// Modified by the Longbow authors; see third_party/README.md.

// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Borrowed runs in Parquet's RLE/bit-packed hybrid encoding.

use super::{run_len, runs};
use crate::errors::{ParquetError, Result};

/// A run of levels or non-null dictionary indices.
#[derive(Debug, PartialEq, Eq)]
pub enum CodeRun<'a> {
    /// One code repeated across the run.
    Repeated {
        /// The code.
        code: u32,
        /// Number of logical entries.
        len: usize,
    },
    /// Eight-code groups stored in the hybrid encoding's little-endian bit order.
    Packed {
        /// Bits per code.
        width: u8,
        /// Encoded groups, including any final padding.
        bytes: &'a [u8],
        /// Number of logical entries, excluding padding.
        len: usize,
    },
}

impl CodeRun<'_> {
    /// Logical entries in the run, excluding packed padding.
    pub fn len(&self) -> usize {
        match *self {
            Self::Repeated { len, .. } | Self::Packed { len, .. } => len,
        }
    }

    /// Whether the run has no logical entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the code at this logical offset.
    #[inline]
    pub fn code_at(&self, offset: usize) -> Option<u32> {
        if offset >= self.len() {
            return None;
        }
        match *self {
            Self::Repeated { code, .. } => Some(code),
            Self::Packed { width, bytes, .. } if width <= 32 => {
                let width = usize::from(width);
                let group = (offset / 8).checked_mul(width)?;
                let bytes = bytes.get(group..group.checked_add(width)?)?;
                Some(packed_code(bytes, width, offset % 8) as u32)
            }
            _ => None,
        }
    }

    /// Counts set bits in part of a flat definition-level run.
    #[cfg(feature = "arrow")]
    pub(crate) fn count_ones(&self, start: usize, end: usize) -> usize {
        debug_assert!(start <= end && end <= self.len());
        if start == end {
            return 0;
        }
        match *self {
            Self::Repeated { code, .. } => usize::from(code == 1) * (end - start),
            Self::Packed {
                width: 1, bytes, ..
            } => {
                let first = start / 8;
                let last = (end - 1) / 8;
                let first_mask = u8::MAX << (start % 8);
                let last_mask = u8::MAX >> (7 - (end - 1) % 8);
                if first == last {
                    return (bytes[first] & first_mask & last_mask).count_ones() as usize;
                }
                (bytes[first] & first_mask).count_ones() as usize
                    + bytes[first + 1..last]
                        .iter()
                        .map(|b| b.count_ones() as usize)
                        .sum::<usize>()
                    + (bytes[last] & last_mask).count_ones() as usize
            }
            _ => unreachable!("flat definition levels use width 1"),
        }
    }
}

/// A page-local cursor over hybrid RLE runs. For dictionary indices, pass the
/// number of non-null values as `len` and the chunk's entry count as `code_limit`.
/// For definition levels, pass the page's level count and the maximum level plus one.
#[derive(Debug)]
pub struct CodeCursor<'a> {
    data: &'a [u8],
    remaining: usize,
    code_limit: u64,
    width: u8,
    failed: bool,
}

impl<'a> CodeCursor<'a> {
    /// Creates a cursor over a hybrid stream without its width or length prefix.
    pub fn new(width: u8, data: &'a [u8], len: usize, code_limit: u64) -> Result<Self> {
        if width > 32 {
            return Err(general_err!("Invalid code width {width}"));
        }
        Ok(Self {
            data,
            remaining: len,
            code_limit,
            width,
            failed: false,
        })
    }

    /// Reads a validated run. Packed bytes borrow the page and can contain up
    /// to seven padding entries past the logical end of the final run.
    pub fn next_run(&mut self) -> Result<Option<CodeRun<'a>>> {
        if self.failed {
            return Err(general_err!("Invalid code stream"));
        }
        if self.remaining == 0 {
            return Ok(None);
        }
        let result = self.read_run();
        self.failed = result.is_err();
        result.map(Some)
    }

    fn read_run(&mut self) -> Result<CodeRun<'a>> {
        let (indicator, offset) =
            runs::header(self.data, 0).ok_or_else(|| general_err!("Incomplete code run header"))?;
        if offset == 10 && self.data[9] > 1 {
            return Err(general_err!("Code run header exceeds 64 bits"));
        }
        let run = run_len(indicator)? as usize;
        if run == 0 {
            return Err(general_err!("Empty code run"));
        }
        let width = usize::from(self.width);
        let data = &self.data[offset..];
        let len = run.min(self.remaining);
        let (result, consumed) = if indicator & 1 == 0 {
            let consumed = width.div_ceil(8);
            if run != len || data.len() < consumed {
                return Err(general_err!("Invalid repeated code run length"));
            }
            let code = runs::repeated_id(data, 0, consumed);
            if code >= 1u64 << width {
                return Err(general_err!(
                    "Repeated code {code} exceeds bit width {width}"
                ));
            }
            self.validate_code(code)?;
            (
                CodeRun::Repeated {
                    code: code as u32,
                    len,
                },
                consumed,
            )
        } else {
            let consumed = (run / 8)
                .checked_mul(width)
                .ok_or_else(|| general_err!("Packed code run byte length overflows usize"))?;
            let bytes = data
                .get(..consumed)
                .filter(|_| run - len < 8)
                .ok_or_else(|| general_err!("Invalid packed code run length"))?;
            if self.code_limit < 1u64 << width {
                for index in 0..len {
                    self.validate_code(packed_code(bytes, width, index))?;
                }
            }
            (
                CodeRun::Packed {
                    width: self.width,
                    bytes,
                    len,
                },
                consumed,
            )
        };
        self.data = &data[consumed..];
        self.remaining -= len;
        Ok(result)
    }

    #[inline]
    fn validate_code(&self, code: u64) -> Result<()> {
        if code >= self.code_limit {
            return Err(runs::out_of_bounds(self.code_limit as usize, code));
        }
        Ok(())
    }

    /// Returns keys at strictly increasing offsets relative to the cursor's
    /// current position. Validates unselected codes too.
    pub fn selected_keys(&mut self, offsets: impl IntoIterator<Item = usize>) -> Result<Vec<u32>> {
        let mut offsets = offsets.into_iter().peekable();
        let mut start = 0;
        let mut keys = Vec::new();
        while let Some(run) = self.next_run()? {
            let len = match run {
                CodeRun::Repeated { len, .. } | CodeRun::Packed { len, .. } => len,
            };
            let mut next = start;
            while let Some(offset) = offsets.next_if(|&offset| offset < start + len) {
                if offset < next {
                    return Err(general_err!("Code selections must be strictly increasing"));
                }
                let code = match run {
                    CodeRun::Repeated { code, .. } => code,
                    CodeRun::Packed { width, bytes, .. } => {
                        packed_code(bytes, usize::from(width), offset - start) as u32
                    }
                };
                keys.push(code);
                next = offset + 1;
            }
            start += len;
        }
        if offsets.peek().is_some() {
            return Err(general_err!("Code selection past the end of the page"));
        }
        Ok(keys)
    }
}

fn packed_code(bytes: &[u8], width: usize, index: usize) -> u64 {
    // Each group occupies exactly `width` bytes. Work within one group so
    // computing a bit offset cannot overflow even on a 32-bit target.
    let bit = (index % 8) * width;
    let first = (index / 8) * width + bit / 8;
    let len = ((bit % 8) + width).div_ceil(8);
    let mut word = [0; 8];
    word[..len].copy_from_slice(&bytes[first..first + len]);
    (u64::from_le_bytes(word) >> (bit % 8)) & ((1u64 << width) - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encodings::rle::{RleDecoder, RleEncoder};
    use bytes::Bytes;

    #[test]
    fn codes_match_value_decoder_at_every_width() {
        crate::util::native::dispatch::for_each_tier(|| {
            let mut seed = 7u32;
            for width in 0..=32 {
                let entries = (1u64 << width).min(257);
                let mut input = Vec::new();
                for i in 0..19 {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    let code = (u64::from(seed) % entries) as u32;
                    input.extend(std::iter::repeat_n(code, if i % 5 == 0 { 35 } else { 1 }));
                }
                let mut encoder = RleEncoder::new(width, 0);
                for &code in &input {
                    encoder.put(u64::from(code));
                }
                let data = encoder.consume();
                let mut cursor = CodeCursor::new(width, &data, input.len(), entries).unwrap();
                let selected: Vec<_> = (0..input.len()).filter(|i| i % 3 != 0).collect();
                let actual = cursor.selected_keys(selected.iter().copied()).unwrap();

                let dictionary: Vec<i64> = (0..entries as i64).map(|value| -3 * value).collect();
                let mut decoder = RleDecoder::new(width);
                decoder.set_data(Bytes::from(data)).unwrap();
                let mut values = vec![0; input.len()];
                assert_eq!(
                    decoder
                        .get_batch_with_dict(&dictionary, &mut values, input.len())
                        .unwrap(),
                    input.len()
                );
                assert_eq!(
                    actual
                        .iter()
                        .map(|&code| dictionary[code as usize])
                        .collect::<Vec<_>>(),
                    selected.iter().map(|&i| values[i]).collect::<Vec<_>>()
                );
            }
        });
    }

    #[test]
    fn packed_codes_keep_the_high_bit_and_ignore_final_padding() {
        let mut data = vec![3];
        data.extend_from_slice(&u32::MAX.to_le_bytes());
        data.extend_from_slice(&[0xff; 28]);
        let mut cursor = CodeCursor::new(32, &data, 1, 1u64 << 32).unwrap();
        assert_eq!(cursor.selected_keys([0]).unwrap(), [u32::MAX]);

        let mut cursor = CodeCursor::new(1, &[3, 0xfe], 1, 1).unwrap();
        assert_eq!(cursor.selected_keys([0]).unwrap(), [0]);
    }

    #[test]
    fn rejects_invalid_codes_and_malformed_runs() {
        for data in [&[2, 1][..], &[3, 1], &[2], &[3], &[0], &[1], &[0x80]] {
            let mut cursor = CodeCursor::new(1, data, 1, 1).unwrap();
            assert!(
                cursor.selected_keys(std::iter::empty()).is_err(),
                "{data:?}"
            );
            assert!(cursor.next_run().is_err());
        }
        let mut cursor = CodeCursor::new(1, &[2, 0], 1, 0).unwrap();
        assert!(cursor.next_run().is_err());
        let mut cursor = CodeCursor::new(0, &[], 0, 0).unwrap();
        assert_eq!(cursor.next_run().unwrap(), None);
    }
}
