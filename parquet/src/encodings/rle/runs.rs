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

//! Run loop of the native dictionary decoder: run headers, repeated values
//! and byte-aligned packed runs read from one borrowed slice, with the cursor
//! in a register. Run-heavy index streams (surrogate keys written in order of
//! a parent key) have about one run per ten values, so the cost of a run is
//! the cost of the column.
use std::mem::MaybeUninit;

use super::{RleDecoder, fused};
use crate::errors::{ParquetError, Result};

/// Values stored for a repeated run without a length check when the output
/// has room. Most repeated runs of surrogate keys are shorter.
const SHORT: usize = 16;

#[cold]
#[inline(never)]
fn out_of_bounds(dict_len: usize, id: u64) -> ParquetError {
    general_err!(
        "dictionary index out of bounds: the len is {} but the index is {}",
        dict_len,
        id
    )
}

/// Unsigned LEB128 run header at `pos`: its value and its length in bytes.
/// `None` when the slice ends first or the header is longer than ten bytes.
#[inline(always)]
fn header(data: &[u8], pos: usize) -> Option<(u64, usize)> {
    let first = *data.get(pos)?;
    if first < 0x80 {
        return Some((first as u64, 1));
    }
    let mut value = (first & 0x7f) as u64;
    for i in 1..10 {
        let byte = *data.get(pos + i)?;
        value |= ((byte & 0x7f) as u64) << (7 * i);
        if byte < 0x80 {
            return Some((value, i + 1));
        }
    }
    None
}

/// Little-endian value of `bytes` (at most four) bytes at `pos`, which the
/// caller has checked against the slice.
#[inline(always)]
fn repeated_id(data: &[u8], pos: usize, bytes: usize) -> u64 {
    debug_assert!(bytes <= 4 && pos + bytes <= data.len());
    if bytes == 0 {
        0
    } else if data.len() - pos >= 4 {
        let word = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as u64;
        word & (u64::MAX >> (64 - 8 * bytes))
    } else {
        let mut word = [0u8; 4];
        word[..bytes].copy_from_slice(&data[pos..pos + bytes]);
        u32::from_le_bytes(word) as u64
    }
}

/// Writes `value` into the first `n` slots of `out`. A short run stores one
/// block of `SHORT` copies when `out` has room; the slots after `n` are
/// overwritten by the values that follow, or stay outside the initialized
/// prefix that the caller reports. A longer run is a plain slice fill, which
/// compiles to a vector store loop.
#[inline(always)]
pub(super) fn fill<T: Clone>(out: &mut [MaybeUninit<T>], n: usize, value: &T) {
    assert!(n <= out.len());
    if n <= SHORT && out.len() >= SHORT {
        let block: [T; SHORT] = std::array::from_fn(|_| value.clone());
        // SAFETY: `SHORT <= out.len()`, so the unaligned block store writes
        // inside `out`; MaybeUninit<T> has T's layout. The callers pass
        // native primitives, which have no drop glue to skip.
        unsafe { out.as_mut_ptr().cast::<[T; SHORT]>().write_unaligned(block) };
    } else {
        for slot in &mut out[..n] {
            slot.write(value.clone());
        }
    }
}

impl RleDecoder {
    /// Decodes whole runs from the current header while the reader is at a
    /// run boundary, byte-aligned. Returns the values written to `out`. On
    /// return the decoder state is as `reload` and the generic run handling
    /// would have left it: a partial run is in `rle_left`/`bit_packed_left`
    /// with the reader after its header (and value), or within its packed
    /// bytes at a value boundary. An empty result with no run pending means
    /// that the stream ended.
    ///
    /// Only exact native primitives enter; see `get_batch_with_dict_uninit`.
    pub(super) fn dictionary_runs<T: Clone + 'static>(
        &mut self,
        dict: &[T],
        out: &mut [MaybeUninit<T>],
    ) -> Result<usize> {
        debug_assert!(self.rle_left == 0 && self.bit_packed_left == 0);
        let Some(reader) = self.bit_reader.as_mut() else {
            return Err(general_err!("bit_reader should be set"));
        };
        let Some(data) = reader.aligned_remaining() else {
            return Ok(0);
        };
        let width = self.bit_width as usize;
        // Dictionary indices are at most 32 bits wide (the page decoder rejects
        // wider ones); a wider decoder built directly takes the checked
        // `reload` path instead of reading a truncated repeated id.
        if width > 32 {
            return Ok(0);
        }
        let value_bytes = width.div_ceil(8);
        let mut pos = 0;
        let mut written = 0;
        let mut result = Ok(());
        while written < out.len() {
            // fastparquet pads pages with zero bytes; a zero header ends the
            // stream as in `reload`.
            let Some((indicator, len)) = header(data, pos) else {
                break;
            };
            if indicator == 0 {
                break;
            }
            pos += len;
            // The counters keep the low 32 bits of the run length, as `reload`.
            let run = indicator >> 1;
            if indicator & 1 == 0 {
                let run = run as u32;
                if data.len() - pos < value_bytes {
                    // A repeated run header without its value: the same
                    // persistent error state that `reload` leaves.
                    self.rle_left = run;
                    self.current_value = None;
                    pos = data.len();
                    result = Err(general_err!(
                        "parquet_data_error: not enough data for RLE decoding"
                    ));
                    break;
                }
                let id = repeated_id(data, pos, value_bytes);
                pos += value_bytes;
                let Some(value) = dict.get(id as usize) else {
                    self.rle_left = run;
                    self.current_value = Some(id);
                    result = Err(out_of_bounds(dict.len(), id));
                    break;
                };
                let n = (run as usize).min(out.len() - written);
                fill(&mut out[written..], n, value);
                written += n;
                if n < run as usize {
                    self.rle_left = run - n as u32;
                    self.current_value = Some(id);
                }
            } else {
                let mut left = run.wrapping_mul(8) as u32 as usize;
                while left > 0 && written < out.len() {
                    let n = left.min(out.len() - written);
                    let Some(decoded) =
                        fused::decode(&data[pos..], self.bit_width, dict, &mut out[written..written + n])
                    else {
                        break;
                    };
                    match decoded {
                        Ok(0) => break,
                        Ok(n) => {
                            // Fused blocks hold multiples of eight values, so
                            // the cursor stays byte-aligned.
                            pos += n * width / 8;
                            written += n;
                            left -= n;
                        }
                        Err(error) => {
                            result = Err(error);
                            break;
                        }
                    }
                }
                // The generic path decodes a tail or a width without a
                // fused kernel, and handles a truncated final run.
                self.bit_packed_left = left as u32;
                if left > 0 || result.is_err() {
                    break;
                }
            }
        }
        reader.skip(pos, 8);
        result.map(|()| written)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_header_and_value_bytes() {
        assert_eq!(header(&[0x05], 0), Some((5, 1)));
        assert_eq!(header(&[0xff, 0x01], 0), Some((255, 2)));
        assert_eq!(header(&[0x80, 0x80, 0x01], 0), Some((1 << 14, 3)));
        assert_eq!(header(&[0x80], 0), None);
        assert_eq!(header(&[0x80; 10], 0), None);
        assert_eq!(header(&[0xff; 9].iter().copied().chain([0x01]).collect::<Vec<_>>(), 0), Some((u64::MAX, 10)));
        for bytes in 0..=4 {
            let data = [0x11, 0x22, 0x33, 0x44, 0x55];
            let expected = (0..bytes).fold(0u64, |v, i| v | (data[i] as u64) << (8 * i));
            assert_eq!(repeated_id(&data, 0, bytes), expected);
            assert_eq!(repeated_id(&data[5 - bytes..], 0, bytes), (0..bytes).fold(0u64, |v, i| v | (data[5 - bytes + i] as u64) << (8 * i)));
        }
    }

    #[test]
    fn test_fill_overshoot_stays_in_bounds() {
        for len in 0..40 {
            for n in 0..=len {
                let mut out = vec![MaybeUninit::new(-1i64); len];
                fill(&mut out, n, &7i64);
                for slot in &out[..n] {
                    // SAFETY: every slot was initialized before fill.
                    assert_eq!(unsafe { slot.assume_init() }, 7);
                }
            }
        }
    }
}
