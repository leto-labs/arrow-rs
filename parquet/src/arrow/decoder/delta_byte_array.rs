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

use bytes::Bytes;

use crate::data_type::Int32Type;
use crate::encodings::decoding::{Decoder, DeltaBitPackDecoder};
use crate::errors::{ParquetError, Result};

/// Decoder for `Encoding::DELTA_BYTE_ARRAY`
pub struct DeltaByteArrayDecoder {
    prefix_lengths: Vec<i32>,
    suffix_lengths: Vec<i32>,
    data: Bytes,
    length_offset: usize,
    data_offset: usize,
    last_value: Vec<u8>,
}

/// Decode all lengths of one DELTA_BYTE_ARRAY length block.
///
/// Checks once for the page that no length is negative, so `read` and `skip`
/// can convert each length to `usize` without a check for each value.
fn read_lengths(decoder: &mut DeltaBitPackDecoder<Int32Type>, name: &str) -> Result<Vec<i32>> {
    let num_lengths = decoder.values_left();
    let mut lengths = vec![0; num_lengths];
    let read = decoder.get(&mut lengths)?;
    if read != num_lengths {
        return Err(general_err!(
            "insufficient DELTA_BYTE_ARRAY {} lengths, expected {}, got {}",
            name,
            num_lengths,
            read
        ));
    }
    if lengths.iter().any(|length| *length < 0) {
        return Err(general_err!("negative DELTA_BYTE_ARRAY {} length", name));
    }
    Ok(lengths)
}

impl DeltaByteArrayDecoder {
    /// Create a new [`DeltaByteArrayDecoder`] with the provided data page
    pub fn new(data: Bytes) -> Result<Self> {
        let mut prefix = DeltaBitPackDecoder::<Int32Type>::new();
        prefix.set_data(data.clone(), 0)?;

        let prefix_lengths = read_lengths(&mut prefix, "prefix")?;
        let num_prefix = prefix_lengths.len();

        let prefix_end = prefix.get_offset();
        if prefix_end > data.len() {
            return Err(ParquetError::EOF(
                "eof decoding DELTA_BYTE_ARRAY prefix lengths".into(),
            ));
        }
        let mut suffix = DeltaBitPackDecoder::<Int32Type>::new();
        suffix.set_data(data.slice(prefix_end..), 0)?;

        let suffix_lengths = read_lengths(&mut suffix, "suffix")?;
        let num_suffix = suffix_lengths.len();

        if num_prefix != num_suffix {
            return Err(general_err!(format!(
                "inconsistent DELTA_BYTE_ARRAY lengths, prefixes: {num_prefix}, suffixes: {num_suffix}"
            )));
        }

        // Lengths are below 2^31 and this offset is inside the data, so the
        // offset arithmetic of `read` and `skip` cannot overflow.
        let data_offset = prefix_end
            .checked_add(suffix.get_offset())
            .filter(|offset| *offset <= data.len())
            .ok_or_else(|| {
                ParquetError::EOF("eof decoding DELTA_BYTE_ARRAY suffix lengths".into())
            })?;

        Ok(Self {
            prefix_lengths,
            suffix_lengths,
            data,
            length_offset: 0,
            data_offset,
            last_value: vec![],
        })
    }

    /// Returns the number of values remaining
    pub fn remaining(&self) -> usize {
        self.prefix_lengths.len() - self.length_offset
    }

    /// Read up to `len` values, returning the number of values read
    /// and calling `f` with each decoded byte slice
    ///
    /// Will short-circuit and return on error
    pub fn read<F>(&mut self, len: usize, mut f: F) -> Result<usize>
    where
        F: FnMut(&[u8]) -> Result<()>,
    {
        let to_read = len.min(self.remaining());

        let length_range = self.length_offset..self.length_offset + to_read;
        let iter = self.prefix_lengths[length_range.clone()]
            .iter()
            .zip(&self.suffix_lengths[length_range]);

        let data = self.data.as_ref();

        for (prefix_length, suffix_length) in iter {
            let prefix_length = *prefix_length as usize;
            let suffix_length = *suffix_length as usize;

            if self.data_offset + suffix_length > self.data.len() {
                return Err(ParquetError::EOF("eof decoding byte array".into()));
            }
            self.last_value.truncate(prefix_length);
            self.last_value
                .extend_from_slice(&data[self.data_offset..self.data_offset + suffix_length]);
            f(&self.last_value)?;

            self.data_offset += suffix_length;
        }

        self.length_offset += to_read;
        Ok(to_read)
    }

    /// Skip up to `to_skip` values, returning the number of values skipped
    pub fn skip(&mut self, to_skip: usize) -> Result<usize> {
        let to_skip = to_skip.min(self.prefix_lengths.len() - self.length_offset);

        let length_range = self.length_offset..self.length_offset + to_skip;
        let iter = self.prefix_lengths[length_range.clone()]
            .iter()
            .zip(&self.suffix_lengths[length_range]);

        let data = self.data.as_ref();

        for (prefix_length, suffix_length) in iter {
            let prefix_length = *prefix_length as usize;
            let suffix_length = *suffix_length as usize;

            if self.data_offset + suffix_length > self.data.len() {
                return Err(ParquetError::EOF("eof decoding byte array".into()));
            }
            self.last_value.truncate(prefix_length);
            self.last_value
                .extend_from_slice(&data[self.data_offset..self.data_offset + suffix_length]);
            self.data_offset += suffix_length;
        }
        self.length_offset += to_skip;
        Ok(to_skip)
    }
}

#[cfg(test)]
mod invalid_input_tests {
    use super::*;
    use crate::encodings::encoding::{DeltaBitPackEncoder, Encoder};

    /// DELTA_BYTE_ARRAY data: the prefix lengths, the suffix lengths, then
    /// the suffix bytes.
    fn encode(prefix_lengths: &[i32], suffix_lengths: &[i32], suffixes: &[u8]) -> Bytes {
        let mut data = Vec::new();
        for lengths in [prefix_lengths, suffix_lengths] {
            let mut encoder = DeltaBitPackEncoder::<Int32Type>::new();
            encoder.put(lengths).unwrap();
            data.extend_from_slice(&encoder.flush_buffer().unwrap());
        }
        data.extend_from_slice(suffixes);
        Bytes::from(data)
    }

    fn read_all(data: Bytes) -> Result<Vec<Vec<u8>>> {
        let mut decoder = DeltaByteArrayDecoder::new(data)?;
        let mut values = Vec::new();
        decoder.read(usize::MAX, |value| {
            values.push(value.to_vec());
            Ok(())
        })?;
        Ok(values)
    }

    #[test]
    fn valid_data_decodes() {
        let values = read_all(encode(&[0, 2], &[3, 1], b"abcd")).unwrap();
        assert_eq!(values, vec![b"abc".to_vec(), b"abd".to_vec()]);
    }

    /// A negative length became a very large `usize`: the offset arithmetic
    /// then overflowed, or the slice range was invalid.
    #[test]
    fn negative_lengths_fail() {
        let err = read_all(encode(&[0, 2], &[3, -1], b"abcd")).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Parquet error: negative DELTA_BYTE_ARRAY suffix length"
        );
        let err = read_all(encode(&[0, -2], &[3, 1], b"abcd")).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Parquet error: negative DELTA_BYTE_ARRAY prefix length"
        );
    }

    #[test]
    fn suffix_past_end_fails() {
        let err = read_all(encode(&[0, 2], &[3, 2], b"abcd")).unwrap_err();
        assert_eq!(err.to_string(), "EOF: eof decoding byte array");

        let mut decoder = DeltaByteArrayDecoder::new(encode(&[0, 2], &[3, 2], b"abcd")).unwrap();
        let err = decoder.skip(2).unwrap_err();
        assert_eq!(err.to_string(), "EOF: eof decoding byte array");
    }
}
