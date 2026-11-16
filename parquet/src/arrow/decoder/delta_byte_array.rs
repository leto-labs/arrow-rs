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

use crate::encodings::decoding::DeltaLengths;
use crate::errors::{ParquetError, Result};

/// Decoder for `Encoding::DELTA_BYTE_ARRAY`
///
/// The prefix and suffix lengths are decoded a window at a time, in step,
/// so memory does not depend on the count that the length blocks declare
/// (see [`DeltaLengths`]).
pub struct DeltaByteArrayDecoder {
    // Boxed: the decoder enums hold this beside much smaller decoders.
    prefix_lengths: Box<DeltaLengths>,
    suffix_lengths: Box<DeltaLengths>,
    data: Bytes,
    data_offset: usize,
    last_value: Vec<u8>,
}

impl DeltaByteArrayDecoder {
    /// Create a new [`DeltaByteArrayDecoder`] with the provided data page,
    /// which has at most `max_values` values
    pub fn new(data: Bytes, max_values: usize) -> Result<Self> {
        let prefix_lengths =
            DeltaLengths::try_new(data.clone(), max_values, "DELTA_BYTE_ARRAY prefix")?;
        let prefix_end = prefix_lengths.end_offset();
        if prefix_end > data.len() {
            return Err(ParquetError::EOF(
                "eof decoding DELTA_BYTE_ARRAY prefix lengths".into(),
            ));
        }
        let suffix_lengths = DeltaLengths::try_new(
            data.slice(prefix_end..),
            max_values,
            "DELTA_BYTE_ARRAY suffix",
        )?;

        let num_prefix = prefix_lengths.remaining();
        let num_suffix = suffix_lengths.remaining();
        if num_prefix != num_suffix {
            return Err(general_err!(format!(
                "inconsistent DELTA_BYTE_ARRAY lengths, prefixes: {num_prefix}, suffixes: {num_suffix}"
            )));
        }

        // Lengths are below 2^31 and this offset is inside the data, so the
        // offset arithmetic of `read` and `skip` cannot overflow.
        let data_offset = prefix_end
            .checked_add(suffix_lengths.end_offset())
            .filter(|offset| *offset <= data.len())
            .ok_or_else(|| {
                ParquetError::EOF("eof decoding DELTA_BYTE_ARRAY suffix lengths".into())
            })?;

        Ok(Self {
            prefix_lengths: Box::new(prefix_lengths),
            suffix_lengths: Box::new(suffix_lengths),
            data,
            data_offset,
            last_value: vec![],
        })
    }

    /// Returns the number of values remaining
    pub fn remaining(&self) -> usize {
        self.prefix_lengths.remaining()
    }

    /// Decodes up to `len` values, calling `f` with each if it is `Some`.
    fn decode<F>(&mut self, len: usize, mut f: Option<F>) -> Result<usize>
    where
        F: FnMut(&[u8]) -> Result<()>,
    {
        let to_decode = len.min(self.remaining());
        let mut decoded = 0;
        while decoded < to_decode {
            // Both blocks hold the same number of lengths and are decoded
            // in windows of the same size, so they stay in step.
            let count = self.prefix_lengths.peek(to_decode - decoded)?.len();
            let count = self.suffix_lengths.peek(count)?.len();
            if count == 0 {
                return Err(general_err!("insufficient DELTA_BYTE_ARRAY lengths"));
            }
            let data = self.data.as_ref();
            for (prefix_length, suffix_length) in self
                .prefix_lengths
                .window(count)
                .iter()
                .zip(self.suffix_lengths.window(count))
            {
                let prefix_length = *prefix_length as usize;
                let suffix_length = *suffix_length as usize;

                if self.data_offset + suffix_length > data.len() {
                    return Err(ParquetError::EOF("eof decoding byte array".into()));
                }
                self.last_value.truncate(prefix_length);
                self.last_value
                    .extend_from_slice(&data[self.data_offset..self.data_offset + suffix_length]);
                if let Some(f) = f.as_mut() {
                    f(&self.last_value)?;
                }
                self.data_offset += suffix_length;
            }
            self.prefix_lengths.consume(count);
            self.suffix_lengths.consume(count);
            decoded += count;
        }
        Ok(to_decode)
    }

    /// Read up to `len` values, returning the number of values read
    /// and calling `f` with each decoded byte slice
    ///
    /// Will short-circuit and return on error
    pub fn read<F>(&mut self, len: usize, f: F) -> Result<usize>
    where
        F: FnMut(&[u8]) -> Result<()>,
    {
        self.decode(len, Some(f))
    }

    /// Skip up to `to_skip` values, returning the number of values skipped
    pub fn skip(&mut self, to_skip: usize) -> Result<usize> {
        self.decode(to_skip, None::<fn(&[u8]) -> Result<()>>)
    }
}

#[cfg(test)]
mod invalid_input_tests {
    use super::*;
    use crate::data_type::Int32Type;
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
        let mut decoder = DeltaByteArrayDecoder::new(data, 2)?;
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
            "Parquet error: Invalid DELTA_BYTE_ARRAY suffix length -1"
        );
        let err = read_all(encode(&[0, -2], &[3, 1], b"abcd")).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Parquet error: Invalid DELTA_BYTE_ARRAY prefix length -2"
        );
    }

    #[test]
    fn suffix_past_end_fails() {
        let err = read_all(encode(&[0, 2], &[3, 2], b"abcd")).unwrap_err();
        assert_eq!(err.to_string(), "EOF: eof decoding byte array");

        let mut decoder = DeltaByteArrayDecoder::new(encode(&[0, 2], &[3, 2], b"abcd"), 2).unwrap();
        let err = decoder.skip(2).unwrap_err();
        assert_eq!(err.to_string(), "EOF: eof decoding byte array");
    }
}
