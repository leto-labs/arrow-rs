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

//! Decode light blocks into caller-owned buffers.

use super::format::{BLOCK_LEN, Frame, LightBlock, LightCodeBlock, LightError, Values, Words};

impl LightBlock {
    /// Decode into the first `len()` slots of the output.
    pub fn decode_into(&self, out: &mut [i64]) -> Result<(), LightError> {
        self.decode_scalar_into(out)
    }

    /// Decode using the portable reference path.
    pub fn decode_scalar_into(&self, out: &mut [i64]) -> Result<(), LightError> {
        self.decode(out, Frame::unpack_scalar)
    }

    fn decode(&self, out: &mut [i64], unpack: fn(&Frame, &mut [i64])) -> Result<(), LightError> {
        let out = out.get_mut(..self.len).ok_or(LightError::OutputLength)?;
        match &self.values {
            Values::Constant(base) => out.fill(*base),
            Values::Packed(frame) => unpack(frame, out),
            Values::Runs { values, ends } => {
                let mut run_values = [0; BLOCK_LEN];
                let mut run_ends = [0; BLOCK_LEN];
                unpack(values, &mut run_values[..values.len]);
                unpack(ends, &mut run_ends[..ends.len]);
                let mut start = 0;
                for (&value, &end) in run_values[..values.len].iter().zip(&run_ends) {
                    let end = end as usize;
                    out[start..end].fill(value);
                    start = end;
                }
            }
        }
        Ok(())
    }
}

impl Frame {
    fn unpack_scalar(&self, out: &mut [i64]) {
        let width = usize::from(self.width);
        macro_rules! unpack_words {
            ($words:expr, $word:ty, $lanes:expr, $bits:expr) => {{
                let mask = <$word>::MAX >> ($bits - width);
                for (row, output) in out.chunks_mut($lanes).enumerate() {
                    let (word, shift) = (row * width / $bits, row * width % $bits);
                    let lo = &$words[word * $lanes..][..output.len()];
                    if shift + width > $bits {
                        let hi = &$words[(word + 1) * $lanes..][..output.len()];
                        for ((dst, &lo), &hi) in output.iter_mut().zip(lo).zip(hi) {
                            let delta = ((lo >> shift) | (hi << ($bits - shift))) & mask;
                            *dst = self.base.wrapping_add(delta as i64);
                        }
                    } else {
                        for (dst, &lo) in output.iter_mut().zip(lo) {
                            *dst = self.base.wrapping_add(((lo >> shift) & mask) as i64);
                        }
                    }
                }
            }};
        }
        match &self.words {
            Words::Narrow(words) => unpack_words!(words, u32, 32, 32),
            Words::Wide(words) => unpack_words!(words, u64, 16, 64),
        }
    }
}

impl LightCodeBlock {
    /// Recover the original Parquet indices at valid slots. Apply the page's
    /// validity bitmap to interpret NULLs.
    pub fn decode_into(&self, out: &mut [u32]) -> Result<(), LightError> {
        let out = out.get_mut(..self.len()).ok_or(LightError::OutputLength)?;
        let mut codes = [0; BLOCK_LEN];
        self.block.decode_into(&mut codes[..out.len()])?;
        for (dst, src) in out.iter_mut().zip(codes) {
            *dst = src as u32;
        }
        Ok(())
    }

    /// Copy Arrow's 16-byte views from the chunk dictionary. The caller
    /// retains the dictionary's backing buffers and attaches page validity.
    /// The dictionary must already have passed UTF-8 validation.
    pub fn decode_views(&self, dictionary: &[u128], out: &mut [u128]) -> Result<(), LightError> {
        let out = out.get_mut(..self.len()).ok_or(LightError::OutputLength)?;
        if self.all_null {
            out.fill(0);
            return Ok(());
        }
        let mut codes = [0; BLOCK_LEN];
        self.block.decode_into(&mut codes[..out.len()])?;
        for (dst, code) in out.iter_mut().zip(codes) {
            *dst = *dictionary
                .get(code as usize)
                .ok_or(LightError::DictionaryIndex)?;
        }
        Ok(())
    }
}
