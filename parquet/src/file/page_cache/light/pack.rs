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

//! Scalar reference packer and encoding selection.

use super::format::{BLOCK_LEN, Frame, LightBlock, LightCodeBlock, LightError, Values, Words};

fn check_input(len: usize, validity: Option<&[u8]>) -> Result<(), LightError> {
    if !(1..=BLOCK_LEN).contains(&len) {
        return Err(LightError::BlockLength);
    }
    if validity.is_some_and(|bits| bits.len() != len.div_ceil(8)) {
        return Err(LightError::ValidityLength);
    }
    Ok(())
}

fn valid(validity: Option<&[u8]>, index: usize) -> bool {
    validity.is_none_or(|bits| bits[index / 8] & (1 << (index % 8)) != 0)
}

impl LightBlock {
    /// Pack signed logical slots using the smallest encoding. Set bits in
    /// `validity` name values, least-significant bit first, starting at row 0.
    /// A NULL slot's input value is ignored. Ties favor frame of reference.
    pub fn pack(input: &[i64], validity: Option<&[u8]>) -> Result<Self, LightError> {
        check_input(input.len(), validity)?;
        Ok(Self::pack_impl(input, validity))
    }

    /// Pack INT32, Date32 or Decimal32 values using their signed order.
    pub fn pack_i32(input: &[i32], validity: Option<&[u8]>) -> Result<Self, LightError> {
        check_input(input.len(), validity)?;
        let mut widened = [0; BLOCK_LEN];
        for (dst, &src) in widened.iter_mut().zip(input) {
            *dst = i64::from(src);
        }
        Ok(Self::pack_impl(&widened[..input.len()], validity))
    }

    fn pack_impl(input: &[i64], validity: Option<&[u8]>) -> Self {
        let len = input.len();
        let mut min = i64::MAX;
        let mut max = i64::MIN;
        let mut previous = None;
        let mut run_count = 0;
        let mut first_end = 0;
        for (index, &value) in input.iter().enumerate() {
            if !valid(validity, index) {
                continue;
            }
            min = min.min(value);
            max = max.max(value);
            if previous != Some(value) {
                if run_count == 1 {
                    first_end = index;
                }
                previous = Some(value);
                run_count += 1;
            }
        }

        let values = if run_count == 0 {
            Values::Constant(0)
        } else if run_count == 1 {
            Values::Constant(min)
        } else {
            let width = Frame::width(min, max);
            let end_width = Frame::width(first_end as i64, len as i64);
            if Frame::bytes(run_count, width) + Frame::bytes(run_count, end_width)
                < Frame::bytes(len, width)
            {
                let mut run_values = [0; BLOCK_LEN];
                let mut ends = [0; BLOCK_LEN];
                let mut runs = 0;
                for (index, &value) in input.iter().enumerate() {
                    if valid(validity, index) && (runs == 0 || run_values[runs - 1] != value) {
                        if runs > 0 {
                            ends[runs - 1] = index as i64;
                        }
                        run_values[runs] = value;
                        runs += 1;
                    }
                }
                ends[runs - 1] = len as i64;
                Values::Runs {
                    values: Frame::pack(&run_values[..runs], None, min, width),
                    ends: Frame::pack(&ends[..runs], None, first_end as i64, end_width),
                }
            } else {
                Values::Packed(Frame::pack(input, validity, min, width))
            }
        };
        Self { len, values }
    }
}

impl LightCodeBlock {
    /// Pack original dictionary codes. Validate codes only at valid slots.
    /// The same chunk dictionary must be supplied to the consumer.
    pub fn pack(
        codes: &[u32],
        validity: Option<&[u8]>,
        dictionary_len: usize,
    ) -> Result<Self, LightError> {
        check_input(codes.len(), validity)?;
        let mut widened = [0; BLOCK_LEN];
        let mut all_null = true;
        for (index, &code) in codes.iter().enumerate() {
            if valid(validity, index) {
                if code as usize >= dictionary_len {
                    return Err(LightError::DictionaryIndex);
                }
                widened[index] = i64::from(code);
                all_null = false;
            }
        }
        Ok(Self {
            block: LightBlock::pack_impl(&widened[..codes.len()], validity),
            all_null,
        })
    }
}

impl Frame {
    fn pack(input: &[i64], validity: Option<&[u8]>, base: i64, width: u8) -> Self {
        let bits = usize::from(width);
        macro_rules! pack_words {
            ($word:ty, $lanes:expr, $word_bits:expr, $variant:ident) => {{
                let mut words = vec![0; Self::word_count(input.len(), bits, $lanes, $word_bits)];
                for (row, values) in input.chunks($lanes).enumerate() {
                    let (word, shift) = (row * bits / $word_bits, row * bits % $word_bits);
                    for (lane, &value) in values.iter().enumerate() {
                        if valid(validity, row * $lanes + lane) {
                            let delta = value.wrapping_sub(base) as $word;
                            words[word * $lanes + lane] |= delta << shift;
                            if shift + bits > $word_bits {
                                words[(word + 1) * $lanes + lane] |= delta >> ($word_bits - shift);
                            }
                        }
                    }
                }
                Words::$variant(words.into_boxed_slice())
            }};
        }
        let words = if width <= 32 {
            pack_words!(u32, 32, 32, Narrow)
        } else {
            pack_words!(u64, 16, 64, Wide)
        };
        Self {
            len: input.len(),
            base,
            width,
            words,
        }
    }
}
