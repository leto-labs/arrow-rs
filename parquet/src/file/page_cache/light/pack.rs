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

//! Encoding selection and frame packing.

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
        Ok(Self::pack_impl(input, validity, Frame::pack, dense_stats))
    }

    /// Pack using the portable reference path.
    pub fn pack_scalar(input: &[i64], validity: Option<&[u8]>) -> Result<Self, LightError> {
        check_input(input.len(), validity)?;
        Ok(Self::pack_impl(
            input,
            validity,
            Frame::pack_scalar,
            dense_scalar_stats,
        ))
    }

    /// Pack INT32, Date32 or Decimal32 values using their signed order.
    pub fn pack_i32(input: &[i32], validity: Option<&[u8]>) -> Result<Self, LightError> {
        check_input(input.len(), validity)?;
        let mut widened = [0; BLOCK_LEN];
        for (dst, &src) in widened.iter_mut().zip(input) {
            *dst = i64::from(src);
        }
        Self::pack(&widened[..input.len()], validity)
    }

    fn pack_impl(
        input: &[i64],
        validity: Option<&[u8]>,
        pack: PackFrame,
        stats: fn(&[i64]) -> (i64, i64, usize, usize),
    ) -> Self {
        let len = input.len();
        let (min, max, run_count, first_end) = if let Some(bitmap) = validity {
            nullable_stats(input, bitmap)
        } else {
            stats(input)
        };

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
                    values: pack(&run_values[..runs], None, min, width),
                    ends: pack(&ends[..runs], None, first_end as i64, end_width),
                }
            } else {
                Values::Packed(pack(input, validity, min, width))
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
        let mut max_code: Option<u32> = None;
        for (index, &code) in codes.iter().enumerate() {
            if valid(validity, index) {
                if code as usize >= dictionary_len {
                    return Err(LightError::DictionaryIndex);
                }
                widened[index] = i64::from(code);
                max_code = Some(max_code.map_or(code, |max| max.max(code)));
            }
        }
        Ok(Self {
            block: LightBlock::pack_impl(
                &widened[..codes.len()],
                validity,
                Frame::pack,
                dense_stats,
            ),
            max_code,
        })
    }
}

type PackFrame = fn(&[i64], Option<&[u8]>, i64, u8) -> Frame;

fn dense_stats(input: &[i64]) -> (i64, i64, usize, usize) {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if input.len() >= 32 && input.len().is_multiple_of(8) && is_x86_feature_detected!("avx512f") {
        // SAFETY: AVX-512F is available and every chunk contains eight values.
        let (min, max, count) = unsafe { avx512::dense_stats(input) };
        let first_end = input.iter().take_while(|&&value| value == input[0]).count();
        return (min, max, count, first_end);
    }

    dense_scalar_stats(input)
}

fn dense_scalar_stats(input: &[i64]) -> (i64, i64, usize, usize) {
    let (min, max) = input
        .iter()
        .fold((i64::MAX, i64::MIN), |(min, max), &value| {
            (min.min(value), max.max(value))
        });
    if min == max {
        return (min, max, 1, input.len());
    }
    let count = 1 + input[..input.len() - 1]
        .iter()
        .zip(&input[1..])
        .map(|(left, right)| usize::from(left != right))
        .sum::<usize>();
    let first_end = input.iter().take_while(|&&value| value == input[0]).count();
    (min, max, count, first_end)
}

fn nullable_stats(input: &[i64], validity: &[u8]) -> (i64, i64, usize, usize) {
    let mut min = i64::MAX;
    let mut max = i64::MIN;
    let mut previous = None;
    let mut count = 0;
    let mut first_end = 0;
    for (index, &value) in input.iter().enumerate() {
        if !valid(Some(validity), index) {
            continue;
        }
        min = min.min(value);
        max = max.max(value);
        if previous != Some(value) {
            if count == 1 {
                first_end = index;
            }
            previous = Some(value);
            count += 1;
        }
    }
    (min, max, count, first_end)
}

impl Frame {
    fn pack(input: &[i64], validity: Option<&[u8]>, base: i64, width: u8) -> Self {
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            let lanes = if width <= 32 { 32 } else { 16 };
            if input.len().is_multiple_of(lanes) && is_x86_feature_detected!("avx512f") {
                // SAFETY: AVX-512F is available and the input spans complete
                // rows. The validated bitmap has one byte per eight slots.
                return unsafe { avx512::pack(input, validity, base, width) };
            }
        }
        Self::pack_scalar(input, validity, base, width)
    }

    fn pack_scalar(input: &[i64], validity: Option<&[u8]>, base: i64, width: u8) -> Self {
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

#[cfg(all(target_arch = "x86_64", not(miri)))]
mod avx512 {
    use std::arch::x86_64::*;

    use super::{Frame, Words};

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn dense_stats(input: &[i64]) -> (i64, i64, usize) {
        let mut min = _mm512_set1_epi64(i64::MAX);
        let mut max = _mm512_set1_epi64(i64::MIN);
        let mut prior = _mm512_set1_epi64(input[0]);
        let mut count = 1;
        for chunk in input.as_chunks::<8>().0 {
            // SAFETY: The chunk contains exactly eight initialized i64s.
            let next = unsafe { _mm512_loadu_si512(chunk.as_ptr().cast()) };
            min = _mm512_min_epi64(min, next);
            max = _mm512_max_epi64(max, next);
            let predecessor = _mm512_alignr_epi64::<7>(next, prior);
            count += _mm512_cmpneq_epi64_mask(next, predecessor).count_ones() as usize;
            prior = next;
        }
        (
            _mm512_reduce_min_epi64(min),
            _mm512_reduce_max_epi64(max),
            count,
        )
    }

    /// AVX-512F and complete rows are required. A supplied bitmap must cover
    /// the input. The caller computes the signed base and offset width.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn pack(
        input: &[i64],
        validity: Option<&[u8]>,
        base: i64,
        width: u8,
    ) -> Frame {
        let width_bits = usize::from(width);
        let broadcast = _mm512_set1_epi64(base);
        // Accumulate one output word at a time per lane. Each complete group
        // is stored once. A crossing value seeds the next group's low bits.
        macro_rules! pack_rows {
            ($ty:ty, $bits:expr, $lanes:expr, $shl:ident, $shr:ident, $load:expr) => {{
                let mut packed =
                    vec![0 as $ty; Frame::word_count(input.len(), width_bits, $lanes, $bits)];
                for lane in [0, $lanes / 2] {
                    let mut acc = _mm512_setzero_si512();
                    let mut word = 0;
                    let mut shift = 0;
                    for start in (0..input.len()).step_by($lanes) {
                        let delta = $load(start + lane);
                        acc = _mm512_or_si512(acc, $shl(delta, _mm_cvtsi32_si128(shift as i32)));
                        if shift + width_bits >= $bits {
                            let offset = word * $lanes + lane;
                            // SAFETY: The allocation covers every output word
                            // group. Each half is exactly one 64-byte store.
                            unsafe {
                                _mm512_storeu_si512(packed.as_mut_ptr().add(offset).cast(), acc)
                            };
                            word += 1;
                            acc = $shr(delta, _mm_cvtsi32_si128(($bits - shift) as i32));
                        }
                        shift = (shift + width_bits) % $bits;
                    }
                    if shift != 0 {
                        let offset = word * $lanes + lane;
                        // SAFETY: The final partial word group is allocated
                        // in full, including both halves.
                        unsafe { _mm512_storeu_si512(packed.as_mut_ptr().add(offset).cast(), acc) };
                    }
                }
                packed.into_boxed_slice()
            }};
        }

        let load = |start| {
            // SAFETY: Starts are multiples of eight and all eight values
            // belong to the current complete row. The load permits unaligned
            // addresses and the bitmap check happened at the public entry.
            let raw = unsafe { _mm512_loadu_si512(input.as_ptr().add(start).cast()) };
            let mask = validity.map_or(u8::MAX, |bits| bits[start / 8]);
            _mm512_maskz_sub_epi64(mask, raw, broadcast)
        };
        let words = if width <= 32 {
            let narrow = |start| {
                let lo = _mm512_cvtepi64_epi32(load(start));
                let hi = _mm512_cvtepi64_epi32(load(start + 8));
                _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi)
            };
            Words::Narrow(pack_rows!(
                u32,
                32,
                32,
                _mm512_sll_epi32,
                _mm512_srl_epi32,
                narrow
            ))
        } else {
            Words::Wide(pack_rows!(
                u64,
                64,
                16,
                _mm512_sll_epi64,
                _mm512_srl_epi64,
                load
            ))
        };
        Frame {
            len: input.len(),
            base,
            width,
            words,
        }
    }
}
