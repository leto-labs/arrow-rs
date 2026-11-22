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

//! Decode light blocks into caller-owned buffers.

use super::format::{BLOCK_LEN, Frame, LightBlock, LightCodeBlock, LightError, Values, Words};

impl LightBlock {
    /// Decode into the first `len()` slots of the output.
    pub fn decode_into(&self, out: &mut [i64]) -> Result<(), LightError> {
        self.decode(out, Frame::unpack)
    }

    /// Decode INT32, Date32 or Decimal32 slots, validating the output range.
    pub fn decode_i32_into(&self, out: &mut [i32]) -> Result<(), LightError> {
        let out = out.get_mut(..self.len).ok_or(LightError::OutputLength)?;
        let mut widened = [0; BLOCK_LEN];
        self.decode_into(&mut widened[..out.len()])?;
        for (dst, src) in out.iter_mut().zip(widened) {
            *dst = i32::try_from(src).map_err(|_| LightError::ValueOutOfRange)?;
        }
        Ok(())
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
    fn unpack(&self, out: &mut [i64]) {
        #[cfg(any(
            all(target_arch = "x86_64", not(miri)),
            all(target_arch = "aarch64", not(miri))
        ))]
        {
            let lanes = if self.width <= 32 { 32 } else { 16 };
            let full = out.len() / lanes * lanes;
            if full != 0 && simd::available() {
                let (head, tail) = out.split_at_mut(full);
                // SAFETY: The CPU supports this backend and head ends at a
                // complete row of the validated frame.
                unsafe { simd::unpack(self, head) };
                if !tail.is_empty() {
                    self.unpack_scalar_from(full, tail);
                }
                return;
            }
        }
        self.unpack_scalar(out);
    }

    fn unpack_scalar(&self, out: &mut [i64]) {
        self.unpack_scalar_from(0, out);
    }

    fn unpack_scalar_from(&self, start: usize, out: &mut [i64]) {
        let width = usize::from(self.width);
        macro_rules! unpack_words {
            ($words:expr, $word:ty, $lanes:expr, $bits:expr) => {{
                let mask = <$word>::MAX >> ($bits - width);
                for (row, output) in out.chunks_mut($lanes).enumerate() {
                    let row = row + start / $lanes;
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
        let Some(max_code) = self.max_code else {
            out.fill(0);
            return Ok(());
        };
        if max_code as usize >= dictionary.len() {
            return Err(LightError::DictionaryIndex);
        }
        if let Values::Constant(code) = self.block.values {
            out.fill(dictionary[code as usize]);
            return Ok(());
        }
        let mut codes = [0; BLOCK_LEN];
        self.block.decode_into(&mut codes[..out.len()])?;
        for (dst, code) in out.iter_mut().zip(codes) {
            *dst = dictionary[code as usize];
        }
        Ok(())
    }
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
mod simd {
    use std::arch::x86_64::*;

    use super::{Frame, Words};

    pub(super) fn available() -> bool {
        is_x86_feature_detected!("avx512f")
    }

    pub(super) unsafe fn unpack(frame: &Frame, out: &mut [i64]) {
        macro_rules! kernels {
            ($kernel:ident, $($width:literal),+ $(,)?) => {
                [$($kernel::<$width>),+]
            };
        }
        type Kernel<T> = unsafe fn(&[T], i64, &mut [i64]);
        const NARROW: [Kernel<u32>; 32] = kernels!(
            unpack_narrow,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
        );
        const WIDE: [Kernel<u64>; 32] = kernels!(
            unpack_wide,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
        );
        // SAFETY: The caller checked AVX-512F and supplies only complete rows
        // of the frame. Frame construction allocates every word group used
        // by those rows, including the second word when an offset crosses.
        unsafe {
            match &frame.words {
                Words::Narrow(words) => {
                    NARROW[usize::from(frame.width) - 1](words, frame.base, out)
                }
                Words::Wide(words) => WIDE[usize::from(frame.width) - 33](words, frame.base, out),
            }
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn unpack_narrow<const WIDTH: usize>(words: &[u32], base: i64, out: &mut [i64]) {
        let mask = _mm512_set1_epi32((u32::MAX >> (32 - WIDTH)) as i32);
        let base = _mm512_set1_epi64(base);
        // SAFETY: Each iteration processes a complete row of 32 slots. Each
        // input word group contains 32 u32s, and each pair of 64-byte stores
        // stays inside one half of the output row. The loads may be unaligned.
        unsafe {
            for row in 0..out.len() / 32 {
                let (word, shift) = (row * WIDTH / 32, row * WIDTH % 32);
                let right = _mm_cvtsi32_si128(shift as i32);
                for half in [0, 16] {
                    let input = words.as_ptr().add(word * 32 + half);
                    let lo = _mm512_loadu_si512(input.cast());
                    let mut delta = _mm512_srl_epi32(lo, right);
                    if shift + WIDTH > 32 {
                        let hi = _mm512_loadu_si512(input.add(32).cast());
                        delta = _mm512_or_si512(
                            delta,
                            _mm512_sll_epi32(hi, _mm_cvtsi32_si128((32 - shift) as i32)),
                        );
                    }
                    let delta = _mm512_and_si512(delta, mask);
                    let lo = _mm512_cvtepu32_epi64(_mm512_castsi512_si256(delta));
                    let hi = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64::<1>(delta));
                    let dst = out.as_mut_ptr().add(row * 32 + half);
                    _mm512_storeu_si512(dst.cast(), _mm512_add_epi64(lo, base));
                    _mm512_storeu_si512(dst.add(8).cast(), _mm512_add_epi64(hi, base));
                }
            }
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn unpack_wide<const WIDTH: usize>(words: &[u64], base: i64, out: &mut [i64]) {
        let mask = _mm512_set1_epi64((u64::MAX >> (64 - WIDTH)) as i64);
        let base = _mm512_set1_epi64(base);
        // SAFETY: Each iteration processes a complete row of 16 slots and
        // each input word group contains 16 u64s. A crossing reads the next
        // group. The loads and stores allow unaligned addresses.
        unsafe {
            for row in 0..out.len() / 16 {
                let (word, shift) = (row * WIDTH / 64, row * WIDTH % 64);
                let right = _mm_cvtsi32_si128(shift as i32);
                for half in [0, 8] {
                    let input = words.as_ptr().add(word * 16 + half);
                    let lo = _mm512_loadu_si512(input.cast());
                    let mut delta = _mm512_srl_epi64(lo, right);
                    if shift + WIDTH > 64 {
                        let hi = _mm512_loadu_si512(input.add(16).cast());
                        delta = _mm512_or_si512(
                            delta,
                            _mm512_sll_epi64(hi, _mm_cvtsi32_si128((64 - shift) as i32)),
                        );
                    }
                    let value = _mm512_add_epi64(_mm512_and_si512(delta, mask), base);
                    _mm512_storeu_si512(out.as_mut_ptr().add(row * 16 + half).cast(), value);
                }
            }
        }
    }
}

#[cfg(all(target_arch = "aarch64", not(miri)))]
mod simd {
    use std::arch::aarch64::*;

    use super::{Frame, Words};

    pub(super) fn available() -> bool {
        std::arch::is_aarch64_feature_detected!("neon")
    }

    pub(super) unsafe fn unpack(frame: &Frame, out: &mut [i64]) {
        // SAFETY: The caller checked NEON and passes complete output rows.
        // The frame owns every word group read by those rows.
        unsafe { unpack_neon(frame, out) }
    }

    #[target_feature(enable = "neon")]
    unsafe fn unpack_neon(frame: &Frame, out: &mut [i64]) {
        let width = usize::from(frame.width);
        let base = vdupq_n_s64(frame.base);
        // SAFETY: Loads span four u32s or two u64s inside their word group.
        // A crossing consumes the next group. Stores stay inside the
        // complete output row, and NEON loads and stores allow unaligned addresses.
        unsafe {
            match &frame.words {
                Words::Narrow(words) => {
                    let mask = vdupq_n_u32(u32::MAX >> (32 - width));
                    for row in 0..out.len() / 32 {
                        let (word, shift) = (row * width / 32, (row * width % 32) as i32);
                        for lane in (0..32).step_by(4) {
                            let src = words.as_ptr().add(word * 32 + lane);
                            let mut delta = vshlq_u32(vld1q_u32(src), vdupq_n_s32(-shift));
                            if shift as usize + width > 32 {
                                let hi = vshlq_u32(vld1q_u32(src.add(32)), vdupq_n_s32(32 - shift));
                                delta = vorrq_u32(delta, hi);
                            }
                            let delta = vandq_u32(delta, mask);
                            let lo = vreinterpretq_s64_u64(vmovl_u32(vget_low_u32(delta)));
                            let hi = vreinterpretq_s64_u64(vmovl_u32(vget_high_u32(delta)));
                            let dst = out.as_mut_ptr().add(row * 32 + lane);
                            vst1q_s64(dst, vaddq_s64(lo, base));
                            vst1q_s64(dst.add(2), vaddq_s64(hi, base));
                        }
                    }
                }
                Words::Wide(words) => {
                    let mask = vdupq_n_u64(u64::MAX >> (64 - width));
                    for row in 0..out.len() / 16 {
                        let (word, shift) = (row * width / 64, (row * width % 64) as i64);
                        for lane in (0..16).step_by(2) {
                            let src = words.as_ptr().add(word * 16 + lane);
                            let mut delta = vshlq_u64(vld1q_u64(src), vdupq_n_s64(-shift));
                            if shift as usize + width > 64 {
                                let hi = vshlq_u64(vld1q_u64(src.add(16)), vdupq_n_s64(64 - shift));
                                delta = vorrq_u64(delta, hi);
                            }
                            let value = vreinterpretq_s64_u64(vandq_u64(delta, mask));
                            vst1q_s64(
                                out.as_mut_ptr().add(row * 16 + lane),
                                vaddq_s64(value, base),
                            );
                        }
                    }
                }
            }
        }
    }
}
