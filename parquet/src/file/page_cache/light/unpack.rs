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

use std::mem::MaybeUninit;

// The unpackers write initialized values or leave the previous storage
// unchanged, even on error.
fn storage<T>(out: &mut [T]) -> &mut [MaybeUninit<T>] {
    // SAFETY: MaybeUninit<T> has the same layout as T. The borrow carries
    // the original lifetime and is exclusive. The writers never write an
    // uninitialized value through this view.
    unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast(), out.len()) }
}

impl LightBlock {
    /// Decode into the first `len()` slots of the output.
    pub fn decode_into(&self, out: &mut [i64]) -> Result<(), LightError> {
        self.decode(storage(out), Frame::unpack)
    }

    /// Decode INT32, Date32 or Decimal32 slots, validating the output range.
    pub fn decode_i32_into(&self, out: &mut [i32]) -> Result<(), LightError> {
        self.decode_i32(storage(out))
    }

    /// Append every logical slot to the caller's Arrow values buffer.
    pub fn append_i64(&self, out: &mut Vec<i64>) -> Result<(), LightError> {
        self.append(out, |block, dst| block.decode(dst, Frame::unpack))
    }

    /// Append every INT32, Date32 or Decimal32 slot.
    pub fn append_i32(&self, out: &mut Vec<i32>) -> Result<(), LightError> {
        self.append(out, Self::decode_i32)
    }

    fn append<T: Copy>(
        &self,
        out: &mut Vec<T>,
        decode: impl FnOnce(&Self, &mut [MaybeUninit<T>]) -> Result<(), LightError>,
    ) -> Result<(), LightError> {
        let end = out.len() + self.len;
        out.reserve(self.len);
        decode(self, &mut out.spare_capacity_mut()[..self.len])?;
        // SAFETY: Both decoders write every requested slot on success.
        // The reservation covers the new length, and errors keep the old length.
        unsafe { out.set_len(end) };
        Ok(())
    }

    fn decode_i32(&self, out: &mut [MaybeUninit<i32>]) -> Result<(), LightError> {
        let out = out.get_mut(..self.len).ok_or(LightError::OutputLength)?;
        match &self.values {
            Values::Constant(value) => {
                let value = i32::try_from(*value).map_err(|_| LightError::ValueOutOfRange)?;
                out.fill(MaybeUninit::new(value));
                return Ok(());
            }
            Values::Packed(frame) if frame.fits_i32() => {
                frame.unpack_i32(out);
                return Ok(());
            }
            Values::Packed(_) => {}
        }
        // A conservative bound can exceed i32 even if every actual value fits.
        let mut widened = [0; BLOCK_LEN];
        self.decode_into(&mut widened[..out.len()])?;
        for (dst, src) in out.iter_mut().zip(widened) {
            dst.write(i32::try_from(src).map_err(|_| LightError::ValueOutOfRange)?);
        }
        Ok(())
    }

    /// Decode using the portable reference path.
    pub fn decode_scalar_into(&self, out: &mut [i64]) -> Result<(), LightError> {
        self.decode(storage(out), Frame::unpack_scalar)
    }

    fn decode(
        &self,
        out: &mut [MaybeUninit<i64>],
        unpack: fn(&Frame, &mut [MaybeUninit<i64>]),
    ) -> Result<(), LightError> {
        let out = out.get_mut(..self.len).ok_or(LightError::OutputLength)?;
        match &self.values {
            Values::Constant(base) => out.fill(MaybeUninit::new(*base)),
            Values::Packed(frame) => unpack(frame, out),
        }
        Ok(())
    }
}

impl Frame {
    fn unpack(&self, out: &mut [MaybeUninit<i64>]) {
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
                    self.unpack_scalar_from(full, tail, |value| value);
                }
                return;
            }
        }
        self.unpack_scalar(out);
    }

    fn unpack_scalar(&self, out: &mut [MaybeUninit<i64>]) {
        self.unpack_scalar_from(0, out, |value| value);
    }

    fn fits_i32(&self) -> bool {
        self.width <= 32
            && self.base >= i64::from(i32::MIN)
            && self.base <= i64::from(i32::MAX) - ((1i64 << self.width) - 1)
    }

    fn unpack_i32(&self, out: &mut [MaybeUninit<i32>]) {
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            let full = out.len() / 32 * 32;
            if full != 0 && simd::available() {
                let (head, tail) = out.split_at_mut(full);
                // SAFETY: The range check guarantees a narrow frame.
                // AVX-512F is available and head is made of complete rows.
                unsafe { simd::unpack_i32(self, head) };
                self.unpack_scalar_from(full, tail, |value| value as i32);
                return;
            }
        }
        self.unpack_scalar_from(0, out, |value| value as i32);
    }

    fn unpack_scalar_from<T>(
        &self,
        start: usize,
        out: &mut [MaybeUninit<T>],
        narrow: impl Fn(i64) -> T,
    ) {
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
                            dst.write(narrow(self.base.wrapping_add(delta as i64)));
                        }
                    } else {
                        for (dst, &lo) in output.iter_mut().zip(lo) {
                            dst.write(narrow(
                                self.base.wrapping_add(((lo >> shift) & mask) as i64),
                            ));
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

    use super::{Frame, MaybeUninit, Words};

    pub(super) fn available() -> bool {
        is_x86_feature_detected!("avx512f")
    }

    macro_rules! narrow_kernels {
        ($kernel:ident) => {
            [
                $kernel::<1>,
                $kernel::<2>,
                $kernel::<3>,
                $kernel::<4>,
                $kernel::<5>,
                $kernel::<6>,
                $kernel::<7>,
                $kernel::<8>,
                $kernel::<9>,
                $kernel::<10>,
                $kernel::<11>,
                $kernel::<12>,
                $kernel::<13>,
                $kernel::<14>,
                $kernel::<15>,
                $kernel::<16>,
                $kernel::<17>,
                $kernel::<18>,
                $kernel::<19>,
                $kernel::<20>,
                $kernel::<21>,
                $kernel::<22>,
                $kernel::<23>,
                $kernel::<24>,
                $kernel::<25>,
                $kernel::<26>,
                $kernel::<27>,
                $kernel::<28>,
                $kernel::<29>,
                $kernel::<30>,
                $kernel::<31>,
                $kernel::<32>,
            ]
        };
    }

    pub(super) unsafe fn unpack_i32(frame: &Frame, out: &mut [MaybeUninit<i32>]) {
        type Kernel = unsafe fn(&[u32], i32, &mut [MaybeUninit<i32>]);
        const NARROW: [Kernel; 32] = narrow_kernels!(unpack_narrow_i32);
        let Words::Narrow(words) = &frame.words else {
            unreachable!("an i32 frame uses narrow words");
        };
        // SAFETY: The caller validated the range, AVX-512F, and whole
        // output rows. The frame owns all word groups used by those rows.
        unsafe { NARROW[usize::from(frame.width) - 1](words, frame.base as i32, out) };
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn unpack_narrow_i32<const WIDTH: usize>(
        words: &[u32],
        base: i32,
        out: &mut [MaybeUninit<i32>],
    ) {
        let base = _mm512_set1_epi32(base);
        // SAFETY: The caller passes whole output rows and the frame's
        // matching word groups. Each operation writes 16 of 32 slots.
        unsafe {
            for row in 0..out.len() / 32 {
                for half in [0, 16] {
                    let delta = narrow_delta::<WIDTH>(words, row, half);
                    let dst = out.as_mut_ptr().add(row * 32 + half);
                    _mm512_storeu_si512(dst.cast(), _mm512_add_epi32(delta, base));
                }
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn narrow_delta<const WIDTH: usize>(words: &[u32], row: usize, half: usize) -> __m512i {
        let mask = _mm512_set1_epi32((u32::MAX >> (32 - WIDTH)) as i32);
        let (word, shift) = (row * WIDTH / 32, row * WIDTH % 32);
        let right = _mm_cvtsi32_si128(shift as i32);
        // SAFETY: Each row reads its own validated words, and crossing
        // offsets have the next word group in the same allocation.
        unsafe {
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
            _mm512_and_si512(delta, mask)
        }
    }

    pub(super) unsafe fn unpack(frame: &Frame, out: &mut [MaybeUninit<i64>]) {
        macro_rules! kernels {
            ($kernel:ident, $($width:literal),+ $(,)?) => {
                [$($kernel::<$width>),+]
            };
        }
        type Kernel<T> = unsafe fn(&[T], i64, &mut [MaybeUninit<i64>]);
        const NARROW: [Kernel<u32>; 32] = narrow_kernels!(unpack_narrow);
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
    unsafe fn unpack_narrow<const WIDTH: usize>(
        words: &[u32],
        base: i64,
        out: &mut [MaybeUninit<i64>],
    ) {
        let base = _mm512_set1_epi64(base);
        // SAFETY: Each iteration processes a complete row of 32 slots. A
        // half reads 16 lanes and writes two 64-byte stores inside that half.
        unsafe {
            for row in 0..out.len() / 32 {
                for half in [0, 16] {
                    let delta = narrow_delta::<WIDTH>(words, row, half);
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
    unsafe fn unpack_wide<const WIDTH: usize>(
        words: &[u64],
        base: i64,
        out: &mut [MaybeUninit<i64>],
    ) {
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

    use super::{Frame, MaybeUninit, Words};

    pub(super) fn available() -> bool {
        std::arch::is_aarch64_feature_detected!("neon")
    }

    pub(super) unsafe fn unpack(frame: &Frame, out: &mut [MaybeUninit<i64>]) {
        // SAFETY: The caller checked NEON and passes complete output rows.
        // The frame owns every word group read by those rows.
        unsafe { unpack_neon(frame, out) }
    }

    #[target_feature(enable = "neon")]
    unsafe fn unpack_neon(frame: &Frame, out: &mut [MaybeUninit<i64>]) {
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
                            vst1q_s64(dst.cast(), vaddq_s64(lo, base));
                            vst1q_s64(dst.add(2).cast(), vaddq_s64(hi, base));
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
                                out.as_mut_ptr().add(row * 16 + lane).cast(),
                                vaddq_s64(value, base),
                            );
                        }
                    }
                }
            }
        }
    }
}
