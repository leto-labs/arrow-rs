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

//! Byte-aligned dictionary unpack/gather without an intermediate index buffer.
use std::{any::TypeId, mem::MaybeUninit};

use crate::errors::{ParquetError, Result};

/// Returns `None` when this type, width, CPU, or batch cannot use a fused kernel.
/// On success only the returned prefix is initialized. An error never exposes
/// the partially written output as initialized values.
pub(super) fn decode<T: 'static>(
    data: &[u8],
    width: u8,
    dict: &[T],
    out: &mut [MaybeUninit<T>],
) -> Option<Result<usize>> {
    if !(1..=24).contains(&width) || !native::<T>() || dict.len() > i32::MAX as usize {
        return None;
    }
    #[cfg(target_arch = "x86_64")]
    {
        // The narrow shuffle wins for small widths on the measured Zen 4.
        // Wider indices benefit from the full-register byte permutation.
        if width >= 17
            && !cfg!(miri)
            && std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512vbmi")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vl")
        {
            let count = (out.len().min(data.len().saturating_mul(8) / width as usize) / 16) * 16;
            if count >= 16 {
                // SAFETY: feature detection and exact primitive type checks above.
                // The count fits both the packed input and MaybeUninit output.
                return Some(unsafe { avx512(data, width, dict, &mut out[..count]) });
            }
        }
        if !cfg!(miri) && std::is_x86_feature_detected!("avx2") {
            // Each eight-value block consumes width bytes. The two unaligned
            // 16-byte loads may extend at most 16 bytes beyond that block.
            let count = (out
                .len()
                .min((data.len().saturating_sub(16) / width as usize).saturating_mul(8))
                / 8)
                * 8;
            if count >= 8 {
                // SAFETY: AVX2, exact primitive types, input padding, and output
                // extent are checked here. The kernel validates each gather.
                return Some(unsafe { avx2(data, width, dict, &mut out[..count]) });
            }
        }
    }
    None
}

pub(super) fn native<T: 'static>() -> bool {
    let t = TypeId::of::<T>();
    t == TypeId::of::<i32>()
        || t == TypeId::of::<i64>()
        || t == TypeId::of::<f32>()
        || t == TypeId::of::<f64>()
}

#[cold]
fn invalid() -> ParquetError {
    general_err!("dictionary index out of bounds in fused decoder")
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// # Safety
/// AVX2 must be available. T must be i32/i64/f32/f64. Width is 1..=24,
/// output length is a multiple of eight, and data has width bytes per eight
/// outputs plus at least sixteen trailing readable bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2<T>(data: &[u8], width: u8, dict: &[T], out: &mut [MaybeUninit<T>]) -> Result<usize> {
    let w = width as usize;
    let mut shuffle = [0u8; 32];
    let mut shifts = [0i32; 8];
    for lane in 0..8 {
        let bit = lane * w;
        shifts[lane] = (bit % 8) as i32;
        for byte in 0..4 {
            shuffle[lane * 4 + byte] = (bit / 8 - if lane >= 4 { w / 2 } else { 0 } + byte) as u8;
        }
    }
    // SAFETY: shuffle and shifts each contain exactly 32 readable bytes.
    let (shuffle, shifts) = unsafe {
        (
            _mm256_loadu_si256(shuffle.as_ptr().cast()),
            _mm256_loadu_si256(shifts.as_ptr().cast()),
        )
    };
    let mask = _mm256_set1_epi32(((1u32 << width) - 1) as i32);
    let bound = _mm256_set1_epi32(dict.len() as i32);
    for (block, dst) in out.chunks_exact_mut(8).enumerate() {
        // SAFETY: the wrapper leaves 16 readable bytes after every packed block.
        let bytes = unsafe {
            let p = data.as_ptr().add(block * w);
            _mm256_set_m128i(
                _mm_loadu_si128(p.add(w / 2).cast()),
                _mm_loadu_si128(p.cast()),
            )
        };
        let ids = _mm256_and_si256(
            _mm256_srlv_epi32(_mm256_shuffle_epi8(bytes, shuffle), shifts),
            mask,
        );
        // Width <=24 makes all IDs nonnegative signed i32. Empty dicts fail too.
        let valid = _mm256_cmpgt_epi32(bound, ids);
        if _mm256_movemask_epi8(valid) != -1 {
            return Err(invalid());
        }
        // SAFETY: all eight IDs are below dict.len(), T is an exact primitive,
        // and dst contains eight writable slots. Loads/stores are unaligned.
        unsafe {
            if size_of::<T>() == 8 {
                let lo =
                    _mm256_i32gather_epi64::<8>(dict.as_ptr().cast(), _mm256_castsi256_si128(ids));
                let hi = _mm256_i32gather_epi64::<8>(
                    dict.as_ptr().cast(),
                    _mm256_extracti128_si256::<1>(ids),
                );
                _mm256_storeu_si256(dst.as_mut_ptr().cast(), lo);
                _mm256_storeu_si256(dst.as_mut_ptr().add(4).cast(), hi);
            } else {
                let values = _mm256_i32gather_epi32::<4>(dict.as_ptr().cast(), ids);
                _mm256_storeu_si256(dst.as_mut_ptr().cast(), values);
            }
        }
    }
    Ok(out.len())
}

/// # Safety
/// AVX512F/BW/VL/VBMI must be available. T must be i32/i64/f32/f64. Width is
/// 1..=24, output length is a multiple of sixteen, and data contains width*2
/// bytes per sixteen outputs. Masked loads never read past a packed run.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn avx512<T>(
    data: &[u8],
    width: u8,
    dict: &[T],
    out: &mut [MaybeUninit<T>],
) -> Result<usize> {
    let w = width as usize;
    let mut shuffle = [0u8; 64];
    let mut shifts = [0i32; 16];
    for lane in 0..16 {
        shifts[lane] = (lane * w % 8) as i32;
        for byte in 0..4 {
            shuffle[lane * 4 + byte] = (lane * w / 8 + byte) as u8;
        }
    }
    // SAFETY: both arrays contain exactly 64 readable bytes.
    let (shuffle, shifts) = unsafe {
        (
            _mm512_loadu_si512(shuffle.as_ptr().cast()),
            _mm512_loadu_si512(shifts.as_ptr().cast()),
        )
    };
    let mask = _mm512_set1_epi32(((1u32 << width) - 1) as i32);
    let bound = _mm512_set1_epi32(dict.len() as i32);
    let load_mask = (1u64 << (2 * w)) - 1;
    for (block, dst) in out.chunks_exact_mut(16).enumerate() {
        // SAFETY: exactly 2*w bytes exist for this block; masked-off bytes are
        // zeroed without touching memory. Shuffle windows may use those zeros.
        let bytes =
            unsafe { _mm512_maskz_loadu_epi8(load_mask, data.as_ptr().add(block * 2 * w).cast()) };
        let ids = _mm512_and_si512(
            _mm512_srlv_epi32(_mm512_permutexvar_epi8(shuffle, bytes), shifts),
            mask,
        );
        if _mm512_cmplt_epi32_mask(ids, bound) != u16::MAX {
            return Err(invalid());
        }
        // SAFETY: every ID is validated above, T is an exact primitive, and
        // the destination holds sixteen writable slots. No typed T is read.
        unsafe {
            if size_of::<T>() == 8 {
                let lo =
                    _mm512_i32gather_epi64::<8>(_mm512_castsi512_si256(ids), dict.as_ptr().cast());
                let hi = _mm512_i32gather_epi64::<8>(
                    _mm512_extracti64x4_epi64::<1>(ids),
                    dict.as_ptr().cast(),
                );
                _mm512_storeu_si512(dst.as_mut_ptr().cast(), lo);
                _mm512_storeu_si512(dst.as_mut_ptr().add(8).cast(), hi);
            } else {
                let values = _mm512_i32gather_epi32::<4>(ids, dict.as_ptr().cast());
                _mm512_storeu_si512(dst.as_mut_ptr().cast(), values);
            }
        }
    }
    Ok(out.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent little-endian bit writer, including the full unsigned range.
    fn packed(ids: &[u32], width: usize) -> Vec<u8> {
        let mut data = vec![0u8; (ids.len() * width).div_ceil(8)];
        for (i, &id) in ids.iter().enumerate() {
            for bit in 0..width {
                data[(i * width + bit) / 8] |= (((id >> bit) & 1) as u8) << ((i * width + bit) % 8);
            }
        }
        data
    }

    #[test]
    fn test_fused_dictionary_properties() {
        let mut state = 0x52a1bc109d32u64;
        for width in 1..=32 {
            let dict: Vec<i64> = (0..(1usize << width.min(11)))
                .map(|i| (i as i64).wrapping_mul(i64::MIN + 17))
                .collect();
            for len in [0, 1, 7, 8, 15, 16, 17, 31, 32, 33, 127, 1024, 1031] {
                let ids: Vec<u32> = (0..len)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        (state as usize % dict.len()) as u32
                    })
                    .collect();
                let data = packed(&ids, width);
                for cut in 0..=data.len().min(35) {
                    let data = &data[..data.len() - cut];
                    let mut out = vec![MaybeUninit::<i64>::uninit(); len + 3];
                    if let Some(result) = decode(data, width as u8, &dict, &mut out[..len]) {
                        let n = result.unwrap();
                        assert!(n <= len && n * width <= data.len() * 8);
                        for i in 0..n {
                            // SAFETY: successful decode initializes exactly this prefix.
                            assert_eq!(unsafe { out[i].assume_init() }, dict[ids[i] as usize]);
                        }
                    }
                }
                #[cfg(target_arch = "x86_64")]
                if width <= 24 && !cfg!(miri) && std::is_x86_feature_detected!("avx2") {
                    // Exercise the AVX2 fallback even on an AVX512 host. Include
                    // readable padding only in this direct-kernel test.
                    let mut padded = data.clone();
                    padded.extend_from_slice(&[0; 16]);
                    let n = len / 8 * 8;
                    let mut out = vec![MaybeUninit::<i64>::uninit(); n];
                    // SAFETY: exact primitive type, supported feature/width,
                    // eight-value blocks, and sixteen readable trailing bytes.
                    assert_eq!(
                        unsafe { avx2(&padded, width as u8, &dict, &mut out) }.unwrap(),
                        n
                    );
                    for i in 0..n {
                        // SAFETY: successful avx2 initialized every output slot.
                        assert_eq!(unsafe { out[i].assume_init() }, dict[ids[i] as usize]);
                    }
                }
            }
        }
    }

    #[test]
    fn test_fused_dictionary_types_and_invalid_ids() {
        // Floating-point gathers must preserve every bit, including NaN payloads.
        let d32 = [f32::from_bits(0x7fc01234), f32::NEG_INFINITY, -0.0, 9.0];
        let d64 = [
            f64::from_bits(0x7ff8000012345678),
            f64::NEG_INFINITY,
            -0.0,
            9.0,
        ];
        let i32s = [i32::MIN, i32::MAX, -1, 0];
        let ids: Vec<u32> = (0..64).map(|i| i % 4).collect();
        for width in [2, 17] {
            let mut data = packed(&ids, width);
            data.extend_from_slice(&[0; 16]);
            let mut out32 = [MaybeUninit::uninit(); 64];
            let mut out64 = [MaybeUninit::uninit(); 64];
            let mut outi32 = [MaybeUninit::uninit(); 64];
            if let Some(result) = decode(&data, width as u8, &d32, &mut out32) {
                let n = result.unwrap();
                assert_eq!(
                    decode(&data, width as u8, &d64, &mut out64)
                        .unwrap()
                        .unwrap(),
                    n
                );
                assert_eq!(
                    decode(&data, width as u8, &i32s, &mut outi32)
                        .unwrap()
                        .unwrap(),
                    n
                );
                for i in 0..n {
                    // SAFETY: all three calls successfully initialized n slots.
                    unsafe {
                        assert_eq!(
                            out32[i].assume_init().to_bits(),
                            d32[ids[i] as usize].to_bits()
                        );
                        assert_eq!(
                            out64[i].assume_init().to_bits(),
                            d64[ids[i] as usize].to_bits()
                        );
                        assert_eq!(outi32[i].assume_init(), i32s[ids[i] as usize]);
                    }
                }
            }
        }
        for width in 1..=24 {
            for position in [0, 7, 8, 15, 16, 31, 63] {
                let mut ids = vec![0u32; 64];
                ids[position] = 1;
                let data = packed(&ids, width);
                let mut out = [MaybeUninit::uninit(); 64];
                if let Some(result) = decode(&data, width as u8, &[42i64], &mut out) {
                    // AVX2 can leave a short suffix for scalar decoding.
                    if let Ok(n) = result {
                        assert!(position >= n);
                    }
                }
            }
        }
        assert!(decode(&[0; 32], 2, &["abc"], &mut [MaybeUninit::uninit(); 64]).is_none());
    }
}
