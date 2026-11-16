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
pub(super) use crate::util::native::native;
use std::mem::MaybeUninit;

#[cfg(target_arch = "x86_64")]
use crate::errors::ParquetError;
use crate::errors::Result;

/// Returns `None` when this type, width, CPU, or batch cannot use a fused kernel.
/// On success only the returned prefix is initialized. An error never exposes
/// the partially written output as initialized values.
pub(super) fn decode<T: 'static>(
    data: &[u8],
    width: u8,
    dict: &[T],
    out: &mut [MaybeUninit<T>],
) -> Option<Result<usize>> {
    if !(1..=32).contains(&width) || !native::<T>() || dict.len() > i32::MAX as usize {
        return None;
    }
    #[cfg(target_arch = "x86_64")]
    {
        // The narrow shuffle wins for small widths on the measured Zen 4.
        // Wider indices benefit from the full-register byte permutation.
        #[cfg(test)]
        let tier = crate::util::native::dispatch::tier();
        #[cfg(test)]
        let allow_avx512 = matches!(
            tier,
            crate::util::native::dispatch::Tier::Auto | crate::util::native::dispatch::Tier::Avx512
        );
        #[cfg(not(test))]
        let allow_avx512 = true;
        #[cfg(test)]
        let allow_avx2 = !matches!(tier, crate::util::native::dispatch::Tier::Scalar);
        #[cfg(not(test))]
        let allow_avx2 = true;
        // A dictionary that fits a few registers is looked up with a
        // two-table permutation: it is three times the rate of a gather.
        let small = dict.len() <= SMALL_ENTRIES * 4 / size_of::<T>();
        if allow_avx512
            && (width >= 17 || small)
            && !cfg!(miri)
            && std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512vbmi")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vl")
        {
            let count = (out.len().min(data.len().saturating_mul(8) / width as usize) / 8) * 8;
            if count >= 8 {
                // SAFETY: feature detection and exact primitive type checks above.
                // The count fits both the packed input and MaybeUninit output.
                return Some(unsafe {
                    if small {
                        avx512_small(data, width, dict, &mut out[..count])
                    } else {
                        avx512(data, width, dict, &mut out[..count])
                    }
                });
            }
        }
        if allow_avx2 && width <= 24 && !cfg!(miri) && std::is_x86_feature_detected!("avx2") {
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
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (data, out);
    None
}

#[cfg(target_arch = "x86_64")]
#[cold]
fn invalid(dict_len: usize, index: u32) -> ParquetError {
    general_err!(
        "dictionary index out of bounds: the len is {} but the index is {}",
        dict_len,
        index
    )
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
    debug_assert!(matches!(size_of::<T>(), 4 | 8));
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
    for (block, dst) in out.as_chunks_mut::<8>().0.iter_mut().enumerate() {
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
            let mut indices = [0i32; 8];
            // SAFETY: indices is exactly one writable 256-bit vector.
            unsafe {
                _mm256_storeu_si256(indices.as_mut_ptr().cast(), ids);
            }
            return Err(invalid(dict.len(), *indices.iter().max().unwrap() as u32));
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

/// Entries of a 4-byte dictionary that the permutation kernel handles: four
/// pairs of 64-byte tables. An 8-byte dictionary has half as many.
const SMALL_ENTRIES: usize = 128;

/// Byte permutation and shifts that unpack sixteen `width`-bit IDs into
/// 32-bit lanes, and the masks of a block. An ID of more than 24 bits can
/// span five bytes: those widths unpack each half into 64-bit lanes.
#[cfg(target_arch = "x86_64")]
struct Unpack {
    shuffle: [__m512i; 2],
    shifts: [__m512i; 2],
    mask: __m512i,
    w: usize,
}

#[cfg(target_arch = "x86_64")]
impl Unpack {
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
    fn new(width: u8) -> Self {
        let w = width as usize;
        let mut shuffle = [[0u8; 64]; 2];
        let mut shifts = [[0u64; 8]; 2];
        if w <= 24 {
            let shifts = shifts.as_mut_ptr().cast::<i32>();
            for lane in 0..16 {
                // SAFETY: 16 four-byte lanes fill the first 64-byte array.
                unsafe { shifts.add(lane).write((lane * w % 8) as i32) };
                for byte in 0..4 {
                    shuffle[0][lane * 4 + byte] = (lane * w / 8 + byte) as u8;
                }
            }
        } else {
            for half in 0..2 {
                for lane in 0..8 {
                    let bit = (half * 8 + lane) * w;
                    shifts[half][lane] = (bit % 8) as u64;
                    // Bytes past the fifth are masked off below; clamping
                    // keeps the byte index inside the 64-byte register.
                    for byte in 0..8 {
                        shuffle[half][lane * 8 + byte] = (bit / 8 + byte).min(63) as u8;
                    }
                }
            }
        }
        // SAFETY: every array contains exactly 64 readable bytes.
        let load = |p: *const u8| unsafe { _mm512_loadu_si512(p.cast()) };
        Self {
            shuffle: [load(shuffle[0].as_ptr()), load(shuffle[1].as_ptr())],
            shifts: [
                load(shifts[0].as_ptr().cast()),
                load(shifts[1].as_ptr().cast()),
            ],
            mask: if w <= 24 {
                _mm512_set1_epi32(((1u32 << width) - 1) as i32)
            } else {
                _mm512_set1_epi64(((1u64 << width) - 1) as i64)
            },
            w,
        }
    }

    /// The IDs of `lanes` (8 or 16) values packed at `data[start..]`.
    ///
    /// # Safety
    /// `data[start..]` holds at least `lanes * w / 8` bytes. The masked load
    /// reads exactly those bytes; masked-off bytes are zeroed without touching
    /// memory, and the shuffle windows may use those zeros.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
    unsafe fn ids<const WIDE: bool>(&self, data: &[u8], start: usize, lanes: usize) -> __m512i {
        debug_assert!(start + lanes * self.w / 8 <= data.len());
        let load_mask = u64::MAX >> (64 - lanes * self.w / 8);
        // SAFETY: see the function contract.
        let bytes = unsafe { _mm512_maskz_loadu_epi8(load_mask, data.as_ptr().add(start).cast()) };
        debug_assert_eq!(WIDE, self.w > 24);
        if !WIDE {
            return _mm512_and_si512(
                _mm512_srlv_epi32(
                    _mm512_permutexvar_epi8(self.shuffle[0], bytes),
                    self.shifts[0],
                ),
                self.mask,
            );
        }
        let half = |h: usize| {
            let wide = _mm512_and_si512(
                _mm512_srlv_epi64(
                    _mm512_permutexvar_epi8(self.shuffle[h], bytes),
                    self.shifts[h],
                ),
                self.mask,
            );
            _mm512_cvtepi64_epi32(wide)
        };
        _mm512_inserti64x4::<1>(_mm512_castsi256_si512(half(0)), half(1))
    }
}

/// Unpacks blocks of sixteen IDs (and one final block of eight), checks them
/// against the dictionary and stores `lookup`'s values.
///
/// # Safety
/// AVX512F/BW/VL/VBMI must be available. T must be i32/i64/f32/f64. Width is
/// 1..=32, output length is a multiple of eight, and data contains width
/// bytes per eight outputs. `lookup` receives IDs below `dict.len()` in the
/// lanes that `active` sets, and writes that many values to its pointer.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn blocks<T>(
    data: &[u8],
    width: u8,
    dict: &[T],
    out: &mut [MaybeUninit<T>],
    lookup: impl FnMut(__m512i, *mut MaybeUninit<T>, u16),
) -> Result<usize> {
    // SAFETY: the contract is this function's.
    unsafe {
        if width > 24 {
            blocks_of::<T, true>(data, width, dict, out, lookup)
        } else {
            blocks_of::<T, false>(data, width, dict, out, lookup)
        }
    }
}

/// [`blocks`] for one unpacking layout: full blocks of sixteen with constant
/// masks, then at most one block of eight.
///
/// # Safety
/// As [`blocks`]; `WIDE` is `width > 24`.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn blocks_of<T, const WIDE: bool>(
    data: &[u8],
    width: u8,
    dict: &[T],
    out: &mut [MaybeUninit<T>],
    mut lookup: impl FnMut(__m512i, *mut MaybeUninit<T>, u16),
) -> Result<usize> {
    debug_assert!(matches!(size_of::<T>(), 4 | 8) && out.len().is_multiple_of(8));
    let unpack = Unpack::new(width);
    let bound = _mm512_set1_epi32(dict.len() as i32);
    let w = unpack.w;
    let base = out.as_mut_ptr();
    let mut block = |written: usize, lanes: usize, active: u16| -> Result<()> {
        // SAFETY: eight outputs consume `width` bytes, which the caller checked.
        let ids = unsafe { unpack.ids::<WIDE>(data, written * w / 8, lanes) };
        // Unsigned: IDs of 32 bits can exceed i32::MAX.
        if _mm512_mask_cmplt_epu32_mask(active, ids, bound) != active {
            let mut indices = [0i32; 16];
            // SAFETY: indices is exactly one writable 512-bit vector.
            unsafe {
                _mm512_storeu_si512(indices.as_mut_ptr().cast(), ids);
            }
            let max = indices[..lanes].iter().map(|&i| i as u32).max().unwrap();
            return Err(invalid(dict.len(), max));
        }
        // SAFETY: `lanes` slots from `written` are inside `out`.
        lookup(ids, unsafe { base.add(written) }, active);
        Ok(())
    };
    let full = out.len() / 16 * 16;
    let mut written = 0;
    while written < full {
        block(written, 16, u16::MAX)?;
        written += 16;
    }
    if written < out.len() {
        block(written, 8, 0xff)?;
    }
    Ok(out.len())
}

/// # Safety
/// As [`blocks`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn avx512<T>(
    data: &[u8],
    width: u8,
    dict: &[T],
    out: &mut [MaybeUninit<T>],
) -> Result<usize> {
    let base = dict.as_ptr();
    // SAFETY: `blocks` passes only validated IDs in active lanes; inactive
    // lanes are neither gathered nor stored. No typed T is read.
    unsafe {
        blocks(data, width, dict, out, |ids, dst, active| {
            if size_of::<T>() == 8 {
                let lo = _mm512_i32gather_epi64::<8>(_mm512_castsi512_si256(ids), base.cast());
                _mm512_storeu_si512(dst.cast(), lo);
                if active == u16::MAX {
                    let hi = _mm512_i32gather_epi64::<8>(
                        _mm512_extracti64x4_epi64::<1>(ids),
                        base.cast(),
                    );
                    _mm512_storeu_si512(dst.add(8).cast(), hi);
                }
            } else if active == u16::MAX {
                let values = _mm512_i32gather_epi32::<4>(ids, base.cast());
                _mm512_storeu_si512(dst.cast(), values);
            } else {
                let values = _mm512_mask_i32gather_epi32::<4>(
                    _mm512_setzero_si512(),
                    active,
                    ids,
                    base.cast(),
                );
                _mm512_mask_storeu_epi32(dst.cast(), active, values);
            }
        })
    }
}

/// The dictionary in up to eight 64-byte tables, looked up by permutation.
///
/// # Safety
/// As [`blocks`]; also `dict.len() * size_of::<T>() <= 4 * SMALL_ENTRIES`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn avx512_small<T>(
    data: &[u8],
    width: u8,
    dict: &[T],
    out: &mut [MaybeUninit<T>],
) -> Result<usize> {
    let bytes = std::mem::size_of_val(dict);
    debug_assert!(bytes <= 4 * SMALL_ENTRIES);
    let mut table = [0u8; 4 * SMALL_ENTRIES];
    // SAFETY: T is a plain primitive; its bytes are copied, not interpreted.
    table[..bytes]
        .copy_from_slice(unsafe { std::slice::from_raw_parts(dict.as_ptr().cast(), bytes) });
    // SAFETY: the table holds 512 readable bytes, eight 64-byte loads.
    let t: [__m512i; 8] =
        std::array::from_fn(|i| unsafe { _mm512_loadu_si512(table.as_ptr().add(64 * i).cast()) });
    // Each pair of tables covers 32 four-byte or 16 eight-byte entries; the
    // next two index bits pick the pair. Padding entries are never selected:
    // `blocks` rejects IDs at or above the dictionary length.
    let pairs = bytes.div_ceil(128).max(1);
    let lookup = |ids: __m512i, pair_bit: i32| -> __m512i {
        if size_of::<T>() == 8 {
            let bit = |b: i32| _mm512_test_epi64_mask(ids, _mm512_set1_epi64(b as i64));
            let x = _mm512_permutex2var_epi64(t[0], ids, t[1]);
            if pairs == 1 {
                return x;
            }
            let y = _mm512_permutex2var_epi64(t[2], ids, t[3]);
            let low = _mm512_mask_blend_epi64(bit(pair_bit), x, y);
            if pairs == 2 {
                return low;
            }
            let z = _mm512_permutex2var_epi64(t[4], ids, t[5]);
            let w = _mm512_permutex2var_epi64(t[6], ids, t[7]);
            let high = _mm512_mask_blend_epi64(bit(pair_bit), z, w);
            _mm512_mask_blend_epi64(bit(2 * pair_bit), low, high)
        } else {
            let bit = |b: i32| _mm512_test_epi32_mask(ids, _mm512_set1_epi32(b));
            let x = _mm512_permutex2var_epi32(t[0], ids, t[1]);
            if pairs == 1 {
                return x;
            }
            let y = _mm512_permutex2var_epi32(t[2], ids, t[3]);
            let low = _mm512_mask_blend_epi32(bit(pair_bit), x, y);
            if pairs == 2 {
                return low;
            }
            let z = _mm512_permutex2var_epi32(t[4], ids, t[5]);
            let w = _mm512_permutex2var_epi32(t[6], ids, t[7]);
            let high = _mm512_mask_blend_epi32(bit(pair_bit), z, w);
            _mm512_mask_blend_epi32(bit(2 * pair_bit), low, high)
        }
    };
    // SAFETY: `blocks` passes only validated IDs; the stores write the lanes
    // of this block inside `out`.
    unsafe {
        blocks(data, width, dict, out, |ids, dst, active| {
            if size_of::<T>() == 8 {
                let lo = lookup(_mm512_cvtepu32_epi64(_mm512_castsi512_si256(ids)), 16);
                _mm512_storeu_si512(dst.cast(), lo);
                if active == u16::MAX {
                    let hi = lookup(
                        _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64::<1>(ids)),
                        16,
                    );
                    _mm512_storeu_si512(dst.add(8).cast(), hi);
                }
            } else {
                _mm512_mask_storeu_epi32(dst.cast(), active, lookup(ids, 32));
            }
        })
    }
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
        crate::util::native::dispatch::for_each_tier(|| {
            let mut state = 0x52a1bc109d32u64;
            for width in 1..=32 {
                let dict: Vec<i64> = (0..(1usize
                    << if (18..=24).contains(&width) && !cfg!(miri) {
                        width
                    } else {
                        width.min(17)
                    }))
                    .map(|i| (i as i64).wrapping_mul(i64::MIN + 17))
                    .collect();
                for len in [0, 1, 7, 8, 15, 16, 17, 31, 32, 33, 127, 1024, 1031] {
                    let ids: Vec<u32> = (0..len)
                        .map(|i| {
                            state ^= state << 13;
                            state ^= state >> 7;
                            state ^= state << 17;
                            if i == 0 {
                                (dict.len() - 1) as u32
                            } else if i < width && dict.len() == 1usize << width {
                                1u32 << i
                            } else {
                                (state as usize % dict.len()) as u32
                            }
                        })
                        .collect();
                    let data = packed(&ids, width);
                    for cut in 0..=data.len().min(35) {
                        let data = &data[..data.len() - cut];
                        let mut out = vec![MaybeUninit::<i64>::uninit(); len + 3];
                        for slot in &mut out[len..] {
                            slot.write(i64::MAX);
                        }
                        if let Some(result) = decode(data, width as u8, &dict, &mut out[..len]) {
                            let n = result.unwrap();
                            assert!(n <= len && n * width <= data.len() * 8);
                            for i in 0..n {
                                // SAFETY: successful decode initializes exactly this prefix.
                                assert_eq!(unsafe { out[i].assume_init() }, dict[ids[i] as usize]);
                            }
                        }
                        for slot in &out[len..] {
                            // SAFETY: sentinels were initialized before decode.
                            assert_eq!(unsafe { slot.assume_init() }, i64::MAX);
                        }
                    }
                    #[cfg(target_arch = "x86_64")]
                    if width <= 24
                        && crate::util::native::dispatch::tier()
                            != crate::util::native::dispatch::Tier::Scalar
                        && !cfg!(miri)
                        && std::is_x86_feature_detected!("avx2")
                    {
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
        });
    }

    /// Every width decodes its high bits: an ID above the dictionary is
    /// reported exactly, whichever lane and block it is in.
    #[test]
    fn test_fused_reports_exact_invalid_ids_for_all_widths() {
        crate::util::native::dispatch::for_each_tier(|| {
            for width in 1..=32usize {
                let top = ((1u64 << width) - 1) as u32;
                for invalid in [top, top ^ 1, (top >> 1) + 1] {
                    if invalid < 3 {
                        continue;
                    }
                    for position in [0, 5, 8, 15, 16, 23, 47] {
                        let mut ids: Vec<u32> = (0..48).map(|i| i % 3).collect();
                        ids[position] = invalid;
                        let mut data = packed(&ids, width);
                        data.extend_from_slice(&[0; 16]);
                        let mut out = [MaybeUninit::<i64>::uninit(); 48];
                        if let Some(Err(error)) =
                            decode(&data, width as u8, &[1i64, 2, 3], &mut out)
                        {
                            assert!(
                                error
                                    .to_string()
                                    .ends_with(&format!("the index is {invalid}")),
                                "width={width} position={position}: {error}"
                            );
                        }
                    }
                }
            }
        });
    }

    /// Permutation lookups at every table-pair boundary, both value sizes,
    /// against a gather of the same IDs; and the gather for wide IDs.
    #[test]
    fn test_fused_small_dictionaries_and_wide_ids() {
        crate::util::native::dispatch::for_each_tier(|| {
            let mut state = 0x9e37_79b9_7f4a_7c15u64;
            for len in [
                1usize, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 300,
            ] {
                let d64: Vec<i64> = (0..len)
                    .map(|i| (i as i64 + 1).wrapping_mul(-0x1234_5678_9abc))
                    .collect();
                let d32: Vec<f32> = (0..len)
                    .map(|i| f32::from_bits(0x7fc0_0000 | i as u32))
                    .collect();
                for width in [
                    len.next_power_of_two().trailing_zeros().max(1) as usize,
                    9,
                    17,
                    25,
                    32,
                ] {
                    for count in [8usize, 16, 24, 504, 1000] {
                        let ids: Vec<u32> = (0..count)
                            .map(|i| {
                                state ^= state << 13;
                                state ^= state >> 7;
                                state ^= state << 17;
                                if i == 0 {
                                    (len - 1) as u32
                                } else {
                                    (state % len as u64) as u32
                                }
                            })
                            .collect();
                        let data = packed(&ids, width);
                        let mut o64 = vec![MaybeUninit::<i64>::uninit(); count];
                        let mut o32 = vec![MaybeUninit::<f32>::uninit(); count];
                        if let Some(n) = decode(&data, width as u8, &d64, &mut o64) {
                            let n = n.unwrap();
                            assert!(n <= count && n % 8 == 0);
                            for i in 0..n {
                                // SAFETY: a successful decode initializes n slots.
                                assert_eq!(unsafe { o64[i].assume_init() }, d64[ids[i] as usize]);
                            }
                        }
                        if let Some(n) = decode(&data, width as u8, &d32, &mut o32) {
                            let n = n.unwrap();
                            for i in 0..n {
                                // SAFETY: a successful decode initializes n slots.
                                assert_eq!(
                                    unsafe { o32[i].assume_init() }.to_bits(),
                                    d32[ids[i] as usize].to_bits()
                                );
                            }
                        }
                    }
                }
            }
        });
    }

    #[test]
    fn test_fused_dictionary_types_and_invalid_ids() {
        crate::util::native::dispatch::for_each_tier(|| {
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
            for width in 1..=32 {
                for invalid_id in [1u32, 1u32 << (width - 1), ((1u64 << width) - 1) as u32] {
                    for position in [0, 7, 8, 15, 16, 31, 63] {
                        let mut ids = vec![0u32; 64];
                        ids[position] = invalid_id;
                        let data = packed(&ids, width);
                        let mut out = [MaybeUninit::uninit(); 64];
                        // AVX2 can leave a short suffix for scalar decoding.
                        if let Some(Ok(n)) = decode(&data, width as u8, &[42i64], &mut out) {
                            assert!(position >= n);
                        }
                    }
                }
            }
            assert!(decode(&[0; 32], 2, &["abc"], &mut [MaybeUninit::uninit(); 64]).is_none());
        });
    }
}
