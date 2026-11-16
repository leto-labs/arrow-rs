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

//! Flat nullable primitives: dense decode windows, sequential logical output.
pub(super) use crate::util::native::native;
use std::mem::MaybeUninit;

use crate::errors::{ParquetError, Result};

const WINDOW: usize = 1024;

fn low_mask(count: usize) -> u64 {
    if count == 64 {
        u64::MAX
    } else {
        (1u64 << count) - 1
    }
}

/// Read up to 64 bits from a range already checked against the bitmap extent.
fn mask_at(data: &[u8], offset: usize, count: usize) -> u64 {
    let byte = offset / 8;
    let shift = offset % 8;
    let word = if data.len() - byte >= 8 {
        u64::from_le_bytes(data[byte..byte + 8].try_into().unwrap())
    } else {
        let mut bytes = [0; 8];
        let len = data.len() - byte;
        bytes[..len].copy_from_slice(&data[byte..]);
        u64::from_le_bytes(bytes)
    };
    let mut mask = word >> shift;
    if shift != 0 && byte + 8 < data.len() {
        mask |= (data[byte + 8] as u64) << (64 - shift);
    }
    mask & low_mask(count)
}

/// Restore the previous initialized length on a failed or panicking append.
struct RestoreLength<'a, T> {
    out: &'a mut Vec<T>,
    keep: usize,
}
impl<T> Drop for RestoreLength<'_, T> {
    fn drop(&mut self) {
        self.out.truncate(self.keep);
    }
}

/// `fill` appends initialized physical values. Vec's initialized length, rather
/// than a claimed read count alone, bounds every subsequent native load.
pub(super) fn read<T: Default + Clone + 'static>(
    out: &mut Vec<T>,
    dense: &mut Vec<T>,
    physical: usize,
    logical: usize,
    validity: &[u8],
    offset: usize,
    fill: impl FnMut(&mut Vec<T>, usize) -> Result<usize>,
) -> Result<usize> {
    #[cfg(all(test, target_arch = "x86_64"))]
    let allow_native =
        crate::util::native::dispatch::tier() != crate::util::native::dispatch::Tier::Scalar;
    #[cfg(all(not(test), target_arch = "x86_64"))]
    let allow_native = true;
    #[cfg(target_arch = "x86_64")]
    if allow_native && !cfg!(miri) && std::is_x86_feature_detected!("popcnt") {
        // SAFETY: POPCNT is checked here. read_impl validates all value, bitmap,
        // and output extents independently of this CPU specialization.
        return unsafe { read_popcnt(out, dense, physical, logical, validity, offset, fill) };
    }
    read_impl(out, dense, physical, logical, validity, offset, fill)
}

/// # Safety
/// POPCNT must be available. All memory/data validation stays in read_impl.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
unsafe fn read_popcnt<T: Default + Clone + 'static>(
    out: &mut Vec<T>,
    dense: &mut Vec<T>,
    physical: usize,
    logical: usize,
    validity: &[u8],
    offset: usize,
    fill: impl FnMut(&mut Vec<T>, usize) -> Result<usize>,
) -> Result<usize> {
    read_impl(out, dense, physical, logical, validity, offset, fill)
}

#[inline(always)]
fn read_impl<T: Default + Clone + 'static>(
    out: &mut Vec<T>,
    dense: &mut Vec<T>,
    physical: usize,
    logical: usize,
    validity: &[u8],
    offset: usize,
    mut fill: impl FnMut(&mut Vec<T>, usize) -> Result<usize>,
) -> Result<usize> {
    if !native::<T>() {
        return Err(general_err!("unsupported spaced primitive"));
    }
    let end = offset
        .checked_add(logical)
        .ok_or_else(|| general_err!("validity range overflow"))?;
    if end.div_ceil(8) > validity.len() {
        return Err(general_err!("short validity bitmap"));
    }
    let counts_match = if physical == logical {
        (0..logical).step_by(64).all(|pos| {
            let count = (logical - pos).min(64);
            mask_at(validity, offset + pos, count) == low_mask(count)
        })
    } else if physical == 0 {
        (0..logical)
            .step_by(64)
            .all(|pos| mask_at(validity, offset + pos, (logical - pos).min(64)) == 0)
    } else {
        let mut valid_count = 0usize;
        for pos in (0..logical).step_by(64) {
            valid_count +=
                mask_at(validity, offset + pos, (logical - pos).min(64)).count_ones() as usize;
        }
        valid_count == physical
    };
    if !counts_match {
        return Err(general_err!("validity/value count mismatch"));
    }
    if logical == 0 {
        return Ok(0);
    }
    let start = out.len();
    let out_end = start
        .checked_add(logical)
        .ok_or_else(|| general_err!("output length overflow"))?;
    if physical == logical {
        let mut guard = RestoreLength { out, keep: start };
        let n = fill(&mut *guard.out, physical)?;
        if n != physical || guard.out.len() != out_end {
            return Err(general_err!(
                "insufficient values read from column - expected: {}, got: {}",
                physical,
                n
            ));
        }
        guard.keep = out_end;
        return Ok(n);
    }
    if physical == 0 {
        out.resize(out_end, T::default());
        return Ok(0);
    }
    out.reserve(logical);
    dense.clear();
    // Fix the window capacity before appending. Amortized growth after small
    // pages could otherwise allocate more than WINDOW on a later large page.
    if dense.capacity() < WINDOW {
        dense.reserve_exact(WINDOW);
    }
    let mut remaining = physical;
    let mut cursor = 0;
    let mut written = 0;
    while written < logical {
        if cursor == dense.len() && remaining != 0 {
            dense.clear();
            let want = remaining.min(WINDOW);
            let n = fill(dense, want)?;
            if n != want || dense.len() != want {
                let got = (physical - remaining).saturating_add(n);
                return Err(general_err!(
                    "insufficient values read from column - expected: {}, got: {}",
                    physical,
                    got
                ));
            }
            remaining -= n;
            cursor = 0;
        }
        let mut count = (logical - written).min(64);
        let mut mask = mask_at(validity, offset + written, count);
        let available = dense.len() - cursor;
        if mask.count_ones() as usize > available {
            if available == 0 {
                return Err(general_err!("short spaced value window"));
            }
            // End this logical prefix at the final value in the dense window.
            // Every following decode still requests 1024 physical values, so
            // dictionary bit-unpacking retains its byte-aligned bulk path.
            let mut bits = mask;
            for _ in 1..available {
                bits &= bits - 1;
            }
            count = bits.trailing_zeros() as usize + 1;
            mask &= low_mask(count);
        }
        let take = mask.count_ones() as usize;
        expand(
            &dense[cursor..cursor + take],
            &mut out.spare_capacity_mut()[written..written + count],
            mask,
            count,
        )?;
        cursor += take;
        written += count;
    }
    // SAFETY: native T has no drop behavior and every logical slot was written
    // by expand, including initialized zero/default values in NULL positions.
    // Errors and panics before this point leave the original Vec length intact.
    unsafe {
        out.set_len(out_end);
    }
    Ok(physical)
}

#[inline(always)]
fn expand<T: Default + Clone + 'static>(
    dense: &[T],
    out: &mut [MaybeUninit<T>],
    mask: u64,
    count: usize,
) -> Result<()> {
    if !native::<T>()
        || count > 64
        || count > out.len()
        || mask & !low_mask(count) != 0
        || mask.count_ones() as usize > dense.len()
    {
        return Err(general_err!("invalid primitive expansion extent"));
    }
    #[cfg(all(test, target_arch = "x86_64"))]
    let allow_avx512 = matches!(
        crate::util::native::dispatch::tier(),
        crate::util::native::dispatch::Tier::Auto | crate::util::native::dispatch::Tier::Avx512
    );
    #[cfg(all(not(test), target_arch = "x86_64"))]
    let allow_avx512 = true;
    #[cfg(target_arch = "x86_64")]
    if allow_avx512
        && !cfg!(miri)
        && std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("popcnt")
    {
        // SAFETY: exact primitive types, AVX512F/POPCNT, active input lanes and the
        // complete logical output extent were checked above.
        unsafe {
            expand_avx512(dense, &mut out[..count], mask);
        }
        return Ok(());
    }
    let mut cursor = 0;
    for (i, dst) in out[..count].iter_mut().enumerate() {
        if mask & (1u64 << i) != 0 {
            dst.write(dense[cursor].clone());
            cursor += 1;
        } else {
            dst.write(T::default());
        }
    }
    Ok(())
}

/// # Safety
/// AVX512F and POPCNT must be available; T is exactly i32/i64/f32/f64. The mask has no
/// set bits beyond out.len() <=64, and dense contains at least popcount(mask)
/// initialized values. Input and output are disjoint exclusive Rust borrows.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,popcnt")]
unsafe fn expand_avx512<T>(dense: &[T], out: &mut [MaybeUninit<T>], mask: u64) {
    use std::arch::x86_64::*;
    debug_assert!(matches!(size_of::<T>(), 4 | 8));
    let mut cursor = 0;
    let lanes = if size_of::<T>() == 8 { 8 } else { 16 };
    for (block, dst) in out.chunks_mut(lanes).enumerate() {
        let valid = (mask >> (block * lanes)) & low_mask(dst.len());
        // SAFETY: expand-load reads exactly popcount(valid) initialized dense
        // values; inactive lanes do not access memory. Prefix stores initialize
        // every logical slot, including NULLs, and never store beyond dst.
        unsafe {
            if size_of::<T>() == 8 {
                let values =
                    _mm512_maskz_expandloadu_epi64(valid as u8, dense.as_ptr().add(cursor).cast());
                _mm512_mask_storeu_epi64(
                    dst.as_mut_ptr().cast(),
                    low_mask(dst.len()) as u8,
                    values,
                );
            } else {
                let values =
                    _mm512_maskz_expandloadu_epi32(valid as u16, dense.as_ptr().add(cursor).cast());
                _mm512_mask_storeu_epi32(
                    dst.as_mut_ptr().cast(),
                    low_mask(dst.len()) as u16,
                    values,
                );
            }
        }
        cursor += valid.count_ones() as usize;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check<T: Clone + Default + 'static>(make: fn(u64) -> T, bits: fn(&T) -> u64) {
        let lengths: &[usize] = if cfg!(miri) {
            &[0, 7, 16, 65, 1031]
        } else {
            &[
                0, 1, 7, 8, 15, 16, 31, 63, 64, 65, 1023, 1024, 1025, 2057, 8199,
            ]
        };
        for &len in lengths {
            for offset in 0..8 {
                for pattern in 0..5 {
                    let mut bitmap = vec![0xa5; (offset + len).div_ceil(8)];
                    let mut dense = Vec::new();
                    let mut expected = Vec::new();
                    for i in 0..len {
                        let value = make((i as u64).wrapping_mul(0x9e3779b97f4a7c15));
                        let valid = match pattern {
                            0 => true,
                            1 => false,
                            2 => i % 2 == 0,
                            3 => i % 100 == 0,
                            _ => (i * 37 + 19) % 101 < 61,
                        };
                        let pos = offset + i;
                        bitmap[pos / 8] &= !(1 << (pos % 8));
                        if valid {
                            bitmap[pos / 8] |= 1 << (pos % 8);
                            dense.push(value.clone());
                        }
                        expected.push(if valid { value } else { T::default() });
                    }
                    let prefix = make(0x123456789abcdef0);
                    let mut out = vec![prefix.clone(); 3];
                    let mut cache = Vec::new();
                    let mut cursor = 0;
                    let n = read(
                        &mut out,
                        &mut cache,
                        dense.len(),
                        len,
                        &bitmap,
                        offset,
                        |dst, want| {
                            dst.extend_from_slice(&dense[cursor..cursor + want]);
                            cursor += want;
                            Ok(want)
                        },
                    )
                    .unwrap();
                    assert_eq!(n, dense.len());
                    assert_eq!(cursor, dense.len());
                    assert_eq!(out.len(), 3 + len);
                    assert!(out[..3].iter().all(|x| bits(x) == bits(&prefix)));
                    for (a, b) in out[3..].iter().zip(&expected) {
                        assert_eq!(bits(a), bits(b));
                    }
                    assert!(cache.len() <= WINDOW);
                }
            }
        }
    }

    #[test]
    fn test_spaced_primitive_masks_and_prefixes() {
        crate::util::native::dispatch::for_each_tier(|| {
            check(|x| x as i64, |x| *x as u64);
            if !cfg!(miri) {
                check(|x| x as i32, |x| *x as u32 as u64);
                check(f64::from_bits, |x| x.to_bits());
                check(|x| f32::from_bits(x as u32), |x| x.to_bits() as u64);
            }
        });
    }

    #[test]
    fn test_spaced_rejects_bad_extents_and_short_reads() {
        crate::util::native::dispatch::for_each_tier(|| {
            let mut out = vec![-17i64, 42];
            let mut cache = Vec::new();
            for (physical, logical, bitmap, offset) in [
                (1, 1, vec![], 0),
                (3, 8, vec![255], 0),
                (8, 8, vec![254], 0),
                (0, 8, vec![1], 0),
                (1, 2, vec![255], usize::MAX),
            ] {
                assert!(
                    read(
                        &mut out,
                        &mut cache,
                        physical,
                        logical,
                        &bitmap,
                        offset,
                        |_, _| panic!("invalid bitmap must not reach decoder")
                    )
                    .is_err()
                );
                assert_eq!(out, [-17, 42]);
            }
            for all_valid in [false, true] {
                let logical: usize = 3073;
                let mut bitmap = vec![0; logical.div_ceil(8)];
                let mut physical = 0;
                for i in 0..logical {
                    if all_valid || i % 3 != 0 {
                        bitmap[i / 8] |= 1 << (i % 8);
                        physical += 1;
                    }
                }
                for failure in 0..4 {
                    let mut calls = 0;
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        read(
                            &mut out,
                            &mut cache,
                            physical,
                            logical,
                            &bitmap,
                            0,
                            |dst, want| {
                                calls += 1;
                                // For the mixed bitmap fail after some logical slots have
                                // already been initialized in spare capacity.
                                if all_valid || calls == 2 {
                                    if failure == 0 {
                                        return Err(general_err!("injected read error"));
                                    }
                                    if failure == 1 {
                                        dst.extend(std::iter::repeat_n(5, want - 1));
                                        return Ok(want - 1);
                                    }
                                    if failure == 2 {
                                        dst.extend(std::iter::repeat_n(5, want));
                                        return Ok(want + 1);
                                    }
                                    dst.push(9);
                                    panic!("injected read panic");
                                }
                                dst.extend(std::iter::repeat_n(7, want));
                                Ok(want)
                            },
                        )
                    }));
                    assert!(result.is_err() || result.unwrap().is_err());
                    assert_eq!(out, [-17, 42]);
                }
            }
        });
    }

    #[test]
    fn test_review_spaced_short_read_error_fields() {
        crate::util::native::dispatch::for_each_tier(|| {
            for (physical, logical, bitmap) in [(8, 8, [255]), (4, 8, [85])] {
                let mut out = vec![-17i64];
                let error = read(
                    &mut out,
                    &mut Vec::new(),
                    physical,
                    logical,
                    &bitmap,
                    0,
                    |dst, want| {
                        dst.extend(std::iter::repeat_n(7, want - 1));
                        Ok(want - 1)
                    },
                )
                .unwrap_err();
                assert_eq!(
                    error.to_string(),
                    format!(
                        "Parquet error: insufficient values read from column - expected: {physical}, got: {}",
                        physical - 1
                    )
                );
                assert_eq!(out, [-17]);
            }
        });
    }

    #[test]
    fn test_spaced_reuses_bounded_window() {
        crate::util::native::dispatch::for_each_tier(|| {
            let mut out = Vec::<i64>::new();
            let mut cache = Vec::new();
            for physical in [301usize, 601, 1023] {
                let logical = physical + 1;
                let mut bitmap = vec![255; logical.div_ceil(8)];
                bitmap[physical / 8] &= !(1 << (physical % 8));
                let start = out.len();
                read(
                    &mut out,
                    &mut cache,
                    physical,
                    logical,
                    &bitmap,
                    0,
                    |dst, want| {
                        dst.reserve(want);
                        dst.extend(std::iter::repeat_n(7, want));
                        Ok(want)
                    },
                )
                .unwrap();
                assert!(
                    cache.capacity() <= WINDOW,
                    "physical cache capacity {}",
                    cache.capacity()
                );
                assert!(out[start..start + physical].iter().all(|v| *v == 7));
                assert_eq!(out[start + physical], 0);
            }
        });
    }

    #[test]
    fn test_expand_masks_and_output_sentinels() {
        crate::util::native::dispatch::for_each_tier(|| {
            for count in 0..=8 {
                for mask in 0u64..=low_mask(count) {
                    let dense: Vec<i64> = (0..mask.count_ones())
                        .map(|i| i64::MIN + i as i64)
                        .collect();
                    let mut out = [MaybeUninit::new(i64::MAX); 11];
                    expand(&dense, &mut out[..count], mask, count).unwrap();
                    let mut cursor = 0;
                    for (i, slot) in out.iter().enumerate() {
                        let expected = if i >= count {
                            i64::MAX
                        } else if mask & (1 << i) != 0 {
                            let value = dense[cursor];
                            cursor += 1;
                            value
                        } else {
                            0
                        };
                        // SAFETY: all sentinels were initialized before expand,
                        // whose successful call initialized every slot in count.
                        assert_eq!(unsafe { slot.assume_init() }, expected);
                    }
                }
            }
            assert!(expand(&[true], &mut [MaybeUninit::<bool>::uninit(); 1], 1, 1).is_err());
            let mut out = [MaybeUninit::<i64>::uninit(); 64];
            assert!(expand(&[], &mut out, 1, 1).is_err());
            assert!(expand(&[7], &mut out[..1], 2, 1).is_err());
            assert!(expand(&[7], &mut out, 0, 65).is_err());
        });
    }
}
