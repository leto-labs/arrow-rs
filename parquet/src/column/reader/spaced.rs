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

//! Flat nullable primitives: one dense decode into the output, then an
//! in-place expansion to logical slots, backwards, from the validity bitmap.
pub(super) use crate::util::native::native;
use std::mem::MaybeUninit;

use crate::errors::{ParquetError, Result};

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
        return unsafe { read_popcnt(out, physical, logical, validity, offset, fill) };
    }
    read_impl(out, physical, logical, validity, offset, fill)
}

/// # Safety
/// POPCNT must be available. All memory/data validation stays in read_impl.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
unsafe fn read_popcnt<T: Default + Clone + 'static>(
    out: &mut Vec<T>,
    physical: usize,
    logical: usize,
    validity: &[u8],
    offset: usize,
    fill: impl FnMut(&mut Vec<T>, usize) -> Result<usize>,
) -> Result<usize> {
    read_impl(out, physical, logical, validity, offset, fill)
}

#[inline(always)]
fn read_impl<T: Default + Clone + 'static>(
    out: &mut Vec<T>,
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
    if physical == 0 {
        out.resize(out_end, T::default());
        return Ok(0);
    }
    out.reserve(logical);
    let mut guard = RestoreLength { out, keep: start };
    // The physical values land at the start of the logical range, so that
    // the decoder writes one contiguous run however the NULLs fall.
    let n = fill(&mut *guard.out, physical)?;
    if n != physical || guard.out.len() != start + physical {
        return Err(general_err!(
            "insufficient values read from column - expected: {}, got: {}",
            physical,
            n
        ));
    }
    if physical != logical {
        guard.out.reserve(logical - physical);
        // SAFETY: the capacity covers start + logical slots, the first
        // `physical` of which `fill` initialized (checked by the length
        // above). MaybeUninit admits the uninitialized rest. The slice is the
        // only access to the buffer until it ends.
        let slots = unsafe {
            std::slice::from_raw_parts_mut(
                guard.out.as_mut_ptr().add(start).cast::<MaybeUninit<T>>(),
                logical,
            )
        };
        expand(slots, physical, validity, offset)?;
    }
    // SAFETY: native T has no drop behavior and every logical slot is
    // initialized: by fill when all are valid, else by expand, including
    // zero/default values in NULL positions. Errors and panics before this
    // point leave the original Vec length intact.
    unsafe {
        guard.out.set_len(out_end);
    }
    guard.keep = out_end;
    Ok(physical)
}

/// Moves the `physical` dense values at the start of `slots` to their logical
/// slots and zeroes the NULL slots, from the last block to the first. A dense
/// value never sits after its slot (its index counts only the valid slots
/// before it), and a block's own values are loaded before its store, so no
/// value is overwritten before it is moved.
fn expand<T: Default + Clone + 'static>(
    slots: &mut [MaybeUninit<T>],
    physical: usize,
    validity: &[u8],
    offset: usize,
) -> Result<()> {
    let logical = slots.len();
    if !native::<T>()
        || physical > logical
        || offset
            .checked_add(logical)
            .is_none_or(|end| end.div_ceil(8) > validity.len())
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
        // SAFETY: exact primitive types and AVX512F/POPCNT are checked; the
        // bitmap extent and the dense count are checked above, and the kernel
        // checks every block's count against the values left.
        return unsafe { expand_avx512(slots, physical, validity, offset) };
    }
    let mut dense = physical;
    for i in (0..logical).rev() {
        let pos = offset + i;
        if validity[pos / 8] & (1 << (pos % 8)) != 0 {
            let Some(from) = dense.checked_sub(1) else {
                return Err(general_err!("validity/value count mismatch"));
            };
            dense = from;
            // SAFETY: slots below `physical` are initialized, and a slot at
            // or below `from` <= i has not been overwritten yet.
            let value = unsafe { slots[from].assume_init_read() };
            slots[i].write(value);
        } else {
            slots[i].write(T::default());
        }
    }
    if dense != 0 {
        return Err(general_err!("validity/value count mismatch"));
    }
    Ok(())
}

/// # Safety
/// AVX512F and POPCNT must be available; T is exactly i32/i64/f32/f64. The
/// first `physical` slots are initialized and `validity` holds the bits
/// `offset..offset + slots.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,popcnt")]
unsafe fn expand_avx512<T>(
    slots: &mut [MaybeUninit<T>],
    physical: usize,
    validity: &[u8],
    offset: usize,
) -> Result<()> {
    use std::arch::x86_64::*;
    debug_assert!(matches!(size_of::<T>(), 4 | 8));
    let logical = slots.len();
    let lanes = if size_of::<T>() == 8 { 8 } else { 16 };
    let base = slots.as_mut_ptr();
    let mut dense = physical;
    let mut block = logical.div_ceil(lanes);
    while block > 0 {
        block -= 1;
        let first = block * lanes;
        let count = (logical - first).min(lanes);
        let valid = mask_at(validity, offset + first, count);
        let take = valid.count_ones() as usize;
        if take > dense {
            return Err(general_err!("validity/value count mismatch"));
        }
        dense -= take;
        // SAFETY: the masked load reads exactly `take` initialized values at
        // `dense..dense + take` (below `physical`, not yet overwritten: they
        // lie at or below `first`, the lowest slot stored so far); masked-off
        // lanes do not access memory. The expansion is in registers: the
        // memory form of VPEXPAND is microcoded on Zen 4. The store writes
        // `count` slots inside `slots`, NULL lanes as zero.
        unsafe {
            let from = base.add(dense).cast::<u8>();
            let to = base.add(first).cast::<u8>();
            let packed = low_mask(take);
            if size_of::<T>() == 8 {
                let values = _mm512_maskz_loadu_epi64(packed as u8, from.cast());
                let values = _mm512_maskz_expand_epi64(valid as u8, values);
                _mm512_mask_storeu_epi64(to.cast(), low_mask(count) as u8, values);
            } else {
                let values = _mm512_maskz_loadu_epi32(packed as u16, from.cast());
                let values = _mm512_maskz_expand_epi32(valid as u16, values);
                _mm512_mask_storeu_epi32(to.cast(), low_mask(count) as u16, values);
            }
        }
    }
    if dense != 0 {
        return Err(general_err!("validity/value count mismatch"));
    }
    Ok(())
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
                    let mut cursor = 0;
                    let n = read(&mut out, dense.len(), len, &bitmap, offset, |dst, want| {
                        dst.extend_from_slice(&dense[cursor..cursor + want]);
                        cursor += want;
                        Ok(want)
                    })
                    .unwrap();
                    assert_eq!(n, dense.len());
                    assert_eq!(cursor, dense.len());
                    assert_eq!(out.len(), 3 + len);
                    assert!(out[..3].iter().all(|x| bits(x) == bits(&prefix)));
                    for (a, b) in out[3..].iter().zip(&expected) {
                        assert_eq!(bits(a), bits(b));
                    }
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
            for (physical, logical, bitmap, offset) in [
                (1, 1, vec![], 0),
                (3, 8, vec![255], 0),
                (8, 8, vec![254], 0),
                (0, 8, vec![1], 0),
                (1, 2, vec![255], usize::MAX),
            ] {
                assert!(
                    read(&mut out, physical, logical, &bitmap, offset, |_, _| panic!(
                        "invalid bitmap must not reach decoder"
                    ))
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
                        read(&mut out, physical, logical, &bitmap, 0, |dst, want| {
                            calls += 1;
                            // One decode call fills the dense prefix of the output.
                            if calls == 1 {
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
                        })
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
                let error = read(&mut out, physical, logical, &bitmap, 0, |dst, want| {
                    dst.extend(std::iter::repeat_n(7, want - 1));
                    Ok(want - 1)
                })
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

    /// The in-place expansion against a reference: every dense prefix and
    /// bitmap pattern, unaligned offsets, and sentinels after the slots.
    #[test]
    fn test_expand_in_place_masks_and_output_sentinels() {
        crate::util::native::dispatch::for_each_tier(|| {
            let mut state = 0x2545_f491_4f6c_dd1du64;
            for logical in (0usize..=70).chain([127, 128, 129, 1000]) {
                for offset in [0usize, 1, 7, 13] {
                    for density in [0u64, 1, 50, 99, 100] {
                        let mut bitmap = vec![0u8; (offset + logical).div_ceil(8) + 1];
                        let mut physical = 0;
                        for i in 0..logical {
                            state ^= state << 13;
                            state ^= state >> 7;
                            state ^= state << 17;
                            if state % 100 < density {
                                let pos = offset + i;
                                bitmap[pos / 8] |= 1 << (pos % 8);
                                physical += 1;
                            }
                        }
                        let mut slots = vec![MaybeUninit::new(i64::MAX); logical + 3];
                        for (i, slot) in slots[..physical].iter_mut().enumerate() {
                            slot.write(i64::MIN + i as i64);
                        }
                        expand(&mut slots[..logical], physical, &bitmap, offset).unwrap();
                        let mut cursor = 0;
                        for (i, slot) in slots.iter().enumerate() {
                            let pos = offset + i;
                            let expected = if i >= logical {
                                i64::MAX
                            } else if bitmap[pos / 8] & (1 << (pos % 8)) != 0 {
                                cursor += 1;
                                i64::MIN + cursor as i64 - 1
                            } else {
                                0
                            };
                            // SAFETY: all slots were initialized before expand,
                            // whose successful call initialized every logical slot.
                            assert_eq!(unsafe { slot.assume_init() }, expected);
                        }
                        if physical > 0 {
                            // A count that does not match the bitmap is an error.
                            let mut slots = vec![MaybeUninit::new(0i32); logical];
                            assert!(expand(&mut slots, physical - 1, &bitmap, offset).is_err());
                        }
                    }
                }
            }
            let mut slots = [MaybeUninit::<bool>::uninit(); 1];
            assert!(expand(&mut slots, 1, &[1], 0).is_err());
            let mut slots = [MaybeUninit::<i64>::uninit(); 9];
            assert!(expand(&mut slots, 1, &[1], 0).is_err());
            assert!(expand(&mut slots[..8], 9, &[255], 0).is_err());
        });
    }
}
