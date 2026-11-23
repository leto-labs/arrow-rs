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

//! Maintain the decoder's word mask as compact predicate results narrow it.

use arrow_buffer::bit_util::expand;
use arrow_buffer::{BooleanBuffer, Buffer};

/// An empty scratch vector means the current bitmap has not been loaded yet.
pub(super) fn selection_words<'a>(words: &'a mut Vec<u64>, bitmap: &BooleanBuffer) -> &'a [u64] {
    if words.is_empty() {
        words.extend(bitmap.bit_chunks().iter_padded());
    }
    words
}

/// Scatter dense predicate results into the previously kept positions. The
/// decoder reuses the updated words, while previously read columns retain
/// snapshots of the bitmap they were read with.
pub(super) fn narrow_selection(
    words: &mut [u64],
    rows: usize,
    passed: &BooleanBuffer,
) -> BooleanBuffer {
    narrow_words(words, passed);
    #[cfg(target_endian = "little")]
    let buffer = Buffer::from_slice_ref(words);
    #[cfg(target_endian = "big")]
    let buffer = Buffer::from_vec(words.iter().map(|word| word.to_le()).collect());
    BooleanBuffer::new(buffer, 0, rows)
}

fn narrow_words(words: &mut [u64], passed: &BooleanBuffer) {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    if std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("bmi2")
        && std::is_x86_feature_detected!("popcnt")
    {
        // AVX-512 identifies x86 CPUs with fast PDEP, as in the code-page reader.
        // SAFETY: BMI2 and POPCNT have been checked on this CPU.
        unsafe { narrow_words_bmi2(words, passed) };
        return;
    }
    narrow_words_with(words, passed, expand);
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "bmi2,popcnt")]
fn narrow_words_bmi2(words: &mut [u64], passed: &BooleanBuffer) {
    narrow_words_with(words, passed, |bits, mask| {
        std::arch::x86_64::_pdep_u64(bits, mask)
    });
}

#[inline(always)]
fn narrow_words_with(words: &mut [u64], passed: &BooleanBuffer, deposit: impl Fn(u64, u64) -> u64) {
    let chunks = passed.bit_chunks();
    let mut dense = chunks.iter_padded();
    let mut current = dense.next().unwrap_or_default();
    let mut offset = 0;
    for mask in words {
        if *mask == 0 {
            continue;
        }
        let width = mask.count_ones();
        let mut bits = current >> offset;
        let remaining = 64 - offset;
        if width >= remaining {
            current = dense.next().unwrap_or_default();
            if width > remaining {
                bits |= current << remaining;
            }
        }
        *mask = deposit(bits, *mask);
        offset = (offset + width) % 64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staged_word_narrowing_matches_scalar_remap_after_each_predicate() {
        for rows in [0, 1, 63, 64, 65, 127, 129, 1021] {
            for stride in [1, 2, 3, 63, 67] {
                let mut expected =
                    BooleanBuffer::collect_bool(rows + 11, |i| i % stride == 0).slice(5, rows);
                let mut bitmap = expected.clone();
                let mut words = Vec::new();
                let mut scalar = selection_words(&mut words, &bitmap).to_vec();
                for step in 0..3 {
                    let kept = expected.count_set_bits();
                    let passed = BooleanBuffer::collect_bool(kept + 18, |i| (i + step) % 7 < 5)
                        .slice(13, kept);
                    let mut passed_bits = passed.iter();
                    expected = BooleanBuffer::collect_bool(rows, |row| {
                        expected.value(row) && passed_bits.next().unwrap()
                    });

                    narrow_words_with(&mut scalar, &passed, expand);
                    bitmap = narrow_selection(&mut words, rows, &passed);
                    assert_eq!(
                        bitmap, expected,
                        "{rows} rows, stride {stride}, step {step}"
                    );
                    assert_eq!(words, scalar);
                    assert!(
                        words
                            .iter()
                            .copied()
                            .eq(expected.bit_chunks().iter_padded())
                    );
                }
            }
        }
    }
}
