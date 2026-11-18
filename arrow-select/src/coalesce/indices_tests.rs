// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file distributed with
// this work for additional information regarding copyright ownership.
// The ASF licenses this file to You under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with the
// License. You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::*;
use arrow_array::builder::FixedSizeBinaryBuilder;
use arrow_array::cast::AsArray;
use arrow_array::{
    BinaryArray, BinaryViewArray, Int32Array, LargeBinaryArray, LargeStringArray, StringArray,
    StringViewArray, UInt32Array, UInt64Array,
};
use rand::{RngExt, SeedableRng, rngs::StdRng};

fn batch() -> RecordBatch {
    let strings: Vec<Option<String>> = (0..70)
        .map(|i| match i % 4 {
            0 => None,
            1 => Some(format!("short-{i}")),
            _ => Some(format!("long out-of-line view value {i}")),
        })
        .collect();
    let text = || strings.iter().map(Option::as_deref);
    let bytes = || text().map(|s| s.map(str::as_bytes));
    let mut fixed = FixedSizeBinaryBuilder::new(4);
    for i in 0i32..70 {
        if i % 5 == 0 {
            fixed.append_null();
        } else {
            fixed.append_value(i.to_le_bytes()).unwrap();
        }
    }
    let cols: Vec<(&str, ArrayRef)> = vec![
        (
            "int",
            Arc::new(Int32Array::from_iter(
                (0..70).map(|i| (i % 5 != 0).then_some(i)),
            )),
        ),
        (
            "bool",
            Arc::new(BooleanArray::from_iter(
                (0..70).map(|i| (i % 3 != 0).then_some(i % 2 == 0)),
            )),
        ),
        ("utf8", Arc::new(StringArray::from_iter(text()))),
        ("utf8_large", Arc::new(LargeStringArray::from_iter(text()))),
        (
            "utf8_view",
            Arc::new(StringViewArray::from_iter(text()).gc()),
        ),
        ("binary", Arc::new(BinaryArray::from_iter(bytes()))),
        (
            "binary_large",
            Arc::new(LargeBinaryArray::from_iter(bytes())),
        ),
        (
            "binary_view",
            Arc::new(BinaryViewArray::from_iter(bytes()).gc()),
        ),
        ("fixed_binary", Arc::new(fixed.finish())),
    ];
    RecordBatch::try_from_iter(cols).unwrap().slice(3, 64)
}

fn compare_ready(candidate: &mut BatchCoalescer, baseline: &mut BatchCoalescer) {
    assert_eq!(candidate.get_buffered_rows(), baseline.get_buffered_rows());
    while let Some(expected) = baseline.next_completed_batch() {
        let actual = candidate.next_completed_batch().unwrap();
        assert_eq!(actual, expected);
        for i in [4, 7] {
            let caps = |batch: &RecordBatch| {
                let a = batch.column(i);
                if i == 4 {
                    a.as_string_view()
                        .data_buffers()
                        .iter()
                        .map(|b| b.capacity())
                        .sum::<usize>()
                } else {
                    a.as_binary_view()
                        .data_buffers()
                        .iter()
                        .map(|b| b.capacity())
                        .sum()
                }
            };
            assert_eq!(caps(&actual), caps(&expected));
        }
    }
    assert!(!candidate.has_completed_batch());
}

#[test]
fn indexed_push_matches_take_then_push_across_batch_boundaries() {
    let batch = batch();
    let mut rng = StdRng::seed_from_u64(62);
    for target in [1, 2, 7, 16, 64, 128] {
        for bypass in [None, Some(target / 2)] {
            let make = || {
                BatchCoalescer::new(batch.schema(), target).with_biggest_coalesce_batch_size(bypass)
            };
            let (mut candidate, mut baseline) = (make(), make());
            for turn in 0..96 {
                let len = match turn % 8 {
                    0 => 0,
                    1 => target - 1,
                    2 => target,
                    3 => target + 1,
                    _ => rng.random_range(0..=96usize),
                };
                let rows = (0..len)
                    .map(|i| {
                        let j = rng.random_range(0..64);
                        match turn % 8 {
                            // Select only inline views from a source with long views.
                            4 => Some((j / 4) * 4 + 2),
                            // Overlapping/repeated views can be dense enough to retain the source.
                            5 => Some(3),
                            6 if i % 5 == 0 => None,
                            _ => Some(j),
                        }
                    })
                    .collect::<Vec<_>>();
                let indices: ArrayRef = if turn % 2 == 0 {
                    Arc::new(UInt32Array::from(rows))
                } else {
                    Arc::new(UInt64Array::from_iter(
                        rows.into_iter().map(|r| r.map(u64::from)),
                    ))
                };
                baseline
                    .push_batch(take_record_batch(&batch, indices.as_ref()).unwrap())
                    .unwrap();
                candidate
                    .push_batch_with_indices(batch.clone(), indices.as_ref())
                    .unwrap();
                compare_ready(&mut candidate, &mut baseline);
            }
            candidate.finish_buffered_batch().unwrap();
            baseline.finish_buffered_batch().unwrap();
            compare_ready(&mut candidate, &mut baseline);
        }
    }
}

#[test]
fn invalid_indices_leave_state_unchanged() {
    let batch = batch();
    let mut coalescer = BatchCoalescer::new(batch.schema(), 4);
    coalescer.push_batch(batch.slice(0, 2)).unwrap();
    let invalid = UInt32Array::from(vec![3, 64]);
    assert!(
        coalescer
            .push_batch_with_indices(batch.clone(), &invalid)
            .is_err()
    );
    coalescer.finish_buffered_batch().unwrap();
    assert_eq!(coalescer.next_completed_batch(), Some(batch.slice(0, 2)));
}

#[test]
fn staging_reuses_the_coalescer_and_compacts_sparse_views() {
    let batch = batch();
    let mut staging = BatchCoalescer::new(batch.schema(), 2);
    for rows in [&[3, 1, 1, 10, 0][..], &[2, 6, 10, 14], &[], &[0, 4, 8]] {
        let staged = staging.gather_batch_with_u32_indices(&batch, rows).unwrap();
        let expected = take_record_batch(&batch, &UInt32Array::from(rows.to_vec())).unwrap();
        assert_eq!(staged, expected);
        assert!(staging.is_empty());

        for i in [4, 7] {
            let data = staged.column(i);
            let (used, capacity): (usize, usize) = if i == 4 {
                let view = data.as_string_view();
                (
                    view.total_buffer_bytes_used(),
                    view.data_buffers().iter().map(|b| b.capacity()).sum(),
                )
            } else {
                let view = data.as_binary_view();
                (
                    view.total_buffer_bytes_used(),
                    view.data_buffers().iter().map(|b| b.capacity()).sum(),
                )
            };
            assert_eq!(capacity, used);
        }
    }
}
