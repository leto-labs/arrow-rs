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
use arrow_array::builder::{Int32Builder, ListBuilder, StringDictionaryBuilder};
use arrow_array::types::Int32Type;
use arrow_array::{BinaryViewArray, Int64Array, StringArray, StringViewArray, UInt32Array};
use arrow_schema::{Field, Schema};

fn collect(c: &mut BatchCoalescer) -> Vec<RecordBatch> {
    c.finish_buffered_batch().unwrap();
    std::iter::from_fn(|| c.next_completed_batch()).collect()
}

#[test]
fn borrowed_ranges_match_materialized_slices() {
    let mut lists = ListBuilder::new(Int32Builder::new());
    let mut dictionary = StringDictionaryBuilder::<Int32Type>::new();
    for i in 0..41 {
        lists.values().append_value(i);
        lists.values().append_null();
        lists.append(i % 3 != 0);
        if i % 5 == 0 {
            dictionary.append_null();
        } else {
            dictionary.append(format!("key{}", i % 3)).unwrap();
        }
    }
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from_iter(
            (0..41).map(|i| (i % 4 != 0).then_some(i)),
        )),
        Arc::new(StringArray::from_iter((0..41).map(|i| {
            (i % 3 != 0).then_some(format!("long-string-value-{i}"))
        }))),
        Arc::new(StringViewArray::from_iter((0..41).map(|i| {
            (i % 3 != 0).then_some(format!("long-view-string-value-{i}"))
        }))),
        Arc::new(BinaryViewArray::from_iter(
            (0..41).map(|i| (i % 4 != 0).then_some(vec![0xff; 20 + i])),
        )),
        Arc::new(lists.finish()),
        Arc::new(dictionary.finish()),
    ];
    let schema = Arc::new(Schema::new(
        arrays
            .iter()
            .enumerate()
            .map(|(i, a)| Field::new(format!("c{i}"), a.data_type().clone(), true))
            .collect::<Vec<_>>(),
    ));
    let batch = RecordBatch::try_new(schema, arrays).unwrap().slice(2, 37);
    for target in [1, 2, 7, 16, 64] {
        for limit in [None, Some(target / 2)] {
            let mut old =
                BatchCoalescer::new(batch.schema(), target).with_biggest_coalesce_batch_size(limit);
            let mut new =
                BatchCoalescer::new(batch.schema(), target).with_biggest_coalesce_batch_size(limit);
            for (offset, len) in [(0, 0), (1, 3), (7, 21), (4, 2), (0, 37), (36, 1)] {
                old.push_batch(batch.slice(offset, len)).unwrap();
                new.push_batch_range(&batch, offset, len).unwrap();
                assert_eq!(old.get_buffered_rows(), new.get_buffered_rows());
                while let Some(expected) = old.next_completed_batch() {
                    assert_eq!(new.next_completed_batch(), Some(expected));
                }
                assert!(!new.has_completed_batch());
            }
            assert_eq!(collect(&mut old), collect(&mut new));
        }
    }
}

#[test]
fn invalid_ranges_leave_buffer_unchanged_and_empty_schema_keeps_rows() {
    let batch = RecordBatch::try_from_iter(vec![(
        "x",
        Arc::new(UInt32Array::from(vec![1, 2, 3])) as ArrayRef,
    )])
    .unwrap();
    let mut c = BatchCoalescer::new(batch.schema(), 4);
    c.push_batch_range(&batch, 0, 1).unwrap();
    for (offset, len) in [(4, 0), (2, 2), (usize::MAX, 1), (0, usize::MAX)] {
        assert!(c.push_batch_range(&batch, offset, len).is_err());
        assert_eq!(c.get_buffered_rows(), 1);
    }
    let empty = RecordBatch::try_new_with_options(
        Arc::new(Schema::empty()),
        vec![],
        &arrow_array::RecordBatchOptions::new().with_row_count(Some(5)),
    )
    .unwrap();
    assert!(c.push_batch_range(&empty, 0, 1).is_err());
    let mut empty_c = BatchCoalescer::new(empty.schema(), 2);
    empty_c.push_batch_range(&empty, 1, 4).unwrap();
    assert_eq!(
        collect(&mut empty_c)
            .iter()
            .map(RecordBatch::num_rows)
            .collect::<Vec<_>>(),
        vec![2, 2]
    );
}

#[test]
fn borrowed_view_range_preserves_sparse_compaction() {
    let strings = StringViewArray::from_iter_values(
        (0..8192).map(|i| format!("{i:08}{}", "retained-source-buffer".repeat(4))),
    );
    let batch = RecordBatch::try_from_iter(vec![("v", Arc::new(strings) as ArrayRef)]).unwrap();
    let mut old = BatchCoalescer::new(batch.schema(), 32);
    let mut new = BatchCoalescer::new(batch.schema(), 32);
    old.push_batch(batch.slice(407, 3)).unwrap();
    new.push_batch_range(&batch, 407, 3).unwrap();
    let expected = collect(&mut old);
    let actual = collect(&mut new);
    assert_eq!(actual, expected);
    let buffer_bytes = |b: &RecordBatch| {
        b.column(0)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap()
            .data_buffers()
            .iter()
            .map(|buffer| buffer.capacity())
            .sum::<usize>()
    };
    assert_eq!(buffer_bytes(&actual[0]), buffer_bytes(&expected[0]));
    assert!(buffer_bytes(&actual[0]) < buffer_bytes(&batch));
}
