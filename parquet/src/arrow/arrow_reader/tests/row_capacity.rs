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

//! Value buffers are reserved for the rows a reader can still return, not
//! for a whole batch (Longbow family RRC).

use super::*;
use crate::arrow::arrow_reader::RowSelectionPolicy;

const BATCH_SIZE: usize = 8192;

/// A file of flat columns in `rows.len()` row groups of the given sizes.
fn flat_file(rows: &[usize]) -> Bytes {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", ArrowDataType::Int64, false),
        Field::new("v", ArrowDataType::Int64, true),
        Field::new("f", ArrowDataType::Float64, false),
        Field::new("s", ArrowDataType::Utf8, true),
        Field::new("sv", ArrowDataType::Utf8View, false),
        Field::new("d", ArrowDataType::Decimal128(30, 2), false),
    ]));
    let mut buffer = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buffer, schema.clone(), None).unwrap();
    let mut next = 0i64;
    for &rows in rows {
        let ids = next..next + rows as i64;
        next += rows as i64;
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from_iter_values(ids.clone())),
                Arc::new(Int64Array::from_iter(
                    ids.clone().map(|i| (i % 3 != 0).then_some(i)),
                )),
                Arc::new(Float64Array::from_iter_values(
                    ids.clone().map(|i| i as f64),
                )),
                Arc::new(StringArray::from_iter(
                    ids.clone().map(|i| (i % 5 != 0).then(|| format!("s{i}"))),
                )),
                Arc::new(StringViewArray::from_iter_values(
                    ids.clone().map(|i| format!("a string view of row {i}")),
                )),
                Arc::new(
                    Decimal128Array::from_iter_values(ids.map(i128::from))
                        .with_precision_and_scale(30, 2)
                        .unwrap(),
                ),
            ],
        )
        .unwrap();
        writer.write(&batch).unwrap();
        writer.flush().unwrap();
    }
    writer.close().unwrap();
    Bytes::from(buffer)
}

/// The rows that the column's fixed-width buffer (values, offsets or views)
/// has room for.
fn slots(array: &ArrayRef) -> usize {
    let data = array.to_data();
    let capacity = data.buffers()[0].capacity();
    match array.data_type() {
        ArrowDataType::Int64 | ArrowDataType::Float64 => capacity / 8,
        ArrowDataType::Utf8 => capacity / 4 - 1,
        ArrowDataType::Utf8View | ArrowDataType::Decimal128(_, _) => capacity / 16,
        other => panic!("unexpected column type {other}"),
    }
}

/// Every batch of `reader`, checked against the rows it could still have
/// returned: `remaining` rows before the first batch.
fn assert_capacity_follows_rows(reader: ParquetRecordBatchReader, mut remaining: usize) -> usize {
    let mut batches = 0;
    for batch in reader {
        let batch = batch.unwrap();
        let bound = remaining.min(BATCH_SIZE);
        assert!(batch.num_rows() <= bound);
        for (field, column) in batch.schema().fields().iter().zip(batch.columns()) {
            assert!(
                slots(column) <= bound,
                "column {} of a {}-row batch has room for {} rows; {bound} rows were left",
                field.name(),
                batch.num_rows(),
                slots(column),
            );
        }
        remaining -= batch.num_rows();
        batches += 1;
    }
    assert_eq!(remaining, 0);
    batches
}

#[test]
fn a_small_row_group_reserves_its_own_rows() {
    let file = flat_file(&[7, 7, 7]);
    for group in 0..3 {
        let reader = ParquetRecordBatchReaderBuilder::try_new(file.clone())
            .unwrap()
            .with_batch_size(BATCH_SIZE)
            .with_row_groups(vec![group])
            .build()
            .unwrap();
        assert_eq!(assert_capacity_follows_rows(reader, 7), 1);
    }
}

#[test]
fn a_batch_across_small_row_groups_reserves_their_rows() {
    let file = flat_file(&[7; 40]);
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .with_batch_size(BATCH_SIZE)
        .build()
        .unwrap();
    assert_eq!(assert_capacity_follows_rows(reader, 280), 1);
}

#[test]
fn the_last_batch_of_a_row_group_reserves_the_rows_that_remain() {
    let file = flat_file(&[BATCH_SIZE + 100]);
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .with_batch_size(BATCH_SIZE)
        .build()
        .unwrap();
    assert_eq!(assert_capacity_follows_rows(reader, BATCH_SIZE + 100), 2);
}

#[test]
fn a_selection_reserves_its_selected_rows() {
    let file = flat_file(&[3 * BATCH_SIZE]);
    let selection = RowSelection::from(vec![
        RowSelector::skip(10),
        RowSelector::select(3),
        RowSelector::skip(BATCH_SIZE),
        RowSelector::select(2),
        RowSelector::skip(2 * BATCH_SIZE - 15),
    ]);
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .with_batch_size(BATCH_SIZE)
        .with_row_selection(selection)
        .with_row_selection_policy(RowSelectionPolicy::Selectors)
        .build()
        .unwrap();
    assert_eq!(assert_capacity_follows_rows(reader, 5), 1);
}

#[test]
fn full_batches_keep_their_batch_size_reservation() {
    // One allocation per column and batch: the bound must not make a full
    // batch grow its buffers while it decodes.
    let file = flat_file(&[3 * BATCH_SIZE]);
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .with_batch_size(BATCH_SIZE)
        .build()
        .unwrap();
    for batch in reader {
        let batch = batch.unwrap();
        assert_eq!(batch.num_rows(), BATCH_SIZE);
        for column in batch.columns() {
            assert_eq!(slots(column), BATCH_SIZE);
        }
    }
}
