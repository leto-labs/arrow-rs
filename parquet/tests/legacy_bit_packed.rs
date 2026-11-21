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

use bytes::Bytes;
use parquet::data_type::Int32Type;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::Field;

#[path = "data/make_legacy_bit_packed.rs"]
mod generator;

const FIXTURE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/legacy_bit_packed.parquet"
);
const FIXTURE: &[u8] = include_bytes!("data/legacy_bit_packed.parquet");

#[test]
#[ignore = "Run explicitly to regenerate the checked-in fixture"]
fn regenerate_fixture() {
    std::fs::write(FIXTURE_PATH, generator::generate()).unwrap();
}

#[test]
fn fixture_matches_generator() {
    assert_eq!(generator::generate(), FIXTURE);
}

#[test]
fn reads_parquet_mr_value_semantics() {
    let reader = SerializedFileReader::new(Bytes::from_static(FIXTURE)).unwrap();
    let mut rows = reader.get_row_iter(None).unwrap();
    let mut columns = rows.next().unwrap().unwrap().into_columns();
    assert_eq!(columns.pop().unwrap().0, "outer_items");
    let expected = [
        ("required_string", Field::Str("required_string1".into())),
        ("optional_string", Field::Str("optional_string".into())),
        ("required_double", Field::Double(1.0)),
        ("optional_double", Field::Double(1.0)),
        ("required_timestamp", Field::TimestampMillis(12_345_678)),
        ("optional_timestamp", Field::TimestampMillis(12_345_678)),
        ("required_bool", Field::Bool(true)),
        ("optional_bool", Field::Bool(false)),
    ];
    assert_eq!(
        columns,
        expected.map(|(name, value)| (name.to_owned(), value))
    );
    let second = rows.next().unwrap().unwrap().into_columns();
    assert_eq!(second[3], ("optional_double".into(), Field::Double(2.0)));
    assert!(rows.next().is_none());
}

#[test]
fn reads_nested_record_across_two_legacy_pages() {
    use parquet::column::reader::get_typed_column_reader;

    let reader = SerializedFileReader::new(Bytes::from_static(FIXTURE)).unwrap();
    let row_group = reader.get_row_group(0).unwrap();
    let mut column = get_typed_column_reader::<Int32Type>(row_group.get_column_reader(8).unwrap());
    for (values, repetitions) in [
        (&[10, 11, 12, 13][..], &[0, 2, 2, 1][..]),
        (&[20, 21][..], &[0, 2][..]),
    ] {
        let (mut data, mut definition, mut repetition) = (vec![], vec![], vec![]);
        assert_eq!(
            column
                .read_records(1, Some(&mut definition), Some(&mut repetition), &mut data)
                .unwrap(),
            (1, values.len(), values.len())
        );
        assert_eq!(data, values);
        assert_eq!(repetition, repetitions);
        assert_eq!(definition, vec![2; values.len()]);
    }
}
