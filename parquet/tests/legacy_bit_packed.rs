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

use bytes::Bytes;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::Field;

#[test]
fn reads_parquet_mr_msb_first_definition_levels() {
    let fixture = Bytes::from_static(include_bytes!("data/bitpacking_levels.parquet"));
    let reader = SerializedFileReader::new(fixture).unwrap();
    let mut rows = reader.get_row_iter(None).unwrap();
    let columns = rows.next().unwrap().unwrap().into_columns();
    assert!(rows.next().is_none());
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
    let expected = expected.map(|(name, field)| (String::from(name), field));
    assert_eq!(columns, expected);
}
