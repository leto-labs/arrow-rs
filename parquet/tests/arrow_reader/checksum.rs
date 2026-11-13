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

//! This file contains an end to end test for verifying checksums when reading parquet files.

use std::path::PathBuf;

use arrow::util::test_util::parquet_test_data;
use arrow_array::RecordBatch;
use bytes::Bytes;
use parquet::arrow::arrow_reader::{ArrowReaderBuilder, ParquetRecordBatchReaderBuilder};

#[test]
fn test_datapage_v1_corrupt_checksum() {
    let errors = read_file_batch_errors("datapage_v1-corrupt-checksum.parquet");
    assert_eq!(errors, [
        Err("Parquet argument error: Parquet error: Page CRC checksum mismatch: the DATA_PAGE header declares 0xbbce3b9d, the 10240 bytes of page data have 0x0f4f6d0a".to_string()),
        Ok(()),
        Ok(()),
        Err("Parquet argument error: Parquet error: Page CRC checksum mismatch: the DATA_PAGE header declares 0x48850d12, the 10240 bytes of page data have 0x0358a2bc".to_string()),
        Err("Parquet argument error: Parquet error: Not all children array length are the same!".to_string())
    ]);
}

#[test]
fn test_datapage_v1_uncompressed_checksum() {
    let errors = read_file_batch_errors("datapage_v1-uncompressed-checksum.parquet");
    assert_eq!(errors, [Ok(()), Ok(()), Ok(()), Ok(()), Ok(())]);
}

#[test]
#[cfg(feature = "snap")]
fn test_datapage_v1_snappy_compressed_checksum() {
    let errors = read_file_batch_errors("datapage_v1-snappy-compressed-checksum.parquet");
    assert_eq!(errors, [Ok(()), Ok(()), Ok(()), Ok(()), Ok(())]);
}

#[test]
fn test_plain_dict_uncompressed_checksum() {
    let errors = read_file_batch_errors("plain-dict-uncompressed-checksum.parquet");
    assert_eq!(errors, [Ok(())]);
}
#[test]
#[cfg(feature = "snap")]
fn test_rle_dict_snappy_checksum() {
    let errors = read_file_batch_errors("rle-dict-snappy-checksum.parquet");
    assert_eq!(errors, [Ok(())]);
}

/// Reads a file and returns a vector with one element per record batch.
/// The record batch data is replaced with () and errors are stringified.
fn read_file_batch_errors(name: &str) -> Vec<Result<(), String>> {
    let path = PathBuf::from(parquet_test_data()).join(name);
    println!("Reading file: {path:?}");
    let file = std::fs::File::open(&path).unwrap();
    let reader = ArrowReaderBuilder::try_new(file).unwrap().build().unwrap();
    reader
        .map(|x| match x {
            Ok(_) => Ok(()),
            Err(e) => Err(e.to_string()),
        })
        .collect()
}

/// A file of `crafted_files/checksums`. `crafted_files/README.md` describes
/// the files and the script that writes them.
macro_rules! crafted_file {
    ($name:literal) => {
        (
            $name,
            include_bytes!(concat!("../crafted_files/checksums/", $name, ".parquet")).as_slice(),
        )
    };
}

fn read_crafted(data: &'static [u8]) -> Result<Vec<RecordBatch>, String> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(Bytes::from_static(data))
        .map_err(|e| e.to_string())?
        .build()
        .map_err(|e| e.to_string())?;
    reader
        .map(|batch| batch.map_err(|e| e.to_string()))
        .collect()
}

/// The two files hold the same pages. One has a checksum in each page header
/// (two data pages, a dictionary page and a compressed page); the other has
/// none.
#[test]
fn files_with_and_without_checksums_read_the_same_rows() {
    let (_, with_checksums) = crafted_file!("valid_with_checksums");
    let (_, without_checksums) = crafted_file!("valid_without_checksums");
    let with_checksums = read_crafted(with_checksums).unwrap();
    let without_checksums = read_crafted(without_checksums).unwrap();
    assert_eq!(with_checksums, without_checksums);

    let formatted = arrow::util::pretty::pretty_format_batches(&with_checksums)
        .unwrap()
        .to_string();
    let expected = [
        "+---+----+----+",
        "| e | v  | s  |",
        "+---+----+----+",
        "| 0 | x  | 10 |",
        "| 1 | yy | 11 |",
        "| 2 | x  | 12 |",
        "| 3 | yy | 13 |",
        "+---+----+----+",
    ];
    assert_eq!(formatted.lines().collect::<Vec<_>>(), expected);
}

/// In each file one page and its checksum disagree. Without verification
/// three of the files return other values and no error.
#[test]
fn page_that_differs_from_its_checksum_returns_an_error() {
    let files = [
        (
            crafted_file!("data_page_byte_changed"),
            "Page CRC checksum mismatch: the DATA_PAGE header declares 0x9100120a, the 32 bytes of page data have 0x28fbc9e2",
        ),
        (
            crafted_file!("dictionary_page_byte_changed"),
            "Page CRC checksum mismatch: the DICTIONARY_PAGE header declares 0xef625a4c, the 11 bytes of page data have 0x78fd4b65",
        ),
        (
            crafted_file!("compressed_page_byte_changed"),
            "Page CRC checksum mismatch: the DATA_PAGE header declares 0xb035f76a, the 34 bytes of page data have 0x99a83587",
        ),
        (
            crafted_file!("checksum_changed"),
            "Page CRC checksum mismatch: the DATA_PAGE header declares 0x9100120b, the 32 bytes of page data have 0x9100120a",
        ),
    ];
    for ((name, data), expected) in files {
        let error = match read_crafted(data) {
            Ok(batches) => panic!("{name}: the read returned {batches:?}"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            format!("Parquet argument error: Parquet error: {expected}"),
            "{name}"
        );
    }
}

/// The tests above and the files in the directory must agree.
#[test]
fn crafted_checksum_files_are_all_read() {
    let directory = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/crafted_files/checksums");
    let mut on_disk: Vec<String> = std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter_map(|name| name.strip_suffix(".parquet").map(str::to_string))
        .collect();
    on_disk.sort();
    assert_eq!(
        on_disk,
        [
            "checksum_changed",
            "compressed_page_byte_changed",
            "data_page_byte_changed",
            "dictionary_page_byte_changed",
            "valid_with_checksums",
            "valid_without_checksums",
        ]
    );
}
