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

//! Reading a file with invalid page data returns an error. It does not panic
//! and it does not loop.
//!
//! Each file in `invalid_files/` made the reader panic or loop before the
//! Longbow PEH patch family. `invalid_files/README.md` records where each
//! file comes from and what it showed.

use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use parquet::arrow::arrow_reader::{
    ArrowReaderOptions, ParquetRecordBatchReaderBuilder, RowSelection, RowSelector,
};
use parquet::errors::ParquetError;

/// A read that does not end in this time is a loop.
const READ_LIMIT: Duration = Duration::from_secs(20);

struct InvalidFile {
    name: &'static str,
    data: &'static [u8],
    /// The row selection of the read that showed the failure: the kind of the
    /// first run (`select` or `skip`), then the run lengths.
    selection: &'static str,
    /// The selected read skips the invalid page of one file without decoding
    /// it, and succeeds. All other reads return an error.
    selected_read_fails: bool,
    /// Text that the errors of the full read and of the selected read
    /// contain, where the test pins the messages.
    expected_errors: Option<(&'static str, &'static str)>,
}

macro_rules! invalid_file {
    ($name:literal) => {
        invalid_file!($name, true)
    };
    ($name:literal, $selected_read_fails:literal) => {
        InvalidFile {
            expected_errors: None,
            ..invalid_file!($name, $selected_read_fails, "", "")
        }
    };
    ($name:literal, $selected_read_fails:literal, $full_error:literal, $selected_error:literal) => {
        InvalidFile {
            name: $name,
            data: include_bytes!(concat!("invalid_files/", $name, ".parquet")),
            selection: include_str!(concat!("invalid_files/", $name, ".parquet.selection")),
            selected_read_fails: $selected_read_fails,
            expected_errors: Some(($full_error, $selected_error)),
        }
    };
}

static INVALID_FILES: &[InvalidFile] = &[
    invalid_file!("byte_stream_split_short_data"),
    invalid_file!("def_levels_packed_run_no_data"),
    invalid_file!("def_levels_packed_run_past_end", false),
    invalid_file!("def_levels_packed_run_past_end_skip"),
    invalid_file!("def_levels_packed_run_short_data"),
    invalid_file!("delta_binary_packed_zero_values"),
    invalid_file!("delta_byte_array_negative_suffix_length"),
    invalid_file!("dictionary_data_page_without_dictionary"),
    invalid_file!("dictionary_index_width_142"),
    invalid_file!("dictionary_index_width_148"),
    invalid_file!("dictionary_index_width_90"),
    invalid_file!("dictionary_indices_empty"),
    invalid_file!("page_header_map_past_end"),
    invalid_file!("plain_byte_array_length_past_end"),
    invalid_file!("plain_byte_array_no_values_declared"),
    invalid_file!("repetition_levels_end_early"),
    // Arrow 60.1 rejects a run length above 2^31 - 1 (#11517). The Longbow
    // fork of 59.3.2 kept the low 32 bits instead and failed later.
    invalid_file!(
        "rle_levels_run_length_overflow",
        true,
        "Invalid or corrupted RLE run length 18446744073709551615. Max allowed is 2147483647",
        "Invalid or corrupted RLE run length 18446744073709551615. Max allowed is 2147483647"
    ),
    // A run header longer than ten bytes ends the run data
    // (`BitReader::get_vlq_int` gives `None`), so the page has too few values.
    invalid_file!(
        "rle_run_header_too_long",
        true,
        "insufficient values read from column - expected: 111, got: 0",
        "skipped 0 values, expected 20"
    ),
    invalid_file!("v1_page_type_without_header"),
    invalid_file!("v2_levels_longer_than_page"),
    invalid_file!("v2_page_type_without_header"),
];

/// How the reader presents top-level string columns. Each choice uses a
/// different byte array decoder.
#[derive(Clone, Copy, Debug)]
enum Strings {
    Offsets,
    Views,
    Dictionary,
}

fn parse_selection(text: &str) -> RowSelection {
    let mut parts = text.split_whitespace();
    let mut select = match parts.next() {
        Some("select") => true,
        Some("skip") => false,
        other => panic!("invalid selection start {other:?}"),
    };
    let mut selectors = Vec::new();
    for length in parts {
        let length = length.parse().unwrap();
        selectors.push(if select {
            RowSelector::select(length)
        } else {
            RowSelector::skip(length)
        });
        select = !select;
    }
    RowSelection::from(selectors)
}

/// Read the whole file and return the number of rows.
fn read(
    data: Bytes,
    batch_size: usize,
    selection: Option<RowSelection>,
    strings: Strings,
) -> Result<usize, ParquetError> {
    let mut builder = ParquetRecordBatchReaderBuilder::try_new(data.clone())?;
    let string_type = match strings {
        Strings::Offsets => None,
        Strings::Views => Some(DataType::Utf8View),
        Strings::Dictionary => Some(DataType::Dictionary(
            Box::new(DataType::Int32),
            Box::new(DataType::Utf8),
        )),
    };
    if let Some(string_type) = string_type {
        let fields: Vec<Field> = builder
            .schema()
            .fields()
            .iter()
            .map(|field| match field.data_type() {
                DataType::Utf8 => field.as_ref().clone().with_data_type(string_type.clone()),
                _ => field.as_ref().clone(),
            })
            .collect();
        let schema = Schema::new_with_metadata(fields, builder.schema().metadata().clone());
        let options = ArrowReaderOptions::new().with_schema(Arc::new(schema));
        builder = ParquetRecordBatchReaderBuilder::try_new_with_options(data, options)?;
    }
    let mut builder = builder.with_batch_size(batch_size);
    if let Some(selection) = selection {
        builder = builder.with_row_selection(selection);
    }
    let mut rows = 0;
    for batch in builder.build()? {
        rows += batch
            .map_err(|e| ParquetError::General(e.to_string()))?
            .num_rows();
    }
    Ok(rows)
}

/// Run one read on another thread, so a loop or a panic fails the test
/// instead of stopping it. `Err` describes a read that panicked or did not
/// end.
fn read_with_limit(
    file: &InvalidFile,
    batch_size: usize,
    selection: Option<RowSelection>,
    strings: Strings,
) -> Result<Result<usize, ParquetError>, String> {
    let (sender, receiver) = mpsc::channel();
    let data = Bytes::from_static(file.data);
    let handle = std::thread::spawn(move || {
        let _ = sender.send(read(data, batch_size, selection, strings));
    });
    match receiver.recv_timeout(READ_LIMIT) {
        Ok(result) => {
            handle.join().unwrap();
            Ok(result)
        }
        Err(mpsc::RecvTimeoutError::Timeout) => Err(format!("did not end in {READ_LIMIT:?}")),
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            let _ = handle.join();
            Err("panicked".to_string())
        }
    }
}

/// Checks one read: it must end without a panic, fail if `must_fail`, and
/// give an error that contains the expected text of the file, if any.
fn check(
    file: &InvalidFile,
    read: &str,
    strings: Strings,
    must_fail: bool,
    outcome: Result<Result<usize, ParquetError>, String>,
) -> Option<String> {
    let problem = match outcome {
        Err(problem) => problem,
        Ok(Ok(rows)) if must_fail => format!("returned {rows} rows"),
        Ok(Ok(_)) => return None,
        Ok(Err(err)) if !must_fail => format!("returned {err}"),
        Ok(Err(err)) => match file.expected_errors.map(
            |(full, selected)| {
                if read == "full" { full } else { selected }
            },
        ) {
            Some(expected) if !err.to_string().contains(expected) => {
                format!("returned {err}, expected an error with {expected:?}")
            }
            _ => return None,
        },
    };
    Some(format!(
        "{} ({strings:?}): the {read} read {problem}",
        file.name
    ))
}

#[test]
fn invalid_files_return_errors() {
    let mut failures = Vec::new();
    for file in INVALID_FILES {
        for strings in [Strings::Offsets, Strings::Views, Strings::Dictionary] {
            let full = read_with_limit(file, 1024, None, strings);
            failures.extend(check(file, "full", strings, true, full));

            let selection = parse_selection(file.selection);
            let selected = read_with_limit(file, 97, Some(selection), strings);
            failures.extend(check(
                file,
                "selected",
                strings,
                file.selected_read_fails,
                selected,
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The table in this file and the files in the directory must agree.
#[test]
fn invalid_files_are_all_listed() {
    let directory = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/arrow_reader/invalid_files"
    );
    let entries = std::fs::read_dir(directory).unwrap();
    let mut on_disk: Vec<String> = entries
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter_map(|name| name.strip_suffix(".parquet").map(str::to_string))
        .collect();
    on_disk.sort();
    let listed: Vec<&str> = INVALID_FILES.iter().map(|file| file.name).collect();
    assert_eq!(on_disk, listed);
}
