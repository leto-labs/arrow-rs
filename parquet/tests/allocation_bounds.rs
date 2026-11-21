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

//! Reading a file that declares a count or a size that its pages cannot hold
//! returns an error. The reader does not allocate memory for the declared
//! count.
//!
//! An allocation that the host cannot provide ends the process: no error path
//! of a caller can report it. This test program replaces the allocator with
//! one that refuses large allocations during a read, as a host without that
//! memory does. Before the Longbow PAB patch family, the reads of the invalid
//! files end this test process.
//!
//! `crafted_files/README.md` describes the files and the script that writes
//! them. Each invalid file has a valid file that differs only in the declared
//! count.
//!
//! The files use Snappy, LZ4 and GZIP pages; the test needs those codecs.

#![cfg(all(
    feature = "arrow",
    feature = "snap",
    feature = "lz4",
    feature = "flate2"
))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BooleanArray, Int32Array, RecordBatch};
use arrow_buffer::BooleanBuffer;
use arrow_cast::display::array_value_to_string;
use arrow_schema::{ArrowError, DataType, Field, Schema};
use bytes::Bytes;
use parquet::arrow::arrow_reader::{
    ArrowPredicate, ArrowReaderOptions, DictionaryFilterMode, DictionaryFilterOptions,
    DictionaryVerdict, ParquetRecordBatchReaderBuilder, RowFilter, RowSelection, RowSelector,
};
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::errors::ParquetError;
use parquet::file::properties::WriterProperties;

/// The allocator refuses a larger allocation during a read.
const REFUSED_ABOVE: usize = 64 << 20;

/// A read of one of these files needs no allocation above this size. The
/// largest file is 33 KB and each file has four rows.
const LARGEST_EXPECTED: usize = 1 << 20;

thread_local! {
    /// The largest allocation that the allocator grants on this thread.
    static LIMIT: Cell<usize> = const { Cell::new(usize::MAX) };
    /// The largest allocation that the allocator granted on this thread.
    static LARGEST: Cell<usize> = const { Cell::new(0) };
}

struct LimitedAllocator;

/// Returns false for a size above the limit of this thread.
fn grant(size: usize) -> bool {
    // `try_with` fails only while the thread ends; grant those allocations.
    LIMIT.try_with(|limit| size <= limit.get()).unwrap_or(true)
}

/// Records an allocation that the system provided.
fn record(pointer: *mut u8, size: usize) -> *mut u8 {
    if !pointer.is_null() {
        let _ = LARGEST.try_with(|largest| largest.set(largest.get().max(size)));
    }
    pointer
}

// SAFETY: every method forwards to `System` with the arguments that it
// received, or returns null, which the `GlobalAlloc` contract allows for a
// failed allocation.
unsafe impl GlobalAlloc for LimitedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !grant(layout.size()) {
            return std::ptr::null_mut();
        }
        // SAFETY: the caller upholds the contract of `GlobalAlloc::alloc`.
        record(unsafe { System.alloc(layout) }, layout.size())
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if !grant(layout.size()) {
            return std::ptr::null_mut();
        }
        // SAFETY: the caller upholds the contract of `GlobalAlloc::alloc_zeroed`.
        record(unsafe { System.alloc_zeroed(layout) }, layout.size())
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if !grant(new_size) {
            return std::ptr::null_mut();
        }
        // SAFETY: the caller upholds the contract of `GlobalAlloc::realloc`.
        record(unsafe { System.realloc(ptr, layout, new_size) }, new_size)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller upholds the contract of `GlobalAlloc::dealloc`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: LimitedAllocator = LimitedAllocator;

struct PassDictionary(ProjectionMask);

impl ArrowPredicate for PassDictionary {
    fn projection(&self) -> &ProjectionMask {
        &self.0
    }

    fn evaluate(&mut self, batch: RecordBatch) -> Result<BooleanArray, ArrowError> {
        Ok(BooleanArray::new(
            BooleanBuffer::new_set(batch.num_rows()),
            None,
        ))
    }

    fn dictionary_filter_options(&self) -> Option<DictionaryFilterOptions> {
        Some(DictionaryFilterOptions {
            mode: DictionaryFilterMode::On,
            max_percent: 25,
        })
    }

    fn evaluate_dictionary(
        &mut self,
        _leaf: usize,
        entries: &ArrayRef,
    ) -> Result<Option<DictionaryVerdict>, ArrowError> {
        Ok(Some(DictionaryVerdict {
            entries: BooleanBuffer::new_set(entries.len()),
            null: false,
        }))
    }
}

#[test]
fn dictionary_filter_small_unit_does_not_allocate_a_group_bitmap() {
    const UNIT_ROWS: usize = 256 * 1024;
    const GROUP_ROWS: usize = 8 * UNIT_ROWS;
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from_iter_values(
            (0..UNIT_ROWS).map(|row| (row % 2) as i32),
        ))],
    )
    .unwrap();
    let properties = WriterProperties::builder()
        .set_max_row_group_row_count(Some(GROUP_ROWS))
        .set_data_page_row_count_limit(4096)
        .build();
    let mut writer = ArrowWriter::try_new(Vec::new(), schema, Some(properties)).unwrap();
    for _ in 0..GROUP_ROWS / UNIT_ROWS {
        writer.write(&batch).unwrap();
    }
    let file = Bytes::from(writer.into_inner().unwrap());
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let projection = ProjectionMask::leaves(builder.parquet_schema(), [0]);
    let builder = builder
        .with_batch_size(1024)
        .with_row_selection(RowSelection::from(vec![
            RowSelector::skip(GROUP_ROWS - UNIT_ROWS),
            RowSelector::select(UNIT_ROWS),
        ]))
        .with_row_filter(
            RowFilter::new(vec![Box::new(PassDictionary(projection))]).with_dictionary_filter(),
        );
    let (rows, largest) = with_limit(|| {
        builder
            .build()
            .unwrap()
            .map(|batch| batch.unwrap().num_rows())
            .sum::<usize>()
    });
    assert_eq!(rows, UNIT_ROWS);
    assert!(
        largest < GROUP_ROWS / 8,
        "largest allocation was {largest} bytes, enough for a full-group bitmap"
    );
}

/// Runs `read` with the allocation limit and returns its result and the
/// largest allocation that it made.
fn with_limit<T>(read: impl FnOnce() -> T) -> (T, usize) {
    LARGEST.with(|largest| largest.set(0));
    LIMIT.with(|limit| limit.set(REFUSED_ABOVE));
    let result = read();
    LIMIT.with(|limit| limit.set(usize::MAX));
    (result, LARGEST.with(|largest| largest.get()))
}

/// How the reader presents the string column. Each choice uses a different
/// byte array decoder.
#[derive(Clone, Copy, Debug)]
enum Strings {
    Offsets,
    Views,
    Dictionary,
}

const STRINGS: [Strings; 3] = [Strings::Offsets, Strings::Views, Strings::Dictionary];

/// Reads the file. Returns the values of column `v` as text.
fn read(
    data: &'static [u8],
    strings: Strings,
    selection: Option<RowSelection>,
) -> Result<Vec<String>, ParquetError> {
    let data = Bytes::from_static(data);
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
    if let Some(selection) = selection {
        builder = builder.with_row_selection(selection);
    }
    let mut values = Vec::new();
    for batch in builder.build()? {
        let batch: RecordBatch = batch.map_err(|e| ParquetError::General(e.to_string()))?;
        let column = batch.column_by_name("v").expect("column v");
        for row in 0..column.len() {
            values.push(array_value_to_string(column, row)?);
        }
    }
    Ok(values)
}

/// Skips the first row and reads the next two.
fn selection() -> RowSelection {
    RowSelection::from(vec![
        RowSelector::skip(1),
        RowSelector::select(2),
        RowSelector::skip(1),
    ])
}

macro_rules! file {
    ($directory:literal, $name:literal) => {
        (
            $name,
            include_bytes!(concat!(
                "crafted_files/",
                $directory,
                "/",
                $name,
                ".parquet"
            ))
            .as_slice(),
        )
    };
}

/// A crafted file: its name and its bytes.
type CraftedFile = (&'static str, &'static [u8]);

/// The files that read their values, and the values of their column `v`.
static VALID_FILES: &[(CraftedFile, [&str; 4])] = &[
    // A writer that under-reports the value count of a column chunk.
    (
        file!("allocation_bounds", "valid_chunk_count_below_pages"),
        ["0", "1", "2", "3"],
    ),
    (
        file!("allocation_bounds", "valid_delta_length"),
        ["a", "bb", "ccc", "dddd"],
    ),
    (
        file!("allocation_bounds", "valid_delta_byte_array"),
        ["a", "ab", "abc", "abcd"],
    ),
    (
        file!("allocation_bounds", "valid_dictionary_strings"),
        ["x", "yy", "x", "yy"],
    ),
    (
        file!("allocation_bounds", "valid_dictionary_int96"),
        [
            "1970-01-01T00:00:00",
            "1970-01-01T00:00:01",
            "1970-01-01T00:00:00",
            "1970-01-01T00:00:01",
        ],
    ),
    (
        file!("allocation_bounds", "valid_snappy"),
        ["0", "1", "2", "3"],
    ),
    (
        file!("allocation_bounds", "valid_lz4_raw"),
        ["0", "1", "2", "3"],
    ),
    (
        file!("allocation_bounds", "valid_gzip"),
        ["0", "1", "2", "3"],
    ),
];

/// The invalid files and a part of the error that each read returns.
static INVALID_FILES: &[(CraftedFile, &str)] = &[
    (
        file!("allocation_bounds", "delta_length_count_2e40"),
        "DELTA_LENGTH_BYTE_ARRAY length block declares 1099511627776 values, but the page has 4",
    ),
    // A length block of one value under a header of block size 2^61 and
    // 2^56 mini blocks: no block is stored, so nothing checks the header
    // against the page bytes, and nothing may be sized by it.
    (
        file!("allocation_bounds", "delta_length_mini_blocks_2e56"),
        "insufficient",
    ),
    (
        file!("allocation_bounds", "delta_byte_array_mini_blocks_2e56"),
        "insufficient",
    ),
    (
        file!("allocation_bounds", "delta_byte_array_prefix_count_2e40"),
        "DELTA_BYTE_ARRAY prefix length block declares 1099511627776 values, but the page has 4",
    ),
    (
        file!("allocation_bounds", "delta_byte_array_suffix_count_2e40"),
        "DELTA_BYTE_ARRAY suffix length block declares 1099511627776 values, but the page has 4",
    ),
    (
        file!("allocation_bounds", "dictionary_strings_count_i32_max"),
        "dictionary page declares 2147483647 values, but ",
    ),
    (
        file!("allocation_bounds", "dictionary_int96_count_i32_max"),
        "the page header declares 2147483647 values in 24 bytes",
    ),
    (
        file!("allocation_bounds", "snappy_size_i32_max"),
        "cannot hold the 2147483647 bytes that the page header declares",
    ),
    (
        file!("allocation_bounds", "lz4_raw_size_i32_max"),
        "cannot hold the 2147483647 bytes that the page header declares",
    ),
    (
        file!("allocation_bounds", "gzip_longer_than_declared"),
        "decompressed data is longer than the 32 bytes that the page header declares",
    ),
];

/// The tests are in a module with the name of the program, so that a test
/// filter on `allocation_bounds` selects them.
mod allocation_bounds {
    use super::*;

    #[test]
    fn valid_files_read_their_values() {
        for ((name, data), expected) in VALID_FILES {
            for strings in STRINGS {
                let (values, largest) = with_limit(|| read(data, strings, None));
                let values = values.unwrap_or_else(|e| panic!("{name} ({strings:?}): {e}"));
                assert_eq!(values, expected, "{name} ({strings:?})");
                assert!(
                    largest <= LARGEST_EXPECTED,
                    "{name} ({strings:?}): allocated {largest} bytes at once"
                );

                let (values, _) = with_limit(|| read(data, strings, Some(selection())));
                assert_eq!(values.unwrap(), expected[1..3], "{name} ({strings:?})");
            }
        }
    }

    /// Files whose page declares 2^31 - 1 values and whose page bytes hold
    /// a few. No allocation follows the declared count: the full read fails
    /// when the values need more bytes than the page holds, and the selected
    /// read decodes only up to its last row and returns its rows.
    ///
    /// * `data_page_count_above_chunk`: a PLAIN page in a chunk of four. The
    ///   page count of a PLAIN page is not checked against its chunk (some
    ///   writers under-report the chunk count).
    /// * `delta_length_zero_width_count_i32_max`: a zero-width length block
    ///   of 2^31 - 1 lengths in a chunk of four.
    /// * `delta_*_zero_width_chunk_count_i32_max`: zero-width length blocks
    ///   in a chunk and row group that also declare 2^31 - 1 values, so the
    ///   footer agrees with the page and a page of 2^31 - 1 empty strings
    ///   would be valid.
    ///
    /// The delta lengths are decoded a window at a time. Before, a read
    /// decoded every length first: 8 GiB, or 16 GiB for the prefixes and
    /// suffixes of DELTA_BYTE_ARRAY.
    #[test]
    fn counts_that_the_page_bytes_contradict_allocate_nothing() {
        let cases = [
            (
                file!("allocation_bounds", "data_page_count_above_chunk"),
                "EOF: Not enough bytes to decode",
                ["1", "2"],
            ),
            (
                file!(
                    "allocation_bounds",
                    "delta_length_zero_width_chunk_count_i32_max"
                ),
                "Insufficient delta length byte array bytes",
                ["b", "c"],
            ),
            (
                file!(
                    "allocation_bounds",
                    "delta_byte_array_zero_width_chunk_count_i32_max"
                ),
                "eof decoding byte array",
                ["b", "c"],
            ),
        ];
        for ((name, data), full_error, selected_values) in cases {
            for strings in STRINGS {
                let (full, largest) = with_limit(|| read(data, strings, None));
                let error = full.unwrap_err().to_string();
                assert!(error.contains(full_error), "{name} ({strings:?}): {error}");
                assert!(
                    largest <= LARGEST_EXPECTED,
                    "{name} ({strings:?}): allocated {largest} bytes at once"
                );
                let (selected, largest) = with_limit(|| read(data, strings, Some(selection())));
                let selected = selected.unwrap_or_else(|e| panic!("{name} ({strings:?}): {e}"));
                assert_eq!(selected, selected_values, "{name} ({strings:?})");
                assert!(
                    largest <= LARGEST_EXPECTED,
                    "{name} ({strings:?}): allocated {largest} bytes at once"
                );
            }
        }
    }

    #[test]
    fn declared_counts_above_the_page_return_errors() {
        for ((name, data), expected) in INVALID_FILES {
            for strings in STRINGS {
                for selection in [None, Some(selection())] {
                    let selected = selection.is_some();
                    let (result, largest) = with_limit(|| read(data, strings, selection));
                    let error = match result {
                        Ok(values) => panic!(
                            "{name} ({strings:?}, selected: {selected}): the read returned {values:?}"
                        ),
                        Err(error) => error.to_string(),
                    };
                    assert!(
                        error.contains(expected),
                        "{name} ({strings:?}, selected: {selected}): {error}"
                    );
                    assert!(
                        largest <= LARGEST_EXPECTED,
                        "{name} ({strings:?}, selected: {selected}): allocated {largest} bytes at once"
                    );
                }
            }
        }
    }

    /// The file is valid: four NULL values of a fixed length type of 2^31 - 1
    /// bytes. Arrow needs a slot of that length for each NULL, which is 8 GiB.
    /// The reader reserved 2^31 - 1 bytes for each row of a batch (16 TiB for
    /// the default batch size) before it read a page.
    #[test]
    fn fixed_length_type_that_cannot_be_allocated_returns_an_error() {
        let (name, data) = file!("type_length", "fixed_len_type_length_i32_max");
        for selection in [None, Some(selection())] {
            let (result, largest) = with_limit(|| read(data, Strings::Offsets, selection));
            let error = result.expect_err(name).to_string();
            assert!(
                error.contains("fixed length values of 2147483647 bytes"),
                "{name}: {error}"
            );
            assert!(
                largest <= LARGEST_EXPECTED,
                "{name}: allocated {largest} bytes at once"
            );
        }
    }

    /// The tables in this file and the files in the directories must agree.
    #[test]
    fn crafted_files_are_all_listed() {
        let mut listed: Vec<String> = VALID_FILES
            .iter()
            .map(|((name, _), _)| format!("allocation_bounds/{name}"))
            .chain(
                INVALID_FILES
                    .iter()
                    .map(|((name, _), _)| format!("allocation_bounds/{name}")),
            )
            .chain([
                "allocation_bounds/data_page_count_above_chunk".to_string(),
                "allocation_bounds/delta_length_zero_width_count_i32_max".to_string(),
                "allocation_bounds/delta_byte_array_zero_width_chunk_count_i32_max".to_string(),
                "allocation_bounds/delta_length_zero_width_chunk_count_i32_max".to_string(),
                "type_length/fixed_len_type_length_i32_max".to_string(),
            ])
            .collect();
        listed.sort();

        let mut on_disk = Vec::new();
        for directory in ["allocation_bounds", "type_length"] {
            let path = format!(
                "{}/tests/crafted_files/{directory}",
                env!("CARGO_MANIFEST_DIR")
            );
            for entry in std::fs::read_dir(path).unwrap() {
                let name = entry.unwrap().file_name().into_string().unwrap();
                if let Some(name) = name.strip_suffix(".parquet") {
                    on_disk.push(format!("{directory}/{name}"));
                }
            }
        }
        on_disk.sort();
        assert_eq!(on_disk, listed);
    }
}
