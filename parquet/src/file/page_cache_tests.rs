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

// Longbow modification notice: this file is new in the Longbow fork
// (family DPC, decompressed page cache).

//! Tests of the decompressed page cache hook: a reader with a cache returns
//! the pages and batches of a reader without one, a resident page costs no
//! file bytes, and a page that left the cache is read from the file again.

use crate::DecodeResult;
use crate::arrow::ArrowWriter;
use crate::arrow::arrow_reader::{RowSelection, RowSelector};
use crate::arrow::push_decoder::ParquetPushDecoderBuilder;
use crate::basic::Compression;
use crate::column::page::{Page, PageReader};
use crate::errors::{ParquetError, Result};
use crate::file::metadata::{PageIndexPolicy, ParquetMetaData, ParquetMetaDataReader};
use crate::file::page_cache::PageCache;
use crate::file::page_cache::testing::MapPageCache;
use crate::file::properties::{WriterProperties, WriterVersion};
use crate::file::reader::{ChunkReader, Length};
use crate::file::serialized_reader::SerializedPageReader;
use arrow_array::{ArrayRef, Int32Array, Int64Array, RecordBatch, StringArray};
use bytes::Bytes;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::Ordering;

const ROWS: usize = 20_000;

/// A compressed file with several row groups and several pages in each column
/// chunk: a plain column, a nullable dictionary column and a string column.
fn file(version: WriterVersion, compression: Compression) -> Bytes {
    let plain: ArrayRef = Arc::new(Int64Array::from_iter_values(
        (0..ROWS as i64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15_u64 as i64)),
    ));
    let dictionary: ArrayRef = Arc::new(Int32Array::from_iter(
        (0..ROWS as i32).map(|i| (i % 7 != 0).then_some(i % 97)),
    ));
    let strings: ArrayRef = Arc::new(StringArray::from_iter(
        (0..ROWS).map(|i| (i % 11 != 0).then(|| format!("value-{}", i % 1_013))),
    ));
    let batch = RecordBatch::try_from_iter([
        ("plain", plain),
        ("dictionary", dictionary),
        ("strings", strings),
    ])
    .unwrap();
    let properties = WriterProperties::builder()
        .set_writer_version(version)
        .set_compression(compression)
        .set_max_row_group_row_count(Some(8_000))
        .set_data_page_row_count_limit(1_000)
        .set_write_batch_size(250)
        .set_column_dictionary_enabled("plain".into(), false)
        .build();
    let mut buffer = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buffer, batch.schema(), Some(properties)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    Bytes::from(buffer)
}

fn metadata(file: &Bytes, page_index: bool) -> ParquetMetaData {
    let policy = if page_index {
        PageIndexPolicy::Required
    } else {
        PageIndexPolicy::Skip
    };
    ParquetMetaDataReader::new()
        .with_page_index_policy(policy)
        .parse_and_finish(file)
        .unwrap()
}

/// A column chunk source that has no bytes: any read fails the test.
struct NoBytes(u64);

impl Length for NoBytes {
    fn len(&self) -> u64 {
        self.0
    }
}

impl ChunkReader for NoBytes {
    type T = bytes::buf::Reader<Bytes>;

    fn get_read(&self, start: u64) -> Result<Self::T> {
        Err(ParquetError::General(format!(
            "read of file bytes at {start}"
        )))
    }

    fn get_bytes(&self, start: u64, length: usize) -> Result<Bytes> {
        Err(ParquetError::General(format!(
            "read of {length} file bytes at {start}"
        )))
    }
}

/// The fields of a page that a decoder reads.
fn fingerprint(page: &Page) -> String {
    format!(
        "{:?} values={} encoding={:?} bytes={:?}",
        page.page_type(),
        page.num_values(),
        page.encoding(),
        page.buffer()
    )
}

fn reader<R: ChunkReader>(
    source: Arc<R>,
    metadata: &ParquetMetaData,
    row_group: usize,
    column: usize,
    cache: Option<Arc<dyn PageCache>>,
) -> SerializedPageReader<R> {
    let group = metadata.row_group(row_group);
    let locations = metadata
        .offset_index()
        .map(|index| index[row_group][column].page_locations.clone());
    SerializedPageReader::new(
        source,
        group.column(column),
        group.num_rows() as usize,
        locations,
    )
    .unwrap()
    .with_page_cache(cache, None)
}

fn pages<R: ChunkReader>(mut reader: SerializedPageReader<R>) -> Vec<String> {
    let mut pages = vec![];
    while let Some(page) = reader.get_next_page().unwrap() {
        pages.push(fingerprint(&page));
    }
    pages
}

fn chunks(metadata: &ParquetMetaData) -> Vec<(usize, usize)> {
    (0..metadata.num_row_groups())
        .flat_map(|group| (0..metadata.row_group(group).num_columns()).map(move |c| (group, c)))
        .collect()
}

#[test]
fn a_resident_page_needs_no_file_bytes() {
    for version in [WriterVersion::PARQUET_1_0, WriterVersion::PARQUET_2_0] {
        for page_index in [false, true] {
            let file = file(version, Compression::SNAPPY);
            let metadata = metadata(&file, page_index);
            let cache = Arc::new(MapPageCache::default());
            let source = Arc::new(file.clone());
            let mut total = 0;
            for (group, column) in chunks(&metadata) {
                let expected = pages(reader(Arc::clone(&source), &metadata, group, column, None));
                assert!(expected.len() > 2, "several pages per column chunk");
                total += expected.len();
                // The first reader fills the cache and returns the same pages.
                let handle: Arc<dyn PageCache> = cache.clone();
                let filled = pages(reader(
                    Arc::clone(&source),
                    &metadata,
                    group,
                    column,
                    Some(handle.clone()),
                ));
                assert_eq!(filled, expected);
                // The second reader has no bytes at all.
                let empty = Arc::new(NoBytes(file.len() as u64));
                let cached = pages(reader(empty, &metadata, group, column, Some(handle)));
                assert_eq!(cached, expected);
            }
            assert_eq!(cache.len(), total);
            assert_eq!(cache.fills.load(Ordering::Relaxed), total);
            assert_eq!(cache.hits.load(Ordering::Relaxed), total);
        }
    }
}

#[test]
fn an_uncompressed_chunk_is_not_cached() {
    let file = file(WriterVersion::PARQUET_1_0, Compression::UNCOMPRESSED);
    for page_index in [false, true] {
        let metadata = metadata(&file, page_index);
        let cache = Arc::new(MapPageCache::default());
        let source = Arc::new(file.clone());
        for (group, column) in chunks(&metadata) {
            let expected = pages(reader(Arc::clone(&source), &metadata, group, column, None));
            let handle: Arc<dyn PageCache> = cache.clone();
            let read = pages(reader(
                Arc::clone(&source),
                &metadata,
                group,
                column,
                Some(handle),
            ));
            assert_eq!(read, expected);
        }
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.misses.load(Ordering::Relaxed), 0);
    }
}

/// One step of a deterministic sequence.
fn next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state >> 33
}

#[test]
fn get_peek_and_skip_agree_with_an_uncached_reader() {
    for version in [WriterVersion::PARQUET_1_0, WriterVersion::PARQUET_2_0] {
        for page_index in [false, true] {
            let file = file(version, Compression::SNAPPY);
            let metadata = metadata(&file, page_index);
            let source = Arc::new(file.clone());
            for seed in 0..24_u64 {
                let mut state = seed.wrapping_mul(0x9E37_79B9).wrapping_add(17);
                let cache = Arc::new(MapPageCache::default());
                let handle: Arc<dyn PageCache> = cache.clone();
                for (group, column) in chunks(&metadata) {
                    // Fill, then evict a part, so that hits and misses mix.
                    pages(reader(
                        Arc::clone(&source),
                        &metadata,
                        group,
                        column,
                        Some(handle.clone()),
                    ));
                    for offset in cache.offsets() {
                        if next(&mut state) % 3 == 0 {
                            cache.remove(offset);
                        }
                    }
                    let mut plain = reader(Arc::clone(&source), &metadata, group, column, None);
                    let mut cached = reader(
                        Arc::clone(&source),
                        &metadata,
                        group,
                        column,
                        Some(handle.clone()),
                    );
                    loop {
                        match next(&mut state) % 4 {
                            0 => {
                                let expected = plain.peek_next_page().unwrap();
                                let actual = cached.peek_next_page().unwrap();
                                assert_eq!(
                                    actual
                                        .as_ref()
                                        .map(|m| (m.num_rows, m.num_levels, m.is_dict)),
                                    expected.as_ref().map(|m| (
                                        m.num_rows,
                                        m.num_levels,
                                        m.is_dict
                                    )),
                                );
                            }
                            1 => {
                                if plain.peek_next_page().unwrap().is_none() {
                                    assert!(cached.peek_next_page().unwrap().is_none());
                                    break;
                                }
                                plain.skip_next_page().unwrap();
                                cached.skip_next_page().unwrap();
                            }
                            2 => {
                                assert_eq!(
                                    cached.at_record_boundary().unwrap(),
                                    plain.at_record_boundary().unwrap()
                                );
                            }
                            _ => {
                                let expected = plain.get_next_page().unwrap();
                                let actual = cached.get_next_page().unwrap();
                                assert_eq!(
                                    actual.as_ref().map(fingerprint),
                                    expected.as_ref().map(fingerprint)
                                );
                                if expected.is_none() {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Decodes the file and returns its batches with the byte ranges that the
/// decoder asked for.
fn decode(
    metadata: &Arc<ParquetMetaData>,
    file: &Bytes,
    selection: Option<RowSelection>,
) -> (Vec<RecordBatch>, Vec<Range<u64>>) {
    let mut builder = ParquetPushDecoderBuilder::try_new_decoder(Arc::clone(metadata))
        .unwrap()
        .with_batch_size(1_024);
    if let Some(selection) = selection {
        builder = builder.with_row_selection(selection);
    }
    let mut decoder = builder.build().unwrap();
    let (mut batches, mut requested) = (vec![], vec![]);
    loop {
        match decoder.try_decode().unwrap() {
            DecodeResult::NeedsData(ranges) => {
                let data = ranges
                    .iter()
                    .map(|r| file.slice(r.start as usize..r.end as usize))
                    .collect();
                requested.extend(ranges.iter().cloned());
                decoder.push_ranges(ranges, data).unwrap();
            }
            DecodeResult::Data(batch) => batches.push(batch),
            DecodeResult::Finished => return (batches, requested),
        }
    }
}

fn selection() -> RowSelection {
    // Skips whole pages and parts of pages, across row group boundaries.
    let mut selectors = vec![];
    let mut remaining = ROWS;
    let mut state = 7_u64;
    while remaining > 0 {
        let skip = (next(&mut state) as usize % 2_500).min(remaining);
        selectors.push(RowSelector::skip(skip));
        remaining -= skip;
        let select = (1 + next(&mut state) as usize % 700).min(remaining);
        selectors.push(RowSelector::select(select));
        remaining -= select;
    }
    RowSelection::from(selectors)
}

#[test]
fn the_push_decoder_fetches_only_what_the_cache_lacks() {
    for version in [WriterVersion::PARQUET_1_0, WriterVersion::PARQUET_2_0] {
        for page_index in [false, true] {
            for selected in [false, true] {
                let file = file(version, Compression::SNAPPY);
                let plain = Arc::new(metadata(&file, page_index));
                let cache = Arc::new(MapPageCache::default());
                let handle: Arc<dyn PageCache> = cache.clone();
                let cached = Arc::new(plain.as_ref().clone().with_page_cache(Some(handle)));
                let selection = || selected.then(selection);

                let (expected, all) = decode(&plain, &file, selection());
                assert!(!all.is_empty());
                // The first cached execution fetches the same bytes.
                let (first, fetched) = decode(&cached, &file, selection());
                assert_eq!(first, expected);
                assert_eq!(fetched, all);
                assert!(cache.len() > 0);
                // The second one fetches nothing. The exception is a selection
                // without an offset index: the first execution stepped over
                // pages that it never decompressed, and without the index a
                // column chunk is pinned only when all its pages are resident.
                let (second, fetched) = decode(&cached, &file, selection());
                assert_eq!(second, expected);
                if selected && !page_index {
                    assert!(fetched.iter().all(|range| all.contains(range)));
                    continue;
                }
                assert_eq!(fetched, vec![], "a warm scan reads no file bytes");

                // One page leaves the cache: only its column chunk (or, with an
                // offset index and a selection, its pages) is fetched again.
                let offsets = cache.offsets();
                let removed = offsets[offsets.len() / 2];
                cache.remove(removed).unwrap();
                let (third, fetched) = decode(&cached, &file, selection());
                assert_eq!(third, expected);
                assert!(!fetched.is_empty());
                let chunk = chunks(&plain)
                    .into_iter()
                    .map(|(group, column)| plain.row_group(group).column(column).byte_range())
                    .map(|(start, length)| start..start + length)
                    .find(|chunk| chunk.contains(&removed))
                    .unwrap();
                assert!(
                    fetched
                        .iter()
                        .all(|range| chunk.start <= range.start && range.end <= chunk.end),
                    "{fetched:?} outside column chunk {chunk:?}"
                );

                // An empty cache behaves as the first execution.
                cache.clear();
                let (fourth, fetched) = decode(&cached, &file, selection());
                assert_eq!(fourth, expected);
                assert_eq!(fetched, all);
            }
        }
    }
}

#[test]
fn a_selection_served_from_the_cache_matches_a_full_read() {
    // Fill with a full scan, then read a selection: every needed page is
    // resident, also for the pages that the selection skips.
    let file = file(WriterVersion::PARQUET_1_0, Compression::SNAPPY);
    for page_index in [false, true] {
        let plain = Arc::new(metadata(&file, page_index));
        let cache = Arc::new(MapPageCache::default());
        let handle: Arc<dyn PageCache> = cache.clone();
        let cached = Arc::new(plain.as_ref().clone().with_page_cache(Some(handle)));
        decode(&cached, &file, None);
        let (expected, _) = decode(&plain, &file, Some(selection()));
        let (actual, fetched) = decode(&cached, &file, Some(selection()));
        assert_eq!(actual, expected);
        assert_eq!(fetched, vec![]);
    }
}

#[test]
fn metadata_equality_ignores_the_page_cache() {
    let file = file(WriterVersion::PARQUET_1_0, Compression::SNAPPY);
    let plain = metadata(&file, true);
    let handle: Arc<dyn PageCache> = Arc::new(MapPageCache::default());
    let cached = plain.clone().with_page_cache(Some(handle));
    assert!(cached.page_cache().is_some());
    assert!(plain.page_cache().is_none());
    assert_eq!(plain, cached);
    // A copy through the builder keeps the handle.
    assert!(cached.into_builder().build().page_cache().is_some());
}
