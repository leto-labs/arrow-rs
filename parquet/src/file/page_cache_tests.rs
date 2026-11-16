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
        .page_index()
        .filter(|index| index.has_offset_indexes())
        .and_then(|_| {
            metadata
                .page_index_for_row_group(row_group)
                .page_locations(column)
                .cloned()
        });
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
fn offset_index_and_sequential_readers_share_pages() {
    // A page has one key, the offset of its header, and one stored length,
    // whichever reader filled it: the reader that walks the offset index or
    // the reader that steps from header to header.
    for version in [WriterVersion::PARQUET_1_0, WriterVersion::PARQUET_2_0] {
        let file = file(version, Compression::SNAPPY);
        for filled_with_index in [true, false] {
            let filler = metadata(&file, filled_with_index);
            let reader_metadata = metadata(&file, !filled_with_index);
            let cache = Arc::new(MapPageCache::default());
            let handle: Arc<dyn PageCache> = cache.clone();
            let source = Arc::new(file.clone());
            for (group, column) in chunks(&filler) {
                let expected = pages(reader(Arc::clone(&source), &filler, group, column, None));
                let filled = pages(reader(
                    Arc::clone(&source),
                    &filler,
                    group,
                    column,
                    Some(handle.clone()),
                ));
                assert_eq!(filled, expected);
                // The other kind of reader finds every page, with no bytes.
                let empty = Arc::new(NoBytes(file.len() as u64));
                let other = pages(reader(
                    empty,
                    &reader_metadata,
                    group,
                    column,
                    Some(handle.clone()),
                ));
                assert_eq!(other, expected, "filled with index: {filled_with_index}");
            }
            // The push decoder of the other kind fetches nothing either.
            let plain = Arc::new(reader_metadata.clone());
            let cached = Arc::new(reader_metadata.with_page_cache(Some(handle)));
            let (expected, _) = decode(&plain, &file, None);
            let (actual, fetched) = decode(&cached, &file, None);
            assert_eq!(actual, expected);
            assert_eq!(fetched, vec![], "filled with index: {filled_with_index}");
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
                        if next(&mut state).is_multiple_of(3) {
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

/// Decodes the file, pushing every requested range, and returns the first
/// error.
fn try_decode(metadata: &Arc<ParquetMetaData>, file: &Bytes) -> Result<Vec<RecordBatch>> {
    let mut decoder = ParquetPushDecoderBuilder::try_new_decoder(Arc::clone(metadata))?
        .with_batch_size(1_024)
        .build()?;
    let mut batches = vec![];
    loop {
        match decoder.try_decode()? {
            DecodeResult::NeedsData(ranges) => {
                let data = ranges
                    .iter()
                    .map(|r| file.slice(r.start as usize..r.end as usize))
                    .collect();
                decoder.push_ranges(ranges, data)?;
            }
            DecodeResult::Data(batch) => batches.push(batch),
            DecodeResult::Finished => return Ok(batches),
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

#[test]
fn a_page_index_load_keeps_the_page_cache() {
    let file = file(WriterVersion::PARQUET_1_0, Compression::SNAPPY);
    let handle: Arc<dyn PageCache> = Arc::new(MapPageCache::default());
    let without_index = metadata(&file, false).with_page_cache(Some(handle));
    let mut reader = ParquetMetaDataReader::new_with_metadata(without_index)
        .with_page_index_policy(PageIndexPolicy::Required);
    reader.read_page_indexes(&file).unwrap();
    let with_index = reader.finish().unwrap();
    assert!(with_index.page_index().is_some());
    assert!(with_index.page_cache().is_some());
}

/// Fetch planning leases the pages that it pins, and the lease lives as long
/// as the decoder reads the row group. Reclamation (a clear) during the read
/// keeps the leased pages; the lease ends with the read.
#[test]
fn planning_leases_the_pinned_pages_until_the_read_ends() {
    for page_index in [false, true] {
        let file = file(WriterVersion::PARQUET_2_0, Compression::SNAPPY);
        let plain = Arc::new(metadata(&file, page_index));
        let cache = Arc::new(MapPageCache::with_leases());
        let handle: Arc<dyn PageCache> = cache.clone();
        let cached = Arc::new(plain.as_ref().clone().with_page_cache(Some(handle)));
        let (expected, _) = decode(&plain, &file, None);
        decode(&cached, &file, None);
        let resident = cache.len();
        assert_eq!(cache.leased_pages(), 0);

        let mut decoder = ParquetPushDecoderBuilder::try_new_decoder(Arc::clone(&cached))
            .unwrap()
            .with_batch_size(1_024)
            .build()
            .unwrap();
        let mut batches = vec![];
        let mut fetched = vec![];
        loop {
            match decoder.try_decode().unwrap() {
                DecodeResult::NeedsData(ranges) => {
                    // Only a row group that is planned after the clear below
                    // fetches bytes.
                    assert!(!batches.is_empty(), "a warm scan fetched {ranges:?}");
                    let data = ranges
                        .iter()
                        .map(|r| file.slice(r.start as usize..r.end as usize))
                        .collect();
                    fetched.extend(ranges.iter().cloned());
                    decoder.push_ranges(ranges, data).unwrap();
                }
                DecodeResult::Data(batch) => {
                    if batches.is_empty() {
                        // The reader of the first row group holds its leases,
                        // so a clear keeps exactly its pages.
                        let leased = cache.leased_pages();
                        assert!(leased > 0, "page index {page_index}");
                        cache.clear();
                        assert_eq!(cache.len(), leased);
                    }
                    batches.push(batch);
                }
                DecodeResult::Finished => break,
            }
        }
        assert_eq!(batches, expected);
        assert!(!fetched.is_empty(), "later row groups were cleared");
        assert!(cache.pins.load(Ordering::Relaxed) > 0);
        drop(decoder);
        assert_eq!(cache.leased_pages(), 0, "every lease ends with the read");
        // The scan filled the cleared row groups again; without a lease a
        // clear removes every page.
        assert_eq!(cache.len(), resident);
        cache.clear();
        assert_eq!(cache.len(), 0);
    }
}

/// A chunk whose pages left the cache between two scans is fetched again,
/// while a lease taken by another reader keeps its pages resident.
#[test]
fn a_lease_keeps_pages_through_a_clear() {
    let file = file(WriterVersion::PARQUET_1_0, Compression::SNAPPY);
    let plain = Arc::new(metadata(&file, true));
    let cache = Arc::new(MapPageCache::with_leases());
    let handle: Arc<dyn PageCache> = cache.clone();
    let cached = Arc::new(plain.as_ref().clone().with_page_cache(Some(handle.clone())));
    decode(&cached, &file, None);
    let offsets = cache.offsets();
    let lease = handle.pin(&offsets).unwrap();
    // A lease is all or nothing.
    assert!(handle.pin(&[offsets[0], u64::MAX]).is_none());
    cache.clear();
    assert_eq!(cache.len(), offsets.len());
    let (_, fetched) = decode(&cached, &file, None);
    assert_eq!(fetched, vec![]);
    drop(lease);
    cache.clear();
    assert_eq!(cache.len(), 0);
    let (_, fetched) = decode(&cached, &file, None);
    assert!(!fetched.is_empty());
}

/// A skipped dictionary page that is resident is kept without file bytes and
/// is the page that `take_deferred_dictionary` returns.
#[test]
fn a_skipped_resident_dictionary_is_deferred_without_bytes() {
    for page_index in [false, true] {
        let file = file(WriterVersion::PARQUET_1_0, Compression::SNAPPY);
        let metadata = metadata(&file, page_index);
        let source = Arc::new(file.clone());
        // Column 1 is dictionary encoded.
        let mut plain = reader(Arc::clone(&source), &metadata, 0, 1, None);
        let dictionary = plain.get_next_page().unwrap().unwrap();
        assert!(dictionary.is_dictionary_page());

        let cache = Arc::new(MapPageCache::default());
        let handle: Arc<dyn PageCache> = cache.clone();
        pages(reader(
            Arc::clone(&source),
            &metadata,
            0,
            1,
            Some(handle.clone()),
        ));
        let empty = Arc::new(NoBytes(file.len() as u64));
        let mut cached = reader(empty, &metadata, 0, 1, Some(handle));
        assert!(cached.supports_deferred_dictionary());
        assert!(cached.peek_next_page().unwrap().unwrap().is_dict);
        cached.skip_next_page().unwrap();
        let deferred = cached.take_deferred_dictionary().unwrap().unwrap();
        assert_eq!(fingerprint(&deferred), fingerprint(&dictionary));
        assert!(cached.take_deferred_dictionary().unwrap().is_none());
    }
}

/// A deferred dictionary that is read from the file is offered to the cache.
#[test]
fn a_deferred_dictionary_read_from_the_file_fills_the_cache() {
    for page_index in [false, true] {
        let file = file(WriterVersion::PARQUET_1_0, Compression::SNAPPY);
        let metadata = metadata(&file, page_index);
        let source = Arc::new(file.clone());
        let cache = Arc::new(MapPageCache::default());
        let handle: Arc<dyn PageCache> = cache.clone();
        let mut cached = reader(Arc::clone(&source), &metadata, 0, 1, Some(handle.clone()));
        cached.skip_next_page().unwrap();
        let dictionary = cached.take_deferred_dictionary().unwrap().unwrap();
        assert_eq!(cache.fills.load(Ordering::Relaxed), 1);

        let empty = Arc::new(NoBytes(file.len() as u64));
        let mut warm = reader(empty, &metadata, 0, 1, Some(handle));
        assert_eq!(
            fingerprint(&warm.get_next_page().unwrap().unwrap()),
            fingerprint(&dictionary)
        );
    }
}

/// The checks that a hit repeats: a page that does not fit its chunk, a page
/// of the wrong kind, and a data page with more values than its chunk are
/// errors, not wrong values or a panic.
#[test]
fn a_hit_repeats_the_checks_that_guard_decoding() {
    use crate::file::page_cache::{CachedPage, PinnedPages};
    for page_index in [false, true] {
        let file = file(WriterVersion::PARQUET_1_0, Compression::SNAPPY);
        let metadata = metadata(&file, page_index);
        let source = Arc::new(file.clone());
        let (start, length) = metadata.row_group(0).column(1).byte_range();
        let expected = pages(reader(Arc::clone(&source), &metadata, 0, 1, None));
        let fill = || {
            let cache = Arc::new(MapPageCache::default());
            let handle: Arc<dyn PageCache> = cache.clone();
            pages(reader(
                Arc::clone(&source),
                &metadata,
                0,
                1,
                Some(handle.clone()),
            ));
            let offsets = cache.offsets();
            assert_eq!(offsets[0], start, "the dictionary page comes first");
            (cache, handle, offsets)
        };
        // A chunk whose pages fetch planning pinned: its bytes were not
        // fetched, so a page that fails a check is an error.
        let read_pinned = |cache: &MapPageCache, handle: Arc<dyn PageCache>| {
            let pinned = PinnedPages::at_offsets(cache, &cache.offsets()).unwrap();
            let empty = Arc::new(NoBytes(file.len() as u64));
            let group = metadata.row_group(0);
            let locations = metadata
                .page_index()
                .filter(|index| index.has_offset_indexes())
                .and_then(|_| {
                    metadata
                        .page_index_for_row_group(0)
                        .page_locations(1)
                        .cloned()
                });
            let mut reader = SerializedPageReader::new(
                empty,
                group.column(1),
                group.num_rows() as usize,
                locations,
            )
            .unwrap()
            .with_page_cache(Some(handle), Some(Arc::new(pinned)));
            loop {
                match reader.get_next_page() {
                    Ok(Some(_)) => {}
                    Ok(None) => return Ok(()),
                    Err(e) => return Err(e.to_string()),
                }
            }
        };
        // A chunk whose bytes were fetched: a page that fails a check is
        // read from the bytes, as on a miss.
        let read_fetched = |handle: Arc<dyn PageCache>| {
            pages(reader(Arc::clone(&source), &metadata, 0, 1, Some(handle)))
        };

        // A stored length past the end of the chunk.
        let (cache, handle, offsets) = fill();
        let dictionary = cache.probe(offsets[0]).unwrap();
        cache.insert(
            offsets[0],
            CachedPage::new(dictionary.page().clone(), length + 1, dictionary.codec()),
        );
        assert_eq!(read_fetched(handle.clone()), expected);
        let err = read_pinned(&cache, handle).unwrap_err();
        assert!(
            err.contains("does not fit its column chunk") || err.contains("page location"),
            "{err}"
        );

        // A data page where the dictionary page is.
        let (cache, handle, offsets) = fill();
        let data = cache.probe(offsets[1]).unwrap();
        cache.insert(
            offsets[0],
            CachedPage::new(data.page().clone(), dictionary.stored_len(), data.codec()),
        );
        if page_index {
            assert_eq!(read_fetched(handle.clone()), expected);
            let err = read_pinned(&cache, handle).unwrap_err();
            assert!(err.contains("has the wrong page type"), "{err}");
        } else {
            // Without an offset index the page kind is not known before the
            // page is read, so the page reader returns the data page. The
            // column decoder then has no dictionary for it.
            assert!(read_pinned(&cache, handle.clone()).is_ok());
            let cached = Arc::new(metadata.clone().with_page_cache(Some(handle)));
            let err = try_decode(&cached, &file).unwrap_err().to_string();
            assert!(err.contains("missing dictionary page for column"), "{err}");
        }

        // A page that was decompressed with another codec.
        let (cache, handle, offsets) = fill();
        let data = cache.probe(offsets[1]).unwrap();
        cache.insert(
            offsets[1],
            CachedPage::new(
                data.page().clone(),
                data.stored_len(),
                Compression::ZSTD(Default::default()),
            ),
        );
        assert_eq!(read_fetched(handle.clone()), expected);
        let err = read_pinned(&cache, handle).unwrap_err();
        assert!(err.contains("was decompressed with ZSTD"), "{err}");
    }
}

/// A page location that is longer than its page header and body. The cold
/// read through the offset index decodes the page and ignores the rest of
/// the location; a reader without the index steps from header to header. A
/// page filled with the location's length would make that reader jump over
/// the bytes after the body, so the fill keeps such a page out of the cache.
#[test]
fn a_location_longer_than_its_page_is_not_cached() {
    // Brotli ignores bytes after the end of its stream, so the longer
    // location still decodes.
    let file = file(
        WriterVersion::PARQUET_1_0,
        Compression::BROTLI(Default::default()),
    );
    let with_index = metadata(&file, true);
    let without_index = metadata(&file, false);
    let source = Arc::new(file.clone());
    let group = with_index.row_group(0);
    // Column 0 has no dictionary: its first location is its first page.
    let column = group.column(0);
    let mut locations = with_index
        .page_index_for_row_group(0)
        .page_locations(0)
        .unwrap()
        .clone();
    assert!(locations.len() > 2);
    // Location 0 covers pages 0 and 1, as a crafted offset index can.
    let second = locations.remove(1);
    locations[0].compressed_page_size += second.compressed_page_size;
    let pages_with_index = |cache: Option<Arc<dyn PageCache>>| {
        let reader = SerializedPageReader::new(
            Arc::clone(&source),
            column,
            group.num_rows() as usize,
            Some(locations.clone()),
        )
        .unwrap()
        .with_page_cache(cache, None);
        pages(reader)
    };
    let expected_with_index = pages_with_index(None);

    let cache = Arc::new(MapPageCache::default());
    let handle: Arc<dyn PageCache> = cache.clone();
    assert_eq!(pages_with_index(Some(handle.clone())), expected_with_index);
    let first = locations[0].offset as u64;
    assert!(cache.probe(first).is_none(), "the long location was cached");
    assert!(cache.len() > 0, "the other pages are cached");

    // The reader without the index reads every page, as its cold read does.
    let expected = pages(reader(Arc::clone(&source), &without_index, 0, 0, None));
    let warm = pages(reader(
        Arc::clone(&source),
        &without_index,
        0,
        0,
        Some(handle),
    ));
    assert_eq!(warm, expected);
}

/// After a peek the reader knows the page header. A cached page whose length
/// differs from that header is not used: the page is read from its bytes,
/// and the reader does not jump to another offset.
#[test]
fn a_hit_after_a_peek_must_match_the_header() {
    use crate::file::page_cache::CachedPage;
    let file = file(WriterVersion::PARQUET_1_0, Compression::SNAPPY);
    let metadata = metadata(&file, false);
    let source = Arc::new(file.clone());
    // The first two page offsets of column 0, from a fill.
    let offsets = {
        let cache = Arc::new(MapPageCache::default());
        let handle: Arc<dyn PageCache> = cache.clone();
        pages(reader(Arc::clone(&source), &metadata, 0, 0, Some(handle)));
        cache.offsets()
    };
    let first = reader(Arc::clone(&source), &metadata, 0, 0, None)
        .get_next_page()
        .unwrap()
        .unwrap();
    let stored_len = offsets[1] - offsets[0];

    // A pinned page is resident, so a peek does not parse its header: the
    // check after a peek meets only pages whose bytes were fetched, and a
    // wrong length reads the page from them.
    for wrong in [true, false] {
        let cache = Arc::new(MapPageCache::default());
        let handle: Arc<dyn PageCache> = cache.clone();
        let len = if wrong { stored_len + 1 } else { stored_len };
        let mut peeked = reader(Arc::clone(&source), &metadata, 0, 0, Some(handle));
        assert!(peeked.peek_next_page().unwrap().is_some());
        cache.insert(
            offsets[0],
            CachedPage::new(first.clone(), len, Compression::SNAPPY),
        );
        let page = peeked.get_next_page().unwrap().unwrap();
        assert_eq!(fingerprint(&page), fingerprint(&first));
        // The lookup is a hit either way; a wrong length is then read from
        // the file bytes, and the reader stays on the page boundaries.
        assert_eq!(cache.hits.load(Ordering::Relaxed), 1);
        let second = peeked.get_next_page().unwrap().unwrap();
        let mut plain = reader(Arc::clone(&source), &metadata, 0, 0, None);
        plain.get_next_page().unwrap();
        assert_eq!(
            fingerprint(&second),
            fingerprint(&plain.get_next_page().unwrap().unwrap()),
            "wrong length {wrong}: the reader stays on the page boundaries"
        );
    }
}

/// A caller that fetches ahead from a `ScanPlan` skips the ranges that are
/// resident with `resident_offsets` and leases them with `PageCache::pin`.
/// After a warm scan every planned range is resident, and the decoder
/// requests nothing.
#[test]
fn scan_plan_ranges_resolve_against_the_cache() {
    use crate::file::page_cache::resident_offsets;
    for page_index in [false, true] {
        let file = file(WriterVersion::PARQUET_2_0, Compression::SNAPPY);
        let plain = Arc::new(metadata(&file, page_index));
        let cache = Arc::new(MapPageCache::with_leases());
        let handle: Arc<dyn PageCache> = cache.clone();
        let cached = Arc::new(plain.as_ref().clone().with_page_cache(Some(handle.clone())));
        let plan = || {
            ParquetPushDecoderBuilder::try_new_decoder(Arc::clone(&cached))
                .unwrap()
                .build()
                .unwrap()
                .scan_plan()
                .collect::<Vec<_>>()
        };
        // Cold: no planned range is resident.
        assert!(
            plan()
                .iter()
                .all(|r| resident_offsets(handle.as_ref(), r.range.clone()).is_none())
        );
        decode(&cached, &file, None);
        let mut leases = vec![];
        for planned in plan() {
            let offsets = resident_offsets(handle.as_ref(), planned.range.clone())
                .unwrap_or_else(|| panic!("{planned:?} is not resident"));
            assert!(!offsets.is_empty());
            assert_eq!(offsets[0], planned.range.start);
            leases.push(handle.pin(&offsets).unwrap());
        }
        // The leases keep the pages through a clear: the scan reads nothing.
        cache.clear();
        let (_, fetched) = decode(&cached, &file, None);
        assert_eq!(fetched, vec![]);
        drop(leases);
        assert_eq!(cache.leased_pages(), 0);
        // A range that does not start at a page header is not resident.
        let first = plan().remove(0);
        cache.clear();
        decode(&cached, &file, None);
        assert!(
            resident_offsets(handle.as_ref(), first.range.start + 1..first.range.end).is_none()
        );
    }
}

/// The async reader reads through the push decoder, so a warm scan through
/// `ParquetRecordBatchStream` fetches no bytes either.
#[cfg(feature = "async")]
#[test]
fn the_async_reader_uses_the_cache() {
    use crate::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
    use crate::arrow::async_reader::{AsyncFileReader, ParquetRecordBatchStreamBuilder};
    use futures::future::BoxFuture;
    use futures::{FutureExt, TryStreamExt};
    use std::sync::Mutex;

    struct Reader {
        file: Bytes,
        metadata: Arc<ParquetMetaData>,
        requests: Arc<Mutex<Vec<Range<u64>>>>,
    }

    impl AsyncFileReader for Reader {
        fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, Result<Bytes>> {
            self.requests.lock().unwrap().push(range.clone());
            let bytes = self.file.slice(range.start as usize..range.end as usize);
            async move { Ok(bytes) }.boxed()
        }

        fn get_metadata<'a>(
            &'a mut self,
            _options: Option<&'a ArrowReaderOptions>,
        ) -> BoxFuture<'a, Result<Arc<ParquetMetaData>>> {
            let metadata = Arc::clone(&self.metadata);
            async move { Ok(metadata) }.boxed()
        }
    }

    for page_index in [false, true] {
        let file = file(WriterVersion::PARQUET_1_0, Compression::SNAPPY);
        let plain = Arc::new(metadata(&file, page_index));
        let cache = Arc::new(MapPageCache::default());
        let handle: Arc<dyn PageCache> = cache.clone();
        let cached = Arc::new(plain.as_ref().clone().with_page_cache(Some(handle)));
        let scan = |metadata: Arc<ParquetMetaData>| {
            let requests = Arc::new(Mutex::new(vec![]));
            let reader = Reader {
                file: file.clone(),
                metadata: Arc::clone(&metadata),
                requests: Arc::clone(&requests),
            };
            let options = ArrowReaderOptions::new();
            let arrow_metadata = ArrowReaderMetadata::try_new(metadata, options).unwrap();
            let stream = ParquetRecordBatchStreamBuilder::new_with_metadata(reader, arrow_metadata)
                .with_batch_size(1_024)
                .build()
                .unwrap();
            let batches: Vec<RecordBatch> =
                futures::executor::block_on(stream.try_collect()).unwrap();
            let requests = requests.lock().unwrap().clone();
            (batches, requests)
        };
        let (expected, all) = scan(Arc::clone(&plain));
        assert!(!all.is_empty());
        let (first, _) = scan(Arc::clone(&cached));
        assert_eq!(first, expected);
        let (second, fetched) = scan(Arc::clone(&cached));
        assert_eq!(second, expected);
        assert_eq!(fetched, vec![], "page index {page_index}");
    }
}

/// An encrypted column chunk is never cached: its decrypted pages must not
/// reach memory that other readers share. Unencrypted columns of the same
/// file still use the cache.
#[cfg(feature = "encryption")]
#[test]
fn an_encrypted_chunk_is_not_cached() {
    use crate::encryption::decrypt::FileDecryptionProperties;
    use crate::encryption::encrypt::FileEncryptionProperties;

    let footer_key = b"0123456789012345".to_vec();
    let column_key = b"1234567890123450".to_vec();
    let plain_file = file(WriterVersion::PARQUET_1_0, Compression::SNAPPY);
    let batches: Vec<RecordBatch> =
        crate::arrow::arrow_reader::ParquetRecordBatchReader::try_new(plain_file, 4_096)
            .unwrap()
            .collect::<std::result::Result<_, _>>()
            .unwrap();
    let encryption = FileEncryptionProperties::builder(footer_key.clone())
        .with_column_key("plain", column_key.clone())
        .build()
        .unwrap();
    let properties = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_max_row_group_row_count(Some(8_000))
        .set_data_page_row_count_limit(1_000)
        .set_write_batch_size(250)
        .with_file_encryption_properties(encryption)
        .build();
    let mut buffer = Vec::new();
    let mut writer =
        ArrowWriter::try_new(&mut buffer, batches[0].schema(), Some(properties)).unwrap();
    for batch in &batches {
        writer.write(batch).unwrap();
    }
    writer.close().unwrap();
    let file = Bytes::from(buffer);

    let decryption = FileDecryptionProperties::builder(footer_key)
        .with_column_key("plain", column_key)
        .build()
        .unwrap();
    for page_index in [false, true] {
        let policy = if page_index {
            PageIndexPolicy::Required
        } else {
            PageIndexPolicy::Skip
        };
        let plain = Arc::new(
            ParquetMetaDataReader::new()
                .with_page_index_policy(policy)
                .with_decryption_properties(Some(Arc::clone(&decryption)))
                .parse_and_finish(&file)
                .unwrap(),
        );
        let cache = Arc::new(MapPageCache::default());
        let handle: Arc<dyn PageCache> = cache.clone();
        let cached = Arc::new(plain.as_ref().clone().with_page_cache(Some(handle)));
        let (expected, _) = decode(&plain, &file, None);
        let (first, _) = decode(&cached, &file, None);
        assert_eq!(first, expected);
        assert!(cache.len() > 0, "unencrypted columns are cached");

        let encrypted: Vec<Range<u64>> = (0..plain.num_row_groups())
            .map(|group| plain.row_group(group).column(0).byte_range())
            .map(|(start, length)| start..start + length)
            .collect();
        for offset in cache.offsets() {
            assert!(
                !encrypted.iter().any(|chunk| chunk.contains(&offset)),
                "page at {offset} of an encrypted chunk is cached"
            );
        }
        // A warm scan fetches exactly the encrypted chunks.
        let (second, fetched) = decode(&cached, &file, None);
        assert_eq!(second, expected);
        assert!(!fetched.is_empty());
        assert!(
            fetched.iter().all(|range| encrypted
                .iter()
                .any(|chunk| chunk.start <= range.start && range.end <= chunk.end)),
            "{fetched:?}"
        );
    }
}
