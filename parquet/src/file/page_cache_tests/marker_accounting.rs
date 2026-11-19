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

use crate::{
    DecodeResult,
    arrow::{
        ArrowWriter,
        arrow_reader::{RowSelection, RowSelector},
        push_decoder::ParquetPushDecoderBuilder,
    },
    basic::{Compression, Encoding},
    file::{
        metadata::{PageIndexPolicy, ParquetMetaData, ParquetMetaDataReader},
        page_cache::{CachedPage, PageCache, PageFill, PageLease, PageLookup},
        properties::{WriterProperties, WriterVersion},
        reader::{FileReader, SerializedFileReader},
    },
};
use arrow_array::{Int32Array, RecordBatch};
use bytes::Bytes;
use std::{
    collections::BTreeMap,
    ops::Range,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

type Slots = Arc<Mutex<BTreeMap<u64, CachedPage>>>;
#[derive(Debug, Default)]
struct Cache {
    slots: Slots,
    marker_hits: Arc<AtomicUsize>,
}
struct Fill(Slots, u64);
impl PageFill for Fill {
    fn complete(self: Box<Self>, page: CachedPage) {
        self.0.lock().unwrap().insert(self.1, page);
    }
}
impl PageCache for Cache {
    fn get_with_accounting(
        &self,
        offset: u64,
        fill: bool,
        count_read: fn(&CachedPage) -> bool,
    ) -> PageLookup {
        match self.slots.lock().unwrap().get(&offset).cloned() {
            Some(page) => {
                if count_read(&page) && page.is_dictionary_marker() {
                    self.marker_hits.fetch_add(1, Ordering::Relaxed);
                }
                PageLookup::Hit(page)
            }
            None => PageLookup::Miss(
                fill.then(|| Box::new(Fill(self.slots.clone(), offset)) as Box<dyn PageFill>),
            ),
        }
    }
    fn probe(&self, offset: u64) -> Option<CachedPage> {
        self.slots.lock().unwrap().get(&offset).cloned()
    }
}

#[derive(Debug)]
struct ChargedFixtureCache(Cache);

impl PageCache for ChargedFixtureCache {
    fn get_with_accounting(
        &self,
        offset: u64,
        fill: bool,
        count_read: fn(&CachedPage) -> bool,
    ) -> PageLookup {
        self.0.get_with_accounting(offset, fill, count_read)
    }

    fn probe(&self, offset: u64) -> Option<CachedPage> {
        self.0.probe(offset)
    }

    fn lease(&self, sets: &[Vec<Range<u64>>]) -> Option<PageLease> {
        let slots = self.0.slots.lock().unwrap();
        let covered: Vec<_> = sets
            .iter()
            .map(|set| {
                let mut pages = Vec::new();
                for range in set {
                    let mut offset = range.start;
                    while offset < range.end {
                        let page = slots.get(&offset)?;
                        pages.push((offset, page.clone()));
                        offset += page.stored_len();
                    }
                    if offset != range.end {
                        return None;
                    }
                }
                Some(pages)
            })
            .collect();
        covered
            .iter()
            .any(Option::is_some)
            .then(|| PageLease::new(covered, None))
    }
}

fn fixture(version: WriterVersion) -> Bytes {
    let batch = RecordBatch::try_from_iter([(
        "x",
        Arc::new(Int32Array::from_iter(
            (0..20_000).map(|i| (i % 17 != 0).then_some(i)),
        )) as _,
    )])
    .unwrap();
    let props = WriterProperties::builder()
        .set_writer_version(version)
        .set_encoding(Encoding::PLAIN)
        .set_dictionary_page_size_limit(48_000)
        .set_data_page_row_count_limit(1000)
        .set_write_batch_size(250)
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(Vec::new(), batch.schema(), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    Bytes::from(writer.into_inner().unwrap())
}

fn decode(
    metadata: &ParquetMetaData,
    file: &Bytes,
    selection: Option<RowSelection>,
) -> (Vec<RecordBatch>, Vec<Range<u64>>) {
    let mut builder = ParquetPushDecoderBuilder::try_new_decoder(Arc::new(metadata.clone()))
        .unwrap()
        .with_batch_size(1024);
    if let Some(selection) = selection {
        builder = builder.with_row_selection(selection);
    }
    let mut decoder = builder.build().unwrap();
    let (mut batches, mut requested) = (Vec::new(), Vec::new());
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

fn review(default_lease: bool) {
    for version in [WriterVersion::PARQUET_1_0, WriterVersion::PARQUET_2_0] {
        let file = fixture(version);
        let source = SerializedFileReader::new(file.clone()).unwrap();
        let group = source.get_row_group(0).unwrap();
        let encodings: Vec<_> = group
            .get_column_page_reader(0)
            .unwrap()
            .map(|page| {
                let page = page.unwrap();
                (page.is_data_page(), page.encoding())
            })
            .collect();
        assert!(encodings.contains(&(true, Encoding::RLE_DICTIONARY)));
        assert!(encodings.contains(&(true, Encoding::PLAIN)));
        for policy in [PageIndexPolicy::Skip, PageIndexPolicy::Required] {
            let meta = ParquetMetaDataReader::new()
                .with_page_index_policy(policy)
                .parse_and_finish(&file)
                .unwrap();
            let column = meta.row_group(0).column(0);
            assert!(column.encodings().any(|e| e == Encoding::RLE_DICTIONARY));
            let dict = column.dictionary_page_offset().unwrap() as u64;
            let first = column.data_page_offset() as u64;
            let cache = Arc::new(ChargedFixtureCache(Cache::default()));
            let handle: Arc<dyn PageCache> = if default_lease {
                Arc::new(Cache {
                    slots: Arc::clone(&cache.0.slots),
                    marker_hits: Arc::clone(&cache.0.marker_hits),
                })
            } else {
                cache.clone()
            };
            let cached = meta.clone().with_page_cache(Some(handle));
            let cache = &cache.0;
            let expected = decode(&meta, &file, None).0;
            assert_eq!(decode(&cached, &file, None).0, expected);
            assert_eq!(cache.probe(dict).unwrap().buffer_len(), 0);
            assert_eq!(
                cache.probe(first).unwrap().page().encoding(),
                Encoding::PLAIN
            );
            let (warm, requested) = decode(&cached, &file, None);
            assert_eq!(warm, expected);
            assert!(requested.is_empty());
            cache.slots.lock().unwrap().remove(&first).unwrap();
            let (partial, requested) = decode(&cached, &file, None);
            assert_eq!(partial, expected);
            assert!(requested.iter().any(|r| r.contains(&dict)));
            assert_eq!(
                cache.marker_hits.load(Ordering::Relaxed),
                0,
                "empty marker recorded by an accounting lookup"
            );
            cache.slots.lock().unwrap().remove(&dict).unwrap();
            let selection = RowSelection::from(vec![
                RowSelector::skip(990),
                RowSelector::select(75),
                RowSelector::skip(10_000),
                RowSelector::select(3_000),
                RowSelector::skip(6_000),
            ]);
            let (partial, requested) = decode(&cached, &file, Some(selection.clone()));
            assert_eq!(partial, decode(&meta, &file, Some(selection)).0);
            assert!(requested.iter().any(|r| r.contains(&dict)));
        }
    }
}
#[test]
fn incomplete_lease_does_not_count_an_empty_marker() {
    review(false);
}
#[test]
fn default_lease_does_not_count_an_empty_marker() {
    review(true);
}
