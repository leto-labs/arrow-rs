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

use super::*;
use crate::basic::{Encoding, Type};
use crate::column::reader::CodeRun;
use crate::file::page_cache::light::{BLOCK_LEN, LightBlock, LightCodeBlock};
use crate::file::page_cache::{CachedPage, LightDataPage, LightValues, light_chunk_eligible};
use arrow_array::{Array, Date32Array, Decimal32Array, Decimal64Array, StringViewArray};

const PAGE_ROWS: usize = 1301;
const TEST_ROWS: usize = 4 * PAGE_ROWS + 1;

fn validity(row: usize) -> bool {
    match row / PAGE_ROWS {
        0 => !row.is_multiple_of(2),
        1 => true,
        2 => false,
        _ => row % 113 < 67,
    }
}

fn input() -> RecordBatch {
    let ints = (0..TEST_ROWS).map(|row| {
        validity(row).then(|| {
            if row.is_multiple_of(3) {
                i32::MIN + row as i32
            } else {
                i32::MAX - row as i32
            }
        })
    });
    let wide = (0..TEST_ROWS).map(|row| {
        validity(row).then(|| {
            if row.is_multiple_of(3) {
                i64::MIN + row as i64
            } else {
                i64::MAX - row as i64
            }
        })
    });
    let decimal = (0..TEST_ROWS).map(|row| validity(row).then(|| 999_999_999 - row as i32 * 431));
    let wide_decimal = (0..TEST_ROWS)
        .map(|row| validity(row).then(|| -999_999_999_999_999_999 + row as i64 * 4_000_000_001));
    let strings = (0..TEST_ROWS).map(|row| {
        validity(row).then(|| match row % 3 {
            0 => "",
            1 => "á much longer dictionary entry",
            _ => "a second long dictionary entry",
        })
    });
    RecordBatch::try_from_iter([
        ("i32", Arc::new(Int32Array::from_iter(ints)) as ArrayRef),
        ("i64", Arc::new(Int64Array::from_iter(wide)) as _),
        (
            "date",
            Arc::new(Date32Array::from_iter_values(
                (0..TEST_ROWS).map(|i| i as i32 - 4096),
            )) as _,
        ),
        (
            "d32",
            Arc::new(
                Decimal32Array::from_iter(decimal)
                    .with_precision_and_scale(9, 2)
                    .unwrap(),
            ) as _,
        ),
        (
            "d64",
            Arc::new(
                Decimal64Array::from_iter(wide_decimal)
                    .with_precision_and_scale(18, 4)
                    .unwrap(),
            ) as _,
        ),
        ("text", Arc::new(StringViewArray::from_iter(strings)) as _),
    ])
    .unwrap()
}

fn encode(batch: &RecordBatch, version: WriterVersion) -> Bytes {
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_writer_version(version)
        .set_data_page_row_count_limit(PAGE_ROWS)
        .set_write_batch_size(1)
        .set_column_dictionary_enabled("date".into(), false)
        .build();
    let mut buf = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buf, batch.schema(), Some(props)).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
    buf.into()
}

fn bitmap(array: &dyn Array, start: usize, rows: usize, nullable: bool) -> Option<Bytes> {
    nullable.then(|| {
        let mut bits = vec![0; rows.div_ceil(8)];
        for i in 0..rows {
            bits[i / 8] |= u8::from(array.is_valid(start + i)) << (i % 8);
        }
        Bytes::from(bits)
    })
}

fn blocks(
    array: &dyn Array,
    start: usize,
    rows: usize,
    nullable: bool,
    dictionary: Option<&StringViewArray>,
) -> LightValues {
    macro_rules! integers {
        ($ty:ty, $pack:ident, $result:ident) => {
            if let Some(array) = array.as_any().downcast_ref::<$ty>() {
                return LightValues::$result(
                    (0..rows)
                        .step_by(BLOCK_LEN)
                        .map(|offset| {
                            let len = (rows - offset).min(BLOCK_LEN);
                            let valid = bitmap(array, start + offset, len, nullable);
                            LightBlock::$pack(
                                &array.values()[start + offset..][..len],
                                valid.as_deref(),
                            )
                            .unwrap()
                        })
                        .collect(),
                );
            }
        };
    }
    integers!(Int32Array, pack_i32, Int32);
    integers!(Int64Array, pack, Int64);
    integers!(Date32Array, pack_i32, Int32);
    integers!(Decimal32Array, pack_i32, Int32);
    integers!(Decimal64Array, pack, Int64);
    let array = array.as_any().downcast_ref::<StringViewArray>().unwrap();
    let dictionary = dictionary.unwrap();
    LightValues::StringCodes(
        (0..rows)
            .step_by(BLOCK_LEN)
            .map(|offset| {
                let len = (rows - offset).min(BLOCK_LEN);
                let valid = bitmap(array, start + offset, len, nullable);
                let codes = (start + offset..start + offset + len)
                    .map(|row| {
                        if array.is_null(row) {
                            u32::MAX // poisonous NULL payload says nothing about the dictionary
                        } else {
                            dictionary
                                .iter()
                                .position(|entry| entry == Some(array.value(row)))
                                .unwrap() as u32
                        }
                    })
                    .collect::<Vec<_>>();
                LightCodeBlock::pack(&codes, valid.as_deref(), dictionary.len()).unwrap()
            })
            .collect(),
    )
}

fn string_dictionary(page: &Page) -> Arc<StringViewArray> {
    let mut buf = page.buffer().as_ref();
    let values = (0..page.num_values()).map(|_| {
        let len = u32::from_le_bytes(buf[..4].try_into().unwrap()) as usize;
        let value = std::str::from_utf8(&buf[4..4 + len]).unwrap();
        buf = &buf[4 + len..];
        value
    });
    Arc::new(StringViewArray::from_iter_values(values))
}

fn seed(
    file: &Bytes,
    meta: &ParquetMetaData,
    batch: &RecordBatch,
    cache: &Arc<MapPageCache>,
    mixed: bool,
) {
    let indexed = metadata(file, true);
    for column in 0..batch.num_columns() {
        let chunk = meta.row_group(0).column(column);
        let source = reader(Arc::new(file.clone()), meta, 0, column, None);
        let row_group_index = indexed.page_index_for_row_group(0);
        let locations = row_group_index.page_locations(column).unwrap();
        let (start, _) = chunk.byte_range();
        let indexed_pages = locations
            .iter()
            .map(|page| (page.offset as u64, page.compressed_page_size as u64));
        let locations = (start < locations[0].offset as u64)
            .then_some((start, locations[0].offset as u64 - start))
            .into_iter()
            .chain(indexed_pages);
        let mut input = source.zip(locations);
        let mut dictionary = None;
        let mut start = 0;
        let mut index = 0;
        for (original, (at, len)) in input.by_ref() {
            let original = original.unwrap();
            let entry = CachedPage::new(original.clone(), len, chunk.compression());
            cache.insert(at, entry.clone());
            if original.is_dictionary_page() {
                if chunk.column_type() == Type::BYTE_ARRAY {
                    let views = string_dictionary(&original);
                    cache.insert(
                        at,
                        CachedPage::new_light_dictionary(
                            original.clone(),
                            entry.stored_len(),
                            entry.codec(),
                            views.clone(),
                        )
                        .unwrap(),
                    );
                    dictionary = Some(views);
                }
                continue;
            }
            let rows = original.num_values() as usize;
            let array = batch.column(column).as_ref();
            let nullable = chunk.column_descr().max_def_level() == 1;
            if !mixed || index % 3 == 0 {
                let light = LightDataPage::try_new(
                    rows,
                    blocks(array, start, rows, nullable, dictionary.as_deref()),
                    bitmap(array, start, rows, nullable),
                )
                .unwrap();
                cache.insert(
                    at,
                    CachedPage::new_light(
                        original.clone(),
                        entry.stored_len(),
                        entry.codec(),
                        light,
                    )
                    .unwrap(),
                );
            } else if index % 3 == 2 {
                cache.remove(at).unwrap();
            }
            start += rows;
            index += 1;
        }
        assert_eq!(start, TEST_ROWS, "column {column}");
    }
}

#[test]
fn arrow_light_full_partial_mixed_and_evicted_dictionary() {
    let batch = input();
    for version in [WriterVersion::PARQUET_1_0, WriterVersion::PARQUET_2_0] {
        let file = encode(&batch, version);
        for with_index in [false, true] {
            let meta = Arc::new(metadata(&file, with_index));
            for mixed in [false, true] {
                let cache = Arc::new(MapPageCache::with_leases());
                seed(&file, &meta, &batch, &cache, mixed);
                let cached = Arc::new(meta.as_ref().clone().with_page_cache(Some(cache.clone())));
                let selection = RowSelection::from(vec![
                    RowSelector::skip(PAGE_ROWS + 129),
                    RowSelector::select(31),
                    RowSelector::skip(PAGE_ROWS - 161),
                    RowSelector::select(PAGE_ROWS + 19),
                ]);
                for selection in [None, Some(selection)] {
                    let (expected, _) = decode(&meta, &file, selection.clone());
                    let (got, reads) = decode(&cached, &file, selection.clone());
                    assert_eq!(
                        got, expected,
                        "{version:?}, index={with_index}, mixed={mixed}"
                    );
                    if !mixed {
                        assert!(reads.is_empty());
                    }

                    let str_chunk = meta.row_group(0).column(batch.num_columns() - 1);
                    let dict_offset = str_chunk.dictionary_page_offset().unwrap() as u64;
                    cache.remove(dict_offset).unwrap();
                    let (got, reads) = decode(&cached, &file, selection);
                    assert_eq!(got, expected);
                    assert!(reads.iter().any(|range| range.contains(&dict_offset)));
                }
            }
        }
    }
}

#[test]
fn light_interleaving_is_relative_to_lanes_and_not_hybrid_groups() {
    for width in 1..=32 {
        let mask = u32::MAX >> (32 - width);
        let codes: [u32; BLOCK_LEN] = std::array::from_fn(|i| {
            if i == 0 {
                0
            } else if i == 1 {
                mask
            } else {
                (i as u32).wrapping_mul(18_011) & mask
            }
        });
        let block = LightCodeBlock::pack(&codes, None, (mask as usize).saturating_add(1)).unwrap();
        let run = block.code_run();
        assert!(matches!(run, CodeRun::Interleaved { width: actual, .. } if actual == width));
        for (offset, &code) in codes.iter().enumerate() {
            assert_eq!(run.code_at(offset), Some(code));
        }
        assert_eq!(run.code_at(codes.len()), None);
        assert_eq!(
            run.count_ones(13, 877),
            codes[13..877].iter().filter(|&&code| code == 1).count()
        );
    }
    let constant = LightCodeBlock::pack(&[17; 29], None, 18).unwrap();
    assert_eq!(constant.code_run(), CodeRun::Repeated { code: 17, len: 29 });
}

#[test]
fn light_chunk_candidate_respects_code_space_and_dictionary_kind() {
    let file = encode(&input(), WriterVersion::PARQUET_2_0);
    let meta = metadata(&file, true);
    let rg = meta.row_group(0);
    let i32_col = rg.column(0);
    assert!(
        !light_chunk_eligible(i32_col, Some(100)),
        "sparse value domain"
    );
    assert!(
        light_chunk_eligible(i32_col, Some(TEST_ROWS as u32)),
        "outside code space"
    );
    assert!(light_chunk_eligible(i32_col, None), "plain values");
    let text_col = rg.column(5);
    assert!(
        !light_chunk_eligible(text_col, None),
        "no string dictionary"
    );
    assert!(light_chunk_eligible(text_col, Some(3)));
    assert!(matches!(text_col.column_type(), Type::BYTE_ARRAY));

    let width = Page::DataPageV2 {
        buf: Bytes::new(),
        num_values: 1,
        encoding: Encoding::PLAIN,
        num_nulls: 0,
        num_rows: 1,
        def_levels_byte_len: 0,
        rep_levels_byte_len: 0,
        is_compressed: false,
        statistics: None,
    };
    let light = LightDataPage::try_new(
        1,
        LightValues::Int32(vec![LightBlock::pack_i32(&[17], None).unwrap()].into()),
        Some(Bytes::from_static(&[0])),
    )
    .unwrap();
    assert!(CachedPage::new_light(width, 4, Compression::SNAPPY, light).is_err());
}
