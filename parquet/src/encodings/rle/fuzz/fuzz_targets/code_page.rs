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

//! Drives flat dictionary pages through validity and the row verdict bitmap
//! and compares them with an independent reader for the on-wire encoding.

#![no_main]

mod oracle;

use arrow_array::types::Int32Type as ArrowInt32;
use bytes::Bytes;
use libfuzzer_sys::fuzz_target;
use parquet::basic::Encoding;
use parquet::column::page::{Page, PageMetadata, PageReader};
use parquet::column::reader::{CodeColumnPage, CodeColumnReader};
use parquet::data_type::Int32Type;
use parquet::errors::Result;
use parquet::schema::{
    parser::parse_message_type,
    types::{ColumnDescPtr, SchemaDescriptor},
};
use std::sync::{Arc, OnceLock};

struct Pages(std::vec::IntoIter<Page>);

impl Iterator for Pages {
    type Item = Result<Page>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(Ok)
    }
}

impl PageReader for Pages {
    fn get_next_page(&mut self) -> Result<Option<Page>> {
        Ok(self.0.next())
    }

    fn peek_next_page(&mut self) -> Result<Option<PageMetadata>> {
        unimplemented!("the code reader does not peek")
    }

    fn skip_next_page(&mut self) -> Result<()> {
        unimplemented!("the code reader does not skip")
    }
}

fn descriptor(nullable: bool) -> ColumnDescPtr {
    static OPTIONAL: OnceLock<ColumnDescPtr> = OnceLock::new();
    static REQUIRED: OnceLock<ColumnDescPtr> = OnceLock::new();
    let (cache, schema) = if nullable {
        (&OPTIONAL, "message x { OPTIONAL INT32 value; }")
    } else {
        (&REQUIRED, "message x { REQUIRED INT32 value; }")
    };
    cache
        .get_or_init(|| {
            SchemaDescriptor::new(Arc::new(parse_message_type(schema).unwrap())).column(0)
        })
        .clone()
}

fn index_payload(
    bytes: &[u8],
    count: usize,
    entries: usize,
    check_packed_codes: bool,
) -> Option<Vec<u32>> {
    match bytes.split_first() {
        Some((&width, data)) => {
            oracle::hybrid(width, data, count, entries as u64, check_packed_codes)
        }
        None => (count == 0).then(Vec::new),
    }
}

fn repeated_header(count: usize) -> Vec<u8> {
    let mut value = 2 * count;
    let mut bytes = Vec::new();
    while value >= 128 {
        bytes.push((value & 127) as u8 | 128);
        value >>= 7;
    }
    bytes.push(value as u8);
    bytes
}

fuzz_target!(|input: &[u8]| {
    let [
        flags,
        values,
        rows,
        entries,
        width,
        level_len,
        nulls,
        bytes @ ..,
    ] = input
    else {
        return;
    };
    let v2 = flags & 1 != 0;
    let nullable = flags & 2 != 0;
    let legacy = !v2 && nullable && flags & 4 != 0;
    let structured = flags & 8 != 0;
    let count = usize::from(*values) * if flags & 16 != 0 { 4 } else { 1 };
    let header_rows = if flags & 32 != 0 {
        usize::from(*rows)
    } else {
        count
    };
    let entries = usize::from(*entries);

    let (raw_levels, raw_indices) = bytes.split_at(usize::from(*level_len).min(bytes.len()));
    let mut levels = if nullable || (v2 && flags & 64 != 0) {
        raw_levels.to_vec()
    } else {
        Vec::new()
    };
    let mut indices = Vec::new();
    if flags & 128 == 0 {
        indices.push(width % 34);
        indices.extend_from_slice(raw_indices);
    }
    let mut num_nulls = if nullable || flags & 64 != 0 {
        usize::from(*nulls)
    } else {
        0
    };

    if structured {
        let present = !nullable || nulls % 2 == 0;
        num_nulls = if present { 0 } else { count };
        levels.clear();
        if nullable && count != 0 {
            if legacy {
                levels.resize(count.div_ceil(8), if present { u8::MAX } else { 0 });
                if !count.is_multiple_of(8) && present {
                    *levels.last_mut().unwrap() <<= 8 - count % 8;
                }
            } else {
                levels.extend(repeated_header(count));
                levels.push(u8::from(present));
            }
        }
        indices.clear();
        if present && count != 0 {
            indices.push(0);
            indices.extend(repeated_header(count));
        }
    }

    let mut buf = Vec::new();
    if nullable && !v2 && !legacy {
        buf.extend_from_slice(&(levels.len() as u32).to_le_bytes());
    }
    buf.extend_from_slice(&levels);
    let (legacy_levels, legacy_indices) = if legacy {
        // V1 BIT_PACKED has no length prefix. A short levels fragment can
        // consume bytes intended as indices, so split the actual page body.
        buf.extend_from_slice(&indices);
        match buf.split_at_checked(count.div_ceil(8)) {
            Some((def, codes)) => (Some(def.to_vec()), codes.to_vec()),
            None => (None, Vec::new()),
        }
    } else {
        buf.extend_from_slice(&indices);
        (None, Vec::new())
    };
    let buf = Bytes::from(buf);
    let page = if v2 {
        Page::DataPageV2 {
            buf,
            num_values: count as u32,
            encoding: Encoding::RLE_DICTIONARY,
            num_nulls: num_nulls as u32,
            num_rows: header_rows as u32,
            def_levels_byte_len: levels.len() as u32,
            rep_levels_byte_len: 0,
            is_compressed: false,
            statistics: None,
        }
    } else {
        Page::DataPage {
            buf,
            num_values: count as u32,
            encoding: Encoding::RLE_DICTIONARY,
            #[expect(deprecated, reason = "Fuzz legacy V1 levels")]
            def_level_encoding: if legacy {
                Encoding::BIT_PACKED
            } else {
                Encoding::RLE
            },
            rep_level_encoding: Encoding::RLE,
            statistics: None,
        }
    };

    let dictionary = Page::DictionaryPage {
        buf: Bytes::from(
            (0..entries as i32)
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>(),
        ),
        num_values: entries as u32,
        encoding: Encoding::PLAIN,
        is_sorted: false,
    };
    let mut reader = CodeColumnReader::<Int32Type, ArrowInt32>::new(
        descriptor(nullable),
        Box::new(Pages(vec![dictionary, page].into_iter())),
    )
    .unwrap();
    let page = reader.next_page();
    if v2
        && (header_rows != count
            || num_nulls > count
            || (!nullable && (!levels.is_empty() || num_nulls != 0)))
    {
        assert!(page.is_err());
        return;
    }

    if legacy {
        let expected = legacy_levels.and_then(|levels| {
            let valid: Vec<_> = (0..count)
                .map(|i| levels[i / 8] & (128 >> (i % 8)) != 0)
                .collect();
            let present = valid.iter().filter(|&&bit| bit).count();
            let mut keys = index_payload(&legacy_indices, present, entries, true)?.into_iter();
            Some(
                valid
                    .into_iter()
                    .map(|bit| bit.then(|| keys.next().unwrap() as i32))
                    .collect::<Vec<_>>(),
            )
        });
        match page {
            Ok(Some(CodeColumnPage::Values { values, .. })) => {
                assert_eq!(Some(values.iter().collect::<Vec<_>>()), expected);
            }
            Err(_) => assert!(expected.is_none()),
            other => panic!("unexpected legacy page {other:?}"),
        }
        return;
    }

    let expected_levels = if nullable {
        oracle::hybrid(1, &levels, count, 2, true)
    } else {
        Some(vec![1; count])
    }
    .filter(|valid| !v2 || valid.iter().filter(|&&bit| bit == 0).count() == num_nulls);
    let keys = expected_levels.as_ref().and_then(|valid| {
        index_payload(
            &indices,
            valid.iter().filter(|&&bit| bit != 0).count(),
            entries,
            true,
        )
    });
    let page = match page {
        Ok(Some(CodeColumnPage::Dictionary(page))) => page,
        Err(_) => {
            assert!(keys.is_none());
            return;
        }
        other => panic!("unexpected hybrid page {other:?}"),
    };
    let mut valid = Vec::new();
    let validity = page.validity(&mut valid);
    let valid_expected = expected_levels
        .as_ref()
        .filter(|_| nullable || index_payload(&indices, count, entries, false).is_some());
    assert_eq!(
        validity.is_ok(),
        valid_expected.is_some(),
        "validity result"
    );
    if let Some(expected) = valid_expected {
        assert_eq!(
            validity.unwrap(),
            expected.iter().filter(|&&bit| bit != 0).count()
        );
        assert_eq!(valid.len(), count.div_ceil(64));
        for (row, &value) in expected.iter().enumerate() {
            assert_eq!((valid[row / 64] >> (row % 64)) & 1, u64::from(value));
        }
    }

    let mut table = page
        .dictionary()
        .verdict_table((0..entries).map(|key| key % 3 == 0));
    let mut verdicts = Vec::new();
    let result = page.verdicts(&mut table, &mut valid, &mut verdicts);
    assert_eq!(result.is_ok(), keys.is_some(), "verdict result");
    if let Some(keys) = keys {
        assert_eq!(verdicts.len(), count.div_ceil(64));
        let mut keys = keys.into_iter();
        for (row, value) in expected_levels.unwrap().into_iter().enumerate() {
            let expected = value != 0 && keys.next().unwrap() % 3 == 0;
            assert_eq!((verdicts[row / 64] >> (row % 64)) & 1, u64::from(expected));
        }
        if count % 64 != 0 {
            assert_eq!(verdicts.last().unwrap() >> (count % 64), 0);
        }
    }
});
