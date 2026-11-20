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
//! and compares them with the pull cursors over the same level and index bytes.

#![no_main]

use arrow_array::types::Int32Type as ArrowInt32;
use bytes::Bytes;
use libfuzzer_sys::fuzz_target;
use parquet::basic::Encoding;
use parquet::column::page::{Page, PageMetadata, PageReader};
use parquet::column::reader::{CodeColumnPage, CodeColumnReader};
use parquet::data_type::Int32Type;
use parquet::errors::Result;
use parquet::schema::{parser::parse_message_type, types::SchemaDescriptor};
use std::sync::Arc;

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

fuzz_target!(|input: &[u8]| {
    let [
        rows_low,
        flags,
        entries,
        width,
        levels_len,
        nulls,
        bytes @ ..,
    ] = input
    else {
        return;
    };
    let rows = (usize::from(*flags & 0x03) << 8) | usize::from(*rows_low);
    let v2 = flags & 0x04 != 0;
    let entries = usize::from(*entries);
    let levels_len = usize::from(*levels_len).min(bytes.len());
    let (levels, indices) = bytes.split_at(levels_len);
    let indices: Vec<u8> = [width % 34]
        .into_iter()
        .chain(indices.iter().copied())
        .collect();
    let num_nulls = u32::from(*nulls).min(rows as u32);
    let page = if v2 {
        let mut buf = levels.to_vec();
        buf.extend_from_slice(&indices);
        Page::DataPageV2 {
            buf: buf.into(),
            num_values: rows as u32,
            encoding: Encoding::RLE_DICTIONARY,
            num_nulls,
            num_rows: rows as u32,
            def_levels_byte_len: levels_len as u32,
            rep_levels_byte_len: 0,
            is_compressed: false,
            statistics: None,
        }
    } else {
        let mut buf = (levels_len as u32).to_le_bytes().to_vec();
        buf.extend_from_slice(levels);
        buf.extend_from_slice(&indices);
        Page::DataPage {
            buf: buf.into(),
            num_values: rows as u32,
            encoding: Encoding::RLE_DICTIONARY,
            def_level_encoding: Encoding::RLE,
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
    let schema = parse_message_type("message x { OPTIONAL INT32 value; }").unwrap();
    let descriptor = SchemaDescriptor::new(Arc::new(schema)).column(0);
    let Ok(mut reader) = CodeColumnReader::<Int32Type, ArrowInt32>::new(
        descriptor,
        Box::new(Pages(vec![dictionary, page].into_iter())),
    ) else {
        return;
    };
    let Ok(Some(CodeColumnPage::Dictionary(page))) = reader.next_page() else {
        return;
    };

    // Oracle validity from the pull cursor over the level stream.
    let mut expected_valid = Vec::with_capacity(rows);
    let mut levels_ok = true;
    let mut cursor = page.levels().unwrap();
    loop {
        match cursor.next_run() {
            Ok(None) => break,
            Ok(Some(run)) => {
                for i in 0..run.len() {
                    expected_valid.push(run.code_at(i).unwrap() == 1);
                }
            }
            Err(_) => {
                levels_ok = false;
                break;
            }
        }
    }
    let ones = expected_valid.iter().filter(|&&valid| valid).count();
    if levels_ok {
        assert_eq!(expected_valid.len(), rows);
        levels_ok = !v2 || ones == rows - num_nulls as usize;
    }

    let mut validity = Vec::new();
    let result = page.validity(&mut validity);
    assert_eq!(result.is_ok(), levels_ok, "validity result");
    if let Ok(count) = result {
        assert_eq!(count, ones);
        assert_eq!(validity.len(), rows.div_ceil(64));
        for (row, &valid) in expected_valid.iter().enumerate() {
            assert_eq!((validity[row / 64] >> (row % 64)) & 1, u64::from(valid));
        }
        if rows % 64 != 0 {
            assert_eq!(validity.last().unwrap() >> (rows % 64), 0);
        }
    }

    // Oracle keys from the pull cursor over the index stream.
    let keys = page
        .indices()
        .and_then(|mut cursor| cursor.selected_keys(0..ones));
    let mut table = page
        .dictionary()
        .verdict_table((0..entries).map(|key| key % 3 == 0));
    let mut verdicts = Vec::new();
    let result = page.verdicts(&mut table, &mut validity, &mut verdicts);
    let keys = match (levels_ok, keys) {
        (true, Ok(keys)) if keys.len() == ones => Some(keys),
        _ => None,
    };
    assert_eq!(result.is_ok(), keys.is_some(), "verdict result");
    if let Some(keys) = keys {
        assert_eq!(verdicts.len(), rows.div_ceil(64));
        let mut keys = keys.into_iter();
        for (row, &valid) in expected_valid.iter().enumerate() {
            let expected = valid && keys.next().unwrap() % 3 == 0;
            assert_eq!((verdicts[row / 64] >> (row % 64)) & 1, u64::from(expected));
        }
        if rows % 64 != 0 {
            assert_eq!(verdicts.last().unwrap() >> (rows % 64), 0);
        }
    }
});
