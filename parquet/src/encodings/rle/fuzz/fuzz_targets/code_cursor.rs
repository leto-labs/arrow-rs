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

#![no_main]

mod oracle;

use libfuzzer_sys::fuzz_target;
use parquet::encodings::rle::codes::{CodeCursor, CodeRun, RunSink, VerdictTable};

#[derive(Default)]
struct DenseCodes(Vec<u32>);

impl RunSink for DenseCodes {
    fn repeated(&mut self, code: u32, len: usize) {
        self.0.extend(std::iter::repeat_n(code, len));
    }

    fn packed(&mut self, width: u8, bytes: &[u8], len: usize) {
        let run = CodeRun::Packed { width, bytes, len };
        self.0.extend((0..len).map(|i| run.code_at(i).unwrap()));
    }
}

fuzz_target!(|input: &[u8]| {
    let [width, len, entries, bytes @ ..] = input else {
        return;
    };
    let width = width % 34;
    let len = usize::from(*len);
    let entries = u64::from(*entries);
    let expected = oracle::hybrid(width, bytes, len, entries, true);
    let Ok(mut cursor) = CodeCursor::new(width, bytes, len, entries) else {
        assert!(width > 32);
        return;
    };
    let mut push_cursor = CodeCursor::new(width, bytes, len, entries).unwrap();
    let mut pushed = DenseCodes::default();
    let push_result = push_cursor.drain(&mut pushed);
    let mut verdict_cursor = CodeCursor::new(width, bytes, len, entries).unwrap();
    let mut table = VerdictTable::new((0..entries).map(|key| key % 2 == 0));
    let mut bitmap = Vec::new();
    let verdict_result = verdict_cursor.verdicts(&mut table, &mut bitmap);

    let mut pulled = DenseCodes::default();
    let pull_ok = loop {
        match cursor.next_run() {
            Ok(Some(CodeRun::Repeated { code, len })) => pulled.repeated(code, len),
            Ok(Some(CodeRun::Packed { width, bytes, len })) => pulled.packed(width, bytes, len),
            Ok(Some(CodeRun::Interleaved { .. })) => unreachable!(),
            Ok(None) => break true,
            Err(_) => break false,
        }
    };
    let selected = CodeCursor::new(width, bytes, len, entries)
        .unwrap()
        .selected_keys(0..len);
    assert_eq!(pull_ok, expected.is_some());
    assert_eq!(push_result.is_ok(), expected.is_some());
    assert_eq!(verdict_result.is_ok(), expected.is_some());
    assert_eq!(selected.as_ref().ok(), expected.as_ref());
    assert_eq!(pushed.0, pulled.0);
    let Some(keys) = expected else {
        assert!(cursor.next_run().is_err());
        assert!(push_cursor.drain(&mut pushed).is_err());
        assert!(verdict_cursor.next_run().is_err());
        return;
    };

    assert_eq!(pulled.0, keys);
    assert_eq!(bitmap.len(), len.div_ceil(64));
    for (i, &key) in keys.iter().enumerate() {
        assert_eq!((bitmap[i / 64] >> (i % 64)) & 1, u64::from(key % 2 == 0));
    }
});
