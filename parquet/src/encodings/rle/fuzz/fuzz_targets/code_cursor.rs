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

use libfuzzer_sys::fuzz_target;
use parquet::encodings::rle::codes::{CodeCursor, CodeRun};

fuzz_target!(|input: &[u8]| {
    let [width, len, entries, bytes @ ..] = input else {
        return;
    };
    let width = width % 34;
    let len = usize::from(*len);
    let entries = u64::from(*entries);
    let Ok(mut cursor) = CodeCursor::new(width, bytes, len, entries) else {
        return;
    };
    let mut count = 0;
    while let Some(run) = match cursor.next_run() {
        Ok(run) => run,
        Err(_) => {
            assert!(cursor.next_run().is_err());
            return;
        }
    } {
        count += match run {
            CodeRun::Repeated { len, code } => {
                assert!(u64::from(code) < entries);
                len
            }
            CodeRun::Packed { len, .. } => len,
        };
        assert!(count <= len);
    }
    assert_eq!(count, len);
    let keys = CodeCursor::new(width, bytes, len, entries)
        .unwrap()
        .selected_keys(0..len)
        .unwrap();
    assert_eq!(keys.len(), len);
    assert!(keys.iter().all(|&key| u64::from(key) < entries));
});
