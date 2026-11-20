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

//! Independent reader for the hybrid grammar in the Parquet encoding spec.
//! It reads varints byte by byte and unpacks one bit at a time.

pub fn hybrid(
    width: u8,
    mut bytes: &[u8],
    count: usize,
    limit: u64,
    check_packed_codes: bool,
) -> Option<Vec<u32>> {
    if width > 32 || (count != 0 && limit == 0) {
        return None;
    }
    let width = usize::from(width);
    let mut out = Vec::new();
    while out.len() < count {
        let mut header = 0u64;
        for shift in (0..70).step_by(7) {
            let (&byte, rest) = bytes.split_first()?;
            bytes = rest;
            if shift == 63 && byte > 1 {
                return None;
            }
            header |= u64::from(byte & 127) << shift;
            if byte < 128 {
                break;
            }
        }
        let packed = header % 2 != 0;
        let units = header / 2;
        let run_len = units.checked_mul(if packed { 8 } else { 1 })?;
        if run_len == 0 || run_len > i32::MAX as u64 {
            return None;
        }
        let remaining = count - out.len();
        let run_len = usize::try_from(run_len).ok()?;
        if packed {
            if run_len.saturating_sub(remaining) >= 8 {
                return None;
            }
            let (payload, rest) = bytes.split_at_checked((run_len / 8).checked_mul(width)?)?;
            bytes = rest;
            for slot in 0..run_len.min(remaining) {
                let mut value = 0u32;
                for bit in 0..width {
                    let pos = slot * width + bit;
                    value |= u32::from((payload[pos / 8] >> (pos % 8)) & 1) << bit;
                }
                if check_packed_codes && u64::from(value) >= limit {
                    return None;
                }
                out.push(value);
            }
        } else {
            let (payload, rest) = bytes.split_at_checked(width.div_ceil(8))?;
            bytes = rest;
            let value = payload
                .iter()
                .enumerate()
                .map(|(shift, &byte)| u32::from(byte) << (shift * 8))
                .sum::<u32>();
            if run_len > remaining || u64::from(value) >= limit.min(1u64 << width) {
                return None;
            }
            out.extend(std::iter::repeat_n(value, run_len));
        }
    }
    // A final packed group can pad up to seven logical slots. Some
    // fastparquet pages additionally end in zero bytes.
    bytes.iter().all(|&byte| byte == 0).then_some(out)
}
