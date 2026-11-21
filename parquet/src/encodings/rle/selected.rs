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

use crate::encodings::rle::codes::{CodeCursor, CodePosition};
use crate::errors::{ParquetError, Result};

impl CodeCursor<'_> {
    /// Gathers the requested ranks without checking the dictionary bounds
    /// of rejected codes. The caller must bounds-check the collected codes
    /// before indexing the dictionary. `position` belongs to this data page.
    pub(crate) fn read_selected(
        data: &[u8],
        values: usize,
        end: usize,
        ranks: &[u32],
        out: &mut Vec<u32>,
        position: &mut Option<CodePosition>,
    ) -> Result<()> {
        out.clear();
        if ranks.is_empty() {
            return Ok(());
        }
        let (&width, runs) = data
            .split_first()
            .ok_or_else(|| eof_err!("Missing dictionary index width"))?;
        // Checking the run's physical bit width still detects malformed RLE.
        // Only the collected codes need to refer to dictionary entries.
        CodeCursor::new(width, runs, values, 1u64 << 32)?.codes_in_range::<false>(
            ranks,
            out,
            position,
            Some(end),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encodings::rle::RleDecoder;
    use bytes::Bytes;
    use rand::{RngExt, SeedableRng, rngs::SmallRng};

    fn varint(mut value: usize, out: &mut Vec<u8>) {
        while value >= 128 {
            out.push(value as u8 | 128);
            value >>= 7;
        }
        out.push(value as u8);
    }

    #[test]
    fn selected_indices_match_full_decode_across_vectors() {
        let mut rng = SmallRng::seed_from_u64(128);
        for width in 0..=32u8 {
            let mask = ((1u64 << width) - 1) as u32;
            let run_counts: &[usize] = if cfg!(miri) {
                &[1, 13]
            } else {
                &[1, 13, 127, 10_000]
            };
            for &run_count in run_counts {
                let mut encoded = vec![width];
                let mut rows = 0;
                for run in 0..run_count {
                    let len = rng.random_range(1..=20);
                    if run % 2 == 0 {
                        varint(len * 2, &mut encoded);
                        let code = rng.random::<u32>() & mask;
                        encoded.extend_from_slice(
                            &code.to_le_bytes()[..usize::from(width).div_ceil(8)],
                        );
                        rows += len;
                    } else {
                        varint(len * 2 + 1, &mut encoded);
                        encoded.extend((0..len * usize::from(width)).map(|_| rng.random::<u8>()));
                        rows += 8 * len;
                    }
                }

                let encoded = Bytes::from(encoded);
                let mut full = RleDecoder::new(width);
                full.set_data(encoded.slice(1..)).unwrap();
                let mut expected = vec![0u32; rows];
                assert_eq!(full.get_batch(&mut expected).unwrap(), rows);

                for density in [0.0, 0.001, 0.02, 0.1, 0.5, 1.0] {
                    let mut position = None;
                    let mut actual = Vec::new();
                    for start in (0..rows).step_by(8191) {
                        let end = (start + 8191).min(rows);
                        let ranks: Vec<_> = (start..end)
                            .filter(|_| rng.random::<f64>() < density)
                            .map(|row| row as u32)
                            .collect();
                        CodeCursor::read_selected(
                            &encoded,
                            rows,
                            end,
                            &ranks,
                            &mut actual,
                            &mut position,
                        )
                        .unwrap();
                        assert!(
                            actual
                                .iter()
                                .copied()
                                .eq(ranks.iter().map(|&i| expected[i as usize]))
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn corrupt_selected_run_is_reported() {
        for stream in [&[1, 128][..], &[1, 2], &[1, 2, 2], &[1, 3], &[33, 2, 0]] {
            assert!(
                CodeCursor::read_selected(stream, 1, 1, &[0], &mut Vec::new(), &mut None).is_err()
            );
        }
    }
}
