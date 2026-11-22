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

//! Offline light-page block kernels. A page keeps its validity and chunk
//! dictionary separately from these blocks.
//!
//! A packed NULL slot has the block base. A run absorbs NULLs up to the next
//! distinct valid value, with leading NULLs in the first run. The validity
//! bitmap is authoritative in both cases.

mod format;
mod pack;
mod unpack;

pub use format::{BLOCK_LEN, LightBlock, LightCodeBlock, LightError, LightKind};

#[cfg(test)]
mod tests {
    use super::*;

    fn random(state: &mut u64) -> u64 {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        state.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    #[test]
    fn frames_match_input_at_every_width() {
        let mut state = 0xa826_7d81_aeff_26cd;
        let lengths: &[usize] = if cfg!(miri) {
            &[2, 33]
        } else {
            &[2, 17, 33, 127, 1023, BLOCK_LEN]
        };
        for width in 1..=64 {
            for &len in lengths {
                for pattern in 0..5 {
                    let base = i64::MIN + (random(&mut state) % 4096) as i64;
                    let base = if width == 64 { i64::MIN } else { base };
                    let mask = u64::MAX >> (64 - width);
                    let mut input: Vec<_> = (0..len)
                        .map(|_| base.wrapping_add((random(&mut state) & mask) as i64))
                        .collect();
                    input[0] = base;
                    input[len - 1] = base.wrapping_add(mask as i64);
                    let valid: Vec<_> = (0..len)
                        .map(|i| match pattern {
                            0 => true,
                            1 => false,
                            2 => i % 2 == 0,
                            3 => i % 51 < 31,
                            _ => random(&mut state).is_multiple_of(3),
                        })
                        .collect();
                    let mut valid = valid;
                    if pattern != 1 {
                        valid[0] = true;
                        valid[len - 1] = true;
                    }
                    let mut bitmap = vec![0u8; len.div_ceil(8)];
                    let mut expected = input.clone();
                    for (i, &is_valid) in valid.iter().enumerate() {
                        if is_valid {
                            bitmap[i / 8] |= 1 << (i % 8);
                        } else {
                            input[i] = if i % 2 == 0 { i64::MAX } else { i64::MIN };
                            expected[i] = if pattern == 1 { 0 } else { base };
                        }
                    }
                    let validity = (pattern != 0).then_some(bitmap.as_slice());
                    let block = LightBlock::pack(&input, validity).unwrap();
                    #[cfg(not(miri))]
                    assert_eq!(block, LightBlock::pack_scalar(&input, validity).unwrap());
                    if len == BLOCK_LEN && pattern == 0 {
                        assert_eq!(block.kind(), LightKind::Packed(width));
                    }
                    let mut scalar = vec![9; len + 2];
                    let mut dispatched = scalar.clone();
                    block.decode_scalar_into(&mut scalar).unwrap();
                    block.decode_into(&mut dispatched).unwrap();
                    assert_eq!(
                        scalar, dispatched,
                        "width {width}, length {len}, pattern {pattern}"
                    );
                    assert_eq!(&scalar[len..], &[9, 9]);
                    if block.kind() != LightKind::Runs {
                        assert_eq!(
                            &scalar[..len],
                            &expected,
                            "width {width}, length {len}, pattern {pattern}"
                        );
                        assert_eq!(
                            block.kind(),
                            if pattern == 1 {
                                LightKind::Constant
                            } else {
                                LightKind::Packed(width)
                            }
                        );
                    } else {
                        for (i, value) in scalar[..len].iter().enumerate() {
                            if valid[i] {
                                assert_eq!(*value, expected[i]);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn nulls_keep_the_enclosing_run_and_do_not_change_its_end() {
        let mut input = [i64::MAX; BLOCK_LEN];
        let mut bitmap = [0xEE; BLOCK_LEN / 8];
        for (index, value) in input.iter_mut().enumerate() {
            if bitmap[index / 8] & (1 << (index % 8)) != 0 {
                *value = if index < 511 { -13 } else { 17 };
            }
        }
        // The first valid value of the second run occurs at 511. The NULL
        // slot at 508 remains in the first run, and leading NULLs join it.
        let mut output = [0; BLOCK_LEN];
        let block = LightBlock::pack(&input, Some(&bitmap)).unwrap();
        assert_eq!(block.kind(), LightKind::Runs);
        block.decode_into(&mut output).unwrap();
        assert_eq!(output[..511], [-13; 511]);
        assert_eq!(output[511..], [17; BLOCK_LEN - 511]);

        bitmap.fill(0);
        let all_null = LightBlock::pack(&input, Some(&bitmap)).unwrap();
        assert_eq!(all_null.kind(), LightKind::Constant);
        all_null.decode_into(&mut output).unwrap();
        assert_eq!(output, [0; BLOCK_LEN]);
    }

    #[test]
    fn signed_32_bit_frames_and_original_codes() {
        let input = [i32::MIN, -10_000_000, -1, 0, 7, 10_000_000, i32::MAX];
        let block = LightBlock::pack_i32(&input, None).unwrap();
        let mut output = [0; 7];
        block.decode_into(&mut output).unwrap();
        assert_eq!(output, input.map(i64::from));
        let mut narrow = [0; 7];
        block.decode_i32_into(&mut narrow).unwrap();
        assert_eq!(narrow, input);
        assert_eq!(
            LightBlock::pack(&[i64::MAX], None)
                .unwrap()
                .decode_i32_into(&mut narrow),
            Err(LightError::ValueOutOfRange)
        );

        let codes = [u32::MAX, 4, 2, 4, 3, 5, 5];
        let validity = [0xfe];
        let block = LightCodeBlock::pack(&codes, Some(&validity), 6).unwrap();
        let mut output = [0; 7];
        block.decode_into(&mut output).unwrap();
        assert_eq!(output, [2, 4, 2, 4, 3, 5, 5]);
        let dictionary = [10, 20, 30, 40, 50, 60];
        let mut views = [0; 7];
        block.decode_views(&dictionary, &mut views).unwrap();
        assert_eq!(views, [30, 50, 30, 50, 40, 60, 60]);
        assert_eq!(
            block.decode_views(&dictionary[..5], &mut views),
            Err(LightError::DictionaryIndex)
        );
        let all_null = LightCodeBlock::pack(&codes, Some(&[0]), 0).unwrap();
        all_null.decode_views(&[], &mut views).unwrap();
        assert_eq!(views, [0; 7]);
    }

    #[test]
    fn light_code_words_interleave_rows_and_use_the_frame_base() {
        let codes: [u32; 64] = std::array::from_fn(|i| {
            let lane = (i % 4) as u32;
            97 + if i < 32 { lane } else { 3 - lane }
        });
        let light = LightCodeBlock::pack(&codes, None, 101).unwrap();
        let format::Values::Packed(frame) = &light.block.values else {
            panic!("expected a frame")
        };
        let format::Words::Narrow(words) = &frame.words else {
            panic!("expected 32-bit words")
        };
        assert_eq!(frame.base, 97);
        assert_eq!(frame.width, 2);
        assert_eq!(&words[..4], &[0b1100, 0b1001, 0b0110, 0b0011]);
        let mut decoded = [0; 64];
        light.decode_into(&mut decoded).unwrap();
        assert_eq!(decoded, codes);
    }

    #[test]
    fn rejects_invalid_dimensions_before_decoding() {
        assert_eq!(
            LightBlock::pack(&[], None).unwrap_err(),
            LightError::BlockLength
        );
        assert_eq!(
            LightBlock::pack(&[0; BLOCK_LEN + 1], None).unwrap_err(),
            LightError::BlockLength
        );
        assert_eq!(
            LightBlock::pack(&[0; 9], Some(&[0])).unwrap_err(),
            LightError::ValidityLength
        );
        assert_eq!(
            LightCodeBlock::pack(&[2], None, 2).unwrap_err(),
            LightError::DictionaryIndex
        );
        let block = LightBlock::pack(&[0, 1], None).unwrap();
        let mut out = [42];
        assert_eq!(block.decode_into(&mut out), Err(LightError::OutputLength));
        assert_eq!(out, [42]);
    }
}
