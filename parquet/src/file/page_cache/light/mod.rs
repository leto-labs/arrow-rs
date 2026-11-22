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
//! A packed NULL slot holds the block base. The validity bitmap is authoritative.

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
                }
            }
        }
    }

    #[test]
    fn nulls_hold_the_frame_base_even_after_the_value_changes() {
        let input = std::array::from_fn::<_, BLOCK_LEN, _>(|row| if row < 511 { -13 } else { 17 });
        let bitmap = [0xEE; BLOCK_LEN / 8];
        let block = LightBlock::pack(&input, Some(&bitmap)).unwrap();
        assert_eq!(block.kind(), LightKind::Packed(5));
        let mut output = [0; BLOCK_LEN];
        block.decode_into(&mut output).unwrap();
        for (row, &value) in output.iter().enumerate() {
            assert_eq!(
                value,
                if row.is_multiple_of(4) {
                    -13
                } else {
                    input[row]
                }
            );
        }

        let all_null = LightBlock::pack(&input, Some(&[0; BLOCK_LEN / 8])).unwrap();
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
        assert_eq!(output, [0, 4, 2, 4, 3, 5, 5]);
        let dictionary = [10, 20, 30, 40, 50, 60];
        let mut views = [0; 7];
        block.decode_views(&dictionary, &mut views).unwrap();
        assert_eq!(views, [10, 50, 30, 50, 40, 60, 60]);
        assert_eq!(
            block.decode_views(&dictionary[..5], &mut views),
            Err(LightError::DictionaryIndex)
        );
        let all_null = LightCodeBlock::pack(&codes, Some(&[0]), 0).unwrap();
        all_null.decode_views(&[], &mut views).unwrap();
        assert_eq!(views, [0; 7]);
    }

    #[test]
    fn light_code_words_interleave_rows_and_keep_original_indices() {
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
        assert_eq!(frame.base, 0);
        assert_eq!(frame.width, 7);
        assert_eq!(
            &words[..4],
            &[97 | 100 << 7, 98 | 99 << 7, 99 | 98 << 7, 100 | 97 << 7]
        );
        let mut decoded = [0; 64];
        light.decode_into(&mut decoded).unwrap();
        assert_eq!(decoded, codes);
    }

    #[test]
    fn append_initializes_only_the_new_slots_and_keeps_length_on_error() {
        let mut small = vec![123];
        let mut wide = vec![321];
        for width in 1..=32 {
            let offset = u32::MAX >> (32 - width);
            let start = i32::MIN as i64;
            let input: [i32; BLOCK_LEN - 1] = std::array::from_fn(|i| {
                (start
                    + i64::from(if i == 1 {
                        offset
                    } else {
                        (i as u32 * 301) & offset
                    })) as i32
            });
            let block = LightBlock::pack_i32(&input, None).unwrap();
            block.append_i32(&mut small).unwrap();
            assert_eq!(&small[small.len() - input.len()..], &input);
            block.append_i64(&mut wide).unwrap();
            assert_eq!(&wide[wide.len() - input.len()..], &input.map(i64::from));
        }
        assert_eq!(small[0], 123);
        assert_eq!(wide[0], 321);
        let before = small.len();
        let out_of_range = LightBlock::pack(&[0, i64::MAX], None).unwrap();
        assert_eq!(
            out_of_range.append_i32(&mut small),
            Err(LightError::ValueOutOfRange)
        );
        assert_eq!(small.len(), before);
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
