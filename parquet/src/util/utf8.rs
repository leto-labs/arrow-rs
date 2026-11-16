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

//! [`check_valid_utf8`] validation function
use crate::errors::{ParquetError, Result};

/// Check that `val` is a valid UTF-8 sequence.
///
/// If the `simdutf8` feature is enabled, this function will use
/// SIMD-accelerated validation from the [`simdutf8`] crate. Otherwise, it will use
/// [`std::str::from_utf8`].
///
/// # Errors
///
/// Returns `Err::General` with a message compatible with [`std::str::from_utf8`] on failure.
///
/// # Example
/// ```
/// use parquet::utf8::check_valid_utf8;
/// assert!(check_valid_utf8(b"hello").is_ok());
/// assert!(check_valid_utf8(b"hello \xF0\x9F\x98\x8E").is_ok());
/// // invalid UTF-8
/// assert!(check_valid_utf8(b"hello \xF0\x9F\x98").is_err());
/// ```
///
/// [`simdutf8`]: https://crates.io/crates/simdutf8
#[inline(always)]
pub fn check_valid_utf8(val: &[u8]) -> Result<()> {
    #[cfg(feature = "simdutf8")]
    match simdutf8::basic::from_utf8(val) {
        Ok(_) => Ok(()),
        Err(_) => {
            // Use simdutf8::compat to return details about the decoding error
            let e = simdutf8::compat::from_utf8(val).unwrap_err();
            Err(general_err!("encountered non UTF-8 data: {}", e))
        }
    }
    #[cfg(not(feature = "simdutf8"))]
    match std::str::from_utf8(val) {
        Ok(_) => Ok(()),
        Err(e) => Err(general_err!("encountered non UTF-8 data: {}", e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_matches_scalar(bytes: &[u8]) {
        let expected = std::str::from_utf8(bytes)
            .map(|_| ())
            .map_err(|error| ParquetError::General(format!("encountered non UTF-8 data: {error}")));
        assert_eq!(
            check_valid_utf8(bytes).map_err(|error| error.to_string()),
            expected.map_err(|error| error.to_string()),
            "UTF-8 validation differs for {bytes:?}"
        );
    }

    #[test]
    fn runtime_utf8_dispatch_matches_scalar_at_vector_boundaries() {
        let edges: &[&[u8]] = &[
            b"",
            b"\0%_\\",
            "é東京🦀".as_bytes(),
            &[0x80],
            &[0xc0, 0xaf],
            &[0xe0, 0x80, 0x80],
            &[0xed, 0xa0, 0x80],
            &[0xf4, 0x90, 0x80, 0x80],
            &[0xf0, 0x9f, 0xa6],
            &[0xff],
        ];
        for offset in [0, 1, 15, 16, 31, 32, 63, 64, 65, 127, 128, 255, 256] {
            for edge in edges {
                let mut bytes = vec![b'a'; offset];
                bytes.extend_from_slice(edge);
                assert_matches_scalar(&bytes);
                bytes.extend_from_slice(&[b'z'; 129]);
                assert_matches_scalar(&bytes);
            }
        }
    }

    #[test]
    fn runtime_utf8_dispatch_matches_scalar_for_random_valid_and_mutated_strings() {
        let alphabet = [
            '\0',
            '%',
            '_',
            '\\',
            'a',
            'é',
            '中',
            '🦀',
            '\u{d7ff}',
            '\u{e000}',
            '\u{10ffff}',
        ];
        let mut state = 0x5eed_cafe_u64;
        for length in 0..512 {
            let mut text = String::new();
            for _ in 0..length {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                text.push(alphabet[(state >> 32) as usize % alphabet.len()]);
            }
            assert_matches_scalar(text.as_bytes());
            let mut bytes = text.into_bytes();
            if !bytes.is_empty() {
                let index = (state >> 32) as usize % bytes.len();
                bytes[index] = (state >> 16) as u8;
                assert_matches_scalar(&bytes);
            }
            bytes.push(0xff);
            assert_matches_scalar(&bytes);
        }
    }
}
