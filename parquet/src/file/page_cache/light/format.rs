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

//! Owned, in-memory block format. All fields that determine kernel bounds
//! are private to the light module.

/// Maximum number of logical rows in a block.
pub const BLOCK_LEN: usize = 1024;

/// Encoding of one block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightKind {
    /// All rows have one physical value.
    Constant,
    /// Frame of reference over signed values, with unsigned offsets.
    Packed(u8),
    /// Frame of reference over the values and exclusive ends of the runs.
    Runs,
}

/// Integer values of a light page. A page uses full blocks followed by at most
/// one short block. The page owns the validity bitmap separately.
#[derive(Debug)]
pub struct LightBlock {
    pub(super) len: usize,
    pub(super) values: Values,
}

#[derive(Debug)]
pub(super) enum Values {
    Constant(i64),
    Packed(Frame),
    Runs { values: Frame, ends: Frame },
}

/// A frame stores little-endian *bit order* in native words. For a logical
/// index `r * lanes + l`, the offset starts at bit `r * width` of lane `l`.
/// Word `k` of that lane is at index `k * lanes + l`. The last word group
/// is zero-padded. Short frames keep the same lane stride.
#[derive(Debug)]
pub(super) struct Frame {
    pub(super) len: usize,
    pub(super) base: i64,
    pub(super) width: u8,
    pub(super) words: Words,
}

#[derive(Debug)]
pub(super) enum Words {
    Narrow(Box<[u32]>),
    Wide(Box<[u64]>),
}

/// Light codes preserve the original Parquet dictionary indices. Their
/// interleaved frame-of-reference layout is distinct from the eight-code
/// groups of Parquet's hybrid RLE encoding.
#[derive(Debug)]
pub struct LightCodeBlock {
    pub(super) block: LightBlock,
    pub(super) all_null: bool,
}

/// An invalid kernel input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightError {
    /// A block must contain 1 through 1,024 logical values.
    BlockLength,
    /// A validity bitmap must cover the input block.
    ValidityLength,
    /// An output slice is too short for the block.
    OutputLength,
    /// A valid code is outside the supplied dictionary.
    DictionaryIndex,
}

impl std::fmt::Display for LightError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::BlockLength => "light block must contain 1 through 1,024 values",
            Self::ValidityLength => "light validity bitmap has the wrong length",
            Self::OutputLength => "light output has fewer slots than the block",
            Self::DictionaryIndex => "light code is outside the dictionary",
        })
    }
}

impl std::error::Error for LightError {}

impl LightBlock {
    /// Number of logical rows, including NULLs.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this block has zero rows. Constructed blocks always have rows.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Encoding selected by the packer.
    pub fn kind(&self) -> LightKind {
        match &self.values {
            Values::Constant(_) => LightKind::Constant,
            Values::Packed(frame) => LightKind::Packed(frame.width),
            Values::Runs { .. } => LightKind::Runs,
        }
    }

    /// Retained bytes, including the inline block and out-of-line word arrays.
    /// The caller accounts for the containing page and validity separately.
    pub fn retained_bytes(&self) -> usize {
        size_of::<Self>()
            + match &self.values {
                Values::Constant(_) => 0,
                Values::Packed(frame) => frame.words.bytes(),
                Values::Runs { values, ends } => values.words.bytes() + ends.words.bytes(),
            }
    }
}

impl Frame {
    pub(super) fn width(min: i64, max: i64) -> u8 {
        (64 - (max.wrapping_sub(min) as u64).leading_zeros()) as u8
    }

    pub(super) fn word_count(len: usize, width: usize, lanes: usize, bits: usize) -> usize {
        (len.div_ceil(lanes) * width).div_ceil(bits) * lanes
    }

    pub(super) fn bytes(len: usize, width: u8) -> usize {
        let width = usize::from(width);
        if width <= 32 {
            Self::word_count(len, width, 32, 32) * size_of::<u32>()
        } else {
            Self::word_count(len, width, 16, 64) * size_of::<u64>()
        }
    }
}

impl Words {
    fn bytes(&self) -> usize {
        match self {
            Self::Narrow(words) => size_of_val(words.as_ref()),
            Self::Wide(words) => size_of_val(words.as_ref()),
        }
    }
}

impl LightCodeBlock {
    /// Number of logical codes, including NULL slots.
    pub fn len(&self) -> usize {
        self.block.len()
    }

    /// Whether this block has zero rows. Constructed blocks always have rows.
    pub fn is_empty(&self) -> bool {
        self.block.is_empty()
    }

    /// Retained bytes, including the inline block and out-of-line words.
    /// The caller accounts for the chunk dictionary and validity separately.
    pub fn retained_bytes(&self) -> usize {
        self.block.retained_bytes() + size_of::<Self>() - size_of::<LightBlock>()
    }
}
