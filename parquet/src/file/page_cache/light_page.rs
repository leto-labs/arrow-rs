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

//! Read-side representations of cached light pages.

use bytes::Bytes;
use std::sync::Arc;

use super::light::{BLOCK_LEN, LightBlock, LightCodeBlock};
use super::{CachedValidity, PageWithValidity, plain::is_dictionary_encoded};
use crate::basic::{Encoding, Type};
use crate::column::page::Page;
use crate::encodings::rle::{RleEncoder, codes::CodeRun};
use crate::errors::{ParquetError, Result};
use crate::file::{metadata::ColumnChunkMetaData, statistics::Statistics};

/// Excessively large pages stay in their ordinary representation.
pub const MAX_LIGHT_ROWS: usize = 1 << 20;

/// A reader's next page, with cached representations available to full readers.
#[doc(hidden)]
pub enum PageRead {
    /// A Parquet page and any separately cached definition levels.
    Parquet(PageWithValidity),
    /// A light data page.
    Light(Arc<LightDataPage>),
    /// A validated chunk dictionary.
    #[cfg(feature = "arrow")]
    Dictionary(Arc<arrow_array::StringViewArray>),
}

impl PageRead {
    /// Expand a cached representation for callers of the ordinary page API.
    pub(crate) fn into_parquet(self) -> Result<PageWithValidity> {
        match self {
            Self::Parquet(page) => Ok(page),
            Self::Light(page) => Ok((page.materialize()?, page.validity.clone())),
            #[cfg(feature = "arrow")]
            Self::Dictionary(dict) => Ok((materialize_dictionary(&dict)?, None)),
        }
    }
}

/// Logical values of a flat light data page, in consecutive 1,024-row blocks.
#[derive(Debug)]
pub enum LightValues {
    /// Physical INT32, including Date32 and Decimal32.
    Int32(Box<[LightBlock]>),
    /// Physical INT64, including Decimal64.
    Int64(Box<[LightBlock]>),
    /// Original dictionary indices of UTF-8 values.
    StringCodes(Box<[LightCodeBlock]>),
}

/// A flat cached data page, with the original page header retained by its cache entry.
#[derive(Debug)]
pub struct LightDataPage {
    rows: usize,
    pub(crate) values: LightValues,
    pub(crate) validity: Option<Arc<CachedValidity>>,
}

impl LightDataPage {
    /// Assemble already packed blocks. `validity` is Arrow bit order,
    /// present for every nullable page, even when there are zero nulls.
    /// Blocks must have been packed with the corresponding bits of this bitmap.
    pub fn try_new(rows: usize, values: LightValues, validity: Option<Bytes>) -> Result<Self> {
        if !(1..=MAX_LIGHT_ROWS).contains(&rows) {
            return Err(general_err!("invalid light page row count {rows}"));
        }
        let check = |lengths: &mut dyn Iterator<Item = usize>| {
            lengths.eq((0..rows)
                .step_by(BLOCK_LEN)
                .map(|start| (rows - start).min(BLOCK_LEN)))
        };
        let valid_blocks = match &values {
            LightValues::Int32(blocks) | LightValues::Int64(blocks) => {
                check(&mut blocks.iter().map(LightBlock::len))
            }
            LightValues::StringCodes(blocks) => check(&mut blocks.iter().map(LightCodeBlock::len)),
        };
        if !valid_blocks {
            return Err(general_err!("invalid light page block lengths"));
        }
        let validity = validity
            .map(|bits| CachedValidity::try_from_bits(bits, rows).map(Arc::new))
            .transpose()?;
        Ok(Self {
            rows,
            values,
            validity,
        })
    }

    /// Number of logical rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn physical_type(&self) -> Type {
        match self.values {
            LightValues::Int32(_) => Type::INT32,
            LightValues::Int64(_) => Type::INT64,
            LightValues::StringCodes(_) => Type::BYTE_ARRAY,
        }
    }

    pub(crate) fn has_codes(&self) -> bool {
        matches!(self.values, LightValues::StringCodes(_))
    }

    pub(crate) fn is_valid(&self, row: usize) -> bool {
        self.validity
            .as_ref()
            .is_none_or(|validity| validity.bits[row / 8] & (1 << (row % 8)) != 0)
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        let blocks: usize = match &self.values {
            LightValues::Int32(blocks) | LightValues::Int64(blocks) => {
                blocks.iter().map(LightBlock::retained_bytes).sum()
            }
            LightValues::StringCodes(blocks) => {
                blocks.iter().map(LightCodeBlock::retained_bytes).sum()
            }
        };
        size_of::<Self>()
            + 2 * size_of::<usize>()
            + blocks
            + self
                .validity
                .as_deref()
                .map_or(0, CachedValidity::buffer_len)
    }

    pub(crate) fn encoded_levels(&self) -> Bytes {
        let Some(_) = self.validity else {
            return Bytes::new();
        };
        let mut encoder = RleEncoder::new_from_buf(1, Vec::new());
        for row in 0..self.rows {
            encoder.put(u64::from(self.is_valid(row)));
        }
        Bytes::from(encoder.consume())
    }

    // Selected kernels and callers of the public page iterator require
    // ordinary layouts. Expand the entire page before handing it to them.
    fn materialize(&self) -> Result<Page> {
        let levels = self.encoded_levels();
        let mut data = levels.to_vec();
        let encoding = match &self.values {
            LightValues::Int32(blocks) => {
                let mut values = [0i32; BLOCK_LEN];
                for (index, block) in blocks.iter().enumerate() {
                    block.decode_i32_into(&mut values).map_err(light_error)?;
                    for (offset, value) in values[..block.len()].iter().enumerate() {
                        if self.is_valid(index * BLOCK_LEN + offset) {
                            data.extend_from_slice(&value.to_le_bytes());
                        }
                    }
                }
                Encoding::PLAIN
            }
            LightValues::Int64(blocks) => {
                let mut values = [0i64; BLOCK_LEN];
                for (index, block) in blocks.iter().enumerate() {
                    block.decode_into(&mut values).map_err(light_error)?;
                    for (offset, value) in values[..block.len()].iter().enumerate() {
                        if self.is_valid(index * BLOCK_LEN + offset) {
                            data.extend_from_slice(&value.to_le_bytes());
                        }
                    }
                }
                Encoding::PLAIN
            }
            LightValues::StringCodes(blocks) => {
                let width = blocks
                    .iter()
                    .filter_map(LightCodeBlock::max_code)
                    .max()
                    .map_or(0, |max| {
                        crate::util::bit_util::num_required_bits(u64::from(max))
                    });
                data.push(width);
                let mut encoder = RleEncoder::new_from_buf(width, data);
                for (index, block) in blocks.iter().enumerate() {
                    let run = block.code_run();
                    for offset in 0..run.len() {
                        if self.is_valid(index * BLOCK_LEN + offset) {
                            let code = run
                                .code_at(offset)
                                .ok_or_else(|| general_err!("invalid light dictionary code"))?;
                            encoder.put(u64::from(code));
                        }
                    }
                }
                data = encoder.consume();
                Encoding::RLE_DICTIONARY
            }
        };
        Ok(Page::DataPageV2 {
            buf: Bytes::from(data),
            num_values: self.rows as u32,
            encoding,
            num_nulls: self.validity.as_ref().map_or(0, |v| v.nulls as u32),
            num_rows: self.rows as u32,
            def_levels_byte_len: levels.len() as u32,
            rep_levels_byte_len: 0,
            is_compressed: false,
            statistics: None,
        })
    }
}

impl LightCodeBlock {
    /// Read original codes in their native layout. Apply page validity before
    /// using a slot; the code in a NULL slot is only a placeholder.
    pub fn code_run(&self) -> CodeRun<'_> {
        let len = self.len();
        if let Some((base, width, words)) = self.words() {
            CodeRun::Interleaved {
                base,
                width,
                words,
                len,
            }
        } else {
            CodeRun::Repeated {
                code: self.constant_code().expect("constant light code"),
                len,
            }
        }
    }
}

#[cfg(feature = "arrow")]
pub(crate) fn materialize_dictionary(dict: &arrow_array::StringViewArray) -> Result<Page> {
    let mut data = Vec::new();
    for entry in dict {
        let entry = entry.ok_or_else(|| general_err!("NULL in light chunk dictionary"))?;
        data.extend_from_slice(&u32::try_from(entry.len())?.to_le_bytes());
        data.extend_from_slice(entry.as_bytes());
    }
    Ok(Page::DictionaryPage {
        buf: Bytes::from(data),
        num_values: u32::try_from(dict.views().len())?,
        encoding: Encoding::PLAIN,
        is_sorted: false,
    })
}

pub(crate) fn light_error(err: super::light::LightError) -> crate::errors::ParquetError {
    general_err!("invalid cached light data: {err:?}")
}

/// Classify a chunk using its footer and the *original* dictionary count.
/// The fill still checks the encoding and the RLE run coverage of each page.
pub fn light_chunk_eligible(meta: &ColumnChunkMetaData, dictionary_entries: Option<u32>) -> bool {
    let descr = meta.column_descr();
    if descr.max_rep_level() != 0 || descr.max_def_level() > 1 || descr.path().parts().len() != 1 {
        return false;
    }

    let quarter = i128::from(meta.num_values()) / 4;
    match descr.physical_type() {
        Type::INT32 | Type::INT64 => {
            if dictionary_entries.is_none_or(|count| i128::from(count) > quarter) {
                return true;
            }

            let range = match meta.statistics() {
                Some(Statistics::Int32(stats)) => stats
                    .min_opt()
                    .zip(stats.max_opt())
                    .map(|(&min, &max)| i128::from(max) - i128::from(min) + 1),
                Some(Statistics::Int64(stats)) => stats
                    .min_opt()
                    .zip(stats.max_opt())
                    .map(|(&min, &max)| i128::from(max) - i128::from(min) + 1),
                _ => None,
            };
            range.is_some_and(|range| (1..=quarter).contains(&range))
        }
        Type::BYTE_ARRAY => {
            descr.converted_type() == crate::basic::ConvertedType::UTF8
                && dictionary_entries.is_some()
                && meta.page_encoding_stats().is_none_or(|stats| {
                    stats.iter().any(|stat| {
                        matches!(
                            stat.page_type,
                            crate::basic::PageType::DATA_PAGE
                                | crate::basic::PageType::DATA_PAGE_V2
                        ) && stat.count > 0
                            && is_dictionary_encoded(stat.encoding)
                    })
                })
                && meta.page_encoding_stats_mask().is_none_or(|mask| {
                    mask.is_set(Encoding::RLE_DICTIONARY) || mask.is_set(Encoding::PLAIN_DICTIONARY)
                })
        }
        _ => false,
    }
}
