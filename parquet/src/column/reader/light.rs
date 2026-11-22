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

//! Full reads of cached flat light pages. Definition levels advance in the
//! column reader, which gives this decoder the matching logical row span.

use crate::errors::{ParquetError, Result};
use crate::file::page_cache::{LightDataPage, LightValues, light::BLOCK_LEN, light_error};
use std::{any::Any, ops::Range, sync::Arc};

pub(crate) struct LightDecoder {
    page: Arc<LightDataPage>,
    pub(crate) rows: Range<usize>,
}

impl LightDecoder {
    pub(crate) fn new(page: Arc<LightDataPage>) -> Self {
        Self { page, rows: 0..0 }
    }

    pub(super) fn read_native(&self, output: &mut dyn Any, spaced: bool) -> Result<()> {
        match &self.page.values {
            LightValues::Int32(blocks) => {
                let out = output
                    .downcast_mut::<Vec<i32>>()
                    .ok_or_else(|| general_err!("light INT32 page for a different output type"))?;
                self.read_blocks(
                    blocks,
                    out,
                    spaced,
                    |block, out| block.decode_i32_into(out).map_err(light_error),
                    |block, out| block.append_i32(out).map_err(light_error),
                )
            }
            LightValues::Int64(blocks) => {
                let out = output
                    .downcast_mut::<Vec<i64>>()
                    .ok_or_else(|| general_err!("light INT64 page for a different output type"))?;
                self.read_blocks(
                    blocks,
                    out,
                    spaced,
                    |block, out| block.decode_into(out).map_err(light_error),
                    |block, out| block.append_i64(out).map_err(light_error),
                )
            }
            LightValues::StringCodes(_) => Err(general_err!("light numeric reader received codes")),
        }
    }

    #[cfg(feature = "arrow")]
    pub(crate) fn read_views(
        &self,
        output: &mut Vec<u128>,
        dictionary: &[u128],
        buffer_base: u32,
    ) -> Result<()> {
        let LightValues::StringCodes(blocks) = &self.page.values else {
            return Err(general_err!("light string reader received numeric values"));
        };
        let start = output.len();
        if let Some(validity) = &self.page.validity
            && validity.nulls != 0
        {
            // Unpack the block once, then apply validity before touching
            // the dictionary. Code consumers that need one row at a time
            // borrow the native Interleaved run instead.
            let mut codes = [0; BLOCK_LEN];
            let mut row = self.rows.start;
            while row < self.rows.end {
                let block = row / BLOCK_LEN;
                let first = block * BLOCK_LEN;
                let block = &blocks[block];
                let end = (first + block.len()).min(self.rows.end);
                if block
                    .max_code()
                    .is_some_and(|key| key as usize >= dictionary.len())
                {
                    return Err(general_err!("invalid light string dictionary key"));
                }
                block.decode_into(&mut codes).map_err(light_error)?;
                let valid = &validity.bits;
                output.extend(
                    (row..end)
                        .filter(|&row| valid[row / 8] & (1 << (row % 8)) != 0)
                        .map(|row| dictionary[codes[row - first] as usize]),
                );
                row = end;
            }
        } else {
            self.read_blocks(
                blocks,
                output,
                true,
                |block, out| block.decode_views(dictionary, out).map_err(light_error),
                |block, out| {
                    let start = out.len();
                    out.resize(start + block.len(), 0);
                    block
                        .decode_views(dictionary, &mut out[start..])
                        .map_err(light_error)
                },
            )?;
        }
        if buffer_base != 0 {
            for view in &mut output[start..] {
                if *view as u32 > 12 {
                    let mut bytes = arrow_data::ByteView::from(*view);
                    bytes.buffer_index += buffer_base;
                    *view = bytes.into();
                }
            }
        }
        Ok(())
    }

    fn read_blocks<B, T: Copy + Default>(
        &self,
        blocks: &[B],
        output: &mut Vec<T>,
        spaced: bool,
        mut decode: impl FnMut(&B, &mut [T]) -> Result<()>,
        mut append: impl FnMut(&B, &mut Vec<T>) -> Result<()>,
    ) -> Result<()> {
        let logical = spaced || self.page.validity.is_none();
        let mut start = self.rows.start;
        while start < self.rows.end {
            let block = start / BLOCK_LEN;
            let block_start = block * BLOCK_LEN;
            let block_end = (block_start + BLOCK_LEN).min(self.page.rows());
            let end = block_end.min(self.rows.end);
            let block = &blocks[block];
            if logical && start == block_start && end == block_end {
                append(block, output)?;
            } else {
                let mut scratch = [T::default(); BLOCK_LEN];
                decode(block, &mut scratch[..block_end - block_start])?;
                let span = &scratch[start - block_start..end - block_start];
                if logical {
                    output.extend_from_slice(span);
                } else {
                    output.extend(
                        span.iter().copied().enumerate().filter_map(|(i, value)| {
                            self.page.is_valid(start + i).then_some(value)
                        }),
                    );
                }
            }
            start = end;
        }
        Ok(())
    }
}
