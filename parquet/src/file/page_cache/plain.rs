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

use crate::basic::{Encoding, Type};
use crate::column::page::Page;
use crate::encodings::rle::RleDecoder;
use crate::file::metadata::ColumnChunkMetaData;
use crate::util::bit_util::num_required_bits;
use bytes::Bytes;

pub(crate) fn is_dictionary_encoded(encoding: Encoding) -> bool {
    matches!(
        encoding,
        Encoding::PLAIN_DICTIONARY | Encoding::RLE_DICTIONARY
    )
}

/// Fill-side state for flat, fixed-width numeric chunks. Original dictionary
/// pages are never admitted until their cardinality has been checked.
pub(crate) struct PlainChunk {
    min_entries: usize,
    width: usize,
    max_def: u32,
    dictionary: Option<Bytes>,
    classified: bool,
}

impl PlainChunk {
    pub(crate) fn new(meta: &ColumnChunkMetaData) -> Option<Self> {
        let width = match meta.column_type() {
            Type::INT32 | Type::FLOAT => 4,
            Type::INT64 | Type::DOUBLE => 8,
            _ => return None,
        };
        let descr = meta.column_descr();
        if descr.max_rep_level() != 0 {
            return None;
        }
        meta.dictionary_page_offset()?;
        Some(Self {
            min_entries: usize::try_from(meta.num_values()).ok()?.div_ceil(2),
            width,
            max_def: descr.max_def_level() as u32,
            dictionary: None,
            classified: false,
        })
    }

    pub(crate) fn needs_dictionary(&self) -> bool {
        !self.classified
    }

    pub(crate) fn observe_dictionary(&mut self, page: &Page) -> bool {
        let Page::DictionaryPage {
            buf,
            num_values,
            encoding,
            ..
        } = page
        else {
            return false;
        };
        self.classified = true;
        self.dictionary = (matches!(encoding, Encoding::PLAIN | Encoding::PLAIN_DICTIONARY)
            && *num_values as usize >= self.min_entries
            && (*num_values as usize).checked_mul(self.width) == Some(buf.len()))
        .then(|| buf.clone());
        self.dictionary.is_some()
    }

    /// Returns the representation to store. A dictionary data page for which
    /// we have not seen the dictionary, or cannot transcode, stays a miss.
    pub(crate) fn data_page(&self, page: &Page, owns_buffer: bool) -> Option<Page> {
        if !is_dictionary_encoded(page.encoding()) {
            return owns_buffer.then(|| page.clone());
        }
        if let Some(dict) = &self.dictionary {
            self.transcode(page.clone(), dict)
        } else {
            (self.classified && owns_buffer).then(|| page.clone())
        }
    }

    fn transcode(&self, mut page: Page, dictionary: &Bytes) -> Option<Page> {
        let (buf, encoding, prefix_len, count) = match &mut page {
            Page::DataPage {
                buf,
                encoding,
                num_values,
                def_level_encoding,
                ..
            } => {
                let count = *num_values as usize;
                let (prefix_len, count) = if self.max_def == 0 {
                    (0, count)
                } else {
                    if *def_level_encoding != Encoding::RLE {
                        return None;
                    }
                    let len = u32::from_le_bytes(buf.get(..4)?.try_into().ok()?) as usize;
                    let end = 4_usize.checked_add(len)?;
                    buf.get(..end)?;
                    let count = count_defined(buf.slice(4..end), self.max_def, count)?;
                    (end, count)
                };
                (buf, encoding, prefix_len, count)
            }
            Page::DataPageV2 {
                buf,
                encoding,
                num_values,
                num_nulls,
                def_levels_byte_len,
                rep_levels_byte_len,
                ..
            } => {
                let prefix_len =
                    (*def_levels_byte_len as usize).checked_add(*rep_levels_byte_len as usize)?;
                let count = num_values.checked_sub(*num_nulls)? as usize;
                let defined = if self.max_def == 0 {
                    *num_values as usize
                } else {
                    buf.get(..prefix_len)?;
                    count_defined(
                        buf.slice(*rep_levels_byte_len as usize..prefix_len),
                        self.max_def,
                        *num_values as usize,
                    )?
                };
                if count != defined {
                    return None;
                }
                (buf, encoding, prefix_len, count)
            }
            Page::DictionaryPage { .. } => return None,
        };
        let prefix = buf.get(..prefix_len)?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(prefix_len.checked_add(count.checked_mul(self.width)?)?)
            .ok()?;
        output.extend_from_slice(prefix);
        if count != 0 {
            let bit_width = *buf.get(prefix_len)?;
            if bit_width > 32 {
                return None;
            }
            let mut ids = RleDecoder::new(bit_width);
            ids.set_data(buf.slice(prefix_len + 1..)).ok()?;
            let mut scratch = [0_u32; 256];
            let mut left = count;
            while left != 0 {
                let size = left.min(scratch.len());
                if ids.get_batch(&mut scratch[..size]).ok()? != size {
                    return None;
                }
                for &id in &scratch[..size] {
                    let start = (id as usize).checked_mul(self.width)?;
                    output
                        .extend_from_slice(dictionary.get(start..start.checked_add(self.width)?)?);
                }
                left -= size;
            }
        }
        *buf = Bytes::from(output);
        *encoding = Encoding::PLAIN;
        Some(page)
    }
}

fn count_defined(data: Bytes, max_def: u32, count: usize) -> Option<usize> {
    let mut decoder = RleDecoder::new(num_required_bits(u64::from(max_def)));
    decoder.set_data(data).ok()?;
    let (defined, read) = decoder.skip_and_count(count, max_def).ok()?;
    (read == count).then_some(defined)
}
