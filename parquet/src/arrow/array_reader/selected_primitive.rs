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

use crate::basic::Encoding;
use crate::column::page::{Page, PageIterator, PageMetadata, PageReader};
use crate::column::reader::decoder::code_page::{
    ChunkDictionary, DictionaryCodePage, FlatDataPage, gather_plain,
};
use crate::data_type::DataType;
use crate::encodings::decoding::{Decoder, DictDecoder, get_decoder};
use crate::encodings::rle::codes::{CodePosition, low_mask};
use crate::errors::{ParquetError, Result};
use crate::file::page_cache::PageWithValidity;
use crate::schema::types::ColumnDescPtr;
use arrow_buffer::{BooleanBufferBuilder, NullBuffer};
use bytes::Bytes;

pub(super) struct SelectedPrimitive<T: DataType> {
    descr: ColumnDescPtr,
    chunk: Option<Box<dyn PageReader>>,
    dictionary: Option<(Page, ChunkDictionary)>,
    page: Option<SelectedPage>,
    selection: Vec<usize>,
    ranks: Vec<u32>,
    codes: Vec<u32>,
    validity: Vec<u64>,
    prefix: Vec<usize>,
    gathered: Vec<T::T>,
    decoded: Vec<T::T>,
}

struct SelectedPage {
    original: Page,
    values: PageValues,
    offset: usize,
    rows: usize,
    non_nulls: usize,
}

enum PageValues {
    Codes(DictionaryCodePage, Option<CodePosition>),
    Plain(Bytes),
    Decoded,
}

type Batch<T> = (Vec<T>, Option<NullBuffer>);

impl<T: DataType> SelectedPrimitive<T>
where
    T::T: Copy,
{
    pub(super) fn new(descr: ColumnDescPtr) -> Self {
        Self {
            descr,
            chunk: None,
            dictionary: None,
            page: None,
            selection: Vec::new(),
            ranks: Vec::new(),
            codes: Vec::new(),
            validity: Vec::new(),
            prefix: Vec::new(),
            gathered: Vec::new(),
            decoded: Vec::new(),
        }
    }

    pub(super) fn read(
        &mut self,
        pages: &mut dyn PageIterator,
        words: &[u64],
        start: usize,
        end: usize,
    ) -> Result<Batch<T::T>> {
        self.selection.clear();
        for (index, &word) in words[..end.div_ceil(64)]
            .iter()
            .enumerate()
            .skip(start / 64)
        {
            let base = index * 64;
            let mut word =
                word & (u64::MAX << start.saturating_sub(base)) & low_mask((end - base).min(64));
            while word != 0 {
                self.selection.push(base + word.trailing_zeros() as usize);
                word &= word - 1;
            }
        }

        let mut values = Vec::with_capacity(self.selection.len());
        let mut nulls = (self.descr.max_def_level() != 0)
            .then(|| BooleanBufferBuilder::new(self.selection.len()));
        let mut row = start;
        let mut kept = 0;
        while row < end {
            if self.page.is_none() {
                let gap = self.selection.get(kept).copied().unwrap_or(end) - row;
                row += self.advance(pages, gap)?;
                if self.page.is_none() {
                    continue;
                }
            }
            let page = self.page.as_mut().unwrap();
            let rows = (end - row).min(page.rows - page.offset);
            let count = self.selection[kept..].partition_point(|&selected| selected < row + rows);
            if count != 0 {
                let offset = page.offset;
                let mut selected = self.selection[kept..kept + count]
                    .iter()
                    .map(|&selected| selected - row + offset);
                self.ranks.clear();

                if let Some(nulls) = &mut nulls {
                    self.ranks.extend(selected.clone().filter_map(|i| {
                        let word = self.validity[i / 64];
                        ((word >> (i % 64)) & 1 != 0).then(|| {
                            (self.prefix[i / 64] + (word & low_mask(i % 64)).count_ones() as usize)
                                as u32
                        })
                    }));
                    if self.ranks.len() == count {
                        nulls.append_n(count, true);
                    }
                } else {
                    self.ranks.extend(selected.clone().map(|i| i as u32));
                }

                let end_rank = if let Some(&last) = self.ranks.last() {
                    if page.offset + rows == page.rows {
                        page.non_nulls
                    } else {
                        last as usize + 1
                    }
                } else {
                    0
                };
                if self.ranks.len() == count {
                    page.gather(
                        &self.ranks,
                        end_rank,
                        &mut self.codes,
                        &self.decoded,
                        &mut values,
                    )?;
                } else {
                    self.gathered.clear();
                    page.gather(
                        &self.ranks,
                        end_rank,
                        &mut self.codes,
                        &self.decoded,
                        &mut self.gathered,
                    )?;
                    let mut present = self.gathered.iter().copied();
                    let nulls = nulls.as_mut().unwrap();
                    for i in selected.by_ref() {
                        let valid = (self.validity[i / 64] >> (i % 64)) & 1 != 0;
                        nulls.append(valid);
                        values.push(if valid {
                            present.next().unwrap()
                        } else {
                            T::T::default()
                        });
                    }
                }
            }
            row += rows;
            kept += count;
            page.offset += rows;
            if page.offset == page.rows {
                self.page = None;
            }
        }
        let nulls = nulls
            .map(|mut builder| NullBuffer::new(builder.finish()))
            .filter(|nulls| nulls.null_count() != 0);
        Ok((values, nulls))
    }

    // Returns the number of rows skipped, or loads the next data page.
    fn advance(&mut self, pages: &mut dyn PageIterator, gap: usize) -> Result<usize> {
        loop {
            let chunk = match &mut self.chunk {
                Some(chunk) => chunk,
                None => {
                    self.dictionary = None;
                    self.chunk.insert(
                        pages
                            .next()
                            .ok_or_else(|| eof_err!("not enough rows for selected vector"))??,
                    )
                }
            };
            if gap != 0
                && let Some(meta) = chunk.peek_next_page()?
            {
                if meta.is_dict && chunk.supports_deferred_dictionary() {
                    chunk.skip_next_page()?;
                    continue;
                }
                if !meta.is_dict
                    && let Some(rows) = meta.num_rows.or(meta.num_levels)
                    && rows <= gap
                {
                    chunk.skip_unselected_data_page()?;
                    if rows != 0 {
                        return Ok(rows);
                    }
                    continue;
                }
            }
            match chunk.get_next_page()? {
                None => self.chunk = None,
                Some(page) if page.is_dictionary_page() => self.install_dictionary(page)?,
                Some(page) => {
                    self.load(page)?;
                    return Ok(0);
                }
            }
        }
    }

    fn install_dictionary(&mut self, page: Page) -> Result<()> {
        if self.dictionary.is_some() {
            return Err(general_err!("Unexpected dictionary page"));
        }
        let dictionary = ChunkDictionary::new_native(&page, T::get_physical_type())?;
        self.dictionary = Some((page, dictionary));
        Ok(())
    }

    fn load(&mut self, original: Page) -> Result<()> {
        let nullable = self.descr.max_def_level() != 0;
        let page = FlatDataPage::new(&original, nullable)?;
        let rows = page.rows;
        let non_nulls = if nullable {
            let count = page.validity(&mut self.validity)?;
            self.prefix.clear();
            let mut base = 0;
            self.prefix.extend(self.validity.iter().map(|word| {
                let start = base;
                base += word.count_ones() as usize;
                start
            }));
            count
        } else {
            rows
        };
        let dict_encoded = matches!(
            page.encoding,
            Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
        );
        if dict_encoded
            && self.dictionary.is_none()
            && let Some(dict) = self.chunk.as_mut().unwrap().take_deferred_dictionary()?
        {
            self.install_dictionary(dict)?;
        }
        let values = if dict_encoded && page.has_hybrid_levels() {
            let (_, dictionary) = self
                .dictionary
                .as_ref()
                .ok_or_else(|| general_err!("Missing column chunk dictionary"))?;
            PageValues::Codes(page.into_dictionary(dictionary.clone())?, None)
        } else if page.encoding == Encoding::PLAIN {
            PageValues::Plain(page.data)
        } else {
            self.decode_fallback(page, non_nulls, dict_encoded)?;
            PageValues::Decoded
        };
        self.page = Some(SelectedPage {
            original,
            values,
            offset: 0,
            rows,
            non_nulls,
        });
        Ok(())
    }

    fn decode_fallback(
        &mut self,
        page: FlatDataPage,
        count: usize,
        dict_encoded: bool,
    ) -> Result<()> {
        let mut decoder: Box<dyn Decoder<T>> = if dict_encoded {
            let (dictionary, _) = self
                .dictionary
                .as_ref()
                .ok_or_else(|| general_err!("Missing column chunk dictionary"))?;
            let mut decoder = DictDecoder::<T>::new();
            decoder.set_dict_from_page(
                dictionary.buffer().clone(),
                dictionary.num_values() as usize,
                self.descr.type_length(),
            )?;
            Box::new(decoder)
        } else {
            get_decoder::<T>(self.descr.clone(), page.encoding)?
        };
        decoder.set_data(page.data, count)?;
        self.decoded.clear();
        let mut buffer = [T::T::default(); 1024];
        while self.decoded.len() < count {
            let want = (count - self.decoded.len()).min(buffer.len());
            if decoder.get(&mut buffer[..want])? != want {
                return Err(eof_err!("Incomplete value page"));
            }
            self.decoded.extend_from_slice(&buffer[..want]);
        }
        Ok(())
    }

    pub(super) fn into_replay(self) -> Option<(Box<dyn PageReader>, usize)> {
        let tail = self.chunk?;
        let offset = self.page.as_ref().map_or(0, |page| page.offset);
        let reader = ReplayPages {
            dictionary: self.dictionary.map(|(original, _)| original),
            data: self.page.map(|page| page.original),
            tail,
        };
        Some((Box::new(reader), offset))
    }
}

impl SelectedPage {
    #[inline]
    fn gather<T: crate::data_type::private::ParquetValueType + Copy>(
        &mut self,
        ranks: &[u32],
        end: usize,
        codes: &mut Vec<u32>,
        decoded: &[T],
        out: &mut Vec<T>,
    ) -> Result<()> {
        match &mut self.values {
            PageValues::Codes(page, position) => {
                if !ranks.is_empty() {
                    page.codes_in_range(self.non_nulls, end, ranks, codes, position)?;
                    page.dictionary().gather(codes, out)?;
                }
                Ok(())
            }
            PageValues::Plain(data) => gather_plain(data, self.non_nulls, ranks, out),
            PageValues::Decoded => {
                out.extend(ranks.iter().map(|&rank| decoded[rank as usize]));
                Ok(())
            }
        }
    }
}

// The ordinary record reader restarts at the active page, retaining its own
// decoders and null handling. Earlier pages of this chunk stay consumed.
struct ReplayPages {
    dictionary: Option<Page>,
    data: Option<Page>,
    tail: Box<dyn PageReader>,
}

impl Iterator for ReplayPages {
    type Item = Result<Page>;

    fn next(&mut self) -> Option<Self::Item> {
        self.get_next_page().transpose()
    }
}

impl PageReader for ReplayPages {
    fn get_next_page(&mut self) -> Result<Option<Page>> {
        match self.dictionary.take().or_else(|| self.data.take()) {
            Some(page) => Ok(Some(page)),
            None => self.tail.get_next_page(),
        }
    }

    fn get_next_page_with_validity(&mut self) -> Result<Option<PageWithValidity>> {
        match self.dictionary.take().or_else(|| self.data.take()) {
            Some(page) => Ok(Some((page, None))),
            None => self.tail.get_next_page_with_validity(),
        }
    }

    fn peek_next_page(&mut self) -> Result<Option<PageMetadata>> {
        match self.dictionary.as_ref().or(self.data.as_ref()) {
            Some(page) => Ok(Some(PageMetadata {
                num_rows: None,
                num_levels: Some(page.num_values() as usize),
                is_dict: page.is_dictionary_page(),
            })),
            None => self.tail.peek_next_page(),
        }
    }

    fn skip_next_page(&mut self) -> Result<()> {
        if self
            .dictionary
            .take()
            .or_else(|| self.data.take())
            .is_none()
        {
            self.tail.skip_next_page()?;
        }
        Ok(())
    }

    fn take_deferred_dictionary(&mut self) -> Result<Option<Page>> {
        self.tail.take_deferred_dictionary()
    }
}
