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

//! Opt-in page reader for code-space consumers of flat primitive columns.

use std::marker::PhantomData;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, DictionaryArray, PrimitiveArray,
    types::{ArrowPrimitiveType, UInt32Type},
};
use arrow_buffer::{Buffer, ScalarBuffer};
use bytes::Bytes;

use super::super::parse_v1_level;
use super::{ColumnLevelDecoder, DefinitionLevelDecoder, DefinitionLevelDecoderImpl, get_decoder};
use crate::basic::Encoding;
use crate::column::page::{Page, PageReader};
use crate::data_type::DataType;
use crate::encodings::rle::codes::{CodeCursor, CodeRun};
use crate::errors::{ParquetError, Result};
use crate::schema::types::ColumnDescPtr;

/// Decoded values shared by the dictionary pages of a single column chunk.
#[derive(Clone, Debug)]
pub struct ChunkDictionary {
    values: ArrayRef,
    entries: usize,
}

impl ChunkDictionary {
    /// The chunk's single values allocation. Its pointer identifies this dictionary.
    pub fn values(&self) -> &ArrayRef {
        &self.values
    }

    /// Number of valid dictionary codes.
    pub fn entry_count(&self) -> usize {
        self.entries
    }
}

/// A page in either code space or value space.
#[derive(Debug)]
pub enum CodeColumnPage<A: ArrowPrimitiveType> {
    /// An RLE dictionary page with hybrid definition levels and indices.
    Dictionary(DictionaryCodePage),
    /// A page decoded into Arrow values, with the original encoding.
    Values {
        /// Original data-page encoding.
        encoding: Encoding,
        /// Decoded values, including nulls.
        values: PrimitiveArray<A>,
    },
}

/// Independent, restartable cursors over a dictionary-encoded page.
#[derive(Debug)]
pub struct DictionaryCodePage {
    dictionary: ChunkDictionary,
    rows: usize,
    values: usize,
    declared_values: bool,
    levels: Option<Bytes>,
    indices: Bytes,
}

impl DictionaryCodePage {
    /// Rows in this flat page, including nulls.
    pub fn num_rows(&self) -> usize {
        self.rows
    }

    /// The chunk dictionary, shared across dictionary-encoded data pages.
    pub fn dictionary(&self) -> &ChunkDictionary {
        &self.dictionary
    }

    /// Definition levels, or `None` for a required column.
    pub fn levels(&self) -> Option<CodeCursor<'_>> {
        self.levels.as_ref().map(|data| {
            CodeCursor::new(1, data, self.rows, 2).expect("flat definition level width")
        })
    }

    /// Verify that V2 definition levels agree with the header's non-null count.
    /// V1 definition levels were already checked to determine that count.
    pub fn validate_levels(&self) -> Result<()> {
        if self.declared_values {
            if let Some(mut levels) = self.levels() {
                let mut count = 0;
                while let Some(run) = levels.next_run()? {
                    count += run.count_ones(0, run.len());
                }
                if count != self.values {
                    return Err(general_err!(
                        "Definition level count disagrees with DataPage V2"
                    ));
                }
            }
        }
        Ok(())
    }

    /// The index stream, including only physical non-null values.
    pub fn indices(&self) -> Result<CodeCursor<'_>> {
        if self.values == 0 && self.indices.is_empty() {
            return CodeCursor::new(0, &[], 0, self.dictionary.entries as u64);
        }
        let (&width, bytes) = self
            .indices
            .split_first()
            .ok_or_else(|| general_err!("Missing dictionary index width"))?;
        CodeCursor::new(width, bytes, self.values, self.dictionary.entries as u64)
    }

    /// Decode a strictly increasing selection of page-relative row offsets.
    /// Null selected rows have null keys. The values Arc is shared across pages.
    pub fn selected_keys(
        &self,
        rows: impl IntoIterator<Item = usize>,
    ) -> Result<DictionaryArray<UInt32Type>> {
        let mut rows = rows.into_iter().peekable();
        let mut physical_offsets = Vec::new();
        let mut valid = Vec::new();
        let mut previous = None;
        let mut logical = 0;
        let mut physical = 0;
        let mut select_run = |run: CodeRun<'_>| -> Result<()> {
            let len = run.len();
            let mut start = 0;
            while let Some(row) = rows.next_if(|row| *row < logical + len) {
                if previous.is_some_and(|previous| row <= previous) {
                    return Err(general_err!("Row selections must be strictly increasing"));
                }
                previous = Some(row);
                let offset = row - logical;
                physical += run.count_ones(start, offset);
                let present = run.code_at(offset) == Some(1);
                if present {
                    physical_offsets.push(physical);
                    physical += 1;
                }
                valid.push(present);
                start = offset + 1;
            }
            physical += run.count_ones(start, len);
            logical += len;
            Ok(())
        };
        if let Some(mut levels) = self.levels() {
            while let Some(run) = levels.next_run()? {
                select_run(run)?;
            }
        } else {
            select_run(CodeRun::Repeated {
                code: 1,
                len: self.rows,
            })?;
        }
        if rows.next().is_some() || physical != self.values {
            return Err(general_err!(
                "Invalid row selection or definition level count"
            ));
        }
        let keys = self.indices()?.selected_keys(physical_offsets)?;
        let mut keys = keys.into_iter();
        let keys = PrimitiveArray::from_iter(
            valid
                .into_iter()
                .map(|valid| valid.then(|| keys.next().expect("one key per selected valid row"))),
        );
        DictionaryArray::try_new(keys, Arc::clone(&self.dictionary.values)).map_err(Into::into)
    }
}

/// Reads one flat primitive column chunk, exposing its code and value pages.
/// `P` is the Parquet physical type and `A` is its Arrow primitive representation.
pub struct CodeColumnReader<P, A> {
    reader: Box<dyn PageReader>,
    descr: ColumnDescPtr,
    dictionary: Option<ChunkDictionary>,
    types: PhantomData<(P, A)>,
}

impl<P, A> CodeColumnReader<P, A>
where
    P: DataType<T = A::Native>,
    A: ArrowPrimitiveType,
{
    /// Creates an opt-in reader for a required or optional flat primitive column.
    /// The Arrow type may reinterpret the native values (for example as dates).
    pub fn new(descr: ColumnDescPtr, reader: Box<dyn PageReader>) -> Result<Self> {
        if descr.max_rep_level() != 0
            || descr.max_def_level() > 1
            || descr.physical_type() != P::get_physical_type()
        {
            return Err(general_err!(
                "Code reading requires a flat primitive column of the requested physical type"
            ));
        }
        Ok(Self {
            reader,
            descr,
            dictionary: None,
            types: PhantomData,
        })
    }

    /// Returns the shared chunk dictionary once its dictionary page is read.
    pub fn dictionary(&self) -> Option<&ChunkDictionary> {
        self.dictionary.as_ref()
    }

    fn install_dictionary(&mut self, page: Page) -> Result<()> {
        let Page::DictionaryPage {
            buf,
            num_values,
            encoding,
            ..
        } = page
        else {
            return Err(general_err!("Expected dictionary page"));
        };
        if self.dictionary.is_some()
            || !matches!(encoding, Encoding::PLAIN | Encoding::PLAIN_DICTIONARY)
        {
            return Err(general_err!("Unexpected dictionary page or encoding"));
        }
        let entries = num_values as usize;
        if entries > buf.len() / size_of::<A::Native>() {
            return Err(general_err!("Incomplete primitive dictionary"));
        }
        let values = if cfg!(target_endian = "little")
            && buf.as_ptr().align_offset(align_of::<A::Native>()) == 0
        {
            PrimitiveArray::<A>::new(ScalarBuffer::new(Buffer::from(buf), 0, entries), None)
        } else {
            self.decode_values(Encoding::PLAIN, buf, entries, None)?
        };
        let values = Arc::new(values);
        self.dictionary = Some(ChunkDictionary { values, entries });
        Ok(())
    }

    /// Reads the next data page, loading the chunk dictionary at most once.
    pub fn next_page(&mut self) -> Result<Option<CodeColumnPage<A>>> {
        let nullable = self.descr.max_def_level() != 0;
        let (data, rows, encoding, levels, level_encoding, non_null) = loop {
            match self.reader.get_next_page()? {
                None => return Ok(None),
                Some(page @ Page::DictionaryPage { .. }) => self.install_dictionary(page)?,
                Some(Page::DataPage {
                    buf,
                    num_values,
                    encoding,
                    def_level_encoding,
                    ..
                }) => {
                    let (offset, levels) = if nullable {
                        let (offset, levels) =
                            parse_v1_level(1, num_values, def_level_encoding, buf.clone())?;
                        (offset, Some(levels))
                    } else {
                        (0, None)
                    };
                    break (
                        buf.slice(offset..),
                        num_values as usize,
                        encoding,
                        levels,
                        def_level_encoding,
                        None,
                    );
                }
                Some(Page::DataPageV2 {
                    buf,
                    num_values,
                    encoding,
                    num_nulls,
                    def_levels_byte_len,
                    rep_levels_byte_len,
                    ..
                }) => {
                    let len = def_levels_byte_len as usize;
                    if rep_levels_byte_len != 0
                        || len > buf.len()
                        || num_nulls > num_values
                        || (!nullable && (len != 0 || num_nulls != 0))
                    {
                        return Err(general_err!("Invalid flat DataPage V2 levels"));
                    }
                    break (
                        buf.slice(len..),
                        num_values as usize,
                        encoding,
                        nullable.then(|| buf.slice(..len)),
                        Encoding::RLE,
                        Some((num_values - num_nulls) as usize),
                    );
                }
            }
        };
        if matches!(
            encoding,
            Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
        ) {
            if self.dictionary.is_none() {
                if let Some(page) = self.reader.take_deferred_dictionary()? {
                    self.install_dictionary(page)?;
                }
            }
            let dictionary = self
                .dictionary
                .clone()
                .ok_or_else(|| general_err!("Missing column chunk dictionary"))?;
            if level_encoding == Encoding::RLE || levels.is_none() {
                let values = match (non_null, &levels) {
                    (Some(values), _) => values,
                    (None, None) => rows,
                    (None, Some(levels)) => {
                        let mut levels = CodeCursor::new(1, levels, rows, 2)?;
                        let mut values = 0;
                        while let Some(run) = levels.next_run()? {
                            values += run.count_ones(0, run.len());
                        }
                        values
                    }
                };
                let page = DictionaryCodePage {
                    dictionary,
                    rows,
                    values,
                    declared_values: non_null.is_some(),
                    levels,
                    indices: data,
                };
                page.indices()?;
                return Ok(Some(CodeColumnPage::Dictionary(page)));
            }
            // The legacy BIT_PACKED level encoding falls back to value space.
            let (levels, count) = decode_levels(levels, level_encoding, rows)?;
            let page = DictionaryCodePage {
                dictionary,
                rows,
                values: count,
                declared_values: false,
                levels: None,
                indices: data,
            };
            let keys = page.indices()?.selected_keys(0..count)?;
            let dictionary = page
                .dictionary
                .values
                .as_any()
                .downcast_ref::<PrimitiveArray<A>>()
                .unwrap();
            let values = keys.into_iter().map(|key| dictionary.value(key as usize));
            return Ok(Some(CodeColumnPage::Values {
                encoding,
                values: with_levels(values, levels),
            }));
        }
        let (levels, count) = decode_levels(levels, level_encoding, rows)?;
        if non_null.is_some_and(|values| values != count) {
            return Err(general_err!(
                "Definition level count disagrees with DataPage V2"
            ));
        }
        let values = self.decode_values(encoding, data, count, levels)?;
        Ok(Some(CodeColumnPage::Values { encoding, values }))
    }

    fn decode_values(
        &self,
        encoding: Encoding,
        data: Bytes,
        count: usize,
        levels: Option<Vec<i16>>,
    ) -> Result<PrimitiveArray<A>> {
        let mut decoder = get_decoder::<P>(self.descr.clone(), encoding)?;
        decoder.set_data(data, count)?;
        let mut values = Vec::new();
        let mut buffer = [A::Native::default(); 1024];
        while values.len() < count {
            let n = (count - values.len()).min(buffer.len());
            if decoder.get(&mut buffer[..n])? != n {
                return Err(general_err!("Incomplete value page"));
            }
            values.extend_from_slice(&buffer[..n]);
        }
        Ok(with_levels(values.into_iter(), levels))
    }
}

fn decode_levels(
    levels: Option<Bytes>,
    encoding: Encoding,
    rows: usize,
) -> Result<(Option<Vec<i16>>, usize)> {
    let Some(levels) = levels else {
        return Ok((None, rows));
    };
    let mut decoder = DefinitionLevelDecoderImpl::new(1);
    decoder.set_data(encoding, levels)?;
    let mut decoded = Vec::new();
    let mut values = 0;
    while decoded.len() < rows {
        let batch = (rows - decoded.len()).min(1024);
        let (num_values, levels_read) = decoder.read_def_levels(&mut decoded, batch)?;
        if levels_read != batch {
            return Err(general_err!("Incomplete definition levels"));
        }
        values += num_values;
    }
    Ok((Some(decoded), values))
}

fn with_levels<A: ArrowPrimitiveType>(
    mut values: impl ExactSizeIterator<Item = A::Native>,
    levels: Option<Vec<i16>>,
) -> PrimitiveArray<A> {
    match levels {
        Some(levels) => PrimitiveArray::from_iter(levels.into_iter().map(|level| {
            (level == 1).then(|| values.next().expect("one value per set definition level"))
        })),
        None => PrimitiveArray::from_iter_values(values),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::reader::ColumnReaderImpl;
    use crate::data_type::Int32Type;
    use crate::encodings::rle::RleEncoder;
    use crate::schema::{parser::parse_message_type, types::SchemaDescriptor};
    use crate::util::test_common::page_util::InMemoryPageReader;
    use arrow_array::{Int32Array, types::Int32Type as ArrowInt32};

    fn data_page(
        v2: bool,
        definition: &[bool],
        encoding: Encoding,
        data: impl IntoIterator<Item = u8>,
    ) -> Page {
        let mut levels = RleEncoder::new(1, 0);
        for &valid in definition {
            levels.put(u64::from(valid));
        }
        let levels = levels.consume();
        let rows = definition.len() as u32;
        let mut buf = Vec::new();
        if !v2 {
            buf.extend_from_slice(&(levels.len() as u32).to_le_bytes());
        }
        buf.extend_from_slice(&levels);
        buf.extend(data);
        if v2 {
            Page::DataPageV2 {
                buf: buf.into(),
                num_values: rows,
                encoding,
                num_nulls: definition.iter().filter(|&&valid| !valid).count() as u32,
                num_rows: rows,
                def_levels_byte_len: levels.len() as u32,
                rep_levels_byte_len: 0,
                is_compressed: false,
                statistics: None,
            }
        } else {
            Page::DataPage {
                buf: buf.into(),
                num_values: rows,
                encoding,
                def_level_encoding: Encoding::RLE,
                rep_level_encoding: Encoding::RLE,
                statistics: None,
            }
        }
    }

    fn dictionary(values: &[i32]) -> Page {
        Page::DictionaryPage {
            buf: values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
                .into(),
            num_values: values.len() as u32,
            encoding: Encoding::PLAIN,
            is_sorted: false,
        }
    }

    fn descriptor() -> ColumnDescPtr {
        SchemaDescriptor::new(Arc::new(
            parse_message_type("message x { OPTIONAL INT32 value; }").unwrap(),
        ))
        .column(0)
    }

    #[test]
    fn code_page_matches_reader_across_page_boundaries() {
        crate::util::native::dispatch::for_each_tier(|| {
            for v2 in [false, true] {
                for width in 1..=32 {
                    let definition: Vec<_> = (0..183).map(|i| i % 27 < 15 && i % 4 != 3).collect();
                    let mut indices = RleEncoder::new(width, 0);
                    for i in 0..definition.iter().filter(|&&valid| valid).count() {
                        indices.put(u64::from(i % 21 < 13));
                    }
                    let indices: Vec<_> = [width].into_iter().chain(indices.consume()).collect();
                    let code = data_page(v2, &definition, Encoding::RLE_DICTIONARY, indices);
                    let plain = data_page(
                        v2,
                        &[true, false, true],
                        Encoding::PLAIN,
                        [89i32, 97].into_iter().flat_map(i32::to_le_bytes),
                    );
                    let pages = vec![dictionary(&[17, 43]), code.clone(), plain, code];
                    let desc = descriptor();
                    let mut oracle = ColumnReaderImpl::<Int32Type>::new(
                        desc.clone(),
                        Box::new(InMemoryPageReader::new(pages.clone())),
                    );
                    let mut values = Vec::new();
                    let mut levels = Vec::new();
                    let total = definition.len() * 2 + 3;
                    assert_eq!(
                        oracle
                            .read_records(total, Some(&mut levels), None, &mut values)
                            .unwrap()
                            .0,
                        total
                    );
                    let mut values = values.into_iter();
                    let expected: Vec<_> = levels
                        .into_iter()
                        .map(|level| (level == 1).then(|| values.next().unwrap()))
                        .collect();
                    let mut reader = CodeColumnReader::<Int32Type, ArrowInt32>::new(
                        desc,
                        Box::new(InMemoryPageReader::new(pages)),
                    )
                    .unwrap();
                    let mut row = 0;
                    let mut cached_dictionary = None;
                    while let Some(page) = reader.next_page().unwrap() {
                        match page {
                            CodeColumnPage::Dictionary(page) => {
                                let rows = (0..page.num_rows()).filter(|i| i % 7 < 4);
                                let selected = page.selected_keys(rows.clone()).unwrap();
                                let values = selected
                                    .values()
                                    .as_any()
                                    .downcast_ref::<Int32Array>()
                                    .unwrap();
                                let decoded: Vec<_> = selected
                                    .keys()
                                    .iter()
                                    .map(|code| code.map(|code| values.value(code as usize)))
                                    .collect();
                                assert_eq!(
                                    decoded,
                                    rows.map(|i| expected[row + i]).collect::<Vec<_>>()
                                );
                                if let Some(dictionary) = &cached_dictionary {
                                    assert!(Arc::ptr_eq(dictionary, selected.values()));
                                } else {
                                    cached_dictionary = Some(Arc::clone(selected.values()));
                                }
                                assert_eq!(page.dictionary().entry_count(), 2);
                                assert!(page.selected_keys([10, 2]).is_err());
                                assert!(page.selected_keys([page.num_rows()]).is_err());
                                row += page.num_rows();
                            }
                            CodeColumnPage::Values { encoding, values } => {
                                assert_eq!(encoding, Encoding::PLAIN);
                                assert_eq!(
                                    values.iter().collect::<Vec<_>>(),
                                    expected[row..row + 3]
                                );
                                row += 3;
                            }
                        }
                    }
                    assert_eq!(row, total);
                }
            }
        });
    }

    #[test]
    fn code_page_empty_dictionary_and_all_null_chunk() {
        for v2 in [false, true] {
            let plain = data_page(v2, &[false; 14], Encoding::PLAIN, []);
            let encoded = data_page(v2, &[false; 14], Encoding::RLE_DICTIONARY, []);
            let mut reader = CodeColumnReader::<Int32Type, ArrowInt32>::new(
                descriptor(),
                Box::new(InMemoryPageReader::new([plain])),
            )
            .unwrap();
            let Some(CodeColumnPage::Values { values, .. }) = reader.next_page().unwrap() else {
                panic!("expected PLAIN page")
            };
            assert_eq!(values.null_count(), 14);
            assert!(reader.dictionary().is_none());

            let invalid = data_page(v2, &[true], Encoding::RLE_DICTIONARY, [1, 2, 0]);
            let mut reader = CodeColumnReader::<Int32Type, ArrowInt32>::new(
                descriptor(),
                Box::new(InMemoryPageReader::new([dictionary(&[]), encoded, invalid])),
            )
            .unwrap();
            let Some(CodeColumnPage::Dictionary(page)) = reader.next_page().unwrap() else {
                panic!("expected dictionary page")
            };
            assert_eq!(page.dictionary().entry_count(), 0);
            let selected = page.selected_keys([0, 7, 13]).unwrap();
            assert_eq!(selected.keys().null_count(), 3);
            let Some(CodeColumnPage::Dictionary(page)) = reader.next_page().unwrap() else {
                panic!("expected dictionary page")
            };
            assert!(page.selected_keys([]).is_err());
        }
    }
}
