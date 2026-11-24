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

use crate::arrow::array_reader::{
    ArrayReader, check_selection, filter_selection, read_records, selected_rows, skip_records,
};
use crate::arrow::buffer::view_buffer::ViewBuffer;
use crate::arrow::decoder::{DeltaByteArrayDecoder, DictIndexDecoder};
use crate::arrow::record_reader::GenericRecordReader;
use crate::arrow::schema::parquet_to_arrow_field;
use crate::basic::{ConvertedType, Encoding};
use crate::column::page::{Page, PageIterator, PageReader};
use crate::column::reader::decoder::ColumnValueDecoder;
use crate::column::reader::decoder::code_page::FlatDataPage;
use crate::column::reader::{FlatPageReplay, FlatValidityRanks};
use crate::encodings::decoding::DeltaLengths;
use crate::encodings::rle::codes::{CodeCursor, CodePosition};
use crate::errors::{ParquetError, Result};
use crate::schema::types::ColumnDescPtr;
use crate::util::utf8::check_valid_utf8;
use arrow_array::{ArrayRef, BinaryViewArray, StringViewArray, UInt32Array, builder::make_view};
use arrow_buffer::{BooleanBufferBuilder, Buffer, NullBuffer, ScalarBuffer};
use arrow_data::ByteView;
use arrow_schema::DataType as ArrowType;
use bytes::Bytes;
use std::any::Any;
use std::sync::Arc;

/// Returns an [`ArrayReader`] that decodes the provided byte array column to view types.
///
/// `batch_size` is a hint used to pre-allocate internal buffers and avoid
/// reallocations when reading the first batch. It is not a hard capacity
/// contract: with selective null padding (`padding_threshold`), value buffers
/// are sized from the number of values actually decoded.
pub fn make_byte_view_array_reader(
    pages: Box<dyn PageIterator>,
    column_desc: ColumnDescPtr,
    arrow_type: Option<ArrowType>,
    batch_size: usize,
    padding_threshold: Option<i16>,
) -> Result<Box<dyn ArrayReader>> {
    // Check if Arrow type is specified, else create it from Parquet type
    let data_type = match arrow_type {
        Some(t) => t,
        None => match parquet_to_arrow_field(column_desc.as_ref())?.data_type() {
            ArrowType::Utf8 | ArrowType::Utf8View => ArrowType::Utf8View,
            _ => ArrowType::BinaryView,
        },
    };

    match data_type {
        ArrowType::BinaryView | ArrowType::Utf8View => {
            let flat = column_desc.max_rep_level() == 0
                && (column_desc.max_def_level() == 0
                    || (column_desc.max_def_level() == 1 && column_desc.self_type().is_optional()))
                && padding_threshold.is_none();
            let nullable = column_desc.max_def_level() != 0;
            let view_type = SelectedViewType {
                utf8: data_type == ArrowType::Utf8View,
                validate_utf8: data_type == ArrowType::Utf8View
                    || column_desc.converted_type() == ConvertedType::UTF8,
            };
            let mut reader = GenericRecordReader::new(column_desc, batch_size);
            if let Some(threshold) = padding_threshold {
                reader.set_padding_threshold(threshold);
            }
            Ok(Box::new(ByteViewArrayReader {
                data_type,
                pages,
                def_levels_buffer: None,
                rep_levels_buffer: None,
                record_reader: reader,
                flat,
                nullable,
                view_type,
                selected: None,
                full_started: false,
            }))
        }

        _ => Err(general_err!(
            "invalid data type for byte array reader read to view type - {}",
            data_type
        )),
    }
}

/// An [`ArrayReader`] for variable length byte arrays
struct ByteViewArrayReader {
    data_type: ArrowType,
    pages: Box<dyn PageIterator>,
    def_levels_buffer: Option<Vec<i16>>,
    rep_levels_buffer: Option<Vec<i16>>,
    record_reader: GenericRecordReader<ViewBuffer, ByteViewArrayColumnValueDecoder>,
    flat: bool,
    nullable: bool,
    view_type: SelectedViewType,
    selected: Option<Box<SelectedViews>>,
    full_started: bool,
}

impl ByteViewArrayReader {
    fn resume_full(&mut self) -> Result<()> {
        if let Some(mut selected) = self.selected.take()
            && let Some(tail) = selected.chunk.take()
        {
            let replay = FlatPageReplay {
                dictionary: selected.dictionary.map(|(page, _)| page),
                data: selected.page.map(|page| page.original),
                tail,
            };
            self.record_reader.set_page_reader(Box::new(replay))?;
            if self.record_reader.skip_records(selected.offset)? != selected.offset {
                return Err(eof_err!("Incomplete page when resuming view decode"));
            }
        }
        self.full_started = true;
        Ok(())
    }
}

impl ArrayReader for ByteViewArrayReader {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_data_type(&self) -> &ArrowType {
        &self.data_type
    }

    fn read_records(&mut self, batch_size: usize) -> Result<usize> {
        if batch_size != 0 && !self.full_started {
            self.resume_full()?;
        }
        read_records(&mut self.record_reader, self.pages.as_mut(), batch_size)
    }

    fn read_selected(&mut self, words: &[u64], rows: usize) -> Result<ArrayRef> {
        check_selection(words, rows)?;
        if self.record_reader.num_values() != 0 {
            return Err(general_err!(
                "Consume the previous batch before reading a selected vector"
            ));
        }
        if rows == 0 {
            return Ok(arrow_array::new_empty_array(&self.data_type));
        }
        if !self.flat {
            self.read_records(rows)?;
            return filter_selection(self.consume_batch()?, words, rows);
        }

        let mut arrays = Vec::new();
        let mut start = 0;
        if self.full_started {
            start = self.record_reader.read_records(rows)?;
            let array = self.consume_batch()?;
            if start == rows {
                return filter_selection(array, words, rows);
            }
            if start != 0 {
                arrays.push(filter_selection(array, words, start)?);
            }
            self.full_started = false;
        }

        let selected = self
            .selected
            .get_or_insert_with(|| Box::new(SelectedViews::default()));
        selected.read(
            self.pages.as_mut(),
            self.nullable,
            self.view_type,
            selected_rows(words, start, rows),
            start..rows,
            &mut arrays,
        )?;
        self.def_levels_buffer = None;
        self.rep_levels_buffer = None;
        match arrays.len() {
            0 => Ok(arrow_array::new_empty_array(&self.data_type)),
            1 => Ok(arrays.pop().unwrap()),
            _ => Ok(arrow_select::concat::concat(
                &arrays
                    .iter()
                    .map(|array| array.as_ref())
                    .collect::<Vec<_>>(),
            )?),
        }
    }

    fn consume_batch(&mut self) -> Result<ArrayRef> {
        let buffer = self.record_reader.consume_record_data();
        let null_buffer = self.record_reader.consume_compact_bitmap();
        self.def_levels_buffer = self.record_reader.consume_def_levels();
        self.rep_levels_buffer = self.record_reader.consume_rep_levels();
        self.record_reader.reset();

        let array = buffer.into_array(null_buffer, &self.data_type);

        Ok(array)
    }

    fn set_remaining_rows(&mut self, rows: usize) {
        self.record_reader.set_remaining_rows(rows);
    }

    fn skip_records(&mut self, num_records: usize) -> Result<usize> {
        if num_records != 0 && !self.full_started {
            self.resume_full()?;
        }
        skip_records(&mut self.record_reader, self.pages.as_mut(), num_records)
    }

    fn get_def_levels(&self) -> Option<&[i16]> {
        self.def_levels_buffer.as_deref()
    }

    fn get_rep_levels(&self) -> Option<&[i16]> {
        self.rep_levels_buffer.as_deref()
    }

    fn max_def_level(&self) -> i16 {
        self.record_reader.max_def_level()
    }
}

#[derive(Default)]
struct SelectedViews {
    chunk: Option<Box<dyn PageReader>>,
    dictionary: Option<(Page, ViewBuffer)>,
    page: Option<ViewPage>,
    // This can be nonzero even before a page has been loaded. A rejected
    // prefix in one vector need not fetch the page for the next vector.
    offset: usize,
    validity: FlatValidityRanks,
    ranks: Vec<u32>,
    codes: Vec<u32>,
}

struct ViewPage {
    original: Page,
    rows: usize,
    physical_values: usize,
    values: ViewPageValues,
}

enum ViewPageValues {
    Dictionary {
        data: Bytes,
        position: Option<CodePosition>,
    },
    Full(ArrayRef),
}

#[derive(Clone, Copy)]
struct SelectedViewType {
    utf8: bool,
    validate_utf8: bool,
}

impl SelectedViews {
    fn read(
        &mut self,
        pages: &mut dyn PageIterator,
        nullable: bool,
        view_type: SelectedViewType,
        rows: impl Iterator<Item = usize>,
        range: std::ops::Range<usize>,
        arrays: &mut Vec<ArrayRef>,
    ) -> Result<()> {
        let mut rows = rows.peekable();
        let mut start = range.start;
        while start < range.end {
            if self.page.is_none() {
                let gap = rows.peek().copied().unwrap_or(range.end) - start;
                start += self.advance(pages, nullable, view_type, gap, range.end - start)?;
                if self.page.is_none() {
                    continue;
                }
            }
            let page = self.page.as_mut().unwrap();
            let end = start + (range.end - start).min(page.rows - self.offset);
            let mut selected = std::iter::from_fn(|| rows.next_if(|&row| row < end))
                .map(|row| (row - start + self.offset) as u32)
                .peekable();
            if selected.peek().is_some() {
                let array = match &mut page.values {
                    ViewPageValues::Dictionary { data, position } => {
                        self.ranks.clear();
                        let mut nulls = nullable.then(|| BooleanBufferBuilder::new(0));
                        for row in selected {
                            if let Some(nulls) = &mut nulls {
                                let rank = self.validity.rank(row);
                                nulls.append(rank.is_some());
                                self.ranks.extend(rank);
                            } else {
                                self.ranks.push(row);
                            }
                        }
                        let physical_end = if self.offset + end - start == page.rows {
                            page.physical_values
                        } else {
                            self.ranks.last().map_or(0, |&rank| rank as usize + 1)
                        };
                        CodeCursor::read_selected(
                            data,
                            page.physical_values,
                            physical_end,
                            &self.ranks,
                            &mut self.codes,
                            position,
                        )?;
                        let (_, dictionary) = self.dictionary.as_ref().unwrap();
                        gather_views(dictionary, &self.codes, nulls, view_type)?
                    }
                    ViewPageValues::Full(values) => {
                        let indices = UInt32Array::from_iter_values(selected);
                        arrow_select::take::take(values.as_ref(), &indices, None)?
                    }
                };
                arrays.push(array);
            }
            self.offset += end - start;
            start = end;
            if self.offset == page.rows {
                self.page = None;
                self.offset = 0;
            }
        }
        Ok(())
    }

    // Skips a rejected prefix or loads the page containing the next selection.
    fn advance(
        &mut self,
        pages: &mut dyn PageIterator,
        nullable: bool,
        view_type: SelectedViewType,
        gap: usize,
        remaining: usize,
    ) -> Result<usize> {
        loop {
            if self.chunk.is_none() {
                self.dictionary = None;
                self.chunk = Some(
                    pages
                        .next()
                        .ok_or_else(|| eof_err!("not enough rows for selected vector"))??,
                );
            }
            let chunk = self.chunk.as_mut().unwrap();
            let metadata = chunk.peek_next_page()?;
            if metadata.as_ref().is_some_and(|meta| meta.is_dict)
                && chunk.supports_deferred_dictionary()
            {
                chunk.skip_next_page()?;
                continue;
            }
            let span = metadata
                .filter(|meta| !meta.is_dict)
                .and_then(|meta| meta.num_rows.or(meta.num_levels));
            if let Some(rows) = span {
                let unread = rows
                    .checked_sub(self.offset)
                    .ok_or_else(|| general_err!("Invalid buffered view page span"))?;
                if unread <= gap {
                    chunk.skip_unselected_data_page()?;
                    self.offset = 0;
                    if unread == 0 {
                        continue;
                    }
                    return Ok(unread);
                }
                if gap == remaining {
                    self.offset += gap;
                    return Ok(gap);
                }
            }
            match chunk.get_next_page()? {
                None if self.offset != 0 => return Err(eof_err!("Missing buffered view page")),
                None => self.chunk = None,
                Some(page) if page.is_dictionary_page() => self.install_dictionary(page)?,
                Some(page) => {
                    self.load(page, span, nullable, view_type)?;
                    return Ok(0);
                }
            }
        }
    }

    fn install_dictionary(&mut self, page: Page) -> Result<()> {
        if self.dictionary.is_some() {
            return Err(general_err!("Unexpected dictionary page"));
        }
        // Build binary views now and validate UTF-8 only for retained values.
        let mut decoder = ByteViewArrayColumnValueDecoder {
            dict: None,
            decoder: None,
            validate_utf8: false,
        };
        decoder.set_dict(
            page.buffer().clone(),
            page.num_values(),
            page.encoding(),
            false,
        )?;
        self.dictionary = Some((page, decoder.dict.unwrap()));
        Ok(())
    }

    fn load(
        &mut self,
        original: Page,
        span: Option<usize>,
        nullable: bool,
        view_type: SelectedViewType,
    ) -> Result<()> {
        let page = FlatDataPage::new(&original, nullable)?;
        let rows = page.rows;
        if self.offset > rows || span.is_some_and(|span| rows != span) {
            return Err(general_err!("View page count disagrees with its row span"));
        }
        let physical_values = if nullable {
            self.validity.load(&page)?
        } else {
            rows
        };
        let dictionary = matches!(
            page.encoding,
            Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
        );
        if dictionary && self.dictionary.is_none() {
            let dict = self
                .chunk
                .as_mut()
                .unwrap()
                .take_deferred_dictionary()?
                .ok_or_else(|| general_err!("Missing column chunk dictionary"))?;
            self.install_dictionary(dict)?;
        }
        let values = if dictionary && page.has_hybrid_levels() {
            ViewPageValues::Dictionary {
                data: page.data,
                position: None,
            }
        } else {
            let mut decoder = ByteViewArrayDecoder::new(
                page.encoding,
                page.data,
                rows,
                Some(physical_values),
                view_type.validate_utf8,
            )?;
            let mut dense = ViewBuffer::default();
            let dict = self.dictionary.as_ref().map(|(_, dict)| dict);
            if decoder.read(&mut dense, physical_values, dict)? != physical_values {
                return Err(eof_err!("Incomplete view value page"));
            }
            let nulls = nullable.then(|| {
                let mut valid = BooleanBufferBuilder::new(rows);
                let mut dense_views = std::mem::take(&mut dense.views).into_iter();
                dense.views = (0..rows as u32)
                    .map(|row| {
                        let present = self.validity.rank(row).is_some();
                        valid.append(present);
                        if present {
                            dense_views.next().unwrap()
                        } else {
                            0
                        }
                    })
                    .collect();
                NullBuffer::new(valid.finish())
            });
            ViewPageValues::Full(finish_views(dense, nulls, view_type)?)
        };
        self.page = Some(ViewPage {
            original,
            rows,
            physical_values,
            values,
        });
        Ok(())
    }
}

fn gather_views(
    dictionary: &ViewBuffer,
    codes: &[u32],
    mut validity: Option<BooleanBufferBuilder>,
    view_type: SelectedViewType,
) -> Result<ArrayRef> {
    let validity = validity
        .as_mut()
        .map(|builder| NullBuffer::new(builder.finish()));
    let len = validity.as_ref().map_or(codes.len(), NullBuffer::len);
    let mut out = ViewBuffer::with_capacity(len);
    out.buffers.clone_from(&dictionary.buffers);
    let mut codes = codes.iter();
    for row in 0..len {
        let view = if validity.as_ref().is_some_and(|valid| valid.is_null(row)) {
            0
        } else {
            let key = *codes.next().unwrap();
            *dictionary
                .views
                .get(key as usize)
                .ok_or_else(|| general_err!("invalid key={key} for dictionary"))?
        };
        out.views.push(view);
    }
    finish_views(out, validity, view_type)
}

fn finish_views(
    values: ViewBuffer,
    nulls: Option<NullBuffer>,
    view_type: SelectedViewType,
) -> Result<ArrayRef> {
    let views = ScalarBuffer::from(values.views);
    let nulls = nulls.filter(|nulls| nulls.null_count() != 0);
    if view_type.validate_utf8 {
        let strings = StringViewArray::try_new(views, values.buffers, nulls)?;
        if view_type.utf8 {
            Ok(Arc::new(strings))
        } else {
            Ok(Arc::new(strings.to_binary_view()))
        }
    } else {
        Ok(Arc::new(BinaryViewArray::try_new(
            views,
            values.buffers,
            nulls,
        )?))
    }
}

/// A [`ColumnValueDecoder`] for variable length byte arrays
struct ByteViewArrayColumnValueDecoder {
    dict: Option<ViewBuffer>,
    decoder: Option<ByteViewArrayDecoder>,
    validate_utf8: bool,
}

impl ColumnValueDecoder for ByteViewArrayColumnValueDecoder {
    type Buffer = ViewBuffer;

    fn new(desc: &ColumnDescPtr) -> Self {
        let validate_utf8 = desc.converted_type() == ConvertedType::UTF8;
        Self {
            dict: None,
            decoder: None,
            validate_utf8,
        }
    }

    fn set_dict(
        &mut self,
        buf: Bytes,
        num_values: u32,
        encoding: Encoding,
        _is_sorted: bool,
    ) -> Result<()> {
        if !matches!(
            encoding,
            Encoding::PLAIN | Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
        ) {
            return Err(nyi_err!(
                "Invalid/Unsupported encoding type for dictionary: {}",
                encoding
            ));
        }

        let num_values = num_values as usize;
        // The count is from the page header and sizes the dictionary. Each
        // value has a four-byte length, so the page holds at most a quarter
        // of its bytes in values.
        if num_values > buf.len() / 4 {
            return Err(general_err!(
                "dictionary page declares {} values, but its {} bytes hold at most {}",
                num_values,
                buf.len(),
                buf.len() / 4
            ));
        }
        let mut buffer = ViewBuffer::with_capacity(num_values);
        let mut decoder =
            ByteViewArrayDecoderPlain::new(buf, num_values, Some(num_values), self.validate_utf8);
        decoder.read(&mut buffer, usize::MAX)?;
        self.dict = Some(buffer);
        Ok(())
    }

    fn set_data(
        &mut self,
        encoding: Encoding,
        data: Bytes,
        num_levels: usize,
        num_values: Option<usize>,
    ) -> Result<()> {
        self.decoder = Some(ByteViewArrayDecoder::new(
            encoding,
            data,
            num_levels,
            num_values,
            self.validate_utf8,
        )?);
        Ok(())
    }

    fn read(&mut self, out: &mut Self::Buffer, num_values: usize) -> Result<usize> {
        let decoder = self
            .decoder
            .as_mut()
            .ok_or_else(|| general_err!("no decoder set"))?;

        decoder.read(out, num_values, self.dict.as_ref())
    }

    fn skip_values(&mut self, num_values: usize) -> Result<usize> {
        let decoder = self
            .decoder
            .as_mut()
            .ok_or_else(|| general_err!("no decoder set"))?;

        decoder.skip(num_values, self.dict.as_ref())
    }
}

/// A generic decoder from uncompressed parquet value data to [`ViewBuffer`]
pub enum ByteViewArrayDecoder {
    Plain(ByteViewArrayDecoderPlain),
    Dictionary(ByteViewArrayDecoderDictionary),
    DeltaLength(ByteViewArrayDecoderDeltaLength),
    DeltaByteArray(ByteViewArrayDecoderDelta),
}

impl ByteViewArrayDecoder {
    pub fn new(
        encoding: Encoding,
        data: Bytes,
        num_levels: usize,
        num_values: Option<usize>,
        validate_utf8: bool,
    ) -> Result<Self> {
        let decoder = match encoding {
            Encoding::PLAIN => ByteViewArrayDecoder::Plain(ByteViewArrayDecoderPlain::new(
                data,
                num_levels,
                num_values,
                validate_utf8,
            )),
            Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY => {
                ByteViewArrayDecoder::Dictionary(ByteViewArrayDecoderDictionary::new(
                    data, num_levels, num_values,
                )?)
            }
            Encoding::DELTA_LENGTH_BYTE_ARRAY => {
                ByteViewArrayDecoder::DeltaLength(ByteViewArrayDecoderDeltaLength::new(
                    data,
                    num_values.unwrap_or(num_levels),
                    validate_utf8,
                )?)
            }
            Encoding::DELTA_BYTE_ARRAY => {
                ByteViewArrayDecoder::DeltaByteArray(ByteViewArrayDecoderDelta::new(
                    data,
                    num_values.unwrap_or(num_levels),
                    validate_utf8,
                )?)
            }
            _ => {
                return Err(general_err!(
                    "unsupported encoding for byte array: {}",
                    encoding
                ));
            }
        };

        Ok(decoder)
    }

    /// Read up to `len` values to `out` with the optional dictionary
    pub fn read(
        &mut self,
        out: &mut ViewBuffer,
        len: usize,
        dict: Option<&ViewBuffer>,
    ) -> Result<usize> {
        match self {
            ByteViewArrayDecoder::Plain(d) => d.read(out, len),
            ByteViewArrayDecoder::Dictionary(d) => {
                let dict = dict
                    .ok_or_else(|| general_err!("dictionary required for dictionary encoding"))?;
                d.read(out, dict, len)
            }
            ByteViewArrayDecoder::DeltaLength(d) => d.read(out, len),
            ByteViewArrayDecoder::DeltaByteArray(d) => d.read(out, len),
        }
    }

    /// Skip `len` values
    pub fn skip(&mut self, len: usize, dict: Option<&ViewBuffer>) -> Result<usize> {
        match self {
            ByteViewArrayDecoder::Plain(d) => d.skip(len),
            ByteViewArrayDecoder::Dictionary(d) => {
                let dict = dict
                    .ok_or_else(|| general_err!("dictionary required for dictionary encoding"))?;
                d.skip(dict, len)
            }
            ByteViewArrayDecoder::DeltaLength(d) => d.skip(len),
            ByteViewArrayDecoder::DeltaByteArray(d) => d.skip(len),
        }
    }
}

/// Decoder from [`Encoding::PLAIN`] data to [`ViewBuffer`]
pub struct ByteViewArrayDecoderPlain {
    buf: Buffer,
    offset: usize,

    validate_utf8: bool,

    /// This is a maximum as the null count is not always known, e.g. value data from
    /// a v1 data page
    max_remaining_values: usize,
}

impl ByteViewArrayDecoderPlain {
    pub fn new(
        buf: Bytes,
        num_levels: usize,
        num_values: Option<usize>,
        validate_utf8: bool,
    ) -> Self {
        Self {
            buf: Buffer::from(buf),
            offset: 0,
            max_remaining_values: num_values.unwrap_or(num_levels),
            validate_utf8,
        }
    }

    pub fn read(&mut self, output: &mut ViewBuffer, len: usize) -> Result<usize> {
        if self.validate_utf8 {
            self.read_impl::<true>(output, len)
        } else {
            self.read_impl::<false>(output, len)
        }
    }

    fn read_impl<const VALIDATE_UTF8: bool>(
        &mut self,
        output: &mut ViewBuffer,
        len: usize,
    ) -> Result<usize> {
        // avoid creating a new buffer if the last buffer is the same as the current buffer
        // This is especially useful when row-level filtering is applied, where we call lots of small `read` over the same buffer.
        let block_id = {
            if output.buffers.last().is_some_and(|x| x.ptr_eq(&self.buf)) {
                output.buffers.len() as u32 - 1
            } else {
                output.append_block(self.buf.clone())
            }
        };

        let to_read = len.min(self.max_remaining_values);

        let buf: &[u8] = self.buf.as_ref();
        let buf_len = buf.len();
        let mut end_offset = self.offset;
        let mut utf8_validation_begin = end_offset;

        // Each value has a four-byte length. If the remaining bytes cannot
        // hold `to_read` values, the loop below ends with this error; return
        // it before the reservation, which is as large as the header declares.
        if to_read > (buf_len - end_offset) / 4 {
            return Err(ParquetError::EOF("eof decoding byte array".into()));
        }
        output.views.reserve(to_read);

        // Safety: we reserved enough space in output.views
        // and we will only write up to to_read views / track how many views we wrote.
        // Ideally, we would use `Vec::extend` here, but this generates sub-optimal code.
        let views_ptr = output.views.as_mut_ptr().wrapping_add(output.views.len());
        for i in 0..to_read {
            let start_offset = end_offset + 4;

            if start_offset > buf_len {
                return Err(ParquetError::EOF("eof decoding byte array".into()));
            }

            // Safety: we have checked that start_offset <= buf_len
            let len = u32::from_le_bytes(
                unsafe { buf.get_unchecked(end_offset..start_offset) }
                    .try_into()
                    .unwrap(),
            );

            end_offset = start_offset + len as usize;

            if end_offset > buf_len {
                return Err(ParquetError::EOF("eof decoding byte array".into()));
            }

            if VALIDATE_UTF8 {
                // It seems you are trying to understand what's going on here, take a breath and be patient.
                // Utf-8 validation is a non-trivial task, here are some background facts:
                // (1) Validating one 2048-byte string is much faster than validating 128 of 16-byte string.
                //     As shown in https://github.com/apache/arrow-rs/pull/6009#issuecomment-2211174229
                //     Potentially because the SIMD operations favor longer strings.
                // (2) Practical strings are short, 99% of strings are smaller than 100 bytes, as shown in paper:
                //     https://www.vldb.org/pvldb/vol17/p148-zeng.pdf, Figure 5f.
                // (3) Parquet plain encoding makes utf-8 validation harder,
                //     because it stores the length of each string right before the string.
                //     This means naive utf-8 validation will be slow, because the validation need to skip the length bytes.
                //     I.e., the validation cannot validate the buffer in one pass, but instead, validate strings chunk by chunk.
                //
                // Given the above observations, the goal is to do batch validation as much as possible.
                // The key idea is that if the length is smaller than 128 (99% of the case), then the length bytes are valid utf-8, as reasoned below:
                // If the length is smaller than 128, its 4-byte encoding are [0, 0, 0, len].
                // Each of the byte is a valid ASCII character, so they are valid utf-8.
                // Since they are all smaller than 128, the won't break a utf-8 code point (won't mess with later bytes).
                //
                // The implementation keeps a water mark `utf8_validation_begin` to track the beginning of the buffer that is not validated.
                // If the length is smaller than 128, then we continue to next string.
                // If the length is larger than 128, then we validate the buffer before the length bytes, and move the water mark to the beginning of next string.
                if len >= 128 {
                    // unfortunately, the len bytes may not be valid utf8, we need to wrap up and validate everything before it.
                    check_valid_utf8(unsafe {
                        buf.get_unchecked(utf8_validation_begin..start_offset - 4)
                    })?;
                    // move the cursor to skip the len bytes.
                    utf8_validation_begin = start_offset;
                }
            }

            let view = make_view(
                unsafe { buf.get_unchecked(start_offset..end_offset) },
                block_id,
                start_offset as u32,
            );
            // Safety: views_ptr is valid for writes, and we have reserved enough space.
            unsafe {
                views_ptr.add(i).write(view);
            }
        }

        // Safety: we have written `to_read` views to `views_ptr`
        unsafe {
            output.views.set_len(output.views.len() + to_read);
        }
        if VALIDATE_UTF8 {
            // validate values from the previously validated location up to (but not including)
            // the length of this string
            check_valid_utf8(unsafe { buf.get_unchecked(utf8_validation_begin..end_offset) })?;
        }

        self.offset = end_offset;
        self.max_remaining_values -= to_read;

        Ok(to_read)
    }

    pub fn skip(&mut self, to_skip: usize) -> Result<usize> {
        let to_skip = to_skip.min(self.max_remaining_values);
        let mut skip = 0;
        let buf: &[u8] = self.buf.as_ref();

        while self.offset < self.buf.len() && skip != to_skip {
            if self.offset + 4 > buf.len() {
                return Err(ParquetError::EOF("eof decoding byte array".into()));
            }
            let len_bytes: [u8; 4] = buf[self.offset..self.offset + 4].try_into().unwrap();
            let len = u32::from_le_bytes(len_bytes) as usize;
            let end_offset = self.offset + 4 + len;
            // Keep `offset` inside the buffer: `read` slices from it unchecked.
            if end_offset > buf.len() {
                return Err(ParquetError::EOF("eof decoding byte array".into()));
            }
            skip += 1;
            self.offset = end_offset;
        }
        self.max_remaining_values -= skip;
        Ok(skip)
    }
}

pub struct ByteViewArrayDecoderDictionary {
    decoder: DictIndexDecoder,
}

impl ByteViewArrayDecoderDictionary {
    fn new(data: Bytes, num_levels: usize, num_values: Option<usize>) -> Result<Self> {
        Ok(Self {
            decoder: DictIndexDecoder::new(data, num_levels, num_values)?,
        })
    }

    /// Reads the next indexes from self.decoder
    /// the indexes are assumed to be indexes into `dict`
    /// the output values are written to output
    ///
    /// Assumptions / Optimization
    /// This function checks if dict.buffers() are the last buffers in `output`, and if so
    /// reuses the dictionary page buffers directly without copying data
    ///
    /// If the dictionary is empty, the buffer contains empty view.
    fn read(&mut self, output: &mut ViewBuffer, dict: &ViewBuffer, len: usize) -> Result<usize> {
        if dict.is_empty() || len == 0 {
            return Ok(0);
        }

        // Check if the last few buffer of `output`` are the same as the `dict` buffer
        // This is to avoid creating a new buffers if the same dictionary is used for multiple `read`
        let need_to_create_new_buffer = {
            if output.buffers.len() >= dict.buffers.len() {
                let offset = output.buffers.len() - dict.buffers.len();
                output.buffers[offset..]
                    .iter()
                    .zip(dict.buffers.iter())
                    .any(|(a, b)| !a.ptr_eq(b))
            } else {
                true
            }
        };

        if need_to_create_new_buffer {
            for b in &dict.buffers {
                output.buffers.push(b.clone());
            }
        }

        // Calculate the offset of the dictionary buffers in the output buffers
        // For example if the 2nd buffer in the dictionary is the 5th buffer in the output buffers,
        // then the base_buffer_idx is 5 - 2 = 3
        let base_buffer_idx = output.buffers.len() as u32 - dict.buffers.len() as u32;

        // Pre-reserve output capacity to avoid per-chunk reallocation in extend
        output.views.reserve(len);

        let mut error = None;
        let read = self.decoder.read(len, |keys| {
            if base_buffer_idx == 0 {
                // the dictionary buffers are the last buffers in output, we can directly use the views
                output
                    .views
                    .extend(keys.iter().map(|k| match dict.views.get(*k as usize) {
                        Some(&view) => view,
                        None => {
                            if error.is_none() {
                                error = Some(general_err!("invalid key={} for dictionary", *k));
                            }
                            0
                        }
                    }));
            } else {
                output
                    .views
                    .extend(keys.iter().map(|k| match dict.views.get(*k as usize) {
                        Some(&view) => {
                            let len = view as u32;
                            if len <= 12 {
                                view
                            } else {
                                let mut view = ByteView::from(view);
                                view.buffer_index += base_buffer_idx;
                                view.into()
                            }
                        }
                        None => {
                            if error.is_none() {
                                error = Some(general_err!("invalid key={} for dictionary", *k));
                            }
                            0
                        }
                    }));
            }
            Ok(())
        })?;
        if let Some(e) = error {
            return Err(e);
        }
        Ok(read)
    }

    fn skip(&mut self, dict: &ViewBuffer, to_skip: usize) -> Result<usize> {
        if dict.is_empty() {
            return Ok(0);
        }
        self.decoder.skip(to_skip)
    }
}

/// Decoder from [`Encoding::DELTA_LENGTH_BYTE_ARRAY`] data to [`ViewBuffer`]
pub struct ByteViewArrayDecoderDeltaLength {
    /// The lengths, decoded a window at a time. Boxed: the decoder enum
    /// holds this beside much smaller decoders.
    lengths: Box<DeltaLengths>,
    data: Bytes,
    data_offset: usize,
    validate_utf8: bool,
}

impl ByteViewArrayDecoderDeltaLength {
    /// `max_values` is the value count of the page; a length block that
    /// declares more lengths is an error.
    fn new(data: Bytes, max_values: usize, validate_utf8: bool) -> Result<Self> {
        let lengths = Box::new(DeltaLengths::try_new(
            data.clone(),
            max_values,
            "DELTA_LENGTH_BYTE_ARRAY",
        )?);
        Ok(Self {
            data_offset: lengths.end_offset(),
            lengths,
            data,
            validate_utf8,
        })
    }

    /// The end of the data of the next `total_bytes` bytes, if the page holds
    /// them.
    fn data_end(&self, total_bytes: usize) -> Result<usize> {
        self.data_offset
            .checked_add(total_bytes)
            .filter(|end| *end <= self.data.len())
            .ok_or_else(|| {
                ParquetError::General("Insufficient delta length byte array bytes".to_string())
            })
    }

    fn read(&mut self, output: &mut ViewBuffer, len: usize) -> Result<usize> {
        let to_read = len.min(self.lengths.remaining());
        output.views.reserve(to_read);

        // Zero copy convert `bytes::Bytes` into `arrow_buffer::Buffer`
        let bytes = Buffer::from(self.data.clone());
        let block_id = output.append_block(bytes);

        let initial_offset = self.data_offset;
        let mut read = 0;
        while read < to_read {
            let src_lengths = self.lengths.peek(to_read - read)?;
            let count = src_lengths.len();
            let total_bytes: usize = src_lengths.iter().map(|x| *x as usize).sum();
            let data_end = self
                .data_offset
                .checked_add(total_bytes)
                .filter(|end| *end <= self.data.len())
                .ok_or_else(|| {
                    ParquetError::General("Insufficient delta length byte array bytes".to_string())
                })?;

            let mut current_offset = self.data_offset;
            output.views.extend(src_lengths.iter().map(|length| {
                let len = *length as u32;
                let start_offset = current_offset;
                current_offset += len as usize;
                // The window's lengths are non-negative and end at
                // `data_end`, inside the data (checked above).
                make_view(
                    &self.data[start_offset..start_offset + len as usize],
                    block_id,
                    start_offset as u32,
                )
            }));

            self.data_offset = data_end;
            self.lengths.consume(count);
            read += count;
        }

        // Delta length encoding has continuous strings, we can validate utf8 in one go
        if self.validate_utf8 {
            check_valid_utf8(&self.data[initial_offset..self.data_offset])?;
        }

        Ok(to_read)
    }

    fn skip(&mut self, to_skip: usize) -> Result<usize> {
        let to_skip = self.lengths.remaining().min(to_skip);
        let mut skipped = 0;
        while skipped < to_skip {
            let src_lengths = self.lengths.peek(to_skip - skipped)?;
            let count = src_lengths.len();
            let total_bytes: usize = src_lengths.iter().map(|x| *x as usize).sum();
            self.data_offset = self.data_end(total_bytes)?;
            self.lengths.consume(count);
            skipped += count;
        }
        Ok(to_skip)
    }
}

/// Decoder from [`Encoding::DELTA_BYTE_ARRAY`] to [`ViewBuffer`]
pub struct ByteViewArrayDecoderDelta {
    decoder: DeltaByteArrayDecoder,
    validate_utf8: bool,
}

impl ByteViewArrayDecoderDelta {
    fn new(data: Bytes, max_values: usize, validate_utf8: bool) -> Result<Self> {
        Ok(Self {
            decoder: DeltaByteArrayDecoder::new(data, max_values)?,
            validate_utf8,
        })
    }

    // Unlike other encodings, we need to copy the data.
    //
    //  DeltaByteArray data is stored using shared prefixes/suffixes,
    // which results in potentially non-contiguous
    // strings, while Arrow encodings require contiguous strings
    //
    // <https://parquet.apache.org/docs/file-format/data-pages/encodings/#delta-strings-delta_byte_array--7>

    fn read(&mut self, output: &mut ViewBuffer, len: usize) -> Result<usize> {
        let to_reserve = len.min(self.decoder.remaining());
        output.views.reserve(to_reserve);

        // array buffer only have long strings
        let mut array_buffer: Vec<u8> = Vec::with_capacity(4096);

        let buffer_id = output.buffers.len() as u32;

        // Use unsafe ptr writes instead of per-element push to avoid
        // repeated length checks. Safety: we reserved enough space above.
        let views_ptr = output.views.as_mut_ptr();
        let initial_len = output.views.len();
        let mut write_count = 0;

        let read = if !self.validate_utf8 {
            self.decoder.read(len, |bytes| {
                let offset = array_buffer.len();
                let view = make_view(bytes, buffer_id, offset as u32);
                if bytes.len() > 12 {
                    // only copy the data to buffer if the string can not be inlined.
                    array_buffer.extend_from_slice(bytes);
                }

                // Safety: views_ptr is valid for writes, we reserved enough space,
                // and write_count < to_reserve.
                unsafe {
                    views_ptr.add(initial_len + write_count).write(view);
                }
                write_count += 1;
                Ok(())
            })?
        } else {
            // utf8 validation buffer has only short strings. These short
            // strings are inlined into the views but we copy them into a
            // contiguous buffer to accelerate validation.
            let mut utf8_validation_buffer = Vec::with_capacity(4096);

            let v = self.decoder.read(len, |bytes| {
                let offset = array_buffer.len();
                let view = make_view(bytes, buffer_id, offset as u32);
                if bytes.len() > 12 {
                    // only copy the data to buffer if the string can not be inlined.
                    array_buffer.extend_from_slice(bytes);
                } else {
                    utf8_validation_buffer.extend_from_slice(bytes);
                }

                // Safety: views_ptr is valid for writes, we reserved enough space,
                // and write_count < to_reserve. Utf-8 validation is done later.
                unsafe {
                    views_ptr.add(initial_len + write_count).write(view);
                }
                write_count += 1;
                Ok(())
            })?;
            check_valid_utf8(&array_buffer)?;
            check_valid_utf8(&utf8_validation_buffer)?;
            v
        };

        // Safety: we wrote exactly `read` views via ptr writes above
        unsafe {
            output.views.set_len(initial_len + read);
        }

        let actual_block_id = output.append_block(Buffer::from_vec(array_buffer));
        assert_eq!(actual_block_id, buffer_id);
        Ok(read)
    }

    fn skip(&mut self, to_skip: usize) -> Result<usize> {
        self.decoder.skip(to_skip)
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::StringViewArray;
    use arrow_buffer::Buffer;

    use crate::{
        arrow::{
            array_reader::test_util::{byte_array_all_encodings, encode_byte_array, utf8_column},
            buffer::view_buffer::ViewBuffer,
            record_reader::buffer::ValuesBuffer,
        },
        basic::Encoding,
        column::reader::decoder::ColumnValueDecoder,
        data_type::ByteArray,
    };

    use super::*;

    #[test]
    fn test_byte_array_string_view_decoder() {
        let (pages, encoded_dictionary) =
            byte_array_all_encodings(vec!["hello", "world", "large payload over 12 bytes", "b"]);

        let column_desc = utf8_column();
        let mut decoder = ByteViewArrayColumnValueDecoder::new(&column_desc);

        decoder
            .set_dict(encoded_dictionary, 4, Encoding::RLE_DICTIONARY, false)
            .unwrap();

        for (encoding, page) in pages {
            let mut output = ViewBuffer::with_capacity(0);
            decoder.set_data(encoding, page, 4, Some(4)).unwrap();

            assert_eq!(decoder.read(&mut output, 1).unwrap(), 1);
            assert_eq!(decoder.read(&mut output, 1).unwrap(), 1);
            assert_eq!(decoder.read(&mut output, 2).unwrap(), 2);
            assert_eq!(decoder.read(&mut output, 4).unwrap(), 0);

            assert_eq!(output.views.len(), 4);

            let valid = [false, false, true, true, false, true, true, false, false];
            let valid_buffer = Buffer::from_iter(valid.iter().copied());

            output
                .pad_nulls(0, 4, valid.len(), valid_buffer.as_slice())
                .unwrap();
            let array = output.into_array(Some(valid_buffer), &ArrowType::Utf8View);
            let strings = array.as_any().downcast_ref::<StringViewArray>().unwrap();

            assert_eq!(
                strings.iter().collect::<Vec<_>>(),
                vec![
                    None,
                    None,
                    Some("hello"),
                    Some("world"),
                    None,
                    Some("large payload over 12 bytes"),
                    Some("b"),
                    None,
                    None,
                ]
            );
        }
    }

    #[test]
    fn test_byte_view_array_plain_decoder_reuse_buffer() {
        let byte_array = vec!["hello", "world", "large payload over 12 bytes", "b"];
        let byte_array: Vec<ByteArray> = byte_array.into_iter().map(|x| x.into()).collect();
        let pages = encode_byte_array(Encoding::PLAIN, &byte_array);

        let column_desc = utf8_column();
        let mut decoder = ByteViewArrayColumnValueDecoder::new(&column_desc);

        let mut view_buffer = ViewBuffer::with_capacity(0);
        decoder.set_data(Encoding::PLAIN, pages, 4, None).unwrap();
        decoder.read(&mut view_buffer, 1).unwrap();
        decoder.read(&mut view_buffer, 1).unwrap();
        assert_eq!(view_buffer.buffers.len(), 1);

        decoder.read(&mut view_buffer, 1).unwrap();
        assert_eq!(view_buffer.buffers.len(), 1);
    }

    #[test]
    fn selected_dictionary_views_validate_only_selected_values() {
        use crate::column::page::Page;
        use crate::schema::{parser::parse_message_type, types::SchemaDescriptor};
        use crate::util::test_common::page_util::InMemoryPageIterator;
        use std::sync::Arc;

        let schema = parse_message_type("message x { REQUIRED BYTE_ARRAY s (UTF8); }").unwrap();
        let desc = SchemaDescriptor::new(Arc::new(schema)).column(0);
        let dictionary = Page::DictionaryPage {
            buf: Bytes::from_static(b"\x01\0\0\0a\x01\0\0\0\xff"),
            num_values: 2,
            encoding: Encoding::PLAIN,
            is_sorted: false,
        };
        let page = Page::DataPage {
            // Width one, one packed group, codes [0, 1] with six padding slots.
            buf: Bytes::from_static(&[1, 3, 2]),
            num_values: 2,
            encoding: Encoding::RLE_DICTIONARY,
            def_level_encoding: Encoding::RLE,
            rep_level_encoding: Encoding::RLE,
            statistics: None,
        };
        let build = || {
            make_byte_view_array_reader(
                Box::new(InMemoryPageIterator::new([vec![
                    dictionary.clone(),
                    page.clone(),
                ]])),
                desc.clone(),
                None,
                2,
                None,
            )
            .unwrap()
        };

        let valid = build().read_selected(&[1], 2).unwrap();
        let valid = valid.as_any().downcast_ref::<StringViewArray>().unwrap();
        assert_eq!(valid, &StringViewArray::from(vec!["a"]));
        assert!(build().read_selected(&[2], 2).is_err());
        assert!(build().next_batch(2).is_err());
    }

    #[test]
    fn selected_views_skip_a_corrupt_page_across_vectors() {
        use crate::schema::{parser::parse_message_type, types::SchemaDescriptor};
        use crate::util::test_common::page_util::InMemoryPageIterator;

        let schema = parse_message_type("message x { REQUIRED BYTE_ARRAY s (UTF8); }").unwrap();
        let desc = SchemaDescriptor::new(Arc::new(schema)).column(0);
        let plain = |buf| Page::DataPage {
            buf: Bytes::from_static(buf),
            num_values: 1,
            encoding: Encoding::PLAIN,
            def_level_encoding: Encoding::RLE,
            rep_level_encoding: Encoding::RLE,
            statistics: None,
        };
        let corrupt = Page::DataPage {
            buf: Bytes::from_static(b"\xff"),
            num_values: 4,
            encoding: Encoding::PLAIN,
            def_level_encoding: Encoding::RLE,
            rep_level_encoding: Encoding::RLE,
            statistics: None,
        };
        let build = || {
            let pages = vec![
                plain(b"\x01\0\0\0a"),
                corrupt.clone(),
                plain(b"\x01\0\0\0b"),
            ];
            make_byte_view_array_reader(
                Box::new(InMemoryPageIterator::new([pages])),
                desc.clone(),
                None,
                2,
                None,
            )
            .unwrap()
        };
        let mut reader = build();
        let a = reader.read_selected(&[1], 3).unwrap();
        let b = reader.read_selected(&[4], 3).unwrap();
        assert_eq!(
            a.as_any().downcast_ref::<StringViewArray>().unwrap(),
            &StringViewArray::from(vec!["a"])
        );
        assert_eq!(
            b.as_any().downcast_ref::<StringViewArray>().unwrap(),
            &StringViewArray::from(vec!["b"])
        );
        assert!(build().read_selected(&[2], 6).is_err());
    }

    #[test]
    fn selected_views_match_full_across_page_and_chunk_slices() {
        use crate::arrow::ArrowWriter;
        use crate::file::{
            properties::{WriterProperties, WriterVersion},
            reader::{FilePageIterator, FileReader, SerializedFileReader},
        };
        use arrow_array::{BooleanArray, RecordBatch};
        use arrow_schema::{Field, Schema};
        use rand::{RngExt, SeedableRng};

        let mut rng = rand::rngs::SmallRng::seed_from_u64(138);
        for version in [WriterVersion::PARQUET_1_0, WriterVersion::PARQUET_2_0] {
            for nullable in [false, true] {
                let schema = Arc::new(Schema::new(vec![Field::new(
                    "s",
                    ArrowType::Utf8View,
                    nullable,
                )]));
                let values = (0..1703).map(|row| {
                    (!nullable || rng.random_bool(0.7)).then(|| {
                        if row % 339 < 89 {
                            "é".to_owned()
                        } else {
                            format!("long value {row:06} 🥭")
                        }
                    })
                });
                let batch = RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(StringViewArray::from_iter(values))],
                )
                .unwrap();
                let props = WriterProperties::builder()
                    .set_writer_version(version)
                    .set_encoding(Encoding::PLAIN)
                    .set_dictionary_page_size_limit(192)
                    .set_max_row_group_row_count(Some(339))
                    .set_data_page_row_count_limit(71)
                    .set_write_batch_size(37)
                    .build();
                let mut bytes = Vec::new();
                let mut writer = ArrowWriter::try_new(&mut bytes, schema, Some(props)).unwrap();
                writer.write(&batch).unwrap();
                writer.close().unwrap();
                let file = Arc::new(SerializedFileReader::new(Bytes::from(bytes)).unwrap());
                let desc = file.metadata().file_metadata().schema_descr().column(0);
                let mut pages = file
                    .get_row_group(0)
                    .unwrap()
                    .get_column_page_reader(0)
                    .unwrap();
                let encodings: Vec<_> = pages
                    .by_ref()
                    .map(|page| page.unwrap().encoding())
                    .collect();
                assert!(encodings.contains(&Encoding::RLE_DICTIONARY));
                assert!(encodings.contains(&Encoding::PLAIN));
                let build = || {
                    make_byte_view_array_reader(
                        Box::new(FilePageIterator::new(0, file.clone()).unwrap()),
                        desc.clone(),
                        None,
                        79,
                        None,
                    )
                    .unwrap()
                };

                let mut full = build();
                let mut selected = build();
                assert!(selected.read_selected(&[], 65).is_err());
                let mut start = 43;
                assert_eq!(full.skip_records(start).unwrap(), start);
                assert!(selected.read_selected(&[0], start).unwrap().is_empty());
                for vector in 0.. {
                    if start == batch.num_rows() {
                        break;
                    }
                    let rows =
                        [37usize, 79, 64, 139, 1, 507][vector % 6].min(batch.num_rows() - start);
                    let density = [0.0, 0.001, 0.02, 0.1, 0.5, 1.0][(vector / 2) % 6];
                    let mut words = vec![0; rows.div_ceil(64)];
                    let mask = BooleanArray::from_iter((0..rows).map(|row| {
                        let keep = rng.random_bool(density);
                        words[row / 64] |= u64::from(keep) << (row % 64);
                        Some(keep)
                    }));
                    let batch = full.next_batch(rows).unwrap();
                    let (actual, expected) = if vector % 7 == 5 {
                        (selected.next_batch(rows).unwrap(), batch)
                    } else {
                        (
                            selected.read_selected(&words, rows).unwrap(),
                            arrow_select::filter::filter(batch.as_ref(), &mask).unwrap(),
                        )
                    };
                    assert_eq!(
                        actual.as_ref(),
                        expected.as_ref(),
                        "{version:?}, nullable={nullable}, offset={start}"
                    );
                    start += rows;
                }
                assert!(selected.read_selected(&[1], 1).is_err());
            }
        }
    }
}

#[cfg(test)]
mod invalid_input_tests {
    use super::*;
    use crate::arrow::buffer::view_buffer::ViewBuffer;

    /// A skip must not move the offset past the buffer: the read slices the
    /// buffer from that offset without a bounds check.
    #[test]
    fn plain_skip_past_end_fails() {
        // One value of length 100 in a buffer that holds one byte of it.
        let mut data = 100u32.to_le_bytes().to_vec();
        data.push(b'a');

        for validate_utf8 in [false, true] {
            let mut decoder =
                ByteViewArrayDecoderPlain::new(Bytes::from(data.clone()), 2, None, validate_utf8);
            let err = decoder.skip(1).unwrap_err();
            assert_eq!(err.to_string(), "EOF: eof decoding byte array");

            let mut output = ViewBuffer::default();
            assert_eq!(decoder.read(&mut output, 0).unwrap(), 0);
            let err = decoder.read(&mut output, 1).unwrap_err();
            assert_eq!(err.to_string(), "EOF: eof decoding byte array");
        }
    }
}

#[cfg(test)]
mod allocation_bound_tests {
    use super::*;
    use crate::arrow::array_reader::test_util::utf8_column;
    use crate::arrow::buffer::view_buffer::ViewBuffer;

    /// Two PLAIN values: "a" and "bc".
    fn two_values() -> Bytes {
        Bytes::from_static(b"\x01\0\0\0a\x02\0\0\0bc")
    }

    /// The read fails when the page ends before the declared count. The views
    /// were reserved for the declared count before the first value was read.
    #[test]
    fn plain_read_above_the_page_bytes_fails_without_reservation() {
        for validate_utf8 in [false, true] {
            let mut decoder =
                ByteViewArrayDecoderPlain::new(two_values(), 1 << 40, None, validate_utf8);
            let mut output = ViewBuffer::default();
            let err = decoder.read(&mut output, usize::MAX).unwrap_err();
            assert_eq!(err.to_string(), "EOF: eof decoding byte array");
            assert_eq!(output.views.capacity(), 0);

            // The values that the page holds are still read.
            assert_eq!(decoder.read(&mut output, 2).unwrap(), 2);
            assert_eq!(output.views.len(), 2);
        }
    }

    #[test]
    fn dictionary_count_above_the_page_bytes_fails() {
        let mut decoder = ByteViewArrayColumnValueDecoder::new(&utf8_column());
        decoder
            .set_dict(two_values(), 2, Encoding::PLAIN, false)
            .unwrap();
        let err = decoder
            .set_dict(two_values(), i32::MAX as u32, Encoding::PLAIN, false)
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "Parquet error: dictionary page declares 2147483647 values, but its 11 bytes hold at most 2"
        );
    }
}
