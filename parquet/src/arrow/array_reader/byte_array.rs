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

use crate::arrow::array_reader::{ArrayReader, read_records, skip_records};
use crate::arrow::buffer::bit_util::sign_extend_be;
use crate::arrow::buffer::offset_buffer::OffsetBuffer;
use crate::arrow::decoder::{DeltaByteArrayDecoder, DictIndexDecoder};
use crate::arrow::record_reader::GenericRecordReader;
use crate::arrow::schema::parquet_to_arrow_field;
use crate::basic::{ConvertedType, Encoding};
use crate::column::page::PageIterator;
use crate::column::reader::decoder::ColumnValueDecoder;
use crate::encodings::decoding::DeltaLengths;
use crate::errors::{ParquetError, Result};
use crate::schema::types::ColumnDescPtr;
use arrow_array::{
    Array, ArrayRef, BinaryArray, Decimal128Array, Decimal256Array, OffsetSizeTrait,
};
use arrow_buffer::i256;
use arrow_schema::DataType as ArrowType;
use bytes::Bytes;
use std::any::Any;
use std::sync::Arc;

/// Returns an [`ArrayReader`] that decodes the provided byte array column
pub fn make_byte_array_reader(
    pages: Box<dyn PageIterator>,
    column_desc: ColumnDescPtr,
    arrow_type: Option<ArrowType>,
    batch_size: usize,
    padding_threshold: Option<i16>,
) -> Result<Box<dyn ArrayReader>> {
    // Check if Arrow type is specified, else create it from Parquet type
    let data_type = match arrow_type {
        Some(t) => t,
        None => parquet_to_arrow_field(column_desc.as_ref())?
            .data_type()
            .clone(),
    };

    match data_type {
        ArrowType::Binary
        | ArrowType::Utf8
        | ArrowType::Decimal128(_, _)
        | ArrowType::Decimal256(_, _) => {
            let mut reader = GenericRecordReader::new(column_desc, batch_size);
            if let Some(threshold) = padding_threshold {
                reader.set_padding_threshold(threshold);
            }
            Ok(Box::new(ByteArrayReader::<i32>::new(
                pages, data_type, reader,
            )))
        }
        ArrowType::LargeUtf8 | ArrowType::LargeBinary => {
            let mut reader = GenericRecordReader::new(column_desc, batch_size);
            if let Some(threshold) = padding_threshold {
                reader.set_padding_threshold(threshold);
            }
            Ok(Box::new(ByteArrayReader::<i64>::new(
                pages, data_type, reader,
            )))
        }
        _ => Err(general_err!(
            "invalid data type for byte array reader - {}",
            data_type
        )),
    }
}

/// An [`ArrayReader`] for variable length byte arrays
struct ByteArrayReader<I: OffsetSizeTrait> {
    data_type: ArrowType,
    pages: Box<dyn PageIterator>,
    def_levels_buffer: Option<Vec<i16>>,
    rep_levels_buffer: Option<Vec<i16>>,
    record_reader: GenericRecordReader<OffsetBuffer<I>, ByteArrayColumnValueDecoder<I>>,
}

impl<I: OffsetSizeTrait> ByteArrayReader<I> {
    fn new(
        pages: Box<dyn PageIterator>,
        data_type: ArrowType,
        record_reader: GenericRecordReader<OffsetBuffer<I>, ByteArrayColumnValueDecoder<I>>,
    ) -> Self {
        Self {
            data_type,
            pages,
            def_levels_buffer: None,
            rep_levels_buffer: None,
            record_reader,
        }
    }
}

impl<I: OffsetSizeTrait> ArrayReader for ByteArrayReader<I> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_data_type(&self) -> &ArrowType {
        &self.data_type
    }

    fn read_records(&mut self, batch_size: usize) -> Result<usize> {
        read_records(&mut self.record_reader, self.pages.as_mut(), batch_size)
    }

    fn consume_batch(&mut self) -> Result<ArrayRef> {
        let buffer = self.record_reader.consume_record_data();
        let null_buffer = self.record_reader.consume_compact_bitmap();
        self.def_levels_buffer = self.record_reader.consume_def_levels();
        self.rep_levels_buffer = self.record_reader.consume_rep_levels();
        self.record_reader.reset();

        let array: ArrayRef = match self.data_type {
            // Apply conversion to all elements regardless of null slots as the conversions
            // are infallible. This improves performance by avoiding a branch in the inner
            // loop (see docs for `PrimitiveArray::from_unary`).
            ArrowType::Decimal128(p, s) => {
                let array = buffer.into_array(null_buffer, ArrowType::Binary);
                let binary = array.as_any().downcast_ref::<BinaryArray>().unwrap();
                // Null slots will have 0 length, so we need to check for that in the lambda
                // or sign_extend_be will panic.
                let decimal = Decimal128Array::from_unary(binary, |x| match x.len() {
                    0 => i128::default(),
                    _ => i128::from_be_bytes(sign_extend_be(x)),
                })
                .with_precision_and_scale(p, s)?;
                Arc::new(decimal)
            }
            ArrowType::Decimal256(p, s) => {
                let array = buffer.into_array(null_buffer, ArrowType::Binary);
                let binary = array.as_any().downcast_ref::<BinaryArray>().unwrap();
                // Null slots will have 0 length, so we need to check for that in the lambda
                // or sign_extend_be will panic.
                let decimal = Decimal256Array::from_unary(binary, |x| match x.len() {
                    0 => i256::default(),
                    _ => i256::from_be_bytes(sign_extend_be(x)),
                })
                .with_precision_and_scale(p, s)?;
                Arc::new(decimal)
            }
            _ => buffer.into_array(null_buffer, self.data_type.clone()),
        };

        Ok(array)
    }

    fn set_remaining_rows(&mut self, rows: usize) {
        self.record_reader.set_remaining_rows(rows);
    }

    fn skip_records(&mut self, num_records: usize) -> Result<usize> {
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

/// A [`ColumnValueDecoder`] for variable length byte arrays
struct ByteArrayColumnValueDecoder<I: OffsetSizeTrait> {
    dict: Option<OffsetBuffer<I>>,
    decoder: Option<ByteArrayDecoder>,
    validate_utf8: bool,
}

impl<I: OffsetSizeTrait> ColumnValueDecoder for ByteArrayColumnValueDecoder<I> {
    type Buffer = OffsetBuffer<I>;

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

        let mut buffer = OffsetBuffer::with_capacity(0);
        let mut decoder = ByteArrayDecoderPlain::new(
            buf,
            num_values as usize,
            Some(num_values as usize),
            self.validate_utf8,
        );
        let read = decoder.read(&mut buffer, usize::MAX)?;
        check_dictionary_len(read, num_values as usize)?;
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
        self.decoder = Some(ByteArrayDecoder::new(
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

/// The PLAIN decoder stops at the end of the page. A dictionary page that
/// holds fewer values than its header declares is invalid: the indices of the
/// data pages can refer to the missing values.
pub(crate) fn check_dictionary_len(read: usize, declared: usize) -> Result<()> {
    if read != declared {
        return Err(general_err!(
            "dictionary page declares {} values, but holds {}",
            declared,
            read
        ));
    }
    Ok(())
}

/// A generic decoder from uncompressed parquet value data to [`OffsetBuffer`]
pub enum ByteArrayDecoder {
    Plain(ByteArrayDecoderPlain),
    Dictionary(ByteArrayDecoderDictionary),
    DeltaLength(ByteArrayDecoderDeltaLength),
    DeltaByteArray(ByteArrayDecoderDelta),
}

impl ByteArrayDecoder {
    pub fn new(
        encoding: Encoding,
        data: Bytes,
        num_levels: usize,
        num_values: Option<usize>,
        validate_utf8: bool,
    ) -> Result<Self> {
        let decoder = match encoding {
            Encoding::PLAIN => ByteArrayDecoder::Plain(ByteArrayDecoderPlain::new(
                data,
                num_levels,
                num_values,
                validate_utf8,
            )),
            Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY => ByteArrayDecoder::Dictionary(
                ByteArrayDecoderDictionary::new(data, num_levels, num_values)?,
            ),
            Encoding::DELTA_LENGTH_BYTE_ARRAY => {
                ByteArrayDecoder::DeltaLength(ByteArrayDecoderDeltaLength::new(
                    data,
                    num_values.unwrap_or(num_levels),
                    validate_utf8,
                )?)
            }
            Encoding::DELTA_BYTE_ARRAY => ByteArrayDecoder::DeltaByteArray(
                ByteArrayDecoderDelta::new(data, num_values.unwrap_or(num_levels), validate_utf8)?,
            ),
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
    pub fn read<I: OffsetSizeTrait>(
        &mut self,
        out: &mut OffsetBuffer<I>,
        len: usize,
        dict: Option<&OffsetBuffer<I>>,
    ) -> Result<usize> {
        match self {
            ByteArrayDecoder::Plain(d) => d.read(out, len),
            ByteArrayDecoder::Dictionary(d) => {
                let dict =
                    dict.ok_or_else(|| general_err!("missing dictionary page for column"))?;

                d.read(out, dict, len)
            }
            ByteArrayDecoder::DeltaLength(d) => d.read(out, len),
            ByteArrayDecoder::DeltaByteArray(d) => d.read(out, len),
        }
    }

    /// Skip `len` values
    pub fn skip<I: OffsetSizeTrait>(
        &mut self,
        len: usize,
        dict: Option<&OffsetBuffer<I>>,
    ) -> Result<usize> {
        match self {
            ByteArrayDecoder::Plain(d) => d.skip(len),
            ByteArrayDecoder::Dictionary(d) => {
                let dict =
                    dict.ok_or_else(|| general_err!("missing dictionary page for column"))?;

                d.skip(dict, len)
            }
            ByteArrayDecoder::DeltaLength(d) => d.skip(len),
            ByteArrayDecoder::DeltaByteArray(d) => d.skip(len),
        }
    }
}

/// Decoder from [`Encoding::PLAIN`] data to [`OffsetBuffer`]
pub struct ByteArrayDecoderPlain {
    buf: Bytes,
    offset: usize,
    validate_utf8: bool,

    /// This is a maximum as the null count is not always known, e.g. value data from
    /// a v1 data page
    max_remaining_values: usize,
}

impl ByteArrayDecoderPlain {
    pub fn new(
        buf: Bytes,
        num_levels: usize,
        num_values: Option<usize>,
        validate_utf8: bool,
    ) -> Self {
        Self {
            buf,
            validate_utf8,
            offset: 0,
            max_remaining_values: num_values.unwrap_or(num_levels),
        }
    }

    pub fn read<I: OffsetSizeTrait>(
        &mut self,
        output: &mut OffsetBuffer<I>,
        len: usize,
    ) -> Result<usize> {
        let initial_values_length = output.values.len();

        let to_read = len.min(self.max_remaining_values);
        let remaining_bytes = self.buf.len() - self.offset;
        // The value count is from the page header. Each value has a four-byte
        // length, so the remaining bytes limit the values that can be read.
        output.offsets.reserve(to_read.min(remaining_bytes / 4));

        // `to_read == 0` also covers a page that declares zero values but
        // has value bytes; the byte estimate below divides by that count.
        if remaining_bytes == 0 || to_read == 0 {
            return Ok(0);
        }

        let estimated_bytes = remaining_bytes
            .checked_mul(to_read)
            .map(|x| x / self.max_remaining_values)
            .unwrap_or_default();

        output.values.reserve(estimated_bytes);

        let mut read = 0;

        let buf = self.buf.as_ref();
        while self.offset < self.buf.len() && read != to_read {
            if self.offset + 4 > buf.len() {
                return Err(ParquetError::EOF("eof decoding byte array".into()));
            }
            let len_bytes: [u8; 4] = buf[self.offset..self.offset + 4].try_into().unwrap();
            let len = u32::from_le_bytes(len_bytes);

            let start_offset = self.offset + 4;
            let end_offset = start_offset + len as usize;
            if end_offset > buf.len() {
                return Err(ParquetError::EOF("eof decoding byte array".into()));
            }

            output.try_push(&buf[start_offset..end_offset], self.validate_utf8)?;

            self.offset = end_offset;
            read += 1;
        }
        self.max_remaining_values -= read;

        if self.validate_utf8 {
            output.check_valid_utf8(initial_values_length)?;
        }
        Ok(read)
    }

    pub fn skip(&mut self, to_skip: usize) -> Result<usize> {
        let to_skip = to_skip.min(self.max_remaining_values);
        let mut skip = 0;
        let buf = self.buf.as_ref();

        while self.offset < self.buf.len() && skip != to_skip {
            if self.offset + 4 > buf.len() {
                return Err(ParquetError::EOF("eof decoding byte array".into()));
            }
            let len_bytes: [u8; 4] = buf[self.offset..self.offset + 4].try_into().unwrap();
            let len = u32::from_le_bytes(len_bytes) as usize;
            let end_offset = self.offset + 4 + len;
            // Keep `offset` inside the buffer, as `read` does.
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

/// Decoder from [`Encoding::DELTA_LENGTH_BYTE_ARRAY`] data to [`OffsetBuffer`]
pub struct ByteArrayDecoderDeltaLength {
    /// The lengths, decoded a window at a time. Boxed: the decoder enum
    /// holds this beside much smaller decoders.
    lengths: Box<DeltaLengths>,
    data: Bytes,
    data_offset: usize,
    validate_utf8: bool,
}

impl ByteArrayDecoderDeltaLength {
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

    fn read<I: OffsetSizeTrait>(
        &mut self,
        output: &mut OffsetBuffer<I>,
        len: usize,
    ) -> Result<usize> {
        let initial_values_length = output.values.len();
        let to_read = len.min(self.lengths.remaining());
        output.offsets.reserve(to_read);

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

            // Delta length data is contiguous — copy the bytes of the window at once
            output.values.reserve(total_bytes);
            let base_offset = output.values.len();
            output
                .values
                .extend_from_slice(&self.data.as_ref()[self.data_offset..data_end]);

            let mut running = base_offset;
            output.offsets.extend(src_lengths.iter().map(|length| {
                running += *length as usize;
                I::from_usize(running).expect("index overflow decoding byte array")
            }));

            self.data_offset = data_end;
            self.lengths.consume(count);
            read += count;
        }

        if self.validate_utf8 {
            output.check_valid_utf8(initial_values_length)?;
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

/// Decoder from [`Encoding::DELTA_BYTE_ARRAY`] to [`OffsetBuffer`]
pub struct ByteArrayDecoderDelta {
    decoder: DeltaByteArrayDecoder,
    validate_utf8: bool,
}

impl ByteArrayDecoderDelta {
    fn new(data: Bytes, max_values: usize, validate_utf8: bool) -> Result<Self> {
        Ok(Self {
            decoder: DeltaByteArrayDecoder::new(data, max_values)?,
            validate_utf8,
        })
    }

    fn read<I: OffsetSizeTrait>(
        &mut self,
        output: &mut OffsetBuffer<I>,
        len: usize,
    ) -> Result<usize> {
        let initial_values_length = output.values.len();
        output.offsets.reserve(len.min(self.decoder.remaining()));

        let read = self
            .decoder
            .read(len, |bytes| output.try_push(bytes, self.validate_utf8))?;

        if self.validate_utf8 {
            output.check_valid_utf8(initial_values_length)?;
        }
        Ok(read)
    }

    fn skip(&mut self, to_skip: usize) -> Result<usize> {
        self.decoder.skip(to_skip)
    }
}

/// Decoder from [`Encoding::RLE_DICTIONARY`] to [`OffsetBuffer`]
pub struct ByteArrayDecoderDictionary {
    decoder: DictIndexDecoder,
}

impl ByteArrayDecoderDictionary {
    fn new(data: Bytes, num_levels: usize, num_values: Option<usize>) -> Result<Self> {
        Ok(Self {
            decoder: DictIndexDecoder::new(data, num_levels, num_values)?,
        })
    }

    fn read<I: OffsetSizeTrait>(
        &mut self,
        output: &mut OffsetBuffer<I>,
        dict: &OffsetBuffer<I>,
        len: usize,
    ) -> Result<usize> {
        // All data must be NULL
        if dict.is_empty() {
            return Ok(0);
        }

        // Pre-reserve offsets capacity to avoid per-chunk reallocation
        output.offsets.reserve(len);

        self.decoder.read(len, |keys| {
            output.extend_from_dictionary(keys, dict.offsets.as_slice(), dict.values.as_slice())
        })
    }

    fn skip<I: OffsetSizeTrait>(
        &mut self,
        dict: &OffsetBuffer<I>,
        to_skip: usize,
    ) -> Result<usize> {
        // All data must be NULL
        if dict.is_empty() {
            return Ok(0);
        }

        self.decoder.skip(to_skip)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow::array_reader::test_util::{byte_array_all_encodings, utf8_column};
    use crate::arrow::record_reader::buffer::ValuesBuffer;
    use arrow_array::{Array, StringArray};
    use arrow_buffer::Buffer;

    #[test]
    fn test_byte_array_decoder() {
        let (pages, encoded_dictionary) =
            byte_array_all_encodings(vec!["hello", "world", "a", "b"]);

        let column_desc = utf8_column();
        let mut decoder = ByteArrayColumnValueDecoder::new(&column_desc);

        decoder
            .set_dict(encoded_dictionary, 4, Encoding::RLE_DICTIONARY, false)
            .unwrap();

        for (encoding, page) in pages {
            let mut output = OffsetBuffer::<i32>::with_capacity(0);
            decoder.set_data(encoding, page, 4, Some(4)).unwrap();

            assert_eq!(decoder.read(&mut output, 1).unwrap(), 1);

            assert_eq!(output.values.as_slice(), b"hello");
            assert_eq!(output.offsets.as_slice(), &[0, 5]);

            assert_eq!(decoder.read(&mut output, 1).unwrap(), 1);
            assert_eq!(output.values.as_slice(), b"helloworld");
            assert_eq!(output.offsets.as_slice(), &[0, 5, 10]);

            assert_eq!(decoder.read(&mut output, 2).unwrap(), 2);
            assert_eq!(output.values.as_slice(), b"helloworldab");
            assert_eq!(output.offsets.as_slice(), &[0, 5, 10, 11, 12]);

            assert_eq!(decoder.read(&mut output, 4).unwrap(), 0);

            let valid = [false, false, true, true, false, true, true, false, false];
            let valid_buffer = Buffer::from_iter(valid.iter().copied());

            output
                .pad_nulls(0, 4, valid.len(), valid_buffer.as_slice())
                .unwrap();
            let array = output.into_array(Some(valid_buffer), ArrowType::Utf8);
            let strings = array.as_any().downcast_ref::<StringArray>().unwrap();

            assert_eq!(
                strings.iter().collect::<Vec<_>>(),
                vec![
                    None,
                    None,
                    Some("hello"),
                    Some("world"),
                    None,
                    Some("a"),
                    Some("b"),
                    None,
                    None,
                ]
            );
        }
    }

    #[test]
    fn test_byte_array_decoder_skip() {
        let (pages, encoded_dictionary) =
            byte_array_all_encodings(vec!["hello", "world", "a", "b"]);

        let column_desc = utf8_column();
        let mut decoder = ByteArrayColumnValueDecoder::new(&column_desc);

        decoder
            .set_dict(encoded_dictionary, 4, Encoding::RLE_DICTIONARY, false)
            .unwrap();

        for (encoding, page) in pages {
            let mut output = OffsetBuffer::<i32>::with_capacity(0);
            decoder.set_data(encoding, page, 4, Some(4)).unwrap();

            assert_eq!(decoder.read(&mut output, 1).unwrap(), 1);

            assert_eq!(output.values.as_slice(), b"hello");
            assert_eq!(output.offsets.as_slice(), &[0, 5]);

            assert_eq!(decoder.skip_values(1).unwrap(), 1);
            assert_eq!(decoder.skip_values(1).unwrap(), 1);

            assert_eq!(decoder.read(&mut output, 1).unwrap(), 1);
            assert_eq!(output.values.as_slice(), b"hellob");
            assert_eq!(output.offsets.as_slice(), &[0, 5, 6]);

            assert_eq!(decoder.read(&mut output, 4).unwrap(), 0);

            let valid = [false, false, true, true, false, false];
            let valid_buffer = Buffer::from_iter(valid.iter().copied());

            output
                .pad_nulls(0, 2, valid.len(), valid_buffer.as_slice())
                .unwrap();
            let array = output.into_array(Some(valid_buffer), ArrowType::Utf8);
            let strings = array.as_any().downcast_ref::<StringArray>().unwrap();

            assert_eq!(
                strings.iter().collect::<Vec<_>>(),
                vec![None, None, Some("hello"), Some("b"), None, None,]
            );
        }
    }

    #[test]
    fn test_byte_array_decoder_nulls() {
        let (pages, encoded_dictionary) = byte_array_all_encodings(Vec::<&str>::new());

        let column_desc = utf8_column();
        let mut decoder = ByteArrayColumnValueDecoder::new(&column_desc);

        // The dictionary page holds no values. Upstream declared 4 here; a
        // dictionary page that holds fewer values than it declares is an
        // error (`check_dictionary_len`), so the test declares the 0 values
        // that the page holds. NULL reads, the subject of the test, do not
        // change.
        decoder
            .set_dict(encoded_dictionary, 0, Encoding::RLE_DICTIONARY, false)
            .unwrap();

        // test nulls read
        for (encoding, page) in pages.clone() {
            let mut output = OffsetBuffer::<i32>::with_capacity(0);
            decoder.set_data(encoding, page, 4, None).unwrap();
            assert_eq!(decoder.read(&mut output, 1024).unwrap(), 0);
        }

        // test nulls skip
        for (encoding, page) in pages {
            decoder.set_data(encoding, page, 4, None).unwrap();
            assert_eq!(decoder.skip_values(1024).unwrap(), 0);
        }
    }

    #[test]
    fn test_plain_decoder_reports_values_actually_read() {
        // The page claims to contain two values, but its buffer contains only
        // one complete PLAIN-encoded BYTE_ARRAY value.
        let buffer = Bytes::from_static(&[3, 0, 0, 0, b'f', b'o', b'o']);
        let mut decoder = ByteArrayDecoderPlain::new(buffer, 2, Some(2), false);
        let mut output = OffsetBuffer::<i32>::with_capacity(2);

        assert_eq!(decoder.read(&mut output, 2).unwrap(), 1);
        assert_eq!(output.values.as_slice(), b"foo");
        assert_eq!(output.offsets.as_slice(), &[0, 3]);
        assert_eq!(decoder.max_remaining_values, 1);
    }
}

#[cfg(test)]
mod invalid_input_tests {
    use super::*;

    /// One value of length 100 in a buffer that holds one byte of it.
    fn truncated_value() -> Bytes {
        let mut data = 100u32.to_le_bytes().to_vec();
        data.push(b'a');
        Bytes::from(data)
    }

    /// A skip must not move the offset past the buffer. Before, the next read
    /// subtracted that offset from the buffer length.
    #[test]
    fn plain_skip_past_end_fails() {
        let mut decoder = ByteArrayDecoderPlain::new(truncated_value(), 2, None, false);
        let err = decoder.skip(1).unwrap_err();
        assert_eq!(err.to_string(), "EOF: eof decoding byte array");

        let mut output = OffsetBuffer::<i32>::with_capacity(0);
        let err = decoder.read(&mut output, 1).unwrap_err();
        assert_eq!(err.to_string(), "EOF: eof decoding byte array");
    }

    /// A page can declare zero values and still hold value bytes: the
    /// regression file is a dictionary page. Before, the byte estimate
    /// divided by that count.
    #[test]
    fn plain_read_without_declared_values_reads_nothing() {
        let mut decoder = ByteArrayDecoderPlain::new(truncated_value(), 2, Some(0), false);
        let mut output = OffsetBuffer::<i32>::with_capacity(0);
        assert_eq!(decoder.read(&mut output, 1).unwrap(), 0);
    }
}

#[cfg(test)]
mod allocation_bound_tests {
    use super::*;
    use crate::arrow::array_reader::test_util::utf8_column;

    /// Two PLAIN values: "a" and "bc".
    fn two_values() -> Bytes {
        Bytes::from_static(b"\x01\0\0\0a\x02\0\0\0bc")
    }

    /// The reservation for the offsets followed the value count of the page
    /// header: 2^40 offsets for a page of eleven bytes.
    #[test]
    fn plain_read_reserves_what_the_page_can_hold() {
        let mut decoder = ByteArrayDecoderPlain::new(two_values(), 1 << 40, None, false);
        let mut output = OffsetBuffer::<i32>::with_capacity(0);
        assert_eq!(decoder.read(&mut output, usize::MAX).unwrap(), 2);
        assert_eq!(output.len(), 2);
        assert!(
            output.offsets.capacity() <= 8,
            "{}",
            output.offsets.capacity()
        );
    }

    /// The indices of the data pages can refer to each declared value.
    #[test]
    fn dictionary_with_fewer_values_than_declared_fails() {
        let mut decoder = ByteArrayColumnValueDecoder::<i32>::new(&utf8_column());
        decoder
            .set_dict(two_values(), 2, Encoding::PLAIN, false)
            .unwrap();
        let err = decoder
            .set_dict(two_values(), i32::MAX as u32, Encoding::PLAIN, false)
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "Parquet error: dictionary page declares 2147483647 values, but holds 2"
        );
    }
}
