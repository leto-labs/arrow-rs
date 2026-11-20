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
use arrow_buffer::{Buffer, ScalarBuffer, bit_util::expand};
use bytes::Bytes;

use super::super::parse_v1_level;
use super::{ColumnLevelDecoder, DefinitionLevelDecoder, DefinitionLevelDecoderImpl, get_decoder};
use crate::basic::Encoding;
use crate::column::page::{Page, PageReader};
use crate::data_type::DataType;
use crate::encodings::rle::codes::{
    CodeCursor, CodeRun, RunSink, VerdictTable, WordWriter, load_padded, low_mask, padded_tail,
};
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

    /// Builds a reusable verdict table in this chunk's dictionary order.
    pub fn verdict_table(&self, verdicts: impl IntoIterator<Item = bool>) -> VerdictTable {
        VerdictTable::new(verdicts)
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
    values: Option<usize>,
    levels: Option<Bytes>,
    indices: Bytes,
}

#[derive(Default)]
struct LevelCount(usize);

impl RunSink for LevelCount {
    #[inline]
    fn repeated(&mut self, code: u32, len: usize) {
        self.0 += code as usize * len;
    }

    #[inline]
    fn packed(&mut self, width: u8, bytes: &[u8], len: usize) {
        self.0 += CodeRun::Packed { width, bytes, len }.count_ones(0, len);
    }
}

/// Appends flat definition levels to row-ordered validity words.
struct ValidityWriter<'a> {
    writer: WordWriter<'a>,
    tail: [u8; 16],
}

impl RunSink for ValidityWriter<'_> {
    #[inline(always)]
    fn repeated(&mut self, code: u32, len: usize) {
        // The walk has checked `code < 2`.
        self.writer.repeat(code != 0, len);
    }

    /// `bytes` is the rest of the level stream (`drain_padded`), so each
    /// group of 64 levels is one padded load masked to the run.
    #[inline(always)]
    fn packed(&mut self, width: u8, bytes: &[u8], len: usize) {
        debug_assert_eq!(width, 1);
        let mut offset = 0;
        let mut remaining = len;
        loop {
            let count = remaining.min(64);
            let word = load_padded(bytes, offset, &self.tail);
            self.writer.write(word & low_mask(count), count);
            remaining -= count;
            if remaining == 0 {
                break;
            }
            offset += 8;
        }
    }
}

fn count_valid_levels(levels: &mut CodeCursor<'_>) -> Result<usize> {
    let mut count = LevelCount::default();
    levels.drain(&mut count)?;
    Ok(count.0)
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

    /// Checks the level stream and the non-null count, if declared.
    pub fn validate_levels(&self) -> Result<()> {
        if let Some(mut levels) = self.levels() {
            self.check_count(count_valid_levels(&mut levels)?)?;
        }
        Ok(())
    }

    fn check_count(&self, count: usize) -> Result<()> {
        if self.values.is_some_and(|expected| expected != count) {
            return Err(general_err!(
                "Definition level count disagrees with DataPage V2"
            ));
        }
        Ok(())
    }

    /// The index stream, including only physical non-null values. V1 optional
    /// pages derive this count from their definition levels.
    pub fn indices(&self) -> Result<CodeCursor<'_>> {
        let count = match self.values {
            Some(count) => count,
            None => count_valid_levels(&mut self.levels().expect("optional V1 levels"))?,
        };
        self.indices_with_count(count)
    }

    fn indices_with_count(&self, count: usize) -> Result<CodeCursor<'_>> {
        if count == 0 && self.indices.is_empty() {
            return CodeCursor::new(0, &[], 0, self.dictionary.entries as u64);
        }
        let (&width, bytes) = self
            .indices
            .split_first()
            .ok_or_else(|| general_err!("Missing dictionary index width"))?;
        CodeCursor::new(width, bytes, count, self.dictionary.entries as u64)
    }

    /// Builds a row validity bitmap and returns its non-null count.
    /// Word bits are in row order, starting at the least significant bit.
    /// For required pages the index stream must cover the declared row count.
    pub fn validity(&self, out: &mut Vec<u64>) -> Result<usize> {
        if self.levels.is_none() {
            self.indices_with_count(self.rows)?.validate_extent()?;
        } else if self.rows.div_ceil(64) > out.capacity() {
            self.validate_levels()?;
        }
        self.validity_inline(out)
    }

    #[inline(always)]
    fn validity_inline(&self, out: &mut Vec<u64>) -> Result<usize> {
        let Some(levels) = &self.levels else {
            out.clear();
            out.resize(self.rows.div_ceil(64), u64::MAX);
            if let Some(last) = out.last_mut() {
                *last = low_mask((self.rows - 1) % 64 + 1);
            }
            return Ok(self.rows);
        };
        let mut cursor = CodeCursor::new(1, levels, self.rows, 2)?;
        cursor.prepare_bitmap(out)?;
        let mut sink = ValidityWriter {
            writer: WordWriter::new(out),
            tail: padded_tail(levels),
        };
        cursor.drain_padded(&mut sink)?;
        sink.writer.finish();
        // Packed loads are masked to their runs, so bits past the last row
        // are zero.
        let count = out.iter().map(|word| word.count_ones() as usize).sum();
        self.check_count(count)?;
        Ok(count)
    }

    /// Evaluates the dictionary verdict into a row bitmap with nulls cleared.
    /// `validity` is reusable scratch. Discard `out` on error.
    pub fn verdicts(
        &self,
        table: &mut VerdictTable,
        validity: &mut Vec<u64>,
        out: &mut Vec<u64>,
    ) -> Result<()> {
        // The validity allocation also depends on the index stream being
        // complete. A huge valid level run can precede a truncated index.
        if self.rows.div_ceil(64) > validity.capacity() && self.levels.is_some() {
            self.validate_optional_verdict_extent()?;
        }
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        if fast_bitmaps() {
            // SAFETY: `fast_bitmaps` checked that the CPU has BMI2 and POPCNT.
            return unsafe { self.verdicts_bmi2(table, validity, out) };
        }
        self.verdicts_with(table, validity, out, |bits, mask| {
            if mask == u64::MAX {
                bits
            } else {
                expand(bits, mask)
            }
        })
    }

    #[cold]
    fn validate_optional_verdict_extent(&self) -> Result<()> {
        let count = count_valid_levels(&mut self.levels().expect("optional page"))?;
        self.check_count(count)?;
        self.indices_with_count(count)?.validate_extent()
    }

    /// The whole page path compiled with BMI2 and POPCNT: the level walk,
    /// the index walk, the non-null count and the row expansion.
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    #[target_feature(enable = "bmi2,popcnt")]
    fn verdicts_bmi2(
        &self,
        table: &mut VerdictTable,
        validity: &mut Vec<u64>,
        out: &mut Vec<u64>,
    ) -> Result<()> {
        self.verdicts_with(table, validity, out, |bits, mask| {
            std::arch::x86_64::_pdep_u64(bits, mask)
        })
    }

    #[inline(always)]
    fn verdicts_with(
        &self,
        table: &mut VerdictTable,
        validity: &mut Vec<u64>,
        out: &mut Vec<u64>,
        deposit: impl Fn(u64, u64) -> u64,
    ) -> Result<()> {
        let count = if self.levels.is_some() {
            self.validity_inline(validity)?
        } else {
            self.rows
        };
        self.indices_with_count(count)?
            .verdicts_inline(table, out)?;
        if count == self.rows {
            return Ok(());
        }
        out.resize(validity.len(), 0);
        scatter_verdicts_with(out, validity, count, deposit);
        Ok(())
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
        if rows.next().is_some() {
            return Err(general_err!(
                "Invalid row selection or definition level count"
            ));
        }
        self.check_count(physical)?;
        let keys = self
            .indices_with_count(physical)?
            .selected_keys(physical_offsets)?;
        let mut keys = keys.into_iter();
        let keys = PrimitiveArray::from_iter(
            valid
                .into_iter()
                .map(|valid| valid.then(|| keys.next().expect("one key per selected valid row"))),
        );
        DictionaryArray::try_new(keys, Arc::clone(&self.dictionary.values)).map_err(Into::into)
    }
}

/// Whether to run the BMI2 page path. AVX-512 stands in for a fast `pdep`:
/// AMD before Zen 3 implements it in microcode. Tests select by tier.
#[cfg(all(target_arch = "x86_64", not(miri)))]
fn fast_bitmaps() -> bool {
    let detected = std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("bmi2")
        && std::is_x86_feature_detected!("popcnt");
    #[cfg(test)]
    let detected = detected
        && !matches!(
            crate::util::native::dispatch::tier(),
            crate::util::native::dispatch::Tier::Scalar | crate::util::native::dispatch::Tier::Avx2
        );
    detected
}

#[inline(always)]
fn scatter_verdicts_with(
    out: &mut [u64],
    validity: &[u64],
    count: usize,
    deposit: impl Fn(u64, u64) -> u64,
) {
    // Expand in reverse. A row word can only read dense words at or
    // below its own position, so expansion can reuse the output buffer.
    let mut rank = count;
    for (i, &mask) in validity.iter().enumerate().rev() {
        let len = mask.count_ones() as usize;
        rank -= len;
        if len == 0 {
            out[i] = 0;
            continue;
        }
        let word = rank / 64;
        let shift = rank % 64;
        let mut bits = out[word] >> shift;
        if len > 64 - shift {
            bits |= out[word + 1] << (64 - shift);
        }
        out[i] = deposit(bits, mask);
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
                    num_rows,
                    def_levels_byte_len,
                    rep_levels_byte_len,
                    ..
                }) => {
                    // Flat columns have exactly one level slot per row.
                    // See DataPageHeaderV2 in the Parquet Thrift specification.
                    if num_rows != num_values {
                        return Err(general_err!("Flat DataPage V2 row/value counts differ"));
                    }
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
            if self.dictionary.is_none()
                && let Some(page) = self.reader.take_deferred_dictionary()?
            {
                self.install_dictionary(page)?;
            }
            let dictionary = self
                .dictionary
                .clone()
                .ok_or_else(|| general_err!("Missing column chunk dictionary"))?;
            if level_encoding == Encoding::RLE || levels.is_none() {
                let values = non_null.or_else(|| levels.is_none().then_some(rows));
                let page = DictionaryCodePage {
                    dictionary,
                    rows,
                    values,
                    levels,
                    indices: data,
                };
                // Check the encoding width now. For V1 optional pages, an
                // empty payload is only known to be valid after reading levels.
                page.indices_with_count(values.unwrap_or(0))?;
                return Ok(Some(CodeColumnPage::Dictionary(page)));
            }
            // The legacy BIT_PACKED level encoding falls back to value space.
            let (levels, count) = decode_levels(levels, level_encoding, rows)?;
            let page = DictionaryCodePage {
                dictionary,
                rows,
                values: Some(count),
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
    #[expect(deprecated, reason = "Read legacy V1 definition levels")]
    if encoding == Encoding::BIT_PACKED {
        // The deprecated bit-packing starts at the MSB, unlike hybrid RLE.
        // https://parquet.apache.org/docs/file-format/data-pages/encodings/#bit-packed-deprecated-bit_packed--4
        if levels.len() < rows.div_ceil(8) {
            return Err(general_err!("Incomplete BIT_PACKED definition levels"));
        }
        let decoded: Vec<i16> = (0..rows)
            .map(|row| i16::from((levels[row / 8] >> (7 - row % 8)) & 1))
            .collect();
        let values = decoded.iter().map(|&level| level as usize).sum();
        return Ok((Some(decoded), values));
    }
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
        flat_data_page(v2, definition, Some(Encoding::RLE), encoding, data)
    }

    #[expect(deprecated, reason = "Write legacy V1 BIT_PACKED test pages")]
    fn flat_data_page(
        v2: bool,
        definition: &[bool],
        level_encoding: Option<Encoding>,
        encoding: Encoding,
        data: impl IntoIterator<Item = u8>,
    ) -> Page {
        let levels = match level_encoding {
            None => Vec::new(),
            Some(Encoding::BIT_PACKED) => {
                assert!(!v2);
                definition
                    .chunks(8)
                    .map(|chunk| {
                        chunk.iter().enumerate().fold(0, |byte, (bit, &valid)| {
                            byte | (u8::from(valid) << (7 - bit))
                        })
                    })
                    .collect()
            }
            Some(Encoding::RLE) => {
                let mut levels = RleEncoder::new(1, 0);
                for &valid in definition {
                    levels.put(u64::from(valid));
                }
                levels.consume()
            }
            _ => unreachable!(),
        };
        let rows = definition.len() as u32;
        let mut buf = Vec::new();
        if !v2 && level_encoding == Some(Encoding::RLE) {
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
                def_level_encoding: level_encoding.unwrap_or(Encoding::RLE),
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
                    let indices: Vec<_> = std::iter::once(width).chain(indices.consume()).collect();
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
                                let expected = &expected[row..row + page.num_rows()];
                                let mut table = page.dictionary().verdict_table([false, true]);
                                let mut valid = Vec::new();
                                let mut verdicts = Vec::new();
                                assert_eq!(
                                    page.validity(&mut valid).unwrap(),
                                    expected.iter().flatten().count()
                                );
                                page.verdicts(&mut table, &mut valid, &mut verdicts)
                                    .unwrap();
                                for (i, &value) in expected.iter().enumerate() {
                                    assert_eq!(
                                        (valid[i / 64] >> (i % 64)) & 1,
                                        u64::from(value.is_some())
                                    );
                                    assert_eq!(
                                        (verdicts[i / 64] >> (i % 64)) & 1,
                                        u64::from(value == Some(43))
                                    );
                                }
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
    #[expect(
        deprecated,
        reason = "Compare legacy V1 BIT_PACKED pages to expected values"
    )]
    fn code_page_row_bitmaps_match_levels_for_random_null_patterns() {
        crate::util::native::dispatch::for_each_tier(|| {
            let mut seed = 0x9e37_79b9_u32;
            let mut next = |bound: u32| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed % bound
            };
            let mut validity = Vec::new();
            let mut verdicts = Vec::new();
            let required = SchemaDescriptor::new(Arc::new(
                parse_message_type("message x { REQUIRED INT32 value; }").unwrap(),
            ))
            .column(0);
            let optional = descriptor();
            for case in 0..400 {
                let (v2, level_encoding) = [
                    (false, None),
                    (true, None),
                    (false, Some(Encoding::RLE)),
                    (true, Some(Encoding::RLE)),
                    (false, Some(Encoding::BIT_PACKED)),
                ][case % 5];
                let desc = if level_encoding.is_some() {
                    &optional
                } else {
                    &required
                };
                let rows = match case / 5 {
                    0 => 0,
                    1 => 1,
                    2 => 64,
                    3 => 65,
                    _ => 1 + next(if case % 4 == 0 { 700 } else { 140 }) as usize,
                };
                // Mix scattered nulls (packed level runs at every offset) with
                // long valid and null runs (repeated runs, some over 64).
                let density = if level_encoding.is_some() { next(5) } else { 0 };
                let mut definition = Vec::with_capacity(rows);
                while definition.len() < rows {
                    let long = next(3) == 0;
                    let len =
                        (1 + next(if long { 90 } else { 9 }) as usize).min(rows - definition.len());
                    let valid = next(5) >= density;
                    definition.extend(std::iter::repeat_n(valid, len));
                }
                let valid_count = definition.iter().filter(|&&valid| valid).count();
                let entries = 1 + next(40) as usize;
                let mut width = 1u8;
                while 1usize << width < entries {
                    width += 1;
                }
                let mut codes = Vec::with_capacity(valid_count);
                while codes.len() < valid_count {
                    let code = next(entries as u32);
                    let len = (1 + next(14) as usize).min(valid_count - codes.len());
                    codes.extend(std::iter::repeat_n(code, len));
                }
                let mut encoder = RleEncoder::new(width, 0);
                for &code in &codes {
                    encoder.put(u64::from(code));
                }
                let indices: Vec<_> = std::iter::once(width).chain(encoder.consume()).collect();
                let dictionary_values: Vec<i32> = (0..entries as i32).collect();
                let page = flat_data_page(
                    v2,
                    &definition,
                    level_encoding,
                    Encoding::RLE_DICTIONARY,
                    indices,
                );
                let mut mismatched = page.clone();
                if let Page::DataPageV2 { num_rows, .. } = &mut mismatched {
                    let wrong_count = next(*num_rows + 7);
                    *num_rows = wrong_count + u32::from(wrong_count >= *num_rows);
                    let mut reader = CodeColumnReader::<Int32Type, ArrowInt32>::new(
                        desc.clone(),
                        Box::new(InMemoryPageReader::new([
                            dictionary(&dictionary_values),
                            mismatched,
                        ])),
                    )
                    .unwrap();
                    assert!(reader.next_page().is_err(), "case {case}");
                }
                let mut reader = CodeColumnReader::<Int32Type, ArrowInt32>::new(
                    desc.clone(),
                    Box::new(InMemoryPageReader::new([
                        dictionary(&dictionary_values),
                        page,
                    ])),
                )
                .unwrap();
                let page = reader.next_page().unwrap().unwrap();
                if level_encoding == Some(Encoding::BIT_PACKED) {
                    let CodeColumnPage::Values { values, .. } = page else {
                        panic!("expected legacy levels to fall back to values")
                    };
                    let mut codes = codes.iter();
                    let expected = definition.iter().map(|&valid| {
                        valid.then(|| dictionary_values[*codes.next().unwrap() as usize])
                    });
                    assert!(values.iter().eq(expected), "case {case}");
                    continue;
                }
                let CodeColumnPage::Dictionary(page) = page else {
                    panic!("expected dictionary page")
                };
                let pass = |code: u32| code % 3 == 1;
                let mut table = page
                    .dictionary()
                    .verdict_table((0..entries as u32).map(pass));
                assert_eq!(page.validity(&mut validity).unwrap(), valid_count);
                page.verdicts(&mut table, &mut validity, &mut verdicts)
                    .unwrap();
                assert_eq!(validity.len(), rows.div_ceil(64));
                assert_eq!(verdicts.len(), rows.div_ceil(64));
                let mut codes = codes.iter();
                for (row, &valid) in definition.iter().enumerate() {
                    let bit = |words: &[u64]| (words[row / 64] >> (row % 64)) & 1;
                    assert_eq!(bit(&validity), u64::from(valid), "case {case} row {row}");
                    let expected = valid && pass(*codes.next().unwrap());
                    assert_eq!(bit(&verdicts), u64::from(expected), "case {case} row {row}");
                }
                if let Some(last) = rows.checked_sub(1) {
                    let unused = !low_mask(last % 64 + 1);
                    assert_eq!(validity.last().unwrap() & unused, 0, "case {case}");
                    assert_eq!(verdicts.last().unwrap() & unused, 0, "case {case}");
                }
            }
        });
    }

    #[test]
    fn code_page_verdicts_check_deferred_level_and_index_counts() {
        let mut v2 = data_page(true, &[true, false], Encoding::RLE_DICTIONARY, [0, 2]);
        let Page::DataPageV2 { num_nulls, .. } = &mut v2 else {
            unreachable!()
        };
        *num_nulls = 0;
        let v1 = data_page(false, &[true], Encoding::RLE_DICTIONARY, []);
        let pages = [dictionary(&[17]), v2, v1];
        let mut reader = CodeColumnReader::<Int32Type, ArrowInt32>::new(
            descriptor(),
            Box::new(InMemoryPageReader::new(pages)),
        )
        .unwrap();
        let mut validity = Vec::new();
        let mut verdicts = Vec::new();
        for _ in 0..2 {
            let Some(CodeColumnPage::Dictionary(page)) = reader.next_page().unwrap() else {
                panic!("expected dictionary page")
            };
            let mut table = page.dictionary().verdict_table([true]);
            assert!(
                page.verdicts(&mut table, &mut validity, &mut verdicts)
                    .is_err()
            );
        }
    }

    #[test]
    #[expect(deprecated, reason = "Exercise the Parquet V1 BIT_PACKED encoding")]
    fn code_page_legacy_levels_are_msb_first() {
        crate::util::native::dispatch::for_each_tier(|| {
            // The specification's BIT_PACKED encoding starts in the high bit.
            // There is no size prefix in V1. The two levels are 1 and 0.
            for (encoding, payload) in [
                (Encoding::PLAIN, &[0x80, 17, 0, 0, 0][..]),
                (Encoding::RLE_DICTIONARY, &[0x80, 0, 2]),
            ] {
                let page = Page::DataPage {
                    buf: Bytes::copy_from_slice(payload),
                    num_values: 2,
                    encoding,
                    def_level_encoding: Encoding::BIT_PACKED,
                    rep_level_encoding: Encoding::RLE,
                    statistics: None,
                };
                let mut reader = CodeColumnReader::<Int32Type, ArrowInt32>::new(
                    descriptor(),
                    Box::new(InMemoryPageReader::new([dictionary(&[17]), page])),
                )
                .unwrap();
                let Some(CodeColumnPage::Values { values, .. }) = reader.next_page().unwrap()
                else {
                    panic!("expected a value page")
                };
                assert_eq!(values.iter().collect::<Vec<_>>(), [Some(17), None]);
            }
        });
    }

    #[test]
    fn code_page_truncated_levels_do_not_allocate_for_declared_count() {
        crate::util::native::dispatch::for_each_tier(|| {
            // V1 RLE levels have a four-byte length. An empty level stream
            // cannot cover the declared rows. The index width is zero.
            let page = Page::DataPage {
                buf: Bytes::from_static(&[0, 0, 0, 0, 0]),
                num_values: i32::MAX as u32,
                encoding: Encoding::RLE_DICTIONARY,
                def_level_encoding: Encoding::RLE,
                rep_level_encoding: Encoding::RLE,
                statistics: None,
            };
            let mut reader = CodeColumnReader::<Int32Type, ArrowInt32>::new(
                descriptor(),
                Box::new(InMemoryPageReader::new([dictionary(&[17]), page])),
            )
            .unwrap();
            let Some(CodeColumnPage::Dictionary(page)) = reader.next_page().unwrap() else {
                panic!("expected a dictionary page")
            };
            let mut out = Vec::new();
            assert!(page.validity(&mut out).is_err());
            assert_eq!(out.capacity(), 0);
        });
    }

    #[test]
    fn code_page_truncated_indices_do_not_grow_validity() {
        crate::util::native::dispatch::for_each_tier(|| {
            // The level header 2 * i32::MAX declares one all-valid run.
            // The index stream contains its width but no runs.
            let page = Page::DataPage {
                buf: Bytes::from_static(&[6, 0, 0, 0, 0xfe, 0xff, 0xff, 0xff, 0x0f, 1, 0]),
                num_values: i32::MAX as u32,
                encoding: Encoding::RLE_DICTIONARY,
                def_level_encoding: Encoding::RLE,
                rep_level_encoding: Encoding::RLE,
                statistics: None,
            };
            let mut reader = CodeColumnReader::<Int32Type, ArrowInt32>::new(
                descriptor(),
                Box::new(InMemoryPageReader::new([dictionary(&[17]), page])),
            )
            .unwrap();
            let Some(CodeColumnPage::Dictionary(page)) = reader.next_page().unwrap() else {
                panic!("expected a dictionary page")
            };
            let mut validity = Vec::new();
            let mut verdicts = Vec::new();
            let mut table = page.dictionary().verdict_table([true]);
            assert!(
                page.verdicts(&mut table, &mut validity, &mut verdicts)
                    .is_err()
            );
            assert_eq!(validity.capacity(), 0);
            assert_eq!(verdicts.capacity(), 0);
        });
    }

    #[test]
    fn code_page_rejects_extra_index_runs() {
        crate::util::native::dispatch::for_each_tier(|| {
            // One valid row in V2, then one dictionary index at width 1.
            // The second run declares a key outside the one-entry dictionary.
            let page = Page::DataPageV2 {
                buf: Bytes::from_static(&[2, 1, 1, 2, 0, 2, 1]),
                num_values: 1,
                encoding: Encoding::RLE_DICTIONARY,
                num_nulls: 0,
                num_rows: 1,
                def_levels_byte_len: 2,
                rep_levels_byte_len: 0,
                is_compressed: false,
                statistics: None,
            };
            let mut reader = CodeColumnReader::<Int32Type, ArrowInt32>::new(
                descriptor(),
                Box::new(InMemoryPageReader::new([dictionary(&[17]), page])),
            )
            .unwrap();
            let Some(CodeColumnPage::Dictionary(page)) = reader.next_page().unwrap() else {
                panic!("expected a dictionary page")
            };
            let mut cursor = page.indices().unwrap();
            assert_eq!(cursor.next_run().unwrap().unwrap().code_at(0), Some(0));
            assert!(cursor.next_run().is_err());
            assert!(page.selected_keys([0]).is_err());
            let mut table = page.dictionary().verdict_table([true]);
            assert!(
                page.verdicts(&mut table, &mut vec![0], &mut vec![0])
                    .is_err()
            );
        });
    }

    #[test]
    fn code_page_rejects_inconsistent_v2_row_count() {
        for nullable in [false, true] {
            let modifier = if nullable { "OPTIONAL" } else { "REQUIRED" };
            let schema =
                parse_message_type(&format!("message x {{ {modifier} INT32 value; }}")).unwrap();
            let descriptor = SchemaDescriptor::new(Arc::new(schema)).column(0);
            for (encoding, data) in [
                (Encoding::RLE_DICTIONARY, &[0, 4][..]),
                (Encoding::PLAIN, &[17, 0, 0, 0, 17, 0, 0, 0]),
            ] {
                for num_rows in [1, 3] {
                    let levels: &[u8] = if nullable { &[4, 1] } else { &[] };
                    let page = Page::DataPageV2 {
                        buf: levels
                            .iter()
                            .chain(data)
                            .copied()
                            .collect::<Vec<_>>()
                            .into(),
                        num_values: 2,
                        encoding,
                        num_nulls: 0,
                        num_rows,
                        def_levels_byte_len: levels.len() as u32,
                        rep_levels_byte_len: 0,
                        is_compressed: false,
                        statistics: None,
                    };
                    let mut reader = CodeColumnReader::<Int32Type, ArrowInt32>::new(
                        descriptor.clone(),
                        Box::new(InMemoryPageReader::new([dictionary(&[17]), page])),
                    )
                    .unwrap();
                    assert!(reader.next_page().is_err());
                }
            }
        }
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
            let mut table = page.dictionary().verdict_table([]);
            let mut valid = Vec::new();
            let mut verdicts = Vec::new();
            page.verdicts(&mut table, &mut valid, &mut verdicts)
                .unwrap();
            assert_eq!(valid, vec![0; page.num_rows().div_ceil(64)]);
            assert_eq!(verdicts, valid);
            let Some(CodeColumnPage::Dictionary(page)) = reader.next_page().unwrap() else {
                panic!("expected dictionary page")
            };
            assert!(page.selected_keys([]).is_err());
        }
    }
}
