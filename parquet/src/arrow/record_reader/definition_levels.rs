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

use arrow_array::builder::BooleanBufferBuilder;
use arrow_buffer::Buffer;
use arrow_buffer::bit_chunk_iterator::UnalignedBitChunk;
use arrow_buffer::bit_util::compress;
use bytes::Bytes;
use std::sync::Arc;

use crate::arrow::buffer::bit_util::count_set_bits;
use crate::basic::Encoding;
use crate::column::page::Page;
use crate::column::reader::decoder::{
    BitPackedLevelDecoder, ColumnLevelDecoder, DefinitionLevelDecoder, DefinitionLevelDecoderImpl,
};
use crate::column::reader::parse_v1_level;
use crate::encodings::rle::run_len;
use crate::errors::{ParquetError, Result};
use crate::file::page_cache::CachedValidity;
use crate::schema::types::ColumnDescPtr;

enum BufferInner {
    /// Compute levels and null mask
    Full {
        levels: Vec<i16>,
        nulls: BooleanBufferBuilder,
        max_level: i16,
    },
    /// Only compute null bitmask - requires max level to be 1
    ///
    /// This is an optimisation for the common case of a nullable scalar column, as decoding
    /// the definition level data is only required when decoding nested structures
    ///
    Mask { nulls: BooleanBufferBuilder },
}

pub struct DefinitionLevelBuffer {
    inner: BufferInner,

    /// The length of this buffer
    ///
    /// Note: `buffer` and `builder` may contain more elements
    len: usize,
}

impl DefinitionLevelBuffer {
    pub fn new(desc: &ColumnDescPtr, null_mask_only: bool) -> Self {
        let inner = match null_mask_only {
            true => {
                assert_eq!(
                    desc.max_def_level(),
                    1,
                    "max definition level must be 1 to only compute null bitmask"
                );

                assert_eq!(
                    desc.max_rep_level(),
                    0,
                    "max repetition level must be 0 to only compute null bitmask"
                );

                BufferInner::Mask {
                    nulls: BooleanBufferBuilder::new(0),
                }
            }
            false => BufferInner::Full {
                levels: Vec::new(),
                nulls: BooleanBufferBuilder::new(0),
                max_level: desc.max_def_level(),
            },
        };

        Self { inner, len: 0 }
    }

    /// Returns the built level data
    pub fn consume_levels(&mut self) -> Option<Vec<i16>> {
        match &mut self.inner {
            BufferInner::Full { levels, .. } => Some(std::mem::take(levels)),
            BufferInner::Mask { .. } => None,
        }
    }

    /// Returns the built null bitmask, or None if all values are valid
    pub fn consume_bitmask(&mut self) -> Option<Buffer> {
        self.len = 0;
        let nulls = match &mut self.inner {
            BufferInner::Full { nulls, .. } => nulls,
            BufferInner::Mask { nulls } => nulls,
        };

        // Always call finish() to reset the builder state for the next batch.
        let buffer = nulls.finish().into_inner();

        // If no bitmap was constructed, return None
        if buffer.is_empty() {
            return None;
        }

        Some(buffer)
    }

    pub fn nulls(&self) -> &BooleanBufferBuilder {
        match &self.inner {
            BufferInner::Full { nulls, .. } => nulls,
            BufferInner::Mask { nulls } => nulls,
        }
    }

    /// Returns the raw definition levels accumulated so far, if available.
    /// Only available when the buffer is in Full mode (nested columns).
    pub fn levels(&self) -> Option<&[i16]> {
        match &self.inner {
            BufferInner::Full { levels, .. } => Some(levels.as_slice()),
            BufferInner::Mask { .. } => None,
        }
    }
}

/// Build a filtered validity bitmap from definition/repetition levels.
///
/// For each level entry where `d >= include_threshold` (when set) and
/// `r <= max_rep` (when `rep_filter` is provided), appends one bit to
/// `bitmap`: set when `d >= value_level`, unset otherwise.
///
/// Returns the number of bits appended.
///
/// This is the shared implementation behind the compact bitmap (leaf
/// readers with selective padding) and the struct validity bitmap.
/// Processing uses 64-level chunks with [`compress`] for word-at-a-time
/// packing.
pub(crate) fn build_filtered_validity_bitmap(
    def_levels: &[i16],
    rep_filter: Option<(&[i16], i16)>,
    include_threshold: Option<i16>,
    value_level: i16,
    bitmap: &mut BooleanBufferBuilder,
) -> usize {
    const CHUNKS: usize = u64::BITS as usize;
    // Fast path: no filtering — every def level produces a bit.
    if include_threshold.is_none() && rep_filter.is_none() {
        let (chunks, remainder) = def_levels.as_chunks::<CHUNKS>();
        for chunk in chunks {
            let mut word: u64 = 0;
            for (i, &d) in chunk.iter().enumerate() {
                word |= ((d >= value_level) as u64) << i;
            }
            bitmap.append_word(word, u64::BITS as usize);
        }
        for &d in remainder {
            bitmap.append(d >= value_level);
        }
        return def_levels.len();
    }

    // Filtered path: build include mask + value mask per chunk, compress.
    let mut item_count: usize = 0;
    let (chunks, remainder) = def_levels.as_chunks::<CHUNKS>();
    let remainder_offset = def_levels.len() - remainder.len();

    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        let base = chunk_idx * u64::BITS as usize;
        let mut include_mask: u64 = 0;
        let mut value_mask: u64 = 0;
        for (i, &d) in chunk.iter().enumerate() {
            let include = include_threshold.is_none_or(|threshold| threshold <= d)
                && rep_filter.is_none_or(|(reps, max_rep)| reps[base + i] <= max_rep);
            include_mask |= (include as u64) << i;
            value_mask |= ((d >= value_level) as u64) << i;
        }

        let count = include_mask.count_ones() as usize;
        let compact = compress(value_mask, include_mask);
        bitmap.append_word(compact, count);
        item_count += count;
    }

    for idx in remainder_offset..def_levels.len() {
        let d = def_levels[idx];
        if let Some(threshold) = include_threshold
            && d < threshold
        {
            continue;
        }
        if let Some((reps, max_rep)) = rep_filter
            && reps[idx] > max_rep
        {
            continue;
        }
        bitmap.append(d >= value_level);
        item_count += 1;
    }

    item_count
}

/// Materialize the exact number of levels declared by an eligible page.
pub(crate) fn cache_validity(page: &Page) -> Option<CachedValidity> {
    let (data, len, header_nulls) = match page {
        Page::DataPage {
            buf,
            num_values,
            def_level_encoding: Encoding::RLE,
            ..
        } => {
            let (_, data) = parse_v1_level(1, *num_values, Encoding::RLE, buf.clone()).ok()?;
            (data, *num_values as usize, None)
        }
        Page::DataPageV2 {
            buf,
            num_values,
            num_nulls,
            def_levels_byte_len,
            rep_levels_byte_len: 0,
            ..
        } if *def_levels_byte_len as usize <= buf.len() => (
            buf.slice(..*def_levels_byte_len as usize),
            *num_values as usize,
            Some(*num_nulls as usize),
        ),
        _ => return None,
    };

    let mut decoder = PackedDecoder::new();
    decoder.set_data(data);
    // The fill ticket already admitted this allocation and the retained copy.
    // PackedDecoder writes complete words and needs eight bytes past the end.
    let mut bitmap = BooleanBufferBuilder::new(len + 64);
    let (read, valid) = decoder.read(&mut bitmap, len).ok()?;
    if read != len {
        return None;
    }
    let nulls = len - valid;
    if header_nulls.is_some_and(|count| count != nulls) {
        return None;
    }
    Some(CachedValidity {
        bits: Bytes::from(bitmap.as_slice().to_vec()),
        len,
        nulls,
    })
}

struct ValidityCursor {
    data: Arc<CachedValidity>,
    offset: usize,
    remaining_valid: usize,
}

impl ValidityCursor {
    fn consume(&mut self, len: usize, output: Option<&mut BooleanBufferBuilder>) -> (usize, usize) {
        let remaining = self.data.len - self.offset;
        let len = len.min(remaining);
        let range = self.offset..self.offset + len;
        let valid = if len == remaining {
            self.remaining_valid
        } else if self.remaining_valid == 0 || self.remaining_valid == remaining {
            len.min(self.remaining_valid)
        } else {
            UnalignedBitChunk::new(&self.data.bits, range.start, len).count_ones()
        };
        if let Some(output) = output {
            if valid == len || valid == 0 {
                output.append_n(len, valid != 0);
            } else {
                output.append_packed_range(range, &self.data.bits);
            }
        }
        self.offset += len;
        self.remaining_valid -= valid;
        (valid, len)
    }
}

enum MaybePacked {
    Cached(ValidityCursor),
    Packed(PackedDecoder),
    Legacy(BitPackedLevelDecoder),
    Fallback(DefinitionLevelDecoderImpl),
}

pub struct DefinitionLevelBufferDecoder {
    max_level: i16,
    decoder: MaybePacked,
}

impl DefinitionLevelBufferDecoder {
    pub fn new(max_level: i16, packed: bool) -> Self {
        let decoder = match packed {
            true => MaybePacked::Packed(PackedDecoder::new()),
            false => MaybePacked::Fallback(DefinitionLevelDecoderImpl::new(max_level)),
        };

        Self { max_level, decoder }
    }
}

impl ColumnLevelDecoder for DefinitionLevelBufferDecoder {
    type Buffer = DefinitionLevelBuffer;

    fn set_data(&mut self, encoding: Encoding, data: Bytes) -> Result<()> {
        match &mut self.decoder {
            MaybePacked::Fallback(d) => d.set_data(encoding, data)?,
            #[expect(deprecated)]
            _ if encoding == Encoding::BIT_PACKED => {
                self.decoder = MaybePacked::Legacy(BitPackedLevelDecoder::new(data, 1));
            }
            _ if encoding != Encoding::RLE => {
                return Err(general_err!("invalid level encoding: {}", encoding));
            }
            MaybePacked::Packed(d) => d.set_data(data),
            MaybePacked::Cached(_) | MaybePacked::Legacy(_) => {
                let mut decoder = PackedDecoder::new();
                decoder.set_data(data);
                self.decoder = MaybePacked::Packed(decoder);
            }
        }
        Ok(())
    }
}

impl DefinitionLevelDecoder for DefinitionLevelBufferDecoder {
    fn set_data_with_validity(
        &mut self,
        encoding: Encoding,
        data: Bytes,
        validity: Option<Arc<CachedValidity>>,
    ) -> Result<()> {
        if let Some(data) = validity
            && self.max_level == 1
            && !matches!(self.decoder, MaybePacked::Fallback(_))
        {
            self.decoder = MaybePacked::Cached(ValidityCursor {
                offset: 0,
                remaining_valid: data.len - data.nulls,
                data,
            });
            return Ok(());
        }
        self.set_data(encoding, data)
    }

    fn set_light_validity(&mut self, page: &crate::file::page_cache::LightDataPage) -> Result<()> {
        let bytes = if matches!(self.decoder, MaybePacked::Fallback(_)) {
            page.encoded_levels()
        } else {
            Bytes::new()
        };
        self.set_data_with_validity(Encoding::RLE, bytes, page.validity.clone())
    }

    fn read_def_levels(
        &mut self,
        writer: &mut Self::Buffer,
        num_levels: usize,
    ) -> Result<(usize, usize)> {
        match (&mut writer.inner, &mut self.decoder) {
            (
                BufferInner::Full {
                    levels,
                    nulls,
                    max_level,
                },
                MaybePacked::Fallback(decoder),
            ) => {
                assert_eq!(self.max_level, *max_level);

                let start = levels.len();
                let (values_read, levels_read) = decoder.read_def_levels(levels, num_levels)?;

                if values_read == levels_read {
                    // Every level is at `max_level`
                    nulls.append_n(levels_read, true);
                } else if values_read == 0 {
                    // No level is at `max_level`
                    nulls.append_n(levels_read, false);
                } else {
                    // Safety: slice iterator has a trusted length
                    unsafe {
                        nulls.extend_trusted_len(
                            levels[start..].iter().map(|level| level == max_level),
                        );
                    }
                }

                Ok((values_read, levels_read))
            }
            (BufferInner::Mask { nulls }, MaybePacked::Cached(cursor)) => {
                Ok(cursor.consume(num_levels, Some(nulls)))
            }
            (BufferInner::Mask { nulls }, MaybePacked::Legacy(decoder)) => {
                let levels = decoder.take_levels(num_levels);
                let levels_read = levels.len();
                let mut values_read = 0;
                for level in levels {
                    let valid = level == 1;
                    nulls.append(valid);
                    values_read += usize::from(valid);
                }
                Ok((values_read, levels_read))
            }
            (BufferInner::Mask { nulls }, MaybePacked::Packed(decoder)) => {
                assert_eq!(self.max_level, 1);

                // Fast path: if all requested levels are valid (max definition level),
                // we can skip RLE decoding and just append all-ones to the bitmap.
                // This is faster than decoding RLE data.
                if let Some(count) = decoder.try_consume_all_valid(num_levels)? {
                    nulls.append_n(count, true);
                    return Ok((count, count)); // values_read == levels_read when all valid
                }

                // Normal path: decode RLE data into the bitmap, counting the
                // set bits as they are written.
                let start = nulls.len();
                let (levels_read, values_read) = decoder.read(nulls, num_levels)?;
                debug_assert_eq!(
                    values_read,
                    count_set_bits(nulls.as_slice(), start..start + levels_read)
                );
                Ok((values_read, levels_read))
            }
            _ => unreachable!("inconsistent null mask"),
        }
    }

    fn skip_def_levels(&mut self, num_levels: usize) -> Result<(usize, usize)> {
        match &mut self.decoder {
            MaybePacked::Cached(cursor) => Ok(cursor.consume(num_levels, None)),
            MaybePacked::Fallback(decoder) => decoder.skip_def_levels(num_levels),
            MaybePacked::Packed(decoder) => decoder.skip(num_levels),
            MaybePacked::Legacy(decoder) => Ok(decoder.skip_and_count(num_levels, 1)),
        }
    }
}

/// Little-endian word of `data` at byte `byte`, zero past the end.
#[inline(always)]
fn load_word(data: &[u8], byte: usize) -> u64 {
    match data.get(byte..byte + 8) {
        Some(word) => u64::from_le_bytes(word.try_into().unwrap()),
        None => {
            let mut word = [0u8; 8];
            let tail = data.get(byte..).unwrap_or(&[]);
            word[..tail.len()].copy_from_slice(tail);
            u64::from_le_bytes(word)
        }
    }
}

/// Appends bits to a zeroed bitmap through one register word. Only whole
/// little-endian words at multiples of eight bytes are stored, so a run never
/// reloads bytes that the previous run stored.
struct Appender<'a> {
    bits: &'a mut [u8],
    /// Byte offset of the word in `word`.
    byte: usize,
    word: u64,
    /// Bits of `word` in use.
    used: usize,
}

impl<'a> Appender<'a> {
    /// `bits` holds the bitmap and at least eight bytes after bit `pos`.
    #[inline(always)]
    fn new(bits: &'a mut [u8], pos: usize) -> Self {
        let byte = pos / 64 * 8;
        let word = u64::from_le_bytes(bits[byte..byte + 8].try_into().unwrap());
        let used = pos % 64;
        Self {
            bits,
            byte,
            word: word & ((1u64 << used) - 1),
            used,
        }
    }

    /// Appends the low `count` bits of `value`, at most 64. Branch-free: the
    /// current word is stored every time (a later store of the same word
    /// replaces it), and the bits that did not fit start the next word.
    #[inline(always)]
    fn push(&mut self, value: u64, count: usize) {
        debug_assert!(count <= 64 && self.used < 64);
        let value =
            value & (u64::MAX >> ((64 - count) & 63)) & 0u64.wrapping_sub((count != 0) as u64);
        let word = self.word | (value << self.used);
        self.bits[self.byte..self.byte + 8].copy_from_slice(&word.to_le_bytes());
        let total = self.used + count;
        // `value >> (64 - used)`, zero when used is zero (a 64-bit value then
        // fills the word exactly).
        let carry = ((value >> 1) >> (63 - self.used)) & 0u64.wrapping_sub((self.used != 0) as u64);
        let next = total >= 64;
        self.word = if next { carry } else { word };
        self.byte += (next as usize) * 8;
        self.used = total & 63;
    }

    #[inline(always)]
    fn ones(&mut self, mut count: usize) {
        while count > 64 {
            self.push(u64::MAX, 64);
            count -= 64;
        }
        self.push(u64::MAX, count);
    }

    #[inline(always)]
    fn zeros(&mut self, mut count: usize) {
        while count > 64 {
            self.push(0, 64);
            count -= 64;
        }
        self.push(0, count);
    }

    /// Appends `count` bits of `src` from bit `from`. Returns how many are set.
    #[inline(always)]
    fn copy(&mut self, src: &[u8], mut from: usize, mut count: usize) -> usize {
        let mut set = 0;
        while count > 0 {
            let n = count.min(56);
            let value = (load_word(src, from / 8) >> (from % 8)) & ((1u64 << n) - 1);
            self.push(value, n);
            set += value.count_ones() as usize;
            from += n;
            count -= n;
        }
        set
    }

    /// Stores the partial word.
    #[inline(always)]
    fn finish(self) {
        self.bits[self.byte..self.byte + 8].copy_from_slice(&self.word.to_le_bytes());
    }
}

/// An optimized decoder for decoding [RLE] data with a bit width of 1
/// directly into a bitmask
///
/// This is significantly faster than decoding the data into `[i16]` and then computing
/// a bitmask from this, as not only can it skip this buffer allocation and construction,
/// but it can exploit properties of the encoded data to reduce work further
///
/// In particular:
///
/// * Packed runs are already bitmask encoded and can simply be appended
/// * Runs of 1 or 0 bits can be efficiently appended with byte (or larger) operations
///
/// [RLE]: https://github.com/apache/parquet-format/blob/master/Encodings.md#run-length-encoding--bit-packing-hybrid-rle--3
struct PackedDecoder {
    data: Bytes,
    data_offset: usize,
    rle_left: usize,
    rle_value: bool,
    packed_count: usize,
    packed_offset: usize,
}

impl PackedDecoder {
    fn next_rle_block(&mut self) -> Result<()> {
        let indicator_value = self.decode_header()?;
        let num_values = run_len(indicator_value as u64)? as usize;
        if indicator_value & 1 == 1 {
            self.packed_count = num_values;
            self.packed_offset = 0;
        } else {
            self.rle_left = num_values;
            let byte = *self.data.as_ref().get(self.data_offset).ok_or_else(|| {
                ParquetError::EOF(
                    "unexpected end of file whilst decoding definition levels rle value".into(),
                )
            })?;

            self.data_offset += 1;
            self.rle_value = byte != 0;
        }
        Ok(())
    }

    /// Decodes a VLQ encoded little endian integer and returns it
    #[inline]
    fn decode_header(&mut self) -> Result<i64> {
        // Runs shorter than 64 levels have a one-byte header.
        if let Some(&byte) = self.data.as_ref().get(self.data_offset)
            && byte < 0x80
        {
            self.data_offset += 1;
            return Ok(byte as i64);
        }
        let mut offset = 0;
        let mut v: i64 = 0;
        while offset < 10 {
            let byte = *self
                .data
                .as_ref()
                .get(self.data_offset + offset)
                .ok_or_else(|| {
                    ParquetError::EOF(
                        "unexpected end of file whilst decoding definition levels rle header"
                            .into(),
                    )
                })?;

            v |= ((byte & 0x7F) as i64) << (offset * 7);
            offset += 1;
            if byte & 0x80 == 0 {
                self.data_offset += offset;
                return Ok(v);
            }
        }
        Err(general_err!("too many bytes for VLQ"))
    }
}

impl PackedDecoder {
    fn new() -> Self {
        Self {
            data: Bytes::from(vec![]),
            data_offset: 0,
            rle_left: 0,
            rle_value: false,
            packed_count: 0,
            packed_offset: 0,
        }
    }

    fn set_data(&mut self, data: Bytes) {
        self.rle_left = 0;
        self.rle_value = false;
        self.packed_offset = 0;
        self.packed_count = 0;
        self.data = data;
        self.data_offset = 0;
    }

    /// Try to consume `len` levels if all are valid (max definition level).
    ///
    /// Returns `Ok(Some(count))` if successfully consumed `count` all-valid levels.
    /// Returns `Ok(None)` if there are any nulls or packed data that prevents fast path.
    ///
    /// Note: On `None`, the decoder state may have advanced to the next RLE block,
    /// but only if `rle_left` was zero (i.e., the block would have been loaded
    /// on the next read anyway).
    fn try_consume_all_valid(&mut self, len: usize) -> Result<Option<usize>> {
        // If no active run and no packed data pending, try to parse the next RLE block
        if self.rle_left == 0 && self.packed_count == self.packed_offset {
            if self.data_offset < self.data.len() {
                self.next_rle_block()?;
            } else {
                // No more data available
                return Ok(None);
            }
        }

        // Fast path only works when we have an active RLE run of true values
        // that covers the entire requested length.
        if self.rle_left >= len && self.rle_value {
            self.rle_left -= len;
            return Ok(Some(len));
        }

        // Any other case (null run, packed data, or insufficient length)
        // falls back to normal path
        Ok(None)
    }

    /// Appends `len` levels to `buffer` as validity bits. Returns the levels
    /// read and how many of them are set. The bits are written in place with
    /// word stores: a repeated run sets a bit range, a packed run copies its
    /// bytes at the bitmap's bit offset.
    fn read(&mut self, buffer: &mut BooleanBufferBuilder, len: usize) -> Result<(usize, usize)> {
        let start = buffer.len();
        // Zeroed bits for the levels and eight bytes of slack for the last
        // word store; truncated on return.
        buffer.advance(len + 64);
        let mut out = Appender::new(buffer.as_slice_mut(), start);
        let end = start + len;
        let mut pos = start;
        let mut valid = 0;
        let mut result = Ok(());
        while pos != end {
            if self.rle_left != 0 {
                let n = self.rle_left.min(end - pos);
                if self.rle_value {
                    out.ones(n);
                    valid += n;
                } else {
                    out.zeros(n);
                }
                self.rle_left -= n;
                pos += n;
            } else if self.packed_count != self.packed_offset {
                let n = (self.packed_count - self.packed_offset).min(end - pos);
                let offset = self.data_offset * 8 + self.packed_offset;
                // `Appender::copy` reads zeros past the data, which would be
                // NULL levels. A packed run longer than the data is an error
                // (`append_packed_range` panicked here before).
                if offset + n > self.data.len() * 8 {
                    result = Err(ParquetError::EOF(
                        "unexpected end of file whilst decoding definition levels bit-packed run"
                            .into(),
                    ));
                    break;
                }
                valid += out.copy(self.data.as_ref(), offset, n);
                self.packed_offset += n;
                pos += n;

                if self.packed_offset == self.packed_count {
                    self.data_offset += self.packed_count / 8;
                }
            } else if self.data_offset == self.data.len() {
                break;
            } else if let Some((levels, set)) = self.whole_runs(&mut out, end - pos) {
                pos += levels;
                valid += set;
            } else if let Err(error) = self.next_rle_block() {
                result = Err(error);
                break;
            }
        }
        out.finish();
        buffer.truncate(pos);
        result.map(|()| (pos - start, valid))
    }

    /// Decodes runs with one-byte headers from a local cursor until `end`,
    /// the data, or a run that needs `next_rle_block` (a longer header, a
    /// missing value byte). A run that `end` cuts is left in the decoder
    /// state, as `next_rle_block` leaves it. Returns the levels and set bits
    /// written, or `None` when no header was consumed and the next run needs
    /// the general path.
    #[inline(always)]
    fn whole_runs(&mut self, out: &mut Appender<'_>, len: usize) -> Option<(usize, usize)> {
        let data = self.data.as_ref();
        let mut offset = self.data_offset;
        let (start, end) = (0, len);
        let mut pos = start;
        let mut valid = 0;
        while pos != end {
            let Some(&header) = data.get(offset) else {
                break;
            };
            if header >= 0x80 {
                break;
            }
            let run = (header >> 1) as usize;
            if header & 1 == 1 {
                // A packed run longer than the data takes the general path
                // (`next_rle_block`), which reports it.
                let bytes = run;
                if bytes > data.len() - offset - 1 {
                    break;
                }
                let count = bytes * 8;
                let n = count.min(end - pos);
                valid += out.copy(data, (offset + 1) * 8, n);
                pos += n;
                if n < count {
                    self.data_offset = offset + 1;
                    self.packed_count = count;
                    self.packed_offset = n;
                    return Some((pos - start, valid));
                }
                offset += 1 + bytes;
            } else {
                let Some(&value) = data.get(offset + 1) else {
                    break;
                };
                let n = run.min(end - pos);
                if value != 0 {
                    out.ones(n);
                    valid += n;
                } else {
                    out.zeros(n);
                }
                pos += n;
                offset += 2;
                if n < run {
                    self.data_offset = offset;
                    self.rle_left = run - n;
                    self.rle_value = value != 0;
                    return Some((pos - start, valid));
                }
            }
        }
        // Packed state is complete here: the next read starts at a header.
        self.packed_count = 0;
        self.packed_offset = 0;
        let progress = offset != self.data_offset;
        self.data_offset = offset;
        progress.then_some((pos - start, valid))
    }

    /// Skips `level_num` definition levels
    ///
    /// Returns the number of values skipped and the number of levels skipped
    fn skip(&mut self, level_num: usize) -> Result<(usize, usize)> {
        let mut skipped_value = 0;
        let mut skipped_level = 0;
        while skipped_level != level_num {
            if self.rle_left != 0 {
                let to_skip = self.rle_left.min(level_num - skipped_level);
                self.rle_left -= to_skip;
                skipped_level += to_skip;
                if self.rle_value {
                    skipped_value += to_skip;
                }
            } else if self.packed_count != self.packed_offset {
                let to_skip =
                    (self.packed_count - self.packed_offset).min(level_num - skipped_level);
                let offset = self.data_offset * 8 + self.packed_offset;
                // A packed run longer than the data is an error, as in `read`
                // (`UnalignedBitChunk::new` panicked here before).
                if offset + to_skip > self.data.len() * 8 {
                    return Err(ParquetError::EOF(
                        "unexpected end of file whilst decoding definition levels bit-packed run"
                            .into(),
                    ));
                }
                let bit_chunk = UnalignedBitChunk::new(self.data.as_ref(), offset, to_skip);
                skipped_value += bit_chunk.count_ones();
                self.packed_offset += to_skip;
                skipped_level += to_skip;
                if self.packed_offset == self.packed_count {
                    self.data_offset += self.packed_count / 8;
                }
            } else if self.data_offset == self.data.len() {
                break;
            } else {
                self.next_rle_block()?
            }
        }
        Ok((skipped_value, skipped_level))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::encodings::rle::RleEncoder;
    use crate::util::bit_util::BitWriter;
    use rand::{RngExt, rng};

    #[test]
    fn test_build_validity_bitmap_unfiltered_word_chunk() {
        // 65 levels forces the unfiltered path to process one full u64 word
        // with append_word, plus a remainder bit.
        let def_levels = (0..65)
            .map(|i| if i % 3 == 0 { 2 } else { 1 })
            .collect::<Vec<_>>();
        let mut bitmap = BooleanBufferBuilder::new(0);

        assert_eq!(
            build_filtered_validity_bitmap(&def_levels, None, None, 2, &mut bitmap),
            def_levels.len()
        );

        let bitmap = bitmap.finish();
        for (idx, def) in def_levels.iter().enumerate() {
            assert_eq!(bitmap.value(idx), *def >= 2);
        }
    }

    /// The word appender against a bit-by-bit reference: every start offset,
    /// runs of ones and zeros of 0 to 200 bits, and copied bit ranges.
    #[test]
    fn test_appender_matches_reference() {
        let mut state = 0x1234_5678_9abc_def1u64;
        let src: Vec<u8> = (0..64)
            .map(|i| (i as u8).wrapping_mul(151) ^ 0x5a)
            .collect();
        for start in 0..130 {
            let mut bits = vec![0u8; 256];
            let mut expected = vec![false; 2048];
            for i in 0..start {
                if i % 3 == 0 {
                    bits[i / 8] |= 1 << (i % 8);
                    expected[i] = true;
                }
            }
            let mut out = Appender::new(&mut bits, start);
            let mut pos = start;
            while pos < 1400 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let n = (state % 201) as usize;
                match state >> 62 {
                    0 => {
                        out.ones(n);
                        expected[pos..pos + n].fill(true);
                    }
                    1 => out.zeros(n),
                    _ => {
                        let from = (state >> 8) as usize % 200;
                        let set = out.copy(&src, from, n);
                        let mut count = 0;
                        for i in 0..n {
                            let bit = (src[(from + i) / 8] >> ((from + i) % 8)) & 1 == 1;
                            expected[pos + i] = bit;
                            count += bit as usize;
                        }
                        assert_eq!(set, count);
                    }
                }
                pos += n;
            }
            out.finish();
            for (i, &bit) in expected[..pos].iter().enumerate() {
                assert_eq!(
                    (bits[i / 8] >> (i % 8)) & 1 == 1,
                    bit,
                    "start={start} bit={i}"
                );
            }
            assert!(bits[pos.div_ceil(8) + 8..].iter().all(|&b| b == 0));
        }
    }

    #[test]
    fn independent_cached_validity_unaligned_skip_matrix() {
        let mut saw_unaligned_copy = false;
        for len in [1, 7, 8, 9, 63, 64, 65, 193, 65_541] {
            for pattern in 0..3 {
                let bits: Vec<_> = (0..len)
                    .map(|i| match pattern {
                        0 => false,
                        1 => true,
                        _ => i % 11 < 4 || i % 23 == 0,
                    })
                    .collect();
                let mut encoder = RleEncoder::new(1, len);
                for &bit in &bits {
                    encoder.put(bit as u64);
                }
                let data = Bytes::from(encoder.consume());
                let mut body = (data.len() as u32).to_le_bytes().to_vec();
                body.extend_from_slice(&data);
                let page = Page::DataPage {
                    buf: Bytes::from(body),
                    num_values: len as u32,
                    encoding: Encoding::PLAIN,
                    def_level_encoding: Encoding::RLE,
                    rep_level_encoding: Encoding::RLE,
                    statistics: None,
                };
                let validity = Arc::new(cache_validity(&page).unwrap());
                for prefix in 0..8 {
                    for skip in [0, 1, 3, 7, 9, len / 2] {
                        let mut cursor = ValidityCursor {
                            data: validity.clone(),
                            offset: 0,
                            remaining_valid: len - validity.nulls,
                        };
                        let mut expected = vec![true; prefix];
                        let mut output = BooleanBufferBuilder::new(0);
                        output.append_n(prefix, true);
                        let mut offset = 0;
                        for (count, read) in [
                            (skip, false),
                            (17, true),
                            (9, false),
                            (19, true),
                            (len + 5, true),
                            (5, false),
                        ] {
                            let end = (offset + count).min(len);
                            let range = &bits[offset..end];
                            let ones = range.iter().filter(|&&bit| bit).count();
                            if read {
                                saw_unaligned_copy |= offset % 8 != 0
                                    && output.len() % 8 != 0
                                    && ones > 0
                                    && ones < range.len();
                                expected.extend_from_slice(range);
                            }
                            assert_eq!(
                                cursor.consume(count, read.then_some(&mut output)),
                                (ones, range.len()),
                                "len={len} pattern={pattern} prefix={prefix} skip={skip} offset={offset}",
                            );
                            offset = end;
                        }
                        assert_eq!(output.len(), expected.len());
                        for (i, bit) in expected.into_iter().enumerate() {
                            assert_eq!(output.get_bit(i), bit, "len={len} bit={i}");
                        }
                    }
                }
            }
        }
        assert!(saw_unaligned_copy);
    }

    #[test]
    fn cached_validity_matches_every_batch_boundary_and_skip() {
        let len = 193;
        for (pattern, bits) in [
            ("all-valid", vec![true; len]),
            ("all-null", vec![false; len]),
            (
                "mixed",
                (0..len).map(|i| i % 53 < 17 || i % 7 == 0).collect(),
            ),
        ] {
            let mut encoder = RleEncoder::new(1, len);
            for &bit in &bits {
                encoder.put(bit as u64);
            }
            let data = Bytes::from(encoder.consume());
            let page = Page::DataPageV2 {
                buf: data.clone(),
                num_values: len as u32,
                encoding: Encoding::PLAIN,
                num_nulls: bits.iter().filter(|&&bit| !bit).count() as u32,
                num_rows: len as u32,
                def_levels_byte_len: data.len() as u32,
                rep_levels_byte_len: 0,
                is_compressed: false,
                statistics: None,
            };
            let validity = Arc::new(cache_validity(&page).unwrap());
            let mut cached = DefinitionLevelBufferDecoder::new(1, true);
            for split in 0..=len {
                for skip_first in [false, true] {
                    cached
                        .set_data_with_validity(Encoding::RLE, data.clone(), Some(validity.clone()))
                        .unwrap();
                    let mut output = DefinitionLevelBuffer {
                        inner: BufferInner::Mask {
                            nulls: BooleanBufferBuilder::new(0),
                        },
                        len: 0,
                    };
                    let expected = |range: std::ops::Range<usize>| {
                        (
                            bits[range.clone()].iter().filter(|&&bit| bit).count(),
                            range.len(),
                        )
                    };
                    if skip_first {
                        assert_eq!(cached.skip_def_levels(split).unwrap(), expected(0..split));
                    } else {
                        assert_eq!(
                            cached.read_def_levels(&mut output, split).unwrap(),
                            expected(0..split)
                        );
                    }
                    assert_eq!(
                        cached.read_def_levels(&mut output, len).unwrap(),
                        expected(split..len)
                    );
                    assert_eq!(cached.skip_def_levels(1).unwrap(), (0, 0));
                    let BufferInner::Mask { nulls } = output.inner else {
                        unreachable!()
                    };
                    let begin = if skip_first { split } else { 0 };
                    assert_eq!(nulls.len(), len - begin);
                    for (i, &bit) in bits[begin..].iter().enumerate() {
                        assert_eq!(nulls.get_bit(i), bit, "{pattern}: split {split}, bit {i}");
                    }
                }
            }

            cached.set_data(Encoding::RLE, data).unwrap();
            assert_eq!(
                cached.skip_def_levels(len).unwrap(),
                (len - validity.nulls, len)
            );
        }
    }

    #[test]
    fn test_packed_decoder() {
        let mut rng = rng();
        let len: usize = rng.random_range(512..1024);

        let mut expected = BooleanBufferBuilder::new(len);
        let mut encoder = RleEncoder::new(1, 1024);
        for _ in 0..len {
            let bool = rng.random_bool(0.8);
            encoder.put(bool as u64);
            expected.append(bool);
        }
        assert_eq!(expected.len(), len);

        let encoded = encoder.consume();
        let mut decoder = PackedDecoder::new();
        decoder.set_data(encoded.into());

        // Decode data in random length intervals
        let mut decoded = BooleanBufferBuilder::new(len);
        loop {
            let remaining = len - decoded.len();
            if remaining == 0 {
                break;
            }

            let to_read = rng.random_range(1..=remaining);
            decoder.read(&mut decoded, to_read).unwrap();
        }

        assert_eq!(decoded.len(), len);
        assert_eq!(decoded.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_packed_decoder_skip() {
        let mut rng = rng();
        let len: usize = rng.random_range(512..1024);

        let mut expected = BooleanBufferBuilder::new(len);
        let mut encoder = RleEncoder::new(1, 1024);

        let mut total_value = 0;
        for _ in 0..len {
            let bool = rng.random_bool(0.8);
            encoder.put(bool as u64);
            expected.append(bool);
            if bool {
                total_value += 1;
            }
        }
        assert_eq!(expected.len(), len);

        let encoded = encoder.consume();
        let mut decoder = PackedDecoder::new();
        decoder.set_data(encoded.into());

        let mut skip_value = 0;
        let mut read_value = 0;
        let mut skip_level = 0;
        let mut read_level = 0;

        loop {
            let offset = skip_level + read_level;
            let remaining_levels = len - offset;
            if remaining_levels == 0 {
                break;
            }
            let to_read_or_skip_level = rng.random_range(1..=remaining_levels);
            if rng.random_bool(0.5) {
                let (skip_val_num, skip_level_num) = decoder.skip(to_read_or_skip_level).unwrap();
                skip_value += skip_val_num;
                skip_level += skip_level_num
            } else {
                let mut decoded = BooleanBufferBuilder::new(to_read_or_skip_level);
                let read_level_num = decoder.read(&mut decoded, to_read_or_skip_level).unwrap().0;
                read_level += read_level_num;
                for i in 0..read_level_num {
                    assert!(!decoded.is_empty());
                    //check each read bit
                    let read_bit = decoded.get_bit(i);
                    if read_bit {
                        read_value += 1;
                    }
                    let expect_bit = expected.get_bit(i + offset);
                    assert_eq!(read_bit, expect_bit);
                }
            }
        }
        assert_eq!(read_level + skip_level, len);
        assert_eq!(read_value + skip_value, total_value);
    }

    #[test]
    fn test_try_consume_all_valid() {
        // Test with all-valid data (all 1s) - single RLE run
        let len = 100;
        let mut encoder = RleEncoder::new(1, 1024);
        for _ in 0..len {
            encoder.put(1); // all valid
        }
        let encoded = encoder.consume();
        let mut decoder = PackedDecoder::new();
        decoder.set_data(encoded.into());

        // try_consume_all_valid now parses the RLE block itself, no need to read first
        let result = decoder.try_consume_all_valid(len).unwrap();
        assert_eq!(result, Some(len));

        // Test with all-null data (all 0s)
        let mut encoder = RleEncoder::new(1, 1024);
        for _ in 0..len {
            encoder.put(0); // all null
        }
        let encoded = encoder.consume();
        let mut decoder = PackedDecoder::new();
        decoder.set_data(encoded.into());

        // Should return None because rle_value is false (all nulls)
        let result = decoder.try_consume_all_valid(len).unwrap();
        assert_eq!(result, None);

        // Test when requesting more than available in current RLE run
        let mut encoder = RleEncoder::new(1, 1024);
        for _ in 0..10 {
            encoder.put(1); // small run of valid
        }
        for _ in 0..10 {
            encoder.put(0); // followed by nulls
        }
        let encoded = encoder.consume();
        let mut decoder = PackedDecoder::new();
        decoder.set_data(encoded.into());

        // Request more than the valid run - should return None
        // (because we don't look ahead to next block)
        let result = decoder.try_consume_all_valid(20).unwrap();
        assert_eq!(result, None);

        // Reset decoder and try requesting within the run
        decoder.set_data({
            let mut encoder = RleEncoder::new(1, 1024);
            for _ in 0..10 {
                encoder.put(1);
            }
            for _ in 0..10 {
                encoder.put(0);
            }
            encoder.consume().into()
        });

        let result = decoder.try_consume_all_valid(5).unwrap();
        assert_eq!(result, Some(5));

        // After skipping 5, we should have 5 left in the valid RLE run
        let result = decoder.try_consume_all_valid(5).unwrap();
        assert_eq!(result, Some(5));

        // Now the valid run is exhausted, next call should parse the null run and return None
        let result = decoder.try_consume_all_valid(5).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_packed_decoder_invalid_run_len() {
        let mut writer = BitWriter::new(16);
        writer.put_vlq_int(u64::MAX);
        let mut decoder = PackedDecoder::new();
        decoder.set_data(writer.consume().into());

        let mut decoded = BooleanBufferBuilder::new(8);
        let err = decoder.read(&mut decoded, 8).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Parquet error: Invalid or corrupted RLE run length 18446744073709551615. Max allowed is 2147483647"
        );
    }

    /// A bit-packed run that declares more groups than the data holds is an
    /// error on both header paths, and writes no levels. The word appender
    /// reads zeros past the data, so it must not see such a run.
    #[test]
    fn test_packed_decoder_run_past_data() {
        // One-byte header (the local run loop) and a two-byte header
        // (`next_rle_block`), each with one byte of levels.
        for header in [vec![(4 << 1) | 1], vec![0x81, 0x01]] {
            let mut data = header;
            data.push(0b0101_0101);
            let mut decoder = PackedDecoder::new();
            decoder.set_data(Bytes::from(data));
            let mut decoded = BooleanBufferBuilder::new(0);
            decoded.append_n(3, true);
            let err = decoder.read(&mut decoded, 32).unwrap_err();
            assert_eq!(
                err.to_string(),
                "EOF: unexpected end of file whilst decoding definition levels bit-packed run"
            );
            assert_eq!(decoded.len(), 3);
        }
    }
}

#[cfg(test)]
mod invalid_input_tests {
    use super::*;

    /// A bit-packed run that declares more groups than the data holds is an
    /// error in a skip, as in a read. Before, the skip panicked.
    #[test]
    fn skip_of_a_packed_run_past_data_fails() {
        // One-byte and two-byte headers of 4 groups, with one byte of levels.
        for header in [vec![(4 << 1) | 1], vec![0x81, 0x01]] {
            let mut data = header;
            data.push(0b0101_0101);
            let mut decoder = PackedDecoder::new();
            decoder.set_data(Bytes::from(data));
            let err = decoder.skip(32).unwrap_err();
            assert_eq!(
                err.to_string(),
                "EOF: unexpected end of file whilst decoding definition levels bit-packed run"
            );
        }

        // A skip within the data still works.
        let mut decoder = PackedDecoder::new();
        decoder.set_data(Bytes::from(vec![(1 << 1) | 1, 0b0101_0101]));
        assert_eq!(decoder.skip(8).unwrap(), (4, 8));
    }
}
