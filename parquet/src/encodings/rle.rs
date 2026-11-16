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

//! Rle/Bit-Packing Hybrid Encoding
//! The grammar for this encoding looks like the following (copied verbatim
//! from <https://github.com/Parquet/parquet-format/blob/master/Encodings.md>):
//!
//! rle-bit-packed-hybrid: `<length>` `<encoded-data>`
//! length := length of the `<encoded-data>` in bytes stored as 4 bytes little endian
//! encoded-data := `<run>`*
//! run := `<bit-packed-run>` | `<rle-run>`
//! bit-packed-run := `<bit-packed-header>` `<bit-packed-values>`
//! bit-packed-header := varint-encode(`<bit-pack-count>` << 1 | 1)
//! we always bit-pack a multiple of 8 values at a time, so we only store the number of
//! values / 8
//! bit-pack-count := (number of values in this run) / 8
//! bit-packed-values := *see 1 below*
//! rle-run := `<rle-header>` `<repeated-value>`
//! rle-header := varint-encode( (number of times repeated) << 1)
//! repeated-value := value that is repeated, using a fixed-width of
//! round-up-to-next-byte(bit-width)

use std::{cmp, mem::size_of};

use bytes::Bytes;

use crate::errors::{ParquetError, Result};
use crate::util::bit_util::{self, BitPacking, BitReader, BitWriter};

/// Maximum bit width for dictionary page indices encoded with RLE / Bit-Packing hybrid encoding.
///
/// Parquet dictionary indices are represented as 32-bit signed integers, so dictionary
/// index bit widths must not exceed 32.
pub const MAX_RLE_DICTIONARY_BIT_WIDTH: u8 = 32;

/// Number of values in one bit-packed group. The Parquet RLE/bit-packing hybrid
/// format always bit-packs values in multiples of this count (see the
/// [format spec](https://github.com/apache/parquet-format/blob/master/Encodings.md#run-length-encoding--bit-packing-hybrid-rle--3):
/// "we always bit-pack a multiple of 8 values at a time").
const BIT_PACK_GROUP_SIZE: usize = 8;

/// Maximum groups of `BIT_PACK_GROUP_SIZE` values per bit-packed run. Current value is 64.
const MAX_GROUPS_PER_BIT_PACKED_RUN: usize = 1 << 6;

/// Maximum number of values in a single run. The
/// [format spec](https://github.com/apache/parquet-format/blob/master/Encodings.md#run-length-encoding--bit-packing-hybrid-rle--3)
/// requires run lengths to be in the range `[1, 2^31 - 1]`.
const MAX_RUN_LEN: u64 = i32::MAX as u64;

/// Returns the number of values in the run with the given header, or an error if it
/// exceeds [`MAX_RUN_LEN`].
#[inline]
pub(crate) fn run_len(indicator_value: u64) -> Result<u32> {
    let len = if indicator_value & 1 == 1 {
        // A bit-packed header stores the number of groups of `BIT_PACK_GROUP_SIZE` values.
        // Saturate so that a corrupt header which overflows is rejected below.
        (indicator_value >> 1).saturating_mul(BIT_PACK_GROUP_SIZE as u64)
    } else {
        indicator_value >> 1
    };
    if len > MAX_RUN_LEN {
        return Err(general_err!(
            "Invalid or corrupted RLE run length {len}. Max allowed is {MAX_RUN_LEN}"
        ));
    }
    Ok(len as u32)
}

/// A RLE/Bit-Packing hybrid encoder.
// TODO: tracking memory usage
pub struct RleEncoder {
    // Number of bits needed to encode the value. Must be in the range of [0, 64].
    bit_width: u8,

    // Underlying writer which holds an internal buffer.
    bit_writer: BitWriter,

    // Values of the current in-progress bit-packed run, including the partially
    // filled trailing group. Only materialized into `bit_writer` when the run
    // closes, so the whole run can be packed in one vectorised `put_batch` call.
    // Bounded by `MAX_GROUPS_PER_BIT_PACKED_RUN * BIT_PACK_GROUP_SIZE` values.
    pending_values: Vec<u64>,

    // The current (also last) value that was written and the count of how many
    // times in a row that value has been seen.
    current_value: u64,

    // The number of repetitions for `current_value`. If this gets too high we'd
    // switch to use RLE encoding.
    repeat_count: usize,
}

impl RleEncoder {
    #[cfg_attr(all(not(feature = "experimental"), not(test)), expect(unused))]
    pub fn new(bit_width: u8, buffer_len: usize) -> Self {
        let buffer = Vec::with_capacity(buffer_len);
        RleEncoder::new_from_buf(bit_width, buffer)
    }

    /// Initialize the encoder from existing `buffer`
    pub fn new_from_buf(bit_width: u8, buffer: Vec<u8>) -> Self {
        debug_assert!(bit_width <= 64);
        let bit_writer = BitWriter::new_from_buf(buffer);
        RleEncoder {
            bit_width,
            bit_writer,
            pending_values: Vec::new(),
            current_value: 0,
            repeat_count: 0,
        }
    }

    /// Returns the maximum buffer size to encode `num_values` values with
    /// `bit_width`.
    pub fn max_buffer_size(bit_width: u8, num_values: usize) -> usize {
        // The maximum size occurs with the shortest possible runs of BIT_PACK_GROUP_SIZE
        let num_runs = bit_util::ceil(num_values, BIT_PACK_GROUP_SIZE);

        // The number of bytes in a run of BIT_PACK_GROUP_SIZE
        let bytes_per_run = bit_width as usize;

        // The maximum size if stored as shortest possible bit packed runs of BIT_PACK_GROUP_SIZE
        let bit_packed_max_size = num_runs + num_runs * bytes_per_run;

        // The length of `BIT_PACK_GROUP_SIZE` VLQ encoded
        let rle_len_prefix = 1;

        // The length of an RLE run of BIT_PACK_GROUP_SIZE
        let min_rle_run_size =
            rle_len_prefix + bit_util::ceil(bit_width as usize, u8::BITS as usize);

        // The maximum size if stored as shortest possible RLE runs of BIT_PACK_GROUP_SIZE
        let rle_max_size = num_runs * min_rle_run_size;

        bit_packed_max_size.max(rle_max_size)
    }

    /// Returns `true` if the encoder is currently in RLE accumulation mode
    /// for the given value (i.e., `repeat_count >= BIT_PACK_GROUP_SIZE` and `current_value == value`).
    ///
    /// The encoder enters accumulation mode as soon as the 8th consecutive identical
    /// value has been seen: at that point `flush_buffered_values` has committed the
    /// RLE decision and cleared the staging buffer, so no more per-element work is
    /// needed.  Callers may use [`extend_run`](Self::extend_run) to add further
    /// repetitions in O(1) once this returns `true`.
    #[inline]
    pub fn is_accumulating_rle(&self, value: u64) -> bool {
        self.repeat_count >= BIT_PACK_GROUP_SIZE && self.current_value == value
    }

    /// Extends the current RLE run by `count` additional repetitions.
    ///
    /// # Preconditions
    /// The caller **must** have verified [`is_accumulating_rle`](Self::is_accumulating_rle)
    /// returns `true` for the same value before calling this method.
    #[inline]
    pub fn extend_run(&mut self, count: usize) {
        debug_assert!(self.repeat_count >= BIT_PACK_GROUP_SIZE);
        self.repeat_count += count;
    }

    /// Encodes `value`, which must be representable with `bit_width` bits.
    #[inline]
    pub fn put(&mut self, value: u64) {
        // This function buffers BIT_PACK_GROUP_SIZE values at a time. After seeing that
        // many values, it decides whether the current run should be encoded in bit-packed
        // or RLE.
        if self.current_value == value {
            self.repeat_count += 1;
            if self.repeat_count > BIT_PACK_GROUP_SIZE {
                // A continuation of last value. No need to buffer.
                return;
            }
        } else {
            if self.repeat_count >= BIT_PACK_GROUP_SIZE {
                // The current RLE run has ended and we've gathered enough. Flush first.
                debug_assert!(self.pending_values.is_empty());
                self.flush_rle_run();
            }
            self.repeat_count = 1;
            self.current_value = value;
        }

        self.pending_values.push(value);
        if self
            .pending_values
            .len()
            .is_multiple_of(BIT_PACK_GROUP_SIZE)
        {
            // A group of values is complete. Decide its encoding.
            self.commit_group();
        }
    }

    /// Encodes `values`, each of which must be representable with `bit_width` bits.
    ///
    /// Produces output identical to calling [`Self::put`] for each value, but avoids
    /// the per-value run tracking wherever a whole group of values can be examined at
    /// once.
    pub fn put_batch(&mut self, values: &[u64]) {
        let mut i = 0;
        while i < values.len() {
            // Extend an active RLE run with its whole continuation at once
            if self.repeat_count >= BIT_PACK_GROUP_SIZE && values[i] == self.current_value {
                let run_len = values[i..]
                    .iter()
                    .take_while(|&&v| v == self.current_value)
                    .count();
                self.repeat_count += run_len;
                i += run_len;
                continue;
            }

            // Whole groups bypass the per-value state machine while the pending run is
            // group-aligned and no RLE run is being accumulated. `repeat_count` is
            // always 0 here: `commit_group` resets it whenever a group completes, and
            // that is the only way to reach a group-aligned state
            if self.repeat_count < BIT_PACK_GROUP_SIZE
                && self
                    .pending_values
                    .len()
                    .is_multiple_of(BIT_PACK_GROUP_SIZE)
                && values.len() - i >= BIT_PACK_GROUP_SIZE
            {
                debug_assert_eq!(self.repeat_count, 0);
                let group = &values[i..i + BIT_PACK_GROUP_SIZE];
                i += BIT_PACK_GROUP_SIZE;
                if group.iter().all(|&v| v == group[0]) {
                    // The same decision `commit_group` makes for a group holding a
                    // single repeated value, without appending it first
                    if !self.pending_values.is_empty() {
                        self.close_bit_packed_run();
                    }
                    self.current_value = group[0];
                    self.repeat_count = BIT_PACK_GROUP_SIZE;
                } else {
                    self.pending_values.extend_from_slice(group);
                    self.current_value = group[BIT_PACK_GROUP_SIZE - 1];
                    let num_groups = self.pending_values.len() / BIT_PACK_GROUP_SIZE;
                    if num_groups + 1 >= MAX_GROUPS_PER_BIT_PACKED_RUN {
                        self.close_bit_packed_run();
                    }
                }
                continue;
            }

            // Ending an RLE run, filling up a partial group, or a trailing partial
            // group: fall back to the per-value path
            self.put(values[i]);
            i += 1;
        }
    }

    #[inline]
    #[cfg_attr(not(feature = "experimental"), expect(unused))]
    pub fn buffer(&self) -> &[u8] {
        self.bit_writer.buffer()
    }

    #[inline]
    pub fn len(&self) -> usize {
        // Include the eventual size of the pending bit-packed run, so size estimates
        // do not lag behind by up to a full run
        let pending_bytes = match self.pending_values.len() {
            0 => 0,
            n => 1 + bit_util::ceil(n * self.bit_width as usize, u8::BITS as usize),
        };
        self.bit_writer.bytes_written() + pending_bytes
    }

    #[cfg_attr(not(feature = "experimental"), expect(unused))]
    pub fn is_empty(&self) -> bool {
        self.bit_writer.bytes_written() == 0
    }

    #[inline]
    pub fn consume(mut self) -> Vec<u8> {
        self.flush();
        self.bit_writer.consume()
    }

    /// Borrow equivalent of the `consume` method.
    /// Call `clear()` after invoking this method.
    #[inline]
    pub fn flush_buffer(&mut self) -> &[u8] {
        self.flush();
        self.bit_writer.flush_buffer()
    }

    /// Like `flush_buffer`, but returns mutable access to the internal buffer.
    /// Call `clear()` after invoking this method.
    #[inline]
    pub fn flush_buffer_mut(&mut self) -> &mut [u8] {
        self.flush();
        self.bit_writer.flush_buffer_mut()
    }

    /// Clears the internal state so this encoder can be reused (e.g., after becoming
    /// full).
    #[inline]
    pub fn clear(&mut self) {
        self.bit_writer.clear();
        self.pending_values.clear();
        self.current_value = 0;
        self.repeat_count = 0;
    }

    /// Advances the buffer by `num_bytes` zero bytes, delegating to the
    /// underlying [`BitWriter::skip`].
    #[inline]
    pub fn skip(&mut self, num_bytes: usize) {
        self.bit_writer.skip(num_bytes);
    }

    /// Flushes all remaining values and return the final byte buffer maintained by the
    /// internal writer.
    #[inline]
    pub fn flush(&mut self) {
        if !self.pending_values.is_empty() || self.repeat_count > 0 {
            let tail = self.pending_values.len() % BIT_PACK_GROUP_SIZE;
            let all_repeat =
                self.pending_values.len() == tail && (self.repeat_count == tail || tail == 0);
            if self.repeat_count > 0 && all_repeat {
                self.flush_rle_run();
            } else {
                // Pad the last group of bit-packed values to BIT_PACK_GROUP_SIZE with 0s.
                if tail > 0 {
                    self.pending_values
                        .resize(self.pending_values.len() + BIT_PACK_GROUP_SIZE - tail, 0);
                }
                if !self.pending_values.is_empty() {
                    self.close_bit_packed_run();
                }
                self.repeat_count = 0;
            }
        }
    }

    fn flush_rle_run(&mut self) {
        debug_assert!(self.repeat_count > 0);
        let indicator_value = self.repeat_count << 1;
        self.bit_writer.put_vlq_int(indicator_value as u64);
        self.bit_writer.put_aligned(
            self.current_value,
            bit_util::ceil(self.bit_width as usize, u8::BITS as usize),
        );
        self.pending_values.clear();
        self.repeat_count = 0;
    }

    // Called when a group of BIT_PACK_GROUP_SIZE values completes: either keeps it in
    // the pending bit-packed run, or drops it and switches to RLE when it repeats a
    // single value.
    fn commit_group(&mut self) {
        if self.repeat_count >= BIT_PACK_GROUP_SIZE {
            // The group repeats a single value, drop it and let `repeat_count` track
            // the run in RLE mode instead
            self.pending_values
                .truncate(self.pending_values.len() - BIT_PACK_GROUP_SIZE);
            if !self.pending_values.is_empty() {
                // In this case we have chosen to switch to RLE encoding. Close out the
                // previous bit-packed run.
                self.close_bit_packed_run();
            }
            return;
        }

        let num_groups = self.pending_values.len() / BIT_PACK_GROUP_SIZE;
        if num_groups + 1 >= MAX_GROUPS_PER_BIT_PACKED_RUN {
            // We've reached the maximum value that can be hold in a single bit-packed
            // run.
            self.close_bit_packed_run();
        }
        self.repeat_count = 0;
    }

    // Called when ending a bit-packed run. Writes the indicator byte followed by all
    // pending values in packed form. Deferring the packing until here allows the whole
    // run to go through the vectorised `put_batch` in one call.
    fn close_bit_packed_run(&mut self) {
        debug_assert_eq!(self.pending_values.len() % BIT_PACK_GROUP_SIZE, 0);
        let num_groups = self.pending_values.len() / BIT_PACK_GROUP_SIZE;
        let indicator_byte = ((num_groups << 1) | 1) as u8;
        self.bit_writer.put_aligned(indicator_byte, 1);
        self.bit_writer
            .put_batch(&self.pending_values, self.bit_width as usize);
        self.pending_values.clear();
    }

    /// return the estimated memory size of this encoder.
    pub(crate) fn estimated_memory_size(&self) -> usize {
        self.bit_writer.estimated_memory_size()
            + self.pending_values.capacity() * std::mem::size_of::<u64>()
            + std::mem::size_of::<Self>()
    }
}

mod fused;
mod runs;

/// Size, in number of `i32s` of buffer to use for RLE batch reading
const RLE_DECODER_INDEX_BUFFER_SIZE: usize = 1024;

/// A RLE/Bit-Packing hybrid decoder.
pub struct RleDecoder {
    // Number of bits used to encode the value. Must be between [0, 64].
    bit_width: u8,

    // Bit reader loaded with input buffer.
    bit_reader: Option<BitReader>,

    // Buffer used when `bit_reader` is not `None`, for batch reading.
    index_buf: Option<Box<[i32; RLE_DECODER_INDEX_BUFFER_SIZE]>>,

    // The remaining number of values in RLE for this run
    rle_left: u32,

    // The remaining number of values in Bit-Packing for this run
    bit_packed_left: u32,

    // The current value for the case of RLE mode
    current_value: Option<u64>,
}

impl RleDecoder {
    /// Creates a new `RleDecoder` with the specified bit width.
    ///
    /// Bit width must be between 0 and 64 (inclusive). Note that for dictionary indices
    /// specifically, the bit width cannot exceed [`MAX_RLE_DICTIONARY_BIT_WIDTH`] (32).
    pub fn new(bit_width: u8) -> Self {
        debug_assert!(bit_width <= 64, "Bit width must be <= 64");
        RleDecoder {
            bit_width,
            rle_left: 0,
            bit_packed_left: 0,
            bit_reader: None,
            index_buf: None,
            current_value: None,
        }
    }

    #[inline]
    pub fn set_data(&mut self, data: Bytes) -> Result<()> {
        if let Some(ref mut bit_reader) = self.bit_reader {
            bit_reader.reset(data);
        } else {
            self.bit_reader = Some(BitReader::new(data));
        }

        // Initialize decoder state. The boolean only reports whether the first run contained data,
        // and `get`/`get_batch` already interpret that result to drive iteration. We only need
        // errors propagated here, so the flag returned is intentionally ignored.
        let _ = self.reload()?;
        Ok(())
    }

    // These functions inline badly, they tend to inline and then create very large loop unrolls
    // that damage L1d-cache occupancy. This results in a ~18% performance drop
    #[inline(never)]
    #[cfg_attr(all(not(feature = "experimental"), not(test)), expect(unused))]
    pub fn get<T: BitPacking>(&mut self) -> Result<Option<T>> {
        assert!(size_of::<T>() <= size_of::<u64>());

        while self.rle_left == 0 && self.bit_packed_left == 0 {
            if !self.reload()? {
                return Ok(None);
            }
        }

        let value = if self.rle_left > 0 {
            let current_value = self
                .current_value
                .ok_or_else(|| general_err!("current_value should be Some"))?;
            let rle_value = T::from_u64(current_value);
            self.rle_left -= 1;
            rle_value
        } else {
            // self.bit_packed_left > 0
            let bit_reader = self
                .bit_reader
                .as_mut()
                .ok_or_else(|| general_err!("bit_reader should be Some"))?;
            let bit_packed_value = bit_reader
                .get_value(self.bit_width as usize)
                .ok_or_else(|| eof_err!("Not enough data for 'bit_packed_value'"))?;
            self.bit_packed_left -= 1;
            bit_packed_value
        };

        Ok(Some(value))
    }

    #[inline(never)]
    pub fn get_batch<T: BitPacking + Clone>(&mut self, buffer: &mut [T]) -> Result<usize> {
        assert!(size_of::<T>() <= size_of::<u64>());

        let mut values_read = 0;
        while values_read < buffer.len() {
            if self.rle_left > 0 {
                let num_values = cmp::min(buffer.len() - values_read, self.rle_left as usize);
                let repeated_value = T::from_u64(self.current_value.unwrap());
                buffer[values_read..values_read + num_values].fill(repeated_value);
                self.rle_left -= num_values as u32;
                values_read += num_values;
            } else if self.bit_packed_left > 0 {
                let mut num_values =
                    cmp::min(buffer.len() - values_read, self.bit_packed_left as usize);
                let bit_reader = self
                    .bit_reader
                    .as_mut()
                    .ok_or_else(|| ParquetError::General("bit_reader should be set".into()))?;

                num_values = bit_reader.get_batch::<T>(
                    &mut buffer[values_read..values_read + num_values],
                    self.bit_width as usize,
                );
                if num_values == 0 {
                    // Handle writers which truncate the final block
                    self.bit_packed_left = 0;
                    continue;
                }
                self.bit_packed_left -= num_values as u32;
                values_read += num_values;
            } else if !self.reload()? {
                break;
            }
        }

        Ok(values_read)
    }

    #[inline(never)]
    pub fn skip(&mut self, num_values: usize) -> Result<usize> {
        let mut values_skipped = 0;
        while values_skipped < num_values {
            if self.rle_left > 0 {
                let num_values = cmp::min(num_values - values_skipped, self.rle_left as usize);
                self.rle_left -= num_values as u32;
                values_skipped += num_values;
            } else if self.bit_packed_left > 0 {
                let mut num_values =
                    cmp::min(num_values - values_skipped, self.bit_packed_left as usize);
                let bit_reader = self
                    .bit_reader
                    .as_mut()
                    .ok_or_else(|| general_err!("bit_reader should be set"))?;

                num_values = bit_reader.skip(num_values, self.bit_width as usize);
                if num_values == 0 {
                    // Handle writers which truncate the final block
                    self.bit_packed_left = 0;
                    continue;
                }
                self.bit_packed_left -= num_values as u32;
                values_skipped += num_values;
            } else if !self.reload()? {
                break;
            }
        }

        Ok(values_skipped)
    }

    /// Skips up to `num_values`, returning the number equal to `value` and the
    /// number skipped. Counts RLE runs directly, one-bit runs with popcount, and
    /// wider packed values with the existing unpacker and bounded scratch space.
    pub(crate) fn skip_and_count<T: BitPacking + Copy + PartialEq>(
        &mut self,
        num_values: usize,
        value: T,
    ) -> Result<(usize, usize)> {
        let mut matching = 0;
        let mut skipped = 0;
        while skipped < num_values {
            if self.rle_left > 0 {
                let count = (num_values - skipped).min(self.rle_left as usize);
                if T::from_u64(self.current_value.unwrap()) == value {
                    matching += count;
                }
                self.rle_left -= count as u32;
                skipped += count;
            } else if self.bit_packed_left > 0 {
                let count = (num_values - skipped).min(self.bit_packed_left as usize);
                let reader = self
                    .bit_reader
                    .as_mut()
                    .ok_or_else(|| general_err!("bit_reader should be set"))?;
                let (matches, read) = reader.skip_and_count(count, self.bit_width as usize, value);
                matching += matches;
                if read == 0 {
                    // Match get_batch's handling of truncated final packed runs.
                    self.bit_packed_left = 0;
                    continue;
                }
                self.bit_packed_left -= read as u32;
                skipped += read;
            } else if !self.reload()? {
                break;
            }
        }
        Ok((matching, skipped))
    }

    #[inline(never)]
    pub fn get_batch_with_dict<T>(
        &mut self,
        dict: &[T],
        buffer: &mut [T],
        max_values: usize,
    ) -> Result<usize>
    where
        T: Default + Clone + 'static,
    {
        assert!(buffer.len() >= max_values);
        if fused::native::<T>() {
            // SAFETY: MaybeUninit<T> has T's layout and permits initialized T.
            // Only exact no-drop primitives enter here. The mutable borrow is
            // exclusive and the decoder only overwrites slots, never reads them.
            let out =
                unsafe { std::slice::from_raw_parts_mut(buffer.as_mut_ptr().cast(), max_values) };
            return self.get_batch_with_dict_uninit(dict, out);
        }

        let mut values_read = 0;
        while values_read < max_values {
            let index_buf = self.index_buf.get_or_insert_with(|| Box::new([0; 1024]));

            if self.rle_left > 0 {
                let num_values = cmp::min(max_values - values_read, self.rle_left as usize);
                let dict_idx = self.current_value.unwrap() as usize;
                let dict_value = dict
                    .get(dict_idx)
                    .ok_or_else(|| {
                        general_err!(
                            "dictionary index out of bounds: the len is {} but the index is {}",
                            dict.len(),
                            dict_idx
                        )
                    })?
                    .clone();

                buffer[values_read..values_read + num_values].fill(dict_value);

                self.rle_left -= num_values as u32;
                values_read += num_values;
            } else if self.bit_packed_left > 0 {
                let bit_reader = self
                    .bit_reader
                    .as_mut()
                    .ok_or_else(|| general_err!("bit_reader should be set"))?;

                let mut failed = None;
                loop {
                    let to_read = index_buf
                        .len()
                        .min(max_values - values_read)
                        .min(self.bit_packed_left as usize);

                    if to_read == 0 {
                        break;
                    }

                    let num_values = bit_reader
                        .get_batch::<i32>(&mut index_buf[..to_read], self.bit_width as usize);
                    if num_values == 0 {
                        // Handle writers which truncate the final block
                        self.bit_packed_left = 0;
                        break;
                    }
                    let out = &mut buffer[values_read..values_read + num_values];
                    let indices = &index_buf[..num_values];
                    let gathered = if !std::mem::needs_drop::<T>()
                        && matches!(std::mem::size_of::<T>(), 4 | 8)
                    {
                        gather_primitive_dictionary(dict, indices, out)
                    } else {
                        gather_dictionary_chunks(dict, indices, out)
                    };
                    if let Err(err) = gathered {
                        // The reader is past these indices.
                        failed = Some(err);
                        break;
                    }
                    self.bit_packed_left -= num_values as u32;
                    values_read += num_values;
                    if num_values < to_read {
                        break;
                    }
                }
                if let Some(err) = failed {
                    // Fail the decoder, so that a retry cannot return the
                    // values after the indices that were read.
                    return Err(self.fail(err));
                }
            } else if !self.reload()? {
                break;
            }
        }

        Ok(values_read)
    }

    /// Append decoded values without initializing primitive output twice.
    pub(crate) fn append_dictionary<T: Default + Clone + 'static>(
        &mut self,
        dict: &[T],
        out: &mut Vec<T>,
        count: usize,
    ) -> Result<usize> {
        let start = out.len();
        if fused::native::<T>() {
            out.reserve(count);
            let n =
                self.get_batch_with_dict_uninit(dict, &mut out.spare_capacity_mut()[..count])?;
            // SAFETY: successful decode initialized n slots in reserved spare
            // capacity. Native T has no destructor. Errors and panics leave the
            // vector length unchanged, so partial writes cannot be observed.
            unsafe {
                out.set_len(start + n);
            }
            Ok(n)
        } else {
            out.resize(start + count, T::default());
            let n = self.get_batch_with_dict(dict, &mut out[start..], count)?;
            out.truncate(start + n);
            Ok(n)
        }
    }

    /// Decode into spare capacity. Only exact no-drop native primitives qualify.
    /// A successful return initializes the returned prefix. On error the caller
    /// must leave the output uninitialized and must not extend its vector length.
    pub(crate) fn get_batch_with_dict_uninit<T: Clone + 'static>(
        &mut self,
        dict: &[T],
        out: &mut [std::mem::MaybeUninit<T>],
    ) -> Result<usize> {
        assert!(fused::native::<T>());
        let mut written = 0;
        while written < out.len() {
            if self.rle_left > 0 {
                let n = (out.len() - written).min(self.rle_left as usize);
                let id = self.current_value.ok_or_else(|| {
                    general_err!("parquet_data_error: not enough data for RLE decoding")
                })? as usize;
                let value = dict.get(id).ok_or_else(|| {
                    general_err!(
                        "dictionary index out of bounds: the len is {} but the index is {}",
                        dict.len(),
                        id
                    )
                })?;
                runs::fill(&mut out[written..], n, value);
                written += n;
                self.rle_left -= n as u32;
            } else if self.bit_packed_left > 0 {
                let reader = self
                    .bit_reader
                    .as_mut()
                    .ok_or_else(|| general_err!("bit_reader should be set"))?;
                let mut n = (out.len() - written).min(self.bit_packed_left as usize);
                // Packed runs hold whole groups of eight values. A cursor
                // inside a group decodes only up to the group's end, so that
                // the fused kernels take the rest of the run.
                let to_group = self.bit_packed_left as usize % 8;
                if to_group != 0 {
                    n = n.min(to_group);
                }
                if let Some(data) = reader.aligned_remaining()
                    && let Some(result) =
                        fused::decode(data, self.bit_width, dict, &mut out[written..written + n])
                {
                    let n = result?;
                    reader.skip(n, self.bit_width as usize);
                    written += n;
                    self.bit_packed_left -= n as u32;
                    continue;
                }
                let indices = self.index_buf.get_or_insert_with(|| Box::new([0; 1024]));
                let n = reader.get_batch(&mut indices[..n.min(1024)], self.bit_width as usize);
                if n == 0 {
                    self.bit_packed_left = 0;
                    continue;
                }
                let max_index = indices[..n].iter().map(|&i| i as u32).max().unwrap();
                if max_index as usize >= dict.len() {
                    // The reader is past these indices: fail the decoder, so
                    // that a retry cannot return the values after them.
                    return Err(self.fail(general_err!(
                        "dictionary index out of bounds: the len is {} but the index is {}",
                        dict.len(),
                        max_index
                    )));
                }
                for (slot, &id) in out[written..written + n].iter_mut().zip(&indices[..n]) {
                    // SAFETY: every unsigned ID is bounded by the maximum checked above.
                    slot.write(unsafe { dict.get_unchecked(id as u32 as usize) }.clone());
                }
                written += n;
                self.bit_packed_left -= n as u32;
            } else {
                let n = self.dictionary_runs(dict, &mut out[written..])?;
                written += n;
                if n == 0 && self.rle_left == 0 && self.bit_packed_left == 0 && !self.reload()? {
                    break;
                }
            }
        }
        Ok(written)
    }

    /// Ends decoding after an error. The reader may already have consumed
    /// part of the failed run, so a later call must not continue from its
    /// position: it returns an error until [`Self::set_data`] is called.
    #[cold]
    fn fail(&mut self, err: ParquetError) -> ParquetError {
        self.bit_reader = None;
        self.rle_left = 0;
        self.bit_packed_left = 0;
        self.current_value = None;
        err
    }

    /// Reads the next run header. On an error the decoder is failed (see
    /// [`Self::fail`]), so it never keeps the counter or the value of a
    /// previous run.
    #[inline]
    fn reload(&mut self) -> Result<bool> {
        self.reload_run().map_err(|err| self.fail(err))
    }

    #[inline]
    fn reload_run(&mut self) -> Result<bool> {
        let bit_reader = self.bit_reader.as_mut().ok_or_else(|| {
            general_err!("RLE decoder has no data: no page was set or an earlier read failed")
        })?;

        if let Some(indicator_value) = bit_reader.get_vlq_int() {
            // fastparquet adds padding to the end of pages. This is not spec-compliant
            // but is handled by the C++ implementation
            // <https://github.com/apache/arrow/blob/8074496cb41bc8ec8fe9fc814ca5576d89a6eb94/cpp/src/arrow/util/rle_encoding.h#L653>
            if indicator_value == 0 {
                return Ok(false);
            }
            let num_values = run_len(indicator_value as u64)?;
            if indicator_value & 1 == 1 {
                self.bit_packed_left = num_values;
            } else {
                let value_width = bit_util::ceil(self.bit_width as usize, u8::BITS as usize);
                let value = bit_reader.get_aligned::<u64>(value_width).ok_or_else(|| {
                    general_err!("parquet_data_error: not enough data for RLE decoding")
                })?;
                if self.bit_width < u64::BITS as u8 && value >= (1_u64 << self.bit_width) {
                    return Err(general_err!(
                        "Invalid or corrupted RLE value {value} exceeds bit width {}",
                        self.bit_width
                    ));
                }
                // Set the run only after its value is valid.
                self.rle_left = num_values;
                self.current_value = Some(value);
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

/// Gathers `indices` from `dict` into `out` for any value type, 16 indices at
/// a time. Returns an error for an index outside the dictionary.
#[inline]
fn gather_dictionary_chunks<T: Clone>(dict: &[T], idx: &[i32], out: &mut [T]) -> Result<()> {
    #[cold]
    #[inline(never)]
    fn oob(max_idx: u32, dict_len: usize) -> ParquetError {
        general_err!(
            "dictionary index out of bounds: the len is {} but the index is {}",
            dict_len,
            max_idx
        )
    }
    const CHUNK: usize = 16;
    let dict_len = dict.len();
    let (out_chunks, out_chunks_remainder) = out.as_chunks_mut::<CHUNK>();
    let (idx_chunks, idx_chunks_remainder) = idx.as_chunks::<CHUNK>();
    for (out_chunk, idx_chunk) in out_chunks.iter_mut().zip(idx_chunks) {
        // u32 max-reduction instead of `.all(|&i| ..)`: `.all`
        // short-circuits and blocks autovectorisation. Negative
        // i32 cast to u32 becomes a large value so the bounds
        // check still rejects it.
        let max_idx = idx_chunk.iter().fold(0u32, |acc, &i| acc.max(i as u32));
        if (max_idx as usize) >= dict_len {
            return Err(oob(max_idx, dict_len));
        }
        for (b, i) in out_chunk.iter_mut().zip(idx_chunk.iter()) {
            // SAFETY: all indices checked above to be in bounds
            b.clone_from(unsafe { dict.get_unchecked(*i as usize) });
        }
    }
    for (b, i) in out_chunks_remainder.iter_mut().zip(idx_chunks_remainder) {
        let dict_idx = *i as usize;
        if dict_idx >= dict_len {
            return Err(oob(*i as u32, dict_len));
        }
        // SAFETY: bounds checked above
        b.clone_from(unsafe { dict.get_unchecked(dict_idx) });
    }
    Ok(())
}

/// Validate one unpacked scratch batch before gathering its values. A complete
/// reduction allows wider vectorization than checking each 16-value group.
#[inline(always)]
fn gather_primitive_dictionary_impl<T: Clone>(
    dict: &[T],
    indices: &[i32],
    out: &mut [T],
) -> Result<()> {
    if let Some(max_index) = indices.iter().map(|&i| i as u32).max()
        && max_index as usize >= dict.len()
    {
        return Err(general_err!(
            "dictionary index out of bounds: the len is {} but the index is {}",
            dict.len(),
            max_index
        ));
    }
    for (value, &index) in out.iter_mut().zip(indices) {
        // SAFETY: the maximum unsigned index is below dict.len(). Interpret
        // the unpacked ID as u32 in both validation and lookup.
        value.clone_from(unsafe { dict.get_unchecked(index as u32 as usize) });
    }
    Ok(())
}

/// # Safety
/// AVX2 must be available. The body is the checked portable gather, which
/// validates every index before an unchecked lookup.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn gather_primitive_dictionary_avx2<T: Clone>(
    dict: &[T],
    indices: &[i32],
    out: &mut [T],
) -> Result<()> {
    gather_primitive_dictionary_impl(dict, indices, out)
}

fn gather_primitive_dictionary<T: Clone>(dict: &[T], indices: &[i32], out: &mut [T]) -> Result<()> {
    // A forced scalar test tier also keeps this gather portable.
    #[cfg(all(test, target_arch = "x86_64"))]
    let allow_avx2 =
        crate::util::native::dispatch::tier() != crate::util::native::dispatch::Tier::Scalar;
    #[cfg(not(all(test, target_arch = "x86_64")))]
    let allow_avx2 = true;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if allow_avx2 && indices.len() >= 32 && std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is checked at runtime; the function checks all dictionary
        // indices before reading values and uses bounded output iteration.
        return unsafe { gather_primitive_dictionary_avx2(dict, indices, out) };
    }
    gather_primitive_dictionary_impl(dict, indices, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::util::bit_util::ceil;
    use rand::{self, RngExt, SeedableRng, distr::StandardUniform, rng};

    const MAX_WIDTH: usize = 32;

    #[test]
    fn test_review_dictionary_errors_and_reuse() {
        crate::util::native::dispatch::for_each_tier(|| {
            for width in [2, 17, 24, 32] {
                let mut encoder = RleEncoder::new(width, 4096);
                for i in 0..64 {
                    encoder.put(if i % 2 == 0 { 0 } else { 3 });
                }
                let mut decoder = RleDecoder::new(width);
                decoder.set_data(Bytes::from(encoder.consume())).unwrap();
                let error = decoder
                    .append_dictionary(&[42i64], &mut Vec::new(), 64)
                    .unwrap_err();
                assert_eq!(
                    error.to_string(),
                    "Parquet error: dictionary index out of bounds: the len is 1 but the index is 3"
                );
            }
            let mut decoder = RleDecoder::new(8);
            // A repeated run header without its value leaves reload in error.
            assert!(decoder.set_data(Bytes::from_static(&[2])).is_err());
            let mut out = vec![-17i64];
            assert!(decoder.append_dictionary(&[42], &mut out, 1).is_err());
            assert!(decoder.append_dictionary(&[42], &mut out, 1).is_err());
            assert_eq!(out, [-17]);
        });
    }

    #[test]
    fn test_append_dictionary_initialized_prefix_and_error() {
        crate::util::native::dispatch::for_each_tier(|| {
            for width in 1..=32 {
                let dict: Vec<i64> = (0..(1usize
                    << if (18..=24).contains(&width) && !cfg!(miri) {
                        width
                    } else {
                        width.min(10)
                    }))
                    .map(|i| i as i64 - 513)
                    .collect();
                let mut encoder = RleEncoder::new(width, 65536);
                for i in 0..4099 {
                    encoder.put(if i == 0 {
                        (dict.len() - 1) as u64
                    } else {
                        (i * 7919 % dict.len()) as u64
                    });
                }
                let data = Bytes::from(encoder.consume());
                let mut decoder = RleDecoder::new(width);
                decoder.set_data(data.clone()).unwrap();
                let mut scalar = RleDecoder::new(width);
                scalar.set_data(data).unwrap();
                let mut actual = vec![i64::MIN];
                for count in [1, 7, 16, 31, 1024, 2048, 2048, 16] {
                    let mut ids = vec![0i32; count];
                    let n = scalar.get_batch(&mut ids).unwrap();
                    let start = actual.len();
                    assert_eq!(
                        decoder
                            .append_dictionary(&dict, &mut actual, count)
                            .unwrap(),
                        n
                    );
                    assert_eq!(actual[0], i64::MIN);
                    assert_eq!(actual.len(), start + n);
                    for i in 0..n {
                        assert_eq!(actual[start + i], dict[ids[i] as u32 as usize]);
                    }
                }
                for bad in [1, 15, 16, 31, 1023] {
                    let mut encoder = RleEncoder::new(width, 65536);
                    for i in 0..1024 {
                        encoder.put(u64::from(i == bad));
                    }
                    let mut decoder = RleDecoder::new(width);
                    decoder.set_data(Bytes::from(encoder.consume())).unwrap();
                    let mut actual = vec![i64::MIN];
                    assert!(
                        decoder
                            .append_dictionary(&[42i64], &mut actual, 1024)
                            .is_err()
                    );
                    assert_eq!(actual, [i64::MIN]);
                }
            }
        });
    }

    #[test]
    fn test_primitive_dictionary_gather_dispatch_boundaries() {
        let dict = [i64::MIN, -1, 0, i64::MAX];
        for len in 0..=65 {
            let indices: Vec<i32> = (0..len).map(|i| (i % dict.len()) as i32).collect();
            let expected: Vec<i64> = indices.iter().map(|&i| dict[i as usize]).collect();
            let mut scalar = vec![0; len];
            let mut dispatched = vec![0; len];
            gather_primitive_dictionary_impl(&dict, &indices, &mut scalar).unwrap();
            gather_primitive_dictionary(&dict, &indices, &mut dispatched).unwrap();
            assert_eq!(scalar, expected);
            assert_eq!(dispatched, expected);
        }
    }

    #[test]
    fn test_primitive_dictionary_gather_random_slices() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5a31_f022);
        for _ in 0..2048 {
            let dict: Vec<i64> = (0..rng.random_range(0..4097))
                .map(|_| rng.random())
                .collect();
            let indices: Vec<i32> = (0..rng.random_range(0..1031))
                .map(|_| {
                    if !dict.is_empty() && rng.random_ratio(255, 256) {
                        rng.random_range(0..dict.len()) as i32
                    } else {
                        rng.random()
                    }
                })
                .collect();
            let expected = indices
                .iter()
                .map(|&i| dict.get(i as usize).copied())
                .collect::<Option<Vec<_>>>();
            let mut actual = vec![0; indices.len()];
            let result = gather_primitive_dictionary(&dict, &indices, &mut actual);
            match expected {
                Some(expected) => {
                    result.unwrap();
                    assert_eq!(actual, expected);
                }
                None => assert!(result.is_err()),
            }
        }
    }

    // Compare dictionary materialization with independently decoded indices,
    // including partial batches, mixed RLE/packed runs, and all index widths.
    #[test]
    fn test_dictionary_gather_random_batches() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5a31);
        for width in 0..=32 {
            let dict_len = (1usize << width.min(12)).min(3073);
            let dict: Vec<i64> = (0..dict_len).map(|_| rng.random()).collect();
            let mut encoder = RleEncoder::new(width, 65536);
            for i in 0..4099 {
                let index = if i % 257 < 64 {
                    0
                } else {
                    rng.random_range(0..dict_len)
                };
                encoder.put(index as u64);
            }
            let data = Bytes::from(encoder.consume());
            let mut gather = RleDecoder::new(width);
            let mut indices = RleDecoder::new(width);
            gather.set_data(data.clone()).unwrap();
            indices.set_data(data).unwrap();
            let mut remaining = 4099;
            while remaining > 0 {
                let n = rng.random_range(1..=remaining.min(1031));
                if rng.random_ratio(1, 5) {
                    assert_eq!(gather.skip(n).unwrap(), indices.skip(n).unwrap());
                } else {
                    let mut ids = vec![0i32; n];
                    let mut actual = vec![0i64; n];
                    let count = indices.get_batch(&mut ids).unwrap();
                    assert_eq!(
                        gather.get_batch_with_dict(&dict, &mut actual, n).unwrap(),
                        count
                    );
                    assert_eq!(
                        &actual[..count],
                        ids[..count]
                            .iter()
                            .map(|&i| dict[i as usize])
                            .collect::<Vec<_>>()
                    );
                }
                remaining -= n;
            }
        }
    }

    #[test]
    fn test_dictionary_gather_rejects_invalid_indices() {
        for invalid in [17u64, i32::MAX as u64, u32::MAX as u64] {
            for position in [0, 15, 16, 31, 511, 1023] {
                let mut encoder = RleEncoder::new(32, 16384);
                for i in 0..1024 {
                    encoder.put(if i == position {
                        invalid
                    } else {
                        (i % 17) as u64
                    });
                }
                let mut decoder = RleDecoder::new(32);
                decoder.set_data(Bytes::from(encoder.consume())).unwrap();
                let err = decoder
                    .get_batch_with_dict(&[0i64; 17], &mut [0i64; 1024], 1024)
                    .unwrap_err();
                assert!(err.to_string().contains("dictionary index out of bounds"));
            }
        }
    }

    #[test]
    fn test_rle_decode_int32() {
        // Test data: 0-7 with bit width 3
        // 00000011 10001000 11000110 11111010
        let data = vec![0x03, 0x88, 0xC6, 0xFA];
        let mut decoder = RleDecoder::new(3);
        decoder.set_data(data.into()).unwrap();
        let mut buffer = vec![0; BIT_PACK_GROUP_SIZE];
        let expected = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let result = decoder.get_batch::<i32>(&mut buffer);
        assert!(result.is_ok());
        assert_eq!(buffer, expected);
    }

    #[test]
    fn test_rle_skip_int32() {
        // Test data: 0-7 with bit width 3
        // 00000011 10001000 11000110 11111010
        let data = vec![0x03, 0x88, 0xC6, 0xFA];
        let mut decoder = RleDecoder::new(3);
        decoder.set_data(data.into()).unwrap();
        let expected = vec![2, 3, 4, 5, 6, 7];
        let skipped = decoder.skip(2).expect("skipping values");
        assert_eq!(skipped, 2);

        let mut buffer = vec![0; 6];
        let remaining = decoder
            .get_batch::<i32>(&mut buffer)
            .expect("getting remaining");
        assert_eq!(remaining, 6);
        assert_eq!(buffer, expected);
    }

    #[test]
    fn test_skip_and_count() {
        for width in [0, 1, 2, 15, 64] {
            let max = if width == 64 {
                u64::MAX
            } else {
                (1_u64 << width) - 1
            };
            let values: Vec<u64> = std::iter::repeat_n(0, 1024)
                .chain((0..128).map(|i| i & max))
                .chain(std::iter::repeat_n(max, 2056))
                .collect();
            let mut encoder = RleEncoder::new(width, 1024);
            for &value in &values {
                encoder.put(value);
            }
            let data = Bytes::from(encoder.consume());
            for value in [0, max] {
                for prefix in [0, 1, 1023, 1024, 1027, 1152] {
                    for requested in [0, 1, 7, 1024, usize::MAX] {
                        let mut decoder = RleDecoder::new(width);
                        decoder.set_data(data.clone()).unwrap();
                        let mut read = vec![0; prefix];
                        assert_eq!(decoder.get_batch(&mut read).unwrap(), prefix);
                        assert_eq!(read, values[..prefix]);
                        let skipped = requested.min(values.len() - prefix);
                        let count = values[prefix..prefix + skipped]
                            .iter()
                            .filter(|&&v| v == value)
                            .count();
                        assert_eq!(
                            decoder.skip_and_count(requested, value).unwrap(),
                            (count, skipped),
                            "width={width}, value={value}, prefix={prefix}, requested={requested}"
                        );
                        let mut rest = vec![0; values.len() - prefix - skipped];
                        assert_eq!(decoder.get_batch(&mut rest).unwrap(), rest.len());
                        assert_eq!(rest, values[prefix + skipped..]);
                        assert_eq!(decoder.skip_and_count(1, value).unwrap(), (0, 0));
                    }
                }
            }
        }
    }

    #[test]
    fn test_skip_and_count_signed() {
        let data = Bytes::from_static(&[3, 255, 0, 128, 255, 1, 2, 254, 255]);
        let mut decoder = RleDecoder::new(8);
        decoder.set_data(data.clone()).unwrap();
        assert_eq!(decoder.skip_and_count(8, -1_i8).unwrap(), (3, 8));
        decoder.set_data(data.clone()).unwrap();
        assert_eq!(decoder.skip_and_count(8, -1_i16).unwrap(), (0, 8));
        decoder.set_data(data).unwrap();
        assert_eq!(decoder.skip_and_count(8, 511_i16).unwrap(), (0, 8));
    }

    #[test]
    fn test_skip_and_count_truncated() {
        for width in [1, 8] {
            // Header declares 24 values, but the final packed run is truncated.
            let mut data = vec![7];
            data.extend(std::iter::repeat_n(0xff, if width == 1 { 2 } else { 20 }));
            let mut decoder = RleDecoder::new(width);
            decoder.set_data(data.into()).unwrap();
            let count = if width == 1 { 16 } else { 20 };
            assert_eq!(
                decoder
                    .skip_and_count(usize::MAX, (1_u64 << width) - 1)
                    .unwrap(),
                (count, count)
            );
            assert_eq!(decoder.skip_and_count(1, 0_u64).unwrap(), (0, 0));
        }
        let mut decoder = RleDecoder::new(1);
        // A valid two-value run followed by a missing RLE value.
        decoder.set_data(Bytes::from_static(&[4, 1, 4])).unwrap();
        assert!(decoder.skip_and_count(3, 1_u64).is_err());
        decoder.set_data(Bytes::from_static(&[4, 1, 4])).unwrap();
        assert_eq!(decoder.skip_and_count(2, 1_u64).unwrap(), (2, 2));
        assert!(decoder.skip_and_count(1, 1_u64).is_err());
    }

    #[test]
    fn test_skip_and_count_long_run() {
        let mut writer = BitWriter::new(8);
        writer.put_vlq_int(MAX_RUN_LEN << 1);
        writer.put_aligned(1_u8, 1);
        let mut decoder = RleDecoder::new(1);
        decoder.set_data(writer.consume().into()).unwrap();
        // Must not expand this run or allocate scratch space proportional to it.
        assert_eq!(
            decoder.skip_and_count(usize::MAX, 1_u64).unwrap(),
            (MAX_RUN_LEN as usize, MAX_RUN_LEN as usize)
        );
        assert_eq!(decoder.skip_and_count(1, 1_u64).unwrap(), (0, 0));
    }

    #[test]
    fn test_rle_consume_flush_buffer() {
        let data = vec![1, 1, 1, 2, 2, 3, 3, 3];
        let mut encoder1 = RleEncoder::new(3, 256);
        let mut encoder2 = RleEncoder::new(3, 256);
        for value in data {
            encoder1.put(value as u64);
            encoder2.put(value as u64);
        }
        let res1 = encoder1.flush_buffer();
        let res2 = encoder2.consume();
        assert_eq!(res1, &res2[..]);
    }

    #[test]
    fn test_rle_decode_bool() {
        // RLE test data: 50 1s followed by 50 0s
        // 01100100 00000001 01100100 00000000
        let data1 = vec![0x64, 0x01, 0x64, 0x00];

        // Bit-packing test data: alternating 1s and 0s, 100 total
        // 100 / 8 = 13 groups
        // 00011011 10101010 ... 00001010
        let data2 = vec![
            0x1B, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0x0A,
        ];

        let mut decoder = RleDecoder::new(1);
        decoder.set_data(data1.into()).unwrap();
        let mut buffer = vec![false; 100];
        let mut expected = vec![];
        for i in 0..100 {
            if i < 50 {
                expected.push(true);
            } else {
                expected.push(false);
            }
        }
        let result = decoder.get_batch::<bool>(&mut buffer);
        assert!(result.is_ok());
        assert_eq!(buffer, expected);

        decoder.set_data(data2.into()).unwrap();
        let mut buffer = vec![false; 100];
        let mut expected = vec![];
        for i in 0..100 {
            if i % 2 == 0 {
                expected.push(false);
            } else {
                expected.push(true);
            }
        }
        let result = decoder.get_batch::<bool>(&mut buffer);
        assert!(result.is_ok());
        assert_eq!(buffer, expected);
    }

    #[test]
    fn test_rle_skip_bool() {
        // RLE test data: 50 1s followed by 50 0s
        // 01100100 00000001 01100100 00000000
        let data1 = vec![0x64, 0x01, 0x64, 0x00];

        // Bit-packing test data: alternating 1s and 0s, 100 total
        // 100 / 8 = 13 groups
        // 00011011 10101010 ... 00001010
        let data2 = vec![
            0x1B, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0x0A,
        ];

        let mut decoder = RleDecoder::new(1);
        decoder.set_data(data1.into()).unwrap();
        let mut buffer = vec![true; 50];
        let expected = vec![false; 50];

        let skipped = decoder.skip(50).expect("skipping first 50");
        assert_eq!(skipped, 50);
        let remainder = decoder
            .get_batch::<bool>(&mut buffer)
            .expect("getting remaining 50");
        assert_eq!(remainder, 50);
        assert_eq!(buffer, expected);

        decoder.set_data(data2.into()).unwrap();
        let mut buffer = vec![false; 50];
        let mut expected = vec![];
        for i in 0..50 {
            if i % 2 == 0 {
                expected.push(false);
            } else {
                expected.push(true);
            }
        }
        let skipped = decoder.skip(50).expect("skipping first 50");
        assert_eq!(skipped, 50);
        let remainder = decoder
            .get_batch::<bool>(&mut buffer)
            .expect("getting remaining 50");
        assert_eq!(remainder, 50);
        assert_eq!(buffer, expected);
    }

    #[test]
    fn test_rle_decode_with_dict_int32() {
        // Test RLE encoding: 3 0s followed by 4 1s followed by 5 2s
        // 00000110 00000000 00001000 00000001 00001010 00000010
        let dict = vec![10, 20, 30];
        let data = vec![0x06, 0x00, 0x08, 0x01, 0x0A, 0x02];
        let mut decoder = RleDecoder::new(3);
        decoder.set_data(data.into()).unwrap();
        let mut buffer = vec![0; 12];
        let expected = vec![10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30, 30];
        let result = decoder.get_batch_with_dict::<i32>(&dict, &mut buffer, 12);
        assert!(result.is_ok());
        assert_eq!(buffer, expected);

        // Test bit-pack encoding: 345345345455 (2 groups: 8 and 4)
        // 011 100 101 011 100 101 011 100 101 100 101 101
        // 00000011 01100011 11000111 10001110 00000011 01100101 00001011
        let dict = vec!["aaa", "bbb", "ccc", "ddd", "eee", "fff"];
        let data = vec![0x03, 0x63, 0xC7, 0x8E, 0x03, 0x65, 0x0B];
        let mut decoder = RleDecoder::new(3);
        decoder.set_data(data.into()).unwrap();
        let mut buffer = vec![""; 12];
        let expected = vec![
            "ddd", "eee", "fff", "ddd", "eee", "fff", "ddd", "eee", "fff", "eee", "fff", "fff",
        ];
        let result =
            decoder.get_batch_with_dict::<&str>(dict.as_slice(), buffer.as_mut_slice(), 12);
        assert!(result.is_ok());
        assert_eq!(buffer, expected);
    }

    #[test]
    fn test_rle_skip_dict() {
        // Test RLE encoding: 3 0s followed by 4 1s followed by 5 2s
        // 00000110 00000000 00001000 00000001 00001010 00000010
        let dict = vec![10, 20, 30];
        let data = vec![0x06, 0x00, 0x08, 0x01, 0x0A, 0x02];
        let mut decoder = RleDecoder::new(3);
        decoder.set_data(data.into()).unwrap();
        let mut buffer = vec![0; 10];
        let expected = vec![10, 20, 20, 20, 20, 30, 30, 30, 30, 30];
        let skipped = decoder.skip(2).expect("skipping two values");
        assert_eq!(skipped, 2);
        let remainder = decoder
            .get_batch_with_dict::<i32>(&dict, &mut buffer, 10)
            .expect("getting remainder");
        assert_eq!(remainder, 10);
        assert_eq!(buffer, expected);

        // Test bit-pack encoding: 345345345455 (2 groups: 8 and 4)
        // 011 100 101 011 100 101 011 100 101 100 101 101
        // 00000011 01100011 11000111 10001110 00000011 01100101 00001011
        let dict = vec!["aaa", "bbb", "ccc", "ddd", "eee", "fff"];
        let data = vec![0x03, 0x63, 0xC7, 0x8E, 0x03, 0x65, 0x0B];
        let mut decoder = RleDecoder::new(3);
        decoder.set_data(data.into()).unwrap();
        let mut buffer = vec![""; BIT_PACK_GROUP_SIZE];
        let expected = vec!["eee", "fff", "ddd", "eee", "fff", "eee", "fff", "fff"];
        let skipped = decoder.skip(4).expect("skipping four values");
        assert_eq!(skipped, 4);
        let remainder = decoder
            .get_batch_with_dict::<&str>(
                dict.as_slice(),
                buffer.as_mut_slice(),
                BIT_PACK_GROUP_SIZE,
            )
            .expect("getting remainder");
        assert_eq!(remainder, BIT_PACK_GROUP_SIZE);
        assert_eq!(buffer, expected);
    }

    fn validate_rle(
        values: &[i64],
        bit_width: u8,
        expected_encoding: Option<&[u8]>,
        expected_len: i32,
    ) {
        let buffer_len = 64 * 1024;
        let mut encoder = RleEncoder::new(bit_width, buffer_len);
        for v in values {
            encoder.put(*v as u64)
        }
        let buffer: Bytes = encoder.consume().into();
        if expected_len != -1 {
            assert_eq!(buffer.len(), expected_len as usize);
        }
        if let Some(b) = expected_encoding {
            assert_eq!(buffer.as_ref(), b);
        }

        // Verify read
        let mut decoder = RleDecoder::new(bit_width);
        decoder.set_data(buffer.clone()).unwrap();
        for v in values {
            let val: i64 = decoder
                .get()
                .expect("get() should be OK")
                .expect("get() should return more value");
            assert_eq!(val, *v);
        }

        // Verify batch read
        decoder.set_data(buffer).unwrap();
        let mut values_read: Vec<i64> = vec![0; values.len()];
        decoder
            .get_batch(&mut values_read[..])
            .expect("get_batch() should be OK");
        assert_eq!(&values_read[..], values);
    }

    #[test]
    fn test_rle_specific_sequences() {
        let mut expected_buffer = Vec::new();
        let mut values = vec![0; 50];
        values.resize(100, 1);

        expected_buffer.push(50 << 1);
        expected_buffer.push(0);
        expected_buffer.push(50 << 1);
        expected_buffer.push(1);

        for width in 1..9 {
            validate_rle(&values[..], width, Some(&expected_buffer[..]), 4);
        }
        for width in 9..MAX_WIDTH + 1 {
            validate_rle(
                &values[..],
                width as u8,
                None,
                2 * (1 + bit_util::ceil(width as i64, u8::BITS as i64) as i32),
            );
        }

        // Test 100 0's and 1's alternating
        values.clear();
        expected_buffer.clear();
        for i in 0..101 {
            values.push(i % 2);
        }
        let num_groups = bit_util::ceil(100, BIT_PACK_GROUP_SIZE) as u8;
        expected_buffer.push((num_groups << 1) | 1);
        expected_buffer.resize(
            expected_buffer.len() + 100 / BIT_PACK_GROUP_SIZE,
            0b10101010,
        );

        // For the last 4 0 and 1's, padded with 0.
        expected_buffer.push(0b00001010);
        validate_rle(
            &values,
            1,
            Some(&expected_buffer[..]),
            1 + num_groups as i32,
        );
        for width in 2..MAX_WIDTH + 1 {
            let num_values = bit_util::ceil(100, BIT_PACK_GROUP_SIZE) * BIT_PACK_GROUP_SIZE;
            validate_rle(
                &values,
                width as u8,
                None,
                1 + bit_util::ceil(width as i64 * num_values as i64, u8::BITS as i64) as i32,
            );
        }
    }

    #[test]
    fn test_rle_bit_packed_run_cap() {
        // A bit-packed run holds at most MAX_GROUPS_PER_BIT_PACKED_RUN - 1 = 63 groups
        // (504 values), which keeps every run header a single byte. 1024 alternating
        // values must split into runs of 504, 504 and 16 values. A round-trip cannot
        // pin this down (longer runs with multi-byte headers also decode fine), so
        // check the exact bytes.
        let values: Vec<i64> = (0..1024).map(|i| i % 2).collect();

        let mut expected_buffer = Vec::new();
        for num_groups in [63u8, 63, 2] {
            expected_buffer.push((num_groups << 1) | 1);
            expected_buffer.resize(expected_buffer.len() + num_groups as usize, 0b10101010);
        }

        validate_rle(&values, 1, Some(&expected_buffer), 131);
    }

    #[test]
    fn test_put_batch_matches_put() {
        // `put_batch` must produce byte-identical output to a `put` loop, for any
        // input, any split of the input into batches, and mixed use of both APIs
        let mut rng = rng();
        for width in [1, 2, 5, 8, 17, 32] {
            let mask = (1u64 << width) - 1;
            for _ in 0..20 {
                // Mix random stretches with runs long enough to trigger RLE, sized to
                // also cross the bit-packed run cap
                let mut values: Vec<u64> = Vec::new();
                while values.len() < 1300 {
                    let n = rng.random_range(1..40);
                    if rng.random_bool(0.5) {
                        let v = rng.random::<u64>() & mask;
                        values.extend(std::iter::repeat_n(v, n));
                    } else {
                        values.extend((0..n).map(|_| rng.random::<u64>() & mask));
                    }
                }

                let mut expected = RleEncoder::new(width as u8, 1024);
                for &v in &values {
                    expected.put(v);
                }

                let mut actual = RleEncoder::new(width as u8, 1024);
                let mut remaining = &values[..];
                while !remaining.is_empty() {
                    let n = rng.random_range(1..=remaining.len().min(300));
                    if rng.random_bool(0.25) {
                        for &v in &remaining[..n] {
                            actual.put(v);
                        }
                    } else {
                        actual.put_batch(&remaining[..n]);
                    }
                    remaining = &remaining[n..];
                }

                assert_eq!(expected.consume(), actual.consume(), "width = {width}");
            }
        }
    }

    // `validate_rle` on `num_vals` with width `bit_width`. If `value` is -1, that value
    // is used, otherwise alternating values are used.
    fn test_rle_values(bit_width: usize, num_vals: usize, value: i32) {
        let mod_val = if bit_width == 64 {
            1
        } else {
            1u64 << bit_width
        };
        let mut values: Vec<i64> = vec![];
        for v in 0..num_vals {
            let val = if value == -1 {
                v as i64 % mod_val as i64
            } else {
                value as i64
            };
            values.push(val);
        }
        validate_rle(&values, bit_width as u8, None, -1);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Takes too long
    fn test_values() {
        for width in 1..MAX_WIDTH + 1 {
            test_rle_values(width, 1, -1);
            test_rle_values(width, 1024, -1);
            test_rle_values(width, 1024, 0);
            test_rle_values(width, 1024, 1);
        }
    }

    #[test]
    fn test_truncated_rle() {
        // The final bit packed run within a page may not be a multiple of 8 values
        // Unfortunately the specification stores `(bit-packed-run-len) / 8`
        // This means we don't necessarily know how many values are present
        // and some writers may not add padding to compensate for this ambiguity

        // Bit pack encode 20 values with a bit width of 8
        let mut data: Vec<u8> = vec![
            (3 << 1) | 1, // bit-packed run of 3 * 8
        ];
        data.extend(std::iter::repeat_n(0xFF, 20));
        let data: Bytes = data.into();

        let mut decoder = RleDecoder::new(8);
        decoder.set_data(data.clone()).unwrap();

        let mut output = vec![0_u16; 100];
        let read = decoder.get_batch(&mut output).unwrap();

        assert_eq!(read, 20);
        assert!(output.iter().take(20).all(|x| *x == 255));

        // Reset decoder
        decoder.set_data(data).unwrap();

        let dict: Vec<u16> = (0..256).collect();
        let mut output = vec![0_u16; 100];
        let read = decoder
            .get_batch_with_dict(&dict, &mut output, 100)
            .unwrap();

        assert_eq!(read, 20);
        assert!(output.iter().take(20).all(|x| *x == 255));
    }

    #[test]
    fn test_rle_padded() {
        let values: Vec<i16> = vec![0, 1, 1, 3, 1, 0];
        let bit_width = 2;
        let buffer_len = RleEncoder::max_buffer_size(bit_width, values.len());
        let mut encoder = RleEncoder::new(bit_width, buffer_len + 1);
        for v in &values {
            encoder.put(*v as u64)
        }

        let mut buffer = encoder.consume();
        buffer.push(0);

        let mut decoder = RleDecoder::new(bit_width);
        decoder.set_data(buffer.into()).unwrap();

        // We don't always reliably know how many non-null values are contained in a page
        // and so the decoder must work correctly without a precise value count
        let mut actual_values: Vec<i16> = vec![0; 12];
        let r = decoder
            .get_batch(&mut actual_values)
            .expect("get_batch() should be OK");

        // Should decode BIT_PACK_GROUP_SIZE values despite only encoding 6 as length of
        // bit packed run is always a multiple of BIT_PACK_GROUP_SIZE
        assert_eq!(r, BIT_PACK_GROUP_SIZE);
        assert_eq!(actual_values[..6], values);
        assert_eq!(actual_values[6], 0);
        assert_eq!(actual_values[7], 0);
    }

    /// The encoder enters RLE accumulation mode exactly on the 8th consecutive
    /// identical value.
    #[test]
    fn test_is_accumulating_rle_boundary() {
        let bit_width = 2;
        let value = 1u64;

        // 7 identical values: not yet accumulating
        let mut enc = RleEncoder::new(bit_width, 256);
        for _ in 0..7 {
            enc.put(value);
        }
        assert!(
            !enc.is_accumulating_rle(value),
            "should not be accumulating after 7 values"
        );

        // 8th value tips into accumulation
        enc.put(value);
        assert!(
            enc.is_accumulating_rle(value),
            "should be accumulating after 8 values"
        );

        // extend_run from that state and verify the round-trip
        enc.extend_run(92); // total: 100 identical values
        let encoded = enc.consume();

        let mut dec = RleDecoder::new(bit_width);
        dec.set_data(encoded.into()).unwrap();
        let mut out = vec![0i32; 100];
        let n = dec.get_batch::<i32>(&mut out).unwrap();
        assert_eq!(n, 100);
        assert!(out.iter().all(|&v| v == value as i32));
    }

    #[test]
    fn test_long_run() {
        // This writer does not write runs longer than 504 values as this allows
        // encoding the run header as a single byte
        //
        // This tests that the decoder correctly handles longer runs

        let mut writer = BitWriter::new(1024);
        let bit_width = 1;

        // Choose a non-multiple of 8 larger than 1024 so that the length
        // of the run is ambiguous, as the encoding only stores `num_values / 8`
        let num_values = 2002;

        // bit-packed header
        let run_bytes = ceil(num_values * bit_width, u8::BITS as usize) as u64;
        writer.put_vlq_int((run_bytes << 1) | 1);
        for _ in 0..run_bytes {
            writer.put_aligned(0xFF_u8, 1);
        }
        let buffer: Bytes = writer.consume().into();

        let mut decoder = RleDecoder::new(1);
        decoder.set_data(buffer.clone()).unwrap();

        let mut decoded: Vec<i16> = vec![0; num_values];
        let r = decoder.get_batch(&mut decoded).unwrap();
        assert_eq!(r, num_values);
        assert_eq!(vec![1; num_values], decoded);

        decoder.set_data(buffer).unwrap();
        let r = decoder
            .get_batch_with_dict(&[0, 23], &mut decoded, num_values)
            .unwrap();
        assert_eq!(r, num_values);
        assert_eq!(vec![23; num_values], decoded);
    }

    #[test]
    fn test_run_len_limit() {
        let set_data = |indicator_value: u64| {
            let mut writer = BitWriter::new(16);
            writer.put_vlq_int(indicator_value);
            writer.put_aligned(1_u8, 1); // repeated value of an RLE run
            RleDecoder::new(1).set_data(writer.consume().into())
        };
        let max_groups = MAX_RUN_LEN / BIT_PACK_GROUP_SIZE as u64;

        // The longest runs allowed by the spec
        set_data(MAX_RUN_LEN << 1).unwrap();
        set_data((max_groups << 1) | 1).unwrap();

        for (indicator_value, expected_len) in [
            ((MAX_RUN_LEN + 1) << 1, "2147483648"),
            (((max_groups + 1) << 1) | 1, "2147483648"),
            // Overflows `i64` when converted to a number of values
            ((1 << 61) | 1, "9223372036854775808"),
            // Negative when decoded as `i64`
            (0xa4ff_ffff_ffff_ffff, "18446744073709551615"),
        ] {
            let err = set_data(indicator_value).unwrap_err();
            assert_eq!(
                err.to_string(),
                format!(
                    "Parquet error: Invalid or corrupted RLE run length {expected_len}. Max allowed is 2147483647"
                )
            );
        }
    }

    #[test]
    fn test_rle_specific_roundtrip() {
        let bit_width = 1;
        let values: Vec<i16> = vec![0, 1, 1, 1, 1, 0, 0, 0, 0, 1];
        let buffer_len = RleEncoder::max_buffer_size(bit_width, values.len());
        let mut encoder = RleEncoder::new(bit_width, buffer_len);
        for v in &values {
            encoder.put(*v as u64)
        }
        let buffer = encoder.consume();
        let mut decoder = RleDecoder::new(bit_width);
        decoder.set_data(Bytes::from(buffer)).unwrap();
        let mut actual_values: Vec<i16> = vec![0; values.len()];
        decoder
            .get_batch(&mut actual_values)
            .expect("get_batch() should be OK");
        assert_eq!(actual_values, values);
    }

    fn test_round_trip(values: &[i32], bit_width: u8) {
        let buffer_len = 64 * 1024;
        let mut encoder = RleEncoder::new(bit_width, buffer_len);
        for v in values {
            encoder.put(*v as u64)
        }

        let buffer = Bytes::from(encoder.consume());

        // Verify read
        let mut decoder = RleDecoder::new(bit_width);
        decoder.set_data(buffer.clone()).unwrap();
        for v in values {
            let val = decoder
                .get::<i32>()
                .expect("get() should be OK")
                .expect("get() should return value");
            assert_eq!(val, *v);
        }

        // Verify batch read
        let mut decoder = RleDecoder::new(bit_width);
        decoder.set_data(buffer).unwrap();
        let mut values_read: Vec<i32> = vec![0; values.len()];
        decoder
            .get_batch(&mut values_read[..])
            .expect("get_batch() should be OK");
        assert_eq!(&values_read[..], values);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Takes too long
    fn test_random() {
        let seed_len = 32;
        let niters = 50;
        let ngroups = 1000;
        let max_group_size = 15;
        let mut values = vec![];

        for _ in 0..niters {
            values.clear();
            let rng = rng();
            let seed_vec: Vec<u8> = rng
                .sample_iter::<u8, _>(&StandardUniform)
                .take(seed_len)
                .collect();
            let mut seed = [0u8; 32];
            seed.copy_from_slice(&seed_vec[0..seed_len]);
            let mut r#gen = rand::rngs::StdRng::from_seed(seed);

            let mut parity = false;
            for _ in 0..ngroups {
                let mut group_size = r#gen.random_range(1..20);
                if group_size > max_group_size {
                    group_size = 1;
                }
                for _ in 0..group_size {
                    values.push(parity as i32);
                }
                parity = !parity;
            }
            let bit_width = bit_util::num_required_bits(values.len() as u64);
            assert!(bit_width < 64);
            test_round_trip(&values[..], bit_width);
        }
    }
}

#[cfg(test)]
mod invalid_input_tests {
    use super::*;

    /// Bit width 1: an RLE run of two 1s, then an RLE run of three values
    /// whose value byte 0xff does not fit the bit width.
    fn rle_run_then_invalid_value() -> Bytes {
        Bytes::from_static(&[2 << 1, 1, 3 << 1, 0xff])
    }

    /// A failed reload must not leave the counter of the new run with the
    /// value of the previous run. Before, a retry returned three more 1s.
    #[test]
    fn retry_after_an_invalid_run_value_fails() {
        let mut decoder = RleDecoder::new(1);
        decoder.set_data(rle_run_then_invalid_value()).unwrap();
        let mut out = [0i32; 8];
        let err = decoder.get_batch(&mut out).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Parquet error: Invalid or corrupted RLE value 255 exceeds bit width 1"
        );
        for _ in 0..2 {
            assert!(decoder.get_batch(&mut out).is_err());
            assert!(decoder.skip(8).is_err());
        }

        // The dictionary paths: native primitive and any other type.
        let mut decoder = RleDecoder::new(1);
        decoder.set_data(rle_run_then_invalid_value()).unwrap();
        let mut out = [0i64; 8];
        assert!(decoder.get_batch_with_dict(&[10, 20], &mut out, 8).is_err());
        assert!(decoder.get_batch_with_dict(&[10, 20], &mut out, 8).is_err());

        let dict = ["a".to_string(), "b".to_string()];
        let mut decoder = RleDecoder::new(1);
        decoder.set_data(rle_run_then_invalid_value()).unwrap();
        let mut out = vec![String::new(); 8];
        assert!(decoder.get_batch_with_dict(&dict, &mut out, 8).is_err());
        assert!(decoder.get_batch_with_dict(&dict, &mut out, 8).is_err());

        // New page data makes the decoder usable again.
        let mut decoder = RleDecoder::new(1);
        decoder.set_data(rle_run_then_invalid_value()).unwrap();
        let mut out = [0i32; 8];
        assert!(decoder.get_batch(&mut out).is_err());
        decoder.set_data(Bytes::from_static(&[2 << 1, 1])).unwrap();
        assert_eq!(decoder.get_batch(&mut out).unwrap(), 2);
        assert_eq!(&out[..2], &[1, 1]);
    }

    /// Bit width 3: one bit-packed group of eight indices 0, 1, 5, 0, 0, 0,
    /// 0, 0. Index 5 is outside a dictionary of two values.
    fn packed_group_with_invalid_index() -> Bytes {
        let packed: u32 = (1 << 3) | (5 << 6);
        Bytes::from(vec![(1 << 1) | 1, packed as u8, (packed >> 8) as u8, 0])
    }

    /// The scratch path reads the indices before it checks them. A retry
    /// after an index error must not return the values after them. Before,
    /// it returned the next four indices as valid values.
    #[test]
    fn retry_after_an_index_out_of_bounds_fails() {
        // Four values at a time: fewer than a fused kernel takes, so every
        // dispatch tier uses the scratch path.
        let mut decoder = RleDecoder::new(3);
        decoder.set_data(packed_group_with_invalid_index()).unwrap();
        let mut out = [0i32; 4];
        let err = decoder
            .get_batch_with_dict(&[10, 20], &mut out, 4)
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "Parquet error: dictionary index out of bounds: the len is 2 but the index is 5"
        );
        for _ in 0..2 {
            assert!(decoder.get_batch_with_dict(&[10, 20], &mut out, 4).is_err());
        }

        let mut decoder = RleDecoder::new(3);
        decoder.set_data(packed_group_with_invalid_index()).unwrap();
        let mut spare = Vec::<i32>::with_capacity(4);
        assert!(decoder.append_dictionary(&[10, 20], &mut spare, 4).is_err());
        assert!(spare.is_empty());
        assert!(decoder.append_dictionary(&[10, 20], &mut spare, 4).is_err());
        assert!(spare.is_empty());

        let dict = ["a".to_string(), "b".to_string()];
        let mut decoder = RleDecoder::new(3);
        decoder.set_data(packed_group_with_invalid_index()).unwrap();
        let mut out = vec![String::new(); 4];
        assert!(decoder.get_batch_with_dict(&dict, &mut out, 4).is_err());
        assert!(decoder.get_batch_with_dict(&dict, &mut out, 4).is_err());
    }
}
