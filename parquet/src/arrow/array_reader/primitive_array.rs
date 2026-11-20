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

use super::selected_primitive::SelectedPrimitive;
use crate::arrow::array_reader::{
    ArrayReader, check_selection, filter_selection, read_records, skip_records,
};
use crate::arrow::record_reader::RecordReader;
use crate::arrow::schema::parquet_to_arrow_field;
use crate::basic::Type as PhysicalType;
use crate::column::page::PageIterator;
use crate::data_type::{DataType, Int96};
use crate::errors::{ParquetError, Result};
use crate::schema::types::ColumnDescPtr;
use arrow_array::{
    Array, ArrayRef, BooleanArray, Date64Array, Decimal64Array, Decimal128Array, Decimal256Array,
    Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, PrimitiveArray,
    UInt8Array, UInt16Array, builder::PrimitiveDictionaryBuilder, cast::AsArray, downcast_integer,
    types::*,
};
use arrow_array::{
    TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
    TimestampSecondArray, UInt32Array, UInt64Array,
};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, ScalarBuffer, i256};
use arrow_schema::{DataType as ArrowType, TimeUnit};
use std::any::Any;
use std::sync::Arc;

/// Provides conversion from `Vec<T>` to `Buffer`
pub trait IntoBuffer {
    fn into_buffer(self, target_type: &ArrowType) -> Buffer;
}

macro_rules! native_buffer {
    ($($t:ty),*) => {
        $(impl IntoBuffer for Vec<$t> {
            fn into_buffer(self, _target_type: &ArrowType) -> Buffer {
                Buffer::from_vec(self)
            }
        })*
    };
}
native_buffer!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

impl IntoBuffer for Vec<bool> {
    fn into_buffer(self, _target_type: &ArrowType) -> Buffer {
        BooleanBuffer::from(self.as_slice()).into_inner()
    }
}

impl IntoBuffer for Vec<Int96> {
    fn into_buffer(self, target_type: &ArrowType) -> Buffer {
        let mut builder = Vec::with_capacity(self.len());
        match target_type {
            ArrowType::Timestamp(TimeUnit::Second, _) => {
                builder.extend(self.iter().map(|x| x.to_seconds()));
            }
            ArrowType::Timestamp(TimeUnit::Millisecond, _) => {
                builder.extend(self.iter().map(|x| x.to_millis()));
            }
            ArrowType::Timestamp(TimeUnit::Microsecond, _) => {
                builder.extend(self.iter().map(|x| x.to_micros()));
            }
            ArrowType::Timestamp(TimeUnit::Nanosecond, _) => {
                builder.extend(self.iter().map(|x| x.to_nanos()));
            }
            _ => unreachable!("Invalid target_type for Int96."),
        }
        Buffer::from_vec(builder)
    }
}

/// Primitive array readers are leaves of array reader tree. They accept page iterator
/// and read them into primitive arrays.
pub struct PrimitiveArrayReader<T>
where
    T: DataType,
    T::T: Copy + Default,
    Vec<T::T>: IntoBuffer,
{
    data_type: ArrowType,
    pages: Box<dyn PageIterator>,
    def_levels_buffer: Option<Vec<i16>>,
    rep_levels_buffer: Option<Vec<i16>>,
    record_reader: RecordReader<T>,
    selection_descr: Option<ColumnDescPtr>,
    selected: Option<Box<SelectedPrimitive<T>>>,
    full_started: bool,
}

impl<T> PrimitiveArrayReader<T>
where
    T: DataType,
    T::T: Copy + Default,
    Vec<T::T>: IntoBuffer,
{
    /// Construct primitive array reader.
    ///
    /// `padding_threshold` controls how null padding is applied. When
    /// `None`, the reader pads all null positions (full padding). When
    /// `Some(threshold)`, entries with `def < threshold` are excluded
    /// from the value buffer (selective padding for list children).
    pub fn new(
        pages: Box<dyn PageIterator>,
        column_desc: ColumnDescPtr,
        arrow_type: Option<ArrowType>,
        batch_size: usize,
        padding_threshold: Option<i16>,
    ) -> Result<Self> {
        // Check if Arrow type is specified, else create it from Parquet type
        let data_type = match arrow_type {
            Some(t) => t,
            None => parquet_to_arrow_field(column_desc.as_ref())?
                .data_type()
                .clone(),
        };

        let selection_descr = (column_desc.max_rep_level() == 0
            && (column_desc.max_def_level() == 0
                || (column_desc.max_def_level() == 1 && column_desc.self_type().is_optional()))
            && padding_threshold.is_none()
            && matches!(
                T::get_physical_type(),
                PhysicalType::INT32
                    | PhysicalType::INT64
                    | PhysicalType::FLOAT
                    | PhysicalType::DOUBLE
            )
            && !matches!(data_type, ArrowType::Dictionary(..)))
        .then(|| column_desc.clone());
        let mut record_reader = RecordReader::<T>::new(column_desc, batch_size);
        if let Some(threshold) = padding_threshold {
            record_reader.set_padding_threshold(threshold);
        }

        Ok(Self {
            data_type,
            pages,
            def_levels_buffer: None,
            rep_levels_buffer: None,
            record_reader,
            selection_descr,
            selected: None,
            full_started: false,
        })
    }

    #[inline]
    fn native_array(record_data: Buffer, len: usize, nulls: Option<NullBuffer>) -> ArrayRef {
        match T::get_physical_type() {
            PhysicalType::BOOLEAN => Arc::new(BooleanArray::new(
                BooleanBuffer::new(record_data, 0, len),
                nulls,
            )),
            PhysicalType::INT32 => Arc::new(Int32Array::new(
                ScalarBuffer::new(record_data, 0, len),
                nulls,
            )),
            PhysicalType::INT64 => Arc::new(Int64Array::new(
                ScalarBuffer::new(record_data, 0, len),
                nulls,
            )),
            PhysicalType::FLOAT => Arc::new(Float32Array::new(
                ScalarBuffer::new(record_data, 0, len),
                nulls,
            )),
            PhysicalType::DOUBLE => Arc::new(Float64Array::new(
                ScalarBuffer::new(record_data, 0, len),
                nulls,
            )),
            PhysicalType::INT96 => Arc::new(Int64Array::new(
                ScalarBuffer::new(record_data, 0, len),
                nulls,
            )),
            PhysicalType::BYTE_ARRAY | PhysicalType::FIXED_LEN_BYTE_ARRAY => {
                unreachable!("PrimitiveArrayReaders don't support complex physical types");
            }
        }
    }

    fn resume_full(&mut self) -> Result<()> {
        if let Some(selected) = self.selected.take()
            && let Some((pages, offset)) = selected.into_replay()
        {
            self.record_reader.set_page_reader(pages)?;
            if offset != 0 && self.record_reader.skip_records(offset)? != offset {
                return Err(eof_err!("Incomplete data page when resuming full decode"));
            }
        }
        self.full_started = true;
        Ok(())
    }
}

/// Implementation of primitive array reader.
impl<T> ArrayReader for PrimitiveArrayReader<T>
where
    T: DataType,
    T::T: Copy + Default,
    Vec<T::T>: IntoBuffer,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Returns data type of primitive array.
    fn get_data_type(&self) -> &ArrowType {
        &self.data_type
    }

    fn read_records(&mut self, batch_size: usize) -> Result<usize> {
        if batch_size == 0 {
            return Ok(0);
        }
        if !self.full_started {
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
        if self.selection_descr.is_none() {
            self.read_records(rows)?;
            return filter_selection(self.consume_batch()?, words, rows);
        }

        // A preceding ordinary read may own the current chunk. Its decoder
        // finishes that chunk, then selected reads can start with the next.
        let mut start = 0;
        let mut prefix = None;
        if self.full_started {
            start = self.record_reader.read_records(rows)?;
            let array = self.consume_batch()?;
            if start == rows {
                return filter_selection(array, words, rows);
            }
            if start != 0 {
                prefix = Some(filter_selection(array, words, start)?);
            }
            self.full_started = false;
        }

        let selected = self.selected.get_or_insert_with(|| {
            Box::new(SelectedPrimitive::new(
                self.selection_descr.as_ref().unwrap().clone(),
            ))
        });
        let (values, nulls) = selected.read(self.pages.as_mut(), words, start, rows)?;
        let len = values.len();
        let values = values.into_buffer(&self.data_type);
        let array = coerce_array(Self::native_array(values, len, nulls), &self.data_type)?;
        self.def_levels_buffer = None;
        self.rep_levels_buffer = None;
        if let Some(prefix) = prefix {
            return Ok(arrow_select::concat::concat(&[
                prefix.as_ref(),
                array.as_ref(),
            ])?);
        }
        Ok(array)
    }

    fn consume_batch(&mut self) -> Result<ArrayRef> {
        let target_type = &self.data_type;

        // Convert physical data to equivalent arrow type, and then perform
        // coercion as needed
        let len = self.record_reader.values_written();

        let record_data = self
            .record_reader
            .consume_record_data()
            .into_buffer(target_type);

        let nulls = self
            .record_reader
            .consume_compact_bitmap()
            .and_then(|b| NullBuffer::from_unsliced_buffer(b, len));

        let array = coerce_array(Self::native_array(record_data, len, nulls), target_type)?;

        // save definition and repetition buffers
        self.def_levels_buffer = self.record_reader.consume_def_levels();
        self.rep_levels_buffer = self.record_reader.consume_rep_levels();
        self.record_reader.reset();
        Ok(array)
    }

    fn skip_records(&mut self, num_records: usize) -> Result<usize> {
        if num_records == 0 {
            return Ok(0);
        }
        if !self.full_started {
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

/// Coerce the parquet physical type array to the target type
///
/// This should match the logic in schema::primitive::apply_hint
fn coerce_array(array: ArrayRef, target_type: &ArrowType) -> Result<ArrayRef> {
    if let ArrowType::Dictionary(key_type, value_type) = target_type {
        let dictionary = pack_dictionary(key_type, array.as_ref())?;
        let any_dictionary = dictionary.as_any_dictionary();

        let coerced_values =
            coerce_array(Arc::clone(any_dictionary.values()), value_type.as_ref())?;

        return Ok(any_dictionary.with_values(coerced_values));
    }

    match array.data_type() {
        ArrowType::Int32 => coerce_i32(array.as_primitive(), target_type),
        ArrowType::Int64 => coerce_i64(array.as_primitive(), target_type),
        ArrowType::Boolean | ArrowType::Float32 | ArrowType::Float64 => Ok(array),
        _ => unreachable!("Cannot coerce array of type {}", array.data_type()),
    }
}

fn coerce_i32(array: &Int32Array, target_type: &ArrowType) -> Result<ArrayRef> {
    Ok(match target_type {
        ArrowType::UInt8 => {
            let array = array.unary(|i| i as u8) as UInt8Array;
            Arc::new(array) as ArrayRef
        }
        ArrowType::Int8 => {
            let array = array.unary(|i| i as i8) as Int8Array;
            Arc::new(array) as ArrayRef
        }
        ArrowType::UInt16 => {
            let array = array.unary(|i| i as u16) as UInt16Array;
            Arc::new(array) as ArrayRef
        }
        ArrowType::Int16 => {
            let array = array.unary(|i| i as i16) as Int16Array;
            Arc::new(array) as ArrayRef
        }
        ArrowType::Int32 => Arc::new(array.clone()),
        // follow C++ implementation and use overflow/reinterpret cast from  i32 to u32 which will map
        // `i32::MIN..0` to `(i32::MAX as u32)..u32::MAX`
        ArrowType::UInt32 => Arc::new(UInt32Array::new(
            array.values().inner().clone().into(),
            array.nulls().cloned(),
        )) as ArrayRef,
        ArrowType::Date32 => Arc::new(array.reinterpret_cast::<Date32Type>()) as _,
        ArrowType::Date64 => {
            let array: Date64Array = array.unary(|x| x as i64 * 86_400_000);
            Arc::new(array) as ArrayRef
        }
        ArrowType::Time32(TimeUnit::Second) => {
            Arc::new(array.reinterpret_cast::<Time32SecondType>()) as ArrayRef
        }
        ArrowType::Time32(TimeUnit::Millisecond) => {
            Arc::new(array.reinterpret_cast::<Time32MillisecondType>()) as ArrayRef
        }
        ArrowType::Timestamp(time_unit, timezone) => match time_unit {
            TimeUnit::Second => {
                let array: TimestampSecondArray = array
                    .unary(|x| x as i64)
                    .with_timezone_opt(timezone.clone());
                Arc::new(array) as _
            }
            TimeUnit::Millisecond => {
                let array: TimestampMillisecondArray = array
                    .unary(|x| x as i64)
                    .with_timezone_opt(timezone.clone());
                Arc::new(array) as _
            }
            TimeUnit::Microsecond => {
                let array: TimestampMicrosecondArray = array
                    .unary(|x| x as i64)
                    .with_timezone_opt(timezone.clone());
                Arc::new(array) as _
            }
            TimeUnit::Nanosecond => {
                let array: TimestampNanosecondArray = array
                    .unary(|x| x as i64)
                    .with_timezone_opt(timezone.clone());
                Arc::new(array) as _
            }
        },
        ArrowType::Decimal32(p, s) => {
            let array = array
                .reinterpret_cast::<Decimal32Type>()
                .with_precision_and_scale(*p, *s)?;
            Arc::new(array) as ArrayRef
        }
        ArrowType::Decimal64(p, s) => {
            let array: Decimal64Array =
                array.unary(|i| i as i64).with_precision_and_scale(*p, *s)?;
            Arc::new(array) as ArrayRef
        }
        ArrowType::Decimal128(p, s) => {
            let array: Decimal128Array = array
                .unary(|i| i as i128)
                .with_precision_and_scale(*p, *s)?;
            Arc::new(array) as ArrayRef
        }
        ArrowType::Decimal256(p, s) => {
            let array: Decimal256Array = array
                .unary(|i| i256::from_i128(i as i128))
                .with_precision_and_scale(*p, *s)?;
            Arc::new(array) as ArrayRef
        }
        _ => unreachable!("Cannot coerce i32 to {target_type}"),
    })
}

fn coerce_i64(array: &Int64Array, target_type: &ArrowType) -> Result<ArrayRef> {
    Ok(match target_type {
        ArrowType::Int64 => Arc::new(array.clone()) as _,
        // follow C++ implementation and use overflow/reinterpret cast from i64 to u64 which will map
        // `i64::MIN..0` to `(i64::MAX as u64)..u64::MAX`
        ArrowType::UInt64 => Arc::new(UInt64Array::new(
            array.values().inner().clone().into(),
            array.nulls().cloned(),
        )) as ArrayRef,
        ArrowType::Date64 => Arc::new(array.reinterpret_cast::<Date64Type>()) as _,
        ArrowType::Time64(TimeUnit::Microsecond) => {
            Arc::new(array.reinterpret_cast::<Time64MicrosecondType>()) as _
        }
        ArrowType::Time64(TimeUnit::Nanosecond) => {
            Arc::new(array.reinterpret_cast::<Time64NanosecondType>()) as _
        }
        ArrowType::Duration(unit) => match unit {
            TimeUnit::Second => Arc::new(array.reinterpret_cast::<DurationSecondType>()) as _,
            TimeUnit::Millisecond => {
                Arc::new(array.reinterpret_cast::<DurationMillisecondType>()) as _
            }
            TimeUnit::Microsecond => {
                Arc::new(array.reinterpret_cast::<DurationMicrosecondType>()) as _
            }
            TimeUnit::Nanosecond => {
                Arc::new(array.reinterpret_cast::<DurationNanosecondType>()) as _
            }
        },
        ArrowType::Timestamp(time_unit, timezone) => match time_unit {
            TimeUnit::Second => {
                let array = array
                    .reinterpret_cast::<TimestampSecondType>()
                    .with_timezone_opt(timezone.clone());
                Arc::new(array) as _
            }
            TimeUnit::Millisecond => {
                let array = array
                    .reinterpret_cast::<TimestampMillisecondType>()
                    .with_timezone_opt(timezone.clone());
                Arc::new(array) as _
            }
            TimeUnit::Microsecond => {
                let array = array
                    .reinterpret_cast::<TimestampMicrosecondType>()
                    .with_timezone_opt(timezone.clone());
                Arc::new(array) as _
            }
            TimeUnit::Nanosecond => {
                let array = array
                    .reinterpret_cast::<TimestampNanosecondType>()
                    .with_timezone_opt(timezone.clone());
                Arc::new(array) as _
            }
        },
        ArrowType::Decimal64(p, s) => {
            let array = array
                .reinterpret_cast::<Decimal64Type>()
                .with_precision_and_scale(*p, *s)?;
            Arc::new(array) as _
        }
        ArrowType::Decimal128(p, s) => {
            let array: Decimal128Array = array
                .unary(|i| i as i128)
                .with_precision_and_scale(*p, *s)?;
            Arc::new(array) as _
        }
        ArrowType::Decimal256(p, s) => {
            let array: Decimal256Array = array
                .unary(|i| i256::from_i128(i as i128))
                .with_precision_and_scale(*p, *s)?;
            Arc::new(array) as _
        }
        _ => unreachable!("Cannot coerce i64 to {target_type}"),
    })
}

macro_rules! pack_dictionary_helper {
    ($t:ty, $values:ident) => {
        match $values.data_type() {
            ArrowType::Int32 => pack_dictionary_impl::<$t, Int32Type>($values.as_primitive()),
            ArrowType::Int64 => pack_dictionary_impl::<$t, Int64Type>($values.as_primitive()),
            ArrowType::Float32 => pack_dictionary_impl::<$t, Float32Type>($values.as_primitive()),
            ArrowType::Float64 => pack_dictionary_impl::<$t, Float64Type>($values.as_primitive()),
            _ => unreachable!("Invalid physical type"),
        }
    };
}

fn pack_dictionary(key: &ArrowType, values: &dyn Array) -> Result<ArrayRef> {
    downcast_integer! {
        key => (pack_dictionary_helper, values),
        _ => unreachable!("Invalid key type"),
    }
}

fn pack_dictionary_impl<K: ArrowDictionaryKeyType, V: ArrowPrimitiveType>(
    values: &PrimitiveArray<V>,
) -> Result<ArrayRef> {
    let mut builder = PrimitiveDictionaryBuilder::<K, V>::with_capacity(1024, values.len());
    builder.extend(values);
    Ok(Arc::new(builder.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross all dictionary index widths with actual V1/V2 page boundaries,
    /// row groups, partial output batches, and five definition-level patterns.
    #[test]
    fn test_fused_dictionary_widths_pages_and_nulls() {
        crate::util::native::dispatch::for_each_tier(|| {
            use crate::encodings::rle::{RleDecoder, RleEncoder};
            use bytes::Bytes;
            use rand::{RngExt, SeedableRng};
            let mut rng = rand::rngs::StdRng::seed_from_u64(0x5232_2026);
            for width in 1u8..=32 {
                let dictionary: Vec<i64> = (0..if (18..=24).contains(&width) && !cfg!(miri) {
                    1usize << width
                } else {
                    (1usize << width.min(7)).min(67)
                })
                    .map(|i| (i as i64).wrapping_mul(i64::MIN + 123))
                    .collect();
                // Built once per width: dictionaries reach 2^24 entries.
                let dictionary_page = Bytes::from(
                    dictionary
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                );
                for v2 in [false, true] {
                    // Patterns 5 and 6 are run-heavy, as surrogate keys
                    // written in order of a parent key: repeated IDs in runs
                    // of 1 to 40, and (5) NULLs in runs as well.
                    for pattern in 0..7 {
                        let nullable = pattern != 0 && pattern != 6;
                        let (mut run_left, mut run_id, mut null_left) = (0usize, 0usize, 0usize);
                        let schema = parse_message_type(if nullable {
                            "message test { OPTIONAL INT64 value; }"
                        } else {
                            "message test { REQUIRED INT64 value; }"
                        })
                        .unwrap();
                        let desc = SchemaDescriptor::new(Arc::new(schema)).column(0);
                        let mut groups = Vec::new();
                        let mut expected = Vec::new();
                        for _ in 0..2 {
                            let mut pages = vec![Page::DictionaryPage {
                                buf: dictionary_page.clone(),
                                num_values: dictionary.len() as u32,
                                encoding: Encoding::PLAIN,
                                is_sorted: false,
                            }];
                            let page_rows: &[usize] = if pattern >= 5 {
                                &[17, 513, 4099]
                            } else {
                                &[17, 129, 513]
                            };
                            for &rows in page_rows {
                                let mut levels = RleEncoder::new(1, 8192);
                                let mut ids = RleEncoder::new(width, 65536);
                                let mut validity = Vec::new();
                                let mut num_valid = 0;
                                for i in 0..rows {
                                    let valid = match pattern {
                                        0 | 1 | 6 => true,
                                        2 => rng.random_bool(0.65),
                                        3 => false,
                                        5 => {
                                            if null_left == 0 && rng.random_ratio(1, 20) {
                                                null_left = rng.random_range(1..12);
                                            }
                                            null_left = null_left.saturating_sub(1);
                                            null_left == 0
                                        }
                                        _ => i % 2 == 0,
                                    };
                                    validity.push(valid);
                                    levels.put(valid as u64);
                                    if valid {
                                        let id = if i == 0 {
                                            dictionary.len() - 1
                                        } else if pattern >= 5 {
                                            if run_left == 0 {
                                                run_left = rng.random_range(1..=40);
                                                run_id = rng.random_range(0..dictionary.len());
                                            }
                                            run_left -= 1;
                                            run_id
                                        } else if i % 97 < 32 {
                                            0
                                        } else {
                                            rng.random_range(0..dictionary.len())
                                        };
                                        ids.put(id as u64);
                                        num_valid += 1;
                                    }
                                }
                                let encoded_ids = Bytes::from(ids.consume());
                                let mut scalar = RleDecoder::new(width);
                                scalar.set_data(encoded_ids.clone()).unwrap();
                                let mut reference_ids = vec![0i32; num_valid];
                                assert_eq!(
                                    scalar.get_batch(&mut reference_ids).unwrap(),
                                    num_valid
                                );
                                let mut ids = reference_ids.into_iter();
                                expected.extend(validity.iter().map(|&valid| {
                                    valid.then(|| dictionary[ids.next().unwrap() as u32 as usize])
                                }));
                                assert!(ids.next().is_none());
                                let levels = if nullable {
                                    levels.consume()
                                } else {
                                    Vec::new()
                                };
                                let mut data = Vec::new();
                                if nullable && !v2 {
                                    data.extend_from_slice(&(levels.len() as u32).to_le_bytes());
                                }
                                data.extend_from_slice(&levels);
                                data.push(width);
                                data.extend_from_slice(&encoded_ids);
                                pages.push(if v2 {
                                    Page::DataPageV2 {
                                        buf: Bytes::from(data),
                                        num_values: rows as u32,
                                        encoding: Encoding::RLE_DICTIONARY,
                                        num_nulls: (rows - num_valid) as u32,
                                        num_rows: rows as u32,
                                        def_levels_byte_len: levels.len() as u32,
                                        rep_levels_byte_len: 0,
                                        is_compressed: false,
                                        statistics: None,
                                    }
                                } else {
                                    Page::DataPage {
                                        buf: Bytes::from(data),
                                        num_values: rows as u32,
                                        encoding: Encoding::RLE_DICTIONARY,
                                        def_level_encoding: Encoding::RLE,
                                        rep_level_encoding: Encoding::RLE,
                                        statistics: None,
                                    }
                                });
                            }
                            groups.push(pages);
                        }
                        let expected = Int64Array::from(expected);
                        for batch in [1, 7, 31, 1023, 1024, 1025, 8192] {
                            let mut reader = PrimitiveArrayReader::<Int64Type>::new(
                                Box::new(InMemoryPageIterator::new(groups.clone())),
                                desc.clone(),
                                None,
                                batch,
                                None,
                            )
                            .unwrap();
                            let mut offset = 0;
                            loop {
                                let actual = reader.next_batch(batch).unwrap();
                                if actual.is_empty() {
                                    break;
                                }
                                let actual = actual.as_any().downcast_ref::<Int64Array>().unwrap();
                                assert_eq!(
                                    actual,
                                    &expected.slice(offset, actual.len()),
                                    "width={width} v2={v2} pattern={pattern} batch={batch}"
                                );
                                offset += actual.len();
                            }
                            assert_eq!(offset, expected.len());
                        }
                    }
                }
            }
        });
    }

    /// Exercise every integer value encoding through actual page and row-group
    /// boundaries. The input arrays are the oracle, independent of the reader.
    #[test]
    fn test_random_primitive_encodings_pages_and_nulls() {
        use crate::arrow::{ArrowWriter, arrow_reader::ParquetRecordBatchReaderBuilder};
        use crate::file::properties::{WriterProperties, WriterVersion};
        use arrow_array::RecordBatch;
        use bytes::Bytes;
        use rand::{RngExt, SeedableRng};

        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5a31_2026);
        for version in [WriterVersion::PARQUET_1_0, WriterVersion::PARQUET_2_0] {
            for encoding in [
                Encoding::PLAIN,
                Encoding::RLE_DICTIONARY,
                Encoding::DELTA_BINARY_PACKED,
                Encoding::BYTE_STREAM_SPLIT,
            ] {
                for pattern in 0..5 {
                    let values: Vec<Option<i64>> = (0..2053)
                        .map(|i| {
                            let valid = match pattern {
                                0 | 1 => true,
                                2 => rng.random_bool(0.7),
                                3 => false,
                                _ => i % 2 == 0,
                            };
                            valid.then(|| rng.random_range(-131072..131072))
                        })
                        .collect();
                    let nullable = pattern != 0;
                    let schema = Arc::new(arrow_schema::Schema::new(vec![
                        arrow_schema::Field::new("i64", ArrowType::Int64, nullable),
                        arrow_schema::Field::new("i32", ArrowType::Int32, nullable),
                    ]));
                    let expected = RecordBatch::try_new(
                        schema.clone(),
                        vec![
                            Arc::new(Int64Array::from(values.clone())),
                            Arc::new(Int32Array::from(
                                values
                                    .iter()
                                    .map(|v| v.map(|x| x as i32))
                                    .collect::<Vec<_>>(),
                            )),
                        ],
                    )
                    .unwrap();
                    let props = WriterProperties::builder()
                        .set_writer_version(version)
                        .set_dictionary_enabled(encoding == Encoding::RLE_DICTIONARY)
                        .set_encoding(if encoding == Encoding::RLE_DICTIONARY {
                            Encoding::PLAIN
                        } else {
                            encoding
                        })
                        .set_max_row_group_row_count(Some(513))
                        .set_data_page_row_count_limit(67)
                        .set_write_batch_size(37)
                        .build();
                    let mut bytes = Vec::new();
                    let mut writer = ArrowWriter::try_new(&mut bytes, schema, Some(props)).unwrap();
                    writer.write(&expected).unwrap();
                    writer.close().unwrap();
                    let bytes = Bytes::from(bytes);
                    for batch_size in [1, 7, 31, 1023, 1024, 1025, 8192] {
                        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())
                            .unwrap()
                            .with_batch_size(batch_size)
                            .build()
                            .unwrap();
                        let mut offset = 0;
                        for actual in reader {
                            let actual = actual.unwrap();
                            assert_eq!(
                                actual,
                                expected.slice(offset, actual.num_rows()),
                                "{version:?} {encoding:?} pattern={pattern} batch={batch_size}"
                            );
                            offset += actual.num_rows();
                        }
                        assert_eq!(offset, expected.num_rows());
                    }
                }
            }
        }
    }
    #[test]
    fn selected_vectors_match_full_with_plain_fallback_and_row_group_slices() {
        use crate::arrow::ArrowWriter;
        use crate::column::page::PageReader;
        use crate::file::{
            properties::{WriterProperties, WriterVersion},
            reader::{FilePageIterator, FileReader, SerializedFileReader},
        };
        use arrow_array::{BooleanArray, RecordBatch};
        use bytes::Bytes;
        use rand::{RngExt, SeedableRng};

        let mut rng = rand::rngs::SmallRng::seed_from_u64(0x128);
        for version in [WriterVersion::PARQUET_1_0, WriterVersion::PARQUET_2_0] {
            for encoding in [Encoding::RLE_DICTIONARY, Encoding::DELTA_BINARY_PACKED] {
                for pattern in 0..5 {
                    let nullable = pattern != 0;
                    let values = (0..2111)
                        .map(|i| {
                            let valid = match pattern {
                                0 | 1 => true,
                                2 => false,
                                3 => rng.random_bool(0.7),
                                _ => i % 2 == 0,
                            };
                            valid.then_some(if i % 701 < 74 { 42 } else { i as i64 })
                        })
                        .collect::<Vec<_>>();
                    let schema =
                        Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
                            "value",
                            ArrowType::Int64,
                            nullable,
                        )]));
                    let batch = RecordBatch::try_new(
                        Arc::clone(&schema),
                        vec![Arc::new(Int64Array::from(values))],
                    )
                    .unwrap();
                    let props = WriterProperties::builder()
                        .set_writer_version(version)
                        .set_dictionary_enabled(encoding == Encoding::RLE_DICTIONARY)
                        .set_dictionary_page_size_limit(192)
                        .set_encoding(if encoding == Encoding::RLE_DICTIONARY {
                            Encoding::PLAIN
                        } else {
                            encoding
                        })
                        .set_max_row_group_row_count(Some(701))
                        .set_data_page_row_count_limit(71)
                        .set_write_batch_size(37)
                        .build();
                    let mut bytes = Vec::new();
                    let mut writer = ArrowWriter::try_new(&mut bytes, schema, Some(props)).unwrap();
                    writer.write(&batch).unwrap();
                    writer.close().unwrap();
                    let file = Arc::new(SerializedFileReader::new(Bytes::from(bytes)).unwrap());
                    if encoding == Encoding::RLE_DICTIONARY && pattern != 2 {
                        let mut pages: Box<dyn PageReader> = file
                            .get_row_group(0)
                            .unwrap()
                            .get_column_page_reader(0)
                            .unwrap();
                        let mut encodings = Vec::new();
                        while let Some(page) = pages.get_next_page().unwrap() {
                            if page.is_data_page() {
                                encodings.push(page.encoding());
                            }
                        }
                        assert!(encodings.contains(&Encoding::RLE_DICTIONARY));
                        assert!(encodings.contains(&Encoding::PLAIN));
                    }
                    let desc = file.metadata().file_metadata().schema_descr().column(0);
                    let build = || {
                        let pages = Box::new(FilePageIterator::new(0, file.clone()).unwrap());
                        PrimitiveArrayReader::<Int64Type>::new(pages, desc.clone(), None, 79, None)
                            .unwrap()
                    };

                    let mut full = build();
                    let mut selected = build();
                    assert!(selected.read_selected(&[], 65).is_err());
                    let start = if pattern % 2 == 0 { 0 } else { 39 };
                    assert_eq!(full.skip_records(start).unwrap(), start);
                    assert_eq!(selected.skip_records(start).unwrap(), start);
                    let mut offset = start;
                    for vector in 0.. {
                        if offset == batch.num_rows() {
                            break;
                        }
                        let rows = [0usize, 1, 37, 79, 139, 507, 143][vector % 7]
                            .min(batch.num_rows() - offset);
                        let density = [0.0, 0.001, 0.02, 0.1, 0.5, 1.0][vector % 6];
                        let mut words = vec![0; rows.div_ceil(64)];
                        let mask = BooleanArray::from_iter((0..rows).map(|row| {
                            let select = rng.random_bool(density);
                            if select {
                                words[row / 64] |= 1 << (row % 64);
                            }
                            Some(select)
                        }));
                        let reference = full.next_batch(rows).unwrap();
                        let actual = if vector >= 11 && vector % 5 == 4 {
                            // Full and selected reads share the same buffered page.
                            selected.next_batch(rows).unwrap()
                        } else {
                            selected.read_selected(&words, rows).unwrap()
                        };
                        let expected = if vector >= 11 && vector % 5 == 4 {
                            reference
                        } else {
                            arrow_select::filter::filter(reference.as_ref(), &mask).unwrap()
                        };
                        assert_eq!(
                            actual.as_ref(),
                            expected.as_ref(),
                            "{version:?} {encoding:?} pattern={pattern} offset={offset}"
                        );
                        offset += rows;
                    }
                }
            }
        }
    }

    use crate::arrow::array_reader::test_util::EmptyPageIterator;
    use crate::arrow::arrow_reader::DEFAULT_BATCH_SIZE;
    use crate::basic::Encoding;
    use crate::column::page::Page;
    use crate::data_type::{Int32Type, Int64Type};
    use crate::schema::parser::parse_message_type;
    use crate::schema::types::SchemaDescriptor;
    use crate::util::InMemoryPageIterator;
    use crate::util::test_common::rand_gen::make_pages;
    use arrow::datatypes::ArrowPrimitiveType;
    use arrow_array::{Array, Date32Array, PrimitiveArray};

    use arrow::datatypes::DataType::{Date32, Decimal128};
    use rand::distr::uniform::SampleUniform;
    use std::collections::VecDeque;

    #[expect(clippy::too_many_arguments)]
    fn make_column_chunks<T: DataType>(
        column_desc: ColumnDescPtr,
        encoding: Encoding,
        num_levels: usize,
        min_value: T::T,
        max_value: T::T,
        def_levels: &mut Vec<i16>,
        rep_levels: &mut Vec<i16>,
        values: &mut Vec<T::T>,
        page_lists: &mut Vec<Vec<Page>>,
        use_v2: bool,
        num_chunks: usize,
    ) where
        T::T: PartialOrd + SampleUniform + Copy,
    {
        for _i in 0..num_chunks {
            let mut pages = VecDeque::new();
            let mut data = Vec::new();
            let mut page_def_levels = Vec::new();
            let mut page_rep_levels = Vec::new();

            make_pages::<T>(
                column_desc.clone(),
                encoding,
                1,
                num_levels,
                min_value,
                max_value,
                &mut page_def_levels,
                &mut page_rep_levels,
                &mut data,
                &mut pages,
                use_v2,
            );

            def_levels.append(&mut page_def_levels);
            rep_levels.append(&mut page_rep_levels);
            values.append(&mut data);
            page_lists.push(Vec::from(pages));
        }
    }

    #[test]
    fn test_primitive_array_reader_empty_pages() {
        // Construct column schema
        let message_type = "
        message test_schema {
          REQUIRED INT32 leaf;
        }
        ";

        let schema = parse_message_type(message_type)
            .map(|t| Arc::new(SchemaDescriptor::new(Arc::new(t))))
            .unwrap();

        let mut array_reader = PrimitiveArrayReader::<Int32Type>::new(
            Box::<EmptyPageIterator>::default(),
            schema.column(0),
            None,
            DEFAULT_BATCH_SIZE,
            None,
        )
        .unwrap();

        // expect no values to be read
        let array = array_reader.next_batch(50).unwrap();
        assert!(array.is_empty());
    }

    #[test]
    fn test_primitive_array_reader_data() {
        // Construct column schema
        let message_type = "
        message test_schema {
          REQUIRED INT32 leaf;
        }
        ";

        let schema = parse_message_type(message_type)
            .map(|t| Arc::new(SchemaDescriptor::new(Arc::new(t))))
            .unwrap();

        let column_desc = schema.column(0);

        // Construct page iterator
        {
            let mut data = Vec::new();
            let mut page_lists = Vec::new();
            make_column_chunks::<Int32Type>(
                column_desc.clone(),
                Encoding::PLAIN,
                100,
                1,
                200,
                &mut Vec::new(),
                &mut Vec::new(),
                &mut data,
                &mut page_lists,
                true,
                2,
            );
            let page_iterator = InMemoryPageIterator::new(page_lists);

            let mut array_reader = PrimitiveArrayReader::<Int32Type>::new(
                Box::new(page_iterator),
                column_desc,
                None,
                DEFAULT_BATCH_SIZE,
                None,
            )
            .unwrap();

            // Read first 50 values, which are all from the first column chunk
            let array = array_reader.next_batch(50).unwrap();
            let array = array.as_any().downcast_ref::<Int32Array>().unwrap();

            assert_eq!(&Int32Array::from(data[0..50].to_vec()), array);

            // Read next 100 values, the first 50 ones are from the first column chunk,
            // and the last 50 ones are from the second column chunk
            let array = array_reader.next_batch(100).unwrap();
            let array = array.as_any().downcast_ref::<Int32Array>().unwrap();

            assert_eq!(&Int32Array::from(data[50..150].to_vec()), array);

            // Try to read 100 values, however there are only 50 values
            let array = array_reader.next_batch(100).unwrap();
            let array = array.as_any().downcast_ref::<Int32Array>().unwrap();

            assert_eq!(&Int32Array::from(data[150..200].to_vec()), array);
        }
    }

    macro_rules! test_primitive_array_reader_one_type {
        (
            $arrow_parquet_type:ty,
            $physical_type:expr,
            $converted_type_str:expr,
            $result_arrow_type:ty,
            $result_arrow_cast_type:ty,
            $result_primitive_type:ty
            $(, $timezone:expr)?
        ) => {{
            let message_type = format!(
                "
            message test_schema {{
              REQUIRED {:?} leaf ({});
          }}
            ",
                $physical_type, $converted_type_str
            );
            let schema = parse_message_type(&message_type)
                .map(|t| Arc::new(SchemaDescriptor::new(Arc::new(t))))
                .unwrap();

            let column_desc = schema.column(0);

            // Construct page iterator
            {
                let mut data = Vec::new();
                let mut page_lists = Vec::new();
                make_column_chunks::<$arrow_parquet_type>(
                    column_desc.clone(),
                    Encoding::PLAIN,
                    100,
                    1,
                    200,
                    &mut Vec::new(),
                    &mut Vec::new(),
                    &mut data,
                    &mut page_lists,
                    true,
                    2,
                );
                let page_iterator = InMemoryPageIterator::new(page_lists);
                let mut array_reader = PrimitiveArrayReader::<$arrow_parquet_type>::new(
                    Box::new(page_iterator),
                    column_desc.clone(),
                    None,
                    DEFAULT_BATCH_SIZE,
                    None,
                )
                .expect("Unable to get array reader");

                let array = array_reader
                    .next_batch(50)
                    .expect("Unable to get batch from reader");

                let result_data_type = <$result_arrow_type>::DATA_TYPE;
                let array = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<$result_arrow_type>>()
                    .expect(
                        format!(
                            "Unable to downcast {:?} to {:?}",
                            array.data_type(),
                            result_data_type
                        )
                        .as_str(),
                    )
                    $(.clone().with_timezone($timezone))?
                    ;

                // create expected array as primitive, and cast to result type
                #[expect(
                    trivial_numeric_casts,
                    reason = "the cast is a no-op when the source and result types are the same"
                )]
                let expected = PrimitiveArray::<$result_arrow_cast_type>::from(
                    data[0..50]
                        .iter()
                        .map(|x| *x as $result_primitive_type)
                        .collect::<Vec<$result_primitive_type>>(),
                );
                let expected = Arc::new(expected) as ArrayRef;
                let expected = arrow::compute::cast(&expected, &result_data_type)
                    .expect("Unable to cast expected array");
                assert_eq!(expected.data_type(), &result_data_type);
                let expected = expected
                    .as_any()
                    .downcast_ref::<PrimitiveArray<$result_arrow_type>>()
                    .expect(
                        format!(
                            "Unable to downcast expected {:?} to {:?}",
                            expected.data_type(),
                            result_data_type
                        )
                        .as_str(),
                    )
                    $(.clone().with_timezone($timezone))?
                    ;
                assert_eq!(expected, array);
            }
        }};
    }

    #[test]
    fn test_primitive_array_reader_temporal_types() {
        test_primitive_array_reader_one_type!(
            crate::data_type::Int32Type,
            PhysicalType::INT32,
            "DATE",
            arrow::datatypes::Date32Type,
            arrow::datatypes::Int32Type,
            i32
        );
        test_primitive_array_reader_one_type!(
            crate::data_type::Int32Type,
            PhysicalType::INT32,
            "TIME_MILLIS",
            arrow::datatypes::Time32MillisecondType,
            arrow::datatypes::Int32Type,
            i32
        );
        test_primitive_array_reader_one_type!(
            crate::data_type::Int64Type,
            PhysicalType::INT64,
            "TIME_MICROS",
            arrow::datatypes::Time64MicrosecondType,
            arrow::datatypes::Int64Type,
            i64
        );
        test_primitive_array_reader_one_type!(
            crate::data_type::Int64Type,
            PhysicalType::INT64,
            "TIMESTAMP_MILLIS",
            arrow::datatypes::TimestampMillisecondType,
            arrow::datatypes::Int64Type,
            i64,
            "UTC"
        );
        test_primitive_array_reader_one_type!(
            crate::data_type::Int64Type,
            PhysicalType::INT64,
            "TIMESTAMP_MICROS",
            arrow::datatypes::TimestampMicrosecondType,
            arrow::datatypes::Int64Type,
            i64,
            "UTC"
        );
    }

    #[test]
    fn test_primitive_array_reader_def_and_rep_levels() {
        // Construct column schema
        let message_type = "
        message test_schema {
            REPEATED Group test_mid {
                OPTIONAL INT32 leaf;
            }
        }
        ";

        let schema = parse_message_type(message_type)
            .map(|t| Arc::new(SchemaDescriptor::new(Arc::new(t))))
            .unwrap();

        let column_desc = schema.column(0);

        // Construct page iterator
        {
            let mut def_levels = Vec::new();
            let mut rep_levels = Vec::new();
            let mut page_lists = Vec::new();
            make_column_chunks::<Int32Type>(
                column_desc.clone(),
                Encoding::PLAIN,
                100,
                1,
                200,
                &mut def_levels,
                &mut rep_levels,
                &mut Vec::new(),
                &mut page_lists,
                true,
                2,
            );

            let page_iterator = InMemoryPageIterator::new(page_lists);

            let mut array_reader = PrimitiveArrayReader::<Int32Type>::new(
                Box::new(page_iterator),
                column_desc,
                None,
                DEFAULT_BATCH_SIZE,
                None,
            )
            .unwrap();

            let mut accu_len: usize = 0;

            // Read first 50 values, which are all from the first column chunk
            let array = array_reader.next_batch(50).unwrap();
            assert_eq!(
                Some(&def_levels[accu_len..(accu_len + array.len())]),
                array_reader.get_def_levels()
            );
            assert_eq!(
                Some(&rep_levels[accu_len..(accu_len + array.len())]),
                array_reader.get_rep_levels()
            );
            accu_len += array.len();

            // Read next 100 values, the first 50 ones are from the first column chunk,
            // and the last 50 ones are from the second column chunk
            let array = array_reader.next_batch(100).unwrap();
            assert_eq!(
                Some(&def_levels[accu_len..(accu_len + array.len())]),
                array_reader.get_def_levels()
            );
            assert_eq!(
                Some(&rep_levels[accu_len..(accu_len + array.len())]),
                array_reader.get_rep_levels()
            );
            accu_len += array.len();

            // Try to read 100 values, however there are only 50 values
            let array = array_reader.next_batch(100).unwrap();
            assert_eq!(
                Some(&def_levels[accu_len..(accu_len + array.len())]),
                array_reader.get_def_levels()
            );
            assert_eq!(
                Some(&rep_levels[accu_len..(accu_len + array.len())]),
                array_reader.get_rep_levels()
            );
        }
    }

    #[test]
    fn test_primitive_array_reader_decimal_types() {
        // parquet `INT32` to decimal
        let message_type = "
            message test_schema {
                REQUIRED INT32 decimal1 (DECIMAL(8,2));
        }
        ";
        let schema = parse_message_type(message_type)
            .map(|t| Arc::new(SchemaDescriptor::new(Arc::new(t))))
            .unwrap();
        let column_desc = schema.column(0);

        // create the array reader
        {
            let mut data = Vec::new();
            let mut page_lists = Vec::new();
            make_column_chunks::<Int32Type>(
                column_desc.clone(),
                Encoding::PLAIN,
                100,
                -99999999,
                99999999,
                &mut Vec::new(),
                &mut Vec::new(),
                &mut data,
                &mut page_lists,
                true,
                2,
            );
            let page_iterator = InMemoryPageIterator::new(page_lists);

            let mut array_reader = PrimitiveArrayReader::<Int32Type>::new(
                Box::new(page_iterator),
                column_desc,
                None,
                DEFAULT_BATCH_SIZE,
                None,
            )
            .unwrap();

            // read data from the reader
            // the data type is decimal(8,2)
            let array = array_reader.next_batch(50).unwrap();
            assert_eq!(array.data_type(), &Decimal128(8, 2));
            let array = array.as_any().downcast_ref::<Decimal128Array>().unwrap();
            let data_decimal_array = data[0..50]
                .iter()
                .copied()
                .map(|v| Some(v as i128))
                .collect::<Decimal128Array>()
                .with_precision_and_scale(8, 2)
                .unwrap();
            assert_eq!(array, &data_decimal_array);

            // not equal with different data type(precision and scale)
            let data_decimal_array = data[0..50]
                .iter()
                .copied()
                .map(|v| Some(v as i128))
                .collect::<Decimal128Array>()
                .with_precision_and_scale(9, 0)
                .unwrap();
            assert_ne!(array, &data_decimal_array)
        }

        // parquet `INT64` to decimal
        let message_type = "
            message test_schema {
                REQUIRED INT64 decimal1 (DECIMAL(18,4));
        }
        ";
        let schema = parse_message_type(message_type)
            .map(|t| Arc::new(SchemaDescriptor::new(Arc::new(t))))
            .unwrap();
        let column_desc = schema.column(0);

        // create the array reader
        {
            let mut data = Vec::new();
            let mut page_lists = Vec::new();
            make_column_chunks::<Int64Type>(
                column_desc.clone(),
                Encoding::PLAIN,
                100,
                -999999999999999999,
                999999999999999999,
                &mut Vec::new(),
                &mut Vec::new(),
                &mut data,
                &mut page_lists,
                true,
                2,
            );
            let page_iterator = InMemoryPageIterator::new(page_lists);

            let mut array_reader = PrimitiveArrayReader::<Int64Type>::new(
                Box::new(page_iterator),
                column_desc,
                None,
                DEFAULT_BATCH_SIZE,
                None,
            )
            .unwrap();

            // read data from the reader
            // the data type is decimal(18,4)
            let array = array_reader.next_batch(50).unwrap();
            assert_eq!(array.data_type(), &Decimal128(18, 4));
            let array = array.as_any().downcast_ref::<Decimal128Array>().unwrap();
            let data_decimal_array = data[0..50]
                .iter()
                .copied()
                .map(|v| Some(v as i128))
                .collect::<Decimal128Array>()
                .with_precision_and_scale(18, 4)
                .unwrap();
            assert_eq!(array, &data_decimal_array);

            // not equal with different data type(precision and scale)
            let data_decimal_array = data[0..50]
                .iter()
                .copied()
                .map(|v| Some(v as i128))
                .collect::<Decimal128Array>()
                .with_precision_and_scale(34, 0)
                .unwrap();
            assert_ne!(array, &data_decimal_array)
        }
    }

    #[test]
    fn test_primitive_array_reader_date32_type() {
        // parquet `INT32` to date
        let message_type = "
            message test_schema {
                REQUIRED INT32 date1 (DATE);
        }
        ";
        let schema = parse_message_type(message_type)
            .map(|t| Arc::new(SchemaDescriptor::new(Arc::new(t))))
            .unwrap();
        let column_desc = schema.column(0);

        // create the array reader
        {
            let mut data = Vec::new();
            let mut page_lists = Vec::new();
            make_column_chunks::<Int32Type>(
                column_desc.clone(),
                Encoding::PLAIN,
                100,
                -99999999,
                99999999,
                &mut Vec::new(),
                &mut Vec::new(),
                &mut data,
                &mut page_lists,
                true,
                2,
            );
            let page_iterator = InMemoryPageIterator::new(page_lists);

            let mut array_reader = PrimitiveArrayReader::<Int32Type>::new(
                Box::new(page_iterator),
                column_desc,
                None,
                DEFAULT_BATCH_SIZE,
                None,
            )
            .unwrap();

            // read data from the reader
            // the data type is date
            let array = array_reader.next_batch(50).unwrap();
            assert_eq!(array.data_type(), &Date32);
            let array = array.as_any().downcast_ref::<Date32Array>().unwrap();
            let data_date_array = data[0..50]
                .iter()
                .copied()
                .map(Some)
                .collect::<Date32Array>();
            assert_eq!(array, &data_date_array);
        }
    }
}
