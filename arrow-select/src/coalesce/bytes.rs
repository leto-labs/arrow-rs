// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file distributed with
// this work for additional information regarding copyright ownership.
// The ASF licenses this file to You under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with the
// License. You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::{InProgressArray, append_indexed_nulls};
use arrow_array::builder::GenericByteBuilder;
use arrow_array::cast::AsArray;
use arrow_array::types::ByteArrayType;
use arrow_array::{Array, ArrayRef, BooleanArray};
use arrow_buffer::{ArrowNativeType, BooleanBufferBuilder, NullBufferBuilder};
use arrow_schema::ArrowError;
use std::sync::Arc;

#[derive(Debug)]
pub(crate) struct InProgressBooleanArray {
    source: Option<ArrayRef>,
    batch_size: usize,
    values: BooleanBufferBuilder,
    nulls: NullBufferBuilder,
}

impl InProgressBooleanArray {
    pub(crate) fn new(batch_size: usize) -> Self {
        Self {
            source: None,
            batch_size,
            values: BooleanBufferBuilder::new(0),
            nulls: NullBufferBuilder::new(batch_size),
        }
    }
}

impl InProgressArray for InProgressBooleanArray {
    fn set_source(&mut self, source: Option<ArrayRef>) {
        self.source = source;
    }

    fn copy_rows(&mut self, offset: usize, len: usize) -> Result<(), ArrowError> {
        let source = self
            .source
            .as_ref()
            .expect("source set before copy")
            .as_boolean();
        self.values
            .append_buffer(&source.values().slice(offset, len));
        if let Some(nulls) = source.nulls() {
            self.nulls.append_buffer(&nulls.slice(offset, len));
        } else {
            self.nulls.append_n_non_nulls(len);
        }
        Ok(())
    }

    fn copy_rows_by_indices(&mut self, indices: &[u32]) -> Result<(), ArrowError> {
        let source = self
            .source
            .as_ref()
            .expect("source set before copy")
            .as_boolean();
        if self.values.capacity() == 0 {
            self.values.reserve(self.batch_size);
        }
        self.values.reserve(indices.len());
        for &index in indices {
            self.values.append(source.value(index as usize));
        }
        append_indexed_nulls(&mut self.nulls, source.nulls(), indices);
        Ok(())
    }

    fn finish(&mut self) -> Result<ArrayRef, ArrowError> {
        Ok(Arc::new(BooleanArray::new(
            self.values.finish(),
            self.nulls.finish(),
        )))
    }

    fn size(&self) -> usize {
        self.values.capacity().div_ceil(8)
            + self.nulls.allocated_size()
            + self
                .source
                .as_ref()
                .map_or(0, |a| a.get_array_memory_size())
    }
}

pub(crate) struct InProgressByteArray<T: ByteArrayType> {
    source: Option<ArrayRef>,
    batch_size: usize,
    builder: Option<GenericByteBuilder<T>>,
}

impl<T: ByteArrayType> InProgressByteArray<T> {
    pub(crate) fn new(batch_size: usize) -> Self {
        Self {
            source: None,
            batch_size,
            builder: None,
        }
    }
}

impl<T: ByteArrayType> std::fmt::Debug for InProgressByteArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InProgressByteArray")
            .field("batch_size", &self.batch_size)
            .field("builder", &self.builder)
            .finish_non_exhaustive()
    }
}

impl<T: ByteArrayType> InProgressArray for InProgressByteArray<T> {
    fn set_source(&mut self, source: Option<ArrayRef>) {
        self.source = source;
    }

    fn copy_rows(&mut self, offset: usize, len: usize) -> Result<(), ArrowError> {
        let source = self
            .source
            .as_ref()
            .expect("source set before copy")
            .as_bytes::<T>();
        let builder = self
            .builder
            .get_or_insert_with(|| GenericByteBuilder::with_capacity(self.batch_size, 0));
        builder.append_array(&source.slice(offset, len))
    }

    fn copy_rows_by_indices(&mut self, indices: &[u32]) -> Result<(), ArrowError> {
        let source = self
            .source
            .as_ref()
            .expect("source set before copy")
            .as_bytes::<T>();
        let builder = self
            .builder
            .get_or_insert_with(|| GenericByteBuilder::with_capacity(self.batch_size, 0));
        for &index in indices {
            let index = index as usize;
            if source.is_null(index) {
                builder.append_null();
            } else {
                let value = source.value(index);
                let bytes: &[u8] = value.as_ref();
                let end = builder.values_slice().len().saturating_add(bytes.len());
                if T::Offset::from_usize(end).is_none() {
                    return Err(ArrowError::OffsetOverflowError(end));
                }
                builder.append_value(value);
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<ArrayRef, ArrowError> {
        Ok(Arc::new(
            self.builder
                .take()
                .expect("builder set before finish")
                .finish(),
        ))
    }

    fn size(&self) -> usize {
        self.builder.as_ref().map_or(0, |builder| {
            builder.values_capacity()
                + builder.offsets_capacity() * size_of::<T::Offset>()
                + builder.validity_capacity()
        }) + self
            .source
            .as_ref()
            .map_or(0, |a| a.get_array_memory_size())
    }
}
