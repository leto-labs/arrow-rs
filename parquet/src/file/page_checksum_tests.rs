// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for the
// specific language governing permissions and limitations
// under the License.

use super::writer::SerializedFileWriter;
use crate::basic::{Compression, PageType};
use crate::data_type::Int32Type;
use crate::file::metadata::thrift::PageHeader;
use crate::file::properties::{WriterProperties, WriterPropertiesBuilder, WriterVersion};
use crate::file::reader::FileReader;
use crate::file::serialized_reader::SerializedFileReader;
use crate::parquet_thrift::{ReadThrift, ThriftSliceInputProtocol};
use crate::record::RowAccessor;
use crate::schema::parser::parse_message_type;
use bytes::Bytes;
use std::sync::Arc;

#[test]
fn pcw_serialized_writer_checksums_every_page() {
    let schema = Arc::new(parse_message_type("message s { REQUIRED INT32 v; }").unwrap());
    for version in [WriterVersion::PARQUET_1_0, WriterVersion::PARQUET_2_0] {
        for dictionary in [false, true] {
            for enabled in [false, true] {
                let props = WriterProperties::builder()
                    .set_writer_version(version)
                    .set_dictionary_enabled(dictionary)
                    .set_data_page_row_count_limit(128)
                    .set_write_batch_size(128)
                    .set_write_page_checksum(enabled)
                    .build();
                assert_eq!(props.write_page_checksum(), enabled);
                let props = WriterPropertiesBuilder::from(props).build();
                assert_eq!(props.write_page_checksum(), enabled);
                let mut writer =
                    SerializedFileWriter::new(Vec::new(), schema.clone(), Arc::new(props)).unwrap();
                let mut group = writer.next_row_group().unwrap();
                let mut column = group.next_column().unwrap().unwrap();
                let values = (0..1024).map(|i| i % 7).collect::<Vec<_>>();
                column
                    .typed::<Int32Type>()
                    .write_batch(&values, None, None)
                    .unwrap();
                column.close().unwrap();
                group.close().unwrap();
                let bytes = Bytes::from(writer.into_inner().unwrap());
                let pages = check_headers(&bytes, enabled);
                assert!(pages.contains(&match version {
                    WriterVersion::PARQUET_1_0 => PageType::DATA_PAGE,
                    WriterVersion::PARQUET_2_0 => PageType::DATA_PAGE_V2,
                }));
                assert_eq!(pages.contains(&PageType::DICTIONARY_PAGE), dictionary);
                let reader = SerializedFileReader::new(bytes).unwrap();
                let actual = reader
                    .get_row_iter(None)
                    .unwrap()
                    .map(|row| row.unwrap().get_int(0).unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(actual, values);
            }
        }
    }
    assert!(!WriterProperties::default().write_page_checksum());
}

fn check_headers(bytes: &Bytes, enabled: bool) -> Vec<PageType> {
    let reader = SerializedFileReader::new(bytes.clone()).unwrap();
    let mut pages = Vec::new();
    for group in reader.metadata().row_groups() {
        for column in group.columns() {
            let (offset, length) = column.byte_range();
            let mut remaining = &bytes[offset as usize..(offset + length) as usize];
            while !remaining.is_empty() {
                let mut protocol = ThriftSliceInputProtocol::new(remaining);
                let header = PageHeader::read_thrift(&mut protocol).unwrap();
                remaining = protocol.as_slice();
                let (payload, tail) = remaining.split_at(header.compressed_page_size as usize);
                assert_eq!(header.crc, enabled.then(|| crc32fast::hash(payload) as i32));
                pages.push(header.r#type);
                remaining = tail;
            }
        }
    }
    assert!(!pages.is_empty());
    pages
}

#[cfg(feature = "arrow")]
#[test]
fn pcw_arrow_writer_checksums_compressed_v1_and_v2() {
    use crate::arrow::{ArrowWriter, arrow_reader::ParquetRecordBatchReaderBuilder};
    use crate::schema::types::ColumnPath;
    use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringArray};

    let batch = RecordBatch::try_from_iter([
        (
            "v",
            Arc::new(Int32Array::from_iter(
                (0..4096).map(|i| (i % 3 != 0).then_some(i % 7)),
            )) as ArrayRef,
        ),
        (
            "s",
            Arc::new(StringArray::from_iter((0..4096).map(|i| {
                (i % 5 != 0).then(|| format!("dictionary-value-{}", i % 7))
            }))) as ArrayRef,
        ),
    ])
    .unwrap();
    for version in [WriterVersion::PARQUET_1_0, WriterVersion::PARQUET_2_0] {
        for compression in [
            Compression::UNCOMPRESSED,
            Compression::SNAPPY,
            Compression::ZSTD(Default::default()),
        ] {
            for dictionary in [false, true] {
                for enabled in [false, true] {
                    // Resolve column overrides through the upstream property path.
                    // Both the normal and byte-array Arrow page writers use the
                    // global checksum option, independently of these overrides.
                    let props = WriterProperties::builder()
                        .set_writer_version(version)
                        .set_dictionary_enabled(!dictionary)
                        .set_column_dictionary_enabled(ColumnPath::from("v"), dictionary)
                        .set_column_dictionary_enabled(ColumnPath::from("s"), dictionary)
                        .set_column_compression(ColumnPath::from("v"), compression)
                        .set_column_compression(ColumnPath::from("s"), compression)
                        .set_data_page_row_count_limit(128)
                        .set_write_batch_size(128)
                        .set_write_page_checksum(enabled)
                        .build();
                    let mut writer =
                        ArrowWriter::try_new(Vec::new(), batch.schema(), Some(props)).unwrap();
                    writer.write(&batch).unwrap();
                    let bytes = Bytes::from(writer.into_inner().unwrap());
                    let pages = check_headers(&bytes, enabled);
                    assert_eq!(pages.contains(&PageType::DICTIONARY_PAGE), dictionary);
                    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
                        .unwrap()
                        .with_batch_size(4096)
                        .build()
                        .unwrap();
                    let read = reader.collect::<Result<Vec<_>, _>>().unwrap();
                    assert_eq!(read, vec![batch.clone()]);
                }
            }
        }
    }
}
