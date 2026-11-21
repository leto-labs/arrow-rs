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

use bytes::Bytes;
use parquet::basic::Encoding;
use parquet::column::page::{CompressedPage, Page, PageWriter};
use parquet::column::writer::ColumnCloseResult;
use parquet::file::metadata::ColumnChunkMetaData;
use parquet::file::properties::WriterProperties;
use parquet::file::writer::{SerializedFileWriter, SerializedPageWriter, TrackedWrite};
use parquet::schema::{parser::parse_message_type, types::SchemaDescriptor};
use std::sync::Arc;

#[expect(deprecated, reason = "Write a fixture with deprecated V1 levels")]
fn plain_page(
    num_values: u32,
    repetition: &[u8],
    definition: &[u8],
    def_level_encoding: Encoding,
    values: impl IntoIterator<Item = u8>,
) -> Page {
    let mut buf = repetition.to_vec();
    buf.extend_from_slice(definition);
    buf.extend(values);
    Page::DataPage {
        buf: buf.into(),
        num_values,
        encoding: Encoding::PLAIN,
        def_level_encoding,
        rep_level_encoding: Encoding::BIT_PACKED,
        statistics: None,
    }
}

fn strings(values: &[&str]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for value in values {
        bytes.extend((value.len() as u32).to_le_bytes());
        bytes.extend(value.as_bytes());
    }
    bytes
}

#[expect(deprecated, reason = "Write a fixture with deprecated V1 levels")]
pub fn generate() -> Bytes {
    let schema = Arc::new(
        parse_message_type(
            "message fixture {
                REQUIRED BYTE_ARRAY required_string (STRING);
                OPTIONAL BYTE_ARRAY optional_string (STRING);
                REQUIRED DOUBLE required_double;
                OPTIONAL DOUBLE optional_double;
                REQUIRED INT64 required_timestamp (TIMESTAMP(MILLIS,true));
                OPTIONAL INT64 optional_timestamp (TIMESTAMP(MILLIS,true));
                REQUIRED BOOLEAN required_bool;
                OPTIONAL BOOLEAN optional_bool;
                REPEATED group outer_items { REPEATED INT32 inner_items; }
            }",
        )
        .unwrap(),
    );
    let legacy = Encoding::BIT_PACKED;
    // The first row has the same scalar values as the parquet-mr readback.
    // The one-bit definition stream for [1, 0] is 0x80 in MSB-first order.
    let columns = [
        vec![plain_page(
            2,
            &[],
            &[],
            legacy,
            strings(&["required_string1", "required_string2"]),
        )],
        vec![plain_page(
            2,
            &[],
            &[0x80],
            legacy,
            strings(&["optional_string"]),
        )],
        vec![plain_page(
            2,
            &[],
            &[],
            legacy,
            [1.0_f64, 2.0].into_iter().flat_map(f64::to_le_bytes),
        )],
        vec![
            plain_page(1, &[], &[0x80], legacy, 1.0_f64.to_le_bytes()),
            // Two payload bytes follow the V1 RLE length prefix. The
            // repeated-run header 2 and value 1 encode one present level.
            plain_page(
                1,
                &[],
                &[2, 0, 0, 0, 2, 1],
                Encoding::RLE,
                2.0_f64.to_le_bytes(),
            ),
        ],
        vec![plain_page(
            2,
            &[],
            &[],
            legacy,
            [12_345_678_i64, 12_345_679]
                .into_iter()
                .flat_map(i64::to_le_bytes),
        )],
        vec![plain_page(
            2,
            &[],
            &[0x80],
            legacy,
            12_345_678_i64.to_le_bytes(),
        )],
        vec![plain_page(2, &[], &[], legacy, [1])],
        vec![plain_page(2, &[], &[0x80], legacy, [0])],
        // Two-bit repetition levels [0, 2] then [2, 1, 0, 2].
        // The first record crosses the page boundary. Every definition
        // level is 2, meaning the innermost value is present.
        vec![
            plain_page(
                2,
                &[0x20],
                &[0xa0],
                legacy,
                [10_i32, 11].into_iter().flat_map(i32::to_le_bytes),
            ),
            plain_page(
                4,
                &[0x92],
                &[0xaa],
                legacy,
                [12_i32, 13, 20, 21].into_iter().flat_map(i32::to_le_bytes),
            ),
        ],
    ];

    let props = WriterProperties::builder()
        .set_created_by("Arrow Rust legacy BIT_PACKED test fixture".into())
        .build();
    let descriptors = SchemaDescriptor::new(schema.clone());
    assert_eq!(descriptors.num_columns(), columns.len());
    let mut writer = SerializedFileWriter::new(Vec::new(), schema, Arc::new(props)).unwrap();
    let mut group = writer.next_row_group().unwrap();

    for (index, pages) in columns.into_iter().enumerate() {
        let mut out = TrackedWrite::new(Vec::new());
        let mut page_writer = SerializedPageWriter::new(&mut out);
        let mut num_values = 0;
        let mut encodings = vec![Encoding::PLAIN, legacy];
        for page in pages {
            if let Page::DataPage {
                def_level_encoding: Encoding::RLE,
                ..
            } = &page
            {
                encodings.push(Encoding::RLE);
            }
            let size = page.buffer().len();
            let spec = page_writer
                .write_page(CompressedPage::new(page, size))
                .unwrap();
            num_values += i64::from(spec.num_values);
        }
        page_writer.close().unwrap();
        let bytes = Bytes::from(out.into_inner().unwrap());
        let metadata = ColumnChunkMetaData::builder(descriptors.column(index))
            .set_num_values(num_values)
            .set_encodings(encodings)
            .set_total_compressed_size(bytes.len() as i64)
            .set_total_uncompressed_size(bytes.len() as i64)
            .set_data_page_offset(0)
            .build()
            .unwrap();
        group
            .append_column(
                &bytes,
                ColumnCloseResult {
                    rows_written: 2,
                    bytes_written: bytes.len() as u64,
                    metadata,
                    bloom_filter: None,
                    column_index: None,
                    offset_index: None,
                },
            )
            .unwrap();
    }
    group.close().unwrap();
    writer.into_inner().unwrap().into()
}
