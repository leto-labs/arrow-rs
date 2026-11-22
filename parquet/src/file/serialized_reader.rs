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

//! Contains implementations of the reader traits FileReader, RowGroupReader and PageReader
//! Also contains implementations of the ChunkReader for files (with buffering) and byte arrays (RAM)

use crate::basic::{Compression, PageType, Type};
use crate::bloom_filter::Sbbf;
use crate::column::page::{Page, PageMetadata, PageReader};
use crate::compression::{Codec, create_codec};
#[cfg(feature = "encryption")]
use crate::encryption::decrypt::{CryptoContext, read_and_decrypt};
use crate::errors::{ParquetError, Result};
use crate::file::metadata::page_index::RowGroupPageIndex;
use crate::file::metadata::thrift::PageHeader;
use crate::file::page_cache::plain::{PlainChunk, is_dictionary_encoded};
use crate::file::page_cache::{
    CachedPage, PageCache, PageFill, PageLookup, PageWithValidity, PinnedPages, page_cache_applies,
};
use crate::file::page_index::offset_index::PageLocation;
use crate::file::statistics;
use crate::file::{
    metadata::*,
    properties::{ReaderProperties, ReaderPropertiesPtr},
    reader::*,
};
#[cfg(feature = "encryption")]
use crate::parquet_thrift::ThriftSliceInputProtocol;
use crate::parquet_thrift::{ReadThrift, ThriftReadInputProtocol};
use crate::record::Row;
use crate::record::reader::RowIter;
use crate::schema::types::{SchemaDescPtr, Type as SchemaType};
use bytes::Bytes;
use std::collections::VecDeque;
use std::{fs::File, io::Read, path::Path, sync::Arc};

impl TryFrom<File> for SerializedFileReader<File> {
    type Error = ParquetError;

    fn try_from(file: File) -> Result<Self> {
        Self::new(file)
    }
}

impl TryFrom<&Path> for SerializedFileReader<File> {
    type Error = ParquetError;

    fn try_from(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        Self::try_from(file)
    }
}

impl TryFrom<String> for SerializedFileReader<File> {
    type Error = ParquetError;

    fn try_from(path: String) -> Result<Self> {
        Self::try_from(Path::new(&path))
    }
}

impl TryFrom<&str> for SerializedFileReader<File> {
    type Error = ParquetError;

    fn try_from(path: &str) -> Result<Self> {
        Self::try_from(Path::new(&path))
    }
}

/// Conversion into a [`RowIter`]
/// using the full file schema over all row groups.
impl IntoIterator for SerializedFileReader<File> {
    type Item = Result<Row>;
    type IntoIter = RowIter<'static>;

    fn into_iter(self) -> Self::IntoIter {
        RowIter::from_file_into(Box::new(self))
    }
}

// ----------------------------------------------------------------------
// Implementations of file & row group readers

/// A serialized implementation for Parquet [`FileReader`].
pub struct SerializedFileReader<R: ChunkReader> {
    chunk_reader: Arc<R>,
    metadata: Arc<ParquetMetaData>,
    props: ReaderPropertiesPtr,
}

/// A predicate for filtering row groups, invoked with the metadata and index
/// of each row group in the file. Only row groups for which the predicate
/// evaluates to `true` will be scanned
pub type ReadGroupPredicate = Box<dyn FnMut(&RowGroupMetaData, usize) -> bool>;

/// A builder for [`ReadOptions`].
/// For the predicates that are added to the builder,
/// they will be chained using 'AND' to filter the row groups.
#[derive(Default)]
pub struct ReadOptionsBuilder {
    predicates: Vec<ReadGroupPredicate>,
    enable_page_index: bool,
    props: Option<ReaderProperties>,
    metadata_options: ParquetMetaDataOptions,
}

impl ReadOptionsBuilder {
    /// New builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a predicate on row group metadata to the reading option,
    /// Filter only row groups that match the predicate criteria
    pub fn with_predicate(mut self, predicate: ReadGroupPredicate) -> Self {
        self.predicates.push(predicate);
        self
    }

    /// Add a range predicate on filtering row groups if their midpoints are within
    /// the Closed-Open range `[start..end) {x | start <= x < end}`
    ///
    /// # Panics
    ///
    /// Panics if `end <= start`
    pub fn with_range(mut self, start: i64, end: i64) -> Self {
        assert!(start < end);
        let predicate = move |rg: &RowGroupMetaData, _: usize| {
            let mid = get_midpoint_offset(rg);
            mid >= start && mid < end
        };
        self.predicates.push(Box::new(predicate));
        self
    }

    /// Enable reading the page index structures described in
    /// "[Column Index] Layout to Support Page Skipping"
    ///
    /// [Column Index]: https://github.com/apache/parquet-format/blob/master/PageIndex.md
    pub fn with_page_index(mut self) -> Self {
        self.enable_page_index = true;
        self
    }

    /// Set the [`ReaderProperties`] configuration.
    pub fn with_reader_properties(mut self, properties: ReaderProperties) -> Self {
        self.props = Some(properties);
        self
    }

    /// Provide a Parquet schema to use when decoding the metadata. The schema in the Parquet
    /// footer will be skipped.
    pub fn with_parquet_schema(mut self, schema: SchemaDescPtr) -> Self {
        self.metadata_options.set_schema(schema);
        self
    }

    /// Set whether to convert the [`encoding_stats`] in the Parquet `ColumnMetaData` to a bitmask
    /// (defaults to `false`).
    ///
    /// See [`ColumnChunkMetaData::page_encoding_stats_mask`] for an explanation of why this
    /// might be desirable.
    ///
    /// [`encoding_stats`]:
    /// https://github.com/apache/parquet-format/blob/786142e26740487930ddc3ec5e39d780bd930907/src/main/thrift/parquet.thrift#L917
    pub fn with_encoding_stats_as_mask(mut self, val: bool) -> Self {
        self.metadata_options.set_encoding_stats_as_mask(val);
        self
    }

    /// Sets the decoding policy for [`encoding_stats`] in the Parquet `ColumnMetaData`.
    ///
    /// [`encoding_stats`]:
    /// https://github.com/apache/parquet-format/blob/786142e26740487930ddc3ec5e39d780bd930907/src/main/thrift/parquet.thrift#L917
    pub fn with_encoding_stats_policy(mut self, policy: ParquetStatisticsPolicy) -> Self {
        self.metadata_options.set_encoding_stats_policy(policy);
        self
    }

    /// Sets the decoding policy for [`statistics`] in the Parquet `ColumnMetaData`.
    ///
    /// [`statistics`]:
    /// https://github.com/apache/parquet-format/blob/786142e26740487930ddc3ec5e39d780bd930907/src/main/thrift/parquet.thrift#L912
    pub fn with_column_stats_policy(mut self, policy: ParquetStatisticsPolicy) -> Self {
        self.metadata_options.set_column_stats_policy(policy);
        self
    }

    /// Sets the decoding policy for [`size_statistics`] in the Parquet `ColumnMetaData`.
    ///
    /// [`size_statistics`]:
    /// https://github.com/apache/parquet-format/blob/786142e26740487930ddc3ec5e39d780bd930907/src/main/thrift/parquet.thrift#L936
    pub fn with_size_stats_policy(mut self, policy: ParquetStatisticsPolicy) -> Self {
        self.metadata_options.set_size_stats_policy(policy);
        self
    }

    /// Seal the builder and return the read options
    pub fn build(self) -> ReadOptions {
        let props = self
            .props
            .unwrap_or_else(|| ReaderProperties::builder().build());
        ReadOptions {
            predicates: self.predicates,
            enable_page_index: self.enable_page_index,
            props,
            metadata_options: self.metadata_options,
        }
    }
}

/// A collection of options for reading a Parquet file.
///
/// Predicates are currently only supported on row group metadata.
/// All predicates will be chained using 'AND' to filter the row groups.
pub struct ReadOptions {
    predicates: Vec<ReadGroupPredicate>,
    enable_page_index: bool,
    props: ReaderProperties,
    metadata_options: ParquetMetaDataOptions,
}

impl<R: 'static + ChunkReader> SerializedFileReader<R> {
    /// Creates file reader from a Parquet file.
    /// Returns an error if the Parquet file does not exist or is corrupt.
    pub fn new(chunk_reader: R) -> Result<Self> {
        let metadata = ParquetMetaDataReader::new().parse_and_finish(&chunk_reader)?;
        let props = Arc::new(ReaderProperties::builder().build());
        Ok(Self {
            chunk_reader: Arc::new(chunk_reader),
            metadata: Arc::new(metadata),
            props,
        })
    }

    /// Creates file reader from a Parquet file with read options.
    /// Returns an error if the Parquet file does not exist or is corrupt.
    pub fn new_with_options(chunk_reader: R, options: ReadOptions) -> Result<Self> {
        let mut metadata_builder = ParquetMetaDataReader::new()
            .with_metadata_options(Some(options.metadata_options.clone()))
            .parse_and_finish(&chunk_reader)?
            .into_builder();
        let mut predicates = options.predicates;

        // Filter row groups based on the predicates
        for (i, rg_meta) in metadata_builder.take_row_groups().into_iter().enumerate() {
            let mut keep = true;
            for predicate in &mut predicates {
                if !predicate(&rg_meta, i) {
                    keep = false;
                    break;
                }
            }
            if keep {
                metadata_builder = metadata_builder.add_row_group(rg_meta);
            }
        }

        let mut metadata = metadata_builder.build();

        // If page indexes are desired, build them with the filtered set of row groups
        if options.enable_page_index {
            let mut reader = ParquetMetaDataReader::new_with_metadata(metadata)
                .with_page_index_policy(PageIndexPolicy::Required);
            reader.read_page_indexes(&chunk_reader)?;
            metadata = reader.finish()?;
        }

        Ok(Self {
            chunk_reader: Arc::new(chunk_reader),
            metadata: Arc::new(metadata),
            props: Arc::new(options.props),
        })
    }
}

/// Get midpoint offset for a row group
fn get_midpoint_offset(meta: &RowGroupMetaData) -> i64 {
    let col = meta.column(0);
    let mut offset = col.data_page_offset();
    if let Some(dic_offset) = col.dictionary_page_offset()
        && offset > dic_offset
    {
        offset = dic_offset
    }
    offset + meta.compressed_size() / 2
}

impl<R: 'static + ChunkReader> FileReader for SerializedFileReader<R> {
    fn metadata(&self) -> &ParquetMetaData {
        &self.metadata
    }

    fn num_row_groups(&self) -> usize {
        self.metadata.num_row_groups()
    }

    fn get_row_group(&self, i: usize) -> Result<Box<dyn RowGroupReader + '_>> {
        let row_group_metadata = self.metadata.row_group(i);
        // Row groups should be processed sequentially.
        let props = Arc::clone(&self.props);
        let f = Arc::clone(&self.chunk_reader);
        let page_index = self.metadata.page_index_for_row_group(i);
        Ok(Box::new(SerializedRowGroupReader::new(
            f,
            row_group_metadata,
            page_index,
            props,
        )?))
    }

    fn get_row_iter(&self, projection: Option<SchemaType>) -> Result<RowIter<'_>> {
        RowIter::from_file(projection, self)
    }
}

/// A serialized implementation for Parquet [`RowGroupReader`].
pub struct SerializedRowGroupReader<'a, R: ChunkReader> {
    chunk_reader: Arc<R>,
    metadata: &'a RowGroupMetaData,
    page_index: RowGroupPageIndex,
    props: ReaderPropertiesPtr,
    bloom_filters: Vec<Option<Sbbf>>,
}

impl<'a, R: ChunkReader> SerializedRowGroupReader<'a, R> {
    /// Creates new row group reader from a file, row group metadata and custom config.
    pub fn new(
        chunk_reader: Arc<R>,
        metadata: &'a RowGroupMetaData,
        page_index: RowGroupPageIndex,
        props: ReaderPropertiesPtr,
    ) -> Result<Self> {
        let bloom_filters = if props.read_bloom_filter() {
            metadata
                .columns()
                .iter()
                .map(|col| Sbbf::read_from_column_chunk(col, &*chunk_reader))
                .collect::<Result<Vec<_>>>()?
        } else {
            std::iter::repeat_n(None, metadata.columns().len()).collect()
        };
        Ok(Self {
            chunk_reader,
            metadata,
            page_index,
            props,
            bloom_filters,
        })
    }
}

impl<R: 'static + ChunkReader> RowGroupReader for SerializedRowGroupReader<'_, R> {
    fn metadata(&self) -> &RowGroupMetaData {
        self.metadata
    }

    fn num_columns(&self) -> usize {
        self.metadata.num_columns()
    }

    // TODO: fix PARQUET-816
    fn get_column_page_reader(&self, i: usize) -> Result<Box<dyn PageReader>> {
        let col = self.metadata.column(i);

        // TODO(ets): push page index into page reader so we don't have to clone here
        let page_locations = self.page_index.page_locations(i).cloned();

        let props = Arc::clone(&self.props);
        Ok(Box::new(SerializedPageReader::new_with_properties(
            Arc::clone(&self.chunk_reader),
            col,
            usize::try_from(self.metadata.num_rows())?,
            page_locations,
            props,
        )?))
    }

    /// get bloom filter for the `i`th column
    fn get_column_bloom_filter(&self, i: usize) -> Option<&Sbbf> {
        self.bloom_filters[i].as_ref()
    }

    fn get_row_iter(&self, projection: Option<SchemaType>) -> Result<RowIter<'_>> {
        RowIter::from_row_group(projection, self)
    }
}

/// Decodes a [`Page`] from the provided `buffer`
pub(crate) fn decode_page(
    page_header: PageHeader,
    buffer: Bytes,
    physical_type: Type,
    decompressor: Option<&mut Box<dyn Codec>>,
) -> Result<Page> {
    // Verify the 32-bit CRC checksum of the page, before the page data is
    // decompressed or decoded. The checksum covers the page as stored: for a
    // compressed page, the compressed bytes.
    #[cfg(feature = "crc")]
    if let Some(expected_crc) = page_header.crc {
        let crc = crc32fast::hash(&buffer);
        if crc != expected_crc as u32 {
            return Err(general_err!(
                "Page CRC checksum mismatch: the {} header declares {:#010x}, the {} bytes of page data have {:#010x}",
                page_header.r#type,
                expected_crc as u32,
                buffer.len(),
                crc
            ));
        }
    }

    // When processing data page v2, depending on enabled compression for the
    // page, we should account for uncompressed data ('offset') of
    // repetition and definition levels.
    //
    // We always use 0 offset for other pages other than v2, `true` flag means
    // that compression will be applied if decompressor is defined
    let (offset, can_decompress): (usize, bool) = match page_header.data_page_header_v2 {
        Some(ref header_v2) => {
            // Sum in 64 bits: two valid 32-bit lengths can exceed `i32::MAX`.
            let levels_byte_length = header_v2.definition_levels_byte_length as i64
                + header_v2.repetition_levels_byte_length as i64;
            if header_v2.definition_levels_byte_length < 0
                || header_v2.repetition_levels_byte_length < 0
                || levels_byte_length > page_header.uncompressed_page_size as i64
            {
                return Err(general_err!(
                    "DataPage v2 header contains implausible values \
                        for definition_levels_byte_length ({}) \
                        and repetition_levels_byte_length ({}) \
                        given DataPage header provides uncompressed_page_size ({})",
                    header_v2.definition_levels_byte_length,
                    header_v2.repetition_levels_byte_length,
                    page_header.uncompressed_page_size
                ));
            }
            (
                usize::try_from(levels_byte_length)?,
                // When is_compressed flag is missing the page is considered compressed
                header_v2.is_compressed.unwrap_or(true),
            )
        }
        None => (0, true),
    };

    let buffer = match decompressor {
        Some(decompressor) if can_decompress => {
            let uncompressed_page_size = usize::try_from(page_header.uncompressed_page_size)?;
            if offset > buffer.len() || offset > uncompressed_page_size {
                return Err(general_err!("Invalid page header"));
            }
            let decompressed_size = uncompressed_page_size - offset;
            // The size is from the page header, so nothing is reserved here.
            // Each codec reserves its output after it has compared the size
            // with the compressed data, and limits its output to the size.
            let mut decompressed = Vec::new();
            decompressed.extend_from_slice(&buffer[..offset]);
            // decompressed size of zero corresponds to a page with no non-null values
            // see https://github.com/apache/parquet-format/blob/master/README.md#data-pages
            if decompressed_size > 0 {
                let compressed = &buffer[offset..];
                decompressor.decompress(compressed, &mut decompressed, Some(decompressed_size))?;
            }

            if decompressed.len() != uncompressed_page_size {
                return Err(general_err!(
                    "Actual decompressed size doesn't match the expected one ({} vs {})",
                    decompressed.len(),
                    uncompressed_page_size
                ));
            }

            Bytes::from(decompressed)
        }
        _ => buffer,
    };

    // The header sizes are not trusted. Without a decompressor the buffer has
    // the compressed size, so the levels can still be longer than the page.
    if offset > buffer.len() {
        return Err(general_err!(
            "DataPage v2 levels need {} bytes, but the page has {} bytes",
            offset,
            buffer.len()
        ));
    }

    let result = match page_header.r#type {
        PageType::DICTIONARY_PAGE => {
            let dict_header = page_header.dictionary_page_header.as_ref().ok_or_else(|| {
                ParquetError::General("Missing dictionary page header".to_string())
            })?;
            let is_sorted = dict_header.is_sorted.unwrap_or(false);
            Page::DictionaryPage {
                buf: buffer,
                num_values: dict_header.num_values.try_into()?,
                encoding: dict_header.encoding,
                is_sorted,
            }
        }
        PageType::DATA_PAGE => {
            let header = page_header
                .data_page_header
                .ok_or_else(|| ParquetError::General("Missing V1 data page header".to_string()))?;
            Page::DataPage {
                buf: buffer,
                num_values: header.num_values.try_into()?,
                encoding: header.encoding,
                def_level_encoding: header.definition_level_encoding,
                rep_level_encoding: header.repetition_level_encoding,
                statistics: statistics::from_thrift_page_stats(physical_type, header.statistics)?,
            }
        }
        PageType::DATA_PAGE_V2 => {
            let header = page_header
                .data_page_header_v2
                .ok_or_else(|| ParquetError::General("Missing V2 data page header".to_string()))?;
            let is_compressed = header.is_compressed.unwrap_or(true);
            Page::DataPageV2 {
                buf: buffer,
                num_values: header.num_values.try_into()?,
                encoding: header.encoding,
                num_nulls: header.num_nulls.try_into()?,
                num_rows: header.num_rows.try_into()?,
                def_levels_byte_len: header.definition_levels_byte_length.try_into()?,
                rep_levels_byte_len: header.repetition_levels_byte_length.try_into()?,
                is_compressed,
                statistics: statistics::from_thrift_page_stats(physical_type, header.statistics)?,
            }
        }
        PageType::INDEX_PAGE => {
            // For unknown page type (e.g., INDEX_PAGE), skip and read next.
            return Err(general_err!(
                "Page type {:?} is not supported",
                page_header.r#type
            ));
        }
    };

    Ok(result)
}

/// Location of a dictionary page skipped without decoding.
///
/// ```text
///            skip, and the reader can defer         first data page that
///            (record location, zero decompress)     needs the dictionary
///   NotSeen ─────────────────────────────▶ Deferred ─────────────────────▶ Decoded
///      │                                                                      ▲
///      ├── read hits the dictionary page (eager path, unchanged) ─────────────┤
///      │                                                                      │
///      └── skip, but the reader cannot defer ─────────────────────────────────┘
///          (`supports_deferred_dictionary()` is false: read it eagerly,
///           because a skipped page this reader cannot hand back is lost)
/// ```
///
/// A chunk skipped end to end stays `Deferred`: its dictionary is never
/// decompressed, which is the point. Skipping *values* needs only the RLE
/// index cursor, and skipping whole pages needs nothing at all, so the only
/// consumer of dictionary *contents* is value decoding. Random re-read is
/// what makes the deferral safe: [`ChunkReader`] hands out readers at any
/// offset, and the async path's in-memory column chunk already holds the
/// page's bytes.
enum DeferredDictionaryPage {
    /// `Values` state: the header was already parsed on the skip path, only
    /// the page body was left untouched.
    Header {
        /// File offset of the page header, the key of the page in the page
        /// cache.
        page_start: u64,
        data_start: u64,
        header: Box<PageHeader>,
    },
    /// `Pages` state: the offset index records the page location; header and
    /// body are both still unread.
    Location(PageLocation),
    /// The cache fill already loaded this dictionary.
    Decoded(Page),
    /// A resident page of the page cache, stepped over without file bytes,
    /// and the file offset of its header: taking it marks the read of a
    /// leased page.
    Cached {
        /// File offset of the page header.
        offset: u64,
        page: Page,
    },
}

enum SerializedPageReaderState {
    Values {
        /// The current byte offset in the reader
        /// Note that offset is u64 (i.e., not usize) to support 32-bit architectures such as WASM
        offset: u64,

        /// The length of the chunk in bytes
        /// Note that remaining_bytes is u64 (i.e., not usize) to support 32-bit architectures such as WASM
        remaining_bytes: u64,

        // If the next page header has already been "peeked", we will cache it and it`s length here
        next_page_header: Option<Box<PageHeader>>,

        /// The index of the data page within this column chunk
        page_index: usize,

        /// Whether the next page is expected to be a dictionary page
        require_dictionary: bool,
    },
    Pages {
        /// Remaining page locations
        page_locations: VecDeque<PageLocation>,
        /// Remaining dictionary location if any
        dictionary_page: Option<PageLocation>,
        /// The total number of rows in this column chunk
        total_rows: usize,
        /// The index of the data page within this column chunk
        page_index: usize,
    },
}

#[derive(Default)]
pub(crate) struct SerializedPageReaderContext {
    /// Controls decoding of page-level statistics
    pub(crate) read_stats: bool,
    /// Crypto context carrying objects required for decryption
    #[cfg(feature = "encryption")]
    pub(crate) crypto_context: Option<Arc<CryptoContext>>,
}

/// A serialized implementation for Parquet [`PageReader`].
pub struct SerializedPageReader<R: ChunkReader> {
    /// The chunk reader
    reader: Arc<R>,

    /// A dictionary page skipped past without decoding. See
    /// [`DeferredDictionaryPage`] for the state diagram.
    deferred_dictionary: Option<DeferredDictionaryPage>,

    /// The compression codec for this column chunk. Only set for non-PLAIN codec.
    decompressor: Option<Box<dyn Codec>>,

    /// Column chunk type.
    physical_type: Type,

    state: SerializedPageReaderState,

    context: SerializedPageReaderContext,

    /// The decompressed page cache of this file, when the column chunk is
    /// eligible. See [`crate::file::page_cache`].
    page_cache: Option<PageCacheReader>,

    /// The codec of the column chunk if its pages can be cached
    /// ([`page_cache_applies`]), else `None`.
    page_cache_codec: Option<Compression>,

    plain_chunk: Option<PlainChunk>,
    flat_nullable: bool,
}

/// The result of [`PageCacheReader::lookup`]: a hit that passed its check,
/// with the check's result, or the ticket (if any) of a read from the file
/// bytes.
type Lookup<T> = std::result::Result<(CachedPage, T), Option<Box<dyn PageFill>>>;

/// The view of a file's decompressed page cache that one column chunk reader
/// has.
struct PageCacheReader {
    cache: Arc<dyn PageCache>,
    /// Pages that fetch planning took from the cache. Their bytes were not
    /// fetched, so the reader must find them here.
    pinned: Option<Arc<PinnedPages>>,
    /// File offset of the page whose header `peek_next_page` already parsed.
    peeked_start: Option<u64>,
}

impl PageCacheReader {
    /// The page that starts at `offset`, or the ticket to publish it, and
    /// whether it is a leased page. A leased page is not marked here: the
    /// caller marks it with [`Self::mark_read`] once the page passed its
    /// checks.
    fn get(&self, offset: u64) -> (PageLookup, bool) {
        if let Some(page) = self.pinned.as_ref().and_then(|pinned| pinned.get(offset)) {
            return (PageLookup::Hit(page.clone()), true);
        }
        (self.cache.get(offset, true), false)
    }

    /// Records a read of the leased page at `offset` in its lease (the
    /// release records it). Nothing for a page that is not leased: its
    /// lookup in the cache counted the read.
    fn mark_read(&self, offset: u64) {
        if let Some(pinned) = &self.pinned {
            pinned.read(offset);
        }
    }

    /// The page that starts at `offset` if it is resident; records no access.
    fn probe(&self, offset: u64) -> Option<CachedPage> {
        if let Some(page) = self.pinned.as_ref().and_then(|pinned| pinned.get(offset)) {
            return Some(page.clone());
        }
        self.cache.probe(offset)
    }

    /// Whether fetch planning pinned the page at `offset`: its bytes were
    /// not fetched, so the reader has only the cached page.
    fn is_pinned(&self, offset: u64) -> bool {
        self.pinned
            .as_ref()
            .is_some_and(|pinned| pinned.get(offset).is_some())
    }

    /// Decides what a hit that failed `checked` means: a pinned page is an
    /// error, because its bytes were not fetched; any other page is read
    /// from the file bytes, as if it were not resident.
    fn on_failed_check<T>(&self, offset: u64, checked: Result<T>) -> Result<Option<T>> {
        match checked {
            Ok(value) => Ok(Some(value)),
            Err(e) if self.is_pinned(offset) => Err(e),
            Err(_) => Ok(None),
        }
    }

    /// Looks the page at `offset` up for a read. `Ok(Ok(..))` is a hit that
    /// passed `check`, with its result. `Ok(Err(ticket))` reads the page from
    /// the file bytes, with the ticket to publish it, if any.
    fn lookup<T>(
        &self,
        offset: u64,
        check: impl FnOnce(&CachedPage) -> Result<T>,
    ) -> Result<Lookup<T>> {
        match self.get(offset) {
            (PageLookup::Hit(cached), leased) => {
                if cached.is_dictionary_marker() && !leased {
                    return Ok(Err(None));
                }
                let checked = check(&cached);
                Ok(match self.on_failed_check(offset, checked)? {
                    Some(value) => {
                        // A leased page counts as read only once it passed
                        // its checks: a rejected page records no read.
                        if leased && !cached.is_dictionary_marker() {
                            self.mark_read(offset);
                        }
                        Ok((cached, value))
                    }
                    None => Err(None),
                })
            }
            (PageLookup::Miss(ticket), _) => Ok(Err(ticket)),
        }
    }

    /// The resident page at `offset` that passes `check`, for a step that
    /// needs no access record (peek, skip). `None` reads the file bytes.
    fn resident(
        &self,
        offset: u64,
        check: impl FnOnce(&CachedPage) -> Result<()>,
    ) -> Result<Option<CachedPage>> {
        let Some(cached) = self.probe(offset) else {
            return Ok(None);
        };
        let checked = check(&cached);
        Ok(self.on_failed_check(offset, checked)?.map(|()| cached))
    }
}

/// Whether a V2 page in a compressed chunk uses the chunk's codec.
fn page_body_is_compressed(header: &PageHeader) -> bool {
    header
        .data_page_header_v2
        .as_ref()
        .is_none_or(|v2| v2.is_compressed.unwrap_or(true))
}

/// Classifies a dictionary even when this reader has no fill ticket. The
/// caller keeps the original dictionary because some pages may still use it.
fn publish_page(
    fill: Option<Box<dyn PageFill>>,
    owns_buffer: bool,
    page: Page,
    stored_len: u64,
    codec: Compression,
    plain_chunk: Option<&mut PlainChunk>,
    flat_nullable: bool,
) -> Page {
    let mut owns_buffer = owns_buffer && codec != Compression::UNCOMPRESSED;
    let mut page = page;
    if let Some(chunk) = plain_chunk {
        if page.is_dictionary_page() {
            let plain = chunk.observe_dictionary(&page);
            if plain && let Some(fill) = fill {
                if owns_buffer {
                    fill.complete(CachedPage::dictionary_marker(&page, stored_len, codec));
                }
                return page;
            }
        } else if fill.is_some() && is_dictionary_encoded(page.encoding()) {
            let Some((stored, converted)) = chunk.dictionary_data_page(&page) else {
                return page;
            };
            page = stored;
            owns_buffer |= converted;
        }
    }
    if let Some(fill) = fill {
        let cached = CachedPage::new(page.clone(), stored_len, codec);
        if owns_buffer {
            cached.publish(fill, flat_nullable);
        } else {
            cached.publish_with_copy(fill, flat_nullable);
        }
    }
    page
}

/// Whether the decoded page of a page location can be cached with the
/// location's length: the header and the body that the header declares are
/// exactly the location. A cold read decodes the whole location after the
/// header, but a reader without the offset index steps by the header's body
/// length, so a cached length must be both.
fn location_matches_header(header_len: usize, header: &PageHeader, location_len: usize) -> bool {
    usize::try_from(header.compressed_page_size)
        .ok()
        .and_then(|body| body.checked_add(header_len))
        == Some(location_len)
}

/// The checks that a cached page repeats on each hit (see
/// [`crate::file::page_cache`]): it was decompressed with the codec of the
/// chunk and it fits the `remaining` bytes of the chunk.
fn check_cached_page(
    cached: &CachedPage,
    page_start: u64,
    remaining: u64,
    codec: Compression,
) -> Result<()> {
    if cached.codec() != codec {
        return Err(general_err!(
            "Cached page at offset {} was decompressed with {}, but its column chunk uses {}",
            page_start,
            cached.codec(),
            codec
        ));
    }
    if cached.stored_len() == 0 || cached.stored_len() > remaining {
        return Err(general_err!(
            "Cached page at offset {} does not fit its column chunk",
            page_start
        ));
    }
    Ok(())
}

/// The bytes of a cached page after the `consumed` bytes of its header that
/// a peek parsed. After a peek the header is known: the page must end where
/// the body that the header declares ends.
fn cached_rest(
    cached: &CachedPage,
    page_start: u64,
    consumed: u64,
    header: Option<&PageHeader>,
) -> Result<u64> {
    let rest = cached.stored_len().checked_sub(consumed).ok_or_else(|| {
        general_err!(
            "Cached page at offset {} is shorter than its header",
            page_start
        )
    })?;
    if let Some(header) = header
        && i64::try_from(rest).ok() != Some(i64::from(header.compressed_page_size))
    {
        return Err(general_err!(
            "Cached page at offset {} has {} bytes after its header, but the header declares {}",
            page_start,
            rest,
            header.compressed_page_size
        ));
    }
    Ok(rest)
}

/// [`check_cached_page`] for a page with an offset index location: the
/// stored length is the location's length and the page kind is the
/// location's kind.
fn check_cached_location(
    cached: &CachedPage,
    location: &PageLocation,
    is_dictionary_page: bool,
    codec: Compression,
) -> Result<()> {
    if cached.stored_len() != location.compressed_page_size as u64 {
        return Err(general_err!(
            "Cached page at offset {} has {} bytes, but its page location has {}",
            location.offset,
            cached.stored_len(),
            location.compressed_page_size
        ));
    }
    if cached.page().is_dictionary_page() != is_dictionary_page {
        return Err(general_err!(
            "Cached page at offset {} has the wrong page type",
            location.offset
        ));
    }
    check_cached_page(cached, location.offset as u64, u64::MAX, codec)
}

impl<R: ChunkReader> SerializedPageReader<R> {
    /// Creates a new serialized page reader from a chunk reader and metadata
    pub fn new(
        reader: Arc<R>,
        column_chunk_metadata: &ColumnChunkMetaData,
        total_rows: usize,
        page_locations: Option<Vec<PageLocation>>,
    ) -> Result<Self> {
        let props = Arc::new(ReaderProperties::builder().build());
        SerializedPageReader::new_with_properties(
            reader,
            column_chunk_metadata,
            total_rows,
            page_locations,
            props,
        )
    }

    /// Stub No-op implementation when encryption is disabled.
    #[cfg(all(feature = "arrow", not(feature = "encryption")))]
    pub(crate) fn add_crypto_context(
        self,
        _rg_idx: usize,
        _column_idx: usize,
        _parquet_meta_data: &ParquetMetaData,
        _column_chunk_metadata: &ColumnChunkMetaData,
    ) -> Result<SerializedPageReader<R>> {
        Ok(self)
    }

    /// Adds any necessary crypto context to this page reader, if encryption is enabled.
    #[cfg(feature = "encryption")]
    pub(crate) fn add_crypto_context(
        mut self,
        rg_idx: usize,
        column_idx: usize,
        parquet_meta_data: &ParquetMetaData,
        column_chunk_metadata: &ColumnChunkMetaData,
    ) -> Result<SerializedPageReader<R>> {
        let Some(file_decryptor) = parquet_meta_data.file_decryptor() else {
            return Ok(self);
        };
        let Some(crypto_metadata) = column_chunk_metadata.crypto_metadata() else {
            return Ok(self);
        };
        let crypto_context =
            CryptoContext::for_column(file_decryptor, crypto_metadata, rg_idx, column_idx)?;
        self.context.crypto_context = Some(Arc::new(crypto_context));
        Ok(self)
    }

    /// Creates a new serialized page with custom options.
    pub fn new_with_properties(
        reader: Arc<R>,
        meta: &ColumnChunkMetaData,
        total_rows: usize,
        page_locations: Option<Vec<PageLocation>>,
        props: ReaderPropertiesPtr,
    ) -> Result<Self> {
        let decompressor = create_codec(meta.compression(), props.codec_options())?;
        let (start, len) = meta.byte_range();

        let state = match page_locations {
            Some(locations) => {
                // If the offset of the first page doesn't match the start of the column chunk
                // then the preceding space must contain a dictionary page.
                let dictionary_page = match locations.first() {
                    Some(dict_offset) if dict_offset.offset as u64 != start => Some(PageLocation {
                        offset: start as i64,
                        compressed_page_size: (dict_offset.offset as u64 - start) as i32,
                        first_row_index: 0,
                    }),
                    _ => None,
                };

                SerializedPageReaderState::Pages {
                    page_locations: locations.into(),
                    dictionary_page,
                    total_rows,
                    page_index: 0,
                }
            }
            None => SerializedPageReaderState::Values {
                offset: start,
                remaining_bytes: len,
                next_page_header: None,
                page_index: 0,
                require_dictionary: meta.dictionary_page_offset().is_some(),
            },
        };
        let mut context = SerializedPageReaderContext::default();
        if props.read_page_stats() {
            context.read_stats = true;
        }
        Ok(Self {
            reader,
            deferred_dictionary: None,
            decompressor,
            state,
            physical_type: meta.column_type(),
            context,
            page_cache: None,
            plain_chunk: (meta.compression() != Compression::UNCOMPRESSED)
                .then(|| PlainChunk::new(meta))
                .flatten(),
            flat_nullable: meta.column_descr().max_rep_level() == 0
                && meta.column_descr().max_def_level() == 1
                && meta.column_descr().path().parts().len() == 1,
            page_cache_codec: page_cache_applies(meta, props.read_page_stats())
                .then(|| meta.compression()),
        })
    }

    /// Reads this column chunk through the decompressed page cache of its
    /// file. `pinned` holds the pages that fetch planning already took from
    /// the cache.
    ///
    /// The cache is used only where [`page_cache_applies`]. Fetch planning
    /// decides with the same function.
    #[cfg(feature = "arrow")]
    pub(crate) fn with_page_cache(
        mut self,
        cache: Option<Arc<dyn PageCache>>,
        pinned: Option<Arc<PinnedPages>>,
    ) -> Self {
        self.page_cache = cache
            .filter(|_| self.page_cache_codec.is_some())
            .map(|cache| PageCacheReader {
                cache,
                pinned,
                peeked_start: None,
            });
        if self.page_cache.is_none() {
            self.plain_chunk = None;
        }
        self
    }

    // Unselected reads can fill an admitted compressed page without recording
    // a read of an already resident or leased page. The full read path does
    // not use this operation.
    fn fill_skipped_data_page(&mut self) -> Result<()> {
        let Some(cache) = &self.page_cache else {
            return Ok(());
        };
        if self.decompressor.is_none() {
            return Ok(());
        }
        let start = match &self.state {
            SerializedPageReaderState::Values {
                next_page_header, ..
            } => {
                let Some(header) = next_page_header.as_deref() else {
                    // The preceding peek saw a resident page or no page.
                    return Ok(());
                };
                if !matches!(header.r#type, PageType::DATA_PAGE | PageType::DATA_PAGE_V2)
                    || !page_body_is_compressed(header)
                {
                    return Ok(());
                }
                let Some(start) = cache.peeked_start else {
                    return Ok(());
                };
                start
            }
            SerializedPageReaderState::Pages {
                dictionary_page,
                page_locations,
                ..
            } => {
                let Some(front) = page_locations.front().filter(|_| dictionary_page.is_none())
                else {
                    return Ok(());
                };
                u64::try_from(front.offset)?
            }
        };
        if cache.is_pinned(start) {
            return Ok(());
        }
        let PageLookup::Miss(Some(fill)) = cache.cache.get_with_accounting(start, true, |_| false)
        else {
            return Ok(());
        };
        let (header, data, page_index, stored_len) = match &self.state {
            SerializedPageReaderState::Values {
                offset,
                remaining_bytes,
                next_page_header,
                page_index,
                ..
            } => {
                let header = next_page_header.as_deref().expect("peeked header");
                verify_page_size(
                    header.compressed_page_size,
                    header.uncompressed_page_size,
                    *remaining_bytes,
                )?;
                let len = header.compressed_page_size as usize;
                let data = self.reader.get_bytes(*offset, len)?;
                (
                    header.clone(),
                    data,
                    *page_index,
                    *offset - start + len as u64,
                )
            }
            SerializedPageReaderState::Pages {
                page_locations,
                page_index,
                ..
            } => {
                let front = page_locations.front().expect("next data page");
                let len = usize::try_from(front.compressed_page_size)?;
                let buffer = self.reader.get_bytes(start, len)?;
                let (header_len, header) = read_page_header_len_from_bytes(
                    &self.context,
                    buffer.as_ref(),
                    *page_index,
                    false,
                )?;
                if !matches!(header.r#type, PageType::DATA_PAGE | PageType::DATA_PAGE_V2)
                    || !location_matches_header(header_len, &header, len)
                    || !page_body_is_compressed(&header)
                {
                    return Ok(());
                }
                (header, buffer.slice(header_len..), *page_index, len as u64)
            }
        };
        let data = self.context.decrypt_page_data(data, page_index, false)?;
        let page = decode_page(header, data, self.physical_type, self.decompressor.as_mut())?;
        if is_dictionary_encoded(page.encoding())
            && self
                .plain_chunk
                .as_ref()
                .is_some_and(PlainChunk::needs_dictionary)
            && let Some(dictionary) = self.take_deferred_dictionary()?
        {
            self.deferred_dictionary = Some(DeferredDictionaryPage::Decoded(dictionary));
        }
        let codec = self.cache_codec();
        publish_page(
            Some(fill),
            true,
            page,
            stored_len,
            codec,
            self.plain_chunk.as_mut(),
            self.flat_nullable,
        );
        Ok(())
    }

    /// The codec that cached pages of this chunk must have.
    fn cache_codec(&self) -> Compression {
        self.page_cache_codec.unwrap_or(Compression::UNCOMPRESSED)
    }

    /// Similar to `peek_next_page`, but returns the offset of the next page instead of the page metadata.
    /// Unlike page metadata, an offset can uniquely identify a page.
    ///
    /// This is used when we need to read parquet with row-filter, and we don't want to decompress the page twice.
    /// This function allows us to check if the next page is being cached or read previously.
    #[cfg(test)]
    fn peek_next_page_offset(&mut self) -> Result<Option<u64>> {
        match &mut self.state {
            SerializedPageReaderState::Values {
                offset,
                remaining_bytes,
                next_page_header,
                page_index,
                require_dictionary,
            } => {
                loop {
                    if *remaining_bytes == 0 {
                        return Ok(None);
                    }
                    return if let Some(header) = next_page_header.as_ref() {
                        if let Ok(_page_meta) = PageMetadata::try_from(&**header) {
                            Ok(Some(*offset))
                        } else {
                            // For unknown page type (e.g., INDEX_PAGE), skip and read next.
                            *next_page_header = None;
                            continue;
                        }
                    } else {
                        let mut read = self.reader.get_read(*offset)?;
                        let (header_len, header) = Self::read_page_header_len(
                            &self.context,
                            &mut read,
                            *page_index,
                            *require_dictionary,
                        )?;
                        *offset += header_len as u64;
                        *remaining_bytes -= header_len as u64;
                        let page_meta = if let Ok(_page_meta) = PageMetadata::try_from(&header) {
                            Ok(Some(*offset))
                        } else {
                            // For unknown page type (e.g., INDEX_PAGE), skip and read next.
                            continue;
                        };
                        *next_page_header = Some(Box::new(header));
                        page_meta
                    };
                }
            }
            SerializedPageReaderState::Pages {
                page_locations,
                dictionary_page,
                ..
            } => {
                if let Some(page) = dictionary_page {
                    Ok(Some(page.offset as u64))
                } else if let Some(page) = page_locations.front() {
                    Ok(Some(page.offset as u64))
                } else {
                    Ok(None)
                }
            }
        }
    }

    fn read_page_header_len<T: Read>(
        context: &SerializedPageReaderContext,
        input: &mut T,
        page_index: usize,
        dictionary_page: bool,
    ) -> Result<(usize, PageHeader)> {
        /// A wrapper around a [`std::io::Read`] that keeps track of the bytes read
        struct TrackedRead<R> {
            inner: R,
            bytes_read: usize,
        }

        impl<R: Read> Read for TrackedRead<R> {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                let v = self.inner.read(buf)?;
                self.bytes_read += v;
                Ok(v)
            }
        }

        let mut tracked = TrackedRead {
            inner: input,
            bytes_read: 0,
        };
        let header = context.read_page_header(&mut tracked, page_index, dictionary_page)?;
        Ok((tracked.bytes_read, header))
    }
}

/// Reads (and decrypts, if `context` carries a crypto context) the page header stored
/// at the front of `buffer`, returning the header and the number of bytes of `buffer`
/// it occupies.
///
/// This is exposed for callers that need to decode a single page directly from an
/// already-fetched byte range, outside of the normal [`SerializedPageReader`] iteration
/// -- e.g. decoding a dictionary page standalone.
pub(crate) fn read_page_header_len_from_bytes(
    context: &SerializedPageReaderContext,
    buffer: &[u8],
    page_index: usize,
    dictionary_page: bool,
) -> Result<(usize, PageHeader)> {
    let mut input = std::io::Cursor::new(buffer);
    let header = context.read_page_header(&mut input, page_index, dictionary_page)?;
    let header_len = input.position() as usize;
    Ok((header_len, header))
}

#[cfg(not(feature = "encryption"))]
impl SerializedPageReaderContext {
    pub(crate) fn read_page_header<T: Read>(
        &self,
        input: &mut T,
        _page_index: usize,
        _dictionary_page: bool,
    ) -> Result<PageHeader> {
        let mut prot = ThriftReadInputProtocol::new(input);
        if self.read_stats {
            Ok(PageHeader::read_thrift(&mut prot)?)
        } else {
            Ok(PageHeader::read_thrift_without_stats(&mut prot)?)
        }
    }

    pub(crate) fn decrypt_page_data<T>(
        &self,
        buffer: T,
        _page_index: usize,
        _dictionary_page: bool,
    ) -> Result<T> {
        Ok(buffer)
    }
}

#[cfg(feature = "encryption")]
impl SerializedPageReaderContext {
    pub(crate) fn read_page_header<T: Read>(
        &self,
        input: &mut T,
        page_index: usize,
        dictionary_page: bool,
    ) -> Result<PageHeader> {
        match self.page_crypto_context(page_index, dictionary_page) {
            None => {
                let mut prot = ThriftReadInputProtocol::new(input);
                if self.read_stats {
                    Ok(PageHeader::read_thrift(&mut prot)?)
                } else {
                    use crate::file::metadata::thrift::PageHeader;

                    Ok(PageHeader::read_thrift_without_stats(&mut prot)?)
                }
            }
            Some(page_crypto_context) => {
                let data_decryptor = page_crypto_context.data_decryptor();
                let aad = page_crypto_context.create_page_header_aad()?;

                let buf = read_and_decrypt(data_decryptor, input, aad.as_ref()).map_err(|_| {
                    ParquetError::General(format!(
                        "Error decrypting page header for column {}, decryption key may be wrong",
                        page_crypto_context.column_ordinal
                    ))
                })?;

                let mut prot = ThriftSliceInputProtocol::new(buf.as_slice());
                if self.read_stats {
                    Ok(PageHeader::read_thrift(&mut prot)?)
                } else {
                    Ok(PageHeader::read_thrift_without_stats(&mut prot)?)
                }
            }
        }
    }

    pub(crate) fn decrypt_page_data<T>(
        &self,
        buffer: T,
        page_index: usize,
        dictionary_page: bool,
    ) -> Result<T>
    where
        T: AsRef<[u8]>,
        T: From<Vec<u8>>,
    {
        let page_crypto_context = self.page_crypto_context(page_index, dictionary_page);
        if let Some(page_crypto_context) = page_crypto_context {
            let decryptor = page_crypto_context.data_decryptor();
            let aad = page_crypto_context.create_page_aad()?;
            let decrypted = decryptor.decrypt(buffer.as_ref(), &aad)?;
            Ok(T::from(decrypted))
        } else {
            Ok(buffer)
        }
    }

    fn page_crypto_context(
        &self,
        page_index: usize,
        dictionary_page: bool,
    ) -> Option<Arc<CryptoContext>> {
        self.crypto_context.as_ref().map(|c| {
            Arc::new(if dictionary_page {
                c.for_dictionary_page()
            } else {
                c.with_page_ordinal(page_index)
            })
        })
    }
}

impl<R: ChunkReader> Iterator for SerializedPageReader<R> {
    type Item = Result<Page>;

    fn next(&mut self) -> Option<Self::Item> {
        self.get_next_page().transpose()
    }
}

/// Metadata of a peeked page header.
///
/// Returns `None` for a page type that the reader skips (for example
/// INDEX_PAGE), and an error for a data page whose header is invalid.
fn peeked_page_metadata(header: &PageHeader) -> Result<Option<PageMetadata>> {
    match PageMetadata::try_from(header) {
        Ok(page_meta) => Ok(Some(page_meta)),
        Err(e) if matches!(header.r#type, PageType::DATA_PAGE | PageType::DATA_PAGE_V2) => Err(e),
        Err(_) => Ok(None),
    }
}

fn verify_page_header_len(header_len: usize, remaining_bytes: u64) -> Result<()> {
    if header_len as u64 > remaining_bytes {
        return Err(eof_err!("Invalid page header"));
    }
    Ok(())
}

pub(crate) fn verify_page_size(
    compressed_size: i32,
    uncompressed_size: i32,
    remaining_bytes: u64,
) -> Result<()> {
    // The page's compressed size should not exceed the remaining bytes that are
    // available to read. The page's uncompressed size is the expected size
    // after decompression, which can never be negative.
    if compressed_size < 0 || compressed_size as u64 > remaining_bytes || uncompressed_size < 0 {
        return Err(eof_err!("Invalid page header"));
    }
    Ok(())
}

impl<R: ChunkReader> PageReader for SerializedPageReader<R> {
    fn get_next_page(&mut self) -> Result<Option<Page>> {
        self.get_next_page_with_validity()
            .map(|page| page.map(|(page, _)| page))
    }

    fn get_next_page_with_validity(&mut self) -> Result<Option<PageWithValidity>> {
        let codec = self.cache_codec();
        loop {
            let (page, fill, owns_buffer, stored_len) = match &mut self.state {
                SerializedPageReaderState::Values {
                    offset,
                    remaining_bytes: remaining,
                    next_page_header,
                    page_index,
                    require_dictionary,
                } => {
                    if *remaining == 0 {
                        return Ok(None);
                    }

                    // The page starts at `offset`, or at the recorded start
                    // when its header was already peeked.
                    let peeked_start = self
                        .page_cache
                        .as_mut()
                        .and_then(|cache| cache.peeked_start.take());
                    let page_start = match next_page_header {
                        Some(_) => peeked_start,
                        None => Some(*offset),
                    };
                    let mut fill = None;
                    if let Some((cache, page_start)) = self.page_cache.as_ref().zip(page_start) {
                        // Bytes of this page that the peek consumed.
                        let consumed = *offset - page_start;
                        let header = next_page_header.as_deref();
                        match cache.lookup(page_start, |cached| {
                            check_cached_page(cached, page_start, *remaining + consumed, codec)?;
                            cached_rest(cached, page_start, consumed, header)
                        })? {
                            Ok((cached, rest)) => {
                                *next_page_header = None;
                                *offset += rest;
                                *remaining -= rest;
                                let dictionary_marker = cached.is_dictionary_marker();
                                let (page, validity) = cached.into_page_with_validity();
                                if page.is_data_page() {
                                    *page_index += 1;
                                } else if page.is_dictionary_page() {
                                    *require_dictionary = false;
                                }
                                if dictionary_marker {
                                    continue;
                                }
                                if page.is_dictionary_page()
                                    && let Some(chunk) = &mut self.plain_chunk
                                {
                                    chunk.observe_dictionary(&page);
                                }
                                return Ok(Some((page, validity)));
                            }
                            Err(ticket) => fill = ticket,
                        }
                    }
                    let page_start = page_start.unwrap_or(*offset);

                    let mut read = self.reader.get_read(*offset)?;
                    let header = if let Some(header) = next_page_header.take() {
                        *header
                    } else {
                        let (header_len, header) = Self::read_page_header_len(
                            &self.context,
                            &mut read,
                            *page_index,
                            *require_dictionary,
                        )?;
                        verify_page_header_len(header_len, *remaining)?;
                        *offset += header_len as u64;
                        *remaining -= header_len as u64;
                        header
                    };
                    verify_page_size(
                        header.compressed_page_size,
                        header.uncompressed_page_size,
                        *remaining,
                    )?;
                    let data_len = header.compressed_page_size as usize;
                    let data_start = *offset;
                    *offset += data_len as u64;
                    *remaining -= data_len as u64;

                    if header.r#type == PageType::INDEX_PAGE {
                        continue;
                    }

                    let buffer = self.reader.get_bytes(data_start, data_len)?;

                    let buffer =
                        self.context
                            .decrypt_page_data(buffer, *page_index, *require_dictionary)?;

                    let owns_buffer = fill.is_some() && page_body_is_compressed(&header);
                    let page = decode_page(
                        header,
                        buffer,
                        self.physical_type,
                        self.decompressor.as_mut(),
                    )?;
                    if page.is_data_page() {
                        *page_index += 1;
                    } else if page.is_dictionary_page() {
                        *require_dictionary = false;
                    }
                    (page, fill, owns_buffer, *offset - page_start)
                }
                SerializedPageReaderState::Pages {
                    page_locations,
                    dictionary_page,
                    page_index,
                    ..
                } => {
                    let (front, is_dictionary_page) = match dictionary_page.take() {
                        Some(front) => (front, true),
                        None => match page_locations.pop_front() {
                            Some(front) => (front, false),
                            None => return Ok(None),
                        },
                    };

                    let mut fill = None;
                    if let Some(cache) = &self.page_cache {
                        match cache.lookup(front.offset as u64, |cached| {
                            check_cached_location(cached, &front, is_dictionary_page, codec)
                        })? {
                            Ok((cached, ())) => {
                                if !is_dictionary_page {
                                    *page_index += 1;
                                }
                                if cached.is_dictionary_marker() {
                                    continue;
                                }
                                if is_dictionary_page && let Some(chunk) = &mut self.plain_chunk {
                                    chunk.observe_dictionary(cached.page());
                                }
                                return Ok(Some(cached.into_page_with_validity()));
                            }
                            Err(ticket) => fill = ticket,
                        }
                    }

                    let page_len = usize::try_from(front.compressed_page_size)?;
                    let buffer = self.reader.get_bytes(front.offset as u64, page_len)?;

                    let (offset, header) = read_page_header_len_from_bytes(
                        &self.context,
                        buffer.as_ref(),
                        *page_index,
                        is_dictionary_page,
                    )?;
                    let bytes = buffer.slice(offset..);
                    let bytes =
                        self.context
                            .decrypt_page_data(bytes, *page_index, is_dictionary_page)?;

                    if !is_dictionary_page {
                        *page_index += 1;
                    }
                    let fill = fill.filter(|_| location_matches_header(offset, &header, page_len));
                    let owns_buffer = fill.is_some() && page_body_is_compressed(&header);
                    let page = decode_page(
                        header,
                        bytes,
                        self.physical_type,
                        self.decompressor.as_mut(),
                    )?;
                    (page, fill, owns_buffer, page_len as u64)
                }
            };

            // A selection may have skipped the dictionary. Load it only
            // when a dictionary-encoded data page has a fill ticket.
            if fill.is_some()
                && page.is_data_page()
                && is_dictionary_encoded(page.encoding())
                && self
                    .plain_chunk
                    .as_ref()
                    .is_some_and(PlainChunk::needs_dictionary)
                && let Some(dictionary) = self.take_deferred_dictionary()?
            {
                self.deferred_dictionary = Some(DeferredDictionaryPage::Decoded(dictionary));
            }
            let page = publish_page(
                fill,
                owns_buffer,
                page,
                stored_len,
                codec,
                self.plain_chunk.as_mut(),
                self.flat_nullable,
            );
            return Ok(Some((page, None)));
        }
    }

    fn peek_next_page(&mut self) -> Result<Option<PageMetadata>> {
        let codec = self.cache_codec();
        match &mut self.state {
            SerializedPageReaderState::Values {
                offset,
                remaining_bytes,
                next_page_header,
                page_index,
                require_dictionary,
            } => {
                loop {
                    if *remaining_bytes == 0 {
                        return Ok(None);
                    }
                    return if let Some(header) = next_page_header.as_ref() {
                        if let Some(page_meta) = peeked_page_metadata(header)? {
                            Ok(Some(page_meta))
                        } else {
                            // For unknown page type (e.g., INDEX_PAGE), skip and read next.
                            *next_page_header = None;
                            continue;
                        }
                    } else {
                        // A resident page answers without its file bytes, and
                        // the reader does not move.
                        if let Some(cache) = &self.page_cache
                            && let Some(cached) = cache.resident(*offset, |cached| {
                                check_cached_page(cached, *offset, *remaining_bytes, codec)
                            })?
                        {
                            return Ok(Some(cached.metadata()));
                        }
                        let page_start = *offset;
                        let mut read = self.reader.get_read(*offset)?;
                        let (header_len, header) = Self::read_page_header_len(
                            &self.context,
                            &mut read,
                            *page_index,
                            *require_dictionary,
                        )?;
                        verify_page_header_len(header_len, *remaining_bytes)?;
                        *offset += header_len as u64;
                        *remaining_bytes -= header_len as u64;
                        if let Some(cache) = self.page_cache.as_mut() {
                            cache.peeked_start = Some(page_start);
                        }
                        let page_meta = if let Some(page_meta) = peeked_page_metadata(&header)? {
                            Ok(Some(page_meta))
                        } else {
                            // For unknown page type (e.g., INDEX_PAGE), skip and read next.
                            continue;
                        };
                        *next_page_header = Some(Box::new(header));
                        page_meta
                    };
                }
            }
            SerializedPageReaderState::Pages {
                page_locations,
                dictionary_page,
                total_rows,
                page_index: _,
            } => {
                if dictionary_page.is_some() {
                    Ok(Some(PageMetadata {
                        num_rows: None,
                        num_levels: None,
                        is_dict: true,
                    }))
                } else if let Some(page) = page_locations.front() {
                    let next_rows = page_locations
                        .get(1)
                        .map(|x| x.first_row_index as usize)
                        .unwrap_or(*total_rows);

                    Ok(Some(PageMetadata {
                        num_rows: Some(next_rows - page.first_row_index as usize),
                        num_levels: None,
                        is_dict: false,
                    }))
                } else {
                    Ok(None)
                }
            }
        }
    }

    fn skip_unselected_data_page(&mut self) -> Result<()> {
        self.fill_skipped_data_page()?;
        self.skip_next_page()
    }

    fn skip_next_page(&mut self) -> Result<()> {
        let codec = self.cache_codec();
        // Retain skipped dictionary pages so they can be decoded on demand.
        // See [`DeferredDictionaryPage`] for the state diagram.
        let deferred = match &mut self.state {
            SerializedPageReaderState::Values {
                offset,
                remaining_bytes,
                next_page_header,
                page_index,
                require_dictionary,
            } => {
                let mut deferred = None;
                let mut skipped_marker = false;
                if let Some(buffered_header) = next_page_header.take() {
                    verify_page_size(
                        buffered_header.compressed_page_size,
                        buffered_header.uncompressed_page_size,
                        *remaining_bytes,
                    )?;
                    // The next page header has already been peeked, so the
                    // offset already points at the page body.
                    let body_len = buffered_header.compressed_page_size as u64;
                    let peeked_start = self
                        .page_cache
                        .as_mut()
                        .and_then(|cache| cache.peeked_start.take());
                    if buffered_header.r#type == PageType::DICTIONARY_PAGE {
                        deferred = Some(DeferredDictionaryPage::Header {
                            page_start: peeked_start.unwrap_or(*offset),
                            data_start: *offset,
                            header: buffered_header,
                        });
                    }
                    *offset += body_len;
                    *remaining_bytes -= body_len;
                } else if let Some(cached) = match &self.page_cache {
                    Some(cache) => cache.resident(*offset, |cached| {
                        check_cached_page(cached, *offset, *remaining_bytes, codec)?;
                        if cached.is_dictionary_marker() && !cache.is_pinned(*offset) {
                            return Err(general_err!(
                                "Dictionary values must be read from the file"
                            ));
                        }
                        Ok(())
                    })?,
                    None => None,
                } {
                    // A resident page is stepped over by its stored length,
                    // without its file bytes. A dictionary page is kept for
                    // `take_deferred_dictionary`.
                    let stored_len = cached.stored_len();
                    if cached.is_dictionary_marker() {
                        skipped_marker = true;
                        *require_dictionary = false;
                    } else if cached.page().is_dictionary_page() {
                        deferred = Some(DeferredDictionaryPage::Cached {
                            offset: *offset,
                            page: cached.page().clone(),
                        });
                    }
                    *offset += stored_len;
                    *remaining_bytes -= stored_len;
                } else {
                    let mut read = self.reader.get_read(*offset)?;
                    let (header_len, header) = Self::read_page_header_len(
                        &self.context,
                        &mut read,
                        *page_index,
                        *require_dictionary,
                    )?;
                    verify_page_header_len(header_len, *remaining_bytes)?;
                    verify_page_size(
                        header.compressed_page_size,
                        header.uncompressed_page_size,
                        *remaining_bytes,
                    )?;
                    let data_page_size = header.compressed_page_size as u64;
                    let is_dict = header.r#type == PageType::DICTIONARY_PAGE;
                    if is_dict {
                        deferred = Some(DeferredDictionaryPage::Header {
                            page_start: *offset,
                            data_start: *offset + header_len as u64,
                            header: Box::new(header),
                        });
                    }
                    *offset += header_len as u64 + data_page_size;
                    *remaining_bytes -= header_len as u64 + data_page_size;
                }
                if deferred.is_some() {
                    *require_dictionary = false;
                } else if !skipped_marker {
                    *page_index += 1;
                }
                deferred
            }
            SerializedPageReaderState::Pages {
                page_locations,
                dictionary_page,
                page_index,
                ..
            } => {
                if dictionary_page.is_some() {
                    // Consume the dictionary page, keeping its location.
                    dictionary_page.take().map(DeferredDictionaryPage::Location)
                } else {
                    // If no dictionary page exists, simply pop the data page from page_locations
                    if page_locations.pop_front().is_some() {
                        *page_index += 1;
                    }
                    None
                }
            }
        };
        if deferred.is_some() {
            self.deferred_dictionary = deferred;
        }
        Ok(())
    }

    fn supports_deferred_dictionary(&self) -> bool {
        match &self.state {
            // Walking headers identifies dictionaries independently of metadata.
            SerializedPageReaderState::Values { .. } => true,
            // With an offset index, a dictionary is recoverable only when its
            // location can be inferred before the first data-page location, in
            // which case `dictionary_page` will be `Some`.
            SerializedPageReaderState::Pages {
                dictionary_page, ..
            } => dictionary_page.is_some(),
        }
    }

    fn take_deferred_dictionary(&mut self) -> Result<Option<Page>> {
        let Some(deferred) = self.deferred_dictionary.take() else {
            return Ok(None);
        };
        let codec = self.cache_codec();
        let page = match deferred {
            DeferredDictionaryPage::Decoded(page) => page,
            DeferredDictionaryPage::Header {
                page_start,
                data_start,
                header,
            } => {
                let data_len = header.compressed_page_size as usize;
                let stored_len = data_start - page_start + data_len as u64;
                let lookup = match &self.page_cache {
                    Some(cache) => cache.lookup(page_start, |cached| {
                        check_cached_page(cached, page_start, stored_len, codec)?;
                        if !cached.page().is_dictionary_page() || cached.is_dictionary_marker() {
                            return Err(general_err!(
                                "Cached page at offset {} has the wrong page type",
                                page_start
                            ));
                        }
                        // The header is known: the cached page must be the
                        // header and the body that it declares.
                        if cached.stored_len() != stored_len {
                            return Err(general_err!(
                                "Cached page at offset {} has {} bytes, but its header and body have {}",
                                page_start,
                                cached.stored_len(),
                                stored_len
                            ));
                        }
                        Ok(())
                    })?,
                    _ => Err(None),
                };
                let fill = match lookup {
                    Ok((cached, ())) => {
                        if let Some(chunk) = &mut self.plain_chunk {
                            chunk.observe_dictionary(cached.page());
                        }
                        return Ok(Some(cached.page().clone()));
                    }
                    Err(ticket) => ticket,
                };
                let buffer = self.reader.get_bytes(data_start, data_len)?;
                let buffer = self.context.decrypt_page_data(buffer, 0, true)?;
                let cacheable = fill.is_some() && page_body_is_compressed(&header);
                let page = decode_page(
                    *header,
                    buffer,
                    self.physical_type,
                    self.decompressor.as_mut(),
                )?;
                publish_page(
                    fill,
                    cacheable,
                    page,
                    stored_len,
                    codec,
                    self.plain_chunk.as_mut(),
                    false,
                )
            }
            DeferredDictionaryPage::Location(front) => {
                let lookup = match &self.page_cache {
                    Some(cache) => cache.lookup(front.offset as u64, |cached| {
                        if cached.is_dictionary_marker() {
                            return Err(general_err!(
                                "Leased data page still needs its dictionary"
                            ));
                        }
                        check_cached_location(cached, &front, true, codec)
                    })?,
                    _ => Err(None),
                };
                let fill = match lookup {
                    Ok((cached, ())) => {
                        if let Some(chunk) = &mut self.plain_chunk {
                            chunk.observe_dictionary(cached.page());
                        }
                        return Ok(Some(cached.page().clone()));
                    }
                    Err(ticket) => ticket,
                };
                let page_len = usize::try_from(front.compressed_page_size)?;
                let buffer = self.reader.get_bytes(front.offset as u64, page_len)?;
                let (offset, header) =
                    read_page_header_len_from_bytes(&self.context, buffer.as_ref(), 0, true)?;
                let bytes = buffer.slice(offset..);
                let bytes = self.context.decrypt_page_data(bytes, 0, true)?;
                let fill = fill.filter(|_| location_matches_header(offset, &header, page_len));
                let owns_buffer = fill.is_some() && page_body_is_compressed(&header);
                let page = decode_page(
                    header,
                    bytes,
                    self.physical_type,
                    self.decompressor.as_mut(),
                )?;
                publish_page(
                    fill,
                    owns_buffer,
                    page,
                    page_len as u64,
                    codec,
                    self.plain_chunk.as_mut(),
                    false,
                )
            }
            DeferredDictionaryPage::Cached { offset, page } => {
                // The skip stepped over this page without a read; taking it
                // is the read (of a leased page; a page of the cache without
                // a lease counted no read on the skip, as before leases).
                if let Some(cache) = &self.page_cache {
                    cache.mark_read(offset);
                }
                if let Some(chunk) = &mut self.plain_chunk {
                    chunk.observe_dictionary(&page);
                }
                page
            }
        };
        Ok(Some(page))
    }

    fn at_record_boundary(&mut self) -> Result<bool> {
        match &mut self.state {
            SerializedPageReaderState::Values { .. } => match self.peek_next_page()? {
                None => Ok(true),
                // V2 data pages must start at record boundaries per the parquet
                // spec, so the current page ends at one.
                Some(metadata) => Ok(metadata.num_rows.is_some()),
            },
            SerializedPageReaderState::Pages { .. } => Ok(true),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use bytes::Buf;

    use crate::file::page_index::column_index::{
        ByteArrayColumnIndex, ColumnIndexMetaData, PrimitiveColumnIndex,
    };
    use crate::file::properties::{EnabledStatistics, WriterProperties};

    use crate::basic::{self, BoundaryOrder, ColumnOrder, Compression, Encoding, SortOrder};
    use crate::column::reader::ColumnReader;
    use crate::data_type::private::ParquetValueType;
    use crate::data_type::{AsBytes, FixedLenByteArrayType, Int32Type};
    use crate::file::metadata::thrift::DataPageHeaderV2;
    use crate::file::writer::SerializedFileWriter;
    use crate::parquet_thrift::{ReadThrift, ThriftSliceInputProtocol};
    use crate::record::RowAccessor;
    use crate::schema::parser::parse_message_type;
    use crate::util::test_common::file_util::{get_test_file, get_test_path};

    use super::*;

    #[test]
    fn test_decode_page_invalid_offset() {
        let page_header = PageHeader {
            r#type: PageType::DATA_PAGE_V2,
            uncompressed_page_size: 10,
            compressed_page_size: 10,
            data_page_header: None,
            index_page_header: None,
            dictionary_page_header: None,
            crc: None,
            data_page_header_v2: Some(DataPageHeaderV2 {
                num_nulls: 0,
                num_rows: 0,
                num_values: 0,
                encoding: Encoding::PLAIN,
                definition_levels_byte_length: 11,
                repetition_levels_byte_length: 0,
                is_compressed: None,
                statistics: None,
            }),
        };

        let buffer = Bytes::new();
        let err = decode_page(page_header, buffer, Type::INT32, None).unwrap_err();
        assert!(
            err.to_string()
                .contains("DataPage v2 header contains implausible values")
        );
    }

    #[test]
    fn test_decode_unsupported_page() {
        let mut page_header = PageHeader {
            r#type: PageType::INDEX_PAGE,
            uncompressed_page_size: 10,
            compressed_page_size: 10,
            data_page_header: None,
            index_page_header: None,
            dictionary_page_header: None,
            crc: None,
            data_page_header_v2: None,
        };
        let buffer = Bytes::new();
        let err = decode_page(page_header.clone(), buffer.clone(), Type::INT32, None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Parquet error: Page type INDEX_PAGE is not supported"
        );

        page_header.data_page_header_v2 = Some(DataPageHeaderV2 {
            num_nulls: 0,
            num_rows: 0,
            num_values: 0,
            encoding: Encoding::PLAIN,
            definition_levels_byte_length: 11,
            repetition_levels_byte_length: 0,
            is_compressed: None,
            statistics: None,
        });
        let err = decode_page(page_header, buffer, Type::INT32, None).unwrap_err();
        assert!(
            err.to_string()
                .contains("DataPage v2 header contains implausible values")
        );
    }

    #[test]
    fn test_cursor_and_file_has_the_same_behaviour() {
        let mut buf: Vec<u8> = Vec::new();
        get_test_file("alltypes_plain.parquet")
            .read_to_end(&mut buf)
            .unwrap();
        let cursor = Bytes::from(buf);
        let read_from_cursor = SerializedFileReader::new(cursor).unwrap();

        let test_file = get_test_file("alltypes_plain.parquet");
        let read_from_file = SerializedFileReader::new(test_file).unwrap();

        let file_iter = read_from_file.get_row_iter(None).unwrap();
        let cursor_iter = read_from_cursor.get_row_iter(None).unwrap();

        for (a, b) in file_iter.zip(cursor_iter) {
            assert_eq!(a.unwrap(), b.unwrap())
        }
    }

    #[test]
    fn test_file_reader_try_from() {
        // Valid file path
        let test_file = get_test_file("alltypes_plain.parquet");
        let test_path_buf = get_test_path("alltypes_plain.parquet");
        let test_path = test_path_buf.as_path();
        let test_path_str = test_path.to_str().unwrap();

        let reader = SerializedFileReader::try_from(test_file);
        assert!(reader.is_ok());

        let reader = SerializedFileReader::try_from(test_path);
        assert!(reader.is_ok());

        let reader = SerializedFileReader::try_from(test_path_str);
        assert!(reader.is_ok());

        let reader = SerializedFileReader::try_from(test_path_str.to_string());
        assert!(reader.is_ok());

        // Invalid file path
        let test_path = Path::new("invalid.parquet");
        let test_path_str = test_path.to_str().unwrap();

        let reader = SerializedFileReader::try_from(test_path);
        assert!(reader.is_err());

        let reader = SerializedFileReader::try_from(test_path_str);
        assert!(reader.is_err());

        let reader = SerializedFileReader::try_from(test_path_str.to_string());
        assert!(reader.is_err());
    }

    #[test]
    fn test_file_reader_into_iter() {
        let path = get_test_path("alltypes_plain.parquet");
        let reader = SerializedFileReader::try_from(path.as_path()).unwrap();
        let iter = reader.into_iter();
        let values: Vec<_> = iter.flat_map(|x| x.unwrap().get_int(0)).collect();

        assert_eq!(values, &[4, 5, 6, 7, 2, 3, 0, 1]);
    }

    #[test]
    fn test_file_reader_into_iter_project() {
        let path = get_test_path("alltypes_plain.parquet");
        let reader = SerializedFileReader::try_from(path.as_path()).unwrap();
        let schema = "message schema { OPTIONAL INT32 id; }";
        let proj = parse_message_type(schema).ok();
        let iter = reader.into_iter().project(proj).unwrap();
        let values: Vec<_> = iter.flat_map(|x| x.unwrap().get_int(0)).collect();

        assert_eq!(values, &[4, 5, 6, 7, 2, 3, 0, 1]);
    }

    #[test]
    fn test_reuse_file_chunk() {
        // This test covers the case of maintaining the correct start position in a file
        // stream for each column reader after initializing and moving to the next one
        // (without necessarily reading the entire column).
        let test_file = get_test_file("alltypes_plain.parquet");
        let reader = SerializedFileReader::new(test_file).unwrap();
        let row_group = reader.get_row_group(0).unwrap();

        let mut page_readers = Vec::new();
        for i in 0..row_group.num_columns() {
            page_readers.push(row_group.get_column_page_reader(i).unwrap());
        }

        // Now buffer each col reader, we do not expect any failures like:
        // General("underlying Thrift error: end of file")
        for mut page_reader in page_readers {
            assert!(page_reader.get_next_page().is_ok());
        }
    }

    #[test]
    fn test_file_reader() {
        let test_file = get_test_file("alltypes_plain.parquet");
        let reader_result = SerializedFileReader::new(test_file);
        assert!(reader_result.is_ok());
        let reader = reader_result.unwrap();

        // Test contents in Parquet metadata
        let metadata = reader.metadata();
        assert_eq!(metadata.num_row_groups(), 1);

        // Test contents in file metadata
        let file_metadata = metadata.file_metadata();
        assert!(file_metadata.created_by().is_some());
        assert_eq!(
            file_metadata.created_by().unwrap(),
            "impala version 1.3.0-INTERNAL (build 8a48ddb1eff84592b3fc06bc6f51ec120e1fffc9)"
        );
        assert!(file_metadata.key_value_metadata().is_none());
        assert_eq!(file_metadata.num_rows(), 8);
        assert_eq!(file_metadata.version(), 1);
        assert_eq!(file_metadata.column_orders(), None);

        // Test contents in row group metadata
        let row_group_metadata = metadata.row_group(0);
        assert_eq!(row_group_metadata.num_columns(), 11);
        assert_eq!(row_group_metadata.num_rows(), 8);
        assert_eq!(row_group_metadata.total_byte_size(), 671);
        // Check each column order
        for i in 0..row_group_metadata.num_columns() {
            assert_eq!(file_metadata.column_order(i), ColumnOrder::UNDEFINED);
        }

        // Test row group reader
        let row_group_reader_result = reader.get_row_group(0);
        assert!(row_group_reader_result.is_ok());
        let row_group_reader: Box<dyn RowGroupReader> = row_group_reader_result.unwrap();
        assert_eq!(
            row_group_reader.num_columns(),
            row_group_metadata.num_columns()
        );
        assert_eq!(
            row_group_reader.metadata().total_byte_size(),
            row_group_metadata.total_byte_size()
        );

        // Test page readers
        // TODO: test for every column
        let page_reader_0_result = row_group_reader.get_column_page_reader(0);
        assert!(page_reader_0_result.is_ok());
        let mut page_reader_0: Box<dyn PageReader> = page_reader_0_result.unwrap();
        let mut page_count = 0;
        while let Some(page) = page_reader_0.get_next_page().unwrap() {
            let is_expected_page = match page {
                Page::DictionaryPage {
                    buf,
                    num_values,
                    encoding,
                    is_sorted,
                } => {
                    assert_eq!(buf.len(), 32);
                    assert_eq!(num_values, 8);
                    assert_eq!(encoding, Encoding::PLAIN_DICTIONARY);
                    assert!(!is_sorted);
                    true
                }
                Page::DataPage {
                    buf,
                    num_values,
                    encoding,
                    def_level_encoding,
                    rep_level_encoding,
                    statistics,
                } => {
                    assert_eq!(buf.len(), 11);
                    assert_eq!(num_values, 8);
                    assert_eq!(encoding, Encoding::PLAIN_DICTIONARY);
                    assert_eq!(def_level_encoding, Encoding::RLE);
                    #[expect(deprecated)]
                    let expected_rep_level_encoding = Encoding::BIT_PACKED;
                    assert_eq!(rep_level_encoding, expected_rep_level_encoding);
                    assert!(statistics.is_none());
                    true
                }
                Page::DataPageV2 { .. } => false,
            };
            assert!(is_expected_page);
            page_count += 1;
        }
        assert_eq!(page_count, 2);
    }

    #[test]
    fn test_file_reader_datapage_v2() {
        let test_file = get_test_file("datapage_v2.snappy.parquet");
        let reader_result = SerializedFileReader::new(test_file);
        assert!(reader_result.is_ok());
        let reader = reader_result.unwrap();

        // Test contents in Parquet metadata
        let metadata = reader.metadata();
        assert_eq!(metadata.num_row_groups(), 1);

        // Test contents in file metadata
        let file_metadata = metadata.file_metadata();
        assert!(file_metadata.created_by().is_some());
        assert_eq!(
            file_metadata.created_by().unwrap(),
            "parquet-mr version 1.8.1 (build 4aba4dae7bb0d4edbcf7923ae1339f28fd3f7fcf)"
        );
        assert!(file_metadata.key_value_metadata().is_some());
        assert_eq!(file_metadata.key_value_metadata().unwrap().len(), 1);

        assert_eq!(file_metadata.num_rows(), 5);
        assert_eq!(file_metadata.version(), 1);
        assert_eq!(file_metadata.column_orders(), None);

        let row_group_metadata = metadata.row_group(0);

        // Check each column order
        for i in 0..row_group_metadata.num_columns() {
            assert_eq!(file_metadata.column_order(i), ColumnOrder::UNDEFINED);
        }

        // Test row group reader
        let row_group_reader_result = reader.get_row_group(0);
        assert!(row_group_reader_result.is_ok());
        let row_group_reader: Box<dyn RowGroupReader> = row_group_reader_result.unwrap();
        assert_eq!(
            row_group_reader.num_columns(),
            row_group_metadata.num_columns()
        );
        assert_eq!(
            row_group_reader.metadata().total_byte_size(),
            row_group_metadata.total_byte_size()
        );

        // Test page readers
        // TODO: test for every column
        let page_reader_0_result = row_group_reader.get_column_page_reader(0);
        assert!(page_reader_0_result.is_ok());
        let mut page_reader_0: Box<dyn PageReader> = page_reader_0_result.unwrap();
        let mut page_count = 0;
        while let Some(page) = page_reader_0.get_next_page().unwrap() {
            let is_expected_page = match page {
                Page::DictionaryPage {
                    buf,
                    num_values,
                    encoding,
                    is_sorted,
                } => {
                    assert_eq!(buf.len(), 7);
                    assert_eq!(num_values, 1);
                    assert_eq!(encoding, Encoding::PLAIN);
                    assert!(!is_sorted);
                    true
                }
                Page::DataPageV2 {
                    buf,
                    num_values,
                    encoding,
                    num_nulls,
                    num_rows,
                    def_levels_byte_len,
                    rep_levels_byte_len,
                    is_compressed,
                    statistics,
                } => {
                    assert_eq!(buf.len(), 4);
                    assert_eq!(num_values, 5);
                    assert_eq!(encoding, Encoding::RLE_DICTIONARY);
                    assert_eq!(num_nulls, 1);
                    assert_eq!(num_rows, 5);
                    assert_eq!(def_levels_byte_len, 2);
                    assert_eq!(rep_levels_byte_len, 0);
                    assert!(is_compressed);
                    assert!(statistics.is_none()); // page stats are no longer read
                    true
                }
                Page::DataPage { .. } => false,
            };
            assert!(is_expected_page);
            page_count += 1;
        }
        assert_eq!(page_count, 2);
    }

    #[cfg_attr(miri, ignore)] // calls native Zstd code unsupported by Miri
    #[test]
    fn test_file_reader_empty_compressed_datapage_v2() {
        // this file has a compressed datapage that un-compresses to 0 bytes
        let test_file = get_test_file("page_v2_empty_compressed.parquet");
        let reader_result = SerializedFileReader::new(test_file);
        assert!(reader_result.is_ok());
        let reader = reader_result.unwrap();

        // Test contents in Parquet metadata
        let metadata = reader.metadata();
        assert_eq!(metadata.num_row_groups(), 1);

        // Test contents in file metadata
        let file_metadata = metadata.file_metadata();
        assert!(file_metadata.created_by().is_some());
        assert_eq!(
            file_metadata.created_by().unwrap(),
            "parquet-cpp-arrow version 14.0.2"
        );
        assert!(file_metadata.key_value_metadata().is_some());
        assert_eq!(file_metadata.key_value_metadata().unwrap().len(), 1);

        assert_eq!(file_metadata.num_rows(), 10);
        assert_eq!(file_metadata.version(), 2);
        let expected_order = ColumnOrder::TYPE_DEFINED_ORDER(SortOrder::SIGNED);
        assert_eq!(
            file_metadata.column_orders(),
            Some(vec![expected_order].as_ref())
        );

        let row_group_metadata = metadata.row_group(0);

        // Check each column order
        for i in 0..row_group_metadata.num_columns() {
            assert_eq!(file_metadata.column_order(i), expected_order);
        }

        // Test row group reader
        let row_group_reader_result = reader.get_row_group(0);
        assert!(row_group_reader_result.is_ok());
        let row_group_reader: Box<dyn RowGroupReader> = row_group_reader_result.unwrap();
        assert_eq!(
            row_group_reader.num_columns(),
            row_group_metadata.num_columns()
        );
        assert_eq!(
            row_group_reader.metadata().total_byte_size(),
            row_group_metadata.total_byte_size()
        );

        // Test page readers
        let page_reader_0_result = row_group_reader.get_column_page_reader(0);
        assert!(page_reader_0_result.is_ok());
        let mut page_reader_0: Box<dyn PageReader> = page_reader_0_result.unwrap();
        let mut page_count = 0;
        while let Some(page) = page_reader_0.get_next_page().unwrap() {
            let is_expected_page = match page {
                Page::DictionaryPage {
                    buf,
                    num_values,
                    encoding,
                    is_sorted,
                } => {
                    assert_eq!(buf.len(), 0);
                    assert_eq!(num_values, 0);
                    assert_eq!(encoding, Encoding::PLAIN);
                    assert!(!is_sorted);
                    true
                }
                Page::DataPageV2 {
                    buf,
                    num_values,
                    encoding,
                    num_nulls,
                    num_rows,
                    def_levels_byte_len,
                    rep_levels_byte_len,
                    is_compressed,
                    statistics,
                } => {
                    assert_eq!(buf.len(), 3);
                    assert_eq!(num_values, 10);
                    assert_eq!(encoding, Encoding::RLE_DICTIONARY);
                    assert_eq!(num_nulls, 10);
                    assert_eq!(num_rows, 10);
                    assert_eq!(def_levels_byte_len, 2);
                    assert_eq!(rep_levels_byte_len, 0);
                    assert!(is_compressed);
                    assert!(statistics.is_none()); // page stats are no longer read
                    true
                }
                Page::DataPage { .. } => false,
            };
            assert!(is_expected_page);
            page_count += 1;
        }
        assert_eq!(page_count, 2);
    }

    #[test]
    fn test_file_reader_empty_datapage_v2() {
        // this file has 0 bytes compressed datapage that un-compresses to 0 bytes
        let test_file = get_test_file("datapage_v2_empty_datapage.snappy.parquet");
        let reader_result = SerializedFileReader::new(test_file);
        assert!(reader_result.is_ok());
        let reader = reader_result.unwrap();

        // Test contents in Parquet metadata
        let metadata = reader.metadata();
        assert_eq!(metadata.num_row_groups(), 1);

        // Test contents in file metadata
        let file_metadata = metadata.file_metadata();
        assert!(file_metadata.created_by().is_some());
        assert_eq!(
            file_metadata.created_by().unwrap(),
            "parquet-mr version 1.13.1 (build db4183109d5b734ec5930d870cdae161e408ddba)"
        );
        assert!(file_metadata.key_value_metadata().is_some());
        assert_eq!(file_metadata.key_value_metadata().unwrap().len(), 2);

        assert_eq!(file_metadata.num_rows(), 1);
        assert_eq!(file_metadata.version(), 1);
        let expected_order = ColumnOrder::TYPE_DEFINED_ORDER(SortOrder::SIGNED);
        assert_eq!(
            file_metadata.column_orders(),
            Some(vec![expected_order].as_ref())
        );

        let row_group_metadata = metadata.row_group(0);

        // Check each column order
        for i in 0..row_group_metadata.num_columns() {
            assert_eq!(file_metadata.column_order(i), expected_order);
        }

        // Test row group reader
        let row_group_reader_result = reader.get_row_group(0);
        assert!(row_group_reader_result.is_ok());
        let row_group_reader: Box<dyn RowGroupReader> = row_group_reader_result.unwrap();
        assert_eq!(
            row_group_reader.num_columns(),
            row_group_metadata.num_columns()
        );
        assert_eq!(
            row_group_reader.metadata().total_byte_size(),
            row_group_metadata.total_byte_size()
        );

        // Test page readers
        let page_reader_0_result = row_group_reader.get_column_page_reader(0);
        assert!(page_reader_0_result.is_ok());
        let mut page_reader_0: Box<dyn PageReader> = page_reader_0_result.unwrap();
        let mut page_count = 0;
        while let Some(page) = page_reader_0.get_next_page().unwrap() {
            let is_expected_page = match page {
                Page::DataPageV2 {
                    buf,
                    num_values,
                    encoding,
                    num_nulls,
                    num_rows,
                    def_levels_byte_len,
                    rep_levels_byte_len,
                    is_compressed,
                    statistics,
                } => {
                    assert_eq!(buf.len(), 2);
                    assert_eq!(num_values, 1);
                    assert_eq!(encoding, Encoding::PLAIN);
                    assert_eq!(num_nulls, 1);
                    assert_eq!(num_rows, 1);
                    assert_eq!(def_levels_byte_len, 2);
                    assert_eq!(rep_levels_byte_len, 0);
                    assert!(is_compressed);
                    assert!(statistics.is_none());
                    true
                }
                _ => false,
            };
            assert!(is_expected_page);
            page_count += 1;
        }
        assert_eq!(page_count, 1);
    }

    fn get_serialized_page_reader<R: ChunkReader>(
        file_reader: &SerializedFileReader<R>,
        row_group_idx: usize,
        column: usize,
    ) -> Result<SerializedPageReader<R>> {
        let row_group = {
            let row_group_metadata = file_reader.metadata.row_group(row_group_idx);
            let props = Arc::clone(&file_reader.props);
            let f = Arc::clone(&file_reader.chunk_reader);
            let page_index = file_reader.metadata.page_index_for_row_group(row_group_idx);
            SerializedRowGroupReader::new(f, row_group_metadata, page_index, props)?
        };

        let col = row_group.metadata.column(column);
        let page_locations = file_reader
            .metadata
            .page_index()
            .map(|pi| pi.page_locations(row_group_idx, column))
            .unwrap_or(None)
            .cloned();

        let props = Arc::clone(&row_group.props);
        SerializedPageReader::new_with_properties(
            Arc::clone(&row_group.chunk_reader),
            col,
            usize::try_from(row_group.metadata.num_rows())?,
            page_locations,
            props,
        )
    }

    #[test]
    fn test_peek_next_page_offset_matches_actual() -> Result<()> {
        let test_file = get_test_file("alltypes_plain.parquet");
        let reader = SerializedFileReader::new(test_file)?;

        let mut offset_set = HashSet::new();
        let num_row_groups = reader.metadata.num_row_groups();
        for row_group in 0..num_row_groups {
            let num_columns = reader.metadata.row_group(row_group).num_columns();
            for column in 0..num_columns {
                let mut page_reader = get_serialized_page_reader(&reader, row_group, column)?;

                while let Ok(Some(page_offset)) = page_reader.peek_next_page_offset() {
                    match &page_reader.state {
                        SerializedPageReaderState::Pages {
                            page_locations,
                            dictionary_page,
                            ..
                        } => {
                            if let Some(page) = dictionary_page {
                                assert_eq!(page.offset as u64, page_offset);
                            } else if let Some(page) = page_locations.front() {
                                assert_eq!(page.offset as u64, page_offset);
                            } else {
                                unreachable!()
                            }
                        }
                        SerializedPageReaderState::Values {
                            offset,
                            next_page_header,
                            ..
                        } => {
                            assert!(next_page_header.is_some());
                            assert_eq!(*offset, page_offset);
                        }
                    }
                    let page = page_reader.get_next_page()?;
                    assert!(page.is_some());
                    let newly_inserted = offset_set.insert(page_offset);
                    assert!(newly_inserted);
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_page_iterator() {
        let file = get_test_file("alltypes_plain.parquet");
        let file_reader = Arc::new(SerializedFileReader::new(file).unwrap());

        let mut page_iterator = FilePageIterator::new(0, file_reader.clone()).unwrap();

        // read first page
        let page = page_iterator.next();
        assert!(page.is_some());
        assert!(page.unwrap().is_ok());

        // reach end of file
        let page = page_iterator.next();
        assert!(page.is_none());

        let row_group_indices = Box::new(0..1);
        let mut page_iterator =
            FilePageIterator::with_row_groups(0, row_group_indices, file_reader).unwrap();

        // read first page
        let page = page_iterator.next();
        assert!(page.is_some());
        assert!(page.unwrap().is_ok());

        // reach end of file
        let page = page_iterator.next();
        assert!(page.is_none());
    }

    #[test]
    fn test_file_reader_key_value_metadata() {
        let file = get_test_file("binary.parquet");
        let file_reader = Arc::new(SerializedFileReader::new(file).unwrap());

        let metadata = file_reader
            .metadata
            .file_metadata()
            .key_value_metadata()
            .unwrap();

        assert_eq!(metadata.len(), 3);

        assert_eq!(metadata[0].key, "parquet.proto.descriptor");

        assert_eq!(metadata[1].key, "writer.model.name");
        assert_eq!(metadata[1].value, Some("protobuf".to_owned()));

        assert_eq!(metadata[2].key, "parquet.proto.class");
        assert_eq!(metadata[2].value, Some("foo.baz.Foobaz$Event".to_owned()));
    }

    #[test]
    fn test_file_reader_optional_metadata() {
        // file with optional metadata: bloom filters, encoding stats, column index and offset index.
        let file = get_test_file("data_index_bloom_encoding_stats.parquet");
        let options = ReadOptionsBuilder::new()
            .with_encoding_stats_as_mask(false)
            .build();
        let file_reader = Arc::new(SerializedFileReader::new_with_options(file, options).unwrap());

        let row_group_metadata = file_reader.metadata.row_group(0);
        let col0_metadata = row_group_metadata.column(0);

        // test optional bloom filter offset
        assert_eq!(col0_metadata.bloom_filter_offset().unwrap(), 192);

        // test page encoding stats
        let page_encoding_stats = &col0_metadata.page_encoding_stats().unwrap()[0];

        assert_eq!(page_encoding_stats.page_type, basic::PageType::DATA_PAGE);
        assert_eq!(page_encoding_stats.encoding, Encoding::PLAIN);
        assert_eq!(page_encoding_stats.count, 1);

        // test optional column index offset
        assert_eq!(col0_metadata.column_index_offset().unwrap(), 156);
        assert_eq!(col0_metadata.column_index_length().unwrap(), 25);

        // test optional offset index offset
        assert_eq!(col0_metadata.offset_index_offset().unwrap(), 181);
        assert_eq!(col0_metadata.offset_index_length().unwrap(), 11);
    }

    #[test]
    fn test_file_reader_page_stats_mask() {
        let file = get_test_file("alltypes_tiny_pages.parquet");
        let options = ReadOptionsBuilder::new()
            .with_encoding_stats_as_mask(true)
            .build();
        let file_reader = Arc::new(SerializedFileReader::new_with_options(file, options).unwrap());

        let row_group_metadata = file_reader.metadata.row_group(0);

        // test page encoding stats
        let page_encoding_stats = row_group_metadata
            .column(0)
            .page_encoding_stats_mask()
            .unwrap();
        assert!(page_encoding_stats.is_only(Encoding::PLAIN));
        let page_encoding_stats = row_group_metadata
            .column(2)
            .page_encoding_stats_mask()
            .unwrap();
        assert!(page_encoding_stats.is_only(Encoding::PLAIN_DICTIONARY));
    }

    #[test]
    fn test_file_reader_page_stats_skipped() {
        let file = get_test_file("alltypes_tiny_pages.parquet");

        // test skipping all
        let options = ReadOptionsBuilder::new()
            .with_encoding_stats_policy(ParquetStatisticsPolicy::SkipAll)
            .with_column_stats_policy(ParquetStatisticsPolicy::SkipAll)
            .build();
        let file_reader = Arc::new(
            SerializedFileReader::new_with_options(file.try_clone().unwrap(), options).unwrap(),
        );

        let row_group_metadata = file_reader.metadata.row_group(0);
        for column in row_group_metadata.columns() {
            assert!(column.page_encoding_stats().is_none());
            assert!(column.page_encoding_stats_mask().is_none());
            assert!(column.statistics().is_none());
        }

        // test skipping all but one column
        let options = ReadOptionsBuilder::new()
            .with_encoding_stats_as_mask(true)
            .with_encoding_stats_policy(ParquetStatisticsPolicy::skip_except(&[0]))
            .with_column_stats_policy(ParquetStatisticsPolicy::skip_except(&[0]))
            .build();
        let file_reader = Arc::new(
            SerializedFileReader::new_with_options(file.try_clone().unwrap(), options).unwrap(),
        );

        let row_group_metadata = file_reader.metadata.row_group(0);
        for (idx, column) in row_group_metadata.columns().iter().enumerate() {
            assert!(column.page_encoding_stats().is_none());
            assert_eq!(column.page_encoding_stats_mask().is_some(), idx == 0);
            assert_eq!(column.statistics().is_some(), idx == 0);
        }
    }

    #[test]
    fn test_file_reader_size_stats_skipped() {
        let file = get_test_file("repeated_primitive_no_list.parquet");

        // test skipping all
        let options = ReadOptionsBuilder::new()
            .with_size_stats_policy(ParquetStatisticsPolicy::SkipAll)
            .build();
        let file_reader = Arc::new(
            SerializedFileReader::new_with_options(file.try_clone().unwrap(), options).unwrap(),
        );

        let row_group_metadata = file_reader.metadata.row_group(0);
        for column in row_group_metadata.columns() {
            assert!(column.repetition_level_histogram().is_none());
            assert!(column.definition_level_histogram().is_none());
            assert!(column.unencoded_byte_array_data_bytes().is_none());
        }

        // test skipping all but one column
        let options = ReadOptionsBuilder::new()
            .with_encoding_stats_as_mask(true)
            .with_size_stats_policy(ParquetStatisticsPolicy::skip_except(&[1]))
            .build();
        let file_reader = Arc::new(
            SerializedFileReader::new_with_options(file.try_clone().unwrap(), options).unwrap(),
        );

        let row_group_metadata = file_reader.metadata.row_group(0);
        for (idx, column) in row_group_metadata.columns().iter().enumerate() {
            assert_eq!(column.repetition_level_histogram().is_some(), idx == 1);
            assert_eq!(column.definition_level_histogram().is_some(), idx == 1);
            assert_eq!(column.unencoded_byte_array_data_bytes().is_some(), idx == 1);
        }
    }

    #[test]
    fn test_file_reader_with_no_filter() -> Result<()> {
        let test_file = get_test_file("alltypes_plain.parquet");
        let origin_reader = SerializedFileReader::new(test_file)?;
        // test initial number of row groups
        let metadata = origin_reader.metadata();
        assert_eq!(metadata.num_row_groups(), 1);
        Ok(())
    }

    #[test]
    fn test_file_reader_filter_row_groups_with_predicate() -> Result<()> {
        let test_file = get_test_file("alltypes_plain.parquet");
        let read_options = ReadOptionsBuilder::new()
            .with_predicate(Box::new(|_, _| false))
            .build();
        let reader = SerializedFileReader::new_with_options(test_file, read_options)?;
        let metadata = reader.metadata();
        assert_eq!(metadata.num_row_groups(), 0);
        Ok(())
    }

    #[test]
    fn test_file_reader_filter_row_groups_with_range() -> Result<()> {
        let test_file = get_test_file("alltypes_plain.parquet");
        let origin_reader = SerializedFileReader::new(test_file)?;
        // test initial number of row groups
        let metadata = origin_reader.metadata();
        assert_eq!(metadata.num_row_groups(), 1);
        let mid = get_midpoint_offset(metadata.row_group(0));

        let test_file = get_test_file("alltypes_plain.parquet");
        let read_options = ReadOptionsBuilder::new().with_range(0, mid + 1).build();
        let reader = SerializedFileReader::new_with_options(test_file, read_options)?;
        let metadata = reader.metadata();
        assert_eq!(metadata.num_row_groups(), 1);

        let test_file = get_test_file("alltypes_plain.parquet");
        let read_options = ReadOptionsBuilder::new().with_range(0, mid).build();
        let reader = SerializedFileReader::new_with_options(test_file, read_options)?;
        let metadata = reader.metadata();
        assert_eq!(metadata.num_row_groups(), 0);
        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Takes too long
    fn test_file_reader_filter_row_groups_and_range() -> Result<()> {
        let test_file = get_test_file("alltypes_tiny_pages.parquet");
        let origin_reader = SerializedFileReader::new(test_file)?;
        let metadata = origin_reader.metadata();
        let mid = get_midpoint_offset(metadata.row_group(0));

        // true, true predicate
        let test_file = get_test_file("alltypes_tiny_pages.parquet");
        let read_options = ReadOptionsBuilder::new()
            .with_page_index()
            .with_predicate(Box::new(|_, _| true))
            .with_range(mid, mid + 1)
            .build();
        let reader = SerializedFileReader::new_with_options(test_file, read_options)?;
        let metadata = reader.metadata();
        assert_eq!(metadata.num_row_groups(), 1);
        assert!(metadata.page_index().is_some());

        // true, false predicate
        let test_file = get_test_file("alltypes_tiny_pages.parquet");
        let read_options = ReadOptionsBuilder::new()
            .with_page_index()
            .with_predicate(Box::new(|_, _| true))
            .with_range(0, mid)
            .build();
        let reader = SerializedFileReader::new_with_options(test_file, read_options)?;
        let metadata = reader.metadata();
        assert_eq!(metadata.num_row_groups(), 0);
        assert!(metadata.page_index().is_none());

        // false, true predicate
        let test_file = get_test_file("alltypes_tiny_pages.parquet");
        let read_options = ReadOptionsBuilder::new()
            .with_page_index()
            .with_predicate(Box::new(|_, _| false))
            .with_range(mid, mid + 1)
            .build();
        let reader = SerializedFileReader::new_with_options(test_file, read_options)?;
        let metadata = reader.metadata();
        assert_eq!(metadata.num_row_groups(), 0);
        assert!(metadata.page_index().is_none());

        // false, false predicate
        let test_file = get_test_file("alltypes_tiny_pages.parquet");
        let read_options = ReadOptionsBuilder::new()
            .with_page_index()
            .with_predicate(Box::new(|_, _| false))
            .with_range(0, mid)
            .build();
        let reader = SerializedFileReader::new_with_options(test_file, read_options)?;
        let metadata = reader.metadata();
        assert_eq!(metadata.num_row_groups(), 0);
        assert!(metadata.page_index().is_none());
        Ok(())
    }

    #[test]
    fn test_file_reader_invalid_metadata() {
        let data = [
            255, 172, 1, 0, 50, 82, 65, 73, 1, 0, 0, 0, 169, 168, 168, 162, 87, 255, 16, 0, 0, 0,
            80, 65, 82, 49,
        ];
        let ret = SerializedFileReader::new(Bytes::copy_from_slice(&data));
        assert_eq!(
            ret.err().unwrap().to_string(),
            "Parquet error: Expected list element type of Struct but got List"
        );
    }

    #[test]
    // Use java parquet-tools get below pageIndex info
    // !```
    // parquet-tools column-index ./data_index_bloom_encoding_stats.parquet
    // row group 0:
    // column index for column String:
    // Boundary order: ASCENDING
    // page-0  :
    // null count                 min                                  max
    // 0                          Hello                                today
    //
    // offset index for column String:
    // page-0   :
    // offset   compressed size       first row index
    // 4               152                     0
    ///```
    //
    fn test_page_index_reader() {
        let test_file = get_test_file("data_index_bloom_encoding_stats.parquet");
        let builder = ReadOptionsBuilder::new();
        //enable read page index
        let options = builder.with_page_index().build();
        let reader_result = SerializedFileReader::new_with_options(test_file, options);
        let reader = reader_result.unwrap();

        // Test contents in Parquet metadata
        let metadata = reader.metadata();
        assert_eq!(metadata.num_row_groups(), 1);

        let page_index = metadata.page_index().expect("page index should be present");

        // only one row group
        let Some(ColumnIndexMetaData::BYTE_ARRAY(index)) = page_index.column_index(0, 0) else {
            unreachable!()
        };

        assert_eq!(index.boundary_order, BoundaryOrder::ASCENDING);

        //only one page group
        assert_eq!(index.num_pages(), 1);

        let min = index.min_value(0).unwrap();
        let max = index.max_value(0).unwrap();
        assert_eq!(b"Hello", min.as_bytes());
        assert_eq!(b"today", max.as_bytes());

        // only one row group
        let offset_index = page_index
            .offset_index(0, 0)
            .expect("offset index should be present");
        let page_offset = offset_index
            .page_locations()
            .first()
            .expect("offset index too small");

        assert_eq!(4, page_offset.offset);
        assert_eq!(152, page_offset.compressed_page_size);
        assert_eq!(0, page_offset.first_row_index);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Takes too long
    fn test_page_index_reader_all_type() {
        let test_file = get_test_file("alltypes_tiny_pages_plain.parquet");
        let builder = ReadOptionsBuilder::new();
        //enable read page index
        let options = builder.with_page_index().build();
        let reader_result = SerializedFileReader::new_with_options(test_file, options);
        let reader = reader_result.unwrap();

        // Test contents in Parquet metadata
        let metadata = reader.metadata();
        assert_eq!(metadata.num_row_groups(), 1);

        let page_index = metadata.page_index().unwrap();
        let row_group_offset_indexes = metadata.page_index_for_row_group(0);

        // only one row group
        let row_group_metadata = metadata.row_group(0);

        //col0->id: INT32 UNCOMPRESSED DO:0 FPO:4 SZ:37325/37325/1.00 VC:7300 ENC:BIT_PACKED,RLE,PLAIN ST:[min: 0, max: 7299, num_nulls: 0]
        let ci = page_index.column_index(0, 0).unwrap();
        assert!(!ci.is_sorted());
        assert!(matches!(
            ci.get_boundary_order(),
            Some(BoundaryOrder::UNORDERED)
        ));
        if let ColumnIndexMetaData::INT32(index) = ci {
            check_native_page_index(
                index,
                325,
                get_row_group_min_max_bytes(row_group_metadata, 0),
                BoundaryOrder::UNORDERED,
            );
            assert_eq!(
                row_group_offset_indexes.page_locations(0).unwrap().len(),
                325
            );
        } else {
            unreachable!()
        }
        //col1->bool_col:BOOLEAN UNCOMPRESSED DO:0 FPO:37329 SZ:3022/3022/1.00 VC:7300 ENC:BIT_PACKED,RLE,PLAIN ST:[min: false, max: true, num_nulls: 0]
        let ci = page_index.column_index(0, 1).unwrap();
        assert!(ci.is_sorted());
        if let ColumnIndexMetaData::BOOLEAN(index) = ci {
            assert_eq!(index.num_pages(), 82);
            assert_eq!(
                row_group_offset_indexes.page_locations(1).unwrap().len(),
                82
            );
        } else {
            unreachable!()
        }
        //col2->tinyint_col: INT32 UNCOMPRESSED DO:0 FPO:40351 SZ:37325/37325/1.00 VC:7300 ENC:BIT_PACKED,RLE,PLAIN ST:[min: 0, max: 9, num_nulls: 0]
        let ci = page_index.column_index(0, 2).unwrap();
        assert!(ci.is_sorted());
        if let ColumnIndexMetaData::INT32(index) = ci {
            check_native_page_index(
                index,
                325,
                get_row_group_min_max_bytes(row_group_metadata, 2),
                BoundaryOrder::ASCENDING,
            );
            assert_eq!(
                row_group_offset_indexes.page_locations(2).unwrap().len(),
                325
            );
        } else {
            unreachable!()
        }
        //col4->smallint_col: INT32 UNCOMPRESSED DO:0 FPO:77676 SZ:37325/37325/1.00 VC:7300 ENC:BIT_PACKED,RLE,PLAIN ST:[min: 0, max: 9, num_nulls: 0]
        let ci = page_index.column_index(0, 3).unwrap();
        assert!(ci.is_sorted());
        if let ColumnIndexMetaData::INT32(index) = ci {
            check_native_page_index(
                index,
                325,
                get_row_group_min_max_bytes(row_group_metadata, 3),
                BoundaryOrder::ASCENDING,
            );
            assert_eq!(
                row_group_offset_indexes.page_locations(3).unwrap().len(),
                325
            );
        } else {
            unreachable!()
        }
        //col5->smallint_col: INT32 UNCOMPRESSED DO:0 FPO:77676 SZ:37325/37325/1.00 VC:7300 ENC:BIT_PACKED,RLE,PLAIN ST:[min: 0, max: 9, num_nulls: 0]
        let ci = page_index.column_index(0, 4).unwrap();
        assert!(ci.is_sorted());
        if let ColumnIndexMetaData::INT32(index) = ci {
            check_native_page_index(
                index,
                325,
                get_row_group_min_max_bytes(row_group_metadata, 4),
                BoundaryOrder::ASCENDING,
            );
            assert_eq!(
                row_group_offset_indexes.page_locations(4).unwrap().len(),
                325
            );
        } else {
            unreachable!()
        }
        //col6->bigint_col: INT64 UNCOMPRESSED DO:0 FPO:152326 SZ:71598/71598/1.00 VC:7300 ENC:BIT_PACKED,RLE,PLAIN ST:[min: 0, max: 90, num_nulls: 0]
        let ci = page_index.column_index(0, 5).unwrap();
        assert!(!ci.is_sorted());
        if let ColumnIndexMetaData::INT64(index) = ci {
            check_native_page_index(
                index,
                528,
                get_row_group_min_max_bytes(row_group_metadata, 5),
                BoundaryOrder::UNORDERED,
            );
            assert_eq!(
                row_group_offset_indexes.page_locations(5).unwrap().len(),
                528
            );
        } else {
            unreachable!()
        }
        //col7->float_col: FLOAT UNCOMPRESSED DO:0 FPO:223924 SZ:37325/37325/1.00 VC:7300 ENC:BIT_PACKED,RLE,PLAIN ST:[min: -0.0, max: 9.9, num_nulls: 0]
        let ci = page_index.column_index(0, 6).unwrap();
        assert!(ci.is_sorted());
        if let ColumnIndexMetaData::FLOAT(index) = ci {
            check_native_page_index(
                index,
                325,
                get_row_group_min_max_bytes(row_group_metadata, 6),
                BoundaryOrder::ASCENDING,
            );
            assert_eq!(
                row_group_offset_indexes.page_locations(6).unwrap().len(),
                325
            );
        } else {
            unreachable!()
        }
        //col8->double_col: DOUBLE UNCOMPRESSED DO:0 FPO:261249 SZ:71598/71598/1.00 VC:7300 ENC:BIT_PACKED,RLE,PLAIN ST:[min: -0.0, max: 90.89999999999999, num_nulls: 0]
        let ci = page_index.column_index(0, 7).unwrap();
        assert!(!ci.is_sorted());
        if let ColumnIndexMetaData::DOUBLE(index) = ci {
            check_native_page_index(
                index,
                528,
                get_row_group_min_max_bytes(row_group_metadata, 7),
                BoundaryOrder::UNORDERED,
            );
            assert_eq!(
                row_group_offset_indexes.page_locations(7).unwrap().len(),
                528
            );
        } else {
            unreachable!()
        }
        //col9->date_string_col: BINARY UNCOMPRESSED DO:0 FPO:332847 SZ:111948/111948/1.00 VC:7300 ENC:BIT_PACKED,RLE,PLAIN ST:[min: 01/01/09, max: 12/31/10, num_nulls: 0]
        let ci = page_index.column_index(0, 8).unwrap();
        assert!(!ci.is_sorted());
        if let ColumnIndexMetaData::BYTE_ARRAY(index) = ci {
            check_byte_array_page_index(
                index,
                974,
                get_row_group_min_max_bytes(row_group_metadata, 8),
                BoundaryOrder::UNORDERED,
            );
            assert_eq!(
                row_group_offset_indexes.page_locations(8).unwrap().len(),
                974
            );
        } else {
            unreachable!()
        }
        //col10->string_col: BINARY UNCOMPRESSED DO:0 FPO:444795 SZ:45298/45298/1.00 VC:7300 ENC:BIT_PACKED,RLE,PLAIN ST:[min: 0, max: 9, num_nulls: 0]
        let ci = page_index.column_index(0, 9).unwrap();
        assert!(ci.is_sorted());
        if let ColumnIndexMetaData::BYTE_ARRAY(index) = ci {
            check_byte_array_page_index(
                index,
                352,
                get_row_group_min_max_bytes(row_group_metadata, 9),
                BoundaryOrder::ASCENDING,
            );
            assert_eq!(
                row_group_offset_indexes.page_locations(9).unwrap().len(),
                352
            );
        } else {
            unreachable!()
        }
        //col11->timestamp_col: INT96 UNCOMPRESSED DO:0 FPO:490093 SZ:111948/111948/1.00 VC:7300 ENC:BIT_PACKED,RLE,PLAIN ST:[num_nulls: 0, min/max not defined]
        // this columns lacks an index
        assert!(page_index.column_index(0, 10).is_none());
        //col12->year: INT32 UNCOMPRESSED DO:0 FPO:602041 SZ:37325/37325/1.00 VC:7300 ENC:BIT_PACKED,RLE,PLAIN ST:[min: 2009, max: 2010, num_nulls: 0]
        let ci = page_index.column_index(0, 11).unwrap();
        assert!(ci.is_sorted());
        if let ColumnIndexMetaData::INT32(index) = ci {
            check_native_page_index(
                index,
                325,
                get_row_group_min_max_bytes(row_group_metadata, 11),
                BoundaryOrder::ASCENDING,
            );
            assert_eq!(
                row_group_offset_indexes.page_locations(11).unwrap().len(),
                325
            );
        } else {
            unreachable!()
        }
        //col13->month: INT32 UNCOMPRESSED DO:0 FPO:639366 SZ:37325/37325/1.00 VC:7300 ENC:BIT_PACKED,RLE,PLAIN ST:[min: 1, max: 12, num_nulls: 0]
        let ci = page_index.column_index(0, 12).unwrap();
        assert!(!ci.is_sorted());
        if let ColumnIndexMetaData::INT32(index) = ci {
            check_native_page_index(
                index,
                325,
                get_row_group_min_max_bytes(row_group_metadata, 12),
                BoundaryOrder::UNORDERED,
            );
            assert_eq!(
                row_group_offset_indexes.page_locations(12).unwrap().len(),
                325
            );
        } else {
            unreachable!()
        }
    }

    fn check_native_page_index<T: ParquetValueType>(
        row_group_index: &PrimitiveColumnIndex<T>,
        page_size: usize,
        min_max: (&[u8], &[u8]),
        boundary_order: BoundaryOrder,
    ) {
        assert_eq!(row_group_index.num_pages() as usize, page_size);
        assert_eq!(row_group_index.boundary_order, boundary_order);
        assert!(row_group_index.min_values().iter().all(|x| {
            x >= &T::try_from_le_slice(min_max.0).unwrap()
                && x <= &T::try_from_le_slice(min_max.1).unwrap()
        }));
    }

    fn check_byte_array_page_index(
        row_group_index: &ByteArrayColumnIndex,
        page_size: usize,
        min_max: (&[u8], &[u8]),
        boundary_order: BoundaryOrder,
    ) {
        assert_eq!(row_group_index.num_pages() as usize, page_size);
        assert_eq!(row_group_index.boundary_order, boundary_order);
        for i in 0..row_group_index.num_pages() as usize {
            let x = row_group_index.min_value(i).unwrap();
            assert!(x >= min_max.0 && x <= min_max.1);
        }
    }

    fn get_row_group_min_max_bytes(r: &RowGroupMetaData, col_num: usize) -> (&[u8], &[u8]) {
        let statistics = r.column(col_num).statistics().unwrap();
        (
            statistics.min_bytes_opt().unwrap_or_default(),
            statistics.max_bytes_opt().unwrap_or_default(),
        )
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Takes too long
    fn test_skip_next_page_with_dictionary_page() {
        let test_file = get_test_file("alltypes_tiny_pages.parquet");
        let builder = ReadOptionsBuilder::new();
        // enable read page index
        let options = builder.with_page_index().build();
        let reader_result = SerializedFileReader::new_with_options(test_file, options);
        let reader = reader_result.unwrap();

        let row_group_reader = reader.get_row_group(0).unwrap();

        // use 'string_col', Boundary order: UNORDERED, total 352 data pages and 1 dictionary page.
        let mut column_page_reader = row_group_reader.get_column_page_reader(9).unwrap();

        let mut vec = vec![];

        // Step 1: Peek and ensure dictionary page is correctly identified
        let meta = column_page_reader.peek_next_page().unwrap().unwrap();
        assert!(meta.is_dict);

        // Step 2: Call skip_next_page to skip the dictionary page
        column_page_reader.skip_next_page().unwrap();

        // Step 3: Read the next data page after skipping the dictionary page
        let page = column_page_reader.get_next_page().unwrap().unwrap();
        assert!(matches!(page.page_type(), basic::PageType::DATA_PAGE));

        // Step 4: Continue reading remaining data pages and verify correctness
        for _i in 0..351 {
            // 352 total pages, 1 dictionary page is skipped
            let meta = column_page_reader.peek_next_page().unwrap().unwrap();
            assert!(!meta.is_dict); // Verify no dictionary page here
            vec.push(meta);

            let page = column_page_reader.get_next_page().unwrap().unwrap();
            assert!(matches!(page.page_type(), basic::PageType::DATA_PAGE));
        }

        // Step 5: Check if all pages are read
        assert!(column_page_reader.peek_next_page().unwrap().is_none());
        assert!(column_page_reader.get_next_page().unwrap().is_none());

        // Step 6: Verify the number of data pages read (should be 351 data pages)
        assert_eq!(vec.len(), 351);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Takes too long
    fn test_skip_page_with_offset_index() {
        let test_file = get_test_file("alltypes_tiny_pages_plain.parquet");
        let builder = ReadOptionsBuilder::new();
        //enable read page index
        let options = builder.with_page_index().build();
        let reader_result = SerializedFileReader::new_with_options(test_file, options);
        let reader = reader_result.unwrap();

        let row_group_reader = reader.get_row_group(0).unwrap();

        //use 'int_col', Boundary order: ASCENDING, total 325 pages.
        let mut column_page_reader = row_group_reader.get_column_page_reader(4).unwrap();

        let mut vec = vec![];

        for i in 0..325 {
            if i % 2 == 0 {
                vec.push(column_page_reader.get_next_page().unwrap().unwrap());
            } else {
                column_page_reader.skip_next_page().unwrap();
            }
        }
        //check read all pages.
        assert!(column_page_reader.peek_next_page().unwrap().is_none());
        assert!(column_page_reader.get_next_page().unwrap().is_none());

        assert_eq!(vec.len(), 163);
    }

    #[test]
    fn test_skip_page_without_offset_index() {
        let test_file = get_test_file("alltypes_tiny_pages_plain.parquet");

        // use default SerializedFileReader without read offsetIndex
        let reader_result = SerializedFileReader::new(test_file);
        let reader = reader_result.unwrap();

        let row_group_reader = reader.get_row_group(0).unwrap();

        //use 'int_col', Boundary order: ASCENDING, total 325 pages.
        let mut column_page_reader = row_group_reader.get_column_page_reader(4).unwrap();

        let mut vec = vec![];

        for i in 0..325 {
            if i % 2 == 0 {
                vec.push(column_page_reader.get_next_page().unwrap().unwrap());
            } else {
                column_page_reader.peek_next_page().unwrap().unwrap();
                column_page_reader.skip_next_page().unwrap();
            }
        }
        //check read all pages.
        assert!(column_page_reader.peek_next_page().unwrap().is_none());
        assert!(column_page_reader.get_next_page().unwrap().is_none());

        assert_eq!(vec.len(), 163);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Takes too long
    fn test_peek_page_with_dictionary_page() {
        let test_file = get_test_file("alltypes_tiny_pages.parquet");
        let builder = ReadOptionsBuilder::new();
        //enable read page index
        let options = builder.with_page_index().build();
        let reader_result = SerializedFileReader::new_with_options(test_file, options);
        let reader = reader_result.unwrap();
        let row_group_reader = reader.get_row_group(0).unwrap();

        //use 'string_col', Boundary order: UNORDERED, total 352 data ages and 1 dictionary page.
        let mut column_page_reader = row_group_reader.get_column_page_reader(9).unwrap();

        let mut vec = vec![];

        let meta = column_page_reader.peek_next_page().unwrap().unwrap();
        assert!(meta.is_dict);
        let page = column_page_reader.get_next_page().unwrap().unwrap();
        assert!(matches!(page.page_type(), basic::PageType::DICTIONARY_PAGE));

        for i in 0..352 {
            let meta = column_page_reader.peek_next_page().unwrap().unwrap();
            // have checked with `parquet-tools column-index   -c string_col  ./alltypes_tiny_pages.parquet`
            // page meta has two scenarios(21, 20) of num_rows expect last page has 11 rows.
            if i != 351 {
                assert!((meta.num_rows == Some(21)) || (meta.num_rows == Some(20)));
            } else {
                // last page first row index is 7290, total row count is 7300
                // because first row start with zero, last page row count should be 10.
                assert_eq!(meta.num_rows, Some(10));
            }
            assert!(!meta.is_dict);
            vec.push(meta);
            let page = column_page_reader.get_next_page().unwrap().unwrap();
            assert!(matches!(page.page_type(), basic::PageType::DATA_PAGE));
        }

        //check read all pages.
        assert!(column_page_reader.peek_next_page().unwrap().is_none());
        assert!(column_page_reader.get_next_page().unwrap().is_none());

        assert_eq!(vec.len(), 352);
    }

    #[test]
    fn test_peek_page_with_dictionary_page_without_offset_index() {
        let test_file = get_test_file("alltypes_tiny_pages.parquet");

        let reader_result = SerializedFileReader::new(test_file);
        let reader = reader_result.unwrap();
        let row_group_reader = reader.get_row_group(0).unwrap();

        //use 'string_col', Boundary order: UNORDERED, total 352 data ages and 1 dictionary page.
        let mut column_page_reader = row_group_reader.get_column_page_reader(9).unwrap();

        let mut vec = vec![];

        let meta = column_page_reader.peek_next_page().unwrap().unwrap();
        assert!(meta.is_dict);
        let page = column_page_reader.get_next_page().unwrap().unwrap();
        assert!(matches!(page.page_type(), basic::PageType::DICTIONARY_PAGE));

        for i in 0..352 {
            let meta = column_page_reader.peek_next_page().unwrap().unwrap();
            // have checked with `parquet-tools column-index   -c string_col  ./alltypes_tiny_pages.parquet`
            // page meta has two scenarios(21, 20) of num_rows expect last page has 11 rows.
            if i != 351 {
                assert!((meta.num_levels == Some(21)) || (meta.num_levels == Some(20)));
            } else {
                // last page first row index is 7290, total row count is 7300
                // because first row start with zero, last page row count should be 10.
                assert_eq!(meta.num_levels, Some(10));
            }
            assert!(!meta.is_dict);
            vec.push(meta);
            let page = column_page_reader.get_next_page().unwrap().unwrap();
            assert!(matches!(page.page_type(), basic::PageType::DATA_PAGE));
        }

        //check read all pages.
        assert!(column_page_reader.peek_next_page().unwrap().is_none());
        assert!(column_page_reader.get_next_page().unwrap().is_none());

        assert_eq!(vec.len(), 352);
    }

    #[test]
    fn test_fixed_length_index() {
        let message_type = "
        message test_schema {
          OPTIONAL FIXED_LEN_BYTE_ARRAY (11) value (DECIMAL(25,2));
        }
        ";

        let schema = parse_message_type(message_type).unwrap();
        let mut out = Vec::with_capacity(1024);
        let mut writer =
            SerializedFileWriter::new(&mut out, Arc::new(schema), Default::default()).unwrap();

        let mut r = writer.next_row_group().unwrap();
        let mut c = r.next_column().unwrap().unwrap();
        c.typed::<FixedLenByteArrayType>()
            .write_batch(
                &[vec![0; 11].into(), vec![5; 11].into(), vec![3; 11].into()],
                Some(&[1, 1, 0, 1]),
                None,
            )
            .unwrap();
        c.close().unwrap();
        r.close().unwrap();
        writer.close().unwrap();

        let b = Bytes::from(out);
        let options = ReadOptionsBuilder::new().with_page_index().build();
        let reader = SerializedFileReader::new_with_options(b, options).unwrap();
        let page_index = reader.metadata().page_index().unwrap();

        match page_index.column_index(0, 0) {
            Some(ColumnIndexMetaData::FIXED_LEN_BYTE_ARRAY(v)) => {
                assert_eq!(v.num_pages(), 1);
                assert_eq!(v.null_count(0).unwrap(), 1);
                assert_eq!(v.min_value(0).unwrap(), &[0; 11]);
                assert_eq!(v.max_value(0).unwrap(), &[5; 11]);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_multi_gz() {
        let file = get_test_file("concatenated_gzip_members.parquet");
        let reader = SerializedFileReader::new(file).unwrap();
        let row_group_reader = reader.get_row_group(0).unwrap();
        match row_group_reader.get_column_reader(0).unwrap() {
            ColumnReader::Int64ColumnReader(mut reader) => {
                let mut buffer = Vec::with_capacity(1024);
                let mut def_levels = Vec::with_capacity(1024);
                let (num_records, num_values, num_levels) = reader
                    .read_records(1024, Some(&mut def_levels), None, &mut buffer)
                    .unwrap();

                assert_eq!(num_records, 513);
                assert_eq!(num_values, 513);
                assert_eq!(num_levels, 513);

                let expected: Vec<i64> = (1..514).collect();
                assert_eq!(&buffer, &expected);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Takes too long
    fn test_byte_stream_split_extended() {
        let path = format!(
            "{}/byte_stream_split_extended.gzip.parquet",
            arrow::util::test_util::parquet_test_data(),
        );
        let file = File::open(path).unwrap();
        let reader = Box::new(SerializedFileReader::new(file).expect("Failed to create reader"));

        // Use full schema as projected schema
        let mut iter = reader
            .get_row_iter(None)
            .expect("Failed to create row iterator");

        let mut start = 0;
        let end = reader.metadata().file_metadata().num_rows();

        let check_row = |row: Result<Row, ParquetError>| {
            assert!(row.is_ok());
            let r = row.unwrap();
            assert_eq!(r.get_float16(0).unwrap(), r.get_float16(1).unwrap());
            assert_eq!(r.get_float(2).unwrap(), r.get_float(3).unwrap());
            assert_eq!(r.get_double(4).unwrap(), r.get_double(5).unwrap());
            assert_eq!(r.get_int(6).unwrap(), r.get_int(7).unwrap());
            assert_eq!(r.get_long(8).unwrap(), r.get_long(9).unwrap());
            assert_eq!(r.get_bytes(10).unwrap(), r.get_bytes(11).unwrap());
            assert_eq!(r.get_decimal(12).unwrap(), r.get_decimal(13).unwrap());
        };

        while start < end {
            match iter.next() {
                Some(row) => check_row(row),
                None => break,
            }
            start += 1;
        }
    }

    #[test]
    fn test_filtered_rowgroup_metadata() {
        let message_type = "
            message test_schema {
                REQUIRED INT32 a;
            }
        ";
        let schema = Arc::new(parse_message_type(message_type).unwrap());
        let props = Arc::new(
            WriterProperties::builder()
                .set_statistics_enabled(EnabledStatistics::Page)
                .build(),
        );
        let mut file: File = tempfile::tempfile().unwrap();
        let mut file_writer = SerializedFileWriter::new(&mut file, schema, props).unwrap();
        let data = [1, 2, 3, 4, 5];

        // write 5 row groups
        for idx in 0..5 {
            let data_i: Vec<i32> = data.iter().map(|x| x * (idx + 1)).collect();
            let mut row_group_writer = file_writer.next_row_group().unwrap();
            if let Some(mut writer) = row_group_writer.next_column().unwrap() {
                writer
                    .typed::<Int32Type>()
                    .write_batch(data_i.as_slice(), None, None)
                    .unwrap();
                writer.close().unwrap();
            }
            row_group_writer.close().unwrap();
            file_writer.flushed_row_groups();
        }
        let file_metadata = file_writer.close().unwrap();

        assert_eq!(file_metadata.file_metadata().num_rows(), 25);
        assert_eq!(file_metadata.num_row_groups(), 5);

        // read only the 3rd row group
        let read_options = ReadOptionsBuilder::new()
            .with_page_index()
            .with_predicate(Box::new(|rgmeta, _| rgmeta.ordinal().unwrap_or(0) == 2))
            .build();
        let reader =
            SerializedFileReader::new_with_options(file.try_clone().unwrap(), read_options)
                .unwrap();
        let metadata = reader.metadata();

        // check we got the expected row group
        assert_eq!(metadata.num_row_groups(), 1);
        assert_eq!(metadata.row_group(0).ordinal(), Some(2));

        // check we only got the relevant page indexes
        assert!(metadata.page_index().is_some_and(|idx| idx.is_complete()));
        let page_index = metadata.page_index().unwrap();

        let col_stats = metadata.row_group(0).column(0).statistics().unwrap();
        let pg_idx = page_index.column_index(0, 0);
        let off_idx_i = page_index.offset_index(0, 0);

        // test that we got the index matching the row group
        match pg_idx {
            Some(ColumnIndexMetaData::INT32(int_idx)) => {
                let min = col_stats.min_bytes_opt().unwrap().get_i32_le();
                let max = col_stats.max_bytes_opt().unwrap().get_i32_le();
                assert_eq!(int_idx.min_value(0), Some(min).as_ref());
                assert_eq!(int_idx.max_value(0), Some(max).as_ref());
            }
            _ => panic!("wrong stats type"),
        }

        // check offset index matches too
        assert_eq!(
            off_idx_i.as_ref().unwrap().page_locations[0].offset,
            metadata.row_group(0).column(0).data_page_offset()
        );

        // read non-contiguous row groups
        let read_options = ReadOptionsBuilder::new()
            .with_page_index()
            .with_predicate(Box::new(|rgmeta, _| rgmeta.ordinal().unwrap_or(0) % 2 == 1))
            .build();
        let reader =
            SerializedFileReader::new_with_options(file.try_clone().unwrap(), read_options)
                .unwrap();
        let metadata = reader.metadata();

        // check we got the expected row groups
        assert_eq!(metadata.num_row_groups(), 2);
        assert_eq!(metadata.row_group(0).ordinal(), Some(1));
        assert_eq!(metadata.row_group(1).ordinal(), Some(3));

        // check we only got the relevant page indexes
        assert!(metadata.page_index().is_some_and(|idx| idx.is_complete()));

        let page_index = metadata.page_index().unwrap();

        for rg_idx in 0..metadata.num_row_groups() {
            let col_stats = metadata.row_group(rg_idx).column(0).statistics().unwrap();
            let pg_idx = page_index.column_index(rg_idx, 0);
            let off_idx_i = page_index.offset_index(rg_idx, 0);

            // test that we got the index matching the row group
            match pg_idx {
                Some(ColumnIndexMetaData::INT32(int_idx)) => {
                    let min = col_stats.min_bytes_opt().unwrap().get_i32_le();
                    let max = col_stats.max_bytes_opt().unwrap().get_i32_le();
                    assert_eq!(int_idx.min_value(0), Some(min).as_ref());
                    assert_eq!(int_idx.max_value(0), Some(max).as_ref());
                }
                _ => panic!("wrong stats type"),
            }

            // check offset index matches too
            assert_eq!(
                off_idx_i.as_ref().unwrap().page_locations[0].offset,
                metadata.row_group(rg_idx).column(0).data_page_offset()
            );
        }
    }

    #[test]
    fn test_reuse_schema() {
        let file = get_test_file("alltypes_plain.parquet");
        let file_reader = SerializedFileReader::new(file.try_clone().unwrap()).unwrap();
        let schema = file_reader.metadata().file_metadata().schema_descr_ptr();
        let expected = file_reader.metadata;

        let options = ReadOptionsBuilder::new()
            .with_parquet_schema(schema)
            .build();
        let file_reader = SerializedFileReader::new_with_options(file, options).unwrap();

        assert_eq!(expected.as_ref(), file_reader.metadata.as_ref());
        // Should have used the same schema instance
        assert!(Arc::ptr_eq(
            &expected.file_metadata().schema_descr_ptr(),
            &file_reader.metadata.file_metadata().schema_descr_ptr()
        ));
    }

    #[test]
    fn test_read_unknown_logical_type() {
        let file = get_test_file("unknown-logical-type.parquet");
        let reader = SerializedFileReader::new(file).expect("Error opening file");

        let schema = reader.metadata().file_metadata().schema_descr();
        assert_eq!(
            schema.column(0).logical_type_ref(),
            Some(&basic::LogicalType::String)
        );
        assert_eq!(
            schema.column(1).logical_type_ref(),
            Some(&basic::LogicalType::_Unknown { field_id: 2555 })
        );
        assert_eq!(schema.column(1).physical_type(), Type::BYTE_ARRAY);

        let mut iter = reader
            .get_row_iter(None)
            .expect("Failed to create row iterator");

        let mut num_rows = 0;
        while iter.next().is_some() {
            num_rows += 1;
        }
        assert_eq!(num_rows, reader.metadata().file_metadata().num_rows());
    }

    /// Builds an in-memory single-row-group file with one snappy-compressed,
    /// dictionary-encoded INT32 column holding `1..=5`, and returns its bytes
    /// together with the byte range of the dictionary page *body* (the
    /// compressed payload after the page header), so tests can corrupt
    /// exactly the part lazy skipping promises never to touch.
    fn dict_column_file() -> (Bytes, std::ops::Range<usize>) {
        let message_type = "
            message test_schema {
                REQUIRED INT32 a;
            }
        ";
        let schema = Arc::new(parse_message_type(message_type).unwrap());
        let props = Arc::new(
            WriterProperties::builder()
                .set_compression(Compression::SNAPPY)
                .build(),
        );
        let mut out = Vec::new();
        let mut writer = SerializedFileWriter::new(&mut out, schema, props).unwrap();
        let mut rg = writer.next_row_group().unwrap();
        let mut col = rg.next_column().unwrap().unwrap();
        col.typed::<Int32Type>()
            .write_batch(&[1, 2, 3, 4, 5], None, None)
            .unwrap();
        col.close().unwrap();
        rg.close().unwrap();
        writer.close().unwrap();
        let bytes = Bytes::from(out);

        let reader = SerializedFileReader::new(bytes.clone()).unwrap();
        let col_meta = reader.metadata().row_group(0).column(0).clone();
        let dict_offset = col_meta.dictionary_page_offset().unwrap() as usize;
        let data_offset = col_meta.data_page_offset() as usize;
        // Parse the dictionary page header to find where its body starts.
        let mut prot = ThriftSliceInputProtocol::new(&bytes[dict_offset..data_offset]);
        let _header = PageHeader::read_thrift(&mut prot).unwrap();
        let header_len = (data_offset - dict_offset) - prot.as_slice().len();
        (bytes, dict_offset + header_len..data_offset)
    }

    /// Zeroes the dictionary page body: an all-zero snappy stream declares an
    /// uncompressed length of zero, so decoding it fails deterministically on
    /// the decompressed-size check. Any code path that decompresses the
    /// dictionary therefore errors; a path that genuinely never touches it
    /// succeeds.
    fn corrupt_dictionary_body(bytes: &Bytes, body: &std::ops::Range<usize>) -> Bytes {
        let mut v = bytes.to_vec();
        v[body.clone()].fill(0);
        Bytes::from(v)
    }

    fn int32_reader(bytes: Bytes) -> crate::column::reader::ColumnReaderImpl<Int32Type> {
        let reader = SerializedFileReader::new(bytes.clone()).unwrap();
        let meta = reader.metadata();
        let descr = meta.file_metadata().schema_descr().column(0);
        let col_meta = meta.row_group(0).column(0).clone();
        let total_rows = meta.row_group(0).num_rows() as usize;
        let pages =
            SerializedPageReader::new(Arc::new(bytes), &col_meta, total_rows, None).unwrap();
        crate::column::reader::ColumnReaderImpl::<Int32Type>::new(descr, Box::new(pages))
    }

    /// Skipping the whole chunk never decompresses the dictionary: with the
    /// dictionary body corrupted so that any decompression errors, the skip
    /// still succeeds. Reverting the lazy branch in
    /// `GenericColumnReader::skip_records` turns this test red.
    #[test]
    fn skipping_a_whole_chunk_never_decodes_the_dictionary() {
        let (bytes, body) = dict_column_file();
        let corrupted = corrupt_dictionary_body(&bytes, &body);
        let mut reader = int32_reader(corrupted);
        let skipped = reader.skip_records(5).unwrap();
        assert_eq!(skipped, 5);
    }

    /// The deferred dictionary is genuinely decoded the moment a data page
    /// needs it: the same corruption that whole-chunk skipping never notices
    /// fails a skip that has to open the data page.
    #[test]
    fn a_partial_skip_pays_the_deferred_dictionary() {
        let (bytes, body) = dict_column_file();
        let corrupted = corrupt_dictionary_body(&bytes, &body);
        let mut reader = int32_reader(corrupted);
        let err = reader.skip_records(2).unwrap_err().to_string();
        assert!(
            err.contains("corrupt") || err.contains("size"),
            "expected the deferred dictionary decode to fail on \
             decompression, got: {err}"
        );
    }

    /// Skip-then-read returns the right values: the dictionary deferred by
    /// the skip is installed before the first value decodes.
    #[test]
    fn skip_then_read_decodes_correct_values() {
        let (bytes, _) = dict_column_file();
        let mut reader = int32_reader(bytes);
        assert_eq!(reader.skip_records(2).unwrap(), 2);
        let mut values = Vec::new();
        let (records, _, read) = reader.read_records(3, None, None, &mut values).unwrap();
        assert_eq!(records, 3);
        assert_eq!(read, 3);
        assert_eq!(values, vec![3, 4, 5]);
    }

    /// The eager path is untouched: reading from the start still installs the
    /// dictionary through `get_next_page` and decodes every value.
    #[test]
    fn plain_read_is_unchanged() {
        let (bytes, _) = dict_column_file();
        let mut reader = int32_reader(bytes);
        let mut values = Vec::new();
        let (records, _, read) = reader.read_records(5, None, None, &mut values).unwrap();
        assert_eq!((records, read), (5, 5));
        assert_eq!(values, vec![1, 2, 3, 4, 5]);
    }

    /// A [`PageReader`] that implements only what existed before deferral: it
    /// forwards the required trait methods and takes the defaults for
    /// `supports_deferred_dictionary` (`false`) and `take_deferred_dictionary`
    /// (`Ok(None)`). This stands in for every third-party implementation, none
    /// of which can retain a dictionary that `skip_next_page` skips past.
    struct NonDeferringPageReader(SerializedPageReader<Bytes>);

    impl Iterator for NonDeferringPageReader {
        type Item = Result<Page>;

        fn next(&mut self) -> Option<Self::Item> {
            self.get_next_page().transpose()
        }
    }

    impl PageReader for NonDeferringPageReader {
        fn get_next_page(&mut self) -> Result<Option<Page>> {
            self.0.get_next_page()
        }

        fn peek_next_page(&mut self) -> Result<Option<PageMetadata>> {
            self.0.peek_next_page()
        }

        fn skip_next_page(&mut self) -> Result<()> {
            self.0.skip_next_page()
        }
    }

    fn non_deferring_int32_reader(
        bytes: Bytes,
    ) -> crate::column::reader::ColumnReaderImpl<Int32Type> {
        let reader = SerializedFileReader::new(bytes.clone()).unwrap();
        let meta = reader.metadata();
        let descr = meta.file_metadata().schema_descr().column(0);
        let col_meta = meta.row_group(0).column(0).clone();
        let total_rows = meta.row_group(0).num_rows() as usize;
        let pages =
            SerializedPageReader::new(Arc::new(bytes), &col_meta, total_rows, None).unwrap();
        crate::column::reader::ColumnReaderImpl::<Int32Type>::new(
            descr,
            Box::new(NonDeferringPageReader(pages)),
        )
    }

    /// A page reader that cannot give a skipped dictionary back must still get
    /// the eager path. Skipping past its dictionary would drop it, and the
    /// first dictionary-encoded data page decoded afterwards would have no
    /// dictionary to decode against — a panic in `set_data`, not an error.
    ///
    /// Removing the `supports_deferred_dictionary` gate in
    /// `GenericColumnReader::skip_records` turns this test red.
    #[test]
    fn a_reader_that_cannot_defer_still_gets_its_dictionary() {
        let (bytes, _) = dict_column_file();
        let mut reader = non_deferring_int32_reader(bytes);

        assert_eq!(reader.skip_records(2).unwrap(), 2);

        let mut values = Vec::new();
        let (records, _, read) = reader.read_records(3, None, None, &mut values).unwrap();
        assert_eq!(records, 3);
        assert_eq!(read, 3);
        assert_eq!(values, vec![3, 4, 5]);
    }

    /// The same reader must also survive a whole-chunk skip. Nothing is
    /// decoded, so nothing needs the dictionary, but the eager read still has
    /// to leave the page cursor in a consistent place.
    #[test]
    fn a_reader_that_cannot_defer_survives_a_whole_chunk_skip() {
        let (bytes, _) = dict_column_file();
        let mut reader = non_deferring_int32_reader(bytes);

        assert_eq!(reader.skip_records(5).unwrap(), 5);

        let mut values = Vec::new();
        let (records, _, _) = reader.read_records(1, None, None, &mut values).unwrap();
        assert_eq!(records, 0, "chunk is exhausted");
        assert!(values.is_empty());
    }

    #[test]
    fn unselected_page_default_still_calls_skip_next_page() {
        let (bytes, _) = dict_column_file();
        let reader = SerializedFileReader::new(bytes.clone()).unwrap();
        let group = reader.metadata().row_group(0);
        let pages = SerializedPageReader::new(Arc::new(bytes), group.column(0), 5, None).unwrap();
        let mut pages = NonDeferringPageReader(pages);
        assert!(pages.peek_next_page().unwrap().unwrap().is_dict);
        pages.skip_unselected_data_page().unwrap();
        assert!(!pages.peek_next_page().unwrap().unwrap().is_dict);
        pages.skip_unselected_data_page().unwrap();
        assert!(pages.peek_next_page().unwrap().is_none());
    }

    #[cfg(all(feature = "arrow", feature = "snap"))]
    #[test]
    fn uncompressed_v2_page_cache_owns_a_body_in_a_compressed_chunk() {
        use crate::file::page_cache::testing::MapPageCache;
        use crate::file::properties::WriterVersion;

        let schema = Arc::new(parse_message_type("message test { REQUIRED INT32 a; }").unwrap());
        let props = Arc::new(
            WriterProperties::builder()
                .set_writer_version(WriterVersion::PARQUET_2_0)
                .set_compression(Compression::SNAPPY)
                .set_dictionary_enabled(false)
                .set_data_page_v2_compression_ratio_threshold(0.001)
                .build(),
        );
        let mut writer = SerializedFileWriter::new(Vec::new(), schema, props).unwrap();
        let mut group = writer.next_row_group().unwrap();
        let mut column = group.next_column().unwrap().unwrap();
        column
            .typed::<Int32Type>()
            .write_batch(&[42; 128], None, None)
            .unwrap();
        column.close().unwrap();
        group.close().unwrap();
        let bytes = Arc::new(Bytes::from(writer.into_inner().unwrap()));
        let options = ReadOptionsBuilder::new().with_page_index().build();
        let reader =
            SerializedFileReader::new_with_options(bytes.as_ref().clone(), options).unwrap();
        let metadata = reader.metadata();
        let column = metadata.row_group(0).column(0);
        let indexed = metadata
            .page_index_for_row_group(0)
            .page_locations(0)
            .unwrap()
            .clone();
        assert_eq!(column.compression(), Compression::SNAPPY);

        for locations in [None, Some(indexed)] {
            let cache = Arc::new(MapPageCache::with_leases());
            let make = || {
                SerializedPageReader::new(bytes.clone(), column, 128, locations.clone())
                    .unwrap()
                    .with_page_cache(Some(cache.clone()), None)
            };
            let page = make().get_next_page().unwrap().unwrap();
            assert!(matches!(
                &page,
                Page::DataPageV2 {
                    is_compressed: false,
                    ..
                }
            ));
            let offset = column.data_page_offset() as u64;
            let cached = cache.probe(offset).expect("V2 page was admitted");
            assert_eq!(cached.codec(), Compression::SNAPPY);
            assert_eq!(cached.page().buffer(), page.buffer());
            assert_ne!(cached.page().buffer().as_ptr(), page.buffer().as_ptr());
            assert_eq!(
                make().get_next_page().unwrap().unwrap().buffer().as_ptr(),
                cached.page().buffer().as_ptr()
            );
        }
    }

    #[cfg(feature = "arrow")]
    #[test]
    fn unselected_page_cache_fills_once_and_skips_hits_without_reading() {
        use crate::file::page_cache::testing::MapPageCache;
        use std::sync::atomic::Ordering::Relaxed;

        let (bytes, _) = dict_column_file();
        let reader = SerializedFileReader::new(bytes.clone()).unwrap();
        let group = reader.metadata().row_group(0);
        let cache = Arc::new(MapPageCache::with_leases());
        let make = || {
            SerializedPageReader::new(Arc::new(bytes.clone()), group.column(0), 5, None)
                .unwrap()
                .with_page_cache(Some(cache.clone()), None)
        };
        let mut pages = make();
        assert!(pages.get_next_page().unwrap().unwrap().is_dictionary_page());
        assert_eq!(cache.fills.load(Relaxed), 1);
        assert!(!pages.peek_next_page().unwrap().unwrap().is_dict);
        pages.skip_unselected_data_page().unwrap();
        assert_eq!(cache.fills.load(Relaxed), 2);
        assert_eq!(cache.hits.load(Relaxed), 0);

        let mut pages = make();
        assert!(pages.peek_next_page().unwrap().unwrap().is_dict);
        pages.skip_next_page().unwrap();
        pages.skip_unselected_data_page().unwrap();
        assert_eq!(cache.fills.load(Relaxed), 2);
        assert_eq!(cache.hits.load(Relaxed), 0);

        let mut pages = make();
        pages.skip_next_page().unwrap();
        assert!(pages.get_next_page().unwrap().unwrap().is_data_page());
        assert_eq!(cache.hits.load(Relaxed), 1);

        let (start, len) = group.column(0).byte_range();
        let chunk = start..start + len;
        let lease = Arc::new(cache.lease(&[vec![chunk.clone()]]).unwrap());
        let pinned = Arc::new(PinnedPages::new(lease, chunk));
        let mut pages = SerializedPageReader::new(Arc::new(bytes), group.column(0), 5, None)
            .unwrap()
            .with_page_cache(Some(cache.clone()), Some(pinned));
        pages.skip_next_page().unwrap();
        pages.skip_unselected_data_page().unwrap();
        drop(pages);
        assert_eq!(cache.hits.load(Relaxed), 1);
        assert_eq!(cache.lease_reads.load(Relaxed), 0);
    }

    #[test]
    fn unselected_page_skips_corrupt_compressed_body_until_read() {
        let (bytes, _) = dict_column_file();
        let reader = SerializedFileReader::new(bytes.clone()).unwrap();
        let group = reader.metadata().row_group(0);
        let column = group.column(0);
        let start = column.data_page_offset() as usize;
        let original = SerializedPageReader::new(Arc::new(bytes.clone()), column, 5, None).unwrap();
        let (header_len, header) =
            read_page_header_len_from_bytes(&original.context, &bytes[start..], 0, false).unwrap();
        let mut corrupt = bytes.to_vec();
        let start = start + header_len;
        corrupt[start..start + header.compressed_page_size as usize].fill(0);
        let corrupt = Arc::new(Bytes::from(corrupt));
        let make = || SerializedPageReader::new(corrupt.clone(), column, 5, None).unwrap();
        let mut pages = make();
        assert!(pages.get_next_page().unwrap().unwrap().is_dictionary_page());
        pages.skip_unselected_data_page().unwrap();
        assert!(pages.get_next_page().unwrap().is_none());

        let mut pages = make();
        assert!(pages.get_next_page().unwrap().unwrap().is_dictionary_page());
        assert!(pages.get_next_page().is_err());
    }

    /// The probe has to describe what the reader can actually do. Without page
    /// locations the reader walks page headers, so it recognises a dictionary
    /// by its header type and can always defer.
    #[test]
    fn values_state_supports_deferral() {
        let (bytes, _) = dict_column_file();
        let reader = SerializedFileReader::new(bytes.clone()).unwrap();
        let meta = reader.metadata();
        let col_meta = meta.row_group(0).column(0).clone();
        let total_rows = meta.row_group(0).num_rows() as usize;
        let pages =
            SerializedPageReader::new(Arc::new(bytes), &col_meta, total_rows, None).unwrap();

        assert!(pages.supports_deferred_dictionary());
    }

    /// With page locations, a dictionary is only representable when the
    /// metadata recorded an offset for it. `dict_column_file` writes one, so
    /// the synthesised `dictionary_page` is present and deferral is safe.
    #[test]
    fn offset_index_state_with_a_dictionary_offset_supports_deferral() {
        let (bytes, _) = dict_column_file();
        let reader = SerializedFileReader::new(bytes.clone()).unwrap();
        let meta = reader.metadata();
        let col_meta = meta.row_group(0).column(0).clone();
        let total_rows = meta.row_group(0).num_rows() as usize;
        let (dict_offset, data_offset) = (
            col_meta.dictionary_page_offset().unwrap(),
            col_meta.data_page_offset(),
        );
        assert_ne!(
            dict_offset, data_offset,
            "fixture must record a dictionary offset"
        );

        let locations = vec![PageLocation {
            offset: data_offset,
            compressed_page_size: 0,
            first_row_index: 0,
        }];
        let pages =
            SerializedPageReader::new(Arc::new(bytes), &col_meta, total_rows, Some(locations))
                .unwrap();

        assert!(pages.supports_deferred_dictionary());
    }

    /// When the metadata records no dictionary offset, `byte_range()` starts
    /// at the first data page, so the synthesised `dictionary_page` is `None`
    /// and a dictionary inlined ahead of that page is indistinguishable from a
    /// data page location. Skipping it would drop it, so the probe must say
    /// so and let the column reader read it eagerly instead.
    #[test]
    fn offset_index_state_without_a_dictionary_offset_declines_deferral() {
        let (bytes, _) = dict_column_file();
        let reader = SerializedFileReader::new(bytes.clone()).unwrap();
        let meta = reader.metadata();
        let total_rows = meta.row_group(0).num_rows() as usize;

        // Drop the dictionary offset, reproducing the older parquet-mr writers
        // that inline a dictionary without recording where it is.
        let col_meta = meta.row_group(0).column(0).clone();
        let stripped = col_meta
            .clone()
            .into_builder()
            .set_dictionary_page_offset(None)
            .build()
            .unwrap();

        let locations = vec![PageLocation {
            offset: stripped.data_page_offset(),
            compressed_page_size: 0,
            first_row_index: 0,
        }];
        let pages =
            SerializedPageReader::new(Arc::new(bytes), &stripped, total_rows, Some(locations))
                .unwrap();

        assert!(
            !pages.supports_deferred_dictionary(),
            "an offset-less dictionary cannot be recovered from the offset-index state"
        );
        // Note this combination is unlikely to occur in practice: a writer old
        // enough to omit `dictionary_page_offset` predates the offset index,
        // so it would not emit page locations either. The probe answers
        // defensively rather than relying on that.
    }

    /// Alternating skip and read is the shape a `RowSelection` produces, and
    /// it exercises the deferral more than once per chunk: the first skip
    /// defers, the first read installs, and every later skip has to cope with
    /// a dictionary that is already installed.
    #[test]
    fn alternating_skip_and_read_decodes_correct_values() {
        let (bytes, _) = dict_column_file();
        let mut reader = int32_reader(bytes);

        assert_eq!(reader.skip_records(1).unwrap(), 1);
        let mut first = Vec::new();
        let (records, _, _) = reader.read_records(2, None, None, &mut first).unwrap();
        assert_eq!(records, 2);
        assert_eq!(first, vec![2, 3]);

        assert_eq!(reader.skip_records(1).unwrap(), 1);
        let mut second = Vec::new();
        let (records, _, _) = reader.read_records(1, None, None, &mut second).unwrap();
        assert_eq!(records, 1);
        assert_eq!(second, vec![5]);
    }

    /// A wrapper that forwards `take_deferred_dictionary` but not
    /// `supports_deferred_dictionary`, which is the easy mistake to make when
    /// adding the new methods to an existing wrapper.
    struct HalfForwardingPageReader(SerializedPageReader<Bytes>);

    impl Iterator for HalfForwardingPageReader {
        type Item = Result<Page>;

        fn next(&mut self) -> Option<Self::Item> {
            self.get_next_page().transpose()
        }
    }

    impl PageReader for HalfForwardingPageReader {
        fn get_next_page(&mut self) -> Result<Option<Page>> {
            self.0.get_next_page()
        }

        fn peek_next_page(&mut self) -> Result<Option<PageMetadata>> {
            self.0.peek_next_page()
        }

        fn skip_next_page(&mut self) -> Result<()> {
            self.0.skip_next_page()
        }

        fn take_deferred_dictionary(&mut self) -> Result<Option<Page>> {
            self.0.take_deferred_dictionary()
        }
    }

    /// Forwarding only half the pair must stay correct: the probe still
    /// reports `false`, so the skip path reads the dictionary eagerly and the
    /// forwarded `take_deferred_dictionary` simply never has anything to
    /// return. Slower than forwarding both, never wrong.
    #[test]
    fn forwarding_only_take_deferred_dictionary_is_still_correct() {
        let (bytes, _) = dict_column_file();
        let reader = SerializedFileReader::new(bytes.clone()).unwrap();
        let meta = reader.metadata();
        let descr = meta.file_metadata().schema_descr().column(0);
        let col_meta = meta.row_group(0).column(0).clone();
        let total_rows = meta.row_group(0).num_rows() as usize;
        let pages =
            SerializedPageReader::new(Arc::new(bytes), &col_meta, total_rows, None).unwrap();
        let mut reader = crate::column::reader::ColumnReaderImpl::<Int32Type>::new(
            descr,
            Box::new(HalfForwardingPageReader(pages)),
        );

        assert_eq!(reader.skip_records(2).unwrap(), 2);
        let mut values = Vec::new();
        let (records, _, _) = reader.read_records(3, None, None, &mut values).unwrap();
        assert_eq!(records, 3);
        assert_eq!(values, vec![3, 4, 5]);
    }
}

#[cfg(test)]
mod invalid_input_tests {
    use super::*;
    use crate::basic::Encoding;
    use crate::file::metadata::thrift::DataPageHeaderV2;

    fn v2_header(
        uncompressed_page_size: i32,
        definition_levels_byte_length: i32,
        repetition_levels_byte_length: i32,
    ) -> PageHeader {
        PageHeader {
            r#type: PageType::DATA_PAGE_V2,
            uncompressed_page_size,
            compressed_page_size: 10,
            crc: None,
            data_page_header: None,
            index_page_header: None,
            dictionary_page_header: None,
            data_page_header_v2: Some(DataPageHeaderV2 {
                num_values: 4,
                num_nulls: 0,
                num_rows: 4,
                encoding: Encoding::PLAIN,
                definition_levels_byte_length,
                repetition_levels_byte_length,
                is_compressed: Some(false),
                statistics: None,
            }),
        }
    }

    /// The sum of two valid 32-bit level lengths can exceed `i32::MAX`.
    #[test]
    fn v2_level_lengths_sum_does_not_overflow() {
        let header = v2_header(i32::MAX, i32::MAX, i32::MAX);
        let err = decode_page(header, Bytes::from(vec![0; 10]), Type::INT32, None).unwrap_err();
        assert!(
            err.to_string()
                .contains("DataPage v2 header contains implausible values"),
            "{err}"
        );
    }

    /// An uncompressed page has the compressed size, so the uncompressed size
    /// in the header does not bound the levels.
    #[test]
    fn v2_levels_longer_than_uncompressed_page_fails() {
        let header = v2_header(100, 8, 8);
        let err = decode_page(header, Bytes::from(vec![0; 10]), Type::INT32, None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Parquet error: DataPage v2 levels need 16 bytes, but the page has 10 bytes"
        );
    }

    #[test]
    fn peeked_data_page_without_its_header_fails() {
        let mut header = v2_header(10, 0, 0);
        header.data_page_header_v2 = None;
        let err = peeked_page_metadata(&header).err().unwrap();
        assert_eq!(
            err.to_string(),
            "Parquet error: Missing V2 data page header"
        );

        // A page type that the reader skips is not an error.
        header.r#type = PageType::INDEX_PAGE;
        assert!(peeked_page_metadata(&header).unwrap().is_none());
    }
}
