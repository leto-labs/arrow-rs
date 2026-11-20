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

//! A hook for a cache of decompressed pages, shared by every reader of one
//! file generation.
//!
//! This crate defines the contract and uses it in
//! [`SerializedPageReader`](crate::file::serialized_reader::SerializedPageReader)
//! and in the push decoder's fetch planning. It holds no pages itself: the
//! embedding engine implements [`PageCache`], owns the memory, the eviction
//! policy and the identity of the file, and attaches one handle to the
//! [`ParquetMetaData`](crate::file::metadata::ParquetMetaData) of that file
//! with
//! [`ParquetMetaData::with_page_cache`](crate::file::metadata::ParquetMetaData::with_page_cache).
//!
//! A cached page is the [`Page`] that the page reader returns: the page
//! header is parsed and checked, the CRC is verified and the body is
//! decompressed. Its values keep their Parquet encoding.
//!
//! # Checks of a cached page
//!
//! A hit skips the file bytes, the page header parse, the CRC and the
//! decompression. These checks guard the file bytes, and the cached page
//! passed them when it was filled. The reader repeats, for each hit, the
//! checks that depend on where the page is read and that guard decoding:
//!
//! * the page was decompressed with the codec of its column chunk;
//! * the page fits the bytes left in its column chunk, and with an offset
//!   index its stored length is the length of its page location;
//! * with an offset index, the page kind (dictionary or data) is the kind
//!   of its location;
//! * when the reader knows the page header (after a peek, or for a skipped
//!   dictionary), the stored length is the header and the body that the
//!   header declares.
//!
//! A page that fails a check is read from the file bytes, as on a miss, when
//! its column chunk was fetched. It is an error only for a page that fetch
//! planning pinned: the bytes of its chunk were not fetched.
//!
//! The column reader checks the contents of every page against its buffer
//! (level lengths, value counts, dictionary indices), whichever page reader
//! supplied it. A cache that returns the page of another file generation is
//! an error of the engine; the checks above make most such errors a cold
//! read or a [`ParquetError`](crate::errors::ParquetError), not a wrong
//! result.
//!
//! Only pages of compressed, unencrypted column chunks are cached. The buffer
//! of an uncompressed page is a slice of the fetched column chunk, so caching
//! it would keep the whole chunk alive and save no decompression.
//!
//! # Planning reads with the cache
//!
//! The push decoder does not fetch a column chunk whose needed pages are all
//! resident. Its fetch planning takes the pages of each decoding stage of a
//! row group with one [`PageCache::lease`] call, for all the stage's column
//! chunks; the page readers hold the lease until they are dropped, and its
//! release records the reads.
//!
//! A caller that already holds a lease of a row group's pages (an engine that
//! admits the row group before it decodes it) hands it to the decoder with
//! [`ParquetPushDecoderBuilder::with_page_lease`](crate::arrow::push_decoder::ParquetPushDecoderBuilder::with_page_lease).
//! The decoder then takes the pages of a column chunk from that lease when
//! the lease tiles its needed ranges, and reads the bytes that the caller
//! pushed for the other chunks: it makes no cache call of its own at
//! planning. A cached row group then takes the cache's lock twice, for the
//! lease and its release.
//!
//! A [`ScanPlan`](crate::arrow::push_decoder::ScanPlan) does not know the
//! cache and lists every range. A caller that fetches ahead from a scan plan
//! skips a range whose lease [`is_charged`](PageLease::is_charged), and holds
//! the lease until the decoder has read the row group.

pub(crate) mod plain;

use crate::basic::Compression;
use crate::column::page::{Page, PageMetadata};
use crate::file::metadata::ColumnChunkMetaData;
use bytes::Bytes;
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

/// The definition levels of a flat nullable data page, in Arrow bit order.
#[doc(hidden)]
#[derive(Debug)]
pub struct CachedValidity {
    pub(crate) bits: Bytes,
    #[cfg(feature = "arrow")]
    pub(crate) len: usize,
    #[cfg(feature = "arrow")]
    pub(crate) nulls: usize,
}

/// A decompressed page and its optional cached validity.
#[doc(hidden)]
pub type PageWithValidity = (Page, Option<Arc<CachedValidity>>);

impl CachedValidity {
    fn charge(bitmap_bytes: usize) -> usize {
        bitmap_bytes + std::mem::size_of::<Self>() + 2 * std::mem::size_of::<usize>()
    }

    fn buffer_len(&self) -> usize {
        Self::charge(self.bits.len())
    }
}

/// One decompressed page, the number of bytes it occupies in the file and
/// the codec that it was decompressed with.
#[derive(Clone, Debug)]
pub struct CachedPage {
    page: Page,
    stored_len: u64,
    codec: Compression,
    dictionary_marker: bool,
    validity: Option<Arc<CachedValidity>>,
}

impl CachedPage {
    /// `stored_len` is the length of the page in the file: its header and its
    /// stored (compressed) body. The next page of the column chunk starts
    /// `stored_len` bytes after this one. `codec` is the compression of its
    /// column chunk; a reader of a chunk with another codec rejects the page.
    pub fn new(page: Page, stored_len: u64, codec: Compression) -> Self {
        Self {
            page,
            stored_len,
            codec,
            dictionary_marker: false,
            validity: None,
        }
    }

    /// Offers the page to a fill ticket, with optional flat nullable validity.
    #[doc(hidden)]
    pub fn publish(self, fill: Box<dyn PageFill>, flat_nullable: bool) {
        #[cfg(feature = "arrow")]
        if flat_nullable && let Some(bytes) = self.validity_build_charge() {
            return fill.complete_with_validity(self, bytes, Self::build_validity);
        }
        let _ = flat_nullable;
        fill.complete(self);
    }

    #[cfg(feature = "arrow")]
    fn validity_build_charge(&self) -> Option<usize> {
        // Bound work and temporary memory even when the cache has ample room.
        const MAX_BITMAP_BYTES: usize = 1024 * 1024;
        match &self.page {
            Page::DataPage {
                def_level_encoding: crate::basic::Encoding::RLE,
                ..
            }
            | Page::DataPageV2 {
                rep_levels_byte_len: 0,
                ..
            } => {}
            _ => return None,
        }
        let bitmap_bytes = (self.page.num_values() as usize).div_ceil(8);
        (bitmap_bytes <= MAX_BITMAP_BYTES).then(|| {
            // The builder holds the bitmap and eight bytes of word-store slack,
            // rounded up by MutableBuffer. Its allocation overlaps the copy.
            CachedValidity::charge(bitmap_bytes) + (bitmap_bytes + 8).next_multiple_of(64)
        })
    }

    #[cfg(feature = "arrow")]
    fn build_validity(mut self) -> Self {
        self.validity = crate::arrow::record_reader::definition_levels::cache_validity(&self.page)
            .map(Arc::new);
        self
    }

    pub(crate) fn into_page_with_validity(self) -> PageWithValidity {
        (self.page, self.validity)
    }

    /// Retains the dictionary's file range for lease tiling without keeping its values.
    pub(crate) fn dictionary_marker(page: &Page, stored_len: u64, codec: Compression) -> Self {
        let mut page = page.clone();
        if let Page::DictionaryPage { buf, .. } = &mut page {
            *buf = Default::default();
        }
        let mut cached = Self::new(page, stored_len, codec);
        cached.dictionary_marker = true;
        cached
    }

    pub(crate) fn is_dictionary_marker(&self) -> bool {
        self.dictionary_marker
    }

    /// The decompressed page.
    pub fn page(&self) -> &Page {
        &self.page
    }

    /// Length of the page in the file, header included.
    pub fn stored_len(&self) -> u64 {
        self.stored_len
    }

    /// The codec that the page was decompressed with.
    pub fn codec(&self) -> Compression {
        self.codec
    }

    /// Bytes retained by the decompressed buffer and optional validity sidecar.
    pub fn buffer_len(&self) -> usize {
        self.page.buffer().len()
            + self
                .validity
                .as_deref()
                .map_or(0, CachedValidity::buffer_len)
    }

    /// What `peek_next_page` reports for this page when the column chunk has
    /// no offset index: the same fields as the parsed page header gives.
    pub(crate) fn metadata(&self) -> PageMetadata {
        match &self.page {
            Page::DataPage { num_values, .. } => PageMetadata {
                num_rows: None,
                num_levels: Some(*num_values as usize),
                is_dict: false,
            },
            Page::DataPageV2 {
                num_values,
                num_rows,
                ..
            } => PageMetadata {
                num_rows: Some(*num_rows as usize),
                num_levels: Some(*num_values as usize),
                is_dict: false,
            },
            Page::DictionaryPage { .. } => PageMetadata {
                num_rows: None,
                num_levels: None,
                is_dict: true,
            },
        }
    }
}

/// The result of [`PageCache::get`].
pub enum PageLookup {
    /// The page is resident.
    Hit(CachedPage),
    /// The page is not resident. With a ticket, the caller is the one reader
    /// that decodes this page: it hands the page to
    /// [`PageFill::complete`], or drops the ticket when it cannot (an error,
    /// a page that is not cacheable). Without a ticket the cache does not
    /// want the page.
    Miss(Option<Box<dyn PageFill>>),
}

/// The right to publish one page. Dropping it without a call to
/// [`complete`](Self::complete) abandons the fill and releases any reader
/// that waits for it.
pub trait PageFill: Send {
    /// Offers the decoded page to the cache. The cache may refuse it.
    fn complete(self: Box<Self>, page: CachedPage);

    /// Offers a page with optional validity. Invoke `build` only after
    /// admitting the page plus `extra_bytes`, which covers temporary and
    /// retained bitmap allocations. Charge the returned page's actual size
    /// for residency. A cache that declines the extra bytes can keep the
    /// encoded page.
    fn complete_with_validity(
        self: Box<Self>,
        page: CachedPage,
        _extra_bytes: usize,
        _build: fn(CachedPage) -> CachedPage,
    ) {
        self.complete(page);
    }
}

/// The engine's side of a [`PageLease`]: it unpins the leased pages and
/// records their reads.
pub trait LeaseRelease: Send + Sync {
    /// Called once, when the last holder drops the lease. `pages` lists every
    /// page of the lease, in [`PageLease::pages`] order (by file offset), with
    /// the reads that the decoder marked; a page with no read records no
    /// access. One acquisition of the engine's lock should cover the call.
    fn release(self: Box<Self>, pages: &[LeasedPage]);
}

/// One page that a [`PageLease`] holds, with the reads that the decoder
/// marked on it.
#[derive(Debug)]
pub struct LeasedPage {
    offset: u64,
    page: CachedPage,
    reads: AtomicU32,
}

impl LeasedPage {
    /// The file offset of the page header.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// The page.
    pub fn page(&self) -> &CachedPage {
        &self.page
    }

    /// The reads that page readers marked: one each time a reader reads
    /// this page (not a peek or a skip), as a `get` counts one. A page that
    /// planning took but no reader read (a cancelled or dropped request, a
    /// page skipped by the selection) has none.
    pub fn reads(&self) -> u32 {
        self.reads.load(Ordering::Relaxed)
    }

    fn mark(&self) {
        self.reads.fetch_add(1, Ordering::Relaxed);
    }
}

/// Pages that one call of [`PageCache::lease`] took from the cache, for the
/// sets of ranges that it was given. The pages of a set are all there, or
/// none are.
///
/// The lease holds the pages themselves, so a page stays readable through the
/// lease whatever the cache does meanwhile. A cache that supports leases also
/// keeps the pages resident and charged until the lease drops (see
/// [`PageCache::lease`]). A page reader marks each page that it reads;
/// dropping the lease hands the marks to the engine's [`LeaseRelease`].
///
/// Share a lease with [`Arc`]: the decoders of the slices of one row group
/// can read from one lease, and it is released when the last one drops it.
pub struct PageLease {
    /// The pages, sorted by file offset.
    pages: Vec<LeasedPage>,
    /// Whether each requested set was leased.
    leased: Vec<bool>,
    release: Option<Box<dyn LeaseRelease>>,
}

impl Debug for PageLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PageLease")
            .field("pages", &self.pages.len())
            .field("leased", &self.leased)
            .field("charged", &self.release.is_some())
            .finish()
    }
}

impl PageLease {
    /// A lease of `sets`: for each requested set, its pages (offset and
    /// page) or `None` when the set was not leased. `release` is called when
    /// the lease drops; `None` for a cache that keeps no pins.
    pub fn new(
        sets: Vec<Option<Vec<(u64, CachedPage)>>>,
        release: Option<Box<dyn LeaseRelease>>,
    ) -> Self {
        let leased = sets.iter().map(Option::is_some).collect();
        let mut pages: Vec<LeasedPage> = sets
            .into_iter()
            .flatten()
            .flatten()
            .map(|(offset, page)| LeasedPage {
                offset,
                page,
                reads: AtomicU32::new(0),
            })
            .collect();
        // Stable: the engine can list its pins in the same order.
        pages.sort_by_key(|page| page.offset);
        Self {
            pages,
            leased,
            release,
        }
    }

    /// Whether the cache keeps the leased pages resident and charged until
    /// the lease drops (a lease with a [`LeaseRelease`]).
    pub fn is_charged(&self) -> bool {
        self.release.is_some()
    }

    /// Whether set `set` of the request was leased.
    pub fn is_leased(&self, set: usize) -> bool {
        self.leased.get(set).copied().unwrap_or(false)
    }

    /// The number of sets of the request.
    pub fn sets(&self) -> usize {
        self.leased.len()
    }

    /// Every leased page, by file offset.
    pub fn pages(&self) -> &[LeasedPage] {
        &self.pages
    }

    /// The leased page that starts at `offset`.
    pub fn page(&self, offset: u64) -> Option<&LeasedPage> {
        self.pages
            .binary_search_by_key(&offset, |page| page.offset)
            .ok()
            .map(|index| &self.pages[index])
    }

    /// Whether the leased pages tile every range of `ranges` (see
    /// [`PageCache::lease`]). Marks nothing: a page's read is marked when a
    /// page reader reads it.
    pub(crate) fn covers(&self, ranges: impl Iterator<Item = Range<u64>>) -> bool {
        let mut pages = vec![];
        ranges.into_iter().all(|range| {
            tile(range, &mut pages, |offset| {
                let page = self.page(offset)?;
                Some(((), page.page.stored_len()))
            })
        })
    }
}

impl Drop for PageLease {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            release.release(&self.pages);
        }
    }
}

/// Follows the pages that cover `range` without a gap from its start, each
/// to the next by its stored length, and appends them to `pages`. Returns
/// `false` (with `pages` in an unspecified state) when a page is missing,
/// has a stored length of zero, or the pages do not end exactly at
/// `range.end`. `page_at` gives the page that starts at an offset and its
/// stored length.
fn tile<P>(
    range: Range<u64>,
    pages: &mut Vec<P>,
    mut page_at: impl FnMut(u64) -> Option<(P, u64)>,
) -> bool {
    let mut offset = range.start;
    while offset < range.end {
        let Some((page, len)) = page_at(offset) else {
            return false;
        };
        match offset.checked_add(len) {
            Some(next) if next > offset && next <= range.end => offset = next,
            _ => return false,
        }
        pages.push(page);
    }
    true
}

/// Decompressed pages of one file generation, keyed by the file offset of the
/// page header.
///
/// The implementation decides the identity of the file. Two handles for
/// different generations of a file must never return each other's pages.
/// Every method is called from decode threads and must not block on I/O.
pub trait PageCache: Debug + Send + Sync {
    /// Looks up the page that starts at `offset` and records the access.
    ///
    /// With `fill`, a miss can carry a ticket, and the call may wait for a
    /// reader that already holds the ticket of this page, so that one reader
    /// decompresses a page and the others share the result. Without `fill`
    /// the call neither waits nor returns a ticket.
    fn get(&self, offset: u64, fill: bool) -> PageLookup {
        self.get_with_accounting(offset, fill, |page| !page.is_dictionary_marker())
    }

    /// Looks up a resident entry and records a read only if `count_read`
    /// accepts that same entry. Apply the predicate and accounting together,
    /// before a concurrent fill or eviction can replace the entry.
    ///
    /// A rejected read still returns the resident entry. In particular, a
    /// dictionary marker can cover its file range without counting as a read.
    fn get_with_accounting(
        &self,
        offset: u64,
        fill: bool,
        count_read: fn(&CachedPage) -> bool,
    ) -> PageLookup;

    /// Returns the page that starts at `offset` if it is resident, without
    /// recording an access. Used to step over a page that is not read.
    fn probe(&self, offset: u64) -> Option<CachedPage>;

    /// Leases the pages of each set of byte ranges whose pages are all
    /// resident, and nothing of a set that has a page missing. Returns `None`
    /// when no set is leased. This method never waits.
    ///
    /// Each range is a column chunk, a page or a run of pages and starts at
    /// a page header. Its pages are the resident pages that cover it without
    /// a gap, each followed to the next by its stored length; they must end
    /// exactly at the end of the range. A range from an offset index (one
    /// page) is one lookup, and the stored length of its page must be the
    /// length of its location.
    ///
    /// A cache that supports leases takes every set under one acquisition of
    /// its lock and gives the lease a [`LeaseRelease`]. Until the lease drops,
    /// `get` and `probe` must still find its pages, including after
    /// reclamation or a clear, and the engine keeps their memory charge.
    /// Taking the lease records no access: the release records the reads
    /// that page readers marked, so a leased page that is not read (a unit
    /// that a filter removed, a cancelled or dropped request, a page that the
    /// selection skips) records none.
    ///
    /// The default holds the pages without pins. It walks each set with
    /// `probe`, which records nothing, then takes each page of a set that
    /// tiles with one `get`, which records reads of payload pages at once (best effort:
    /// the read of a page that is not then read by a decoder is counted). A
    /// set whose page `get` misses is not leased: it supplies only pages
    /// that the accounting lookup returned. The cache can evict a page that
    /// the lease holds. An external prefetcher must then fetch the encoded
    /// bytes instead: it asks [`Self::holds_leases`] first.
    fn lease(&self, sets: &[Vec<Range<u64>>]) -> Option<PageLease> {
        let mut any = false;
        let sets = sets
            .iter()
            .map(|ranges| {
                let mut offsets = vec![];
                let tiled = ranges.iter().all(|range| {
                    tile(range.clone(), &mut offsets, |offset| {
                        let len = self.probe(offset)?.stored_len();
                        Some((offset, len))
                    })
                });
                if !tiled {
                    return None;
                }
                let pages = offsets
                    .into_iter()
                    .map(|offset| match self.get(offset, false) {
                        PageLookup::Hit(page) => Some((offset, page)),
                        PageLookup::Miss(_) => None,
                    })
                    .collect::<Option<Vec<_>>>()?;
                any = true;
                Some(pages)
            })
            .collect();
        any.then(|| PageLease::new(sets, None))
    }

    /// Whether [`Self::lease`] keeps its pages resident and charged and
    /// records their reads at release ([`PageLease::is_charged`]). A
    /// prefetcher that skips the bytes of a resident range asks this before
    /// it leases, so that a cache without lease support counts no read.
    fn holds_leases(&self) -> bool {
        false
    }
}

/// Whether the pages of a column chunk can be in the page cache: the chunk
/// is compressed and not encrypted, and its reader does not decode page
/// statistics (`ReaderProperties::read_page_stats`). The page reader and the
/// push decoder's fetch planning both decide with this function.
///
/// The buffer of an uncompressed page is a slice of the fetched chunk, so
/// caching it would keep the chunk alive and save no decompression. The
/// decrypted pages of an encrypted chunk must not reach memory that other
/// readers share. A cached page has no statistics.
pub(crate) fn page_cache_applies(column: &ColumnChunkMetaData, read_page_stats: bool) -> bool {
    #[cfg(feature = "encryption")]
    if column.crypto_metadata().is_some() {
        return false;
    }
    column.compression() != Compression::UNCOMPRESSED && !read_page_stats
}

/// The pages of one column chunk that fetch planning took from a
/// [`PageLease`]. The reader of the chunk holds the lease, so an eviction
/// between planning and decoding cannot remove a page whose bytes were not
/// fetched.
pub(crate) struct PinnedPages {
    lease: Arc<PageLease>,
    /// The byte range of the column chunk: the pages of the lease in it.
    chunk: Range<u64>,
}

impl Debug for PinnedPages {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedPages")
            .field("chunk", &self.chunk)
            .field("lease", &self.lease)
            .finish()
    }
}

impl PinnedPages {
    /// The pages of `lease` in the column chunk at byte range `chunk`.
    #[cfg(feature = "arrow")]
    pub(crate) fn new(lease: Arc<PageLease>, chunk: Range<u64>) -> Self {
        Self { lease, chunk }
    }

    /// The leased page at `offset`, without marking a read (a peek, a skip
    /// or a check).
    pub(crate) fn get(&self, offset: u64) -> Option<&CachedPage> {
        self.leased(offset).map(LeasedPage::page)
    }

    /// The leased page at `offset` for a read: marks one read in the lease,
    /// which its release records.
    pub(crate) fn read(&self, offset: u64) -> Option<&CachedPage> {
        let page = self.leased(offset)?;
        page.mark();
        Some(page.page())
    }

    fn leased(&self, offset: u64) -> Option<&LeasedPage> {
        if !self.chunk.contains(&offset) {
            return None;
        }
        self.lease.page(offset)
    }
}

#[cfg(test)]
pub(crate) mod testing {
    //! An unbounded map that counts its calls, for the tests of this crate.
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicUsize;

    #[derive(Debug, Default)]
    pub(crate) struct MapPageCache {
        pages: Arc<Mutex<HashMap<u64, CachedPage>>>,
        pub(crate) hits: Arc<AtomicUsize>,
        pub(crate) misses: AtomicUsize,
        pub(crate) fills: Arc<AtomicUsize>,
        /// With leases, `lease` pins its pages, and `remove` and `clear` keep
        /// the leased pages until the last lease is dropped. Without, `lease`
        /// holds the pages as the trait's default does.
        pub(crate) leases: Option<Arc<Mutex<HashMap<u64, usize>>>>,
        /// Leases that pinned pages.
        pub(crate) pins: AtomicUsize,
        /// Reads that released leases recorded.
        pub(crate) lease_reads: Arc<AtomicUsize>,
        /// Acquisitions of the map's lock by `get`, `probe`, `lease` and a
        /// lease's release: what an engine's lock would see.
        pub(crate) calls: Arc<AtomicUsize>,
    }

    impl MapPageCache {
        pub(crate) fn with_leases() -> Self {
            Self {
                leases: Some(Default::default()),
                ..Default::default()
            }
        }
        pub(crate) fn len(&self) -> usize {
            self.pages.lock().unwrap().len()
        }
        fn leased(&self, offset: u64) -> bool {
            self.leases
                .as_ref()
                .is_some_and(|leases| leases.lock().unwrap().contains_key(&offset))
        }
        pub(crate) fn clear(&self) {
            self.pages
                .lock()
                .unwrap()
                .retain(|offset, _| self.leased(*offset));
        }
        pub(crate) fn remove(&self, offset: u64) -> Option<CachedPage> {
            if self.leased(offset) {
                return None;
            }
            self.pages.lock().unwrap().remove(&offset)
        }
        /// Replaces the page at `offset`, as a faulty engine could.
        pub(crate) fn insert(&self, offset: u64, page: CachedPage) {
            self.pages.lock().unwrap().insert(offset, page);
        }
        pub(crate) fn offsets(&self) -> Vec<u64> {
            let mut offsets: Vec<_> = self.pages.lock().unwrap().keys().copied().collect();
            offsets.sort_unstable();
            offsets
        }
        /// The number of pages that a live lease holds.
        pub(crate) fn leased_pages(&self) -> usize {
            self.leases
                .as_ref()
                .map_or(0, |leases| leases.lock().unwrap().len())
        }
    }

    struct Fill {
        offset: u64,
        pages: Arc<Mutex<HashMap<u64, CachedPage>>>,
        fills: Arc<AtomicUsize>,
    }

    impl PageFill for Fill {
        fn complete(self: Box<Self>, page: CachedPage) {
            self.fills.fetch_add(1, Ordering::Relaxed);
            self.pages.lock().unwrap().insert(self.offset, page);
        }

        fn complete_with_validity(
            self: Box<Self>,
            page: CachedPage,
            extra_bytes: usize,
            build: fn(CachedPage) -> CachedPage,
        ) {
            // Test fixtures permit bounded builds. Fail before allocating if
            // an oversized page ever makes it past the preflight.
            assert!(extra_bytes < 3 * 1024 * 1024);
            self.complete(build(page));
        }
    }

    struct Release {
        leases: Arc<Mutex<HashMap<u64, usize>>>,
        hits: Arc<AtomicUsize>,
        reads: Arc<AtomicUsize>,
        calls: Arc<AtomicUsize>,
    }

    impl LeaseRelease for Release {
        fn release(self: Box<Self>, pages: &[LeasedPage]) {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let mut leases = self.leases.lock().unwrap();
            for page in pages {
                let count = leases.get_mut(&page.offset()).unwrap();
                *count -= 1;
                if *count == 0 {
                    leases.remove(&page.offset());
                }
                self.reads
                    .fetch_add(page.reads() as usize, Ordering::Relaxed);
                self.hits
                    .fetch_add(page.reads() as usize, Ordering::Relaxed);
            }
        }
    }

    impl PageCache for MapPageCache {
        fn get_with_accounting(
            &self,
            offset: u64,
            fill: bool,
            count_read: fn(&CachedPage) -> bool,
        ) -> PageLookup {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if let Some(page) = self.pages.lock().unwrap().get(&offset) {
                if count_read(page) {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                }
                return PageLookup::Hit(page.clone());
            }
            self.misses.fetch_add(1, Ordering::Relaxed);
            PageLookup::Miss(fill.then(|| {
                Box::new(Fill {
                    offset,
                    pages: Arc::clone(&self.pages),
                    fills: Arc::clone(&self.fills),
                }) as Box<dyn PageFill>
            }))
        }

        fn probe(&self, offset: u64) -> Option<CachedPage> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.pages.lock().unwrap().get(&offset).cloned()
        }

        fn holds_leases(&self) -> bool {
            self.leases.is_some()
        }

        /// One lock for the request; a lease records its reads when it
        /// drops (counted in `hits` and `lease_reads`).
        fn lease(&self, sets: &[Vec<Range<u64>>]) -> Option<PageLease> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let Some(leases) = self.leases.as_ref() else {
                // As the default: hold the pages, recording each read of a
                // set that tiles now.
                let pages = self.pages.lock().unwrap();
                let sets: Vec<_> = sets
                    .iter()
                    .map(|ranges| {
                        let mut taken = vec![];
                        ranges
                            .iter()
                            .all(|range| {
                                tile(range.clone(), &mut taken, |offset| {
                                    let page = pages.get(&offset)?.clone();
                                    let len = page.stored_len();
                                    Some(((offset, page), len))
                                })
                            })
                            .then_some(taken)
                    })
                    .collect();
                let reads = sets
                    .iter()
                    .flatten()
                    .flatten()
                    .filter(|(_, page)| !page.is_dictionary_marker())
                    .count();
                self.hits.fetch_add(reads, Ordering::Relaxed);
                return sets
                    .iter()
                    .any(Option::is_some)
                    .then(|| PageLease::new(sets, None));
            };
            let pages = self.pages.lock().unwrap();
            let mut counts = leases.lock().unwrap();
            let sets: Vec<_> = sets
                .iter()
                .map(|ranges| {
                    let mut taken = vec![];
                    let tiled = ranges.iter().all(|range| {
                        tile(range.clone(), &mut taken, |offset| {
                            let page = pages.get(&offset)?.clone();
                            let len = page.stored_len();
                            Some(((offset, page), len))
                        })
                    });
                    if !tiled {
                        return None;
                    }
                    for (offset, _) in &taken {
                        *counts.entry(*offset).or_default() += 1;
                    }
                    Some(taken)
                })
                .collect();
            if sets.iter().all(Option::is_none) {
                return None;
            }
            self.pins.fetch_add(1, Ordering::Relaxed);
            Some(PageLease::new(
                sets,
                Some(Box::new(Release {
                    leases: Arc::clone(leases),
                    hits: Arc::clone(&self.hits),
                    reads: Arc::clone(&self.lease_reads),
                    calls: Arc::clone(&self.calls),
                })),
            ))
        }
    }
}

#[cfg(all(test, feature = "arrow"))]
mod validity_admission_tests {
    use super::*;
    use crate::basic::Encoding;
    use crate::util::bit_util::BitWriter;

    #[test]
    fn oversized_validity_is_not_built() {
        // Fit millions of NULL levels into one short run.
        let levels = i32::MAX as u32;
        let mut run = BitWriter::new(8);
        run.put_vlq_int(u64::from(levels) * 2);
        run.put_value(0, 8);
        let run = run.consume();
        let mut buf = (run.len() as u32).to_le_bytes().to_vec();
        buf.extend_from_slice(&run);
        let page = CachedPage::new(
            Page::DataPage {
                buf: buf.into(),
                num_values: levels,
                encoding: Encoding::PLAIN,
                def_level_encoding: Encoding::RLE,
                rep_level_encoding: Encoding::RLE,
                statistics: None,
            },
            1,
            Compression::SNAPPY,
        );
        let encoded_len = page.buffer_len();
        use crate::column::reader::ColumnReaderImpl;
        use crate::data_type::Int32Type;
        use crate::schema::types::{ColumnDescriptor, ColumnPath, Type};
        use crate::util::test_common::page_util::InMemoryPageReader;

        let cache = testing::MapPageCache::with_leases();
        let PageLookup::Miss(Some(ticket)) = cache.get(0, true) else {
            panic!("expected fill ticket");
        };
        page.publish(ticket, true);
        let cached = cache.probe(0).unwrap();
        assert_eq!(cached.buffer_len(), encoded_len);

        let desc = Arc::new(ColumnDescriptor::new(
            Arc::new(
                Type::primitive_type_builder("i", crate::basic::Type::INT32)
                    .with_repetition(crate::basic::Repetition::OPTIONAL)
                    .build()
                    .unwrap(),
            ),
            1,
            0,
            ColumnPath::from("i"),
        ));
        let mut reader = ColumnReaderImpl::<Int32Type>::new(
            desc,
            Box::new(InMemoryPageReader::new(vec![cached.page().clone()])),
        );
        let mut values = Vec::new();
        let mut levels = Vec::new();
        assert_eq!(
            reader
                .read_records(17, Some(&mut levels), None, &mut values)
                .unwrap(),
            (17, 0, 17)
        );
        assert_eq!(levels, vec![0; 17]);
        assert!(values.is_empty());
    }
}
