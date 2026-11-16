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
//! resident: its fetch planning pins them (see [`PageCache::pin`]) and the
//! page reader holds the pins until it is dropped. A
//! [`ScanPlan`](crate::arrow::push_decoder::ScanPlan) does not know the
//! cache and lists every range. A caller that fetches ahead from a scan plan
//! skips a resident range with [`resident_offsets`] and [`PageCache::pin`],
//! and holds the lease until the decoder has read the row group.

use crate::basic::Compression;
use crate::column::page::{Page, PageMetadata};
use crate::file::metadata::ColumnChunkMetaData;
use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

/// One decompressed page, the number of bytes it occupies in the file and
/// the codec that it was decompressed with.
#[derive(Clone, Debug)]
pub struct CachedPage {
    page: Page,
    stored_len: u64,
    codec: Compression,
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
        }
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

    /// Bytes that this entry keeps alive: the decompressed buffer.
    pub fn buffer_len(&self) -> usize {
        self.page.buffer().len()
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
}

/// A lease from [`PageCache::pin`]. The pages stay resident until the last
/// clone is dropped.
pub type PageLease = Arc<dyn Send + Sync>;

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
    fn get(&self, offset: u64, fill: bool) -> PageLookup;

    /// Returns the page that starts at `offset` if it is resident, without
    /// recording an access. Used to step over a page that is not read.
    fn probe(&self, offset: u64) -> Option<CachedPage>;

    /// Atomically leases all named pages, or returns `None` without a lease.
    /// Until the last lease is dropped, `get` and `probe` must still find these
    /// pages, including after reclamation or a clear. The embedding engine
    /// retains their memory charge. Acquiring a lease records no page access;
    /// the decoder's later `get` records its read. This method never waits.
    ///
    /// The default does not support leases. An external prefetcher must then
    /// fetch the encoded bytes instead of relying on residency probes alone.
    /// The decoder's own planning does not need leases: it holds the pages
    /// that it found.
    fn pin(&self, _offsets: &[u64]) -> Option<PageLease> {
        None
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

/// The file offsets of the resident pages that cover the byte range `range`
/// of a column chunk without a gap, following each page's stored length to
/// the next one. Returns `None` if a page in the range is not resident or the
/// pages do not end exactly at `range.end`.
///
/// `range` is a column chunk, a page, or a run of pages, for example the
/// range of a [`PlannedRange`](crate::arrow::push_decoder::PlannedRange). It
/// must start at a page header. This records no access and holds no page:
/// pass the offsets to [`PageCache::pin`] to keep the pages resident.
pub fn resident_offsets(cache: &dyn PageCache, range: Range<u64>) -> Option<Vec<u64>> {
    let mut offsets = vec![];
    let mut offset = range.start;
    while offset < range.end {
        let page = cache.probe(offset)?;
        let next = offset
            .checked_add(page.stored_len())
            .filter(|next| *next > offset && *next <= range.end)?;
        offsets.push(offset);
        offset = next;
    }
    Some(offsets)
}

/// Pages of one column chunk that fetch planning took from the cache, by the
/// file offset of the page. The reader of the chunk holds them, so an eviction
/// between planning and decoding cannot remove a page whose bytes were not
/// fetched.
#[derive(Default)]
pub(crate) struct PinnedPages {
    pages: HashMap<u64, CachedPage>,
    /// The engine's lease of these pages, if it supports leases: it keeps
    /// their memory charged while the reader holds them.
    lease: Option<PageLease>,
}

impl Debug for PinnedPages {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedPages")
            .field("pages", &self.pages.len())
            .field("leased", &self.lease.is_some())
            .finish()
    }
}

impl PinnedPages {
    pub(crate) fn get(&self, offset: u64) -> Option<&CachedPage> {
        self.pages.get(&offset)
    }

    /// Pins the pages that start at `offsets`. Returns `None` unless every
    /// one is resident.
    ///
    /// A lease, if the cache gives one, is taken first, so that no page can
    /// leave the cache between the lookups of one chunk. Each lookup records
    /// one access: the read of the page that the plan stands for.
    #[cfg(feature = "arrow")]
    pub(crate) fn at_offsets(cache: &dyn PageCache, offsets: &[u64]) -> Option<Self> {
        let lease = cache.pin(offsets);
        let mut pages = HashMap::with_capacity(offsets.len());
        for &offset in offsets {
            match cache.get(offset, false) {
                PageLookup::Hit(page) => pages.insert(offset, page),
                PageLookup::Miss(_) => return None,
            };
        }
        Some(Self { pages, lease })
    }

    /// Pins every page of the byte range `start..end` of a column chunk that
    /// has no offset index (see [`resident_offsets`]). Returns `None` unless
    /// the resident pages cover the range exactly.
    #[cfg(feature = "arrow")]
    pub(crate) fn contiguous(cache: &dyn PageCache, start: u64, end: u64) -> Option<Self> {
        let offsets = resident_offsets(cache, start..end)?;
        let pinned = Self::at_offsets(cache, &offsets)?;
        // A page can have been replaced between the probe and the pin: the
        // pinned pages must still cover the range.
        let mut offset = start;
        for &page_start in &offsets {
            let page = pinned.get(page_start)?;
            if page_start != offset {
                return None;
            }
            offset = page_start.checked_add(page.stored_len())?;
        }
        (offset == end).then_some(pinned)
    }
}

#[cfg(test)]
pub(crate) mod testing {
    //! An unbounded map that counts its calls, for the tests of this crate.
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug, Default)]
    pub(crate) struct MapPageCache {
        pages: Arc<Mutex<HashMap<u64, CachedPage>>>,
        pub(crate) hits: AtomicUsize,
        pub(crate) misses: AtomicUsize,
        pub(crate) fills: Arc<AtomicUsize>,
        /// With leases, `pin` returns a lease, and `remove` and `clear` keep
        /// the leased pages until the last lease is dropped.
        pub(crate) leases: Option<Arc<Mutex<HashMap<u64, usize>>>>,
        pub(crate) pins: AtomicUsize,
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
    }

    struct Lease {
        offsets: Vec<u64>,
        leases: Arc<Mutex<HashMap<u64, usize>>>,
    }

    impl Drop for Lease {
        fn drop(&mut self) {
            let mut leases = self.leases.lock().unwrap();
            for offset in &self.offsets {
                let count = leases.get_mut(offset).unwrap();
                *count -= 1;
                if *count == 0 {
                    leases.remove(offset);
                }
            }
        }
    }

    impl PageCache for MapPageCache {
        fn get(&self, offset: u64, fill: bool) -> PageLookup {
            if let Some(page) = self.pages.lock().unwrap().get(&offset) {
                self.hits.fetch_add(1, Ordering::Relaxed);
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
            self.pages.lock().unwrap().get(&offset).cloned()
        }

        fn pin(&self, offsets: &[u64]) -> Option<PageLease> {
            let leases = self.leases.as_ref()?;
            let pages = self.pages.lock().unwrap();
            if !offsets.iter().all(|offset| pages.contains_key(offset)) {
                return None;
            }
            let mut counts = leases.lock().unwrap();
            for offset in offsets {
                *counts.entry(*offset).or_default() += 1;
            }
            self.pins.fetch_add(1, Ordering::Relaxed);
            Some(Arc::new(Lease {
                offsets: offsets.to_vec(),
                leases: Arc::clone(leases),
            }))
        }
    }
}
