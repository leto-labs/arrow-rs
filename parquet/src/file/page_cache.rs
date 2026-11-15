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

// Longbow modification notice: this file is new in the Longbow fork
// (family DPC, decompressed page cache).

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
//! header is parsed, the CRC is verified and the body is decompressed. Its
//! values keep their Parquet encoding. A hit therefore skips the file bytes,
//! the header, the checksum and the decompression, and nothing else.
//!
//! Only pages of compressed, unencrypted column chunks are cached. The buffer
//! of an uncompressed page is a slice of the fetched column chunk, so caching
//! it would keep the whole chunk alive and save no decompression.

use crate::column::page::{Page, PageMetadata};
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

/// One decompressed page and the number of bytes it occupies in the file.
#[derive(Clone, Debug)]
pub struct CachedPage {
    page: Page,
    stored_len: u64,
}

impl CachedPage {
    /// `stored_len` is the length of the page in the file: its header and its
    /// stored (compressed) body. The next page of the column chunk starts
    /// `stored_len` bytes after this one.
    pub fn new(page: Page, stored_len: u64) -> Self {
        Self { page, stored_len }
    }

    /// The decompressed page.
    pub fn page(&self) -> &Page {
        &self.page
    }

    /// Length of the page in the file, header included.
    pub fn stored_len(&self) -> u64 {
        self.stored_len
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
}

/// The handle of a file's page cache as a field of its metadata. It takes no
/// part in the equality of two metadata values.
#[derive(Clone, Debug, Default)]
pub(crate) struct PageCacheSlot(pub(crate) Option<Arc<dyn PageCache>>);

impl PartialEq for PageCacheSlot {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

/// Pages of one column chunk that fetch planning took from the cache, by the
/// file offset of the page. The reader of the chunk holds them, so an eviction
/// between planning and decoding cannot remove a page whose bytes were not
/// fetched.
#[derive(Debug, Default)]
pub(crate) struct PinnedPages {
    pages: HashMap<u64, CachedPage>,
}

impl PinnedPages {
    pub(crate) fn get(&self, offset: u64) -> Option<&CachedPage> {
        self.pages.get(&offset)
    }

    /// Pins the pages that start at `offsets`. Returns `None` unless every
    /// one is resident.
    pub(crate) fn at_offsets(
        cache: &dyn PageCache,
        offsets: impl IntoIterator<Item = u64>,
    ) -> Option<Self> {
        let mut pages = HashMap::new();
        for offset in offsets {
            match cache.get(offset, false) {
                PageLookup::Hit(page) => pages.insert(offset, page),
                PageLookup::Miss(_) => return None,
            };
        }
        Some(Self { pages })
    }

    /// Pins every page of the byte range `start..end` of a column chunk that
    /// has no offset index, following each page's stored length to the next
    /// one. Returns `None` unless the resident pages cover the range exactly.
    pub(crate) fn contiguous(cache: &dyn PageCache, start: u64, end: u64) -> Option<Self> {
        let mut pages = HashMap::new();
        let mut offset = start;
        while offset < end {
            let PageLookup::Hit(page) = cache.get(offset, false) else {
                return None;
            };
            let next = offset.checked_add(page.stored_len())?;
            if page.stored_len() == 0 || next > end {
                return None;
            }
            pages.insert(offset, page);
            offset = next;
        }
        Some(Self { pages })
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
    }

    impl MapPageCache {
        pub(crate) fn len(&self) -> usize {
            self.pages.lock().unwrap().len()
        }
        pub(crate) fn clear(&self) {
            self.pages.lock().unwrap().clear();
        }
        pub(crate) fn remove(&self, offset: u64) -> Option<CachedPage> {
            self.pages.lock().unwrap().remove(&offset)
        }
        pub(crate) fn offsets(&self) -> Vec<u64> {
            let mut offsets: Vec<_> = self.pages.lock().unwrap().keys().copied().collect();
            offsets.sort_unstable();
            offsets
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
    }
}
