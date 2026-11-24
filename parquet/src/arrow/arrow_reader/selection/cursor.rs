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

//! Execution time iteration over a [`RowSelection`].
//!
//! A [`ReadPlan`](crate::arrow::arrow_reader::ReadPlan) resolves a
//! [`RowSelectionPolicy`] into a [`RowSelectionStrategy`] and builds the
//! matching [`RowSelectionCursor`], which keeps the per-reader position while
//! the selection itself stays immutable.

use super::{MaskRunIter, RowSelection, RowSelector};
use crate::errors::ParquetError;
use arrow_array::BooleanArray;
use arrow_buffer::BooleanBuffer;
use arrow_buffer::bit_chunk_iterator::BitChunks;
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;

#[cfg(test)]
std::thread_local! {
    /// Run boundary discoveries of mask-backed selector cursors, to check the
    /// work without timing.
    static MASK_RUN_DISCOVERIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Policy for picking a strategy to materialize [`RowSelection`] during execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RowSelectionPolicy {
    /// Use a queue of [`RowSelector`] values
    Selectors,
    /// Use a boolean mask to materialize the selection
    Mask,
    /// Choose between [`Self::Mask`] and [`Self::Selectors`] based on selector density
    Auto {
        /// Average selector length below which masks are preferred
        threshold: usize,
    },
}

impl Default for RowSelectionPolicy {
    fn default() -> Self {
        Self::Auto { threshold: 32 }
    }
}

impl RowSelectionPolicy {
    /// Resolve this policy for a selection without changing its backing.
    pub(crate) fn resolve(self, selection: Option<&RowSelection>) -> RowSelectionStrategy {
        match self {
            Self::Selectors => RowSelectionStrategy::Selectors,
            Self::Mask => RowSelectionStrategy::Mask,
            Self::Auto { threshold } => selection
                .map(|selection| selection.auto_selection_strategy(threshold))
                .unwrap_or(RowSelectionStrategy::Selectors),
        }
    }

    /// Materialize a selection according to this policy, preserving its full
    /// length, including trailing skips needed for selection composition.
    pub(crate) fn apply(self, selection: RowSelection) -> RowSelection {
        match (
            self.resolve(Some(&selection)),
            selection.as_mask().is_some(),
        ) {
            (RowSelectionStrategy::Mask, true) | (RowSelectionStrategy::Selectors, false) => {
                selection
            }
            (RowSelectionStrategy::Mask, false) => selection.force_mask(),
            (RowSelectionStrategy::Selectors, true) => {
                RowSelection::from(Vec::<RowSelector>::from(selection))
            }
        }
    }
}

/// Fully resolved strategy for materializing [`RowSelection`] during execution.
///
/// This is determined by [`RowSelectionPolicy`], including selector density for
/// [`RowSelectionPolicy::Auto`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RowSelectionStrategy {
    /// Use a queue of [`RowSelector`] values
    Selectors,
    /// Use a boolean mask to materialize the selection
    Mask,
}

impl RowSelectionStrategy {
    /// Build a cursor using this strategy. The selection must be trimmed of
    /// trailing skips before constructing a mask cursor.
    pub(crate) fn build_cursor(
        self,
        selection: RowSelection,
        loaded_row_ranges: Option<Arc<LoadedRowRanges>>,
    ) -> RowSelectionCursor {
        match self {
            Self::Mask => RowSelectionCursor::new_mask_from_buffer(
                selection.into_boolean_buffer(),
                loaded_row_ranges,
            ),
            // A mask-backed selection keeps its bitmap: the cursor streams
            // its runs instead of building the selector queue up front.
            Self::Selectors if selection.as_mask().is_some() => {
                RowSelectionCursor::new_mask_selectors(selection.into_boolean_buffer())
            }
            Self::Selectors => RowSelectionCursor::new_selectors(selection.into()),
        }
    }
}

/// Cursor for iterating a [`RowSelection`] during execution within a
/// [`ReadPlan`](crate::arrow::arrow_reader::ReadPlan).
///
/// This keeps per-reader state such as the current position and delegates the
/// actual storage strategy to the internal `RowSelectionInner`.
#[derive(Debug)]
pub enum RowSelectionCursor {
    /// Reading all rows
    All,
    /// Use a bitmask to back the selection (dense selections)
    Mask(MaskCursor),
    /// Use a queue of selectors to back the selection (sparse selections)
    Selectors(SelectorsCursor),
}

impl RowSelectionCursor {
    /// Create a [`MaskCursor`] cursor backed by an existing bitmask.
    pub(crate) fn new_mask_from_buffer(
        mask: BooleanBuffer,
        loaded_row_ranges: Option<Arc<LoadedRowRanges>>,
    ) -> Self {
        debug_assert!(
            mask.is_empty() || mask.value(mask.len() - 1),
            "Mask selections must not end with a skip"
        );
        Self::Mask(MaskCursor {
            mask,
            position: 0,
            loaded_row_ranges,
        })
    }

    /// Create a [`RowSelectionCursor::Selectors`] from the provided selectors
    pub(crate) fn new_selectors(selectors: Vec<RowSelector>) -> Self {
        Self::Selectors(SelectorsCursor {
            selectors: SelectorSource::Queue(selectors.into()),
            position: 0,
            known_run_end: 0,
        })
    }

    /// Create a [`RowSelectionCursor::Selectors`] that streams the runs of a
    /// bitmap, which must not end with a skip.
    pub(crate) fn new_mask_selectors(mask: BooleanBuffer) -> Self {
        Self::Selectors(SelectorsCursor {
            selectors: SelectorSource::Mask(mask),
            position: 0,
            known_run_end: 0,
        })
    }

    /// Create a cursor that selects all rows
    pub(crate) fn new_all() -> Self {
        Self::All
    }
}

/// The runs of a [`SelectorsCursor`]: a queue of selectors, or the bitmap of
/// a mask-backed selection, whose runs are found as the cursor reaches them.
#[derive(Debug)]
enum SelectorSource {
    Queue(VecDeque<RowSelector>),
    Mask(BooleanBuffer),
}

#[cfg(test)]
impl SelectorSource {
    /// Selectors that the source holds memory for.
    fn capacity(&self) -> usize {
        match self {
            Self::Queue(queue) => queue.capacity(),
            Self::Mask(_) => 0,
        }
    }
}

/// Cursor for iterating a [`RowSelection`] as selected and skipped runs.
///
/// This is best for sparse selections where large contiguous
/// blocks of rows are selected or skipped. A mask-backed selection keeps its
/// bitmap, and the cursor finds each run when it reaches it, so the cursor
/// holds no selector queue.
#[derive(Debug)]
pub struct SelectorsCursor {
    selectors: SelectorSource,
    /// Current absolute offset into the selection
    position: usize,
    /// For a bitmap: the end of the run that contains `position`, once found.
    /// A run that the reader returns in part is not searched again.
    known_run_end: usize,
}

impl SelectorsCursor {
    /// Returns `true` when no further rows remain
    pub fn is_empty(&self) -> bool {
        match &self.selectors {
            SelectorSource::Mask(mask) => self.position >= mask.len(),
            SelectorSource::Queue(queue) => queue.is_empty(),
        }
    }

    /// The selected rows from the current position to the end.
    pub(crate) fn selected_rows(&self) -> usize {
        match &self.selectors {
            SelectorSource::Mask(mask) => mask
                .slice(self.position, mask.len().saturating_sub(self.position))
                .count_set_bits(),
            SelectorSource::Queue(queue) => queue
                .iter()
                .filter(|selector| !selector.skip)
                .map(|selector| selector.row_count)
                .sum(),
        }
    }

    /// Return the next [`RowSelector`]
    pub(crate) fn next_selector(&mut self) -> RowSelector {
        let selector = match &mut self.selectors {
            SelectorSource::Mask(mask) => {
                if self.position < self.known_run_end {
                    RowSelector {
                        row_count: self.known_run_end - self.position,
                        skip: !mask.value(self.position),
                    }
                } else {
                    let remaining = mask.slice(self.position, mask.len() - self.position);
                    #[cfg(test)]
                    MASK_RUN_DISCOVERIES.with(|count| count.set(count.get() + 1));
                    let run = MaskRunIter::new(&remaining).next().unwrap();
                    self.known_run_end = self.position + run.row_count;
                    run
                }
            }
            SelectorSource::Queue(queue) => queue.pop_front().unwrap(),
        };
        self.position += selector.row_count;
        selector
    }

    /// Return a selector to the front, rewinding the position
    pub(crate) fn return_selector(&mut self, selector: RowSelector) {
        self.position = self.position.saturating_sub(selector.row_count);
        // A bitmap rewinds within the run that it found: the next call
        // returns the rest of that run without searching it again.
        if let SelectorSource::Queue(queue) = &mut self.selectors {
            queue.push_front(selector);
        }
    }
}

/// Cursor for iterating a mask-backed [`RowSelection`]
///
/// This is best for dense selections where there are many small skips
/// or selections. For example, selecting every other row.
///
/// When page pruning produces sparse column data, `loaded_row_ranges` limits
/// each decoded chunk to rows whose pages are loaded for every projected leaf.
/// For example, two projected columns can have different page boundaries:
///
/// ```text
/// Row ranges:       [0, 4) [4, 6) [6, 8) [8, 10) [10, 12)
/// Selection mask:   1000   00     00     00      01
/// Column A pages:   loaded | missing [4, 8) | loaded [8, 12)
/// Column B pages:   loaded [0, 6) | missing [6, 10) | loaded
/// LoadedRowRanges:  [0, 4)                         [10, 12)
/// ```
///
/// The first chunk decodes `[0, 1)` with mask `1`. The next chunk skips to row
/// 11 and decodes `[11, 12)` with mask `1`. When loaded ranges are present,
/// every returned chunk ends at a selected row and never includes trailing
/// unselected rows. [`ParquetRecordBatchReader`] still accumulates both chunks
/// and applies the combined mask `11` once.
///
/// [`ParquetRecordBatchReader`]: crate::arrow::arrow_reader::ParquetRecordBatchReader
#[derive(Debug)]
pub struct MaskCursor {
    mask: BooleanBuffer,
    /// Current absolute offset into the selection
    position: usize,
    /// Row ranges whose backing pages are loaded for every projected column.
    loaded_row_ranges: Option<Arc<LoadedRowRanges>>,
}

impl MaskCursor {
    /// Returns `true` when no further rows remain
    pub fn is_empty(&self) -> bool {
        self.position >= self.mask.len()
    }

    /// Advance through the mask representation, producing the next chunk summary
    pub fn next_mask_chunk(&mut self, batch_size: usize) -> Option<MaskChunk> {
        if self.is_empty() {
            return None;
        }

        Some(self.next_mask_chunk_non_empty(batch_size))
    }

    /// Produces the next chunk for a non-empty, trailing-skip-free mask.
    fn next_mask_chunk_non_empty(&mut self, batch_size: usize) -> MaskChunk {
        debug_assert!(!self.is_empty());
        let mask_start = self.next_selected_row().unwrap_or(self.mask.len());
        debug_assert!(
            mask_start < self.mask.len(),
            "ReadPlan must remove trailing skips from Mask selections"
        );
        self.scan_chunk(mask_start, self.mask.len(), batch_size)
    }

    /// Find the first selected row without inspecting every skipped bit.
    fn next_selected_row(&self) -> Option<usize> {
        let chunks = BitChunks::new(
            self.mask.values(),
            self.mask.offset() + self.position,
            self.mask.len() - self.position,
        );
        chunks.iter_padded().enumerate().find_map(|(idx, word)| {
            (word != 0).then(|| self.position + idx * 64 + word.trailing_zeros() as usize)
        })
    }

    /// Count whole words until the batch is full or the loaded range ends.
    /// `BitChunks` normalizes words to little-endian and masks the remainder,
    /// so padding and bits in an unloaded page cannot contribute to the count.
    // Keep the scanning loop out of the reader's selection dispatch.
    #[inline(never)]
    fn scan_chunk(&mut self, mask_start: usize, end: usize, batch_size: usize) -> MaskChunk {
        let chunks = BitChunks::new(
            self.mask.values(),
            self.mask.offset() + mask_start,
            end - mask_start,
        );
        let mut selected_rows = 0;
        let mut chunk_end = mask_start;

        if batch_size > 0 {
            for (idx, mut word) in chunks.iter_padded().enumerate() {
                let count = word.count_ones() as usize;
                if count == 0 {
                    continue;
                }

                let remaining = batch_size - selected_rows;
                let last_bit = if count <= remaining {
                    selected_rows += count;
                    63 - word.leading_zeros() as usize
                } else {
                    // Only the final word needs individual set-bit selection.
                    for _ in 1..remaining {
                        word &= word - 1;
                    }
                    selected_rows = batch_size;
                    word.trailing_zeros() as usize
                };
                chunk_end = mask_start + idx * 64 + last_bit + 1;
                if selected_rows == batch_size {
                    break;
                }
            }
        }

        let initial_skip = mask_start - self.position;
        // Stop immediately after the last selected row. Unset bits at the
        // end of a loaded range belong to the next chunk's initial skip.
        self.position = chunk_end;
        MaskChunk {
            initial_skip,
            chunk_rows: chunk_end - mask_start,
            selected_rows,
            mask_start,
        }
    }

    /// Returns the next non-empty mask chunk without crossing an unloaded row range.
    /// When loaded ranges are present, every returned chunk ends immediately after a
    /// selected row and therefore never contains trailing unselected rows. Those rows
    /// remain for the next call's initial skip.
    ///
    /// The [`ReadPlan`](crate::arrow::arrow_reader::ReadPlan) removes trailing
    /// skips before constructing this cursor. Callers therefore only invoke
    /// this method for a non-empty mask that has another selected row.
    pub(crate) fn next_chunk(&mut self, batch_size: usize) -> Result<MaskChunk, ParquetError> {
        debug_assert!(batch_size > 0);
        debug_assert!(!self.is_empty());

        if self.loaded_row_ranges.is_none() {
            return Ok(self.next_mask_chunk_non_empty(batch_size));
        }

        let mask_start = self.next_selected_row().ok_or_else(|| {
            ParquetError::General(
                "Internal Error: Mask cursor reached the end without finding a selected row; \
                 ReadPlan must remove trailing skips"
                    .to_string(),
            )
        })?;

        let loaded_range_end = self
            .loaded_row_ranges
            .as_ref()
            .and_then(|ranges| ranges.end_containing(mask_start))
            .ok_or_else(|| {
                ParquetError::General(format!(
                    "Internal Error: selected row {mask_start} has no loaded page range"
                ))
            })?;

        Ok(self.scan_chunk(
            mask_start,
            loaded_range_end.min(self.mask.len()),
            batch_size,
        ))
    }

    /// Materialise the boolean values for a mask-backed chunk
    pub fn mask_values_for(&self, chunk: &MaskChunk) -> Result<BooleanArray, ParquetError> {
        if chunk.mask_start.saturating_add(chunk.chunk_rows) > self.mask.len() {
            return Err(ParquetError::General(
                "Internal Error: MaskChunk exceeds mask length".to_string(),
            ));
        }
        Ok(BooleanArray::from(
            self.mask.slice(chunk.mask_start, chunk.chunk_rows),
        ))
    }
}

/// Result of computing the next chunk to read when using a [`MaskCursor`]
#[derive(Debug)]
pub struct MaskChunk {
    /// Number of leading rows to skip before reaching selected rows
    pub initial_skip: usize,
    /// Total rows covered by this chunk (selected + skipped)
    pub chunk_rows: usize,
    /// Rows actually selected within the chunk
    pub selected_rows: usize,
    /// Starting offset within the mask where the chunk begins
    pub mask_start: usize,
}

/// Row ranges whose backing pages are loaded for every projected column.
#[derive(Clone, Debug)]
pub(crate) struct LoadedRowRanges(Vec<Range<usize>>);

impl LoadedRowRanges {
    pub(crate) fn from_selection(selection: RowSelection) -> Self {
        let selectors: Vec<RowSelector> = selection.into();
        let mut position = 0;
        let ranges = selectors
            .into_iter()
            .filter_map(|selector| {
                let start = position;
                position += selector.row_count;
                (!selector.skip).then_some(start..position)
            })
            .collect();
        Self(ranges)
    }

    fn end_containing(&self, row: usize) -> Option<usize> {
        let idx = self.0.partition_point(|range| range.end <= row);
        self.0
            .get(idx)
            .filter(|range| range.start <= row)
            .map(|range| range.end)
    }

    #[cfg(test)]
    pub(crate) fn ranges(&self) -> &[Range<usize>] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compare chunks with the selected row positions, independently of how
    /// the cursor scans the bitmap. A chunk may consume at most `batch_size`
    /// selected rows and may not leave the loaded range containing its first.
    fn assert_mask_chunks(mask: BooleanBuffer, loaded: Option<LoadedRowRanges>, batch_size: usize) {
        let selected: Vec<_> = (0..mask.len()).filter(|&row| mask.value(row)).collect();
        let loaded = loaded.map(Arc::new);
        let selection = RowSelection::from_boolean_buffer(mask).trim();
        let RowSelectionCursor::Mask(mut cursor) =
            RowSelectionStrategy::Mask.build_cursor(selection, loaded.clone())
        else {
            unreachable!()
        };
        let mut next = 0;
        let mut position = 0;
        while next < selected.len() {
            let first = selected[next];
            let end = loaded
                .as_ref()
                .map_or(usize::MAX, |ranges| ranges.end_containing(first).unwrap());
            let count = selected[next..]
                .iter()
                .take(batch_size)
                .take_while(|&&row| row < end)
                .count();
            let last = selected[next + count - 1];
            let chunk = cursor.next_chunk(batch_size).unwrap();
            assert_eq!(chunk.initial_skip, first - position);
            assert_eq!(chunk.mask_start, first);
            assert_eq!(chunk.chunk_rows, last + 1 - first);
            assert_eq!(chunk.selected_rows, count);
            assert_eq!(cursor.position, last + 1);
            let values = cursor.mask_values_for(&chunk).unwrap();
            assert_eq!(values.true_count(), count);
            assert!(values.value(0));
            assert!(values.value(values.len() - 1));
            next += count;
            position = last + 1;
        }
        assert!(cursor.is_empty());
        assert!(cursor.next_mask_chunk(batch_size).is_none());
    }

    #[test]
    fn test_mask_chunks_sparse_aligned_words() {
        // Distinct byte positions expose native-endian word interpretation
        // on big-endian platforms. Both selections must form one-row chunks.
        let mask = BooleanBuffer::from_iter((0..256).map(|row| row == 0 || row == 255));
        assert_mask_chunks(mask, None, 1);
    }

    #[test]
    fn test_mask_chunks_word_boundaries_and_offsets() {
        use rand::{RngExt, SeedableRng, rngs::StdRng};

        for len in [0, 1, 7, 8, 63, 64, 65, 127, 128, 129, 511, 2049] {
            for offset in [0, 1, 3, 7, 8, 31, 63, 64, 65] {
                for pattern in 0..7 {
                    let mut rng = StdRng::seed_from_u64(42);
                    let mask = BooleanBuffer::from_iter((0..offset + len + 64).map(|row| {
                        if !(offset..offset + len).contains(&row) {
                            // Set padding on both sides to catch reads outside
                            // the logical slice, including its final word.
                            return true;
                        }
                        let row = row - offset;
                        match pattern {
                            0 => false,
                            1 => true,
                            2 => row % 2 == 0,
                            3 => row % 129 == 128,
                            4 => rng.random_bool(0.1),
                            5 => rng.random_bool(0.5),
                            _ => rng.random_bool(0.9),
                        }
                    }))
                    .slice(offset, len);
                    for batch_size in [1, 2, 3, 63, 64, 65, 1024] {
                        assert_mask_chunks(mask.clone(), None, batch_size);
                    }
                }
            }
        }
    }

    #[test]
    fn test_mask_chunks_loaded_ranges_word_boundaries() {
        for page_rows in [1, 31, 63, 64, 65, 127, 257] {
            let len = page_rows * 7 + 13;
            let ranges: Vec<_> = (0..len)
                .step_by(page_rows)
                .enumerate()
                .filter(|(page, _)| page % 3 != 1)
                .map(|(_, start)| start..(start + page_rows).min(len))
                .collect();
            let loaded = LoadedRowRanges::from_selection(RowSelection::from_consecutive_ranges(
                ranges.iter().cloned(),
                len,
            ));
            for offset in [0, 1, 3, 7, 31, 63, 64, 65] {
                for pattern in 0..4 {
                    let mask = BooleanBuffer::from_iter((0..offset + len + 64).map(|row| {
                        if !(offset..offset + len).contains(&row) {
                            return true;
                        }
                        let row = row - offset;
                        ranges.iter().any(|range| range.contains(&row))
                            && match pattern {
                                0 => true,
                                1 => row % 2 == 0,
                                // Leave an unset tail in every loaded range.
                                2 => row % page_rows == 0,
                                // Select directly on either side of page and
                                // word boundaries, including the final page.
                                _ => {
                                    row % page_rows == page_rows - 1
                                        || row % 64 == 0
                                        || row % 64 == 63
                                }
                            }
                    }))
                    .slice(offset, len);
                    for batch_size in [1, 2, 3, 63, 64, 65, 1024] {
                        assert_mask_chunks(mask.clone(), Some(loaded.clone()), batch_size);
                    }
                }
            }
        }
    }

    #[test]
    fn test_mask_chunk_does_not_read_selected_rows_outside_loaded_range() {
        let mut cursor = MaskCursor {
            mask: BooleanBuffer::new_set(130),
            position: 0,
            loaded_row_ranges: Some(Arc::new(LoadedRowRanges(vec![0..65, 66..130]))),
        };
        let chunk = cursor.next_chunk(1024).unwrap();
        assert_eq!(chunk.chunk_rows, 65);
        assert_eq!(chunk.selected_rows, 65);
        assert_eq!(cursor.position, 65);
        let err = cursor.next_chunk(1024).unwrap_err();
        assert!(
            err.to_string()
                .contains("selected row 65 has no loaded page range")
        );
        assert_eq!(cursor.position, 65);
    }

    #[test]
    fn selection_respects_policy() {
        let mask = BooleanBuffer::from(vec![true, true, false, false, true, true, false, false]);
        for (policy, expect_mask) in [
            (RowSelectionPolicy::Mask, true),
            (RowSelectionPolicy::Selectors, false),
            (RowSelectionPolicy::Auto { threshold: 2 }, false),
            (RowSelectionPolicy::Auto { threshold: 3 }, true),
        ] {
            for selection in [
                RowSelection::from_boolean_buffer(mask.clone()),
                RowSelection::from(vec![
                    RowSelector::select(2),
                    RowSelector::skip(2),
                    RowSelector::select(2),
                    RowSelector::skip(2),
                ]),
            ] {
                let selection = policy.apply(selection);
                assert_eq!(selection.as_mask().is_some(), expect_mask, "{policy:?}");
                assert_eq!(selection.into_boolean_buffer(), mask);
            }
        }
    }

    #[test]
    fn adaptive_selection_uses_selectors_for_long_runs() {
        let mask = BooleanBuffer::from_iter((0..1_024).map(|idx| (256..768).contains(&idx)));
        let selection =
            RowSelectionPolicy::default().apply(RowSelection::from_boolean_buffer(mask));
        assert!(selection.as_mask().is_none());
    }

    #[test]
    fn adaptive_selection_keeps_fragmented_masks() {
        let mask = BooleanBuffer::from_iter((0..1_024).map(|idx| idx % 2 == 0));
        let selection =
            RowSelectionPolicy::default().apply(RowSelection::from_boolean_buffer(mask));
        assert!(selection.as_mask().is_some());
    }

    #[test]
    fn test_loaded_mask_chunk_stops_at_trimmed_mask_end() {
        let loaded = LoadedRowRanges::from_selection(RowSelection::from_consecutive_ranges(
            std::iter::once(0..5),
            10,
        ));
        let RowSelectionCursor::Mask(mut cursor) = RowSelectionStrategy::Mask.build_cursor(
            RowSelection::from(vec![RowSelector::select(1)]),
            Some(loaded.into()),
        ) else {
            unreachable!()
        };

        let chunk = cursor.next_chunk(10).unwrap();
        assert_eq!(chunk.chunk_rows, 1);
        assert!(cursor.is_empty());
    }

    #[test]
    fn test_next_mask_chunk_until_cursor_is_empty() {
        let RowSelectionCursor::Mask(mut cursor) = RowSelectionStrategy::Mask.build_cursor(
            RowSelection::from(vec![
                RowSelector::skip(2),
                RowSelector::select(2),
                RowSelector::skip(1),
                RowSelector::select(1),
            ]),
            None,
        ) else {
            unreachable!()
        };

        let first = cursor.next_mask_chunk(2).unwrap();
        assert_eq!(first.initial_skip, 2);
        assert_eq!(first.chunk_rows, 2);
        assert_eq!(first.selected_rows, 2);

        let second = cursor.next_mask_chunk(2).unwrap();
        assert_eq!(second.initial_skip, 1);
        assert_eq!(second.chunk_rows, 1);
        assert_eq!(second.selected_rows, 1);

        assert!(cursor.next_mask_chunk(2).is_none());
    }

    #[test]
    fn million_row_alternating_mask_streams_runs_without_run_vector() {
        use crate::arrow::arrow_reader::ReadPlanBuilder;
        let rows = 1_048_576;
        let mask = BooleanBuffer::collect_bool(rows, |i| i % 2 == 0);
        let mut plan = ReadPlanBuilder::new(8192)
            .with_selection(Some(RowSelection::from_boolean_buffer(mask)))
            .with_row_selection_policy(RowSelectionPolicy::Selectors)
            .build();
        let RowSelectionCursor::Selectors(cursor) = plan.row_selection_cursor_mut() else {
            panic!("expected selector execution");
        };
        // The bitmap is the selection's own storage; the cursor adds no
        // selector queue, also while runs are consumed.
        assert_eq!(cursor.selectors.capacity(), 0, "eager selector queue");
        let mut position = 0;
        while !cursor.is_empty() {
            let run = cursor.next_selector();
            assert_eq!(run.row_count, 1);
            assert_eq!(run.skip, position % 2 != 0);
            position += 1;
            assert_eq!(cursor.selectors.capacity(), 0, "run traversal allocated");
        }
        // The plan trims the final skipped row.
        assert_eq!(position, rows - 1);
    }

    #[test]
    fn mask_selector_partial_run_rewind_does_not_allocate() {
        use crate::arrow::arrow_reader::ReadPlanBuilder;
        let mut plan = ReadPlanBuilder::new(8192)
            .with_selection(Some(RowSelection::from_boolean_buffer(
                BooleanBuffer::collect_bool(32768, |i| i >= 7),
            )))
            .with_row_selection_policy(RowSelectionPolicy::Selectors)
            .build();
        let RowSelectionCursor::Selectors(cursor) = plan.row_selection_cursor_mut() else {
            panic!("expected selector execution");
        };
        assert_eq!(cursor.selectors.capacity(), 0);
        assert_eq!(cursor.next_selector(), RowSelector::skip(7));
        assert_eq!(cursor.next_selector(), RowSelector::select(32761));
        cursor.return_selector(RowSelector::select(32761 - 8192));
        assert_eq!(cursor.selectors.capacity(), 0);
        assert_eq!(cursor.next_selector(), RowSelector::select(32761 - 8192));
        assert!(cursor.is_empty());
    }

    #[test]
    fn mask_selectors_match_the_selector_queue() {
        // A sliced bitmap with a nonzero offset; the queue is the reference.
        let bits = BooleanBuffer::from(vec![false, true, true, false, false, true, true, true]);
        let mask = bits.slice(1, 7);
        let selectors: Vec<RowSelector> = MaskRunIter::new(&mask).collect();
        let mut from_mask = RowSelectionCursor::new_mask_selectors(mask);
        let mut from_queue = RowSelectionCursor::new_selectors(selectors);
        let (RowSelectionCursor::Selectors(a), RowSelectionCursor::Selectors(b)) =
            (&mut from_mask, &mut from_queue)
        else {
            unreachable!()
        };
        assert_eq!(a.next_selector(), b.next_selector());
        assert_eq!(a.next_selector(), b.next_selector());
        let third = a.next_selector();
        assert_eq!(third, b.next_selector());
        assert_eq!(third, RowSelector::select(3));
        // Return part of a run to both.
        a.return_selector(RowSelector::select(2));
        b.return_selector(RowSelector::select(2));
        assert_eq!(a.next_selector(), b.next_selector());
        assert!(a.is_empty() && b.is_empty());
    }

    #[test]
    fn long_mask_run_reuses_discovery_across_partial_batches() {
        use crate::arrow::arrow_reader::ReadPlanBuilder;
        const ROWS: usize = 1_048_576;
        const BATCH: usize = 8192;
        // A nonzero bit offset, and trailing skips that the plan trims.
        let bits = BooleanBuffer::collect_bool(ROWS + 14, |i| (5..ROWS + 5).contains(&i));
        let mut plan = ReadPlanBuilder::new(BATCH)
            .with_selection(Some(RowSelection::from_boolean_buffer(
                bits.slice(5, ROWS + 9),
            )))
            .with_row_selection_policy(RowSelectionPolicy::Selectors)
            .build();
        let RowSelectionCursor::Selectors(cursor) = plan.row_selection_cursor_mut() else {
            panic!("expected selector execution");
        };
        MASK_RUN_DISCOVERIES.with(|count| count.set(0));
        let mut remaining = ROWS;
        while remaining > 0 {
            assert_eq!(cursor.next_selector(), RowSelector::select(remaining));
            remaining -= remaining.min(BATCH);
            if remaining > 0 {
                cursor.return_selector(RowSelector::select(remaining));
            }
            // One search for the run, then only known remainders: N/B
            // partial batches must not search N + (N - B) + ... rows.
            assert_eq!(MASK_RUN_DISCOVERIES.with(|count| count.get()), 1);
            assert_eq!(cursor.selectors.capacity(), 0);
        }
        eprintln!(
            "Q_RUN_DISCOVERY rows={ROWS} batch={BATCH} partial_batches={} discoveries={}",
            ROWS.div_ceil(BATCH),
            MASK_RUN_DISCOVERIES.with(|count| count.get())
        );
        assert!(cursor.is_empty());
    }

    // Longbow Q: this characterization runs unchanged on stock and Q.
    #[test]
    fn q_resource_storage_characterization() {
        for rows in [1_024, 65_536, 1_048_576] {
            for run_length in [1, 257, rows] {
                let mask = BooleanBuffer::collect_bool(rows, |row| (row / run_length) % 2 == 0);
                let selection = RowSelection::from_boolean_buffer(mask).trim();
                let bits = selection.clone().into_boolean_buffer();
                let runs = super::super::MaskRunIter::new(&bits).count();
                let RowSelectionCursor::Selectors(mut cursor) =
                    RowSelectionStrategy::Selectors.build_cursor(selection, None)
                else {
                    unreachable!()
                };
                let mut peak_capacity = cursor.selectors.capacity();
                let mut seen = 0;
                while !cursor.is_empty() {
                    let selector = cursor.next_selector();
                    let take = selector.row_count.min(8192);
                    if !selector.skip {
                        seen += take;
                    }
                    if selector.row_count > take {
                        cursor.return_selector(RowSelector {
                            row_count: selector.row_count - take,
                            skip: selector.skip,
                        });
                    }
                    peak_capacity = peak_capacity.max(cursor.selectors.capacity());
                }
                assert_eq!(seen, bits.count_set_bits());
                eprintln!(
                    "Q_STORAGE rows={rows} run_length={run_length} runs={runs} selector_bytes={} cursor_bytes={} peak_queue_capacity={peak_capacity} peak_owner_bytes={}",
                    std::mem::size_of::<RowSelector>(),
                    std::mem::size_of_val(&cursor),
                    std::mem::size_of_val(&cursor)
                        + peak_capacity * std::mem::size_of::<RowSelector>()
                );
            }
        }
    }

    #[test]
    fn q_fragmented_cursor_owner_storage_is_bounded() {
        let mask = BooleanBuffer::collect_bool(1_048_576, |row| row % 2 == 0);
        let RowSelectionCursor::Selectors(cursor) = RowSelectionStrategy::Selectors
            .build_cursor(RowSelection::from_boolean_buffer(mask).trim(), None)
        else {
            unreachable!()
        };
        let owner_bytes = std::mem::size_of_val(&cursor)
            + cursor.selectors.capacity() * std::mem::size_of::<RowSelector>();
        assert!(
            owner_bytes <= 64 * 1024,
            "cursor owns {owner_bytes} bytes, exceeding the 64 KiB bound (the input bitmap is excluded)"
        );
    }

    #[test]
    fn q_selector_mask_differential() {
        fn selected_from_selectors(mut cursor: SelectorsCursor, batch: usize) -> Vec<usize> {
            let mut selected = Vec::new();
            let mut position = 0;
            while !cursor.is_empty() {
                let run = cursor.next_selector();
                let take = run.row_count.min(batch);
                if !run.skip {
                    selected.extend(position..position + take);
                }
                position += take;
                if take < run.row_count {
                    cursor.return_selector(RowSelector {
                        row_count: run.row_count - take,
                        skip: run.skip,
                    });
                }
                assert_eq!(cursor.position, position);
            }
            selected
        }
        let mut cases = 0;
        for len in [0, 1, 2, 63, 64, 65, 129, 2049, 16_385] {
            for offset in [0, 1, 7, 63, 64, 65] {
                for pattern in 0..7 {
                    let mask = BooleanBuffer::collect_bool(offset + len + 67, |i| {
                        if !(offset..offset + len).contains(&i) {
                            return true;
                        }
                        let row = i - offset;
                        match pattern {
                            0 => false,
                            1 => true,
                            2 => row % 2 == 0,
                            3 => row % 129 == 128,
                            4 => (7..len.saturating_sub(9)).contains(&row),
                            5 => {
                                (row.wrapping_mul(6364136223846793005_u64 as usize)
                                    .wrapping_add(17)
                                    >> 9)
                                    % 11
                                    < 3
                            }
                            _ => row % 257 < 191,
                        }
                    })
                    .slice(offset, len);
                    let expected: Vec<_> = (0..len).filter(|&row| mask.value(row)).collect();
                    for batch in [1, 2, 7, 63, 64, 65, 8192] {
                        let selection = RowSelection::from_boolean_buffer(mask.clone()).trim();
                        let RowSelectionCursor::Selectors(from_mask) =
                            RowSelectionStrategy::Selectors.build_cursor(selection.clone(), None)
                        else {
                            unreachable!()
                        };
                        let RowSelectionCursor::Selectors(from_queue) =
                            RowSelectionCursor::new_selectors(selection.clone().into())
                        else {
                            unreachable!()
                        };
                        let RowSelectionCursor::Mask(mut from_bits) =
                            RowSelectionStrategy::Mask.build_cursor(selection, None)
                        else {
                            unreachable!()
                        };
                        assert_eq!(
                            selected_from_selectors(from_mask, batch),
                            expected,
                            "mask selectors: len={len} offset={offset} pattern={pattern} batch={batch}"
                        );
                        assert_eq!(selected_from_selectors(from_queue, batch), expected);
                        let mut actual = Vec::new();
                        while let Some(chunk) = from_bits.next_mask_chunk(batch) {
                            let values = from_bits.mask_values_for(&chunk).unwrap();
                            assert!(chunk.selected_rows <= batch);
                            for row in 0..chunk.chunk_rows {
                                if values.value(row) {
                                    actual.push(chunk.mask_start + row);
                                }
                            }
                        }
                        assert_eq!(actual, expected);
                        cases += 1;
                    }
                }
            }
        }
        eprintln!(
            "Q_DIFFERENTIAL cases={cases} (mask-backed selectors, explicit queue, mask cursor, independent bit positions)"
        );
    }
}
