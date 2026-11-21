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

//! Apply a chunk verdict to selected data pages without materializing their values.

use super::{PredicateOptions, ReadPlanBuilder};
use crate::arrow::array_reader::{ArrayReaderBuilder, RowGroups};
use crate::arrow::arrow_reader::metrics::ArrowReaderMetrics;
use crate::arrow::arrow_reader::{
    ArrowPredicate, DictionaryFilterMetrics, DictionaryFilterMode, DictionaryFilterOptions,
    RowSelection,
};
use crate::arrow::schema::{ParquetField, ParquetFieldType};
use crate::basic::{Encoding, EncodingMask, PageType, Type};
use crate::column::page::{Page, PageIterator, PageMetadata, PageReader};
use crate::column::reader::decoder::code_page::{ChunkDictionary, FlatDataPage};
use crate::errors::{ParquetError, Result};
use crate::file::metadata::{ColumnChunkMetaData, ParquetMetaData, RowGroupMetaData};
use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder, Buffer};

impl ReadPlanBuilder {
    pub(crate) fn with_dictionary_predicate(
        self,
        options: PredicateOptions<'_>,
        groups: &dyn RowGroups,
        fields: Option<&ParquetField>,
    ) -> Result<Self> {
        let Some(policy) = options
            .predicate
            .dictionary_filter_options()
            .filter(|policy| policy.mode != DictionaryFilterMode::Off)
        else {
            return self.with_predicate_options(options);
        };
        let PredicateOptions {
            array_reader,
            predicate,
            limit,
            total_rows,
        } = options;
        let rows = groups.num_rows();
        // The verdict uses dense bitmaps over the span from the first to the
        // last selected row. A native unit can cover a small part of a much
        // larger row group, so the bitmaps never cover the whole group for
        // it. `SelectedSpan::of` refuses the verdict before it allocates if
        // the span exceeds four times the selected row count.
        let span = (limit.is_none() && predicate.dictionary_filter_eligible())
            .then(|| SelectedSpan::of(&self, rows))
            .flatten();
        let leaf = span
            .as_ref()
            .and_then(|_| flat_leaf(fields, groups.metadata(), predicate));
        let mut inconsistent = false;
        let code = match (span, leaf) {
            (Some(span), Some((index, field))) => {
                match evaluate_chunks(&span, groups, predicate, policy, index, field)? {
                    Some(code) => Some((span, code)),
                    None => {
                        inconsistent = true;
                        None
                    }
                }
            }
            _ => None,
        };

        let Some((span, code)) = code else {
            let chunks = count_selected_chunks(&self, groups);
            let several_leaves = (0..groups
                .metadata()
                .file_metadata()
                .schema_descr()
                .num_columns())
                .filter(|&i| predicate.projection().leaf_included(i))
                .nth(1)
                .is_some();
            let mut rows_value = 0;
            let result = self.apply_predicate(
                PredicateOptions {
                    array_reader,
                    predicate: &mut *predicate,
                    limit,
                    total_rows,
                },
                |rows| rows_value = rows,
            )?;
            let metrics = if inconsistent {
                // Page counts disagree somewhere in the unit. The ordinary
                // reader defines how such a unit reads.
                DictionaryFilterMetrics {
                    chunks_value: chunks,
                    rows_value,
                    fallback_error: chunks,
                    ..Default::default()
                }
            } else {
                DictionaryFilterMetrics {
                    chunks_value: chunks,
                    rows_value,
                    fallback_columns: if several_leaves { chunks } else { 0 },
                    fallback_unsupported: if several_leaves { 0 } else { chunks },
                    ..Default::default()
                }
            };
            predicate.record_dictionary_filter(metrics);
            return Ok(result);
        };
        let CodeSelection {
            covered,
            passed,
            mut metrics,
        } = code;
        let verify = policy.mode == DictionaryFilterMode::Verify && metrics.chunks_verdict != 0;
        let result = if verify {
            metrics.rows_verified = span.selected;
            let mut coverage = BooleanBufferBuilder::new(span.selected);
            span.append_selected(&covered, &mut coverage);
            let coverage = coverage.finish();
            let mut offset = 0;
            let oracle = self.clone().apply_predicate_with(
                PredicateOptions::new(array_reader, predicate),
                |predicate, batch| {
                    let count = batch.num_rows();
                    if count > coverage.len() - offset {
                        return Err(arrow_schema::ArrowError::ComputeError(
                            "Dictionary verification read past the selected span".into(),
                        ));
                    }
                    let batch_coverage = coverage.slice(offset, count);
                    offset += count;
                    predicate.evaluate_dictionary_oracle(batch, &batch_coverage)
                },
                |_| {},
            )?;
            if (&span.window(oracle.selection(), rows) & &covered) != passed {
                return Err(general_err!(
                    "Dictionary and value predicate selections differ"
                ));
            }
            oracle
        } else if metrics.chunks_verdict == 0 {
            self.with_predicate(array_reader, predicate)?
        } else {
            let mask = if metrics.rows_value == 0 {
                passed
            } else {
                let remaining = &span.mask & &!&covered;
                let fallback = self
                    .clone()
                    .with_selection(Some(span.restrict(&self, &remaining)))
                    .with_predicate(array_reader, predicate)?;
                &span.window(fallback.selection(), rows) | &passed
            };
            if mask.count_set_bits() == span.selected {
                self
            } else {
                let selection = span.restrict(&self, &mask);
                self.with_selection(Some(selection))
            }
        };
        predicate.record_dictionary_filter(metrics);
        Ok(result)
    }
}

/// The rows that a plan selects between its first and last selected row.
struct SelectedSpan {
    /// The first selected row in the reader's row range.
    start: usize,
    /// The number of rows from the first to the last selected row.
    len: usize,
    /// The number of selected rows.
    selected: usize,
    /// The selected rows of `start..start + len`.
    mask: BooleanBuffer,
}

impl SelectedSpan {
    /// Returns the selected span of `plan`, or `None` if it selects no rows
    /// or fewer than one row in four of its span. The check counts the
    /// selected runs and allocates nothing.
    fn of(plan: &ReadPlanBuilder, rows: usize) -> Option<Self> {
        let Some(selection) = plan.selection() else {
            return (rows != 0).then(|| Self {
                start: 0,
                len: rows,
                selected: rows,
                mask: BooleanBuffer::new_set(rows),
            });
        };
        let mut bounds = None::<(usize, usize)>;
        let mut selected = 0;
        for_each_selected_run(selection, rows, |run| {
            selected += run.len();
            let first = bounds.map_or(run.start, |(first, _)| first);
            bounds = Some((first, run.end));
        });
        let (start, end) = bounds?;
        let len = end - start;
        if selected < len.div_ceil(4) {
            return None;
        }
        let mask = window_mask(Some(selection), start, len, rows);
        debug_assert_eq!(mask.count_set_bits(), selected);
        Some(Self {
            start,
            len,
            selected,
            mask,
        })
    }

    /// Returns the rows that `selection` selects in this span.
    fn window(&self, selection: Option<&RowSelection>, rows: usize) -> BooleanBuffer {
        window_mask(selection, self.start, self.len, rows)
    }

    /// Returns the selection of `plan` limited to the rows set in `rows`, a
    /// mask over this span that sets only selected rows.
    fn restrict(&self, plan: &ReadPlanBuilder, rows: &BooleanBuffer) -> RowSelection {
        // Gather the bits of the selected rows only, as a predicate result
        // does, so the plan's own selection keeps its representation.
        let mut kept = BooleanBufferBuilder::new(self.selected);
        self.append_selected(rows, &mut kept);
        match plan.selection() {
            Some(selection) => {
                // A selection can extend past the reader's rows; it reads
                // none of them.
                kept.append_n(selection.row_count() - self.selected, false);
                selection.and_then(&RowSelection::from_boolean_buffer(kept.finish()))
            }
            None => RowSelection::from_boolean_buffer(kept.finish()),
        }
    }

    fn append_selected(&self, rows: &BooleanBuffer, kept: &mut BooleanBufferBuilder) {
        for (start, end) in self.mask.set_slices() {
            kept.append_buffer(&rows.slice(start, end - start));
        }
    }
}

/// Calls `f` with each selected run of rows in `0..rows`, in order.
fn for_each_selected_run(
    selection: &RowSelection,
    rows: usize,
    mut f: impl FnMut(std::ops::Range<usize>),
) {
    if let Some(mask) = selection.as_mask() {
        let mask = mask.slice(0, mask.len().min(rows));
        for (start, end) in mask.set_slices() {
            f(start..end);
        }
        return;
    }
    let mut offset = 0;
    for selector in selection.iter() {
        let end = (offset + selector.row_count).min(rows);
        if !selector.skip && end > offset {
            f(offset..end);
        }
        offset = end;
        if offset == rows {
            break;
        }
    }
}

/// Returns the rows that `selection` selects in `start..start + len`. Rows
/// past the end of the selection are not selected. This allocates only the
/// window, not the reader's whole row range.
fn window_mask(
    selection: Option<&RowSelection>,
    start: usize,
    len: usize,
    rows: usize,
) -> BooleanBuffer {
    let Some(selection) = selection else {
        return BooleanBuffer::new_set(len);
    };
    let end = start + len;
    if let Some(mask) = selection.as_mask()
        && mask.len() >= end
    {
        return mask.slice(start, len);
    }
    let mut window = BooleanBufferBuilder::new(len);
    for_each_selected_run(selection, rows.min(end), |run| {
        if run.end > start {
            let run_start = run.start.max(start);
            window.append_n(run_start - start - window.len(), false);
            window.append_n(run.end - run_start, true);
        }
    });
    window.append_n(len - window.len(), false);
    window.finish()
}

fn count_selected_chunks(plan: &ReadPlanBuilder, groups: &dyn RowGroups) -> usize {
    if groups.row_groups().nth(1).is_none() {
        return usize::from(plan.selects_any());
    }
    let Some(mut selected) = plan.selection().cloned() else {
        return groups.row_groups().count();
    };
    groups
        .row_groups()
        .filter(|group| selected.split_off(group.num_rows() as usize).selects_any())
        .count()
}

fn flat_leaf<'a>(
    fields: Option<&'a ParquetField>,
    metadata: &ParquetMetaData,
    predicate: &dyn ArrowPredicate,
) -> Option<(usize, &'a ParquetField)> {
    let schema = metadata.file_metadata().schema_descr();
    let mut leaves = (0..schema.num_columns()).filter(|&i| predicate.projection().leaf_included(i));
    let index = leaves.next()?;
    if leaves.next().is_some() {
        return None;
    }
    let descriptor = schema.column(index);
    if descriptor.path().parts().len() != 1
        || descriptor.max_rep_level() != 0
        || descriptor.max_def_level() > 1
    {
        return None;
    }
    let ParquetFieldType::Group { children } = &fields?.field_type else {
        return None;
    };
    children.iter().find(|field| {
        matches!(field.field_type, ParquetFieldType::Primitive { col_idx, .. } if col_idx == index)
    }).map(|field| (index, field))
}

struct CodeSelection {
    covered: BooleanBuffer,
    passed: BooleanBuffer,
    metrics: DictionaryFilterMetrics,
}

/// Evaluates the verdict for each chunk in the selected span. Returns `None`
/// if page counts disagree in any chunk that the ordinary reader positions
/// rows through. The ordinary reader then evaluates the whole unit, because
/// such a disagreement can shift the rows of later pages and row groups.
fn evaluate_chunks(
    span: &SelectedSpan,
    groups: &dyn RowGroups,
    predicate: &mut dyn ArrowPredicate,
    policy: DictionaryFilterOptions,
    leaf: usize,
    field: &ParquetField,
) -> Result<Option<CodeSelection>> {
    let mut chunks = groups.column_chunks(leaf)?;
    // Both bitmaps cover only the span of selected rows.
    let mut covered = BooleanBufferBuilder::new(span.len);
    let mut passed = BooleanBufferBuilder::new(span.len);
    let mut metrics = DictionaryFilterMetrics::default();
    let mut scratch = VerdictScratch {
        validity: Vec::new(),
        verdict: Vec::new(),
    };
    let span_end = span.start + span.len;
    let mut offset = 0;
    for row_group in groups.row_groups() {
        if offset >= span_end {
            // Later row groups cannot move selected rows.
            break;
        }
        let length = row_group.num_rows() as usize;
        let group_start = offset;
        offset += length;
        let pages = chunks
            .next()
            .ok_or_else(|| general_err!("Missing predicate column chunk"))?;
        // The part of this row group inside the span, in group rows.
        let window_start = span.start.max(group_start);
        let window_end = span_end.min(offset);
        let window = (window_start < window_end)
            .then(|| ChunkWindow {
                rows: length,
                start: window_start - group_start,
                selection: span
                    .mask
                    .slice(window_start - span.start, window_end - window_start),
            })
            .filter(|window| window.selection.count_set_bits() != 0);
        // Check the page counts of every chunk up to the end of the span
        // before using any verdict. A chunk before the span still positions
        // the rows after it.
        let Some(scanned) = scan_chunk(&mut *pages?, length, window.as_ref())? else {
            return Ok(None);
        };
        let Some(window) = window else {
            continue;
        };
        let selected_rows = window.selection.count_set_bits();
        let chunk_result = evaluate_chunk(
            scanned,
            groups,
            predicate,
            policy,
            leaf,
            field,
            row_group,
            &window,
            &mut covered,
            &mut passed,
            &mut scratch,
            &mut metrics,
        )?;
        // A rejected chunk does not append any bitmap. Mixed chunks append
        // both covered and uncovered page spans in file order.
        let appended = window_end - span.start;
        debug_assert!(covered.len() <= appended && passed.len() <= appended);
        covered.append_n(appended - covered.len(), false);
        passed.append_n(appended - passed.len(), false);
        let value_rows = selected_rows - chunk_result.rows;
        metrics.chunks_verdict += usize::from(chunk_result.rows > 0);
        metrics.rows_verdict += chunk_result.rows;
        if let Some(runtime) = chunk_result.runtime {
            if runtime.exact {
                metrics.rows_runtime_exact += chunk_result.rows;
            } else {
                metrics.rows_runtime_conservative += chunk_result.rows;
            }
        }
        metrics.chunks_value += usize::from(value_rows > 0);
        metrics.rows_value += value_rows;
        metrics.fallback_plain += usize::from(chunk_result.plain);
        metrics.fallback_limit += usize::from(chunk_result.limit);
        metrics.fallback_error += usize::from(chunk_result.error);
        metrics.fallback_unsupported += usize::from(chunk_result.unsupported);
    }
    // Row groups that hold no selected rows append nothing.
    covered.append_n(span.len - covered.len(), false);
    passed.append_n(span.len - passed.len(), false);
    let passed = passed.finish();
    metrics.rows_verdict_matched = passed.count_set_bits();
    Ok(Some(CodeSelection {
        covered: covered.finish(),
        passed,
        metrics,
    }))
}

/// The rows of one row group that lie inside the selected span.
struct ChunkWindow {
    /// The number of rows in the row group.
    rows: usize,
    /// The first window row, relative to the row group.
    start: usize,
    /// The selected rows of the window.
    selection: BooleanBuffer,
}

impl ChunkWindow {
    fn end(&self) -> usize {
        self.start + self.selection.len()
    }

    /// Returns the selected rows of the window inside group rows
    /// `start..end`, and the first of those rows.
    fn part(&self, start: usize, end: usize) -> Option<(usize, BooleanBuffer)> {
        let first = start.max(self.start);
        let last = end.min(self.end());
        (first < last).then(|| {
            (
                first,
                self.selection.slice(first - self.start, last - first),
            )
        })
    }
}

/// The pages of one column chunk, read and checked before any verdict work.
struct ScannedChunk {
    /// The dictionary page, if the chunk starts with one.
    dictionary: Option<Page>,
    /// The data pages inside the window, in file order. Only pages with
    /// selected rows hold their contents.
    pages: Vec<ScannedPage>,
}

struct ScannedPage {
    /// The first row of the page, relative to the row group.
    start: usize,
    rows: usize,
    page: Option<Page>,
}

/// Reads the page spans of one chunk and the pages that hold selected rows
/// in `window`. Returns `None` if a page's counts or the chunk's total
/// disagree, so that only the ordinary reader positions its rows:
///
/// * A page header's DataPage v2 row count differs from its value count.
/// * An offset index span differs from the value or row count of a page
///   that both readers decode.
/// * The page spans do not add up to the row group.
///
/// Without an offset index, every page header is checked. With one, both
/// readers skip pages without selected rows by their index span, so their
/// headers cannot move a row. Sparse and cached chunks do not hold their
/// bytes.
fn scan_chunk(
    pages: &mut dyn PageReader,
    rows: usize,
    window: Option<&ChunkWindow>,
) -> Result<Option<ScannedChunk>> {
    let mut dictionary = None;
    if pages.peek_next_page()?.is_some_and(|page| page.is_dict) {
        if window.is_none() {
            pages.skip_next_page()?;
        } else {
            let Ok(Some(page)) = pages.get_next_page() else {
                // An ordinary read might only touch selected PLAIN pages in
                // this chunk, and it decides whether they read.
                return Ok(None);
            };
            dictionary = Some(page);
        }
    }
    let mut scanned = Vec::new();
    let mut offset = 0;
    while let Some(header) = pages.peek_next_page()? {
        // A flat leaf has one value per row. A page header gives its value
        // count as the level count. Only an offset index omits it.
        let span = match (header.is_dict, header.num_levels, header.num_rows) {
            (true, _, _) => return Ok(None),
            (_, Some(levels), Some(rows)) if levels != rows => return Ok(None),
            (_, Some(levels), _) => levels,
            (_, None, Some(rows)) => rows,
            (_, None, None) => return Ok(None),
        };
        if span > rows - offset {
            return Ok(None);
        }
        let start = offset;
        offset += span;
        let part = window.and_then(|window| window.part(start, offset));
        let wanted = part
            .as_ref()
            .is_some_and(|(_, selection)| selection.count_set_bits() != 0);
        let page = if wanted {
            let page = pages
                .get_next_page()?
                .ok_or_else(|| general_err!("Missing predicate data page"))?;
            let page_rows = match &page {
                Page::DataPageV2 { num_rows, .. } => *num_rows as usize,
                _ => span,
            };
            if page.num_values() as usize != span || page_rows != span {
                return Ok(None);
            }
            Some(page)
        } else {
            pages.skip_next_page()?;
            None
        };
        // Keep only pages inside the window, so this list is bounded by the
        // selected span rather than by the row group.
        if part.is_some() {
            scanned.push(ScannedPage {
                start,
                rows: span,
                page,
            });
        }
    }
    if offset != rows {
        return Ok(None);
    }
    Ok(Some(ScannedChunk {
        dictionary,
        pages: scanned,
    }))
}

#[derive(Default)]
struct ChunkResult {
    runtime: Option<crate::arrow::arrow_reader::RuntimeVerdict>,
    rows: usize,
    plain: bool,
    limit: bool,
    error: bool,
    unsupported: bool,
}

struct VerdictScratch {
    validity: Vec<u64>,
    verdict: Vec<u64>,
}

#[expect(clippy::too_many_arguments)]
fn evaluate_chunk(
    scanned: ScannedChunk,
    groups: &dyn RowGroups,
    predicate: &mut dyn ArrowPredicate,
    policy: DictionaryFilterOptions,
    leaf: usize,
    field: &ParquetField,
    group: &RowGroupMetaData,
    window: &ChunkWindow,
    covered: &mut BooleanBufferBuilder,
    passed: &mut BooleanBufferBuilder,
    scratch: &mut VerdictScratch,
    metrics: &mut DictionaryFilterMetrics,
) -> Result<ChunkResult> {
    let mut result = ChunkResult::default();
    let Some(dictionary_page) = scanned.dictionary else {
        result.plain = true;
        return Ok(result);
    };
    let entries = dictionary_page.num_values() as usize;
    if entries as u128 * 100 > window.rows as u128 * u128::from(policy.max_percent) {
        result.limit = true;
        return Ok(result);
    }
    let column = group.column(leaf);
    let values = match decode_entries(&dictionary_page, entries, column, groups, field, predicate) {
        Ok(values) => values,
        Err(_) => {
            result.error = true;
            return Ok(result);
        }
    };
    metrics.entries_tested += entries;
    let verdict = match predicate.evaluate_dictionary(leaf, &values) {
        Ok(Some(verdict)) if verdict.entries.len() == entries => verdict,
        Ok(None) => {
            metrics.entries_tested -= entries;
            result.unsupported = true;
            return Ok(result);
        }
        _ => {
            result.error = true;
            return Ok(result);
        }
    };
    let no_nulls = field.def_level == 0
        || column
            .statistics()
            .is_some_and(|s| s.null_count_opt() == Some(0));
    if (!verdict.null || no_nulls)
        && verdict.entries.count_set_bits() == 0
        && all_dictionary_pages(column)
    {
        covered.append_buffer(&window.selection);
        passed.append_n(window.selection.len(), false);
        result.rows = window.selection.count_set_bits();
        result.runtime = verdict.runtime;
        metrics.row_groups_pruned_dictionary += 1;
        metrics
            .pruned_row_group_offsets
            .push(group.column(0).data_page_offset());
        return Ok(result);
    }

    let dictionary = ChunkDictionary::for_verdict(values, column.column_type());
    let mut table = dictionary.verdict_table(verdict.entries.iter());
    for ScannedPage { start, rows, page } in scanned.pages {
        let Some((first, selection)) = window.part(start, start + rows) else {
            continue;
        };
        let part = selection.len();
        let Some(page) = page else {
            // The page has no selected rows.
            covered.append_n(part, false);
            passed.append_n(part, false);
            continue;
        };
        let selected_rows = selection.count_set_bits();
        if !dictionary_encoding(page.encoding()) {
            result.plain = true;
            covered.append_n(part, false);
            passed.append_n(part, false);
            continue;
        }
        let decoded = FlatDataPage::new(&page, field.def_level != 0).and_then(|flat| {
            if !flat.has_hybrid_levels() {
                return Ok(false);
            }
            let page = flat.into_dictionary(dictionary.clone())?;
            page.verdicts(&mut table, &mut scratch.validity, &mut scratch.verdict)?;
            Ok(true)
        });
        if !matches!(decoded, Ok(true)) {
            // The ordinary reader's acceptance rules are authoritative. It
            // can stop after the page's logical index count even when the
            // final packed header advertises a larger run, for example.
            result.error |= decoded.is_err();
            result.unsupported |= decoded.is_ok();
            covered.append_n(part, false);
            passed.append_n(part, false);
            continue;
        }
        if verdict.null && field.def_level != 0 {
            for (word, valid) in scratch.verdict.iter_mut().zip(&scratch.validity) {
                *word |= !valid;
            }
        }
        let page_mask = words_to_mask(&scratch.verdict, rows).slice(first - start, part);
        covered.append_buffer(&selection);
        passed.append_buffer(&(&page_mask & &selection));
        result.rows += selected_rows;
    }
    result.runtime = verdict.runtime;
    metrics.runs_evaluated += table.runs_evaluated();
    Ok(result)
}

fn words_to_mask(words: &[u64], rows: usize) -> BooleanBuffer {
    // Arrow bitmap bytes have little-endian bit order on every platform.
    let bytes = if cfg!(target_endian = "little") {
        Buffer::from_slice_ref(words)
    } else {
        Buffer::from_vec(
            words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    };
    BooleanBuffer::new(bytes, 0, rows)
}

fn dictionary_encoding(encoding: Encoding) -> bool {
    matches!(
        encoding,
        Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
    )
}

fn all_dictionary_pages(column: &ColumnChunkMetaData) -> bool {
    if let Some(mask) = column.page_encoding_stats_mask() {
        let dictionary = EncodingMask::new_from_encodings(
            [&Encoding::RLE_DICTIONARY, &Encoding::PLAIN_DICTIONARY].into_iter(),
        )
        .as_i32();
        return mask.as_i32() != 0 && mask.as_i32() & !dictionary == 0;
    }
    column.page_encoding_stats().is_some_and(|stats| {
        let mut data = stats
            .iter()
            .filter(|stat| {
                matches!(stat.page_type, PageType::DATA_PAGE | PageType::DATA_PAGE_V2)
                    && stat.count > 0
            })
            .peekable();
        data.peek().is_some() && data.all(|stat| dictionary_encoding(stat.encoding))
    })
}

fn decode_entries(
    dictionary: &Page,
    entries: usize,
    column: &ColumnChunkMetaData,
    groups: &dyn RowGroups,
    field: &ParquetField,
    predicate: &dyn ArrowPredicate,
) -> Result<arrow_array::ArrayRef> {
    let Page::DictionaryPage {
        buf,
        num_values,
        encoding,
        ..
    } = dictionary
    else {
        return Err(general_err!("Expected dictionary page"));
    };
    if !matches!(encoding, Encoding::PLAIN | Encoding::PLAIN_DICTIONARY) {
        return Err(general_err!("Unsupported dictionary page encoding"));
    }
    // Check physical lower bounds before the Arrow decoder reserves entry
    // buffers. In particular, a corrupted count cannot amplify a short page.
    let max_entries = match column.column_type() {
        Type::BOOLEAN => buf.len().saturating_mul(8),
        Type::INT32 | Type::FLOAT | Type::BYTE_ARRAY => buf.len() / 4,
        Type::INT64 | Type::DOUBLE => buf.len() / 8,
        Type::INT96 => buf.len() / 12,
        Type::FIXED_LEN_BYTE_ARRAY => {
            let width = usize::try_from(column.column_descr().type_length()).unwrap_or(0);
            buf.len().checked_div(width).unwrap_or(0)
        }
    };
    if entries > max_entries {
        return Err(general_err!("Dictionary count exceeds its PLAIN payload"));
    }
    // Decode the PLAIN dictionary bytes through the same logical type reader
    // as the column. The dictionary never contains definition levels.
    let page = Page::DataPageV2 {
        buf: buf.clone(),
        num_values: *num_values,
        encoding: Encoding::PLAIN,
        num_nulls: 0,
        num_rows: *num_values,
        def_levels_byte_len: 0,
        rep_levels_byte_len: 0,
        is_compressed: false,
        statistics: None,
    };
    let field = ParquetField {
        def_level: 0,
        nullable: false,
        ..field.clone()
    };
    let source = EntrySource {
        original: groups,
        page,
        entries,
    };
    let mut reader = ArrayReaderBuilder::new(&source, &ArrowReaderMetrics::disabled())
        .with_batch_size(entries.max(1))
        .build_array_reader(Some(&field), predicate.projection())?;
    if reader.read_records(entries)? != entries {
        return Err(general_err!("Incomplete predicate dictionary"));
    }
    reader.consume_batch()
}

struct EntrySource<'a> {
    original: &'a dyn RowGroups,
    page: Page,
    entries: usize,
}

impl RowGroups for EntrySource<'_> {
    fn num_rows(&self) -> usize {
        self.entries
    }

    fn column_chunks(&self, _i: usize) -> Result<Box<dyn PageIterator>> {
        Ok(Box::new(EntryChunk(Some(EntryPage(Some(
            self.page.clone(),
        ))))))
    }

    fn row_groups(&self) -> Box<dyn Iterator<Item = &RowGroupMetaData> + '_> {
        Box::new(std::iter::empty())
    }

    fn metadata(&self) -> &ParquetMetaData {
        self.original.metadata()
    }
}

struct EntryChunk(Option<EntryPage>);

impl Iterator for EntryChunk {
    type Item = Result<Box<dyn PageReader>>;
    fn next(&mut self) -> Option<Self::Item> {
        self.0
            .take()
            .map(|page| Ok(Box::new(page) as Box<dyn PageReader>))
    }
}

impl PageIterator for EntryChunk {}

struct EntryPage(Option<Page>);

impl Iterator for EntryPage {
    type Item = Result<Page>;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.take().map(Ok)
    }
}

impl PageReader for EntryPage {
    fn get_next_page(&mut self) -> Result<Option<Page>> {
        self.next().transpose()
    }

    fn peek_next_page(&mut self) -> Result<Option<PageMetadata>> {
        Ok(self.0.as_ref().map(|page| PageMetadata {
            num_rows: Some(page.num_values() as usize),
            num_levels: Some(page.num_values() as usize),
            is_dict: false,
        }))
    }

    fn skip_next_page(&mut self) -> Result<()> {
        self.0.take();
        Ok(())
    }
}
