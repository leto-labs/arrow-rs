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

use std::ops::Range;

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
        let span = selected_span(&self, rows);
        // Bound dense scratch space relative to selected rows even when the
        // first and last selected rows are separated by large gaps.
        let bounded = self
            .num_rows_selected()
            .is_none_or(|selected| selected >= span.len().div_ceil(4));
        let use_codes = bounded && limit.is_none() && predicate.dictionary_filter_eligible();
        let leaf = use_codes
            .then(|| flat_leaf(fields, groups.metadata(), predicate))
            .flatten();
        let code = leaf
            .map(|(index, field)| {
                let selected = selection_mask(&self, &span);
                evaluate_chunks(groups, predicate, policy, index, field, &span, selected)
            })
            .transpose()?;

        let Some(code) = code else {
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
            let metrics = DictionaryFilterMetrics {
                chunks_value: chunks,
                rows_value,
                fallback_columns: if several_leaves { chunks } else { 0 },
                fallback_unsupported: if several_leaves { 0 } else { chunks },
                ..Default::default()
            };
            predicate.record_dictionary_filter(metrics);
            return Ok(result);
        };
        let CodeSelection {
            selected,
            covered,
            passed,
            mut metrics,
        } = code;
        let verify = policy.mode == DictionaryFilterMode::Verify && metrics.chunks_verdict != 0;
        let result = if verify {
            metrics.rows_verified = self.num_rows_selected().unwrap_or(rows);
            let oracle = self.clone().with_predicate(array_reader, predicate)?;
            if (&selection_mask(&oracle, &span) & &covered) != passed {
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
                let remaining = &selected & &!&covered;
                let fallback = self
                    .clone()
                    .with_selection(Some(selection_from_span(remaining, &span, rows)))
                    .with_predicate(array_reader, predicate)?;
                &selection_mask(&fallback, &span) | &passed
            };
            if mask.count_set_bits() == self.num_rows_selected().unwrap_or(rows) {
                self
            } else {
                self.with_selection(Some(selection_from_span(mask, &span, rows)))
            }
        };
        predicate.record_dictionary_filter(metrics);
        Ok(result)
    }
}

fn selected_span(plan: &ReadPlanBuilder, rows: usize) -> Range<usize> {
    let Some(selection) = plan.selection() else {
        return 0..rows;
    };
    if let Some(mask) = selection.as_mask() {
        let mask = mask.slice(0, mask.len().min(rows));
        let Some(start) = mask.set_indices().next() else {
            return 0..0;
        };
        let end = RowSelection::from_boolean_buffer(mask)
            .trim()
            .total_row_count();
        return start..end;
    }
    let mut start = None;
    let mut end = 0;
    let mut offset = 0;
    for selector in selection.iter() {
        if offset >= rows {
            break;
        }
        let next = offset.saturating_add(selector.row_count).min(rows);
        if !selector.skip {
            start.get_or_insert(offset);
            end = next;
        }
        offset = next;
    }
    start.unwrap_or_default()..end
}

fn selection_mask(plan: &ReadPlanBuilder, span: &Range<usize>) -> BooleanBuffer {
    let Some(selection) = plan.selection() else {
        return BooleanBuffer::new_set(span.len());
    };
    let mut builder;
    if let Some(mask) = selection.as_mask() {
        let start = span.start.min(mask.len());
        let end = span.end.min(mask.len());
        let slice = mask.slice(start, end - start);
        if slice.len() == span.len() {
            return slice;
        }
        builder = BooleanBufferBuilder::new(span.len());
        builder.append_buffer(&slice);
    } else {
        builder = BooleanBufferBuilder::new(span.len());
        let mut offset: usize = 0;
        for selector in selection.iter() {
            let next = offset.saturating_add(selector.row_count);
            let overlap = offset.max(span.start)..next.min(span.end);
            builder.append_n(overlap.len(), !selector.skip);
            if next >= span.end {
                break;
            }
            offset = next;
        }
    }
    builder.append_n(span.len() - builder.len(), false);
    builder.finish()
}

fn selection_from_span(mask: BooleanBuffer, span: &Range<usize>, rows: usize) -> RowSelection {
    if span.len() == rows {
        return RowSelection::from_boolean_buffer(mask);
    }
    // Keep skipped prefixes and suffixes out of the bitmap when run-length
    // encoding is compact. Highly fragmented output still needs a bitmap.
    let selector_bytes = std::mem::size_of::<super::RowSelector>();
    let max_ranges = rows.div_ceil(8) / (2 * selector_bytes);
    if mask.set_slices().take(max_ranges + 1).count() <= max_ranges {
        return RowSelection::from_consecutive_ranges(
            mask.set_slices()
                .map(|(start, end)| start + span.start..end + span.start),
            rows,
        );
    }
    let mut result = BooleanBufferBuilder::new(span.end);
    result.append_n(span.start, false);
    result.append_buffer(&mask);
    RowSelection::from_boolean_buffer(result.finish())
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
    selected: BooleanBuffer,
    covered: BooleanBuffer,
    passed: BooleanBuffer,
    metrics: DictionaryFilterMetrics,
}

fn evaluate_chunks(
    groups: &dyn RowGroups,
    predicate: &mut dyn ArrowPredicate,
    policy: DictionaryFilterOptions,
    leaf: usize,
    field: &ParquetField,
    span: &Range<usize>,
    selected: BooleanBuffer,
) -> Result<CodeSelection> {
    let mut chunks = groups.column_chunks(leaf)?;
    let mut covered = BooleanBufferBuilder::new(selected.len());
    let mut passed = BooleanBufferBuilder::new(selected.len());
    let mut metrics = DictionaryFilterMetrics::default();
    let mut scratch = VerdictScratch {
        validity: Vec::new(),
        verdict: Vec::new(),
    };
    let mut offset = 0;
    for row_group in groups.row_groups() {
        let length = row_group.num_rows() as usize;
        let group_end = offset + length;
        let pages = chunks
            .next()
            .ok_or_else(|| general_err!("Missing predicate column chunk"))?;
        let overlap = offset.max(span.start)..group_end.min(span.end);
        let selection = (!overlap.is_empty())
            .then(|| selected.slice(overlap.start - span.start, overlap.len()));
        let selected_rows = selection.as_ref().map_or(0, BooleanBuffer::count_set_bits);
        let mut chunk_result = ChunkResult::default();
        if let Some(selection) = selection.filter(|_| selected_rows != 0) {
            chunk_result = evaluate_chunk(
                &mut *pages?,
                groups,
                predicate,
                policy,
                leaf,
                field,
                row_group,
                overlap.start - offset,
                &selection,
                &mut covered,
                &mut passed,
                &mut scratch,
                &mut metrics,
            )?;
        }
        offset = group_end;
        // A rejected chunk does not append any bitmap. Mixed chunks append
        // both covered and uncovered page spans in file order.
        let bitmap_end = offset.clamp(span.start, span.end) - span.start;
        debug_assert!(covered.len() <= bitmap_end && passed.len() <= bitmap_end);
        covered.append_n(bitmap_end - covered.len(), false);
        passed.append_n(bitmap_end - passed.len(), false);
        let value_rows = selected_rows - chunk_result.rows;
        metrics.chunks_verdict += usize::from(chunk_result.rows > 0);
        metrics.rows_verdict += chunk_result.rows;
        metrics.chunks_value += usize::from(value_rows > 0);
        metrics.rows_value += value_rows;
        metrics.fallback_plain += usize::from(chunk_result.plain);
        metrics.fallback_limit += usize::from(chunk_result.limit);
        metrics.fallback_error += usize::from(chunk_result.error);
        metrics.fallback_unsupported += usize::from(chunk_result.unsupported);
        if offset >= span.end {
            break;
        }
    }
    if offset < span.end {
        return Err(general_err!(
            "Predicate row groups did not reach the selected row span"
        ));
    }
    let passed = passed.finish();
    metrics.rows_verdict_matched = passed.count_set_bits();
    Ok(CodeSelection {
        selected,
        covered: covered.finish(),
        passed,
        metrics,
    })
}

#[derive(Default)]
struct ChunkResult {
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
    pages: &mut dyn PageReader,
    groups: &dyn RowGroups,
    predicate: &mut dyn ArrowPredicate,
    policy: DictionaryFilterOptions,
    leaf: usize,
    field: &ParquetField,
    group: &RowGroupMetaData,
    selected_start: usize,
    selected: &BooleanBuffer,
    covered: &mut BooleanBufferBuilder,
    passed: &mut BooleanBufferBuilder,
    scratch: &mut VerdictScratch,
    metrics: &mut DictionaryFilterMetrics,
) -> Result<ChunkResult> {
    let mut result = ChunkResult::default();
    if !pages.peek_next_page()?.is_some_and(|page| page.is_dict) {
        result.plain = true;
        return Ok(result);
    }
    let Ok(Some(dictionary_page)) = pages.get_next_page() else {
        // An ordinary read might only touch selected PLAIN pages in this chunk.
        result.error = true;
        return Ok(result);
    };
    let entries = dictionary_page.num_values() as usize;
    if entries as u128 * 100 > group.num_rows() as u128 * u128::from(policy.max_percent) {
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
        covered.append_buffer(selected);
        passed.append_n(selected.len(), false);
        result.rows = selected.count_set_bits();
        metrics.row_groups_pruned_dictionary += 1;
        metrics
            .pruned_row_group_offsets
            .push(group.column(0).data_page_offset());
        return Ok(result);
    }

    let dictionary = ChunkDictionary::for_verdict(values, column.column_type());
    let mut table = dictionary.verdict_table(verdict.entries.iter());
    let group_rows = group.num_rows() as usize;
    let selected_end = selected_start + selected.len();
    let mut offset = 0;
    while offset < selected_end || selected_end == group_rows {
        let Some(header) = pages.peek_next_page()? else {
            break;
        };
        if header.is_dict {
            return Err(general_err!("Unexpected dictionary after first page"));
        }
        // Match the ordinary flat reader. An offset index supplies num_rows
        // only. Without one, the page's num_values is the level-slot count.
        let rows = header
            .num_levels
            .or(header.num_rows)
            .ok_or_else(|| general_err!("Missing flat predicate page row count"))?;
        if rows > group_rows - offset {
            return Err(general_err!("Predicate page exceeds row group"));
        }
        let page_start = offset;
        offset += rows;
        let overlap = page_start.max(selected_start)..offset.min(selected_end);
        if overlap.is_empty() {
            pages.skip_next_page()?;
            continue;
        }
        let selection = selected.slice(overlap.start - selected_start, overlap.len());
        let selected_rows = selection.count_set_bits();
        if selected_rows == 0 {
            pages.skip_next_page()?;
            covered.append_n(overlap.len(), false);
            passed.append_n(overlap.len(), false);
            continue;
        }
        let page = pages
            .get_next_page()?
            .ok_or_else(|| general_err!("Missing predicate data page"))?;
        if page.num_values() as usize != rows {
            return Err(general_err!(
                "Flat predicate page row count differs from page metadata"
            ));
        }
        if !dictionary_encoding(page.encoding()) {
            result.plain = true;
            covered.append_n(overlap.len(), false);
            passed.append_n(overlap.len(), false);
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
            covered.append_n(overlap.len(), false);
            passed.append_n(overlap.len(), false);
            continue;
        }
        if verdict.null && field.def_level != 0 {
            for (word, valid) in scratch.verdict.iter_mut().zip(&scratch.validity) {
                *word |= !valid;
            }
        }
        let page_mask =
            words_to_mask(&scratch.verdict, rows).slice(overlap.start - page_start, overlap.len());
        covered.append_buffer(&selection);
        passed.append_buffer(&(&page_mask & &selection));
        result.rows += selected_rows;
    }
    if offset < selected_end {
        return Err(general_err!(
            "Predicate pages did not reach the selected row span"
        ));
    }
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
