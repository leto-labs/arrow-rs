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
        let Some(policy) = options.predicate.dictionary_filter_options() else {
            return self.with_predicate_options(options);
        };
        let PredicateOptions {
            array_reader,
            predicate,
            limit,
            total_rows,
        } = options;
        let rows = groups.num_rows();
        let enabled = policy.mode != DictionaryFilterMode::Off;
        let use_codes = enabled && limit.is_none() && predicate.dictionary_filter_eligible();
        let leaf = use_codes
            .then(|| flat_leaf(fields, groups.metadata(), predicate))
            .flatten();
        let code = match (use_codes, leaf) {
            (true, Some((index, field))) => Some(evaluate_chunks(
                &self, groups, predicate, policy, index, field,
            )?),
            _ => None,
        };
        let selected = || selection_mask(&self, rows);

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
            let (result, rows_value) = self.with_value_predicate(PredicateOptions {
                array_reader,
                predicate: &mut *predicate,
                limit,
                total_rows,
            })?;
            let metrics = DictionaryFilterMetrics {
                chunks_value: chunks,
                rows_value,
                fallback_columns: if enabled && several_leaves { chunks } else { 0 },
                fallback_unsupported: if enabled && !several_leaves {
                    chunks
                } else {
                    0
                },
                ..Default::default()
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
            metrics.rows_verified = self.num_rows_selected().unwrap_or(rows);
            let oracle = self.clone().with_predicate(array_reader, predicate)?;
            if (&selection_mask(&oracle, rows) & &covered) != passed {
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
                let remaining = &selected() & &!&covered;
                let fallback = self
                    .clone()
                    .with_selection(Some(RowSelection::from_boolean_buffer(remaining)))
                    .with_predicate(array_reader, predicate)?;
                &selection_mask(&fallback, rows) | &passed
            };
            if mask.count_set_bits() == self.num_rows_selected().unwrap_or(rows) {
                self
            } else {
                self.with_selection(Some(RowSelection::from_boolean_buffer(mask)))
            }
        };
        predicate.record_dictionary_filter(metrics);
        Ok(result)
    }
}

fn selection_mask(plan: &ReadPlanBuilder, rows: usize) -> BooleanBuffer {
    plan.selection().map_or_else(
        || BooleanBuffer::new_set(rows),
        |selection| {
            let mask = selection.clone().into_boolean_buffer();
            if mask.len() >= rows {
                return mask.slice(0, rows);
            }
            let mut padded = BooleanBufferBuilder::new(rows);
            padded.append_buffer(&mask);
            padded.append_n(rows - mask.len(), false);
            padded.finish()
        },
    )
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

fn evaluate_chunks(
    plan: &ReadPlanBuilder,
    groups: &dyn RowGroups,
    predicate: &mut dyn ArrowPredicate,
    policy: DictionaryFilterOptions,
    leaf: usize,
    field: &ParquetField,
) -> Result<CodeSelection> {
    let selected = selection_mask(plan, groups.num_rows());
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
        let selection = selected.slice(offset, length);
        offset += length;
        let pages = chunks
            .next()
            .ok_or_else(|| general_err!("Missing predicate column chunk"))?;
        let selected_rows = selection.count_set_bits();
        let mut chunk_result = ChunkResult::default();
        if selected_rows != 0 {
            chunk_result = evaluate_chunk(
                &mut *pages?,
                groups,
                predicate,
                policy,
                leaf,
                field,
                row_group,
                &selection,
                &mut covered,
                &mut passed,
                &mut scratch,
                &mut metrics,
            )?;
        }
        // A rejected chunk does not append any bitmap. Mixed chunks append
        // both covered and uncovered page spans in file order.
        debug_assert!(covered.len() <= offset && passed.len() <= offset);
        covered.append_n(offset - covered.len(), false);
        passed.append_n(offset - passed.len(), false);
        let value_rows = selected_rows - chunk_result.rows;
        metrics.chunks_verdict += usize::from(chunk_result.rows > 0);
        metrics.rows_verdict += chunk_result.rows;
        metrics.chunks_value += usize::from(value_rows > 0);
        metrics.rows_value += value_rows;
        metrics.fallback_plain += usize::from(chunk_result.plain);
        metrics.fallback_limit += usize::from(chunk_result.limit);
        metrics.fallback_error += usize::from(chunk_result.error);
        metrics.fallback_unsupported += usize::from(chunk_result.unsupported);
    }
    let passed = passed.finish();
    metrics.rows_verdict_matched = passed.count_set_bits();
    Ok(CodeSelection {
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
    if entries as u128 * 100 > selected.len() as u128 * u128::from(policy.max_percent) {
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
    let mut offset = 0;
    while let Some(header) = pages.peek_next_page()? {
        if header.is_dict {
            return Err(general_err!("Unexpected dictionary after first page"));
        }
        let rows = header
            .num_rows
            .or(header.num_levels)
            .ok_or_else(|| general_err!("Missing flat predicate page row count"))?;
        if rows > selected.len() - offset {
            return Err(general_err!("Predicate page exceeds row group"));
        }
        let selection = selected.slice(offset, rows);
        offset += rows;
        let selected_rows = selection.count_set_bits();
        if selected_rows == 0 {
            pages.skip_next_page()?;
            covered.append_n(rows, false);
            passed.append_n(rows, false);
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
        let flat = FlatDataPage::new(&page, field.def_level != 0)?;
        if !dictionary_encoding(flat.encoding) || !flat.has_hybrid_levels() {
            result.plain |= !dictionary_encoding(flat.encoding);
            result.unsupported |= !flat.has_hybrid_levels();
            covered.append_n(rows, false);
            passed.append_n(rows, false);
            continue;
        }
        let page = flat.into_dictionary(dictionary.clone())?;
        page.verdicts(&mut table, &mut scratch.validity, &mut scratch.verdict)?;
        if verdict.null && field.def_level != 0 {
            for (word, valid) in scratch.verdict.iter_mut().zip(&scratch.validity) {
                *word |= !valid;
            }
        }
        let page_mask = words_to_mask(&scratch.verdict, rows);
        covered.append_buffer(&selection);
        passed.append_buffer(&(&page_mask & &selection));
        result.rows += selected_rows;
    }
    if offset != selected.len() {
        return Err(general_err!("Predicate pages did not cover the row group"));
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
