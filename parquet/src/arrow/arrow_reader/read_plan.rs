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

//! [`ReadPlan`] and [`ReadPlanBuilder`] for determining which rows to read
//! from a Parquet file

use crate::arrow::ProjectionMask;
use crate::arrow::array_reader::ArrayReader;
use crate::arrow::array_reader::{ArrayReaderBuilder, RowGroups};
use crate::arrow::arrow_reader::metrics::ArrowReaderMetrics;
use crate::arrow::arrow_reader::selection::{
    LoadedRowRanges, RowSelectionPolicy, RowSelectionStrategy,
};
use crate::arrow::arrow_reader::{
    ArrowPredicate, ParquetRecordBatchReader, RowSelection, RowSelectionCursor, RowSelector,
};
use crate::arrow::schema::{ParquetField, ParquetFieldType};
use crate::errors::{ParquetError, Result};
use crate::file::metadata::{ColumnChunkMetaData, RowGroupMetaData};
use arrow_array::{Array, BooleanArray};
use arrow_array::{ArrayRef, RecordBatch, RecordBatchOptions, StructArray};
use arrow_buffer::BooleanBuffer;
use arrow_schema::{DataType, Schema, SchemaRef};
use arrow_select::filter::filter;
use arrow_select::filter::prep_null_mask_filter;
use std::sync::Arc;
use std::time::Instant;
mod dictionary;

/// Options for [`ReadPlanBuilder::with_predicate_options`].
pub struct PredicateOptions<'a> {
    array_reader: Box<dyn ArrayReader>,
    predicate: &'a mut dyn ArrowPredicate,
    limit: Option<usize>,
    total_rows: usize,
}

impl<'a> PredicateOptions<'a> {
    /// Create options for evaluating `predicate` against rows produced by
    /// `array_reader`.
    ///
    /// By default there is no match-count limit; the predicate is evaluated
    /// over every row the reader yields. Use [`Self::with_limit`] to enable
    /// early termination.
    pub fn new(array_reader: Box<dyn ArrayReader>, predicate: &'a mut dyn ArrowPredicate) -> Self {
        Self {
            array_reader,
            predicate,
            limit: None,
            total_rows: 0,
        }
    }

    /// Stop scanning `array_reader` once `limit` matches have accumulated.
    ///
    /// Performance optimization for `LIMIT` / TopK: when the cumulative
    /// `true_count` reaches `limit`, the current filter batch is truncated
    /// at the `limit`-th match and remaining batches are never decoded.
    ///
    /// `limit` counts predicate matches, not output rows — callers applying
    /// an offset must pass `offset + limit`.
    ///
    /// `total_rows` is the row count `array_reader` would yield if iterated
    /// to completion. It is used to pad un-evaluated trailing rows as "not
    /// selected" so the returned [`RowSelection`] covers the full row group.
    ///
    /// Only valid for the *last* predicate in a filter chain: intermediate
    /// predicates' match counts do not map 1:1 to output rows.
    pub fn with_limit(mut self, limit: usize, total_rows: usize) -> Self {
        self.limit = Some(limit);
        self.total_rows = total_rows;
        self
    }
}

/// A builder for [`ReadPlan`]
#[derive(Clone, Debug)]
pub struct ReadPlanBuilder {
    batch_size: usize,
    /// Which rows to select. Includes the result of all filters applied so far
    selection: Option<RowSelection>,
    /// Policy to use when materializing the row selection
    row_selection_policy: RowSelectionPolicy,
    /// Row ranges with page data loaded for the current projection.
    loaded_row_ranges: Option<Arc<LoadedRowRanges>>,
}

impl ReadPlanBuilder {
    /// Create a `ReadPlanBuilder` with the given batch size
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            selection: None,
            row_selection_policy: RowSelectionPolicy::default(),
            loaded_row_ranges: None,
        }
    }

    /// Set the current selection to the given value
    pub fn with_selection(mut self, selection: Option<RowSelection>) -> Self {
        self.selection = selection;
        self
    }

    /// Configure the policy to use when materialising the [`RowSelection`]
    ///
    /// Defaults to [`RowSelectionPolicy::Auto`]
    pub fn with_row_selection_policy(mut self, policy: RowSelectionPolicy) -> Self {
        self.row_selection_policy = policy;
        self
    }

    pub(crate) fn with_loaded_row_ranges(mut self, ranges: Option<LoadedRowRanges>) -> Self {
        self.loaded_row_ranges = ranges.map(Arc::new);
        self
    }

    /// Returns the current row selection policy
    pub fn row_selection_policy(&self) -> &RowSelectionPolicy {
        &self.row_selection_policy
    }

    /// Returns the current selection, if any
    pub fn selection(&self) -> Option<&RowSelection> {
        self.selection.as_ref()
    }

    /// Specifies the number of rows in the row group, before filtering is applied.
    ///
    /// Returns a [`LimitedReadPlanBuilder`] that can apply
    /// offset and limit.
    ///
    /// Call [`LimitedReadPlanBuilder::build_limited`] to apply the limits to this
    /// selection.
    pub(crate) fn limited(self, row_count: usize) -> LimitedReadPlanBuilder {
        LimitedReadPlanBuilder::new(self, row_count)
    }

    /// Returns true if the current plan selects any rows
    pub fn selects_any(&self) -> bool {
        self.selection
            .as_ref()
            .map(|s| s.selects_any())
            .unwrap_or(true)
    }

    /// Returns the number of rows selected, or `None` if all rows are selected.
    pub fn num_rows_selected(&self) -> Option<usize> {
        self.selection.as_ref().map(|s| s.row_count())
    }

    /// Returns the [`RowSelectionStrategy`] for this plan.
    ///
    /// Guarantees to return either `Selectors` or `Mask`, never `Auto`.
    pub(crate) fn resolve_selection_strategy(&self) -> RowSelectionStrategy {
        self.row_selection_policy.resolve(self.selection.as_ref())
    }

    /// Evaluates an [`ArrowPredicate`], updating this plan's `selection`
    ///
    /// If the current `selection` is `Some`, the resulting [`RowSelection`]
    /// will be the conjunction of the existing selection and the rows selected
    /// by `predicate`.
    ///
    /// Note: pre-existing selections may come from evaluating a previous predicate
    /// or if the [`ParquetRecordBatchReader`] specified an explicit
    /// [`RowSelection`] in addition to one or more predicates.
    pub fn with_predicate(
        self,
        array_reader: Box<dyn ArrayReader>,
        predicate: &mut dyn ArrowPredicate,
    ) -> Result<Self> {
        self.with_predicate_options(PredicateOptions::new(array_reader, predicate))
    }

    /// Evaluates an [`ArrowPredicate`] with the given [`PredicateOptions`],
    /// updating this plan's `selection`.
    ///
    /// Like [`Self::with_predicate`], but allows additional options such as a
    /// match-count limit for early termination (see
    /// [`PredicateOptions::with_limit`]).
    pub fn with_predicate_options(self, options: PredicateOptions<'_>) -> Result<Self> {
        self.apply_predicate(options, |_| {})
    }

    fn apply_predicate(
        mut self,
        options: PredicateOptions<'_>,
        rows_evaluated: impl FnOnce(usize),
    ) -> Result<Self> {
        let PredicateOptions {
            array_reader,
            predicate,
            limit,
            total_rows,
        } = options;

        // Target length for the concatenated filter output:
        // - Prior selection ⇒ the reader yields that many rows; `and_then`
        //   below requires the filter output to match.
        // - No prior selection ⇒ the reader yields `total_rows`. We only
        //   need to pad when `limit` may short-circuit the loop; otherwise
        //   iteration naturally exhausts.
        let expected_rows = match self.selection.as_ref() {
            Some(s) => Some(s.row_count()),
            None => limit.map(|_| total_rows),
        };

        let reader = ParquetRecordBatchReader::new(array_reader, self.clone().build());
        let mut filters = vec![];
        let mut processed_rows: usize = 0;
        let mut matched_rows: usize = 0;
        for maybe_batch in reader {
            let maybe_batch = maybe_batch?;
            let input_rows = maybe_batch.num_rows();
            let filter = predicate.evaluate(maybe_batch)?;
            // Since user supplied predicate, check error here to catch bugs quickly
            if filter.len() != input_rows {
                return Err(arrow_err!(
                    "ArrowPredicate predicate returned {} rows, expected {input_rows}",
                    filter.len()
                ));
            }
            let filter = match filter.null_count() {
                0 => filter,
                // RowSelection::from_filters expects non-null filters. Convert
                // NULL predicate results to false so they are not selected.
                _ => prep_null_mask_filter(&filter),
            };

            processed_rows += input_rows;

            match limit {
                Some(limit) if limit - matched_rows <= filter.len() => {
                    let truncated = filter.take_n_true(limit - matched_rows);
                    matched_rows += truncated.true_count();
                    filters.push(truncated);
                    if matched_rows >= limit {
                        break;
                    }
                }
                _ => {
                    matched_rows += filter.true_count();
                    filters.push(filter);
                }
            }
        }

        rows_evaluated(processed_rows);

        // Pad the tail so the filters cover `expected_rows` total. This keeps
        // the invariant that the resulting `RowSelection` spans every row the
        // reader would have produced — rows past the early break are marked
        // "not selected". When no limit is set the loop always exhausts and
        // no padding is needed.
        if let Some(expected) = expected_rows
            && processed_rows < expected
        {
            let pad_len = expected - processed_rows;
            filters.push(BooleanArray::new(BooleanBuffer::new_unset(pad_len), None));
        }

        // If the predicate selected all rows, applying it is a no-op. With no
        // prior selection this keeps selection as None, enabling coalesced page
        // fetches; with a prior selection it avoids rebuilding the same
        // selection.
        let all_selected = filters.iter().all(|f| f.true_count() == f.len());
        if all_selected {
            return Ok(self);
        }
        let raw = match (self.selection.as_ref(), self.row_selection_policy) {
            (Some(selection), _) if selection.as_mask().is_some() => {
                RowSelection::from_filters_mask(&filters)
            }
            (None, RowSelectionPolicy::Auto { threshold }) => {
                RowSelection::from_filters_auto(&filters, threshold)
            }
            _ => RowSelection::from_filters(&filters),
        };
        self.selection = match self.selection.take() {
            Some(selection) => Some(selection.and_then(&raw)),
            None => Some(raw),
        };
        Ok(self)
    }

    /// Create a final `ReadPlan` the read plan for the scan
    pub fn build(mut self) -> ReadPlan {
        // If selection is empty, truncate
        if !self.selects_any() {
            self.selection = Some(RowSelection::from(vec![]));
        }

        // Preferred strategy must not be Auto
        let selection_strategy = self.resolve_selection_strategy();

        let Self {
            batch_size,
            selection,
            row_selection_policy: _,
            loaded_row_ranges,
        } = self;

        let row_selection_cursor = selection
            .map(|s| selection_strategy.build_cursor(s.trim(), loaded_row_ranges))
            .unwrap_or_else(RowSelectionCursor::new_all);

        ReadPlan {
            batch_size,
            row_selection_cursor,
        }
    }
}

/// Builder for [`ReadPlan`] that applies a limit and offset to the read plan
///
/// See [`ReadPlanBuilder::limited`] to create this builder.
pub(crate) struct LimitedReadPlanBuilder {
    /// The underlying builder
    inner: ReadPlanBuilder,
    /// Total number of rows in the row group before the selection, limit or
    /// offset are applied
    row_count: usize,
    /// The offset to apply, if any
    offset: Option<usize>,
    /// The limit to apply, if any
    limit: Option<usize>,
}

impl LimitedReadPlanBuilder {
    /// Create a new `LimitedReadPlanBuilder` from the existing builder and number of rows
    fn new(inner: ReadPlanBuilder, row_count: usize) -> Self {
        Self {
            inner,
            row_count,
            offset: None,
            limit: None,
        }
    }

    /// Set the offset to apply to the read plan
    pub(crate) fn with_offset(mut self, offset: Option<usize>) -> Self {
        self.offset = offset;
        self
    }

    /// Set the limit to apply to the read plan
    pub(crate) fn with_limit(mut self, limit: Option<usize>) -> Self {
        self.limit = limit;
        self
    }

    /// Apply offset and limit, updating the selection on the underlying builder
    /// and returning it.
    pub(crate) fn build_limited(self) -> ReadPlanBuilder {
        let Self {
            mut inner,
            row_count,
            offset,
            limit,
        } = self;

        // If the selection is empty, truncate
        if !inner.selects_any() {
            inner.selection = Some(RowSelection::from(vec![]));
        }

        // If an offset is defined, apply it to the `selection`
        if let Some(offset) = offset {
            inner.selection = Some(match row_count.checked_sub(offset) {
                None => RowSelection::from(vec![]),
                Some(remaining) => inner
                    .selection
                    .map(|selection| selection.offset(offset))
                    .unwrap_or_else(|| {
                        RowSelection::from(vec![
                            RowSelector::skip(offset),
                            RowSelector::select(remaining),
                        ])
                    }),
            });
        }

        // If a limit is defined, apply it to the final `selection`
        if let Some(limit) = limit {
            inner.selection = Some(
                inner
                    .selection
                    .map(|selection| selection.limit(limit))
                    .unwrap_or_else(|| {
                        RowSelection::from(vec![RowSelector::select(limit.min(row_count))])
                    }),
            );
        }

        inner
    }
}

/// A plan reading specific rows from a Parquet Row Group.
///
/// See [`ReadPlanBuilder`] to create `ReadPlan`s
#[derive(Debug)]
pub struct ReadPlan {
    /// The number of rows to read in each batch
    batch_size: usize,
    /// Row ranges to be selected from the data source
    row_selection_cursor: RowSelectionCursor,
}

impl ReadPlan {
    /// Returns a mutable reference to the row selection cursor
    pub fn row_selection_cursor_mut(&mut self) -> &mut RowSelectionCursor {
        &mut self.row_selection_cursor
    }

    /// Return the number of rows to read in each output batch
    #[inline(always)]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

/// One total Boolean expression and the Parquet leaves it reads in file order.
#[derive(Clone)]
pub struct StagedPredicate {
    leaves: Vec<usize>,
    evaluate: Arc<StagedEvaluator>,
}

type StagedEvaluator = dyn Fn(&RecordBatch) -> Result<BooleanArray> + Send + Sync;

impl StagedPredicate {
    /// The evaluator receives a batch of these leaves, in physical schema order.
    pub fn new(
        leaves: Vec<usize>,
        evaluate: impl Fn(&arrow_array::RecordBatch) -> Result<BooleanArray> + Send + Sync + 'static,
    ) -> Self {
        Self {
            leaves,
            evaluate: Arc::new(evaluate),
        }
    }
}

/// Work done by the staged reader on one input vector.
#[derive(Default)]
pub struct StagedScanWork {
    /// Number of rows entering the filter stage.
    pub input_rows: usize,
    /// Number of rows passing every conjunct.
    pub output_rows: usize,
    /// Whether this vector used a selected read.
    pub selected: bool,
    /// Number of payload column slots decoded in full.
    pub decoded_values: usize,
    /// Number of payload column slots gathered by selected reads.
    pub gathered_values: usize,
    /// Elapsed time in predicate reads and evaluation.
    pub filter_time: std::time::Duration,
    /// Elapsed time in payload reads and output assembly.
    pub payload_time: std::time::Duration,
}

/// Optional per-vector decoding for a projection of flat columns.
pub struct StagedScan {
    predicates: Vec<StagedPredicate>,
    fully_matched: Option<Arc<[bool]>>,
    observe: Arc<dyn Fn(StagedScanWork) + Send + Sync>,
}

impl std::fmt::Debug for StagedScan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StagedScan")
            .field("predicates", &self.predicates.len())
            .finish()
    }
}

impl StagedScan {
    /// The caller keeps residual predicates. On a full vector the ordinary
    /// iterator returns the entire vector, while the selection-aware reader
    /// can also obtain the exact mask.
    pub fn new(
        predicates: Vec<StagedPredicate>,
        observe: impl Fn(StagedScanWork) + Send + Sync + 'static,
    ) -> Self {
        Self {
            predicates,
            fully_matched: None,
            observe: Arc::new(observe),
        }
    }

    /// Skip staging on physical row groups proved to match every predicate.
    pub fn with_fully_matched(mut self, groups: Option<Arc<[bool]>>) -> Self {
        self.fully_matched = groups;
        self
    }

    pub(crate) fn applies_to(&self, group: usize) -> bool {
        !self
            .fully_matched
            .as_ref()
            .and_then(|groups| groups.get(group))
            .copied()
            .unwrap_or(false)
    }
}

// Ratios from the SMT rows in the 5.12a cost table. Unknown encodings use
// full decode. Footer metadata cannot identify individual index run widths.
#[derive(Clone, Copy, Debug)]
enum SelectionCost {
    LargeDictionary,
    DateDictionary,
    NullableDictionary,
    ViewDictionary,
    PlainFixed,
    Unknown,
}

impl SelectionCost {
    fn for_column(column: &ColumnChunkMetaData, data_type: &DataType) -> Self {
        use crate::basic::{Encoding, Type};
        let primitive = matches!(column.column_type(), Type::INT32 | Type::INT64)
            && matches!(
                data_type,
                DataType::Int32 | DataType::Int64 | DataType::Date32 | DataType::Decimal64(_, _)
            );
        let view = matches!(data_type, DataType::Utf8View | DataType::BinaryView);
        let dictionary = column.page_encoding_stats_mask().is_some_and(|mask| {
            mask.is_only(Encoding::RLE_DICTIONARY) || mask.is_only(Encoding::PLAIN_DICTIONARY)
        });
        let required = column.column_descr().max_def_level() == 0;
        match (primitive, view, dictionary, required, data_type) {
            (true, _, true, true, DataType::Date32) => Self::DateDictionary,
            (true, _, true, true, _) => Self::LargeDictionary,
            (true, _, true, false, _) => Self::NullableDictionary,
            (_, true, true, true, _) => Self::ViewDictionary,
            (true, _, false, true, _)
                if column
                    .page_encoding_stats_mask()
                    .is_some_and(|mask| mask.is_only(Encoding::PLAIN)) =>
            {
                Self::PlainFixed
            }
            _ => Self::Unknown,
        }
    }

    fn uses_kernel(self, fraction: f64) -> bool {
        const FRACTIONS: [f64; 4] = [0.001, 0.02, 0.10, 0.20];
        let ratios = match self {
            Self::LargeDictionary => [0.105, 0.324, 0.740, 0.960],
            Self::DateDictionary => [0.210, 0.472, 0.988, 2.0],
            Self::NullableDictionary => [0.626, 0.922, 1.785, 3.0],
            Self::ViewDictionary => [0.103, 0.361, 1.297, 2.5],
            Self::PlainFixed => [0.106, 0.308, 0.993, 2.0],
            Self::Unknown => return false,
        };
        if fraction >= FRACTIONS[3] {
            return false;
        }
        let first = FRACTIONS.partition_point(|&bound| bound < fraction);
        let ratio = if first == 0 {
            ratios[0]
        } else {
            let low = first - 1;
            let position = (fraction - FRACTIONS[low]) / (FRACTIONS[first] - FRACTIONS[low]);
            ratios[low] + position * (ratios[first] - ratios[low])
        };
        ratio <= 0.80
    }
}

struct StagedColumn {
    reader: Box<dyn ArrayReader>,
    cost: SelectionCost,
    ordinary: bool,
    payload: bool,
}

impl StagedColumn {
    fn full(&mut self, rows: usize) -> Result<ArrayRef> {
        self.ordinary = true;
        let actual = self.reader.read_records(rows)?;
        if actual != rows {
            return Err(general_err!(
                "Expected {rows} staged rows, decoded {actual}"
            ));
        }
        self.reader.consume_batch()
    }

    fn skip(&mut self, rows: usize, empty: &[u64]) -> Result<()> {
        if !self.ordinary && !matches!(self.cost, SelectionCost::Unknown) {
            self.reader.read_selected(empty, rows)?;
        } else {
            self.ordinary = true;
            let actual = self.reader.skip_records(rows)?;
            if actual != rows {
                return Err(general_err!(
                    "Expected to skip {rows} staged rows, skipped {actual}"
                ));
            }
        }
        Ok(())
    }
}

struct StagedConjunct {
    columns: Vec<usize>,
    schema: SchemaRef,
    predicate: StagedPredicate,
}

#[derive(Clone, Copy, Default)]
struct ObservedRank {
    score: f64,
    samples: usize,
}

impl ObservedRank {
    fn observe(&mut self, nanos: f64, rejected: usize) {
        let rank = nanos / rejected.max(1) as f64;
        self.score = if self.samples == 0 {
            rank
        } else {
            self.score * 0.75 + rank * 0.25
        };
        self.samples += 1;
    }
}

struct StagedOrder {
    indices: Vec<usize>,
    ranks: Vec<ObservedRank>,
    pending: Option<(usize, usize)>,
    settled: Vec<(usize, usize)>,
}

impl StagedOrder {
    fn new(initial_scores: &[f64]) -> Self {
        let mut indices: Vec<_> = (0..initial_scores.len()).collect();
        indices.sort_by(|&left, &right| initial_scores[left].total_cmp(&initial_scores[right]));
        Self {
            indices,
            ranks: vec![ObservedRank::default(); initial_scores.len()],
            pending: None,
            settled: Vec::new(),
        }
    }

    fn adapt(&mut self) {
        let candidate = self
            .indices
            .windows(2)
            .enumerate()
            .find_map(|(position, pair)| {
                let (left, right) = (&self.ranks[pair[0]], &self.ranks[pair[1]]);
                (left.samples >= 2
                    && right.samples >= 2
                    && right.score < 0.8 * left.score
                    && !self.settled.contains(&(pair[1], pair[0])))
                .then_some((position, pair[1]))
            });
        if let Some((position, _)) = candidate.filter(|pair| self.pending == Some(*pair)) {
            // Conditional pass rates may change after a swap. Keep the
            // first stable change for this pair within the row group.
            self.settled
                .push((self.indices[position], self.indices[position + 1]));
            self.indices.swap(position, position + 1);
            self.ranks.fill(ObservedRank::default());
            self.pending = None;
        } else {
            self.pending = candidate;
        }
    }
}

struct VectorColumn {
    values: ArrayRef,
    // A subset of the vector in original row order when read by the kernel.
    selection: Option<BooleanBuffer>,
}

impl VectorColumn {
    fn compact(&self, selection: &BooleanBuffer, kept: usize) -> Result<ArrayRef> {
        if self.values.len() == kept {
            return Ok(Arc::clone(&self.values));
        }
        let mask = match &self.selection {
            Some(previous) => BooleanBuffer::collect_bool(previous.count_set_bits(), {
                let mut indices = previous.set_indices();
                move |_| selection.value(indices.next().unwrap())
            }),
            None => selection.clone(),
        };
        Ok(filter(&self.values, &BooleanArray::from(mask))?)
    }
}

/// ArrayReader adapter for the record-batch reader's existing All cursor.
/// Every read advances one physical vector, or skips rejected vectors until
/// a vector with output is ready. A physical selection bounds each read.
pub(crate) struct StagedArrayReader {
    schema: SchemaRef,
    data_type: DataType,
    columns: Vec<StagedColumn>,
    conjuncts: Vec<StagedConjunct>,
    order: StagedOrder,
    vectors: usize,
    rows: std::collections::VecDeque<RowSelector>,
    scan: Arc<StagedScan>,
    batch: Option<ArrayRef>,
    selection: Option<BooleanBuffer>,
}

impl StagedArrayReader {
    pub(crate) fn try_new(
        rows: &dyn RowGroups,
        row_group: &RowGroupMetaData,
        fields: Option<&ParquetField>,
        projection: &ProjectionMask,
        plan: &ReadPlanBuilder,
        metrics: &ArrowReaderMetrics,
        scan: &Arc<StagedScan>,
    ) -> Result<Option<Self>> {
        let Some(fields) = fields else {
            return Ok(None);
        };
        let (ParquetFieldType::Group { children }, DataType::Struct(arrow_fields)) =
            (&fields.field_type, &fields.arrow_type)
        else {
            return Ok(None);
        };
        let mut leaves = Vec::new();
        for (field, arrow_field) in children.iter().zip(arrow_fields) {
            let ParquetFieldType::Primitive { col_idx, .. } = field.field_type else {
                // A mixed projection is left to the original reader.
                return Ok(None);
            };
            if projection.leaf_included(col_idx) {
                leaves.push((col_idx, field, Arc::clone(arrow_field)));
            }
        }
        if leaves.is_empty() || scan.predicates.is_empty() {
            return Ok(None);
        }
        let schema = Arc::new(Schema::new(
            leaves
                .iter()
                .map(|(_, _, field)| Arc::clone(field))
                .collect::<Vec<_>>(),
        ));
        let mut used = vec![false; leaves.len()];
        let mut conjuncts = Vec::with_capacity(scan.predicates.len());
        for predicate in &scan.predicates {
            let Some(columns) = predicate
                .leaves
                .iter()
                .map(|leaf| leaves.iter().position(|(index, _, _)| index == leaf))
                .collect::<Option<Vec<_>>>()
            else {
                return Ok(None);
            };
            for &column in &columns {
                used[column] = true;
            }
            let schema = Arc::new(schema.project(&columns)?);
            conjuncts.push(StagedConjunct {
                columns,
                schema,
                predicate: predicate.clone(),
            });
        }
        let costs: Vec<_> = leaves
            .iter()
            .map(|(leaf, _, field)| {
                let column = row_group.column(*leaf);
                (
                    SelectionCost::for_column(column, field.data_type()),
                    column.uncompressed_size().max(0) as usize,
                )
            })
            .collect();
        let total = costs
            .iter()
            .fold(0usize, |total, (_, bytes)| total.saturating_add(*bytes));
        let eligible_payload: usize = costs
            .iter()
            .zip(&used)
            .filter_map(|(&(cost, bytes), &used)| {
                (!used && !matches!(cost, SelectionCost::Unknown)).then_some(bytes)
            })
            .fold(0usize, usize::saturating_add);
        if eligible_payload == 0 || eligible_payload < total.div_ceil(4) {
            return Ok(None);
        }
        let initial_scores = conjuncts
            .iter()
            .map(|conjunct| {
                // Until a vector is observed, assume a pass rate of one half.
                let estimated_bytes = conjunct
                    .columns
                    .iter()
                    .map(|&index| costs[index].1 as f64)
                    .sum::<f64>();
                2.0 * (1.0 + estimated_bytes / rows.num_rows().max(1) as f64)
            })
            .collect::<Vec<_>>();
        let order = StagedOrder::new(&initial_scores);
        let builder = ArrayReaderBuilder::new(rows, metrics)
            .with_batch_size(plan.batch_size)
            .with_parquet_metadata(rows.metadata());
        let columns = leaves
            .into_iter()
            .zip(costs)
            .zip(used)
            .map(|(((_, field, _), (cost, _)), used)| {
                Ok(StagedColumn {
                    reader: builder.build_array_reader(Some(field), projection)?,
                    cost,
                    ordinary: false,
                    payload: !used,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let rows = match plan.selection() {
            Some(selection) => selection.clone().trim().into(),
            None => std::collections::VecDeque::from([RowSelector::select(rows.num_rows())]),
        };
        Ok(Some(Self {
            data_type: DataType::Struct(schema.fields().clone()),
            schema,
            columns,
            order,
            conjuncts,
            vectors: 0,
            rows,
            scan: Arc::clone(scan),
            batch: None,
            selection: None,
        }))
    }

    pub(crate) fn selection(&self) -> Option<BooleanBuffer> {
        self.selection.clone()
    }

    fn read_vector(&mut self, rows: usize) -> Result<usize> {
        let filter_start = Instant::now();
        let mut selection = BooleanBuffer::new_set(rows);
        let mut kept = rows;
        let mut selected = false;
        let mut arrays: Vec<Option<VectorColumn>> = (0..self.columns.len()).map(|_| None).collect();
        let mut words = Vec::new();
        let sample = self.vectors.is_multiple_of(4);
        self.vectors = self.vectors.wrapping_add(1);
        for &index in &self.order.indices {
            let begin = sample.then(Instant::now);
            let conjunct = &self.conjuncts[index];
            let mut inputs = Vec::with_capacity(conjunct.columns.len());
            let fraction = kept as f64 / rows as f64;
            for &index in &conjunct.columns {
                if let Some(array) = &arrays[index] {
                    inputs.push(array.compact(&selection, kept)?);
                    continue;
                }
                let column = &mut self.columns[index];
                let array = if !column.ordinary && column.cost.uses_kernel(fraction) {
                    fill_words(&mut words, &selection);
                    selected = true;
                    VectorColumn {
                        values: column.reader.read_selected(&words, rows)?,
                        selection: Some(selection.clone()),
                    }
                } else {
                    VectorColumn {
                        values: column.full(rows)?,
                        selection: None,
                    }
                };
                inputs.push(array.compact(&selection, kept)?);
                arrays[index] = Some(array);
            }
            let options = RecordBatchOptions::new().with_row_count(Some(kept));
            let batch =
                RecordBatch::try_new_with_options(Arc::clone(&conjunct.schema), inputs, &options)?;
            let result = (conjunct.predicate.evaluate)(&batch)?;
            let result = if result.null_count() == 0 {
                result
            } else {
                prep_null_mask_filter(&result)
            };
            if result.len() != kept {
                return Err(general_err!("Staged predicate returned the wrong length"));
            }
            let before = kept;
            kept = result.true_count();
            if kept != before {
                if before == rows {
                    selection = result.values().clone();
                } else {
                    let previous = selection;
                    let mut passed = result.values().iter();
                    selection = BooleanBuffer::collect_bool(rows, |i| {
                        previous.value(i) && passed.next().unwrap()
                    });
                }
            }
            if let Some(begin) = begin {
                self.order.ranks[index].observe(begin.elapsed().as_nanos() as f64, before - kept);
            }
            if kept == 0 {
                break;
            }
        }
        if sample {
            self.order.adapt();
        }
        let filter_time = filter_start.elapsed();
        let payload_start = Instant::now();
        let mut work = StagedScanWork {
            input_rows: rows,
            output_rows: kept,
            filter_time,
            ..Default::default()
        };
        if kept == 0 {
            words.clear();
            words.resize(rows.div_ceil(64), 0);
            for (column, array) in self.columns.iter_mut().zip(&arrays) {
                if array.is_none() {
                    column.skip(rows, &words)?;
                }
            }
            work.payload_time = payload_start.elapsed();
            (self.scan.observe)(work);
            return Ok(0);
        }
        let fraction = kept as f64 / rows as f64;
        words.clear();
        for (column, array) in self.columns.iter_mut().zip(&mut arrays) {
            if array.is_some() {
                continue;
            }
            *array = Some(if !column.ordinary && column.cost.uses_kernel(fraction) {
                if words.is_empty() {
                    fill_words(&mut words, &selection);
                }
                selected = true;
                if column.payload {
                    work.gathered_values += kept;
                }
                VectorColumn {
                    values: column.reader.read_selected(&words, rows)?,
                    selection: Some(selection.clone()),
                }
            } else {
                if column.payload {
                    work.decoded_values += rows;
                }
                VectorColumn {
                    values: column.full(rows)?,
                    selection: None,
                }
            });
        }
        let values = arrays
            .into_iter()
            .map(|array| {
                let array = array.expect("every output column was read");
                if selected {
                    array.compact(&selection, kept)
                } else {
                    Ok(array.values)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        self.batch = Some(Arc::new(StructArray::try_new(
            self.schema.fields().clone(),
            values,
            None,
        )?));
        self.selection = Some(if selected {
            BooleanBuffer::new_set(kept)
        } else {
            selection
        });
        work.selected = selected;
        work.payload_time = payload_start.elapsed();
        (self.scan.observe)(work);
        Ok(if selected { kept } else { rows })
    }
}

fn fill_words(words: &mut Vec<u64>, selection: &BooleanBuffer) {
    words.clear();
    words.extend(selection.bit_chunks().iter_padded());
}

impl ArrayReader for StagedArrayReader {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn get_data_type(&self) -> &DataType {
        &self.data_type
    }

    fn read_records(&mut self, batch_size: usize) -> Result<usize> {
        self.selection = None;
        while let Some(mut run) = self.rows.pop_front() {
            if run.skip {
                // Offset-index readers skip unrequested pages without reading
                // them. Small steps could try to load a missing page prefix.
                let empty = vec![0; run.row_count.div_ceil(64)];
                for column in &mut self.columns {
                    column.skip(run.row_count, &empty)?;
                }
                continue;
            }
            let rows = run.row_count.min(batch_size);
            if rows == 0 {
                continue;
            }
            run.row_count -= rows;
            if run.row_count != 0 {
                self.rows.push_front(run);
            }
            let output = self.read_vector(rows)?;
            if output != 0 {
                return Ok(output);
            }
        }
        Ok(0)
    }

    fn consume_batch(&mut self) -> Result<ArrayRef> {
        Ok(self
            .batch
            .take()
            .unwrap_or_else(|| arrow_array::new_empty_array(&self.data_type)))
    }

    fn skip_records(&mut self, _: usize) -> Result<usize> {
        Err(general_err!(
            "Staged record reader owns the physical cursor"
        ))
    }

    fn get_def_levels(&self) -> Option<&[i16]> {
        None
    }

    fn get_rep_levels(&self) -> Option<&[i16]> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staged_order_requires_a_stable_difference_and_cannot_swap_back() {
        let mut order = StagedOrder::new(&[1.0, 2.0]);
        for _ in 0..2 {
            order.ranks[0].observe(600.0, 2);
            order.ranks[1].observe(800.0, 8);
        }
        order.adapt();
        assert_eq!(order.indices, [0, 1]);
        order.adapt();
        assert_eq!(order.indices, [1, 0]);
        for _ in 0..8 {
            order.ranks[0].observe(100.0, 8);
            order.ranks[1].observe(900.0, 2);
            order.adapt();
            assert_eq!(order.indices, [1, 0]);
        }
    }

    #[test]
    fn staged_cost_table_reserves_twenty_percent() {
        assert!(SelectionCost::LargeDictionary.uses_kernel(0.10));
        assert!(!SelectionCost::LargeDictionary.uses_kernel(0.15));
        assert!(SelectionCost::NullableDictionary.uses_kernel(0.001));
        assert!(!SelectionCost::NullableDictionary.uses_kernel(0.02));
        assert!(!SelectionCost::Unknown.uses_kernel(0.001));
    }

    const DEFAULT_AUTO_THRESHOLD: usize = 32;

    fn builder_with_selection(selection: RowSelection) -> ReadPlanBuilder {
        ReadPlanBuilder::new(1024).with_selection(Some(selection))
    }

    fn predicate_plan(
        pattern: Vec<bool>,
        batch_size: usize,
        limit: Option<usize>,
    ) -> ReadPlanBuilder {
        use crate::arrow::ProjectionMask;
        use crate::arrow::array_reader::StructArrayReader;
        use crate::arrow::array_reader::test_util::make_int32_page_reader;
        use crate::arrow::arrow_reader::ArrowPredicateFn;
        use arrow_schema::{DataType as ArrowType, Field, Fields};

        let total_rows = pattern.len();
        let data: Vec<i32> = (0..total_rows as i32).collect();
        let levels = vec![0; total_rows];
        let leaf = make_int32_page_reader(&data, &levels, &levels, 0, 0, None);
        let struct_type = ArrowType::Struct(Fields::from(vec![Field::new(
            "c0",
            ArrowType::Int32,
            false,
        )]));
        let struct_reader = StructArrayReader::new(struct_type, vec![leaf], 0, 0, false, None);

        let mut offset = 0usize;
        let mut predicate = ArrowPredicateFn::new(ProjectionMask::all(), move |batch| {
            let end = offset + batch.num_rows();
            let filter = BooleanArray::from(pattern[offset..end].to_vec());
            offset = end;
            Ok(filter)
        });
        let options = PredicateOptions::new(Box::new(struct_reader), &mut predicate);
        let options = match limit {
            Some(limit) => options.with_limit(limit, total_rows),
            None => options,
        };

        ReadPlanBuilder::new(batch_size)
            .with_predicate_options(options)
            .unwrap()
    }

    fn first_n_matches(pattern: &[bool], limit: usize) -> Vec<bool> {
        let mut remaining = limit;
        pattern
            .iter()
            .map(|selected| {
                if *selected && remaining != 0 {
                    remaining -= 1;
                    true
                } else {
                    false
                }
            })
            .collect()
    }

    fn assert_limit_case(name: &str, pattern: Vec<bool>, batch_size: usize, limit: usize) {
        let expected_bits = first_n_matches(&pattern, limit);
        let expected = RowSelection::from_filters(&[BooleanArray::from(expected_bits)]);
        let builder = predicate_plan(pattern, batch_size, Some(limit));
        let actual = builder
            .selection()
            .unwrap_or_else(|| panic!("{name}: limited mixed predicate must produce a selection"));

        assert_eq!(actual, &expected, "{name}: logical selection");

        let current_strategy = expected.auto_selection_strategy(DEFAULT_AUTO_THRESHOLD);
        assert_eq!(
            builder.resolve_selection_strategy(),
            current_strategy,
            "{name}: Auto strategy"
        );
        assert_eq!(
            actual.as_mask().is_some(),
            current_strategy == RowSelectionStrategy::Mask,
            "{name}: backing selected by capped Auto"
        );
    }

    #[test]
    fn preferred_selection_strategy_prefers_mask_by_default() {
        let selection = RowSelection::from(vec![RowSelector::select(8)]);
        let builder = builder_with_selection(selection);
        assert_eq!(
            builder.resolve_selection_strategy(),
            RowSelectionStrategy::Mask
        );
    }

    #[test]
    fn preferred_selection_strategy_prefers_selectors_when_threshold_small() {
        let selection = RowSelection::from(vec![RowSelector::select(8)]);
        let builder = builder_with_selection(selection)
            .with_row_selection_policy(RowSelectionPolicy::Auto { threshold: 1 });
        assert_eq!(
            builder.resolve_selection_strategy(),
            RowSelectionStrategy::Selectors
        );
    }

    #[test]
    fn preferred_selection_strategy_handles_dense_mask_backing() {
        let bits: Vec<_> = (0..16).map(|i| i % 2 == 0).collect();
        let selection = RowSelection::from_boolean_buffer(BooleanBuffer::from(bits));
        let builder = builder_with_selection(selection)
            .with_row_selection_policy(RowSelectionPolicy::Auto { threshold: 4 });
        assert_eq!(
            builder.resolve_selection_strategy(),
            RowSelectionStrategy::Mask
        );
    }

    #[test]
    fn preferred_selection_strategy_handles_sparse_mask_backing() {
        let bits: Vec<_> = (0..128).map(|i| i < 64).collect();
        let selection = RowSelection::from_boolean_buffer(BooleanBuffer::from(bits));
        let builder = builder_with_selection(selection)
            .with_row_selection_policy(RowSelectionPolicy::Auto { threshold: 32 });
        assert_eq!(
            builder.resolve_selection_strategy(),
            RowSelectionStrategy::Selectors
        );
    }

    #[test]
    fn preferred_selection_strategy_preserves_mask_threshold_boundaries() {
        let mask = BooleanBuffer::from(vec![true; 8]);

        let empty = builder_with_selection(RowSelection::from_boolean_buffer(
            BooleanBuffer::new_unset(0),
        ));
        assert_eq!(
            empty.resolve_selection_strategy(),
            RowSelectionStrategy::Mask
        );

        let disabled = builder_with_selection(RowSelection::from_boolean_buffer(mask.clone()))
            .with_row_selection_policy(RowSelectionPolicy::Auto { threshold: 0 });
        assert_eq!(
            disabled.resolve_selection_strategy(),
            RowSelectionStrategy::Selectors
        );

        // Equality does not satisfy the policy's strict inequality: 8 < 1 * 8.
        let equal = builder_with_selection(RowSelection::from_boolean_buffer(mask.clone()))
            .with_row_selection_policy(RowSelectionPolicy::Auto { threshold: 8 });
        assert_eq!(
            equal.resolve_selection_strategy(),
            RowSelectionStrategy::Selectors
        );

        let above = builder_with_selection(RowSelection::from_boolean_buffer(mask))
            .with_row_selection_policy(RowSelectionPolicy::Auto { threshold: 9 });
        assert_eq!(
            above.resolve_selection_strategy(),
            RowSelectionStrategy::Mask
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Takes too long
    fn preferred_selection_strategy_mask_matches_selector_backing() {
        use rand::{RngExt, rng};

        let mut rand = rng();
        for _ in 0..200 {
            let len = rand.random_range(0..256);
            let bits: Vec<_> = (0..len).map(|_| rand.random_bool(0.5)).collect();
            let mask_backed = RowSelection::from_boolean_buffer(BooleanBuffer::from(bits.clone()));
            let selector_backed = RowSelection::from_filters(&[BooleanArray::from(bits)]);

            for threshold in [0, 1, 2, 8, 32, 64, usize::MAX] {
                assert_eq!(
                    mask_backed.auto_selection_strategy(threshold),
                    selector_backed.auto_selection_strategy(threshold),
                    "strategy differs for len {len} and threshold {threshold}"
                );
            }
        }
    }

    #[test]
    fn mask_plan_trims_trailing_skips_before_chunking() {
        let mut plan = ReadPlanBuilder::new(8)
            .with_selection(Some(RowSelection::from(vec![
                RowSelector::select(1),
                RowSelector::skip(7),
            ])))
            .with_row_selection_policy(RowSelectionPolicy::Mask)
            .build();
        let RowSelectionCursor::Mask(cursor) = plan.row_selection_cursor_mut() else {
            panic!("expected a Mask cursor");
        };

        let chunk = cursor.next_chunk(8).unwrap();
        assert_eq!(chunk.chunk_rows, 1);
        assert_eq!(chunk.selected_rows, 1);
        assert!(cursor.is_empty());
    }

    #[test]
    fn selectors_policy_lowers_mask_backed_selection() {
        let selection = RowSelection::from_boolean_buffer(BooleanBuffer::from(vec![
            true, false, false, true, true,
        ]));
        let mut plan = ReadPlanBuilder::new(4)
            .with_selection(Some(selection))
            .with_row_selection_policy(RowSelectionPolicy::Selectors)
            .build();
        let RowSelectionCursor::Selectors(cursor) = plan.row_selection_cursor_mut() else {
            panic!("expected a Selectors cursor");
        };

        assert_eq!(cursor.next_selector(), RowSelector::select(1));
        assert_eq!(cursor.next_selector(), RowSelector::skip(2));
        assert_eq!(cursor.next_selector(), RowSelector::select(2));
        assert!(cursor.is_empty());
    }

    #[test]
    fn mask_backed_plan_respects_loaded_row_ranges() {
        // Start from a non-byte-aligned BooleanBuffer::slice so the mask-backed
        // cursor path (new_mask_from_buffer) is exercised with a bit offset.
        // Rows 0 and 11 of 12 are selected; pages for rows [4, 10) are unloaded.
        let mut bits = vec![false; 5];
        bits.push(true); // row 0
        bits.extend(std::iter::repeat_n(false, 10)); // rows 1..=10
        bits.push(true); // row 11
        bits.extend([true, false, true]); // trailing padding outside the slice
        let mask = BooleanBuffer::from(bits).slice(5, 12);
        let selection = RowSelection::from_boolean_buffer(mask);
        assert!(selection.as_mask().is_some());

        let loaded = LoadedRowRanges::from_selection(RowSelection::from(vec![
            RowSelector::select(4),
            RowSelector::skip(6),
            RowSelector::select(2),
        ]));

        let mut plan = ReadPlanBuilder::new(12)
            .with_selection(Some(selection))
            .with_row_selection_policy(RowSelectionPolicy::Mask)
            .with_loaded_row_ranges(Some(loaded))
            .build();
        let RowSelectionCursor::Mask(cursor) = plan.row_selection_cursor_mut() else {
            panic!("expected a Mask cursor");
        };

        // The first chunk stops at its final selected row instead of carrying
        // trailing skipped rows to the loaded range boundary.
        let first = cursor.next_chunk(12).unwrap();
        assert_eq!(first.initial_skip, 0);
        assert_eq!(first.chunk_rows, 1);
        assert_eq!(first.selected_rows, 1);

        // The second chunk skips directly to the next selected row.
        let second = cursor.next_chunk(12).unwrap();
        assert_eq!(second.initial_skip, 10);
        assert_eq!(second.chunk_rows, 1);
        assert_eq!(second.selected_rows, 1);
        assert!(cursor.is_empty());
    }

    #[test]
    fn with_predicate_options_capped_auto_preserves_limit_and_padding_boundaries() {
        let fragmented_early_limit = (0..37)
            .map(|row| matches!(row, 0 | 3 | 7 | 9 | 12 | 18 | 24 | 36))
            .collect();
        assert_limit_case("fragmented early limit", fragmented_early_limit, 16, 3);

        assert_limit_case(
            "selector-friendly padded tail",
            vec![true; 4_097],
            1_024,
            1_024,
        );
    }

    #[test]
    fn with_predicate_options_limit_pads_tail_when_no_prior_selection() {
        use crate::arrow::ProjectionMask;
        use crate::arrow::array_reader::StructArrayReader;
        use crate::arrow::array_reader::test_util::make_int32_page_reader;
        use crate::arrow::arrow_reader::ArrowPredicateFn;
        use arrow_schema::{DataType as ArrowType, Field, Fields};

        // 100 rows, all match the predicate. Limit stops the loop after 10
        // matches — but the resulting RowSelection must still describe the
        // full 100-row row group (90 trailing rows as "not selected"), not
        // only the 10 rows we happened to evaluate before breaking.
        const TOTAL_ROWS: usize = 100;
        const LIMIT: usize = 10;

        let data: Vec<i32> = (0..TOTAL_ROWS as i32).collect();
        let levels = vec![0; TOTAL_ROWS];
        let leaf = make_int32_page_reader(&data, &levels, &levels, 0, 0, None);
        let struct_type = ArrowType::Struct(Fields::from(vec![Field::new(
            "c0",
            ArrowType::Int32,
            false,
        )]));
        let struct_reader = StructArrayReader::new(struct_type, vec![leaf], 0, 0, false, None);

        let mut predicate = ArrowPredicateFn::new(ProjectionMask::all(), |batch| {
            Ok(BooleanArray::from(vec![true; batch.num_rows()]))
        });

        let builder = ReadPlanBuilder::new(16)
            .with_predicate_options(
                PredicateOptions::new(Box::new(struct_reader), &mut predicate)
                    .with_limit(LIMIT, TOTAL_ROWS),
            )
            .unwrap();

        let selection = builder
            .selection()
            .expect("limit-driven early break must produce a selection");

        // `row_count` counts selected rows — must equal the limit.
        assert_eq!(selection.row_count(), LIMIT);

        // Total rows covered (selects + skips) must equal the full row group
        // so downstream offset/limit math stays in absolute-row space.
        assert_eq!(
            selection.total_row_count(),
            TOTAL_ROWS,
            "selection must span the full row group, not only the prefix evaluated before the limit"
        );
    }

    #[test]
    fn with_predicate_options_preserves_mask_selection() {
        use crate::arrow::ProjectionMask;
        use crate::arrow::array_reader::StructArrayReader;
        use crate::arrow::array_reader::test_util::make_int32_page_reader;
        use crate::arrow::arrow_reader::ArrowPredicateFn;
        use arrow_schema::{DataType as ArrowType, Field, Fields};

        let data: Vec<i32> = (0..6).collect();
        let levels = vec![0; data.len()];
        let leaf = make_int32_page_reader(&data, &levels, &levels, 0, 0, None);
        let struct_type = ArrowType::Struct(Fields::from(vec![Field::new(
            "c0",
            ArrowType::Int32,
            false,
        )]));
        let struct_reader = StructArrayReader::new(struct_type, vec![leaf], 0, 0, false, None);

        let prior = RowSelection::from_boolean_buffer(BooleanBuffer::from(vec![
            true, false, true, true, false, true,
        ]));
        let mut filters = vec![BooleanArray::from(vec![true, false, true, false])];
        let mut predicate = ArrowPredicateFn::new(ProjectionMask::all(), move |batch| {
            assert_eq!(batch.num_rows(), 4);
            Ok(filters.remove(0))
        });

        let builder = ReadPlanBuilder::new(16)
            .with_selection(Some(prior))
            .with_predicate_options(PredicateOptions::new(
                Box::new(struct_reader),
                &mut predicate,
            ))
            .unwrap();

        let selection = builder.selection().unwrap();
        assert!(selection.as_mask().is_some());

        let expected = RowSelection::from_boolean_buffer(BooleanBuffer::from(vec![
            true, false, false, true, false, false,
        ]));
        assert_eq!(selection, &expected);
    }

    #[test]
    fn with_predicate_options_limit_handles_null_filters() {
        use crate::arrow::ProjectionMask;
        use crate::arrow::array_reader::StructArrayReader;
        use crate::arrow::array_reader::test_util::make_int32_page_reader;
        use crate::arrow::arrow_reader::ArrowPredicateFn;
        use arrow_schema::{DataType as ArrowType, Field, Fields};

        const TOTAL_ROWS: usize = 100;
        const LIMIT: usize = 10;

        let data: Vec<i32> = (0..TOTAL_ROWS as i32).collect();
        let levels = vec![0; TOTAL_ROWS];
        let leaf = make_int32_page_reader(&data, &levels, &levels, 0, 0, None);
        let struct_type = ArrowType::Struct(Fields::from(vec![Field::new(
            "c0",
            ArrowType::Int32,
            false,
        )]));
        let struct_reader = StructArrayReader::new(struct_type, vec![leaf], 0, 0, false, None);

        let mut predicate = ArrowPredicateFn::new(ProjectionMask::all(), |batch| {
            Ok((0..batch.num_rows())
                .map(|i| match i % 4 {
                    0 | 2 => Some(true),
                    1 => None,
                    _ => Some(false),
                })
                .collect::<BooleanArray>())
        });

        let builder = ReadPlanBuilder::new(16)
            .with_predicate_options(
                PredicateOptions::new(Box::new(struct_reader), &mut predicate)
                    .with_limit(LIMIT, TOTAL_ROWS),
            )
            .unwrap();

        let selection = builder
            .selection()
            .expect("limit-driven early break must produce a selection");

        assert_eq!(selection.row_count(), LIMIT);

        assert_eq!(selection.total_row_count(), TOTAL_ROWS);
    }
}
