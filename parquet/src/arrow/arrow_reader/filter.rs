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

use crate::arrow::ProjectionMask;
use crate::arrow::arrow_reader::{RowSelection, RowSelectionPolicy};
use crate::schema::types::SchemaDescriptor;
use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch};
use arrow_buffer::BooleanBuffer;
use arrow_schema::ArrowError;
use arrow_select::filter::{SlicesIterator, filter_record_batch, prep_null_mask_filter};
use std::fmt::{Debug, Formatter};

/// Whether to evaluate a Parquet row filter in dictionary code space.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DictionaryFilterMode {
    /// Evaluate only decoded rows.
    #[default]
    Off,
    /// Evaluate eligible chunks in code space.
    On,
    /// Check the code-space selection against ordinary value evaluation.
    Verify,
}

/// Per-predicate dictionary filter policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DictionaryFilterOptions {
    /// Execution mode.
    pub mode: DictionaryFilterMode,
    /// Maximum dictionary entries as a percentage of the chunk's row count.
    pub max_percent: u8,
}

/// Predicate result for the dictionary entries, in code order, and for NULL.
#[derive(Debug)]
pub struct DictionaryVerdict {
    /// One bit per dictionary entry.
    pub entries: BooleanBuffer,
    /// Whether a row with a NULL value passes.
    pub null: bool,
}

/// Work performed while applying a dictionary-aware predicate.
/// Chunk counts refer to predicate evaluations. A mixed chunk may count once
/// under each of the verdict and value paths.
#[derive(Debug, Default)]
pub struct DictionaryFilterMetrics {
    /// Chunks for which code space supplied a selection.
    pub chunks_verdict: usize,
    /// Selected input rows tested in code space.
    pub rows_verdict: usize,
    /// Code-space rows that passed.
    pub rows_verdict_matched: usize,
    /// Chunks for which selected rows were evaluated as values.
    pub chunks_value: usize,
    /// Selected input rows evaluated in value space, excluding the verify oracle.
    pub rows_value: usize,
    /// Rows evaluated by the verify oracle, including any value fallback.
    pub rows_verified: usize,
    /// Chunks with selected data in a non-dictionary encoding.
    pub fallback_plain: usize,
    /// Chunks whose dictionary exceeds the configured percentage.
    pub fallback_limit: usize,
    /// Predicate/chunk evaluations with more than one projected leaf.
    pub fallback_columns: usize,
    /// Chunks where dictionary evaluation failed.
    pub fallback_error: usize,
    /// Chunks refused for other reasons (e.g. applicability or legacy levels).
    pub fallback_unsupported: usize,
    /// Dictionary entries presented to predicate evaluation.
    pub entries_tested: usize,
    /// Dictionary index runs evaluated, counting a packed run as one.
    pub runs_evaluated: usize,
    /// Row groups discarded by a verdict that rejects the entire dictionary.
    pub row_groups_pruned_dictionary: usize,
    /// First data-page offsets identifying those row groups within a file.
    /// A prepared scan can evaluate the same row group in several units.
    pub pruned_row_group_offsets: Vec<i64>,
}

/// A predicate operating on [`RecordBatch`]
///
/// See also:
/// * [`RowFilter`] for more information  on applying filters during the
///   Parquet decoding process.
/// * [`ArrowPredicateFn`] for a concrete implementation based on a function
pub trait ArrowPredicate: Send + 'static {
    /// Returns the [`ProjectionMask`] that describes the columns required
    /// to evaluate this predicate.
    ///
    /// All projected columns will be provided in the `batch` passed to
    /// [`evaluate`](Self::evaluate). The projection mask should be as small as
    /// possible because any columns needed for the overall projection mask are
    /// decoded again after a predicate is applied.
    fn projection(&self) -> &ProjectionMask;

    /// Evaluate this predicate for the given [`RecordBatch`] containing the columns
    /// identified by [`Self::projection`]
    ///
    /// Must return a [`BooleanArray`] that has the same length as the input
    /// `batch` where each row indicates whether the row should be returned:
    /// * `true`:the row should be returned
    /// * `false` or `null`: the row should not be returned
    fn evaluate(&mut self, batch: RecordBatch) -> Result<BooleanArray, ArrowError>;

    /// Opt into dictionary filtering and its metrics. The projection must be
    /// one flat leaf, and a verdict must be a deterministic function of its value.
    fn dictionary_filter_options(&self) -> Option<DictionaryFilterOptions> {
        None
    }

    /// Whether this predicate can evaluate dictionary entries in this scan.
    fn dictionary_filter_eligible(&self) -> bool {
        true
    }

    /// Evaluate a chunk's dictionary and NULL. Return `None` when the predicate
    /// cannot evaluate dictionary entries. Errors are retried through values.
    fn evaluate_dictionary(
        &mut self,
        _leaf: usize,
        _dictionary: &ArrayRef,
    ) -> Result<Option<DictionaryVerdict>, ArrowError> {
        Ok(None)
    }

    /// Report the dictionary and value work for this predicate.
    fn record_dictionary_filter(&mut self, _metrics: DictionaryFilterMetrics) {}
}

/// An [`ArrowPredicate`] created from an [`FnMut`] and a [`ProjectionMask`]
///
/// See [`RowFilter`] for more information on applying filters during the
/// Parquet decoding process.
///
/// The function is passed `RecordBatch`es with only the columns specified in
/// the [`ProjectionMask`].
///
/// The function must return a [`BooleanArray`] that has the same length as the
/// input `batch` where each row indicates whether the row should be returned:
/// * `true`: the row should be returned
/// * `false` or `null`: the row should not be returned
///
/// # Example:
///
/// Given an input schema: `"a:int64", "b:int64"`, you can create a predicate that
/// evaluates `b > 0` like this:
///
/// ```
/// # use std::sync::Arc;
/// # use arrow::compute::kernels::cmp::gt;
/// # use arrow_array::{BooleanArray, Int64Array, RecordBatch};
/// # use arrow_array::cast::AsArray;
/// # use arrow_array::types::Int64Type;
/// # use parquet::arrow::arrow_reader::ArrowPredicateFn;
/// # use parquet::arrow::ProjectionMask;
/// # use parquet::schema::types::{SchemaDescriptor, Type};
/// # use parquet::basic; // note there are two `Type`s that are different
/// # // Schema for a table with one columns: "a" (int64) and "b" (int64)
/// # let descriptor = SchemaDescriptor::new(
/// #  Arc::new(
/// #    Type::group_type_builder("my_schema")
/// #      .with_fields(vec![
/// #        Arc::new(
/// #         Type::primitive_type_builder("a", basic::Type::INT64)
/// #          .build().unwrap()
/// #        ),
/// #        Arc::new(
/// #         Type::primitive_type_builder("b", basic::Type::INT64)
/// #          .build().unwrap()
/// #        ),
/// #     ])
/// #     .build().unwrap()
/// #  )
/// # );
/// // Create a mask for selecting only the second column "b" (index 1)
/// let projection_mask = ProjectionMask::leaves(&descriptor, [1]);
/// // Closure that evaluates "b > 0"
/// let predicate = |batch: RecordBatch| {
///    let scalar_0 = Int64Array::new_scalar(0);
///    let column = batch.column(0).as_primitive::<Int64Type>();
///    // call the gt kernel to compute `>` which returns a BooleanArray
///    gt(column, &scalar_0)
///  };
/// // Create ArrowPredicateFn that can be passed to RowFilter
/// let arrow_predicate = ArrowPredicateFn::new(projection_mask, predicate);
/// ```
pub struct ArrowPredicateFn<F> {
    f: F,
    projection: ProjectionMask,
}

impl<F> ArrowPredicateFn<F>
where
    F: FnMut(RecordBatch) -> Result<BooleanArray, ArrowError> + Send + 'static,
{
    /// Create a new [`ArrowPredicateFn`] that invokes `f` on the columns
    /// specified in `projection`.
    pub fn new(projection: ProjectionMask, f: F) -> Self {
        Self { f, projection }
    }
}

impl<F> ArrowPredicate for ArrowPredicateFn<F>
where
    F: FnMut(RecordBatch) -> Result<BooleanArray, ArrowError> + Send + 'static,
{
    fn projection(&self) -> &ProjectionMask {
        &self.projection
    }

    fn evaluate(&mut self, batch: RecordBatch) -> Result<BooleanArray, ArrowError> {
        (self.f)(batch)
    }
}

/// Filter applied *during* the parquet read process
///
/// See example on [`ArrowReaderBuilder::with_row_filter`]
///
/// [`RowFilter`] applies predicates in order, after decoding only the columns
/// required. As predicates eliminate rows, fewer rows from subsequent columns
/// may be required, thus potentially reducing IO and decode. This process is
/// also known as *push down* filtering and  *late materialization*.
///
/// A `RowFilter` consists of a list of [`ArrowPredicate`]s. Only the rows for which
/// all the predicates evaluate to `true` will be returned.
/// Any [`RowSelection`] provided to the reader will be applied prior
/// to the first predicate, and each predicate in turn will then be used to compute
/// a more refined [`RowSelection`] used when evaluating the subsequent predicates.
///
/// Once all predicates have been evaluated, the final [`RowSelection`] is applied
/// to the top-level [`ProjectionMask`] to produce the final output [`RecordBatch`].
///
/// This design has a couple of implications:
///
/// * [`RowFilter`] can be used to skip entire pages, and thus IO, in addition to CPU decode overheads
/// * Columns may be decoded multiple times if they appear in multiple [`ProjectionMask`].
///   Consecutive predicates whose projection is the same single top-level, non-repeated
///   column are evaluated together on one decoded stream, so ordering such predicates
///   next to each other avoids decoding that column again
/// * IO will be deferred until needed by a [`ProjectionMask`]
///
/// As such there is a trade-off between a single large predicate, or multiple predicates,
/// that will depend on the shape of the data. Whilst multiple smaller predicates may
/// minimise the amount of data scanned/decoded, it may not be faster overall.
///
/// For example, if a predicate that needs a single column of data filters out all but
/// 1% of the rows, applying it as one of the early `ArrowPredicateFn` will likely significantly
/// improve performance.
///
/// As a counter example, if a predicate needs several columns of data to evaluate but
/// leaves 99% of the rows, it may be better to not filter the data from parquet and
/// apply the filter after the RecordBatch has been fully decoded.
///
/// Additionally, even if a predicate eliminates a moderate number of rows, it may still be faster
/// to filter the data after the RecordBatch has been fully decoded, if the eliminated rows are
/// not contiguous.
///
/// [`RowSelection`]: crate::arrow::arrow_reader::RowSelection
/// [`ArrowReaderBuilder::with_row_filter`]: crate::arrow::arrow_reader::ArrowReaderBuilder::with_row_filter
pub struct RowFilter {
    /// A list of [`ArrowPredicate`]
    pub(crate) predicates: Vec<Box<dyn ArrowPredicate>>,
}

impl Debug for RowFilter {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "RowFilter {{ {} predicates: }}", self.predicates.len())
    }
}

impl RowFilter {
    /// Create a new [`RowFilter`] from an array of [`ArrowPredicate`]
    pub fn new(predicates: Vec<Box<dyn ArrowPredicate>>) -> Self {
        Self { predicates }
    }
    /// Returns the inner predicates
    pub fn predicates(&self) -> &Vec<Box<dyn ArrowPredicate>> {
        &self.predicates
    }
    /// Returns the inner predicates, consuming self
    pub fn into_predicates(self) -> Vec<Box<dyn ArrowPredicate>> {
        self.predicates
    }

    /// Fuse consecutive predicates on the same single top-level, non-repeated leaf.
    /// This avoids repeated decoding or predicate-cache replay of that column.
    pub(crate) fn fuse_same_projection(self, parquet_schema: &SchemaDescriptor) -> Self {
        let mut predicates: Vec<Box<dyn ArrowPredicate>> =
            Vec::with_capacity(self.predicates.len());
        let mut group: Vec<Box<dyn ArrowPredicate>> = Vec::new();
        let flush_group = |predicates: &mut Vec<Box<dyn ArrowPredicate>>,
                           group: &mut Vec<Box<dyn ArrowPredicate>>| {
            if group.len() > 1 && can_fuse_projection(group[0].projection(), parquet_schema) {
                let group = std::mem::take(group);
                predicates.push(Box::new(FusedPredicate::new(group)));
            } else {
                predicates.append(group);
            }
        };

        for predicate in self.predicates {
            if predicate
                .dictionary_filter_options()
                .is_some_and(|options| options.mode != DictionaryFilterMode::Off)
            {
                flush_group(&mut predicates, &mut group);
                predicates.push(predicate);
                continue;
            }
            if group
                .last()
                .is_some_and(|last| last.projection() != predicate.projection())
            {
                flush_group(&mut predicates, &mut group);
            }
            group.push(predicate);
        }
        flush_group(&mut predicates, &mut group);

        Self { predicates }
    }
}

/// Restrict fusion to one top-level, non-repeated leaf to limit compaction costs.
fn can_fuse_projection(projection: &ProjectionMask, parquet_schema: &SchemaDescriptor) -> bool {
    let mut leaf_indices =
        (0..parquet_schema.num_columns()).filter(|idx| projection.leaf_included(*idx));
    let Some(leaf_idx) = leaf_indices.next() else {
        return false;
    };
    if leaf_indices.next().is_some() {
        return false;
    }

    let column = parquet_schema.column(leaf_idx);
    column.path().parts().len() == 1 && column.max_rep_level() == 0
}

/// Evaluate same-projection predicates in order on one decoded batch.
/// Later predicates see only surviving rows, sliced when contiguous or compacted
/// otherwise.
struct FusedPredicate {
    /// At least two predicates, all with the same projection.
    predicates: Vec<Box<dyn ArrowPredicate>>,
    /// Chooses how intermediate selections are composed. The output is always
    /// a [`BooleanArray`], so this is independent of the reader's
    /// [`RowSelectionPolicy`], which governs decoding.
    composition_policy: RowSelectionPolicy,
}

impl FusedPredicate {
    /// Create a fused predicate from at least two predicates that share one
    /// projection.
    fn new(predicates: Vec<Box<dyn ArrowPredicate>>) -> Self {
        debug_assert!(predicates.len() > 1);
        debug_assert!(
            predicates
                .windows(2)
                .all(|pair| pair[0].projection() == pair[1].projection())
        );
        Self {
            predicates,
            composition_policy: RowSelectionPolicy::default(),
        }
    }
}

impl Debug for FusedPredicate {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FusedPredicate {{ {} predicates }}",
            self.predicates.len()
        )
    }
}

impl ArrowPredicate for FusedPredicate {
    fn projection(&self) -> &ProjectionMask {
        self.predicates[0].projection()
    }

    fn dictionary_filter_options(&self) -> Option<DictionaryFilterOptions> {
        self.predicates
            .iter()
            .find_map(|predicate| predicate.dictionary_filter_options())
    }

    fn record_dictionary_filter(&mut self, metrics: DictionaryFilterMetrics) {
        if let Some(predicate) = self
            .predicates
            .iter_mut()
            .find(|p| p.dictionary_filter_options().is_some())
        {
            predicate.record_dictionary_filter(metrics);
        }
    }

    fn evaluate(&mut self, batch: RecordBatch) -> Result<BooleanArray, ArrowError> {
        let num_rows = batch.num_rows();
        // Positions in the original batch; None until a predicate rejects rows.
        let mut selection: Option<RowSelection> = None;
        let mut filtered_batch = batch;
        let last_predicate_idx = self.predicates.len() - 1;

        for (idx, predicate) in self.predicates.iter_mut().enumerate() {
            let filter = evaluate_predicate(predicate.as_mut(), filtered_batch.clone())?;
            // No mapping is needed if all preceding predicates accepted every row.
            if idx == last_predicate_idx && selection.is_none() {
                return Ok(filter);
            }
            let true_count = filter.true_count();
            if true_count == 0 {
                return Ok(BooleanArray::new(BooleanBuffer::new_unset(num_rows), None));
            }
            if true_count == filter.len() {
                continue;
            }

            let predicate_selection = RowSelection::from_boolean_buffer(filter.values().clone());
            selection = Some(match selection.take() {
                // Only the accumulated selection drives the composition
                // algorithm, so adapt it once, right before it is used.
                Some(prev) => self
                    .composition_policy
                    .apply(prev)
                    .and_then(&predicate_selection),
                None => predicate_selection,
            });
            if idx != last_predicate_idx {
                filtered_batch = narrow_batch(&filtered_batch, &filter, true_count)?;
            }
        }

        let mask = match selection {
            Some(selection) => selection.into_boolean_buffer(),
            None => BooleanBuffer::new_set(num_rows),
        };
        debug_assert_eq!(mask.len(), num_rows);
        Ok(BooleanArray::new(mask, None))
    }
}

/// Validate the predicate result length and treat nulls as false.
fn evaluate_predicate(
    predicate: &mut dyn ArrowPredicate,
    batch: RecordBatch,
) -> Result<BooleanArray, ArrowError> {
    let input_rows = batch.num_rows();
    let filter = predicate.evaluate(batch)?;
    if filter.len() != input_rows {
        return Err(ArrowError::InvalidArgumentError(format!(
            "ArrowPredicate predicate returned {} rows, expected {input_rows}",
            filter.len()
        )));
    }
    Ok(match filter.null_count() {
        0 => filter,
        _ => prep_null_mask_filter(&filter),
    })
}

/// Restrict `batch` to the rows `filter` accepts, slicing zero-copy when they
/// form one contiguous range and compacting otherwise.
///
/// `filter` must be null-free and accept some but not all rows.
fn narrow_batch(
    batch: &RecordBatch,
    filter: &BooleanArray,
    true_count: usize,
) -> Result<RecordBatch, ArrowError> {
    let mut slices = SlicesIterator::new(filter);
    let (start, end) = slices
        .next()
        .expect("a partially selected filter has a true slice");
    if end - start == true_count {
        Ok(batch.slice(start, end - start))
    } else {
        filter_record_batch(batch, filter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow::array_reader::ArrayReader;
    use crate::arrow::array_reader::StructArrayReader;
    use crate::arrow::array_reader::test_util::make_int32_page_reader;
    use crate::arrow::arrow_reader::ReadPlanBuilder;
    use crate::schema::parser::parse_message_type;
    use arrow_array::Int32Array;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int32Type;
    use arrow_schema::{DataType, Field, Fields, Schema};
    use std::sync::Arc;

    fn test_schema() -> SchemaDescriptor {
        let schema = parse_message_type(
            "message schema {
                REQUIRED INT32 a;
                REQUIRED INT32 b;
                REQUIRED INT32 c;
                REQUIRED GROUP nested { REQUIRED INT32 d; }
                REPEATED INT32 e;
            }",
        )
        .unwrap();
        SchemaDescriptor::new(Arc::new(schema))
    }

    #[test]
    fn can_fuse_projection_requires_one_top_level_leaf() {
        let schema = test_schema();

        let one_leaf = ProjectionMask::leaves(&schema, [1]);
        let two_leaves = ProjectionMask::leaves(&schema, [0, 2]);
        let nested_leaf = ProjectionMask::leaves(&schema, [3]);
        let repeated_leaf = ProjectionMask::leaves(&schema, [4]);
        assert!(can_fuse_projection(&one_leaf, &schema));
        assert!(!can_fuse_projection(&two_leaves, &schema));
        assert!(!can_fuse_projection(&nested_leaf, &schema));
        assert!(!can_fuse_projection(&repeated_leaf, &schema));
        assert!(!can_fuse_projection(
            &ProjectionMask::none(schema.num_columns()),
            &schema
        ));
    }

    fn accept_all(projection: ProjectionMask) -> Box<dyn ArrowPredicate> {
        Box::new(ArrowPredicateFn::new(projection, |batch| {
            Ok(BooleanArray::from(vec![true; batch.num_rows()]))
        }))
    }

    #[test]
    fn fuse_same_projection_groups_consecutive_eligible_predicates() {
        let schema = test_schema();
        let a = ProjectionMask::leaves(&schema, [0]);
        let b = ProjectionMask::leaves(&schema, [1]);
        let ac = ProjectionMask::leaves(&schema, [0, 2]);
        let nested = ProjectionMask::leaves(&schema, [3]);

        let filter = RowFilter::new(vec![
            // fused
            accept_all(a.clone()),
            accept_all(a.clone()),
            // single predicate: kept as-is
            accept_all(b.clone()),
            // two leaves: not eligible for fusion
            accept_all(ac.clone()),
            accept_all(ac.clone()),
            // nested leaf: not eligible for fusion
            accept_all(nested.clone()),
            accept_all(nested.clone()),
            // fused
            accept_all(a.clone()),
            accept_all(a.clone()),
            accept_all(a.clone()),
        ])
        .fuse_same_projection(&schema);

        let projections: Vec<_> = filter
            .predicates()
            .iter()
            .map(|predicate| predicate.projection().clone())
            .collect();
        assert_eq!(
            projections,
            vec![a.clone(), b, ac.clone(), ac, nested.clone(), nested, a]
        );
        assert_eq!(format!("{filter:?}"), "RowFilter { 7 predicates: }");
    }

    #[test]
    fn fuse_same_projection_keeps_single_predicates_and_empty_filters() {
        let schema = test_schema();
        let a = ProjectionMask::leaves(&schema, [0]);
        let b = ProjectionMask::leaves(&schema, [1]);

        let filter =
            RowFilter::new(vec![accept_all(a), accept_all(b)]).fuse_same_projection(&schema);
        assert_eq!(filter.predicates().len(), 2);

        let filter = RowFilter::new(vec![]).fuse_same_projection(&schema);
        assert!(filter.predicates().is_empty());
    }

    fn int_batch(values: impl IntoIterator<Item = i32>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let values = Int32Array::from_iter_values(values);
        RecordBatch::try_new(schema, vec![Arc::new(values)]).unwrap()
    }

    /// A predicate over the `v` column that asserts `precondition` holds for
    /// every row it sees before returning `f(v)`.
    fn int_predicate(
        precondition: impl Fn(i32) -> bool + Send + 'static,
        f: impl Fn(i32) -> Option<bool> + Send + 'static,
    ) -> Box<dyn ArrowPredicate> {
        Box::new(ArrowPredicateFn::new(ProjectionMask::all(), move |batch| {
            let values = batch.column(0).as_primitive::<Int32Type>();
            assert!(
                values.values().iter().all(|v| precondition(*v)),
                "predicate saw rows rejected by a predecessor"
            );
            Ok(values.values().iter().map(|v| f(*v)).collect())
        }))
    }

    /// Fragmented survivors: roughly every other row.
    fn even() -> Box<dyn ArrowPredicate> {
        int_predicate(|_| true, |v| Some(v % 2 == 0))
    }

    /// Clustered survivors: one contiguous range.
    fn in_range() -> Box<dyn ArrowPredicate> {
        int_predicate(|_| true, |v| Some((10..40).contains(&v)))
    }

    /// Applies after `even`: mixes nulls (rejected) into the result.
    fn divisible_by_three_after_even() -> Box<dyn ArrowPredicate> {
        int_predicate(
            |v| v % 2 == 0,
            |v| if v % 7 == 0 { None } else { Some(v % 3 == 0) },
        )
    }

    fn all_true() -> Box<dyn ArrowPredicate> {
        int_predicate(|_| true, |_| Some(true))
    }

    fn all_false() -> Box<dyn ArrowPredicate> {
        int_predicate(|_| true, |_| Some(false))
    }

    fn never_called() -> Box<dyn ArrowPredicate> {
        Box::new(ArrowPredicateFn::new(ProjectionMask::all(), |_| {
            panic!("predicate evaluated after every row was rejected")
        }))
    }

    /// Evaluate `predicates` one after another, each on the rows kept by its
    /// predecessors, the way `RowFilter` applies separate predicates.
    fn sequential(batch: &RecordBatch, predicates: &mut [Box<dyn ArrowPredicate>]) -> Vec<bool> {
        let mut kept = vec![true; batch.num_rows()];
        for predicate in predicates {
            // Like `RowFilter`, stop once every row has been rejected.
            if !kept.contains(&true) {
                break;
            }
            let survivors = filter_record_batch(batch, &BooleanArray::from(kept.clone())).unwrap();
            let filter = predicate.evaluate(survivors).unwrap();
            for (filter_idx, keep) in kept.iter_mut().filter(|keep| **keep).enumerate() {
                *keep = filter.is_valid(filter_idx) && filter.value(filter_idx);
            }
        }
        kept
    }

    const POLICIES: [RowSelectionPolicy; 4] = [
        RowSelectionPolicy::Selectors,
        RowSelectionPolicy::Mask,
        RowSelectionPolicy::Auto { threshold: 32 },
        RowSelectionPolicy::Auto { threshold: 2 },
    ];

    type PredicateChain = fn() -> Vec<Box<dyn ArrowPredicate>>;

    #[test]
    fn fused_predicate_matches_sequential_evaluation() {
        let cases: Vec<PredicateChain> = vec![
            || vec![even(), divisible_by_three_after_even()],
            || vec![even(), in_range(), divisible_by_three_after_even()],
            || vec![in_range(), even()],
            || vec![even(), all_true(), divisible_by_three_after_even()],
            || vec![all_true(), even()],
            || vec![all_true(), all_true()],
            || vec![even(), all_false(), never_called()],
            || vec![all_false(), never_called()],
        ];

        for num_rows in [0, 1, 7, 97, 200] {
            let batch = int_batch(0..num_rows);
            for make_predicates in &cases {
                let expected = sequential(&batch, &mut make_predicates());
                for policy in POLICIES {
                    let mut fused = FusedPredicate {
                        predicates: make_predicates(),
                        composition_policy: policy,
                    };
                    let actual = fused.evaluate(batch.clone()).unwrap();
                    assert_eq!(actual.null_count(), 0);
                    assert_eq!(
                        actual.values().iter().collect::<Vec<_>>(),
                        expected,
                        "{num_rows} rows, {policy:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn fused_predicate_rejects_wrong_result_length() {
        let short = Box::new(ArrowPredicateFn::new(ProjectionMask::all(), |_| {
            Ok(BooleanArray::from(vec![true]))
        }));
        let mut fused = FusedPredicate::new(vec![all_true(), short]);
        let err = fused.evaluate(int_batch(0..4)).unwrap_err();
        assert!(
            err.to_string()
                .contains("ArrowPredicate predicate returned 1 rows, expected 4"),
            "{err}"
        );
    }

    #[test]
    fn narrow_batch_slices_contiguous_survivors() {
        let batch = int_batch(0..6);
        let filter = BooleanArray::from(vec![false, false, true, true, true, false]);
        let narrowed = narrow_batch(&batch, &filter, 3).unwrap();
        assert_eq!(
            narrowed.column(0).as_primitive::<Int32Type>().values(),
            &[2, 3, 4]
        );
        // Slicing shares the input buffer instead of copying it.
        assert_eq!(narrowed.column(0).to_data().buffers()[0].as_ptr(), unsafe {
            batch.column(0).to_data().buffers()[0].as_ptr().add(2 * 4)
        });

        let filter = BooleanArray::from(vec![false, true, false, true, false, false]);
        let narrowed = narrow_batch(&batch, &filter, 2).unwrap();
        assert_eq!(
            narrowed.column(0).as_primitive::<Int32Type>().values(),
            &[1, 3]
        );
    }

    /// Fusing predicates into one `ReadPlanBuilder::with_predicate` call
    /// produces the same selection as applying them one at a time.
    #[test]
    fn fused_predicate_matches_predicate_major_read_plan() {
        let data: Vec<i32> = (0..97).collect();
        let make_reader = || {
            let levels = vec![0; data.len()];
            let leaf = make_int32_page_reader(&data, &levels, &levels, 0, 0, None);
            let struct_type =
                DataType::Struct(Fields::from(vec![Field::new("c0", DataType::Int32, false)]));
            Box::new(StructArrayReader::new(
                struct_type,
                vec![leaf],
                0,
                0,
                false,
                None,
            )) as Box<dyn ArrayReader>
        };
        let make_predicates = || vec![even(), divisible_by_three_after_even()];

        let prior = RowSelection::from_filters(&[BooleanArray::from(
            (0..data.len()).map(|idx| idx % 5 != 0).collect::<Vec<_>>(),
        )]);
        for initial in [None, Some(prior)] {
            let mut sequential = ReadPlanBuilder::new(7).with_selection(initial.clone());
            for predicate in &mut make_predicates() {
                sequential = sequential
                    .with_predicate(make_reader(), predicate.as_mut())
                    .unwrap();
            }

            for policy in POLICIES {
                let mut fused = FusedPredicate::new(make_predicates());
                let plan = ReadPlanBuilder::new(7)
                    .with_selection(initial.clone())
                    .with_row_selection_policy(policy)
                    .with_predicate(make_reader(), &mut fused)
                    .unwrap();
                assert_eq!(plan.selection(), sequential.selection(), "{policy:?}");
            }
        }
    }
}

#[cfg(test)]
mod dictionary_tests {
    use super::*;
    use crate::DecodeResult;
    use crate::arrow::ArrowWriter;
    use crate::arrow::arrow_reader::{
        ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder, RowSelector,
    };
    use crate::arrow::push_decoder::ParquetPushDecoderBuilder;
    use crate::file::metadata::PageIndexPolicy;
    use crate::file::properties::{WriterProperties, WriterVersion};
    use arrow_array::{ArrayRef, Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use bytes::Bytes;
    use std::sync::{Arc, Mutex};

    struct StaticFilter {
        projection: ProjectionMask,
        mode: DictionaryFilterMode,
        accept_null: bool,
        accept_yes: bool,
        fail_dictionary: bool,
        metrics: Arc<Mutex<DictionaryFilterMetrics>>,
    }

    impl ArrowPredicate for StaticFilter {
        fn projection(&self) -> &ProjectionMask {
            &self.projection
        }

        fn evaluate(&mut self, batch: RecordBatch) -> Result<BooleanArray, ArrowError> {
            let values = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            Ok(values
                .iter()
                .map(|v| v.map_or(self.accept_null, |v| self.accept_yes && v == "yes"))
                .collect())
        }

        fn dictionary_filter_options(&self) -> Option<DictionaryFilterOptions> {
            Some(DictionaryFilterOptions {
                mode: self.mode,
                max_percent: 25,
            })
        }

        fn evaluate_dictionary(
            &mut self,
            leaf: usize,
            dictionary: &ArrayRef,
        ) -> Result<Option<DictionaryVerdict>, ArrowError> {
            assert_eq!(leaf, 0);
            if self.fail_dictionary {
                return Err(ArrowError::ComputeError("entry evaluation failed".into()));
            }
            let values = dictionary.as_any().downcast_ref::<StringArray>().unwrap();
            Ok(Some(DictionaryVerdict {
                entries: BooleanBuffer::collect_bool(values.len(), |i| {
                    self.accept_yes && values.value(i) == "yes"
                }),
                null: self.accept_null,
            }))
        }

        fn record_dictionary_filter(&mut self, metrics: DictionaryFilterMetrics) {
            *self.metrics.lock().unwrap() = metrics;
        }
    }

    #[test]
    fn dictionary_filter_in_push_decoder_with_sparse_page_requests() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("word", DataType::Utf8, true),
            Field::new("id", DataType::Int32, false),
        ]));
        let words = (0..512)
            .map(|i| match i % 3 {
                0 => None,
                1 => Some("yes"),
                _ => Some("no"),
            })
            .collect::<StringArray>();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(words),
                Arc::new(Int32Array::from_iter_values(0..512)),
            ],
        )
        .unwrap();
        let props = WriterProperties::builder()
            .set_data_page_row_count_limit(32)
            .set_write_batch_size(32)
            .build();
        let mut writer = ArrowWriter::try_new(Vec::new(), schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        let file = Bytes::from(writer.into_inner().unwrap());
        let metadata = ArrowReaderMetadata::load(
            &file,
            ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Required),
        )
        .unwrap();
        let builder = ParquetPushDecoderBuilder::new_with_metadata(metadata);
        let schema = builder.metadata().file_metadata().schema_descr();
        let output = ProjectionMask::leaves(schema, [1]);
        let metrics = Arc::new(Mutex::new(DictionaryFilterMetrics::default()));
        let predicate = StaticFilter {
            projection: ProjectionMask::leaves(schema, [0]),
            mode: DictionaryFilterMode::Verify,
            accept_null: true,
            accept_yes: true,
            fail_dictionary: false,
            metrics: Arc::clone(&metrics),
        };
        let mut decoder = builder
            .with_projection(output)
            .with_row_selection(RowSelection::from(vec![
                RowSelector::skip(32),
                RowSelector::select(16),
                RowSelector::skip(48),
                RowSelector::select(96),
                RowSelector::skip(320),
            ]))
            .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
            .build()
            .unwrap();
        let mut ids = Vec::new();
        loop {
            match decoder.try_next_reader().unwrap() {
                DecodeResult::Data(reader) => {
                    for batch in reader {
                        let batch = batch.unwrap();
                        ids.extend_from_slice(
                            batch
                                .column(0)
                                .as_any()
                                .downcast_ref::<Int32Array>()
                                .unwrap()
                                .values(),
                        );
                    }
                }
                DecodeResult::NeedsData(ranges) => {
                    let bytes = ranges
                        .iter()
                        .map(|range| file.slice(range.start as usize..range.end as usize))
                        .collect();
                    decoder.push_ranges(ranges, bytes).unwrap();
                }
                DecodeResult::Finished => break,
            }
        }
        let expected = (32..48)
            .chain(96..192)
            .filter(|i| i % 3 != 2)
            .collect::<Vec<_>>();
        assert_eq!(ids, expected);
        assert_eq!(metrics.lock().unwrap().rows_verified, 112);
    }

    #[test]
    fn dictionary_filter_prunes_empty_verdict_without_filtering_nulls_by_accident() {
        let schema = Arc::new(Schema::new(vec![Field::new("word", DataType::Utf8, true)]));
        let values = (0..128)
            .map(|i| (i % 2 == 0).then_some("no"))
            .collect::<StringArray>();
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(values)]).unwrap();
        let mut writer = ArrowWriter::try_new(Vec::new(), schema, None).unwrap();
        writer.write(&batch).unwrap();
        let file = Bytes::from(writer.into_inner().unwrap());
        for (accept_null, expected) in [(false, 0), (true, 64)] {
            let builder = ParquetRecordBatchReaderBuilder::try_new(file.clone()).unwrap();
            let metrics = Arc::new(Mutex::new(DictionaryFilterMetrics::default()));
            let predicate = StaticFilter {
                projection: ProjectionMask::leaves(builder.parquet_schema(), [0]),
                mode: DictionaryFilterMode::Verify,
                accept_null,
                accept_yes: false,
                fail_dictionary: false,
                metrics: Arc::clone(&metrics),
            };
            let rows = builder
                .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
                .build()
                .unwrap()
                .map(|batch| batch.unwrap().num_rows())
                .sum::<usize>();
            assert_eq!(rows, expected);
            let metrics = metrics.lock().unwrap();
            assert_eq!(
                metrics.row_groups_pruned_dictionary,
                usize::from(!accept_null)
            );
            assert_eq!(metrics.rows_verified, 128);
        }
    }

    #[test]
    fn dictionary_filter_matches_values_for_nulls_mixed_pages_and_errors() {
        for version in [WriterVersion::PARQUET_1_0, WriterVersion::PARQUET_2_0] {
            for mixed in [false, true] {
                let schema = Arc::new(Schema::new(vec![
                    Field::new("word", DataType::Utf8, true),
                    Field::new("id", DataType::Int32, false),
                ]));
                let values = (0..1024)
                    .map(|i| match i % 8 {
                        0 => None,
                        1 if mixed && i > 512 => Some(format!("other-{i}")),
                        2 => Some("yes".into()),
                        _ => Some("no".into()),
                    })
                    .collect::<StringArray>();
                let batch = RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(values),
                        Arc::new(Int32Array::from_iter_values(0..1024)),
                    ],
                )
                .unwrap();
                let properties = WriterProperties::builder()
                    .set_writer_version(version)
                    .set_data_page_row_count_limit(64)
                    .set_write_batch_size(64)
                    .set_dictionary_page_size_limit(if mixed { 100 } else { 1024 })
                    .build();
                let mut writer =
                    ArrowWriter::try_new(Vec::new(), schema, Some(properties)).unwrap();
                writer.write(&batch).unwrap();
                let file = Bytes::from(writer.into_inner().unwrap());
                for accept_null in [false, true] {
                    for mode in [
                        DictionaryFilterMode::Off,
                        DictionaryFilterMode::On,
                        DictionaryFilterMode::Verify,
                    ] {
                        for fail_dictionary in [false, true] {
                            let builder =
                                ParquetRecordBatchReaderBuilder::try_new(file.clone()).unwrap();
                            let projected = ProjectionMask::leaves(builder.parquet_schema(), [1]);
                            let projection = ProjectionMask::leaves(builder.parquet_schema(), [0]);
                            let metrics = Arc::new(Mutex::new(DictionaryFilterMetrics::default()));
                            let predicate = StaticFilter {
                                projection,
                                mode,
                                accept_null,
                                accept_yes: true,
                                fail_dictionary,
                                metrics: Arc::clone(&metrics),
                            };
                            let ids = builder
                                .with_projection(projected)
                                .with_batch_size(73)
                                .with_row_selection(RowSelection::from(vec![
                                    crate::arrow::arrow_reader::RowSelector::skip(33),
                                    crate::arrow::arrow_reader::RowSelector::select(823),
                                    crate::arrow::arrow_reader::RowSelector::skip(168),
                                ]))
                                .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
                                .build()
                                .unwrap()
                                .flat_map(|batch| {
                                    let batch = batch.unwrap();
                                    batch
                                        .column(0)
                                        .as_any()
                                        .downcast_ref::<Int32Array>()
                                        .unwrap()
                                        .values()
                                        .to_vec()
                                })
                                .collect::<Vec<_>>();
                            let expected = (33..856)
                                .filter(|i| i % 8 == 2 || (accept_null && i % 8 == 0))
                                .collect::<Vec<_>>();
                            assert_eq!(
                                ids, expected,
                                "{version:?}, mixed={mixed}, {mode:?}, error={fail_dictionary}"
                            );
                            let metrics = metrics.lock().unwrap();
                            if mode != DictionaryFilterMode::Off && !fail_dictionary {
                                assert_eq!(metrics.chunks_verdict, 1);
                                assert!(metrics.runs_evaluated > 0);
                                if mixed {
                                    assert_eq!(metrics.fallback_plain, 1);
                                }
                            }
                            if mode != DictionaryFilterMode::Off && fail_dictionary {
                                assert_eq!(metrics.fallback_error, 1);
                                assert_eq!(metrics.rows_value, 823);
                            }
                        }
                    }
                }
            }
        }
    }
}
