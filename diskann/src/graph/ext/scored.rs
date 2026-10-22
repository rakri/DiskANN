/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Query-local scoring layered over existing search strategies.
//!
//! [`Scored`] changes navigation and final ranking without owning metadata or changing
//! graph construction. Its callbacks can borrow query metadata and a metadata store.
//! Call the original strategy when query metadata is absent to retain the original
//! search path without scoring overhead.
//!
//! Filtering is independent: wrap a filtered strategy to preserve its classification
//! and accept-only expansion, including skipping distances for rejected second-hop
//! neighbors. Scoring alone does not reject candidates.
//!
//! # Example
//!
//! This example assumes internal and output IDs use the same timestamp lookup. For
//! providers that translate IDs, use the appropriate lookup in each callback.
//!
//! ```
//! # use diskann::{ANNResult, graph::{DiskANNIndex, ext::scored::Scored, glue, search::Knn},
//! #     neighbor::Neighbor, provider::DataProvider};
//! # async fn search<DP, S>(index: &DiskANNIndex<DP>, base: S, context: &DP::Context,
//! #     query: &[f32], timestamps: &[f32], query_timestamp: Option<f32>)
//! #     -> ANNResult<Vec<Neighbor<u32>>>
//! # where
//! #     DP: DataProvider<InternalId = u32, ExternalId = u32>,
//! #     S: for<'a> glue::DefaultSearchStrategy<'a, DP, &'a [f32], u32,
//! #         SearchAccessor: glue::SearchAccessor>,
//! # {
//! let knn = Knn::new_default(64)?;
//! let mut output = Vec::new();
//! if let Some(timestamp) = query_timestamp {
//!     let score = |id: u32, distance: f32| {
//!         let penalty = timestamps.get(id as usize)
//!             .map_or(0.0, |stored| (stored - timestamp).abs());
//!         distance + penalty
//!     };
//!     let strategy = Scored::new(base, score, |id: &u32, distance| score(*id, distance));
//!     index.search(knn, &strategy, context, query, &mut output).await?;
//! } else {
//!     index.search(knn, &base, context, query, &mut output).await?;
//! }
//! # Ok(output)
//! # }
//! ```

use std::{fmt::Debug, hash::Hash};

use hashbrown::HashMap;

use crate::{
    ANNError, ANNResult,
    graph::{SearchOutputBuffer, glue},
    neighbor::{self, Neighbor},
    provider::{DataProvider, HasId},
};

/// Adjust navigation distances and final result distances for a single query.
///
/// `navigation` receives internal IDs and the inner accessor's distances. `results`
/// receives output IDs and distances produced by the inner post-processor, including
/// full-precision reranking when supported. Keeping these callbacks separate permits
/// internal/external ID translation and different navigation and reranking precision.
/// Both callbacks must return finite, lower-is-better scores and use consistent metadata
/// throughout the search. Missing metadata and synthetic start points are policy choices
/// for the callbacks.
///
/// Original navigation distances are retained per visited ID so that the inner
/// post-processor never sees an already adjusted distance. Final scoring happens before
/// output truncation. This requires query-local storage proportional to visited IDs and
/// buffering the post-processed candidate set.
///
/// The inner post-processor must support unbounded output. Any candidate selection it
/// performs, such as diversity selection, still precedes final scoring; this adapter
/// does not redefine that selection policy.
///
/// A vector-similarity graph remains approximate under a different scoring policy;
/// increasing the influence of metadata may require more exploration for good recall.
#[derive(Debug, Clone, Copy)]
pub struct Scored<S, N, R> {
    strategy: S,
    navigation: N,
    results: R,
}

impl<S, N, R> Scored<S, N, R> {
    /// Construct a query-scoped strategy with navigation and final-result scoring.
    pub fn new(strategy: S, navigation: N, results: R) -> Self {
        Self {
            strategy,
            navigation,
            results,
        }
    }
}

impl<'a, DP, S, T, N, R> glue::SearchStrategy<'a, DP, T> for Scored<S, N, R>
where
    DP: DataProvider,
    S: glue::SearchStrategy<'a, DP, T>,
    N: Fn(DP::InternalId, f32) -> f32 + Send + Sync + 'a,
    R: Send + Sync,
{
    type SearchAccessor = Accessor<'a, S::SearchAccessor, N>;
    type SearchAccessorError = S::SearchAccessorError;

    fn search_accessor(
        &'a self,
        provider: &'a DP,
        context: &'a DP::Context,
        query: T,
    ) -> Result<Self::SearchAccessor, Self::SearchAccessorError> {
        Ok(Accessor {
            inner: self.strategy.search_accessor(provider, context, query)?,
            navigation: &self.navigation,
            original: HashMap::new(),
        })
    }
}

impl<'a, DP, S, T, O, N, R> glue::DefaultPostProcessor<'a, DP, T, O> for Scored<S, N, R>
where
    DP: DataProvider,
    S: glue::DefaultPostProcessor<'a, DP, T, O>,
    N: Fn(DP::InternalId, f32) -> f32 + Send + Sync + 'a,
    R: Fn(&O, f32) -> f32 + Send + Sync + 'a,
    T: Send,
    O: Send,
{
    type Processor = glue::Pipeline<Rescore<'a, R>, S::Processor>;

    fn default_post_processor(&'a self) -> Self::Processor {
        glue::Pipeline::new(
            Rescore::new(&self.results),
            self.strategy.default_post_processor(),
        )
    }
}

/// An accessor that adjusts emitted distances while retaining the original distances.
pub struct Accessor<'a, A, N>
where
    A: HasId,
{
    inner: A,
    navigation: &'a N,
    original: HashMap<A::Id, f32>,
}

impl<A, N> Accessor<'_, A, N>
where
    A: HasId,
{
    /// Access the backend for custom post-processing or instrumentation.
    pub fn inner(&mut self) -> &mut A {
        &mut self.inner
    }
}

impl<A: HasId, N> HasId for Accessor<'_, A, N> {
    type Id = A::Id;
}

fn score<I, N>(original: &mut HashMap<I, f32>, navigation: &N, id: I, distance: f32) -> f32
where
    I: Copy + Eq + Hash,
    N: Fn(I, f32) -> f32,
{
    original.insert(id, distance);
    navigation(id, distance)
}

impl<A, N> glue::SearchAccessor for Accessor<'_, A, N>
where
    A: glue::SearchAccessor<Id: Copy + Eq + Hash + Send + Sync>,
    N: Fn(A::Id, f32) -> f32 + Sync,
{
    async fn starting_points(&self) -> ANNResult<Vec<Self::Id>> {
        self.inner.starting_points().await
    }

    async fn start_point_distances<F>(&mut self, mut f: F) -> ANNResult<()>
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        let original = &mut self.original;
        let navigation = self.navigation;
        self.inner
            .start_point_distances(|id, distance| f(id, score(original, navigation, id, distance)))
            .await
    }

    async fn expand_beam<Itr, P, F>(
        &mut self,
        ids: Itr,
        pred: P,
        mut on_neighbors: F,
    ) -> ANNResult<()>
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: glue::HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(Self::Id, f32) + Send,
    {
        let original = &mut self.original;
        let navigation = self.navigation;
        self.inner
            .expand_beam(ids, pred, |id, distance| {
                on_neighbors(id, score(original, navigation, id, distance))
            })
            .await
    }

    fn terminate_early(&mut self) -> bool {
        self.inner.terminate_early()
    }

    async fn is_not_start_point(
        &self,
    ) -> ANNResult<impl Fn(Self::Id) -> bool + Send + Sync + 'static> {
        self.inner.is_not_start_point().await
    }

    async fn num_starting_points(&self) -> ANNResult<usize> {
        self.inner.num_starting_points().await
    }
}

impl<A, N> glue::FilteredAccessor for Accessor<'_, A, N>
where
    A: glue::FilteredAccessor<Id: Copy + Eq + Hash + Send + Sync>,
    N: Fn(A::Id, f32) -> f32 + Sync,
{
    async fn start_point_distances<F>(&mut self, mut f: F) -> ANNResult<()>
    where
        F: FnMut(glue::Decision<Self::Id>, f32) + Send,
    {
        let original = &mut self.original;
        let navigation = self.navigation;
        self.inner
            .start_point_distances(|decision, distance| {
                f(
                    decision,
                    score(original, navigation, decision.into_inner(), distance),
                )
            })
            .await
    }

    async fn expand_beam_filtered<Itr, P, F>(
        &mut self,
        ids: Itr,
        pred: P,
        mut on_neighbors: F,
    ) -> ANNResult<()>
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: glue::HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(glue::Decision<Self::Id>, f32) + Send,
    {
        let original = &mut self.original;
        let navigation = self.navigation;
        self.inner
            .expand_beam_filtered(ids, pred, |decision, distance| {
                on_neighbors(
                    decision,
                    score(original, navigation, decision.into_inner(), distance),
                )
            })
            .await
    }

    async fn expand_beam_accept_only<Itr, P, F>(
        &mut self,
        ids: Itr,
        pred: P,
        mut on_neighbors: F,
    ) -> ANNResult<()>
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: glue::Predicate<Self::Id> + glue::PredicateMut<glue::Accept<Self::Id>> + Send + Sync,
        F: FnMut(glue::Accept<Self::Id>, f32) + Send,
    {
        let original = &mut self.original;
        let navigation = self.navigation;
        self.inner
            .expand_beam_accept_only(ids, pred, |accepted, distance| {
                on_neighbors(
                    accepted,
                    score(original, navigation, accepted.into_inner(), distance),
                )
            })
            .await
    }

    fn terminate_early(&mut self) -> bool {
        self.inner.terminate_early()
    }

    async fn num_starting_points(&self) -> ANNResult<usize> {
        self.inner.num_starting_points().await
    }
}

/// Restore base distances for the backend, then score and sort its output before truncation.
///
/// Compose this step with a custom backend post-processor using [`glue::Pipeline`].
pub struct Rescore<'a, R> {
    results: &'a R,
}

impl<'a, R> Rescore<'a, R> {
    /// Apply `results` to backend output distances before sorting and truncation.
    pub fn new(results: &'a R) -> Self {
        Self { results }
    }
}

impl<A, N, R, T, O> glue::SearchPostProcessStep<Accessor<'_, A, N>, T, O> for Rescore<'_, R>
where
    A: HasId<Id: Copy + Eq + Hash + Debug + Send> + Send,
    N: Sync,
    R: Fn(&O, f32) -> f32 + Sync,
    T: Send,
    O: Send,
{
    type Error<NextError>
        = ANNError
    where
        NextError: crate::error::StandardError;

    type NextAccessor = A;

    async fn post_process_step<I, B, Next>(
        &self,
        next: &Next,
        accessor: &mut Accessor<'_, A, N>,
        query: T,
        candidates: I,
        output: &mut B,
    ) -> ANNResult<usize>
    where
        I: Iterator<Item = Neighbor<A::Id>> + Send,
        B: SearchOutputBuffer<O> + Send + ?Sized,
        Next: glue::SearchPostProcess<A, T, O> + Sync,
    {
        let mut candidates = candidates
            .map(|candidate| {
                let id = *candidate.id();
                let distance = accessor.original.get(&id).copied().ok_or_else(|| {
                    ANNError::message(format!(
                        "missing original distance for scored candidate {id:?}"
                    ))
                })?;
                Ok(Neighbor::new(id, distance))
            })
            .collect::<ANNResult<Vec<_>>>()?;
        candidates.sort_unstable_by(neighbor::ord::fast_distance);

        let mut results = Vec::with_capacity(candidates.len());
        next.post_process(
            &mut accessor.inner,
            query,
            candidates.into_iter(),
            &mut results,
        )
        .await
        .map_err(|error| {
            let error: ANNError = error.into();
            error.context("post-processing before metadata scoring")
        })?;

        let mut results: Vec<_> = results
            .into_iter()
            .map(|neighbor| {
                let (id, distance) = neighbor.as_tuple();
                let distance = (self.results)(&id, distance);
                Neighbor::new(id, distance)
            })
            .collect();
        results.sort_unstable_by(neighbor::ord::fast_distance);
        Ok(output.extend(results))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{
        ext::labeled::{self, QueryLabelProvider},
        glue::{
            FilteredAccessor as _, SearchAccessor as _, SearchPostProcess as _, SearchStrategy as _,
        },
        search::{AdaptiveL, InlineFilterSearch, Knn, MultihopFilterSearch},
        search_output_buffer,
        test::{provider as test_provider, synthetic::Grid},
    };
    use diskann_vector::distance::Metric;

    #[derive(Debug)]
    struct Even;

    impl QueryLabelProvider<u32> for Even {
        fn is_match(&self, id: u32) -> bool {
            id.is_multiple_of(2)
        }
    }

    #[tokio::test]
    async fn scores_start_points_and_neighbors() {
        let provider = test_provider::Provider::grid(Grid::Two, 4).unwrap();
        let context = test_provider::Context::new();
        let timestamp = 7.0;
        let strategy = Scored::new(
            test_provider::Strategy::new(),
            |_: u32, distance| distance + timestamp,
            |_: &u32, distance: f32| distance + timestamp,
        );
        let mut accessor = strategy
            .search_accessor(&provider, &context, &[0.0, 0.0])
            .unwrap();

        let mut starts = Vec::new();
        accessor
            .start_point_distances(|id, distance| starts.push((id, distance)))
            .await
            .unwrap();
        assert_eq!(starts.len(), 1);
        for (id, distance) in starts {
            assert_eq!(distance, accessor.original[&id] + timestamp);
        }

        let mut visited = hashbrown::HashSet::new();
        let mut neighbors = Vec::new();
        accessor
            .expand_beam(
                [0, 5].into_iter(),
                glue::NotInMut::new(&mut visited),
                |id, distance| neighbors.push((id, distance)),
            )
            .await
            .unwrap();
        assert!(!neighbors.is_empty());
        for (id, distance) in neighbors {
            assert_eq!(distance, accessor.original[&id] + timestamp);
        }
    }

    #[tokio::test]
    async fn accept_only_preserves_filter_before_distance() {
        let provider = test_provider::Provider::grid(Grid::Two, 4).unwrap();
        let context = test_provider::Context::new();
        let filter = Even;
        let strategy = Scored::new(
            labeled::Filtered::new(test_provider::Strategy::new(), &filter),
            |_: u32, distance| distance + 3.0,
            |_: &u32, distance: f32| distance + 3.0,
        );
        let mut accessor = strategy
            .search_accessor(&provider, &context, &[0.0, 0.0])
            .unwrap();
        let mut visited = hashbrown::HashSet::new();
        let mut results = Vec::new();
        accessor
            .expand_beam_accept_only(
                [0, 5, 10, 15].into_iter(),
                glue::NotInMut::new(&mut visited),
                |accepted, distance| results.push((accepted.into_inner(), distance)),
            )
            .await
            .unwrap();

        assert!(!results.is_empty());
        assert!(results.iter().all(|(id, _)| id.is_multiple_of(2)));
        assert_eq!(results.len(), accessor.original.len());
        let computed = results.len();
        for (id, distance) in results {
            assert!(visited.contains(&id));
            assert_eq!(distance, accessor.original[&id] + 3.0);
        }
        assert!(visited.iter().all(|id| id.is_multiple_of(2)));
        drop(accessor);
        assert_eq!(provider.get_vector.value(), computed);
    }

    struct Backend;

    impl HasId for Backend {
        type Id = u32;
    }

    struct Rerank;

    impl glue::SearchPostProcess<Backend, (), String> for Rerank {
        type Error = ANNError;

        async fn post_process<I, B>(
            &self,
            _accessor: &mut Backend,
            _query: (),
            candidates: I,
            output: &mut B,
        ) -> ANNResult<usize>
        where
            I: Iterator<Item = Neighbor<u32>> + Send,
            B: SearchOutputBuffer<String> + Send + ?Sized,
        {
            let candidates: Vec<_> = candidates.collect();
            assert_eq!(candidates.len(), 2);
            assert_eq!(candidates[0].as_tuple(), (1, 1.0));
            assert_eq!(candidates[1].as_tuple(), (2, 4.0));
            assert_eq!(output.size_hint(), None);
            Ok(output.extend(candidates.into_iter().map(|candidate| {
                Neighbor::new(
                    format!("vector-{}", candidate.id()),
                    2.0 * candidate.distance(),
                )
            })))
        }
    }

    #[tokio::test]
    async fn restores_base_distances_and_scores_before_truncation() {
        let navigation = |_: u32, distance: f32| distance + 1000.0;
        let mut accessor = Accessor {
            inner: Backend,
            navigation: &navigation,
            original: HashMap::from_iter([(1, 1.0), (2, 4.0)]),
        };
        let results = |id: &String, distance| {
            if id == "vector-2" {
                distance
            } else {
                distance + 100.0
            }
        };
        let processor = glue::Pipeline::new(Rescore::new(&results), Rerank);
        let mut ids = [String::new()];
        let mut distances = [0.0];
        let mut output = search_output_buffer::IdDistance::new(&mut ids, &mut distances);
        let count = processor
            .post_process(
                &mut accessor,
                (),
                [Neighbor::new(2, 1004.0), Neighbor::new(1, 1001.0)].into_iter(),
                &mut output,
            )
            .await
            .unwrap();

        assert_eq!(count, 1);
        assert_eq!(ids, ["vector-2"]);
        assert_eq!(distances, [8.0]);
    }

    #[tokio::test]
    async fn missing_original_distance_is_an_error() {
        let navigation = |_: u32, distance: f32| distance;
        let mut accessor = Accessor {
            inner: Backend,
            navigation: &navigation,
            original: HashMap::new(),
        };
        let results = |_: &u32, distance: f32| distance;
        let processor = glue::Pipeline::new(Rescore::new(&results), glue::CopyIds);
        let mut output = Vec::new();
        let error = processor
            .post_process(
                &mut accessor,
                (),
                [Neighbor::new(42, 1.0)].into_iter(),
                &mut output,
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("42"));
        assert!(output.is_empty());
    }

    struct FailedPostProcess;

    impl glue::SearchPostProcess<Backend, ()> for FailedPostProcess {
        type Error = ANNError;

        async fn post_process<I, B>(
            &self,
            _accessor: &mut Backend,
            _query: (),
            _candidates: I,
            _output: &mut B,
        ) -> ANNResult<usize>
        where
            I: Iterator<Item = Neighbor<u32>> + Send,
            B: SearchOutputBuffer<u32> + Send + ?Sized,
        {
            Err(ANNError::message("metadata backend unavailable"))
        }
    }

    #[tokio::test]
    async fn preserves_backend_post_process_errors() {
        let navigation = |_: u32, distance: f32| distance;
        let mut accessor = Accessor {
            inner: Backend,
            navigation: &navigation,
            original: HashMap::from_iter([(1, 1.0)]),
        };
        let results = |_: &u32, distance: f32| distance;
        let processor = glue::Pipeline::new(Rescore::new(&results), FailedPostProcess);
        let mut output = Vec::new();
        let error = processor
            .post_process(
                &mut accessor,
                (),
                [Neighbor::new(1, 1.0)].into_iter(),
                &mut output,
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("metadata backend unavailable"));
        assert!(output.is_empty());
    }

    fn grid_index() -> crate::graph::DiskANNIndex<test_provider::Provider> {
        let config = crate::graph::config::Builder::new(
            4,
            crate::graph::config::MaxDegree::Same,
            32,
            Metric::L2.into(),
        )
        .build()
        .unwrap();
        crate::graph::DiskANNIndex::new(
            config,
            test_provider::Provider::grid(Grid::Two, 4).unwrap(),
            None,
        )
    }

    #[tokio::test]
    async fn optional_query_metadata_keeps_the_vanilla_path() {
        let index = grid_index();
        let context = test_provider::Context::new();
        let query = [0.0, 0.0];
        let mut timestamps = [100.0_f32; 16];
        timestamps[15] = 0.0;
        let base = test_provider::Strategy::new();
        let mut vanilla = Vec::new();
        let baseline = index
            .search(
                Knn::new_default(32).unwrap(),
                &base,
                &context,
                query.as_slice(),
                &mut vanilla,
            )
            .await
            .unwrap();

        for query_timestamp in [None, Some(0.0)] {
            let mut ids = [u32::MAX];
            let mut distances = [f32::NAN];
            let mut output = search_output_buffer::IdDistance::new(&mut ids, &mut distances);
            let knn = Knn::new_default(32).unwrap();
            let stats = if let Some(timestamp) = query_timestamp {
                let metadata_score = |id: u32, distance| {
                    distance
                        + timestamps
                            .get(id as usize)
                            .map_or(0.0, |stored| (stored - timestamp).abs())
                };
                let strategy = Scored::new(base.clone(), metadata_score, |id: &u32, distance| {
                    metadata_score(*id, distance)
                });
                index
                    .search(knn, &strategy, &context, query.as_slice(), &mut output)
                    .await
                    .unwrap()
            } else {
                index
                    .search(knn, &base, &context, query.as_slice(), &mut output)
                    .await
                    .unwrap()
            };

            assert_eq!(stats.result_count, 1);
            if query_timestamp.is_some() {
                assert_eq!(ids, [15]);
                assert_eq!(distances, [18.0]);
            } else {
                assert_eq!(ids, [*vanilla[0].id()]);
                assert_eq!(distances, [*vanilla[0].distance()]);
                assert_eq!(stats.cmps, baseline.cmps);
                assert_eq!(stats.hops, baseline.hops);
            }
        }
    }

    #[rstest::rstest]
    #[case::inline(false, false)]
    #[case::adaptive_inline(false, true)]
    #[case::multihop(true, false)]
    #[tokio::test]
    async fn filtered_search_uses_metadata_scores(#[case] multihop: bool, #[case] adaptive: bool) {
        let index = grid_index();
        let context = test_provider::Context::new();
        let filter = Even;
        let metadata_score = |id: u32, distance| {
            if id == 14 { distance } else { distance + 100.0 }
        };
        let strategy = Scored::new(
            labeled::Filtered::new(test_provider::Strategy::new(), &filter),
            metadata_score,
            |id: &u32, distance| metadata_score(*id, distance),
        );
        let mut ids = [u32::MAX];
        let mut distances = [f32::NAN];
        let mut output = search_output_buffer::IdDistance::new(&mut ids, &mut distances);
        let knn = Knn::new_default(32).unwrap();
        let stats = if multihop {
            index
                .search(
                    MultihopFilterSearch::new(knn),
                    &strategy,
                    &context,
                    [0.0, 0.0].as_slice(),
                    &mut output,
                )
                .await
                .unwrap()
        } else {
            let adaptive = adaptive.then(|| AdaptiveL::new(10, 4.0).unwrap());
            index
                .search(
                    InlineFilterSearch::new(knn, adaptive),
                    &strategy,
                    &context,
                    [0.0, 0.0].as_slice(),
                    &mut output,
                )
                .await
                .unwrap()
        };
        assert_eq!(stats.result_count, 1);
        assert_eq!(ids, [14]);
        assert_eq!(distances, [13.0]);
    }
}
