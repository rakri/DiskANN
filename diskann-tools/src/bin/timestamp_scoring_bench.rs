/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Compare vector-only, post-reranked, and metadata-aware search on held-out queries.
//!
//! # Reproduce
//!
//! ```text
//! cargo run --release -p diskann-tools --bin timestamp_scoring_bench -- \
//!   --data test_data/sift1m/sift_base.fbin --points 100000 --queries 100 \
//!   --l 20,50,100,200 --weights 0,0.25,1 --repeats 3 \
//!   --output target/timestamp-scoring-100k.json
//! ```
//!
//! Input is fbin (u32 rows, u32 dimensions, row-major f32 data, all little-endian).
//! The tool L2-normalizes vectors and uses the following rows as held-out queries.
//! Synthetic timestamps are independent of vectors and uniform over the previous year.
//! Exact GT minimizes squared L2 + weight * age / 365. Relevance is measured separately
//! against each query's vector-only top `relevance_pool`, including ties at the boundary.
//!
//! # Reference Run
//!
//! On 100k SIFT vectors, 100 held-out queries, k=10, seed=42, three repeats, and L=20:
//!
//! | Weight | Method | Combined recall | Relevant fraction | Mean age (days) | Median us |
//! |--------|--------|-----------------|-------------------|-----------------|-----------|
//! | 0.25 | Post-rerank | 0.514 | 1.000 | 97.1 | 47.8 |
//! | 0.25 | Metadata-aware | 0.790 | 1.000 | 41.2 | 112.9 |
//!
//! Both methods stayed within the exact vector top 1,000. The metadata-aware results
//! had mean squared L2 0.2454 versus 0.2246 for post-rerank. Increasing ordinary search
//! to L=100 plus post-reranking achieved recall 0.971, mean age 40.3 days, and median
//! 184.9 us; overfetching is a strong alternative and should not be omitted.
//! These are synthetic recency preferences and local timings, not production relevance
//! judgments or a claim of universal improvement. The JSON report retains all runs.

use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    num::NonZeroUsize,
    path::PathBuf,
    sync::Arc,
    time::Instant,
};

use anyhow::{ensure, Context, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use clap::Parser;
use diskann::{
    graph::{self, ext::scored::Scored, search::Knn},
    neighbor::{self, BackInserter, Neighbor},
    provider::DefaultContext,
};
use diskann_providers::{
    index::diskann_async::{new_index, MemoryIndex},
    model::graph::provider::async_::{
        common::{FullPrecision, NoDeletes},
        inmem::{DefaultProviderParameters, SetStartPoints},
    },
};
use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use diskann_vector::{
    distance::{Metric, SquaredL2},
    PureDistanceFunction,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::Serialize;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    data: PathBuf,
    #[arg(long, default_value_t = 100_000)]
    points: usize,
    #[arg(long, default_value_t = 100)]
    queries: usize,
    #[arg(long, default_value_t = 10)]
    k: usize,
    #[arg(long, value_delimiter = ',', default_value = "20,50,100,200")]
    l: Vec<usize>,
    #[arg(long, value_delimiter = ',', default_value = "0,0.25,1")]
    weights: Vec<f32>,
    #[arg(long, default_value = "8")]
    threads: NonZeroUsize,
    #[arg(long, default_value = "3")]
    repeats: NonZeroUsize,
    #[arg(long, default_value_t = 1000)]
    relevance_pool: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Serialize)]
struct Report {
    dataset: PathBuf,
    points: usize,
    dimensions: usize,
    queries: usize,
    k: usize,
    seed: u64,
    repeats: usize,
    relevance_pool: usize,
    normalized_vectors: bool,
    timestamp_model: &'static str,
    build_seconds: f64,
    ground_truth_seconds: f64,
    measurements: Vec<Measurement>,
}

#[derive(Serialize)]
struct Measurement {
    method: &'static str,
    weight: f32,
    l: usize,
    combined_recall: f64,
    relevant_fraction: f64,
    mean_vector_distance: f64,
    mean_age_days: f64,
    p50_query_us: f64,
    p95_query_us: f64,
    mean_comparisons: f64,
}

struct GroundTruth {
    vector_distances: Vec<f32>,
    relevance_cutoff: f32,
    combined: Vec<Vec<u32>>,
}

#[derive(Default)]
struct Totals {
    hits: usize,
    relevant: usize,
    count: usize,
    vector_distance: f64,
    age_days: f64,
    comparisons: u64,
    durations_us: Vec<f64>,
}

fn read_matrix(
    reader: &mut impl ReadBytesExt,
    rows: usize,
    dimensions: usize,
) -> Result<rowmajor::Owned<f32>> {
    let mut data = rowmajor::Owned::try_from_element(rows, dimensions, 0.0_f32)?;
    reader.read_f32_into::<LittleEndian>(data.as_mut_slice())?;
    for row_index in 0..rows {
        let row = data.row_mut(row_index);
        let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
        ensure!(
            norm.is_finite() && norm > 0.0,
            "invalid vector in row {row_index}"
        );
        for value in row {
            *value /= norm;
        }
    }
    Ok(data)
}

fn nearest(mut candidates: Vec<Neighbor<u32>>, k: usize) -> Vec<Neighbor<u32>> {
    candidates.select_nth_unstable_by(k - 1, neighbor::ord::fast_distance);
    candidates.truncate(k);
    candidates.sort_unstable_by(neighbor::ord::fast_distance);
    candidates
}

fn ground_truth(
    base: rowmajor::Ref<'_, f32>,
    queries: rowmajor::Ref<'_, f32>,
    timestamps: &[f32],
    args: &Args,
) -> Vec<GroundTruth> {
    (0..queries.nrows())
        .map(|query_index| {
            let distances: Vec<_> = (0..base.nrows())
                .map(|id| SquaredL2::evaluate(queries.row(query_index), base.row(id)))
                .collect();
            let relevance_cutoff = *nearest(
                distances
                    .iter()
                    .enumerate()
                    .map(|(id, &distance)| Neighbor::new(id as u32, distance))
                    .collect(),
                args.relevance_pool.min(base.nrows()),
            )
            .last()
            .unwrap()
            .distance();
            let combined = args
                .weights
                .iter()
                .map(|&weight| {
                    nearest(
                        distances
                            .iter()
                            .enumerate()
                            .map(|(id, &distance)| {
                                Neighbor::new(
                                    id as u32,
                                    distance + weight * (365.0 - timestamps[id]) / 365.0,
                                )
                            })
                            .collect(),
                        args.k,
                    )
                    .into_iter()
                    .map(|neighbor| *neighbor.id())
                    .collect()
                })
                .collect();
            GroundTruth {
                vector_distances: distances,
                relevance_cutoff,
                combined,
            }
        })
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.points > 0 && args.points < u32::MAX as usize,
        "invalid point count"
    );
    ensure!(args.queries > 0, "queries must be positive");
    ensure!(args.k > 0 && args.k <= args.points, "invalid k");
    ensure!(args.relevance_pool > 0, "relevance pool must be positive");
    ensure!(
        args.l.iter().all(|&value| value >= args.k),
        "L must be at least k"
    );
    ensure!(
        args.weights
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0),
        "invalid weight"
    );

    let mut reader = BufReader::new(File::open(&args.data).context("open fbin data")?);
    let rows = reader.read_u32::<LittleEndian>()? as usize;
    let dimensions = reader.read_u32::<LittleEndian>()? as usize;
    ensure!(dimensions > 0, "zero dimensions");
    let requested = args
        .points
        .checked_add(args.queries)
        .context("row count overflow")?;
    ensure!(requested <= rows, "need {requested} rows, file has {rows}");
    let base = read_matrix(&mut reader, args.points, dimensions)?;
    let queries = read_matrix(&mut reader, args.queries, dimensions)?;
    let mut rng = StdRng::seed_from_u64(args.seed);
    let timestamps: Vec<f32> = (0..args.points)
        .map(|_| rng.random_range(0.0..365.0))
        .collect();
    let runtime = diskann_benchmark_core::tokio::runtime(args.threads.get())?;
    let config = graph::config::Builder::new(
        32,
        graph::config::MaxDegree::default_slack(),
        100,
        Metric::L2.into(),
    )
    .build()?;
    let parameters = DefaultProviderParameters {
        max_points: args.points,
        frozen_points: diskann::utils::ONE,
        metric: Metric::L2,
        dim: dimensions,
        prefetch_lookahead: None,
        prefetch_cache_line_level: None,
        max_degree: config.max_degree_u32().get(),
    };
    let index: MemoryIndex<f32> = new_index(config, parameters, NoDeletes)?;
    let mut centroid = vec![0.0; dimensions];
    for row in 0..base.nrows() {
        for (sum, value) in centroid.iter_mut().zip(base.row(row)) {
            *sum += value / args.points as f32;
        }
    }
    index
        .provider()
        .set_start_points(std::iter::once(centroid.as_slice()))?;
    let context = DefaultContext;
    let build_start = Instant::now();
    for start in (0..args.points).step_by(1000) {
        let end = start.saturating_add(1000).min(args.points);
        let mut batch = rowmajor::Owned::try_from_element(end - start, dimensions, 0.0_f32)?;
        batch
            .as_mut_slice()
            .copy_from_slice(&base.as_slice()[start * dimensions..end * dimensions]);
        let ids: Arc<[u32]> = (start..end).map(|id| id as u32).collect::<Vec<_>>().into();
        runtime.block_on(index.multi_insert::<_, rowmajor::Owned<f32>>(
            FullPrecision,
            &context,
            Arc::new(batch),
            ids,
        ))?;
        if end % 10_000 == 0 || end == args.points {
            eprintln!(
                "Built {end}/{} vectors in {:.1}s",
                args.points,
                build_start.elapsed().as_secs_f64()
            );
        }
    }
    let build_seconds = build_start.elapsed().as_secs_f64();
    let gt_start = Instant::now();
    let truth = ground_truth(base.as_view(), queries.as_view(), &timestamps, &args);
    let ground_truth_seconds = gt_start.elapsed().as_secs_f64();
    eprintln!("Exact ground truth: {ground_truth_seconds:.2}s");
    let mut measurements = Vec::new();
    println!("weight L method recall relevant mean_L2 age_days p50_us p95_us cmps");

    for (weight_index, &weight) in args.weights.iter().enumerate() {
        let score = |id: u32, distance: f32| {
            distance
                + timestamps
                    .get(id as usize)
                    .map_or(0.0, |timestamp| weight * (365.0 - timestamp) / 365.0)
        };
        let strategy = Scored::new(FullPrecision, score, |id: &u32, distance| {
            score(*id, distance)
        });
        for &search_l in &args.l {
            let mut totals: [Totals; 3] = std::array::from_fn(|_| Totals::default());
            let query_count = args
                .queries
                .checked_mul(args.repeats.get())
                .context("query count overflow")?;
            let run_count = query_count
                .checked_add(args.queries.min(10))
                .context("warmup count overflow")?;
            for run in 0..run_count {
                let warmup = run < args.queries.min(10);
                let query_index = run % args.queries;
                let query = queries.row(query_index);
                for offset in 0..3 {
                    let method = (run + offset) % 3;
                    let mut output = Vec::new();
                    let mut fixed = vec![Neighbor::default(); args.k];
                    let params = Knn::new_default(search_l)?;
                    let started = Instant::now();
                    let stats = if method == 1 {
                        let stats = runtime.block_on(index.search(
                            params,
                            &FullPrecision,
                            &context,
                            query,
                            &mut output,
                        ))?;
                        output = output
                            .into_iter()
                            .map(|neighbor| {
                                let (id, distance) = neighbor.as_tuple();
                                Neighbor::new(id, score(id, distance))
                            })
                            .collect();
                        output.sort_unstable_by(neighbor::ord::fast_distance);
                        output.truncate(args.k);
                        stats
                    } else {
                        let mut buffer = BackInserter::new(&mut fixed);
                        let stats = if method == 0 {
                            runtime.block_on(index.search(
                                params,
                                &FullPrecision,
                                &context,
                                query,
                                &mut buffer,
                            ))?
                        } else {
                            runtime.block_on(index.search(
                                params,
                                &strategy,
                                &context,
                                query,
                                &mut buffer,
                            ))?
                        };
                        fixed.truncate(stats.result_count as usize);
                        output = fixed;
                        stats
                    };
                    let micros = started.elapsed().as_secs_f64() * 1_000_000.0;
                    ensure!(
                        output.len() == args.k,
                        "search returned fewer than k results"
                    );
                    if warmup {
                        continue;
                    }
                    let total = &mut totals[method];
                    total.durations_us.push(micros);
                    total.comparisons += u64::from(stats.cmps);
                    for neighbor in output {
                        let id = *neighbor.id() as usize;
                        total.count += 1;
                        total.hits += usize::from(
                            truth[query_index].combined[weight_index].contains(&(id as u32)),
                        );
                        total.relevant += usize::from(
                            truth[query_index].vector_distances[id]
                                <= truth[query_index].relevance_cutoff,
                        );
                        total.vector_distance += f64::from(truth[query_index].vector_distances[id]);
                        total.age_days += f64::from(365.0 - timestamps[id]);
                    }
                }
            }
            for (method, mut total) in ["vanilla", "post_rerank", "metadata_aware"]
                .into_iter()
                .zip(totals)
            {
                total.durations_us.sort_unstable_by(f64::total_cmp);
                let count = total.count as f64;
                let measurement = Measurement {
                    method,
                    weight,
                    l: search_l,
                    combined_recall: total.hits as f64 / count,
                    relevant_fraction: total.relevant as f64 / count,
                    mean_vector_distance: total.vector_distance / count,
                    mean_age_days: total.age_days / count,
                    p50_query_us: total.durations_us[total.durations_us.len() / 2],
                    p95_query_us: total.durations_us[(total.durations_us.len() - 1) * 95 / 100],
                    mean_comparisons: total.comparisons as f64 / total.durations_us.len() as f64,
                };
                println!(
                    "{weight:.2} {search_l} {method} {:.3} {:.3} {:.4} {:.1} {:.1} {:.1} {:.0}",
                    measurement.combined_recall,
                    measurement.relevant_fraction,
                    measurement.mean_vector_distance,
                    measurement.mean_age_days,
                    measurement.p50_query_us,
                    measurement.p95_query_us,
                    measurement.mean_comparisons
                );
                measurements.push(measurement);
            }
        }
    }
    if let Some(path) = &args.output {
        let report = Report {
            dataset: args.data.clone(),
            points: args.points,
            dimensions,
            queries: args.queries,
            k: args.k,
            seed: args.seed,
            repeats: args.repeats.get(),
            relevance_pool: args.relevance_pool.min(args.points),
            normalized_vectors: true,
            timestamp_model:
                "independent uniform timestamps over 365 days; query timestamp is day 365",
            build_seconds,
            ground_truth_seconds,
            measurements,
        };
        let mut writer = BufWriter::new(File::create(path)?);
        serde_json::to_writer_pretty(&mut writer, &report)?;
        writer.flush()?;
        eprintln!("Report: {}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_selects_and_sorts_exact_top_k() {
        let candidates = vec![
            Neighbor::new(0, 4.0),
            Neighbor::new(1, 1.0),
            Neighbor::new(2, 3.0),
            Neighbor::new(3, 2.0),
        ];
        let selected = nearest(candidates, 2);
        assert_eq!(selected[0].as_tuple(), (1, 1.0));
        assert_eq!(selected[1].as_tuple(), (3, 2.0));
    }

    #[test]
    fn matrix_loading_normalizes_and_rejects_invalid_data() {
        let bytes: Vec<_> = [3.0_f32, 4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let data = read_matrix(&mut bytes.as_slice(), 1, 2).unwrap();
        assert_eq!(data.as_slice(), &[0.6, 0.8]);
        assert!(read_matrix(&mut [0u8; 4].as_slice(), 1, 2).is_err());
        assert!(read_matrix(&mut [0u8; 8].as_slice(), 1, 2).is_err());
    }

    #[test]
    fn ground_truth_separates_vector_relevance_and_recency() {
        let base = rowmajor::Owned::from_fn(3, 2, |position| {
            [1.0_f32, 0.0, 0.96, 0.28, 0.8, 0.6][position.row * 2 + position.col]
        });
        let queries = rowmajor::Owned::row_vector(vec![1.0, 0.0].into_boxed_slice());
        let args = Args::try_parse_from([
            "bench",
            "--data",
            "unused.fbin",
            "--points",
            "3",
            "--queries",
            "1",
            "--k",
            "1",
            "--weights",
            "0,1",
            "--relevance-pool",
            "1",
        ])
        .unwrap();
        let truth = ground_truth(
            base.as_view(),
            queries.as_view(),
            &[0.0, 180.0, 365.0],
            &args,
        );
        assert_eq!(truth[0].combined, [vec![0], vec![2]]);
        assert_eq!(truth[0].relevance_cutoff, 0.0);
        assert!(truth[0].vector_distances[2] > truth[0].relevance_cutoff);
    }
}
