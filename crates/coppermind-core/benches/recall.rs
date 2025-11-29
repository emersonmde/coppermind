//! Recall benchmarks for measuring search quality.
//!
//! Run with: `cargo bench -p coppermind-core --bench recall`
//!
//! These benchmarks measure **search quality** rather than speed:
//!
//! - **Recall@k**: What fraction of true top-k nearest neighbors does HNSW return?
//! - Ground truth is computed via brute-force exact search
//! - Tests various k values and corpus sizes
//!
//! # Why Recall Matters
//!
//! HNSW is an *approximate* nearest neighbor algorithm. It trades some accuracy
//! for O(log n) search time. Understanding this tradeoff is critical:
//!
//! - Recall@10 = 0.95 means 95% of true top-10 neighbors are found
//! - Higher `ef_search` parameter improves recall but slows search
//! - Different use cases have different recall requirements
//!
//! # Benchmark Output
//!
//! These benchmarks report recall as throughput (recall * 100 = percentage).
//! A throughput of 95 means 95% recall.

use coppermind_core::config::EMBEDDING_DIM;
use coppermind_core::search::types::DocId;
use coppermind_core::search::vector::VectorSearchEngine;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::collections::HashSet;
use std::time::Duration;

// =============================================================================
// Configuration
// =============================================================================

/// Corpus sizes for recall testing.
const RECALL_CORPUS_SIZES: &[usize] = &[1_000, 5_000, 10_000];

/// K values to test recall at.
const RECALL_K_VALUES: &[usize] = &[1, 5, 10, 20, 50, 100];

/// Number of queries to average recall over.
const NUM_QUERIES: usize = 100;

/// Base seed for query embeddings.
const QUERY_SEED_BASE: u64 = 1_000_000;

// =============================================================================
// Test Data Generation
// =============================================================================

/// Generate a deterministic L2-normalized embedding.
fn seeded_embedding(seed: u64) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let raw: Vec<f32> = (0..EMBEDDING_DIM)
        .map(|i| {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            i.hash(&mut hasher);
            let h = hasher.finish();
            ((h as f32 / u64::MAX as f32) * 2.0) - 1.0
        })
        .collect();

    let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    raw.into_iter().map(|x| x / norm).collect()
}

/// Compute cosine similarity between two vectors.
///
/// For normalized vectors, this equals dot product.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// =============================================================================
// Ground Truth Computation
// =============================================================================

/// Compute exact k-nearest neighbors via brute-force search.
///
/// This is O(n) and used as ground truth for measuring HNSW recall.
fn brute_force_knn(query: &[f32], corpus: &[(DocId, Vec<f32>)], k: usize) -> Vec<DocId> {
    let mut scored: Vec<_> = corpus
        .iter()
        .map(|(doc_id, embedding)| {
            let sim = cosine_similarity(query, embedding);
            (*doc_id, sim)
        })
        .collect();

    // Sort by similarity descending
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

/// Calculate recall: fraction of ground truth results found by HNSW.
fn calculate_recall(hnsw_results: &[DocId], ground_truth: &[DocId]) -> f64 {
    if ground_truth.is_empty() {
        return 1.0;
    }

    let truth_set: HashSet<_> = ground_truth.iter().collect();
    let found = hnsw_results
        .iter()
        .filter(|id| truth_set.contains(id))
        .count();

    found as f64 / ground_truth.len() as f64
}

// =============================================================================
// Recall Benchmarks
// =============================================================================

/// Benchmark: Recall@k for various k values.
///
/// Reports recall as a percentage (throughput field).
/// Higher is better - 100 means perfect recall.
fn bench_recall_at_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall/at_k");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));

    let corpus_size = 5_000; // Fixed corpus size for k comparison

    // Pre-build corpus
    let corpus: Vec<_> = (0..corpus_size)
        .map(|i| (DocId::from_u64(i as u64), seeded_embedding(i as u64)))
        .collect();

    // Build HNSW index
    let mut engine = VectorSearchEngine::new(EMBEDDING_DIM);
    for (doc_id, embedding) in &corpus {
        let _ = engine.add_document(*doc_id, embedding.clone());
    }

    // Generate query embeddings
    let queries: Vec<_> = (0..NUM_QUERIES)
        .map(|i| seeded_embedding(QUERY_SEED_BASE + i as u64))
        .collect();

    for &k in RECALL_K_VALUES {
        if k > corpus_size {
            continue;
        }

        // Pre-compute ground truth for all queries
        let ground_truths: Vec<_> = queries
            .iter()
            .map(|q| brute_force_knn(q, &corpus, k))
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            b.iter(|| {
                let mut total_recall = 0.0;

                for (query, ground_truth) in queries.iter().zip(&ground_truths) {
                    let hnsw_results: Vec<_> = engine
                        .search(black_box(query), k)
                        .unwrap()
                        .into_iter()
                        .map(|(id, _)| id)
                        .collect();

                    total_recall += calculate_recall(&hnsw_results, ground_truth);
                }

                // Return average recall as percentage
                (total_recall / NUM_QUERIES as f64) * 100.0
            });
        });
    }
    group.finish();
}

/// Benchmark: Recall@10 vs corpus size.
///
/// Tests whether recall degrades as corpus grows (it shouldn't significantly
/// with proper HNSW parameters).
fn bench_recall_vs_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall/vs_corpus_size");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    let k = 10;
    let num_queries = 50; // Fewer queries for larger corpora

    for &corpus_size in RECALL_CORPUS_SIZES {
        // Build corpus
        let corpus: Vec<_> = (0..corpus_size)
            .map(|i| (DocId::from_u64(i as u64), seeded_embedding(i as u64)))
            .collect();

        // Build HNSW index
        let mut engine = VectorSearchEngine::new(EMBEDDING_DIM);
        for (doc_id, embedding) in &corpus {
            let _ = engine.add_document(*doc_id, embedding.clone());
        }

        // Generate queries and ground truth
        let queries: Vec<_> = (0..num_queries)
            .map(|i| seeded_embedding(QUERY_SEED_BASE + i as u64))
            .collect();

        let ground_truths: Vec<_> = queries
            .iter()
            .map(|q| brute_force_knn(q, &corpus, k))
            .collect();

        // Report recall as throughput percentage
        group.throughput(Throughput::Elements(100)); // Max possible recall
        group.bench_with_input(
            BenchmarkId::from_parameter(corpus_size),
            &corpus_size,
            |b, _| {
                b.iter(|| {
                    let mut total_recall = 0.0;

                    for (query, ground_truth) in queries.iter().zip(&ground_truths) {
                        let hnsw_results: Vec<_> = engine
                            .search(black_box(query), k)
                            .unwrap()
                            .into_iter()
                            .map(|(id, _)| id)
                            .collect();

                        total_recall += calculate_recall(&hnsw_results, ground_truth);
                    }

                    (total_recall / num_queries as f64) * 100.0
                });
            },
        );
    }
    group.finish();
}

/// Benchmark: Recall distribution analysis.
///
/// Measures recall variance across queries to identify edge cases
/// where HNSW might perform poorly.
fn bench_recall_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall/distribution");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));

    let corpus_size = 5_000;
    let k = 10;
    let num_queries = 200;

    // Build corpus and index
    let corpus: Vec<_> = (0..corpus_size)
        .map(|i| (DocId::from_u64(i as u64), seeded_embedding(i as u64)))
        .collect();

    let mut engine = VectorSearchEngine::new(EMBEDDING_DIM);
    for (doc_id, embedding) in &corpus {
        let _ = engine.add_document(*doc_id, embedding.clone());
    }

    // Generate queries
    let queries: Vec<_> = (0..num_queries)
        .map(|i| seeded_embedding(QUERY_SEED_BASE + i as u64))
        .collect();

    let ground_truths: Vec<_> = queries
        .iter()
        .map(|q| brute_force_knn(q, &corpus, k))
        .collect();

    // Report minimum recall (worst case) as throughput
    group.bench_function("min_recall", |b| {
        b.iter(|| {
            let recalls: Vec<_> = queries
                .iter()
                .zip(&ground_truths)
                .map(|(query, ground_truth)| {
                    let hnsw_results: Vec<_> = engine
                        .search(black_box(query), k)
                        .unwrap()
                        .into_iter()
                        .map(|(id, _)| id)
                        .collect();
                    calculate_recall(&hnsw_results, ground_truth)
                })
                .collect();

            recalls
                .iter()
                .cloned()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0)
                * 100.0
        });
    });

    // Report p50 recall
    group.bench_function("p50_recall", |b| {
        b.iter(|| {
            let mut recalls: Vec<_> = queries
                .iter()
                .zip(&ground_truths)
                .map(|(query, ground_truth)| {
                    let hnsw_results: Vec<_> = engine
                        .search(black_box(query), k)
                        .unwrap()
                        .into_iter()
                        .map(|(id, _)| id)
                        .collect();
                    calculate_recall(&hnsw_results, ground_truth)
                })
                .collect();

            recalls.sort_by(|a, b| a.partial_cmp(b).unwrap());
            recalls[recalls.len() / 2] * 100.0
        });
    });

    // Report p99 recall (near-worst case)
    group.bench_function("p99_recall", |b| {
        b.iter(|| {
            let mut recalls: Vec<_> = queries
                .iter()
                .zip(&ground_truths)
                .map(|(query, ground_truth)| {
                    let hnsw_results: Vec<_> = engine
                        .search(black_box(query), k)
                        .unwrap()
                        .into_iter()
                        .map(|(id, _)| id)
                        .collect();
                    calculate_recall(&hnsw_results, ground_truth)
                })
                .collect();

            recalls.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p99_idx = (recalls.len() as f64 * 0.01) as usize;
            recalls[p99_idx] * 100.0
        });
    });

    group.finish();
}

criterion_group!(
    name = recall_benches;
    config = Criterion::default()
        .significance_level(0.05);
    targets =
        bench_recall_at_k,
        bench_recall_vs_size,
        bench_recall_distribution,
);

criterion_main!(recall_benches);
