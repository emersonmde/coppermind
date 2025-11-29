//! Throughput benchmarks measuring queries per second (QPS).
//!
//! Run with: `cargo bench -p coppermind-core --bench throughput`
//!
//! These benchmarks measure production-relevant throughput metrics:
//!
//! - **QPS**: Queries per second at various concurrency levels
//! - **Latency percentiles**: p50, p95, p99 under load
//! - **Saturation point**: Where does throughput plateau?
//!
//! # Concurrency Model
//!
//! Tests multi-threaded query execution to simulate production load.
//! Uses Rayon for parallel iteration to stress-test thread safety
//! and measure contention overhead.

use coppermind_core::config::EMBEDDING_DIM;
use coppermind_core::search::keyword::KeywordSearchEngine;
use coppermind_core::search::types::{DocId, Document, DocumentMetadata};
use coppermind_core::search::vector::VectorSearchEngine;
use coppermind_core::search::HybridSearchEngine;
use coppermind_core::storage::InMemoryDocumentStore;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::Arc;
use std::time::{Duration, Instant};

// =============================================================================
// Configuration
// =============================================================================

/// Corpus size for throughput testing.
const CORPUS_SIZE: usize = 10_000;

/// Number of queries per batch for QPS measurement.
const QUERIES_PER_BATCH: usize = 1_000;

/// Concurrency levels to test.
const CONCURRENCY_LEVELS: &[usize] = &[1, 2, 4, 8, 16];

/// Query seed base.
const QUERY_SEED_BASE: u64 = 1_000_000;

// =============================================================================
// Test Data Generation
// =============================================================================

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

fn sample_text(id: u64) -> String {
    let topics = [
        "machine learning algorithms and neural network architectures",
        "semantic search engines and information retrieval systems",
        "natural language processing and text understanding",
        "vector embeddings and similarity metrics",
    ];
    let topic = topics[(id % topics.len() as u64) as usize];
    format!("Document {id} about {topic}. Content for testing. ID: {id}.")
}

fn build_vector_engine(size: usize) -> VectorSearchEngine {
    let mut engine = VectorSearchEngine::new(EMBEDDING_DIM);
    for i in 0..size {
        let _ = engine.add_document(DocId::from_u64(i as u64), seeded_embedding(i as u64));
    }
    engine
}

fn create_document(id: u64) -> Document {
    Document {
        text: sample_text(id),
        metadata: DocumentMetadata {
            filename: Some(format!("doc_{}.txt", id)),
            source: Some(format!("/test/doc_{}.txt", id)),
            created_at: 1700000000 + id,
        },
    }
}

// =============================================================================
// Sequential Throughput Benchmarks
// =============================================================================

/// Benchmark: HNSW queries per second (single-threaded baseline).
fn bench_hnsw_qps_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput/hnsw_sequential");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(10));

    // Build index
    let mut engine = VectorSearchEngine::new(EMBEDDING_DIM);
    for i in 0..CORPUS_SIZE {
        let _ = engine.add_document(DocId::from_u64(i as u64), seeded_embedding(i as u64));
    }

    // Pre-generate queries
    let queries: Vec<_> = (0..QUERIES_PER_BATCH)
        .map(|i| seeded_embedding(QUERY_SEED_BASE + i as u64))
        .collect();

    let k = 10;

    group.throughput(Throughput::Elements(QUERIES_PER_BATCH as u64));
    group.bench_function("baseline", |b| {
        b.iter(|| {
            for query in &queries {
                let _ = engine.search(black_box(query), k);
            }
        });
    });

    group.finish();
}

/// Benchmark: BM25 queries per second (single-threaded baseline).
fn bench_bm25_qps_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput/bm25_sequential");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(10));

    // Build index
    let mut engine = KeywordSearchEngine::new();
    for i in 0..CORPUS_SIZE {
        engine.add_document(DocId::from_u64(i as u64), sample_text(i as u64));
    }

    // Pre-generate queries (varying to avoid caching effects)
    let query_texts = [
        "machine learning",
        "semantic search",
        "neural network",
        "text understanding",
        "vector embeddings",
        "information retrieval",
        "natural language",
        "similarity metrics",
    ];

    let k = 10;

    group.throughput(Throughput::Elements(QUERIES_PER_BATCH as u64));
    group.bench_function("baseline", |b| {
        b.iter(|| {
            for i in 0..QUERIES_PER_BATCH {
                let query = query_texts[i % query_texts.len()];
                let _ = engine.search(black_box(query), k);
            }
        });
    });

    group.finish();
}

// =============================================================================
// Concurrent Throughput Benchmarks
// =============================================================================

/// Benchmark: HNSW concurrent query throughput.
///
/// Uses scoped threads to run queries in parallel and measure total QPS.
fn bench_hnsw_qps_concurrent(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput/hnsw_concurrent");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));

    // Pre-generate queries
    let queries: Vec<_> = (0..QUERIES_PER_BATCH)
        .map(|i| seeded_embedding(QUERY_SEED_BASE + i as u64))
        .collect();

    let k = 10;

    for &num_threads in CONCURRENCY_LEVELS {
        let queries_per_thread = QUERIES_PER_BATCH / num_threads;
        let total_queries = queries_per_thread * num_threads;

        group.throughput(Throughput::Elements(total_queries as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_threads", num_threads)),
            &num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    // Each thread gets its own engine since search requires &mut self
                    std::thread::scope(|s| {
                        let queries_ref = &queries;

                        let handles: Vec<_> = (0..num_threads)
                            .map(|thread_id| {
                                s.spawn(move || {
                                    let mut thread_engine = build_vector_engine(CORPUS_SIZE);
                                    let start = thread_id * queries_per_thread;
                                    let end = start + queries_per_thread;
                                    for query in &queries_ref[start..end] {
                                        let _ = thread_engine.search(black_box(query), k);
                                    }
                                })
                            })
                            .collect();

                        for handle in handles {
                            handle.join().unwrap();
                        }
                    });
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Latency Percentile Benchmarks
// =============================================================================

/// Benchmark: HNSW latency percentiles under load.
///
/// Measures p50, p95, p99 latencies for individual queries.
fn bench_hnsw_latency_percentiles(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput/hnsw_latency");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(10));

    // Build index
    let mut engine = VectorSearchEngine::new(EMBEDDING_DIM);
    for i in 0..CORPUS_SIZE {
        let _ = engine.add_document(DocId::from_u64(i as u64), seeded_embedding(i as u64));
    }

    // Pre-generate queries
    let queries: Vec<_> = (0..500)
        .map(|i| seeded_embedding(QUERY_SEED_BASE + i as u64))
        .collect();

    let k = 10;

    // Measure latencies
    group.bench_function("measure_latencies", |b| {
        b.iter_custom(|iters| {
            let mut total_duration = Duration::ZERO;

            for _ in 0..iters {
                let mut latencies: Vec<Duration> = Vec::with_capacity(queries.len());

                for query in &queries {
                    let start = Instant::now();
                    let _ = engine.search(black_box(query), k);
                    latencies.push(start.elapsed());
                }

                latencies.sort();

                // Sum p50, p95, p99 for this iteration
                let p50 = latencies[latencies.len() / 2];
                let p95 = latencies[(latencies.len() as f64 * 0.95) as usize];
                let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];

                total_duration += p50 + p95 + p99;
            }

            total_duration
        });
    });

    group.finish();
}

// =============================================================================
// Hybrid Search Throughput
// =============================================================================

/// Benchmark: Hybrid search QPS.
fn bench_hybrid_qps(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("throughput/hybrid");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));

    // Build hybrid engine with smaller corpus for feasibility
    let hybrid_corpus_size = 5_000;
    let store = Arc::new(InMemoryDocumentStore::new());
    let mut engine = rt
        .block_on(HybridSearchEngine::new(Arc::clone(&store), EMBEDDING_DIM))
        .unwrap();

    for i in 0..hybrid_corpus_size {
        rt.block_on(engine.add_document(create_document(i as u64), seeded_embedding(i as u64)))
            .unwrap();
    }

    // Pre-generate queries
    let query_count = 100;
    let queries: Vec<_> = (0..query_count)
        .map(|i| {
            (
                seeded_embedding(QUERY_SEED_BASE + i as u64),
                "machine learning semantic search",
            )
        })
        .collect();

    let k = 10;

    group.throughput(Throughput::Elements(query_count as u64));
    group.bench_function("sequential", |b| {
        b.iter(|| {
            rt.block_on(async {
                for (embedding, text) in &queries {
                    let _ = engine
                        .search(black_box(embedding), black_box(*text), k)
                        .await;
                }
            });
        });
    });

    group.finish();
}

// =============================================================================
// Sustained Load Benchmark
// =============================================================================

/// Benchmark: Sustained load over time.
///
/// Measures whether performance degrades under sustained query load
/// (e.g., due to memory pressure, cache pollution).
fn bench_sustained_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput/sustained");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    // Build index
    let mut engine = VectorSearchEngine::new(EMBEDDING_DIM);
    for i in 0..CORPUS_SIZE {
        let _ = engine.add_document(DocId::from_u64(i as u64), seeded_embedding(i as u64));
    }

    // Generate many unique queries to avoid caching
    let num_queries = 10_000;
    let queries: Vec<_> = (0..num_queries)
        .map(|i| seeded_embedding(QUERY_SEED_BASE + i as u64))
        .collect();

    let k = 10;

    group.throughput(Throughput::Elements(num_queries as u64));
    group.bench_function("10k_queries", |b| {
        b.iter(|| {
            for query in &queries {
                let _ = engine.search(black_box(query), k);
            }
        });
    });

    group.finish();
}

criterion_group!(
    name = throughput_benches;
    config = Criterion::default()
        .significance_level(0.05)
        .noise_threshold(0.02);
    targets =
        bench_hnsw_qps_sequential,
        bench_bm25_qps_sequential,
        bench_hnsw_qps_concurrent,
        bench_hnsw_latency_percentiles,
        bench_hybrid_qps,
        bench_sustained_load,
);

criterion_main!(throughput_benches);
