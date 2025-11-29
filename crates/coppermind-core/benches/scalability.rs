//! Scalability benchmarks for large corpus sizes.
//!
//! Run with: `cargo bench -p coppermind-core --bench scalability`
//!
//! These benchmarks measure how search and indexing performance scales
//! with corpus size, testing from 100 to 100K+ documents to validate
//! algorithmic complexity claims:
//!
//! - HNSW search: O(log n) expected
//! - HNSW insertion: O(log n) expected
//! - BM25 search: O(n * q) where q is query terms
//!
//! # Memory Considerations
//!
//! Large corpus benchmarks require significant memory:
//! - 100K documents × 512 dimensions × 4 bytes = ~200MB for vectors alone
//! - Plus HNSW graph structure overhead (~2-3x vector size)
//!
//! Run with `--bench scalability` to isolate these from quick benchmarks.

use coppermind_core::config::EMBEDDING_DIM;
use coppermind_core::search::keyword::KeywordSearchEngine;
use coppermind_core::search::types::DocId;
use coppermind_core::search::vector::VectorSearchEngine;
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration,
    Throughput,
};
use std::time::Duration;

// =============================================================================
// Benchmark Configuration
// =============================================================================

/// Corpus sizes for scalability testing.
///
/// Logarithmically spaced to show O(log n) behavior clearly on log-scale plots.
/// Adjust based on available memory and acceptable benchmark duration.
const CORPUS_SIZES: &[usize] = &[
    100,     // Baseline
    500,     // Small
    1_000,   // 1K
    5_000,   // 5K
    10_000,  // 10K
    25_000,  // 25K
    50_000,  // 50K - memory: ~100MB vectors
    100_000, // 100K - memory: ~200MB vectors (comment out if memory constrained)
];

/// Seed for query embeddings (must differ from document seeds).
const QUERY_EMBEDDING_SEED: u64 = 1_000_000;

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

/// Generate realistic document text.
fn sample_text(id: u64) -> String {
    let topics = [
        "machine learning algorithms and neural network architectures",
        "semantic search engines and information retrieval systems",
        "natural language processing and text understanding",
        "vector embeddings and similarity metrics",
        "transformer models and attention mechanisms",
        "approximate nearest neighbor search algorithms",
        "text classification and sentiment analysis",
        "knowledge graphs and entity recognition",
    ];
    let topic = topics[(id % topics.len() as u64) as usize];

    format!(
        "Document {id} discusses {topic}. This content explores key concepts \
         and practical applications in the field. The implementation covers \
         optimization techniques and best practices developed through research. \
         Performance characteristics vary based on data distribution. ID: {id}.",
        id = id,
        topic = topic
    )
}

// =============================================================================
// Pre-built Index Cache
// =============================================================================

/// Pre-build vector indices for each corpus size to avoid rebuild overhead.
///
/// Building a 100K index takes significant time - we only want to measure
/// search latency, not index construction (that's in indexing.rs).
fn build_vector_engine(size: usize) -> VectorSearchEngine {
    let mut engine = VectorSearchEngine::new(EMBEDDING_DIM);
    for i in 0..size {
        let _ = engine.add_document(DocId::from_u64(i as u64), seeded_embedding(i as u64));
    }
    engine
}

fn build_keyword_engine(size: usize) -> KeywordSearchEngine {
    let mut engine = KeywordSearchEngine::new();
    for i in 0..size {
        engine.add_document(DocId::from_u64(i as u64), sample_text(i as u64));
    }
    engine
}

// =============================================================================
// HNSW Scalability Benchmarks
// =============================================================================

/// Benchmark: HNSW search latency vs corpus size.
///
/// This is the primary scalability test - demonstrates O(log n) search time.
/// Uses log-scale plotting to visualize the logarithmic relationship.
fn bench_hnsw_search_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability/hnsw_search");

    // Configure for log-scale plotting
    group.plot_config(PlotConfiguration::default());
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(10));

    let k = 10;
    let query = seeded_embedding(QUERY_EMBEDDING_SEED);

    for &size in CORPUS_SIZES {
        // Skip very large sizes if they would take too long to build
        if size > 100_000 {
            continue;
        }

        let mut engine = build_vector_engine(size);

        group.throughput(Throughput::Elements(1)); // 1 query per iteration
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| engine.search(black_box(&query), k).unwrap());
        });
    }
    group.finish();
}

/// Benchmark: HNSW insertion latency vs existing index size.
///
/// Tests O(log n) insertion complexity by measuring single-document
/// insertion into indices of varying sizes.
fn bench_hnsw_insert_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability/hnsw_insert");

    group.plot_config(PlotConfiguration::default());
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(10));

    for &size in CORPUS_SIZES {
        if size > 50_000 {
            // Skip largest sizes for insertion benchmark (too slow to rebuild each iteration)
            continue;
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched(
                || {
                    // Setup: build index with `size` documents
                    build_vector_engine(size)
                },
                |mut engine| {
                    // Benchmark: insert one more document
                    let _ = engine.add_document(
                        DocId::from_u64(size as u64),
                        black_box(seeded_embedding(size as u64)),
                    );
                    engine
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

/// Benchmark: HNSW batch build time vs corpus size.
///
/// Measures total time to build an index from scratch.
/// Important for understanding cold-start and rebuild scenarios.
fn bench_hnsw_build_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability/hnsw_build");

    group.plot_config(PlotConfiguration::default());
    group.sample_size(10); // Fewer samples for expensive builds
    group.measurement_time(Duration::from_secs(30));

    // Use smaller sizes for build benchmarks (building 100K is very slow)
    let build_sizes: &[usize] = &[100, 500, 1_000, 2_500, 5_000, 10_000];

    for &size in build_sizes {
        // Pre-generate embeddings outside benchmark
        let embeddings: Vec<_> = (0..size).map(|i| seeded_embedding(i as u64)).collect();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let mut engine = VectorSearchEngine::new(EMBEDDING_DIM);
                for (i, embedding) in embeddings.iter().enumerate() {
                    let _ = engine.add_document(DocId::from_u64(i as u64), embedding.clone());
                }
                engine
            });
        });
    }
    group.finish();
}

// =============================================================================
// BM25 Scalability Benchmarks
// =============================================================================

/// Benchmark: BM25 search latency vs corpus size.
///
/// BM25 is O(n) in corpus size for each query term, but with good constants
/// due to inverted index optimization.
fn bench_bm25_search_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability/bm25_search");

    group.plot_config(PlotConfiguration::default());
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(10));

    let k = 10;
    let query = "machine learning semantic search";

    for &size in CORPUS_SIZES {
        if size > 100_000 {
            continue;
        }

        let engine = build_keyword_engine(size);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| engine.search(black_box(query), k));
        });
    }
    group.finish();
}

// =============================================================================
// Memory Estimation (Informational)
// =============================================================================

/// Benchmark that reports approximate memory usage per corpus size.
///
/// This doesn't measure latency - it provides memory usage data points
/// that can be plotted alongside latency benchmarks.
fn bench_memory_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability/memory_estimate");
    group.sample_size(10);

    // Memory estimation sizes (smaller set for quick runs)
    let memory_sizes: &[usize] = &[1_000, 5_000, 10_000, 25_000];

    for &size in memory_sizes {
        // Rough memory estimate:
        // - Vectors: size * EMBEDDING_DIM * 4 bytes
        // - HNSW graph: ~2-3x vector size (M=16 means ~32 neighbors per node on average)
        let vector_bytes = size * EMBEDDING_DIM * 4;
        let estimated_total = vector_bytes * 3; // Conservative 3x multiplier

        group.throughput(Throughput::Bytes(estimated_total as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            // Build engine of target size and measure a search
            // This gives us timing data correlated with memory size
            b.iter_batched(
                || build_vector_engine(size),
                |mut engine| {
                    let query = seeded_embedding(QUERY_EMBEDDING_SEED);
                    engine.search(black_box(&query), 10).unwrap()
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

criterion_group!(
    name = scalability_benches;
    config = Criterion::default()
        .significance_level(0.05)
        .noise_threshold(0.02);
    targets =
        bench_hnsw_search_scalability,
        bench_hnsw_insert_scalability,
        bench_hnsw_build_scalability,
        bench_bm25_search_scalability,
        bench_memory_scaling,
);

criterion_main!(scalability_benches);
