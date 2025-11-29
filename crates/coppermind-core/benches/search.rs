//! Benchmarks for search operations (HNSW, BM25, hybrid).
//!
//! Run with: `cargo bench -p coppermind-core --bench search`
//!
//! These benchmarks measure the performance of:
//! - Vector similarity search (HNSW)
//! - Keyword search (BM25)
//! - Hybrid search with RRF fusion
//! - Various k values and index sizes
//!
//! # Production Configuration
//!
//! These benchmarks use constants from `coppermind_core::config` to match
//! production settings. If those values change, benchmarks automatically
//! use the updated configuration.

use coppermind_core::config::EMBEDDING_DIM;
use coppermind_core::search::fusion::reciprocal_rank_fusion;
use coppermind_core::search::keyword::KeywordSearchEngine;
use coppermind_core::search::types::{Chunk, ChunkId, ChunkSourceMetadata};
use coppermind_core::search::vector::VectorSearchEngine;
use coppermind_core::search::HybridSearchEngine;
use coppermind_core::storage::InMemoryDocumentStore;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::Arc;

// =============================================================================
// Benchmark Configuration
// =============================================================================

/// Seed used for generating query embeddings.
///
/// This must be different from document seeds (0..N where N is index size)
/// to ensure we're benchmarking realistic search - where the query isn't
/// an exact match for any indexed document.
const QUERY_EMBEDDING_SEED: u64 = 1_000_000;

// =============================================================================
// Test Data Generation
// =============================================================================

/// Generate a deterministic L2-normalized embedding with a seed.
///
/// Produces vectors matching production embedding characteristics:
/// - Dimension matches `EMBEDDING_DIM`
/// - L2-normalized (||v|| = 1) like JinaBERT output
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

    // L2 normalize to match JinaBERT output (which is always normalized)
    let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    raw.into_iter().map(|x| x / norm).collect()
}

/// Generate realistic document text matching production chunk sizes.
///
/// Uses `TARGET_CHUNK_CHARS` sizing. Content varies by id for realistic
/// BM25 term distribution.
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
        "Document {id} provides a comprehensive analysis of {topic}. \
         This section explores the fundamental concepts and practical applications \
         that have emerged from recent advances in the field. Understanding these \
         principles is essential for building effective systems that can process \
         and analyze large volumes of textual data.\n\n\
         The implementation details covered here include various optimization \
         techniques and best practices that have been developed through extensive \
         research and real-world deployment. These methods have proven effective \
         across a wide range of use cases, from small-scale prototypes to \
         production systems handling millions of documents.\n\n\
         Key considerations when working with {topic} include computational \
         efficiency, memory usage, and the trade-offs between accuracy and speed. \
         Modern approaches leverage sophisticated algorithms that can achieve \
         near-optimal results while maintaining reasonable resource requirements. \
         This balance is crucial for practical applications where latency and \
         throughput are important factors.\n\n\
         The techniques described in this document have been validated through \
         rigorous benchmarking and evaluation against standard datasets. Results \
         demonstrate significant improvements over baseline methods, with \
         particularly notable gains in scenarios involving complex queries and \
         large document collections.\n\n\
         Additional context about {topic} reveals important patterns that emerge \
         when scaling these systems to handle enterprise workloads. Performance \
         characteristics vary significantly based on data distribution and query \
         patterns. Document identifier: {id}.",
        id = id,
        topic = topic
    )
}

/// Create a test chunk
fn create_chunk(id: u64) -> Chunk {
    Chunk {
        text: sample_text(id),
        metadata: ChunkSourceMetadata {
            filename: Some(format!("doc_{}.txt", id)),
            source: Some(format!("/test/doc_{}.txt", id)),
            created_at: 1700000000 + id,
        },
    }
}

/// Build a VectorSearchEngine with the given number of documents
fn build_vector_engine(size: usize) -> VectorSearchEngine {
    let mut engine = VectorSearchEngine::new(EMBEDDING_DIM);
    for i in 0..size {
        let _ = engine.add_chunk(ChunkId::from_u64(i as u64), seeded_embedding(i as u64));
    }
    engine
}

/// Build a KeywordSearchEngine with the given number of documents
fn build_keyword_engine(size: usize) -> KeywordSearchEngine {
    let mut engine = KeywordSearchEngine::new();
    for i in 0..size {
        engine.add_chunk(ChunkId::from_u64(i as u64), sample_text(i as u64));
    }
    engine
}

// ============================================================================
// HNSW Vector Search Benchmarks
// ============================================================================

/// Benchmark: Vector search with varying k values
///
/// Tests how search time scales with the number of results requested.
/// HNSW uses ef_search = max(k*2, 50), so larger k means more work.
fn bench_hnsw_search_varying_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw/search_by_k");
    group.sample_size(100);

    // Fixed index size, varying k
    let index_size = 600;
    let mut engine = build_vector_engine(index_size);
    let query = seeded_embedding(QUERY_EMBEDDING_SEED);

    for k in [1, 5, 10, 20, 50, 100] {
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            b.iter(|| engine.search(black_box(&query), k).unwrap());
        });
    }
    group.finish();
}

/// Benchmark: Vector search with varying index sizes
///
/// Tests O(log n) search complexity claim.
fn bench_hnsw_search_varying_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw/search_by_size");
    group.sample_size(100);

    let k = 10; // Fixed k
    let query = seeded_embedding(QUERY_EMBEDDING_SEED);

    for size in [100, 300, 600, 1000, 2000] {
        let mut engine = build_vector_engine(size);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| engine.search(black_box(&query), k).unwrap());
        });
    }
    group.finish();
}

// ============================================================================
// BM25 Keyword Search Benchmarks
// ============================================================================

/// Benchmark: BM25 search with varying index sizes
fn bench_bm25_search_varying_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("bm25/search_by_size");
    group.sample_size(100);

    let k = 10;
    let query = "machine learning semantic search";

    for size in [100, 300, 600, 1000, 2000] {
        let engine = build_keyword_engine(size);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| engine.search(black_box(query), k));
        });
    }
    group.finish();
}

/// Benchmark: BM25 search with varying query lengths
fn bench_bm25_search_varying_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("bm25/search_by_query_length");
    group.sample_size(100);

    let size = 600;
    let k = 10;
    let engine = build_keyword_engine(size);

    let queries = [
        ("1_word", "machine"),
        ("3_words", "machine learning search"),
        ("5_words", "machine learning semantic search retrieval"),
        (
            "10_words",
            "machine learning semantic search retrieval neural network embedding vector similarity",
        ),
    ];

    for (name, query) in queries {
        group.bench_with_input(BenchmarkId::from_parameter(name), &query, |b, query| {
            b.iter(|| engine.search(black_box(query), k));
        });
    }
    group.finish();
}

// ============================================================================
// Hybrid Search Benchmarks
// ============================================================================

/// Build a HybridSearchEngine with the given number of documents
fn build_hybrid_engine(
    rt: &tokio::runtime::Runtime,
    size: usize,
) -> HybridSearchEngine<Arc<InMemoryDocumentStore>> {
    let store = Arc::new(InMemoryDocumentStore::new());
    let mut engine = rt
        .block_on(HybridSearchEngine::new(Arc::clone(&store), EMBEDDING_DIM))
        .unwrap();

    for i in 0..size {
        rt.block_on(engine.add_chunk(create_chunk(i as u64), seeded_embedding(i as u64)))
            .unwrap();
    }
    engine
}

/// Benchmark: Full hybrid search pipeline
///
/// This measures the complete search path:
/// 1. Vector search (HNSW)
/// 2. Keyword search (BM25)
/// 3. RRF fusion
/// 4. Document retrieval from store
fn bench_hybrid_search(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("hybrid/search");
    group.sample_size(50);

    let k = 10;
    let query_text = "machine learning semantic search";
    let query_embedding = seeded_embedding(QUERY_EMBEDDING_SEED);

    for size in [100, 300, 600, 1000] {
        let mut engine = build_hybrid_engine(&rt, size);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                rt.block_on(async {
                    engine
                        .search(black_box(&query_embedding), black_box(query_text), k)
                        .await
                        .unwrap()
                })
            });
        });
    }
    group.finish();
}

/// Benchmark: Hybrid search with varying k values
fn bench_hybrid_search_varying_k(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("hybrid/search_by_k");
    group.sample_size(50);

    let size = 600;
    let query_text = "machine learning semantic search";
    let query_embedding = seeded_embedding(QUERY_EMBEDDING_SEED);

    let mut engine = build_hybrid_engine(&rt, size);

    for k in [1, 5, 10, 20, 50] {
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            b.iter(|| {
                rt.block_on(async {
                    engine
                        .search(black_box(&query_embedding), black_box(query_text), k)
                        .await
                        .unwrap()
                })
            });
        });
    }
    group.finish();
}

// ============================================================================
// Cold vs Warm Search Benchmarks
// ============================================================================

/// Benchmark: First search vs subsequent searches
///
/// HNSW has internal caching in the searcher. This tests if there's
/// a "warm-up" effect we should account for.
fn bench_hnsw_cold_vs_warm(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw/cold_vs_warm");
    group.sample_size(100);

    let size = 600;
    let k = 10;
    let query = seeded_embedding(QUERY_EMBEDDING_SEED);

    // Cold search: fresh engine each iteration (measures setup + search)
    group.bench_function("cold", |b| {
        b.iter_with_setup(
            || build_vector_engine(size),
            |mut engine| engine.search(black_box(&query), k).unwrap(),
        );
    });

    // Warm search: reuse same engine (searcher state persists)
    let mut warm_engine = build_vector_engine(size);
    // Warm up with a few queries
    for _ in 0..5 {
        let _ = warm_engine.search(&query, k);
    }

    group.bench_function("warm", |b| {
        b.iter(|| warm_engine.search(black_box(&query), k).unwrap());
    });

    group.finish();
}

// ============================================================================
// RRF Fusion Benchmarks
// ============================================================================

/// Benchmark: RRF fusion algorithm in isolation
///
/// Tests the fusion step independent of search engines.
fn bench_rrf_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("rrf/fusion");
    group.sample_size(1000);

    for size in [10, 50, 100, 200] {
        // Create mock rankings
        let vector_results: Vec<_> = (0..size)
            .map(|i| (ChunkId::from_u64(i as u64), 1.0 - (i as f32 / size as f32)))
            .collect();

        let keyword_results: Vec<_> = (0..size)
            .rev()
            .map(|i| (ChunkId::from_u64(i as u64), 1.0 - (i as f32 / size as f32)))
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                reciprocal_rank_fusion(
                    black_box(&vector_results),
                    black_box(&keyword_results),
                    size,
                )
            });
        });
    }
    group.finish();
}

// ============================================================================
// Hybrid Search Contribution Analysis
// ============================================================================

/// Benchmark: Compare vector-only vs BM25-only vs hybrid search.
///
/// This helps understand when hybrid search provides value over single-modality.
/// Key insight: hybrid excels when query has both semantic and keyword signals.
fn bench_search_modality_comparison(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("hybrid/modality_comparison");
    group.sample_size(50);

    let size = 1000;

    // Build all three engines with same data
    let mut vector_engine = VectorSearchEngine::new(EMBEDDING_DIM);
    let mut keyword_engine = KeywordSearchEngine::new();
    let store = Arc::new(InMemoryDocumentStore::new());
    let mut hybrid_engine = rt
        .block_on(HybridSearchEngine::new(Arc::clone(&store), EMBEDDING_DIM))
        .unwrap();

    for i in 0..size {
        let doc_id = ChunkId::from_u64(i as u64);
        let embedding = seeded_embedding(i as u64);
        let text = sample_text(i as u64);

        let _ = vector_engine.add_chunk(doc_id, embedding.clone());
        keyword_engine.add_chunk(doc_id, text.clone());
        rt.block_on(hybrid_engine.add_chunk(create_chunk(i as u64), embedding))
            .unwrap();
    }

    let k = 10;
    let query_embedding = seeded_embedding(QUERY_EMBEDDING_SEED);
    let query_text = "machine learning semantic search";

    // Vector-only search
    group.bench_function("vector_only", |b| {
        b.iter(|| {
            vector_engine
                .search(black_box(&query_embedding), k)
                .unwrap()
        });
    });

    // BM25-only search
    group.bench_function("bm25_only", |b| {
        b.iter(|| keyword_engine.search(black_box(query_text), k));
    });

    // Hybrid search
    group.bench_function("hybrid_rrf", |b| {
        b.iter(|| {
            rt.block_on(async {
                hybrid_engine
                    .search(black_box(&query_embedding), black_box(query_text), k)
                    .await
                    .unwrap()
            })
        });
    });

    group.finish();
}

// ============================================================================
// Query Distribution Benchmarks
// ============================================================================

/// Benchmark: Search performance across query complexity.
///
/// Tests how query length and specificity affect performance.
fn bench_query_complexity(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("query/complexity");
    group.sample_size(50);

    let size = 2000;

    // Build hybrid engine
    let store = Arc::new(InMemoryDocumentStore::new());
    let mut engine = rt
        .block_on(HybridSearchEngine::new(Arc::clone(&store), EMBEDDING_DIM))
        .unwrap();

    for i in 0..size {
        rt.block_on(engine.add_chunk(create_chunk(i as u64), seeded_embedding(i as u64)))
            .unwrap();
    }

    let k = 10;

    // Different query types
    let queries = [
        (
            "single_word",
            "machine",
            seeded_embedding(QUERY_EMBEDDING_SEED),
        ),
        (
            "short_phrase",
            "machine learning",
            seeded_embedding(QUERY_EMBEDDING_SEED + 1),
        ),
        (
            "medium_phrase",
            "machine learning neural networks",
            seeded_embedding(QUERY_EMBEDDING_SEED + 2),
        ),
        (
            "long_query",
            "machine learning neural networks for semantic search and information retrieval",
            seeded_embedding(QUERY_EMBEDDING_SEED + 3),
        ),
        (
            "question_format",
            "how do transformer models work for text understanding",
            seeded_embedding(QUERY_EMBEDDING_SEED + 4),
        ),
    ];

    for (name, text, embedding) in queries {
        group.bench_function(name, |b| {
            b.iter(|| {
                rt.block_on(async {
                    engine
                        .search(black_box(&embedding), black_box(text), k)
                        .await
                        .unwrap()
                })
            });
        });
    }

    group.finish();
}

// ============================================================================
// Dimension Sensitivity Benchmarks
// ============================================================================

/// Benchmark: HNSW search performance vs embedding dimension.
///
/// Tests how search latency scales with vector dimensionality.
/// Useful for comparing different embedding models (e.g., 384 vs 512 vs 768 vs 1024).
fn bench_dimension_sensitivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("dimension/search_latency");
    group.sample_size(50);

    let corpus_size = 5000;
    let k = 10;

    // Test various embedding dimensions
    let dimensions = [128, 256, 384, 512, 768, 1024];

    for dim in dimensions {
        // Generate embeddings for this dimension
        let embeddings: Vec<Vec<f32>> = (0..corpus_size)
            .map(|seed| {
                let raw: Vec<f32> = (0..dim)
                    .map(|i| {
                        use std::collections::hash_map::DefaultHasher;
                        use std::hash::{Hash, Hasher};
                        let mut hasher = DefaultHasher::new();
                        (seed as u64).hash(&mut hasher);
                        i.hash(&mut hasher);
                        let h = hasher.finish();
                        ((h as f32 / u64::MAX as f32) * 2.0) - 1.0
                    })
                    .collect();
                let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
                raw.into_iter().map(|x| x / norm).collect()
            })
            .collect();

        // Build index
        let mut engine = VectorSearchEngine::new(dim);
        for (i, embedding) in embeddings.iter().enumerate() {
            let _ = engine.add_chunk(ChunkId::from_u64(i as u64), embedding.clone());
        }

        // Query embedding
        let query: Vec<f32> = {
            let raw: Vec<f32> = (0..dim)
                .map(|i| {
                    use std::collections::hash_map::DefaultHasher;
                    use std::hash::{Hash, Hasher};
                    let mut hasher = DefaultHasher::new();
                    QUERY_EMBEDDING_SEED.hash(&mut hasher);
                    i.hash(&mut hasher);
                    let h = hasher.finish();
                    ((h as f32 / u64::MAX as f32) * 2.0) - 1.0
                })
                .collect();
            let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
            raw.into_iter().map(|x| x / norm).collect()
        };

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| engine.search(black_box(&query), k).unwrap());
        });
    }

    group.finish();
}

/// Benchmark: HNSW index build time vs embedding dimension.
fn bench_dimension_build_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("dimension/build_time");
    group.sample_size(10);

    let corpus_size = 2000;

    let dimensions = [128, 256, 384, 512, 768, 1024];

    for dim in dimensions {
        // Pre-generate embeddings
        let embeddings: Vec<Vec<f32>> = (0..corpus_size)
            .map(|seed| {
                let raw: Vec<f32> = (0..dim)
                    .map(|i| {
                        use std::collections::hash_map::DefaultHasher;
                        use std::hash::{Hash, Hasher};
                        let mut hasher = DefaultHasher::new();
                        (seed as u64).hash(&mut hasher);
                        i.hash(&mut hasher);
                        let h = hasher.finish();
                        ((h as f32 / u64::MAX as f32) * 2.0) - 1.0
                    })
                    .collect();
                let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
                raw.into_iter().map(|x| x / norm).collect()
            })
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, &dim| {
            b.iter(|| {
                let mut engine = VectorSearchEngine::new(dim);
                for (i, embedding) in embeddings.iter().enumerate() {
                    let _ = engine.add_chunk(ChunkId::from_u64(i as u64), embedding.clone());
                }
                engine
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_hnsw_search_varying_k,
    bench_hnsw_search_varying_size,
    bench_bm25_search_varying_size,
    bench_bm25_search_varying_query,
    bench_hybrid_search,
    bench_hybrid_search_varying_k,
    bench_hnsw_cold_vs_warm,
    bench_rrf_fusion,
    bench_search_modality_comparison,
    bench_query_complexity,
    bench_dimension_sensitivity,
    bench_dimension_build_time,
);

criterion_main!(benches);
