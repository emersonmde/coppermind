//! Benchmarks for indexing operations (HNSW, BM25, hybrid).
//!
//! Run with: `cargo bench -p coppermind-core --bench indexing`
//!
//! These benchmarks measure the performance of:
//! - Single document insertion
//! - Batch document insertion
//! - Index rebuild from storage
//! - Vector index compaction
//!
//! # Production Configuration
//!
//! These benchmarks use constants from `coppermind_core::config` to match
//! production settings. If those values change, benchmarks automatically
//! use the updated configuration.

use coppermind_core::config::EMBEDDING_DIM;
use coppermind_core::search::keyword::KeywordSearchEngine;
use coppermind_core::search::types::{DocId, Document, DocumentMetadata, DocumentRecord};
use coppermind_core::search::vector::VectorSearchEngine;
use coppermind_core::search::HybridSearchEngine;
use coppermind_core::storage::{DocumentStore, InMemoryDocumentStore};
use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use std::sync::Arc;
use std::time::Duration;

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
/// Uses `TARGET_CHUNK_CHARS` from config to generate appropriately sized text.
/// Content varies by id for realistic BM25 term distribution.
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

    // Generate text targeting TARGET_CHUNK_CHARS (~2048 chars for 512 tokens)
    // This template produces ~1900-2100 chars depending on topic length
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

/// Create a test document
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

// ============================================================================
// HNSW Vector Index Benchmarks
// ============================================================================

/// Benchmark: Single vector insertion into HNSW
///
/// This measures the O(log n) insertion time for a single document.
/// Useful for detecting regressions in incremental indexing.
fn bench_hnsw_single_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw/insert_single");
    group.sample_size(100);

    // Test with different pre-existing index sizes
    for base_size in [0, 100, 500, 1000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("base_{}", base_size)),
            &base_size,
            |b, &base_size| {
                b.iter_batched(
                    || {
                        // Setup: create engine with base_size documents
                        let mut engine = VectorSearchEngine::new(EMBEDDING_DIM);
                        for i in 0..base_size {
                            let _ = engine.add_document(
                                DocId::from_u64(i as u64),
                                seeded_embedding(i as u64),
                            );
                        }
                        engine
                    },
                    |mut engine| {
                        // Benchmark: insert one more document
                        let _ = engine.add_document(
                            DocId::from_u64(base_size as u64),
                            black_box(seeded_embedding(base_size as u64)),
                        );
                        engine
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

/// Benchmark: Batch vector insertion (building index from scratch)
///
/// This is the critical path for index rebuild - the 10 second issue is here.
/// Measures total time to insert N vectors into an empty index.
fn bench_hnsw_batch_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw/insert_batch");
    group.sample_size(10); // Fewer samples for larger batches

    // Test various corpus sizes - these match real-world usage patterns
    for size in [100, 300, 600, 1000] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            // Pre-generate embeddings outside the benchmark
            let embeddings: Vec<_> = (0..size).map(|i| seeded_embedding(i as u64)).collect();

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

// ============================================================================
// BM25 Keyword Index Benchmarks
// ============================================================================

/// Benchmark: Batch BM25 document insertion
///
/// BM25 is typically much faster than HNSW. This baseline helps identify
/// if slowdowns are in HNSW specifically vs overall indexing.
fn bench_bm25_batch_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("bm25/insert_batch");
    group.sample_size(20);

    for size in [100, 300, 600, 1000] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            // Pre-generate texts
            let texts: Vec<_> = (0..size).map(|i| sample_text(i as u64)).collect();

            b.iter(|| {
                let mut engine = KeywordSearchEngine::new();
                for (i, text) in texts.iter().enumerate() {
                    engine.add_document(DocId::from_u64(i as u64), text.clone());
                }
                engine
            });
        });
    }
    group.finish();
}

// ============================================================================
// Hybrid Engine Benchmarks (Full Pipeline)
// ============================================================================

/// Benchmark: Hybrid engine document addition
///
/// Tests the full pipeline: storage write + HNSW insert + BM25 insert
fn bench_hybrid_add_document(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("hybrid/add_document");
    group.sample_size(50);

    for base_size in [0, 100, 500] {
        // Pre-build the engine with base_size documents
        let store = InMemoryDocumentStore::new();
        let mut engine = rt
            .block_on(HybridSearchEngine::new(store, EMBEDDING_DIM))
            .unwrap();

        for i in 0..base_size {
            rt.block_on(engine.add_document(create_document(i as u64), seeded_embedding(i as u64)))
                .unwrap();
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("base_{}", base_size)),
            &base_size,
            |b, &base_size| {
                let mut counter = base_size;
                b.iter(|| {
                    // Benchmark: add one document
                    rt.block_on(async {
                        engine
                            .add_document(
                                create_document(counter as u64),
                                black_box(seeded_embedding(counter as u64)),
                            )
                            .await
                            .unwrap()
                    });
                    counter += 1;
                });
            },
        );
    }
    group.finish();
}

/// Helper: populate a store with test data (documents + embeddings)
async fn populate_store(store: &InMemoryDocumentStore, size: usize) {
    for i in 0..size {
        let doc_id = DocId::from_u64(i as u64);
        let record = DocumentRecord {
            id: doc_id,
            text: sample_text(i as u64),
            metadata: DocumentMetadata {
                filename: Some(format!("doc_{}.txt", i)),
                source: Some(format!("/test/doc_{}.txt", i)),
                created_at: 1700000000 + i as u64,
            },
        };
        let embedding = seeded_embedding(i as u64);

        store.put_document(doc_id, &record).await.unwrap();
        store.put_embedding(doc_id, &embedding).await.unwrap();
    }
}

/// Benchmark: Index rebuild from storage
///
/// This is the most critical benchmark - simulates app startup with existing data.
/// The 10-second issue manifests here for 600 chunks.
fn bench_hybrid_rebuild(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("hybrid/rebuild_from_store");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30)); // Allow longer measurement

    for size in [100, 300, 600] {
        // Pre-populate a store that can be cloned via Arc
        let store = Arc::new(InMemoryDocumentStore::new());
        rt.block_on(populate_store(&store, size));

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let store_clone = Arc::clone(&store);
                // Benchmark: rebuild indices from store (this is the slow path)
                rt.block_on(async {
                    HybridSearchEngine::try_load_or_new(store_clone, EMBEDDING_DIM)
                        .await
                        .unwrap()
                })
            });
        });
    }
    group.finish();
}

// ============================================================================
// Vector Index Compaction Benchmarks
// ============================================================================

/// Benchmark: HNSW compaction (rebuild to remove tombstones)
///
/// Compaction rebuilds the entire index. This is expensive but necessary
/// when tombstone ratio exceeds threshold.
fn bench_hnsw_compaction(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw/compaction");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    for size in [100, 300, 600] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            // Pre-generate embeddings with DocIds
            let entries: Vec<_> = (0..size)
                .map(|i| (DocId::from_u64(i as u64), seeded_embedding(i as u64)))
                .collect();

            b.iter_batched(
                || {
                    // Setup: create index and mark ~30% as tombstones
                    let mut engine = VectorSearchEngine::new(EMBEDDING_DIM);
                    for (doc_id, embedding) in &entries {
                        let _ = engine.add_document(*doc_id, embedding.clone());
                    }

                    // Mark every 3rd document as tombstone
                    for i in (0..size).step_by(3) {
                        engine.mark_tombstone(i);
                    }

                    (engine, entries.clone())
                },
                |(mut engine, entries)| {
                    // Benchmark: compact the index
                    let _ = engine.compact(entries);
                    engine
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_hnsw_single_insert,
    bench_hnsw_batch_insert,
    bench_bm25_batch_insert,
    bench_hybrid_add_document,
    bench_hybrid_rebuild,
    bench_hnsw_compaction,
);

criterion_main!(benches);
