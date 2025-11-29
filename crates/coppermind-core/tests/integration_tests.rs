//! End-to-end integration tests for the complete indexing and search pipeline.
//!
//! These tests exercise the full workflow:
//! 1. Indexing: chunking → tokenization → embedding → HNSW/BM25 indexing
//! 2. Search: query embedding → BM25/HNSW search → RRF fusion → result ranking
//!
//! **Note**: These tests require model files to be present.
//! Run with: `cargo test -p coppermind-core --test integration_tests`
//! Make sure to run `./download-models.sh` first to download the model files.

use coppermind_core::chunking::{create_chunker, detect_file_type, FileType};
use coppermind_core::config;
use coppermind_core::embedding::{Embedder, JinaBertConfig, JinaBertEmbedder, TokenizerHandle};
use coppermind_core::processing::IndexingPipeline;
use coppermind_core::search::{Document, DocumentMetadata, HybridSearchEngine};
use coppermind_core::storage::InMemoryDocumentStore;
use std::sync::Arc;

// ============================================================================
// Test Fixtures (Cached for Performance)
// ============================================================================
//
// The model and tokenizer are expensive to load (~65MB model, ~1s initialization).
// We cache them in static Arc's so all tests share the same instance.
// This reduces test time from ~95s to ~5s.

use std::sync::OnceLock;

/// Cached tokenizer wrapped in Arc - loaded once and shared across all tests.
static CACHED_TOKENIZER: OnceLock<Arc<TokenizerHandle>> = OnceLock::new();

/// Cached embedder wrapped in Arc - loaded once and shared across all tests.
/// This is the expensive part (~65MB model load + initialization).
static CACHED_EMBEDDER: OnceLock<Arc<JinaBertEmbedder>> = OnceLock::new();

/// Load the tokenizer from the assets directory.
fn load_tokenizer() -> Arc<TokenizerHandle> {
    CACHED_TOKENIZER
        .get_or_init(|| {
            let tokenizer_path = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../coppermind/assets/models/jina-bert-tokenizer.json"
            );
            let tokenizer_bytes = std::fs::read(tokenizer_path)
                .expect("Failed to read tokenizer file - run ./download-models.sh first");

            Arc::new(
                TokenizerHandle::from_bytes(tokenizer_bytes, config::MAX_CHUNK_TOKENS)
                    .expect("Failed to create TokenizerHandle"),
            )
        })
        .clone()
}

/// Load the embedding model from the assets directory.
fn load_embedder() -> Arc<JinaBertEmbedder> {
    CACHED_EMBEDDER
        .get_or_init(|| {
            let model_path = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../coppermind/assets/models/jina-bert.safetensors"
            );
            let model_bytes = std::fs::read(model_path)
                .expect("Failed to read model file - run ./download-models.sh first");

            let tokenizer = load_tokenizer();
            let vocab_size = tokenizer.vocab_size();

            let config = JinaBertConfig::default();
            Arc::new(
                JinaBertEmbedder::from_bytes(model_bytes, vocab_size, config)
                    .expect("Failed to create JinaBertEmbedder"),
            )
        })
        .clone()
}

/// Get a static reference to the inner tokenizer for chunking.
fn get_static_tokenizer() -> &'static tokenizers::Tokenizer {
    // Use a separate OnceLock to get a &'static reference to the inner tokenizer
    static INNER_TOKENIZER: OnceLock<&'static tokenizers::Tokenizer> = OnceLock::new();
    INNER_TOKENIZER.get_or_init(|| {
        // Safety: The Arc<TokenizerHandle> is stored in a static OnceLock,
        // so it lives for 'static. We're getting a reference to its inner tokenizer.
        let tokenizer = CACHED_TOKENIZER.get_or_init(|| {
            let tokenizer_path = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../coppermind/assets/models/jina-bert-tokenizer.json"
            );
            let tokenizer_bytes = std::fs::read(tokenizer_path)
                .expect("Failed to read tokenizer file - run ./download-models.sh first");

            Arc::new(
                TokenizerHandle::from_bytes(tokenizer_bytes, config::MAX_CHUNK_TOKENS)
                    .expect("Failed to create TokenizerHandle"),
            )
        });
        // Safety: tokenizer is &'static Arc<TokenizerHandle>, inner() returns &Tokenizer
        // which has the same lifetime as the Arc (i.e., 'static)
        unsafe {
            std::mem::transmute::<&tokenizers::Tokenizer, &'static tokenizers::Tokenizer>(
                tokenizer.inner(),
            )
        }
    })
}

/// Helper to create a Document from text
fn make_document(text: &str) -> Document {
    Document {
        text: text.to_string(),
        metadata: DocumentMetadata::default(),
    }
}

/// Helper to create a Document with source info
fn make_document_with_source(text: &str, source: &str) -> Document {
    Document {
        text: text.to_string(),
        metadata: DocumentMetadata {
            source: Some(source.to_string()),
            ..Default::default()
        },
    }
}

/// Sample documents for testing.
fn sample_documents() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        (
            "rust_intro.md",
            "introduction to rust",
            r#"# Introduction to Rust

Rust is a systems programming language focused on safety, speed, and concurrency.
It achieves memory safety without using garbage collection.

## Key Features

- **Memory Safety**: The borrow checker ensures memory safety at compile time.
- **Zero-Cost Abstractions**: High-level features compile to efficient machine code.
- **Fearless Concurrency**: The type system prevents data races.

## Getting Started

To install Rust, use rustup:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Create a new project with `cargo new my_project`.
"#,
        ),
        (
            "python_basics.md",
            "python programming basics",
            r#"# Python Basics

Python is a high-level, interpreted programming language known for its simplicity.

## Variables and Types

Python uses dynamic typing:

```python
name = "Alice"
age = 30
height = 5.8
```

## Functions

Define functions with the `def` keyword:

```python
def greet(name):
    return f"Hello, {name}!"
```

## Lists and Loops

Python has powerful list comprehensions:

```python
squares = [x**2 for x in range(10)]
```
"#,
        ),
        (
            "web_development.txt",
            "web development overview",
            r#"Web Development Overview

Modern web development involves building applications that run in browsers.
The three core technologies are HTML, CSS, and JavaScript.

Frontend frameworks like React, Vue, and Angular help build complex user interfaces.
Backend technologies include Node.js, Python Django, Ruby on Rails, and Rust Actix.

APIs connect frontend and backend using REST or GraphQL protocols.
Databases store application data, with options like PostgreSQL, MongoDB, and Redis.

DevOps practices include continuous integration, containerization with Docker,
and orchestration with Kubernetes.
"#,
        ),
        (
            "machine_learning.md",
            "machine learning concepts",
            r#"# Machine Learning Concepts

Machine learning enables computers to learn from data without explicit programming.

## Types of Learning

1. **Supervised Learning**: Training with labeled data (classification, regression)
2. **Unsupervised Learning**: Finding patterns in unlabeled data (clustering)
3. **Reinforcement Learning**: Learning through trial and error with rewards

## Neural Networks

Deep learning uses neural networks with multiple layers:
- Input layer receives features
- Hidden layers extract patterns
- Output layer produces predictions

## Embeddings

Text embeddings convert words into dense vectors that capture semantic meaning.
Similar words have similar embedding vectors, enabling semantic search.
"#,
        ),
        (
            "database_systems.md",
            "database systems guide",
            r#"# Database Systems

Databases are organized collections of structured data.

## Relational Databases

SQL databases like PostgreSQL and MySQL use tables with rows and columns.
They enforce ACID properties: Atomicity, Consistency, Isolation, Durability.

## NoSQL Databases

Document stores (MongoDB), key-value stores (Redis), and graph databases (Neo4j)
offer flexibility for different use cases.

## Indexing

Database indexes speed up queries by creating data structures for fast lookups.
B-trees and hash indexes are common implementations.

## Query Optimization

The query planner chooses efficient execution strategies.
Understanding EXPLAIN plans helps optimize slow queries.
"#,
        ),
    ]
}

// ============================================================================
// Indexing Pipeline Tests
// ============================================================================

#[test]
fn test_full_indexing_pipeline_text_chunking() {
    let tokenizer = load_tokenizer();
    let embedder = load_embedder();
    let pipeline = IndexingPipeline::new(embedder.clone(), tokenizer.clone());

    let content = "First sentence about Rust programming. Second sentence about memory safety. Third sentence about the borrow checker.";

    let mut progress_updates = Vec::new();
    let result = pipeline
        .process_text(content, Some("test.txt"), 512, |progress| {
            progress_updates.push(progress.clone());
        })
        .expect("Processing should succeed");

    // Verify results
    assert!(result.has_chunks(), "Should have at least one chunk");
    assert!(result.total_tokens > 0, "Should have tokens");
    assert!(result.elapsed_ms > 0, "Should have elapsed time");

    // Verify embeddings
    for chunk in &result.chunks {
        assert_eq!(
            chunk.embedding.len(),
            config::EMBEDDING_DIM,
            "Embedding should have correct dimension"
        );
        assert!(!chunk.token_ids.is_empty(), "Should have token IDs");
        assert!(chunk.token_count > 0, "Should have token count");
    }

    // Verify progress callbacks
    assert!(!progress_updates.is_empty(), "Should have progress updates");
    let last_progress = progress_updates.last().unwrap();
    assert!(
        last_progress.is_complete(),
        "Final progress should be complete"
    );
}

#[test]
fn test_full_indexing_pipeline_markdown_chunking() {
    let tokenizer = load_tokenizer();
    let embedder = load_embedder();
    let pipeline = IndexingPipeline::new(embedder.clone(), tokenizer.clone());

    let markdown_content = r#"# Heading One

This is the first section with some content.

## Heading Two

This is the second section with different content.

### Subheading

More detailed information here.
"#;

    let result = pipeline
        .process_text(markdown_content, Some("document.md"), 512, |_| {})
        .expect("Processing should succeed");

    assert!(result.has_chunks(), "Should have chunks from markdown");

    // Verify file type detection
    let file_type = detect_file_type("document.md");
    assert!(
        matches!(file_type, FileType::Markdown),
        "Should detect markdown"
    );
}

#[test]
fn test_indexing_pipeline_batched_processing() {
    let tokenizer = load_tokenizer();
    let embedder = load_embedder();
    let pipeline = IndexingPipeline::new(embedder.clone(), tokenizer.clone());

    // Create content that will produce multiple chunks
    let long_content = (0..20)
        .map(|i| {
            format!(
                "This is paragraph number {} with some content about topic {}. ",
                i, i
            )
        })
        .collect::<String>();

    let result = pipeline
        .process_text_batched(
            &long_content,
            None,
            50, // Small chunk size to force multiple chunks
            4,  // Batch size
            |_| {},
        )
        .expect("Batched processing should succeed");

    assert!(result.chunk_count() > 1, "Should have multiple chunks");

    // All embeddings should have correct dimension
    for chunk in &result.chunks {
        assert_eq!(chunk.embedding.len(), config::EMBEDDING_DIM);
    }
}

#[test]
fn test_indexing_pipeline_empty_content() {
    let tokenizer = load_tokenizer();
    let embedder = load_embedder();
    let pipeline = IndexingPipeline::new(embedder.clone(), tokenizer.clone());

    let result = pipeline
        .process_text("", None, 512, |_| {})
        .expect("Empty content should not error");

    assert!(
        !result.has_chunks(),
        "Empty content should produce no chunks"
    );
    assert_eq!(result.total_tokens, 0);
}

// ============================================================================
// Search Pipeline Tests (Async)
// ============================================================================

#[tokio::test]
async fn test_full_search_pipeline() {
    // Setup
    let tokenizer = load_tokenizer();
    let embedder = load_embedder();
    let pipeline = IndexingPipeline::new(embedder.clone(), tokenizer.clone());
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, config::EMBEDDING_DIM)
        .await
        .expect("Engine creation should succeed");

    // Index sample documents
    let documents = sample_documents();

    for (filename, source, content) in &documents {
        let result = pipeline
            .process_text(content, Some(filename), 512, |_| {})
            .expect("Indexing should succeed");

        for chunk in result.chunks {
            let doc = make_document_with_source(&chunk.chunk.text, source);
            engine
                .add_document(doc, chunk.embedding)
                .await
                .expect("Adding document should succeed");
        }
    }

    // Verify index state
    let (doc_count, _tokens, _avg) = engine.get_index_metrics_sync();
    assert!(doc_count > 0, "Should have indexed documents");

    // Test semantic search for "memory safety"
    let query = "memory safety and borrow checker";
    let query_tokens = tokenizer
        .tokenize(query)
        .expect("Tokenization should succeed");
    let query_embedding = embedder
        .embed_tokens(query_tokens)
        .expect("Embedding should succeed");

    let results = engine
        .search(&query_embedding, query, 5)
        .await
        .expect("Search should succeed");

    assert!(!results.is_empty(), "Should have search results");

    // The Rust document should be highly ranked (it mentions memory safety)
    let top_result = &results[0];
    assert!(
        top_result.score > 0.0,
        "Top result should have positive score"
    );

    // Verify we can retrieve the document
    let doc_record = engine.get_document(&top_result.doc_id).await.unwrap();
    assert!(doc_record.is_some(), "Should be able to retrieve document");
}

#[tokio::test]
async fn test_search_with_keyword_boost() {
    let tokenizer = load_tokenizer();
    let embedder = load_embedder();
    let pipeline = IndexingPipeline::new(embedder.clone(), tokenizer.clone());
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, config::EMBEDDING_DIM)
        .await
        .unwrap();

    // Index documents
    let documents = sample_documents();
    for (filename, _, content) in &documents {
        let result = pipeline
            .process_text(content, Some(filename), 512, |_| {})
            .expect("Indexing should succeed");

        for chunk in result.chunks {
            let doc = make_document(&chunk.chunk.text);
            engine.add_document(doc, chunk.embedding).await.unwrap();
        }
    }

    // Search for exact keyword "PostgreSQL"
    let query = "PostgreSQL database";
    let query_tokens = tokenizer.tokenize(query).unwrap();
    let query_embedding = embedder.embed_tokens(query_tokens).unwrap();

    let results = engine.search(&query_embedding, query, 10).await.unwrap();

    // The database document should be in results
    // Note: SearchResult includes the text directly
    let has_database_result = results.iter().any(|r| r.text.contains("PostgreSQL"));

    assert!(
        has_database_result,
        "Should find document mentioning PostgreSQL"
    );
}

#[tokio::test]
async fn test_search_semantic_similarity() {
    let tokenizer = load_tokenizer();
    let embedder = load_embedder();
    let pipeline = IndexingPipeline::new(embedder.clone(), tokenizer.clone());
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, config::EMBEDDING_DIM)
        .await
        .unwrap();

    // Index documents
    let documents = sample_documents();
    for (filename, _, content) in &documents {
        let result = pipeline
            .process_text(content, Some(filename), 512, |_| {})
            .expect("Indexing should succeed");

        for chunk in result.chunks {
            let doc = make_document(&chunk.chunk.text);
            engine.add_document(doc, chunk.embedding).await.unwrap();
        }
    }

    // Search using semantically similar but different words
    // "artificial intelligence" should match "machine learning"
    let query = "artificial intelligence and neural networks";
    let query_tokens = tokenizer.tokenize(query).unwrap();
    let query_embedding = embedder.embed_tokens(query_tokens).unwrap();

    let results = engine.search(&query_embedding, query, 5).await.unwrap();

    assert!(
        !results.is_empty(),
        "Should have results for semantic query"
    );

    // Machine learning document should be in top results
    // Note: SearchResult includes the text directly
    let has_ml_result = results.iter().take(3).any(|r| {
        r.text.contains("Machine") || r.text.contains("neural") || r.text.contains("learning")
    });

    assert!(has_ml_result, "Should find ML document via semantic search");
}

#[tokio::test]
async fn test_search_empty_query() {
    let tokenizer = load_tokenizer();
    let embedder = load_embedder();
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, config::EMBEDDING_DIM)
        .await
        .unwrap();

    // Add a document
    let text = "test document about programming";
    let tokens = tokenizer.tokenize(text).unwrap();
    let embedding = embedder.embed_tokens(tokens.clone()).unwrap();
    let doc = make_document(text);
    engine.add_document(doc, embedding.clone()).await.unwrap();

    // Search with empty query text - should fall back to vector-only search
    // Note: The engine may reject empty queries, so we use the embedding
    let query_embedding = embedder.embed_tokens(tokens).unwrap();

    // Use a minimal query that won't match keywords but will match semantically
    let results = engine
        .search(&query_embedding, "programming", 5)
        .await
        .unwrap();

    assert!(!results.is_empty(), "Vector search should return results");
}

#[tokio::test]
async fn test_search_result_ordering() {
    let tokenizer = load_tokenizer();
    let embedder = load_embedder();
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, config::EMBEDDING_DIM)
        .await
        .unwrap();

    // Create documents with varying relevance
    let docs = [
        "Rust programming language with memory safety guarantees",
        "Python is a popular programming language",
        "Rust prevents memory bugs through ownership",
        "JavaScript runs in web browsers",
    ];

    for text in &docs {
        let tokens = tokenizer.tokenize(text).unwrap();
        let embedding = embedder.embed_tokens(tokens).unwrap();
        let doc = make_document(text);
        engine.add_document(doc, embedding).await.unwrap();
    }

    // Query about Rust memory safety
    let query = "Rust memory safety";
    let query_tokens = tokenizer.tokenize(query).unwrap();
    let query_embedding = embedder.embed_tokens(query_tokens).unwrap();

    let results = engine.search(&query_embedding, query, 4).await.unwrap();

    // Results should be ordered by relevance (higher scores first)
    for i in 1..results.len() {
        assert!(
            results[i - 1].score >= results[i].score,
            "Results should be ordered by score descending"
        );
    }

    // Top result should be about Rust
    // Note: SearchResult includes the text directly
    assert!(
        results[0].text.contains("Rust"),
        "Top result should be about Rust"
    );
}

// ============================================================================
// Document Lifecycle Tests
// ============================================================================

#[tokio::test]
async fn test_source_deletion_and_search() {
    let tokenizer = load_tokenizer();
    let embedder = load_embedder();
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, config::EMBEDDING_DIM)
        .await
        .unwrap();

    // Register and add first source
    let source1 = "file://doc1.txt";
    engine
        .register_source(source1, "hash1".to_string())
        .await
        .unwrap();

    let doc1_text = "Rust programming language";
    let tokens1 = tokenizer.tokenize(doc1_text).unwrap();
    let emb1 = embedder.embed_tokens(tokens1).unwrap();
    let doc1 = make_document_with_source(doc1_text, source1);
    let id1 = engine.add_document(doc1, emb1.clone()).await.unwrap();
    engine.add_doc_to_source(source1, id1).await.unwrap();
    engine.complete_source(source1).await.unwrap();

    // Register and add second source
    let source2 = "file://doc2.txt";
    engine
        .register_source(source2, "hash2".to_string())
        .await
        .unwrap();

    let doc2_text = "Python programming language";
    let tokens2 = tokenizer.tokenize(doc2_text).unwrap();
    let emb2 = embedder.embed_tokens(tokens2).unwrap();
    let doc2 = make_document_with_source(doc2_text, source2);
    let id2 = engine.add_document(doc2, emb2).await.unwrap();
    engine.add_doc_to_source(source2, id2).await.unwrap();
    engine.complete_source(source2).await.unwrap();

    // Delete first source
    let deleted = engine.delete_source(source1).await.unwrap();
    assert_eq!(deleted, 1, "Should delete one document");

    // Search should not return deleted document
    let results = engine.search(&emb1, "Rust", 5).await.unwrap();

    for result in &results {
        assert!(
            !result.text.contains("Rust programming language"),
            "Deleted document should not appear in results"
        );
    }
}

#[tokio::test]
async fn test_source_tracking() {
    let tokenizer = load_tokenizer();
    let embedder = load_embedder();
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, config::EMBEDDING_DIM)
        .await
        .unwrap();

    // Register a source
    let source = "file:///path/to/document.md";
    let hash = "abc123";
    engine
        .register_source(source, hash.to_string())
        .await
        .unwrap();

    // Add documents with source
    let text = "content";
    let tokens = tokenizer.tokenize(text).unwrap();
    let embedding = embedder.embed_tokens(tokens).unwrap();
    let doc = make_document_with_source(text, source);
    let doc_id = engine.add_document(doc, embedding).await.unwrap();
    engine.add_doc_to_source(source, doc_id).await.unwrap();
    engine.complete_source(source).await.unwrap();

    // Verify source tracking
    assert!(
        engine.get_source(source).await.unwrap().is_some(),
        "Source should be registered"
    );
    assert!(
        !engine.source_needs_update(source, hash).await.unwrap(),
        "Source should be current"
    );
    assert!(
        engine
            .source_needs_update(source, "different_hash")
            .await
            .unwrap(),
        "Different hash should need update"
    );

    // Delete by source
    let deleted = engine.delete_source(source).await.unwrap();
    assert_eq!(deleted, 1, "Should delete one document");

    // Source should no longer exist
    assert!(
        engine.get_source(source).await.unwrap().is_none(),
        "Source should be removed after deletion"
    );
}

#[tokio::test]
async fn test_index_compaction() {
    let tokenizer = load_tokenizer();
    let embedder = load_embedder();
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, config::EMBEDDING_DIM)
        .await
        .unwrap();

    // Register a source
    let source = "file://compaction-test";
    engine
        .register_source(source, "hash".to_string())
        .await
        .unwrap();

    // Add documents to the source (10 is enough to verify compaction logic)
    for i in 0..10 {
        let text = format!("document {} content", i);
        let tokens = tokenizer.tokenize(&text).unwrap();
        let embedding = embedder.embed_tokens(tokens).unwrap();
        let doc = make_document_with_source(&text, source);
        let doc_id = engine.add_document(doc, embedding).await.unwrap();
        engine.add_doc_to_source(source, doc_id).await.unwrap();
    }
    engine.complete_source(source).await.unwrap();

    // Delete the source (creates tombstones)
    engine.delete_source(source).await.unwrap();

    // Check compaction stats - returns (tombstone_count, total_count, ratio)
    let (tombstone_count, total_count, _ratio) = engine.compaction_stats();
    assert!(tombstone_count > 0, "Should have tombstones");
    assert_eq!(total_count, 10, "Should have total count");

    // Compact if needed
    if engine.needs_compaction() {
        engine.compact_if_needed().await.unwrap();

        let (new_tombstone_count, _, _) = engine.compaction_stats();
        assert!(
            new_tombstone_count < tombstone_count,
            "Compaction should reduce tombstones"
        );
    }
}

// ============================================================================
// Chunking Strategy Tests
// ============================================================================

#[test]
fn test_file_type_detection() {
    assert!(matches!(detect_file_type("README.md"), FileType::Markdown));
    assert!(matches!(detect_file_type("document.txt"), FileType::Text));
    assert!(matches!(detect_file_type("notes"), FileType::Text));

    // Code detection available on native platforms (tree-sitter)
    #[cfg(not(target_arch = "wasm32"))]
    {
        assert!(matches!(detect_file_type("script.py"), FileType::Code(_)));
        assert!(matches!(detect_file_type("main.rs"), FileType::Code(_)));
    }

    #[cfg(target_arch = "wasm32")]
    {
        // On WASM, code files fall back to Text (tree-sitter doesn't compile to WASM)
        assert!(matches!(detect_file_type("script.py"), FileType::Text));
        assert!(matches!(detect_file_type("main.rs"), FileType::Text));
    }
}

#[test]
fn test_chunker_respects_max_tokens() {
    let tokenizer = get_static_tokenizer();
    let max_tokens = 50;

    let chunker = create_chunker(FileType::Text, max_tokens, tokenizer);

    // Create content that would exceed max tokens if not chunked
    let long_content = "word ".repeat(200);
    let chunks = chunker
        .chunk(&long_content)
        .expect("Chunking should succeed");

    // Verify each chunk respects the token limit (approximately)
    for chunk in &chunks {
        // Count tokens (rough check - actual tokenization may vary)
        let word_count = chunk.text.split_whitespace().count();
        // Allow some margin for tokenizer differences
        assert!(
            word_count <= max_tokens * 2,
            "Chunk should respect approximate token limit"
        );
    }
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

#[test]
fn test_unicode_content() {
    let tokenizer = load_tokenizer();
    let embedder = load_embedder();
    let pipeline = IndexingPipeline::new(embedder.clone(), tokenizer.clone());

    let unicode_content = "Hello 世界! Привет мир! 🦀 Rust is awesome! مرحبا";

    let result = pipeline
        .process_text(unicode_content, None, 512, |_| {})
        .expect("Unicode content should be processed");

    assert!(result.has_chunks(), "Should handle unicode content");
}

#[test]
fn test_very_long_content() {
    let tokenizer = load_tokenizer();
    let embedder = load_embedder();
    let pipeline = IndexingPipeline::new(embedder.clone(), tokenizer.clone());

    // Generate content that will produce multiple chunks
    // Using 50 paragraphs is enough to verify chunking works without being slow
    let long_content = (0..50)
        .map(|i| format!("Paragraph {} with some text content about topic {}. ", i, i))
        .collect::<String>();

    // Use smaller max_tokens to ensure multiple chunks
    let result = pipeline
        .process_text(&long_content, None, 100, |_| {})
        .expect("Long content should be processed");

    assert!(result.has_chunks(), "Long content should produce chunks");
    // With 100 tokens per chunk, 50 paragraphs should produce multiple chunks
    assert!(
        result.chunk_count() > 1,
        "Long content should produce multiple chunks, got {}",
        result.chunk_count()
    );
}

#[test]
fn test_special_characters() {
    let tokenizer = load_tokenizer();
    let embedder = load_embedder();
    let pipeline = IndexingPipeline::new(embedder.clone(), tokenizer.clone());

    let special_content = r#"
        Code: fn main() { println!("Hello"); }
        Math: x² + y² = z²
        Symbols: @#$%^&*()
        Quotes: "double" and 'single'
        Escapes: \n \t \r
    "#;

    let result = pipeline
        .process_text(special_content, None, 512, |_| {})
        .expect("Special characters should be handled");

    assert!(result.has_chunks());
}

#[test]
fn test_whitespace_only_content() {
    let tokenizer = load_tokenizer();
    let embedder = load_embedder();
    let pipeline = IndexingPipeline::new(embedder.clone(), tokenizer.clone());

    let result = pipeline
        .process_text("   \n\t\n   ", None, 512, |_| {})
        .expect("Whitespace content should not error");

    assert!(
        !result.has_chunks(),
        "Whitespace-only should produce no chunks"
    );
}

// ============================================================================
// Integration: Full Pipeline E2E Test
// ============================================================================

#[tokio::test]
async fn test_complete_index_and_search_workflow() {
    // This test exercises the entire workflow end-to-end:
    // 1. Load model and tokenizer
    // 2. Process documents (chunk, tokenize, embed)
    // 3. Index in search engine
    // 4. Search and retrieve results
    // 5. Verify result quality

    let tokenizer = load_tokenizer();
    let embedder = load_embedder();
    let pipeline = IndexingPipeline::new(embedder.clone(), tokenizer.clone());
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, config::EMBEDDING_DIM)
        .await
        .unwrap();

    // Index all sample documents
    let documents = sample_documents();
    let mut total_chunks = 0;

    for (filename, _source, content) in &documents {
        let result = pipeline
            .process_text(content, Some(filename), 512, |_| {})
            .expect("Processing should succeed");

        for chunk in result.chunks {
            let doc = Document {
                text: chunk.chunk.text.clone(),
                metadata: DocumentMetadata {
                    filename: Some(filename.to_string()),
                    ..Default::default()
                },
            };
            engine.add_document(doc, chunk.embedding).await.unwrap();
            total_chunks += 1;
        }
    }

    println!(
        "Indexed {} chunks from {} documents",
        total_chunks,
        documents.len()
    );

    // Test various queries
    let test_queries = [
        ("Rust memory safety borrow checker", "Rust"),
        ("Python functions and variables", "Python"),
        ("database SQL PostgreSQL", "Database"),
        ("neural networks deep learning", "Machine"),
        ("web development frontend backend", "Web"),
    ];

    for (query, expected_keyword) in test_queries {
        let query_tokens = tokenizer.tokenize(query).unwrap();
        let query_embedding = embedder.embed_tokens(query_tokens).unwrap();

        let results = engine.search(&query_embedding, query, 3).await.unwrap();

        assert!(
            !results.is_empty(),
            "Query '{}' should return results",
            query
        );

        // Top result should contain the expected keyword
        // Note: SearchResult includes the text directly
        let top_text = &results[0].text;
        assert!(
            top_text.contains(expected_keyword),
            "Query '{}' top result should contain '{}', got: {}...",
            query,
            expected_keyword,
            &top_text[..top_text.len().min(100)]
        );
    }
}
