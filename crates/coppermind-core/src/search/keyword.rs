//! BM25 keyword search for exact term matching.
//!
//! This module wraps the [`bm25`](https://crates.io/crates/bm25) crate to provide
//! keyword-based search capabilities. BM25 (Best Matching 25) is a ranking function
//! used by search engines to score documents based on query term frequency.
//!
//! # Algorithm
//!
//! BM25 scores documents based on:
//! - **Term Frequency (TF)**: How often query terms appear in the document
//! - **Inverse Document Frequency (IDF)**: Rarity of terms across the corpus
//! - **Document Length**: Normalized to avoid bias toward longer documents
//!
//! # Usage
//!
//! ```ignore
//! use coppermind_core::search::keyword::KeywordSearchEngine;
//!
//! let mut engine = KeywordSearchEngine::new();
//! engine.add_document(DocId::from_u64(1), "rust programming language".to_string());
//! engine.add_document(DocId::from_u64(2), "python scripting language".to_string());
//!
//! // Search returns (DocId, score) pairs
//! let results = engine.search("rust", 10);
//! ```
//!
//! # Integration with Hybrid Search
//!
//! This engine is used alongside [`VectorSearchEngine`](super::vector::VectorSearchEngine)
//! in the [`HybridSearchEngine`](super::engine::HybridSearchEngine). Results from both
//! are combined using [Reciprocal Rank Fusion](super::fusion::reciprocal_rank_fusion).

use super::types::ChunkId;
use bm25::{Document, Language, SearchEngineBuilder};
use tracing::instrument;

/// BM25-based keyword search engine.
///
/// Provides full-text search using the BM25 ranking algorithm. Documents are
/// indexed by their text content and can be searched using natural language queries.
///
/// # Features
///
/// - **Case-insensitive**: Queries and documents are normalized
/// - **Multi-term queries**: Multiple words are scored independently and combined
/// - **Language-aware**: Uses English tokenization and stemming
///
/// # Thread Safety
///
/// This type is **not thread-safe**. For concurrent access, wrap in appropriate
/// synchronization primitives (e.g., `Mutex`).
pub struct KeywordSearchEngine {
    /// BM25 search engine
    search_engine: bm25::SearchEngine<u64>,
    /// Document count (tracked separately since bm25 crate doesn't expose it)
    document_count: usize,
}

impl KeywordSearchEngine {
    /// Creates a new empty keyword search engine.
    ///
    /// Initializes with English language settings for tokenization and stemming.
    pub fn new() -> Self {
        // Create empty search engine with English language
        // Using with_documents to get proper u64 type
        let empty_docs: Vec<Document<u64>> = vec![];
        let search_engine =
            SearchEngineBuilder::<u64>::with_documents(Language::English, empty_docs).build();

        Self {
            search_engine,
            document_count: 0,
        }
    }

    /// Adds a chunk to the BM25 corpus.
    ///
    /// The chunk text is tokenized and indexed for keyword search.
    /// If a chunk with the same ID already exists, it is updated (upsert semantics).
    ///
    /// # Arguments
    ///
    /// * `chunk_id` - Unique identifier for the chunk
    /// * `text` - Full text content to index
    #[instrument(skip_all, fields(text_len = text.len()))]
    pub fn add_chunk(&mut self, chunk_id: ChunkId, text: String) {
        // Create BM25 document with ChunkId as u64
        let doc = Document {
            id: chunk_id.as_u64(),
            contents: text,
        };

        // Upsert document into search engine
        self.search_engine.upsert(doc);
        self.document_count += 1;
    }

    /// Searches for chunks matching the query.
    ///
    /// Returns up to `k` chunks ranked by BM25 score. Higher scores indicate
    /// better matches based on term frequency and chunk relevance.
    ///
    /// # Arguments
    ///
    /// * `query` - Search query (can contain multiple terms)
    /// * `k` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// Vector of `(ChunkId, score)` pairs sorted by score descending.
    /// Returns empty vector if query is empty or no chunks match.
    pub fn search(&self, query: &str, k: usize) -> Vec<(ChunkId, f32)> {
        // Perform search
        let results = self.search_engine.search(query, k);

        // Convert to (ChunkId, score) pairs
        results
            .into_iter()
            .map(|result| {
                let chunk_id = ChunkId::from_u64(result.document.id);
                let score = result.score;
                (chunk_id, score)
            })
            .collect()
    }

    /// Returns the number of indexed chunks.
    #[allow(dead_code)] // Public API
    pub fn len(&self) -> usize {
        self.document_count
    }

    /// Returns `true` if no chunks have been indexed.
    #[allow(dead_code)] // Public API
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for KeywordSearchEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyword_search() {
        let mut engine = KeywordSearchEngine::new();

        let chunk1 = ChunkId::from_u64(1);
        let chunk2 = ChunkId::from_u64(2);
        let chunk3 = ChunkId::from_u64(3);

        engine.add_chunk(
            chunk1,
            "the quick brown fox jumps over the lazy dog".to_string(),
        );
        engine.add_chunk(chunk2, "the lazy cat sleeps all day".to_string());
        engine.add_chunk(chunk3, "quick brown rabbits hop in the garden".to_string());

        // Search for "quick brown"
        let results = engine.search("quick brown", 2);

        assert!(results.len() <= 2);
        // Should match chunk1 and chunk3 which contain "quick" and "brown"
        if !results.is_empty() {
            assert!(
                results.iter().any(|(id, _)| *id == chunk1)
                    || results.iter().any(|(id, _)| *id == chunk3)
            );
        }
    }

    #[test]
    fn test_empty_query_returns_empty() {
        let mut engine = KeywordSearchEngine::new();

        let chunk1 = ChunkId::from_u64(1);
        engine.add_chunk(chunk1, "test chunk".to_string());

        // Empty query should return no results
        let results = engine.search("", 10);

        assert!(results.is_empty());
    }

    #[test]
    fn test_empty_index_returns_empty() {
        let engine = KeywordSearchEngine::new();

        // Search empty index
        let results = engine.search("query", 10);

        assert!(results.is_empty());
    }

    #[test]
    fn test_multi_term_query() {
        let mut engine = KeywordSearchEngine::new();

        let chunk1 = ChunkId::from_u64(1);
        let chunk2 = ChunkId::from_u64(2);
        let chunk3 = ChunkId::from_u64(3);

        engine.add_chunk(chunk1, "machine learning algorithms".to_string());
        engine.add_chunk(chunk2, "deep learning neural networks".to_string());
        engine.add_chunk(chunk3, "machine vision systems".to_string());

        // Multi-term query
        let results = engine.search("machine learning", 3);

        // Should return chunks containing either "machine" or "learning"
        assert!(!results.is_empty());
        // chunk1 should rank high (has both terms)
        // chunk2 has "learning", chunk3 has "machine"
    }

    #[test]
    fn test_case_insensitivity() {
        let mut engine = KeywordSearchEngine::new();

        let chunk1 = ChunkId::from_u64(1);
        engine.add_chunk(chunk1, "Rust Programming Language".to_string());

        // Lowercase query should match uppercase chunk
        let results_lower = engine.search("rust", 1);
        assert!(!results_lower.is_empty());

        // Uppercase query should also work
        let results_upper = engine.search("RUST", 1);
        assert!(!results_upper.is_empty());

        // Mixed case
        let results_mixed = engine.search("RuSt", 1);
        assert!(!results_mixed.is_empty());
    }

    #[test]
    fn test_special_characters_handling() {
        let mut engine = KeywordSearchEngine::new();

        let chunk1 = ChunkId::from_u64(1);
        engine.add_chunk(chunk1, "Hello, world! This is a test: 123 @#$%".to_string());

        // Query with word should work
        let results = engine.search("hello", 1);
        assert!(!results.is_empty());

        // Query with punctuation
        let results2 = engine.search("test", 1);
        assert!(!results2.is_empty());
    }

    #[test]
    fn test_bm25_scoring() {
        let mut engine = KeywordSearchEngine::new();

        let chunk1 = ChunkId::from_u64(1);
        let chunk2 = ChunkId::from_u64(2);
        let chunk3 = ChunkId::from_u64(3);

        // chunk1: "rust" appears once
        engine.add_chunk(chunk1, "rust programming".to_string());

        // chunk2: "rust" appears three times
        engine.add_chunk(
            chunk2,
            "rust rust rust is a programming language".to_string(),
        );

        // chunk3: no "rust"
        engine.add_chunk(chunk3, "python programming".to_string());

        let results = engine.search("rust", 3);

        // Results should include chunk1 and chunk2
        assert!(results.iter().any(|(id, _)| *id == chunk1));
        assert!(results.iter().any(|(id, _)| *id == chunk2));

        // chunk2 should have higher score (more occurrences)
        let chunk1_score = results.iter().find(|(id, _)| *id == chunk1).map(|(_, s)| s);
        let chunk2_score = results.iter().find(|(id, _)| *id == chunk2).map(|(_, s)| s);

        if let (Some(&score1), Some(&score2)) = (chunk1_score, chunk2_score) {
            assert!(
                score2 > score1,
                "chunk2 (more occurrences) should score higher than chunk1"
            );
        }
    }

    #[test]
    fn test_search_returns_top_k() {
        let mut engine = KeywordSearchEngine::new();

        // Add 10 chunks
        for i in 0..10 {
            let chunk_id = ChunkId::from_u64(i);
            engine.add_chunk(chunk_id, format!("chunk number {}", i));
        }

        // Request top 3
        let results = engine.search("chunk", 3);

        // Should return at most 3 results
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_len_returns_correct_count() {
        let mut engine = KeywordSearchEngine::new();

        // Empty engine
        assert_eq!(engine.len(), 0);
        assert!(engine.is_empty());

        // Add chunks and verify count increases
        engine.add_chunk(ChunkId::from_u64(1), "first chunk".to_string());
        assert_eq!(engine.len(), 1);
        assert!(!engine.is_empty());

        engine.add_chunk(ChunkId::from_u64(2), "second chunk".to_string());
        assert_eq!(engine.len(), 2);

        engine.add_chunk(ChunkId::from_u64(3), "third chunk".to_string());
        assert_eq!(engine.len(), 3);
    }

    #[test]
    fn test_scores_are_non_negative() {
        let mut engine = KeywordSearchEngine::new();

        let chunk1 = ChunkId::from_u64(1);
        engine.add_chunk(chunk1, "test chunk".to_string());

        let results = engine.search("test", 1);

        // All scores should be non-negative
        for (_, score) in &results {
            assert!(*score >= 0.0, "Score should be non-negative, got {}", score);
        }
    }
}
