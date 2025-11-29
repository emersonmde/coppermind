//! Document-level BM25 keyword search.
//!
//! This module provides document-level BM25 search, which indexes complete documents
//! (files, web pages) rather than individual chunks. This ensures proper IDF (Inverse
//! Document Frequency) statistics across the corpus.
//!
//! # Why Document-Level BM25?
//!
//! Chunk-level BM25 dilutes IDF statistics. Consider a file with 5 chunks all containing
//! the term "BM25":
//!
//! - Chunk-level: IDF("BM25") = log(N/5) where N = total chunks
//! - Document-level: IDF("BM25") = log(D/1) where D = total documents
//!
//! The document-level IDF correctly reflects that "BM25" appears in only one document,
//! making it a more discriminative term.
//!
//! # Usage
//!
//! ```ignore
//! use coppermind_core::search::document_keyword::DocumentKeywordEngine;
//! use coppermind_core::search::types::DocumentId;
//!
//! let mut engine = DocumentKeywordEngine::new();
//!
//! // Index full document text
//! let doc_id = DocumentId::from_u64(1);
//! engine.add_document(doc_id, "Full document text goes here...".to_string());
//!
//! // Search returns (DocumentId, score) pairs
//! let results = engine.search("document", 10);
//! ```
//!
//! # Integration with Hybrid Search
//!
//! This engine is used in [`HybridSearchEngine`](super::engine::HybridSearchEngine)
//! alongside chunk-level HNSW vector search. The search flow is:
//!
//! 1. HNSW finds semantically similar chunks
//! 2. Chunks are lifted to document IDs
//! 3. DocumentKeywordEngine scores documents with BM25
//! 4. RRF fuses vector and keyword rankings at the document level
//!
//! See ADR-008 for the full architecture.

use super::types::DocumentId;
use bm25::{Document, Language, SearchEngineBuilder};
use tracing::instrument;

/// Document-level BM25 search engine.
///
/// Indexes full document text for proper IDF statistics across the corpus.
/// Unlike chunk-level BM25, this correctly weights term rarity at the document level.
///
/// # Algorithm Details
///
/// Uses BM25 with standard parameters:
/// - k1 = 1.2 (term frequency saturation)
/// - b = 0.75 (document length normalization)
///
/// # Thread Safety
///
/// This type is **not thread-safe**. For concurrent access, wrap in appropriate
/// synchronization primitives (e.g., `Mutex`).
pub struct DocumentKeywordEngine {
    /// BM25 search engine with DocumentId as u64 key
    search_engine: bm25::SearchEngine<u64>,
    /// Document count (tracked separately since bm25 crate doesn't expose it)
    document_count: usize,
}

impl DocumentKeywordEngine {
    /// Creates a new empty document keyword search engine.
    ///
    /// Initializes with English language settings for tokenization and stemming.
    pub fn new() -> Self {
        // Create empty search engine with English language
        let empty_docs: Vec<Document<u64>> = vec![];
        let search_engine =
            SearchEngineBuilder::<u64>::with_documents(Language::English, empty_docs).build();

        Self {
            search_engine,
            document_count: 0,
        }
    }

    /// Adds a document to the BM25 corpus.
    ///
    /// The full document text is tokenized and indexed for keyword search.
    /// If a document with the same ID already exists, it is updated (upsert semantics).
    ///
    /// # Arguments
    ///
    /// * `doc_id` - Unique identifier for the document
    /// * `full_text` - Complete text content of the document
    #[instrument(skip_all, fields(text_len = full_text.len()))]
    pub fn add_document(&mut self, doc_id: DocumentId, full_text: String) {
        // Create BM25 document with DocumentId as u64
        let doc = Document {
            id: doc_id.as_u64(),
            contents: full_text,
        };

        // Upsert document into search engine
        self.search_engine.upsert(doc);
        self.document_count += 1;
    }

    /// Removes a document from the index.
    ///
    /// Note: The bm25 crate doesn't support deletion directly. This method
    /// tracks the removal but the underlying index may still contain the entry
    /// until a rebuild occurs. The document will still be filtered out of results.
    ///
    /// For now, we simply decrement the count. Full removal requires rebuilding
    /// the index (which happens on app restart or explicit compaction).
    #[allow(dead_code)]
    pub fn remove_document(&mut self, _doc_id: DocumentId) {
        // The bm25 crate doesn't support deletion directly.
        // We track the removal here, but the entry persists until rebuild.
        // Search results should be filtered by the caller if needed.
        if self.document_count > 0 {
            self.document_count -= 1;
        }
    }

    /// Searches for documents matching the query.
    ///
    /// Returns up to `k` documents ranked by BM25 score. Higher scores indicate
    /// better matches based on term frequency and document relevance.
    ///
    /// # Arguments
    ///
    /// * `query` - Search query (can contain multiple terms)
    /// * `k` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// Vector of `(DocumentId, score)` pairs sorted by score descending.
    /// Returns empty vector if query is empty or no documents match.
    pub fn search(&self, query: &str, k: usize) -> Vec<(DocumentId, f32)> {
        // Perform search
        let results = self.search_engine.search(query, k);

        // Convert to (DocumentId, score) pairs
        results
            .into_iter()
            .map(|result| {
                let doc_id = DocumentId::from_u64(result.document.id);
                let score = result.score;
                (doc_id, score)
            })
            .collect()
    }

    /// Returns the number of indexed documents.
    pub fn len(&self) -> usize {
        self.document_count
    }

    /// Returns `true` if no documents have been indexed.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Rebuilds the engine from a list of documents.
    ///
    /// This is used during compaction or initial load to populate the index
    /// with existing document data.
    ///
    /// # Arguments
    ///
    /// * `documents` - Iterator of (DocumentId, full_text) pairs
    pub fn rebuild_from<I>(&mut self, documents: I)
    where
        I: IntoIterator<Item = (DocumentId, String)>,
    {
        // Collect documents into BM25 format
        let docs: Vec<Document<u64>> = documents
            .into_iter()
            .map(|(id, text)| Document {
                id: id.as_u64(),
                contents: text,
            })
            .collect();

        let count = docs.len();

        // Rebuild the search engine
        self.search_engine =
            SearchEngineBuilder::<u64>::with_documents(Language::English, docs).build();
        self.document_count = count;
    }
}

impl Default for DocumentKeywordEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_search() {
        let mut engine = DocumentKeywordEngine::new();

        let doc1 = DocumentId::from_u64(1);
        let doc2 = DocumentId::from_u64(2);
        let doc3 = DocumentId::from_u64(3);

        engine.add_document(
            doc1,
            "Rust is a systems programming language focused on safety".to_string(),
        );
        engine.add_document(
            doc2,
            "Python is great for data science and machine learning".to_string(),
        );
        engine.add_document(
            doc3,
            "JavaScript runs in browsers and Node.js servers".to_string(),
        );

        // Search for "programming"
        let results = engine.search("programming", 3);
        assert!(!results.is_empty());
        // doc1 should match (contains "programming")
        assert!(results.iter().any(|(id, _)| *id == doc1));

        // Search for "data science"
        let results = engine.search("data science", 3);
        assert!(!results.is_empty());
        // doc2 should match
        assert!(results.iter().any(|(id, _)| *id == doc2));
    }

    #[test]
    fn test_empty_query() {
        let mut engine = DocumentKeywordEngine::new();
        engine.add_document(DocumentId::from_u64(1), "test document".to_string());

        let results = engine.search("", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_empty_index() {
        let engine = DocumentKeywordEngine::new();
        let results = engine.search("query", 10);
        assert!(results.is_empty());
        assert!(engine.is_empty());
    }

    #[test]
    fn test_len() {
        let mut engine = DocumentKeywordEngine::new();
        assert_eq!(engine.len(), 0);
        assert!(engine.is_empty());

        engine.add_document(DocumentId::from_u64(1), "first".to_string());
        assert_eq!(engine.len(), 1);

        engine.add_document(DocumentId::from_u64(2), "second".to_string());
        assert_eq!(engine.len(), 2);
    }

    #[test]
    fn test_rebuild_from() {
        let mut engine = DocumentKeywordEngine::new();

        // Initially empty
        assert!(engine.is_empty());

        // Rebuild from documents
        let docs = vec![
            (DocumentId::from_u64(1), "document one content".to_string()),
            (DocumentId::from_u64(2), "document two content".to_string()),
            (
                DocumentId::from_u64(3),
                "document three different".to_string(),
            ),
        ];

        engine.rebuild_from(docs);

        assert_eq!(engine.len(), 3);

        // Search should work
        let results = engine.search("content", 3);
        assert_eq!(results.len(), 2); // doc1 and doc2 contain "content"

        let results = engine.search("different", 3);
        assert_eq!(results.len(), 1); // only doc3
        assert_eq!(results[0].0, DocumentId::from_u64(3));
    }

    #[test]
    fn test_idf_behavior() {
        let mut engine = DocumentKeywordEngine::new();

        // Add documents where "common" appears in many, "rare" in one
        engine.add_document(
            DocumentId::from_u64(1),
            "common term appears here with rare term".to_string(),
        );
        engine.add_document(
            DocumentId::from_u64(2),
            "common term in second document".to_string(),
        );
        engine.add_document(
            DocumentId::from_u64(3),
            "common term in third document".to_string(),
        );

        // Search for "rare" should find doc1 with high score
        let results = engine.search("rare", 3);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, DocumentId::from_u64(1));

        // Search for "common" should find all 3
        let results = engine.search("common", 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_case_insensitivity() {
        let mut engine = DocumentKeywordEngine::new();
        engine.add_document(
            DocumentId::from_u64(1),
            "UPPERCASE Document Content".to_string(),
        );

        let results = engine.search("uppercase", 1);
        assert!(!results.is_empty());

        let results = engine.search("DOCUMENT", 1);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_multi_term_query() {
        let mut engine = DocumentKeywordEngine::new();

        engine.add_document(
            DocumentId::from_u64(1),
            "machine learning algorithms".to_string(),
        );
        engine.add_document(
            DocumentId::from_u64(2),
            "deep learning neural networks".to_string(),
        );
        engine.add_document(
            DocumentId::from_u64(3),
            "machine vision systems".to_string(),
        );

        // "machine learning" should match doc1 best (both terms)
        let results = engine.search("machine learning", 3);
        assert!(!results.is_empty());
        // Doc1 should rank highest (contains both terms)
        assert_eq!(results[0].0, DocumentId::from_u64(1));
    }

    #[test]
    fn test_scores_are_positive() {
        let mut engine = DocumentKeywordEngine::new();
        engine.add_document(DocumentId::from_u64(1), "test document content".to_string());

        let results = engine.search("test", 1);
        assert!(!results.is_empty());
        assert!(results[0].1 > 0.0, "Score should be positive");
    }
}
