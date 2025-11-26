// BM25 keyword search integration

use super::types::DocId;
use bm25::{Document, Language, SearchEngineBuilder};
use tracing::instrument;

/// Keyword search engine using BM25 algorithm
pub struct KeywordSearchEngine {
    /// BM25 search engine
    search_engine: bm25::SearchEngine<u64>,
    /// Document count (tracked separately since bm25 crate doesn't expose it)
    document_count: usize,
}

impl KeywordSearchEngine {
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

    /// Add a document to the BM25 corpus
    #[instrument(skip_all, fields(text_len = text.len()))]
    pub fn add_document(&mut self, doc_id: DocId, text: String) {
        // Create BM25 document with DocId as u64
        let doc = Document {
            id: doc_id.as_u64(),
            contents: text,
        };

        // Upsert document into search engine
        self.search_engine.upsert(doc);
        self.document_count += 1;
    }

    /// Search using BM25 keyword matching
    pub fn search(&self, query: &str, k: usize) -> Vec<(DocId, f32)> {
        // Perform search
        let results = self.search_engine.search(query, k);

        // Convert to (DocId, score) pairs
        results
            .into_iter()
            .map(|result| {
                let doc_id = DocId::from_u64(result.document.id);
                let score = result.score;
                (doc_id, score)
            })
            .collect()
    }

    /// Get number of indexed documents
    #[allow(dead_code)] // Public API
    pub fn len(&self) -> usize {
        self.document_count
    }

    /// Check if index is empty
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

        let doc1 = DocId::from_u64(1);
        let doc2 = DocId::from_u64(2);
        let doc3 = DocId::from_u64(3);

        engine.add_document(
            doc1,
            "the quick brown fox jumps over the lazy dog".to_string(),
        );
        engine.add_document(doc2, "the lazy cat sleeps all day".to_string());
        engine.add_document(doc3, "quick brown rabbits hop in the garden".to_string());

        // Search for "quick brown"
        let results = engine.search("quick brown", 2);

        assert!(results.len() <= 2);
        // Should match doc1 and doc3 which contain "quick" and "brown"
        if !results.is_empty() {
            assert!(
                results.iter().any(|(id, _)| *id == doc1)
                    || results.iter().any(|(id, _)| *id == doc3)
            );
        }
    }

    #[test]
    fn test_empty_query_returns_empty() {
        let mut engine = KeywordSearchEngine::new();

        let doc1 = DocId::from_u64(1);
        engine.add_document(doc1, "test document".to_string());

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

        let doc1 = DocId::from_u64(1);
        let doc2 = DocId::from_u64(2);
        let doc3 = DocId::from_u64(3);

        engine.add_document(doc1, "machine learning algorithms".to_string());
        engine.add_document(doc2, "deep learning neural networks".to_string());
        engine.add_document(doc3, "machine vision systems".to_string());

        // Multi-term query
        let results = engine.search("machine learning", 3);

        // Should return documents containing either "machine" or "learning"
        assert!(!results.is_empty());
        // doc1 should rank high (has both terms)
        // doc2 has "learning", doc3 has "machine"
    }

    #[test]
    fn test_case_insensitivity() {
        let mut engine = KeywordSearchEngine::new();

        let doc1 = DocId::from_u64(1);
        engine.add_document(doc1, "Rust Programming Language".to_string());

        // Lowercase query should match uppercase document
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

        let doc1 = DocId::from_u64(1);
        engine.add_document(doc1, "Hello, world! This is a test: 123 @#$%".to_string());

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

        let doc1 = DocId::from_u64(1);
        let doc2 = DocId::from_u64(2);
        let doc3 = DocId::from_u64(3);

        // doc1: "rust" appears once
        engine.add_document(doc1, "rust programming".to_string());

        // doc2: "rust" appears three times
        engine.add_document(doc2, "rust rust rust is a programming language".to_string());

        // doc3: no "rust"
        engine.add_document(doc3, "python programming".to_string());

        let results = engine.search("rust", 3);

        // Results should include doc1 and doc2
        assert!(results.iter().any(|(id, _)| *id == doc1));
        assert!(results.iter().any(|(id, _)| *id == doc2));

        // doc2 should have higher score (more occurrences)
        let doc1_score = results.iter().find(|(id, _)| *id == doc1).map(|(_, s)| s);
        let doc2_score = results.iter().find(|(id, _)| *id == doc2).map(|(_, s)| s);

        if let (Some(&score1), Some(&score2)) = (doc1_score, doc2_score) {
            assert!(
                score2 > score1,
                "doc2 (more occurrences) should score higher than doc1"
            );
        }
    }

    #[test]
    fn test_search_returns_top_k() {
        let mut engine = KeywordSearchEngine::new();

        // Add 10 documents
        for i in 0..10 {
            let doc_id = DocId::from_u64(i);
            engine.add_document(doc_id, format!("document number {}", i));
        }

        // Request top 3
        let results = engine.search("document", 3);

        // Should return at most 3 results
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_len_returns_correct_count() {
        let mut engine = KeywordSearchEngine::new();

        // Empty engine
        assert_eq!(engine.len(), 0);
        assert!(engine.is_empty());

        // Add documents and verify count increases
        engine.add_document(DocId::from_u64(1), "first document".to_string());
        assert_eq!(engine.len(), 1);
        assert!(!engine.is_empty());

        engine.add_document(DocId::from_u64(2), "second document".to_string());
        assert_eq!(engine.len(), 2);

        engine.add_document(DocId::from_u64(3), "third document".to_string());
        assert_eq!(engine.len(), 3);
    }

    #[test]
    fn test_scores_are_non_negative() {
        let mut engine = KeywordSearchEngine::new();

        let doc1 = DocId::from_u64(1);
        engine.add_document(doc1, "test document".to_string());

        let results = engine.search("test", 1);

        // All scores should be non-negative
        for (_, score) in &results {
            assert!(*score >= 0.0, "Score should be non-negative, got {}", score);
        }
    }
}
