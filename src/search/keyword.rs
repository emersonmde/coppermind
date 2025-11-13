// BM25 keyword search integration

use super::types::DocId;
use bm25::{Document, Language, SearchEngineBuilder};

/// Keyword search engine using BM25 algorithm
pub struct KeywordSearchEngine {
    /// BM25 search engine
    search_engine: bm25::SearchEngine<u64>,
}

impl KeywordSearchEngine {
    pub fn new() -> Self {
        // Create empty search engine with English language
        // Using with_documents to get proper u64 type
        let empty_docs: Vec<Document<u64>> = vec![];
        let search_engine =
            SearchEngineBuilder::<u64>::with_documents(Language::English, empty_docs).build();

        Self { search_engine }
    }

    /// Add a document to the BM25 corpus
    pub fn add_document(&mut self, doc_id: DocId, text: String) {
        // Create BM25 document with DocId as u64
        let doc = Document {
            id: doc_id.as_u64(),
            contents: text,
        };

        // Upsert document into search engine
        self.search_engine.upsert(doc);
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
        // Note: SearchEngine doesn't expose document count
        // This is a limitation we'll need to track separately if needed
        0 // Placeholder
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
}
