// instant-distance vector search integration using HNSW

use super::types::DocId;
use instant_distance::{Builder, HnswMap, Point, Search};
use std::collections::HashMap;

/// Wrapper for embedding vectors that implements Point trait
#[derive(Clone, Debug)]
pub struct EmbeddingPoint(pub Vec<f32>);

impl Point for EmbeddingPoint {
    fn distance(&self, other: &Self) -> f32 {
        // Cosine distance (1 - cosine similarity)
        let dot: f32 = self.0.iter().zip(&other.0).map(|(a, b)| a * b).sum();
        let mag_a: f32 = self.0.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = other.0.iter().map(|x| x * x).sum::<f32>().sqrt();

        if mag_a == 0.0 || mag_b == 0.0 {
            return 1.0; // Maximum distance for zero vectors
        }

        let cosine_sim = dot / (mag_a * mag_b);
        1.0 - cosine_sim // Convert similarity to distance
    }
}

/// Vector search engine using HNSW (Hierarchical Navigable Small World)
pub struct VectorSearchEngine {
    /// HNSW index mapping embeddings to DocIds
    index: Option<HnswMap<EmbeddingPoint, DocId>>,
    /// Map from DocId to embedding (for rebuilding index if needed)
    embeddings: HashMap<DocId, Vec<f32>>,
    /// Dimensionality of embeddings (e.g., 512 for JinaBERT)
    dimension: usize,
}

impl VectorSearchEngine {
    pub fn new(dimension: usize) -> Self {
        Self {
            index: None,
            embeddings: HashMap::new(),
            dimension,
        }
    }

    /// Add a document embedding to the index
    pub fn add_document(&mut self, doc_id: DocId, embedding: Vec<f32>) {
        assert_eq!(
            embedding.len(),
            self.dimension,
            "Embedding dimension mismatch"
        );

        self.embeddings.insert(doc_id, embedding);

        // Rebuild index when documents are added
        self.rebuild_index();
    }

    /// Add a document embedding without rebuilding the index
    /// Use this for batch inserts, then call rebuild_index() once at the end
    pub fn add_document_deferred(&mut self, doc_id: DocId, embedding: Vec<f32>) {
        assert_eq!(
            embedding.len(),
            self.dimension,
            "Embedding dimension mismatch"
        );

        self.embeddings.insert(doc_id, embedding);
    }

    /// Rebuild the HNSW index from all stored embeddings
    ///
    /// NOTE: This is CPU-intensive and rebuilds the entire graph from scratch.
    /// instant-distance HNSW does not support incremental insertion.
    /// Time complexity: O(n log n) where n is the number of documents.
    ///
    /// For large indexes (10k+ documents), this can take several minutes.
    pub fn rebuild_index(&mut self) {
        if self.embeddings.is_empty() {
            self.index = None;
            return;
        }

        // Separate points and values for Builder::build()
        let (points, values): (Vec<EmbeddingPoint>, Vec<DocId>) = self
            .embeddings
            .iter()
            .map(|(doc_id, embedding)| (EmbeddingPoint(embedding.clone()), *doc_id))
            .unzip();

        // Build HNSW index (CPU-intensive for large datasets)
        // instant-distance constructs the entire navigable small world graph
        // Using default Builder parameters (M=12, ef_construction=100)
        let map = Builder::default().build(points, values);

        self.index = Some(map);
    }

    /// Search for k nearest neighbors using vector similarity
    pub fn search(&self, query_embedding: &[f32], k: usize) -> Vec<(DocId, f32)> {
        assert_eq!(
            query_embedding.len(),
            self.dimension,
            "Query embedding dimension mismatch"
        );

        let index = match &self.index {
            Some(idx) => idx,
            None => return vec![], // No documents indexed
        };

        let query_point = EmbeddingPoint(query_embedding.to_vec());

        // Search for k nearest neighbors
        let mut search = Search::default();
        let neighbors = index.search(&query_point, &mut search);

        // Convert to (DocId, score) pairs
        // instant-distance returns Item which contains value (DocId) and distance
        neighbors
            .take(k)
            .map(|item| {
                let doc_id = *item.value;
                let distance = item.distance;
                // Convert distance back to similarity score (higher is better)
                let similarity = 1.0 - distance;
                (doc_id, similarity)
            })
            .collect()
    }

    /// Get number of indexed documents
    #[allow(dead_code)] // Public API
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Check if index is empty
    #[allow(dead_code)] // Public API
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_search() {
        let mut engine = VectorSearchEngine::new(3);

        // Add some test documents
        let doc1 = DocId::from_u64(1);
        let doc2 = DocId::from_u64(2);
        let doc3 = DocId::from_u64(3);

        engine.add_document(doc1, vec![1.0, 0.0, 0.0]);
        engine.add_document(doc2, vec![0.0, 1.0, 0.0]);
        engine.add_document(doc3, vec![1.0, 0.1, 0.0]);

        // Search for something close to doc1 and doc3
        let results = engine.search(&[1.0, 0.0, 0.0], 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, doc1); // Exact match should be first
        assert_eq!(results[1].0, doc3); // Similar vector should be second
    }

    #[test]
    fn test_add_document_updates_count() {
        let mut engine = VectorSearchEngine::new(3);

        assert_eq!(engine.len(), 0);
        assert!(engine.is_empty());

        let doc1 = DocId::from_u64(1);
        engine.add_document(doc1, vec![1.0, 0.0, 0.0]);

        assert_eq!(engine.len(), 1);
        assert!(!engine.is_empty());

        let doc2 = DocId::from_u64(2);
        engine.add_document(doc2, vec![0.0, 1.0, 0.0]);

        assert_eq!(engine.len(), 2);
    }

    #[test]
    fn test_search_empty_index() {
        let engine = VectorSearchEngine::new(3);

        let results = engine.search(&[1.0, 0.0, 0.0], 10);

        assert!(results.is_empty());
    }

    #[test]
    fn test_deferred_add_doesnt_rebuild() {
        let mut engine = VectorSearchEngine::new(3);

        // Add documents deferred
        for i in 0..5 {
            let doc_id = DocId::from_u64(i);
            engine.add_document_deferred(doc_id, vec![i as f32, 0.0, 0.0]);
        }

        // Index should be None (not built yet)
        assert!(engine.index.is_none());
        assert_eq!(engine.len(), 5);

        // Search should return empty (no index)
        let results = engine.search(&[2.0, 0.0, 0.0], 3);
        assert!(results.is_empty());
    }

    #[test]
    fn test_rebuild_index_after_deferred() {
        let mut engine = VectorSearchEngine::new(3);

        // Add documents deferred
        for i in 0..5 {
            let doc_id = DocId::from_u64(i);
            engine.add_document_deferred(doc_id, vec![i as f32, 0.0, 0.0]);
        }

        // Rebuild index
        engine.rebuild_index();

        // Index should now exist
        assert!(engine.index.is_some());

        // Search should work
        let results = engine.search(&[2.0, 0.0, 0.0], 3);
        assert!(!results.is_empty());
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_cosine_distance_calculation() {
        // Test the Point trait implementation
        let p1 = EmbeddingPoint(vec![1.0, 0.0, 0.0]);
        let p2 = EmbeddingPoint(vec![1.0, 0.0, 0.0]);
        let p3 = EmbeddingPoint(vec![0.0, 1.0, 0.0]);

        // Identical vectors should have distance 0
        let dist_same = p1.distance(&p2);
        assert!(
            dist_same.abs() < 0.001,
            "Identical vectors should have ~0 distance"
        );

        // Orthogonal vectors should have distance ~1.0 (cosine similarity = 0)
        let dist_orthogonal = p1.distance(&p3);
        assert!(
            (dist_orthogonal - 1.0).abs() < 0.001,
            "Orthogonal vectors should have ~1.0 distance"
        );

        // Test zero vector handling
        let p_zero = EmbeddingPoint(vec![0.0, 0.0, 0.0]);
        let dist_zero = p1.distance(&p_zero);
        assert_eq!(dist_zero, 1.0, "Zero vector should have max distance");
    }

    #[test]
    fn test_exact_match_returns_high_similarity() {
        let mut engine = VectorSearchEngine::new(3);

        let doc1 = DocId::from_u64(1);
        let embedding = vec![0.5, 0.3, 0.2];

        engine.add_document(doc1, embedding.clone());

        // Search with exact same embedding
        let results = engine.search(&embedding, 1);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, doc1);

        // Similarity should be very close to 1.0 (perfect match)
        assert!(
            results[0].1 > 0.999,
            "Exact match should have similarity ~1.0, got {}",
            results[0].1
        );
    }

    #[test]
    fn test_batch_operations() {
        let mut engine = VectorSearchEngine::new(3);

        // Add 100 documents
        for i in 0..100 {
            let doc_id = DocId::from_u64(i);
            let angle = (i as f32) * 0.01; // Spread vectors around
            let embedding = vec![angle.cos(), angle.sin(), 0.0];
            engine.add_document(doc_id, embedding);
        }

        assert_eq!(engine.len(), 100);

        // Search for something in the middle
        let query = vec![0.5, 0.5, 0.0];
        let results = engine.search(&query, 10);

        // Should return 10 results
        assert_eq!(results.len(), 10);

        // All results should have valid scores
        for (_, score) in &results {
            assert!(*score >= 0.0 && *score <= 1.0);
        }

        // Results should be sorted by similarity (descending)
        for i in 1..results.len() {
            assert!(
                results[i - 1].1 >= results[i].1,
                "Results should be sorted by similarity"
            );
        }
    }

    #[test]
    #[should_panic(expected = "Embedding dimension mismatch")]
    fn test_add_document_dimension_mismatch() {
        let mut engine = VectorSearchEngine::new(3);
        let doc_id = DocId::from_u64(1);

        // Try to add 2D embedding to 3D engine
        engine.add_document(doc_id, vec![1.0, 0.0]); // Should panic
    }

    #[test]
    #[should_panic(expected = "Query embedding dimension mismatch")]
    fn test_search_dimension_mismatch() {
        let mut engine = VectorSearchEngine::new(3);
        let doc_id = DocId::from_u64(1);
        engine.add_document(doc_id, vec![1.0, 0.0, 0.0]);

        // Try to search with 2D query
        engine.search(&[1.0, 0.0], 1); // Should panic
    }

    #[test]
    fn test_rebuild_index_empty() {
        let mut engine = VectorSearchEngine::new(3);

        // Add and then clear by rebuilding empty
        let doc_id = DocId::from_u64(1);
        engine.add_document(doc_id, vec![1.0, 0.0, 0.0]);
        assert!(engine.index.is_some());

        // Clear embeddings
        engine.embeddings.clear();
        engine.rebuild_index();

        // Index should be None now
        assert!(engine.index.is_none());
        assert_eq!(engine.len(), 0);
    }

    #[test]
    fn test_search_returns_top_k() {
        let mut engine = VectorSearchEngine::new(3);

        // Add 20 documents
        for i in 0..20 {
            let doc_id = DocId::from_u64(i);
            engine.add_document(doc_id, vec![i as f32, 0.0, 0.0]);
        }

        // Request top 5
        let results = engine.search(&[10.0, 0.0, 0.0], 5);

        // Should return exactly 5 results
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_similarity_score_range() {
        let mut engine = VectorSearchEngine::new(3);

        let doc1 = DocId::from_u64(1);
        let doc2 = DocId::from_u64(2);

        engine.add_document(doc1, vec![1.0, 0.0, 0.0]);
        engine.add_document(doc2, vec![0.0, 1.0, 0.0]);

        let results = engine.search(&[1.0, 0.0, 0.0], 2);

        // All similarity scores should be in [0, 1] range
        for (_, score) in &results {
            assert!(
                *score >= 0.0 && *score <= 1.0,
                "Score {} out of range [0, 1]",
                score
            );
        }
    }
}
