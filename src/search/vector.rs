// rust-cv/hnsw vector search integration with incremental updates

use super::types::DocId;
use hnsw::{Hnsw, Searcher};
use space::{Metric, Neighbor};

/// Cosine distance metric for embedding vectors
/// Computes 1 - cosine_similarity, scaled to u32
struct CosineDistance;

impl Metric<&[f32]> for CosineDistance {
    type Unit = u32;

    fn distance(&self, a: &&[f32], b: &&[f32]) -> u32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();

        if mag_a == 0.0 || mag_b == 0.0 {
            return u32::MAX; // Maximum distance for zero vectors
        }

        let cosine_sim = dot / (mag_a * mag_b);
        let distance = 1.0 - cosine_sim; // Convert similarity to distance [0, 2]

        // Convert to u32 by scaling to [0, u32::MAX]
        // Distance is in [0, 2], so we scale by u32::MAX/2
        (distance * (u32::MAX as f32 / 2.0)) as u32
    }
}

/// Vector search engine using HNSW (Hierarchical Navigable Small World)
///
/// This implementation uses rust-cv/hnsw which supports incremental updates
/// and is WASM-compatible (no native dependencies).
pub struct VectorSearchEngine {
    /// HNSW index for semantic similarity search using cosine distance
    /// Type parameters: Metric, Data, RNG, M (connections), M0 (layer 0 connections)
    index: Hnsw<CosineDistance, &'static [f32], rand::rngs::StdRng, 16, 32>,
    /// Searcher for performing queries
    searcher: Searcher<u32>,
    /// Storage for embeddings (needed because index stores references)
    embeddings: Vec<Vec<f32>>,
    /// Map from storage index to DocId
    doc_ids: Vec<DocId>,
    /// Dimensionality of embeddings (e.g., 512 for JinaBERT)
    dimension: usize,
}

impl VectorSearchEngine {
    /// Create a new vector search engine
    ///
    /// # Arguments
    /// * `dimension` - Dimensionality of embeddings (must match the model)
    ///
    /// # HNSW Parameters
    /// - `M`: 16 - Number of bidirectional links per node
    ///   - Higher M = better recall, more memory
    ///   - Default is 12, we use 16 for better accuracy
    /// - `M0`: 32 - Number of links for layer 0 (2*M per standard practice)
    pub fn new(dimension: usize) -> Self {
        let index = Hnsw::new(CosineDistance);
        let searcher = Searcher::default();

        Self {
            index,
            searcher,
            embeddings: Vec::new(),
            doc_ids: Vec::new(),
            dimension,
        }
    }

    /// Add a document embedding to the index
    ///
    /// This method supports incremental insertion without rebuilding the entire index.
    /// The HNSW algorithm allows efficient online insertion while maintaining search quality.
    pub fn add_document(&mut self, doc_id: DocId, embedding: Vec<f32>) {
        assert_eq!(
            embedding.len(),
            self.dimension,
            "Embedding dimension mismatch"
        );

        // Store embedding and doc_id
        self.embeddings.push(embedding);
        self.doc_ids.push(doc_id);

        // Insert into HNSW index
        // SAFETY: We're leaking the embedding reference to make it 'static
        // The embeddings vec is never modified after insertion, only appended to
        let embedding_ref: &'static [f32] =
            unsafe { std::mem::transmute(self.embeddings.last().unwrap().as_slice()) };

        self.index.insert(embedding_ref, &mut self.searcher);
    }

    /// Add a document embedding without rebuilding the index
    ///
    /// With rust-cv/hnsw, this is identical to add_document() since incremental
    /// insertion is supported natively. This method is kept for API compatibility.
    pub fn add_document_deferred(&mut self, doc_id: DocId, embedding: Vec<f32>) {
        // rust-cv/hnsw supports incremental insertion, so this is the same as add_document
        self.add_document(doc_id, embedding);
    }

    /// Rebuild the HNSW index from all stored embeddings
    ///
    /// NOTE: With rust-cv/hnsw, this is a no-op since the index supports incremental updates.
    /// This method is kept for API compatibility but does nothing.
    ///
    /// Unlike instant-distance which required full rebuilds, rust-cv/hnsw maintains
    /// the index structure during insertions, so no rebuild is needed.
    #[allow(dead_code)] // Public API for compatibility
    pub fn rebuild_index(&mut self) {
        // No-op: rust-cv/hnsw supports incremental insertion
        // Index is already up-to-date
    }

    /// Search for k nearest neighbors using vector similarity
    ///
    /// # Arguments
    /// * `query_embedding` - Query vector (must match dimension)
    /// * `k` - Number of nearest neighbors to return
    ///
    /// # Returns
    /// Vector of (DocId, similarity_score) pairs, sorted by similarity (descending)
    /// Similarity scores are in [0, 1] range where 1.0 is most similar
    ///
    /// Note: Takes `&mut self` because the internal searcher state is mutated during search
    pub fn search(&mut self, query_embedding: &[f32], k: usize) -> Vec<(DocId, f32)> {
        assert_eq!(
            query_embedding.len(),
            self.dimension,
            "Query embedding dimension mismatch"
        );

        if self.embeddings.is_empty() {
            return vec![]; // No documents indexed
        }

        // Allocate neighbor buffer for min(k, index_size) results
        // If index has fewer items than k, we can only return what's available
        let actual_k = std::cmp::min(k, self.embeddings.len());
        let mut neighbors = vec![
            Neighbor {
                index: !0,
                distance: !0
            };
            actual_k
        ];

        // Search HNSW index
        // ef_search controls search quality (higher = better but slower)
        // We use max(k * 2, 50) for good quality
        let ef_search = std::cmp::max(k * 2, 50);
        self.index.nearest(
            &query_embedding,
            ef_search,
            &mut self.searcher,
            &mut neighbors,
        );

        // Convert results to (DocId, score) pairs
        neighbors
            .into_iter()
            .filter(|n| n.index != !0) // Filter out unfilled entries
            .map(|neighbor| {
                // neighbor.distance is u32 cosine distance
                // Convert back to similarity score
                let distance_f32 = (neighbor.distance as f32) / (u32::MAX as f32 / 2.0);
                let similarity = 1.0 - distance_f32;

                // Clamp to [0, 1] to handle any floating point errors
                let similarity = similarity.clamp(0.0, 1.0);

                let doc_id = self.doc_ids[neighbor.index];
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
        let mut engine = VectorSearchEngine::new(3);

        let results = engine.search(&[1.0, 0.0, 0.0], 10);

        assert!(results.is_empty());
    }

    #[test]
    fn test_incremental_insertion() {
        let mut engine = VectorSearchEngine::new(3);

        // Add documents incrementally (no rebuild needed with rust-cv/hnsw)
        for i in 0..5 {
            let doc_id = DocId::from_u64(i);
            engine.add_document(doc_id, vec![i as f32, 0.0, 0.0]);
        }

        // Index should have all documents
        assert_eq!(engine.len(), 5);

        // Search should work immediately (no rebuild needed)
        let results = engine.search(&[2.0, 0.0, 0.0], 3);
        assert!(!results.is_empty());
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_deferred_add_is_same_as_regular() {
        let mut engine = VectorSearchEngine::new(3);

        // With rust-cv/hnsw, deferred add is the same as regular add
        for i in 0..5 {
            let doc_id = DocId::from_u64(i);
            engine.add_document_deferred(doc_id, vec![i as f32, 0.0, 0.0]);
        }

        // Documents should be searchable immediately
        assert_eq!(engine.len(), 5);
        let results = engine.search(&[2.0, 0.0, 0.0], 3);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_rebuild_index_is_noop() {
        let mut engine = VectorSearchEngine::new(3);

        // Add documents
        for i in 0..5 {
            let doc_id = DocId::from_u64(i);
            engine.add_document_deferred(doc_id, vec![i as f32, 0.0, 0.0]);
        }

        // rebuild_index should do nothing (no-op)
        engine.rebuild_index();

        // Index should still work
        assert_eq!(engine.len(), 5);
        let results = engine.search(&[2.0, 0.0, 0.0], 3);
        assert!(!results.is_empty());
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
        // Allow small floating point error
        assert!(
            results[0].1 > 0.95,
            "Exact match should have similarity ~1.0, got {}",
            results[0].1
        );
    }

    #[test]
    fn test_batch_operations() {
        let mut engine = VectorSearchEngine::new(3);

        // Add 100 documents incrementally
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
