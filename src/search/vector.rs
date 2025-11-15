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

        // Build HNSW index
        // Using default Builder parameters
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
}
