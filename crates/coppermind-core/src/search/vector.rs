//! HNSW vector search for semantic similarity.
//!
//! This module provides vector search using the [rust-cv/hnsw](https://crates.io/crates/hnsw)
//! implementation of Hierarchical Navigable Small World graphs.
//!
//! # Algorithm
//!
//! HNSW builds a multi-layer graph where each layer has decreasing connectivity.
//! Search starts at the top layer (sparse) and traverses down to the bottom layer
//! (dense) for efficient approximate nearest neighbor search.
//!
//! **Time complexity:**
//! - Insert: O(log n)
//! - Search: O(log n)
//!
//! # Usage
//!
//! ```ignore
//! use coppermind_core::search::vector::VectorSearchEngine;
//!
//! let mut engine = VectorSearchEngine::new(512); // 512-dimensional embeddings
//! engine.add_document(DocId::from_u64(1), embedding_vec)?;
//!
//! // Search returns (DocId, similarity_score) pairs
//! let results = engine.search(&query_embedding, 10)?;
//! ```
//!
//! # Tombstone-Based Deletion
//!
//! HNSW doesn't support true deletion (removing nodes would break graph connectivity).
//! Instead, we use tombstone marking:
//!
//! 1. Mark entries as deleted via `mark_tombstone(index)`
//! 2. Filter tombstoned entries during search
//! 3. Periodically `compact()` to rebuild index without tombstones
//!
//! This approach is used by Weaviate, Milvus, and other production vector databases.
//!
//! # Integration with Hybrid Search
//!
//! This engine is used alongside [`KeywordSearchEngine`](super::keyword::KeywordSearchEngine)
//! in the [`HybridSearchEngine`](super::engine::HybridSearchEngine). Results from both
//! are combined using [Reciprocal Rank Fusion](super::fusion::reciprocal_rank_fusion).

use super::types::{validate_dimension, DocId, SearchError};
use hnsw::{Hnsw, Searcher};
use space::{Metric, Neighbor};
use std::collections::HashSet;
use tracing::instrument;

/// Minimum ef_search parameter for HNSW queries.
///
/// ef_search controls recall vs speed tradeoff in HNSW search:
/// - Higher values = better recall but slower
/// - Lower values = faster but may miss relevant results
///
/// We use max(k * 2, MIN_EF_SEARCH) to scale with result count
/// while ensuring a minimum quality floor.
///
/// A value of 50 provides good recall for most use cases.
/// Reference: HNSW paper recommends ef_search >= k for good results.
const MIN_EF_SEARCH: usize = 50;

/// Cosine distance metric for embedding vectors
/// Computes 1 - cosine_similarity, scaled to u32
///
/// Accepts Box<[f32]> for owned, stable heap allocations that avoid lifetime issues.
struct CosineDistance;

impl Metric<Box<[f32]>> for CosineDistance {
    type Unit = u32;

    fn distance(&self, a: &Box<[f32]>, b: &Box<[f32]>) -> u32 {
        // Deref Box to &[f32] - zero cost abstraction
        let a_slice: &[f32] = a;
        let b_slice: &[f32] = b;

        let dot: f32 = a_slice
            .iter()
            .zip(b_slice.iter())
            .map(|(&x, &y)| x * y)
            .sum();
        let mag_a: f32 = a_slice.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b_slice.iter().map(|y| y * y).sum::<f32>().sqrt();

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
///
/// Memory layout: Index owns Box<[f32]> embeddings on the heap. Box provides
/// stable heap allocations that won't move when the index grows, avoiding
/// lifetime issues entirely without requiring unsafe code.
///
/// # HNSW Parameters
///
/// - **M = 16**: Number of bidirectional links per node at layers > 0.
///   Higher values improve recall at cost of memory and build time.
///   Range 12-48 is typical; 16 is the paper's recommendation for balanced performance.
///
/// - **M0 = 32**: Number of links at layer 0 (entry layer). Standard practice
///   is M0 = 2*M for denser connectivity at the base layer.
///
/// Reference: "Efficient and robust approximate nearest neighbor search using
/// Hierarchical Navigable Small World graphs" by Malkov & Yashunin (2018).
/// arXiv:1603.09320
pub struct VectorSearchEngine {
    /// HNSW index for semantic similarity search using cosine distance
    /// Type parameters: <Metric, Data, RNG, M, M0>
    /// - Metric: CosineDistance implementation
    /// - Data: Box<[f32]> - owned heap-allocated embeddings
    /// - RNG: StdRng for WASM compatibility
    /// - M: 16 - bidirectional links per node (paper recommendation)
    /// - M0: 32 - layer 0 connections (2*M per standard practice)
    index: Hnsw<CosineDistance, Box<[f32]>, rand::rngs::StdRng, 16, 32>,
    /// Searcher for performing queries (mutated during search)
    searcher: Searcher<u32>,
    /// Map from HNSW index position to DocId
    doc_ids: Vec<DocId>,
    /// Dimensionality of embeddings (e.g., 512 for JinaBERT)
    dimension: usize,
    /// Set of tombstoned indices (soft-deleted, excluded from search)
    ///
    /// HNSW doesn't support true deletion, so we use tombstones to mark
    /// entries that should be filtered from search results. The indices
    /// are preserved in the HNSW graph but excluded during result filtering.
    /// Background compaction can rebuild the index to reclaim space.
    tombstones: HashSet<usize>,
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
    /// - `ef_construction`: 200 (implicit in Hnsw::new)
    pub fn new(dimension: usize) -> Self {
        let index = Hnsw::new(CosineDistance);
        let searcher = Searcher::default();

        Self {
            index,
            searcher,
            doc_ids: Vec::new(),
            dimension,
            tombstones: HashSet::new(),
        }
    }

    /// Add a document embedding to the index
    ///
    /// This method supports incremental insertion without rebuilding the entire index.
    /// The HNSW algorithm allows efficient online insertion while maintaining search quality.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if embedding dimension doesn't match
    /// the engine's configured dimension.
    ///
    /// # Memory Safety
    /// Embeddings are converted to Box<[f32]> (stable heap allocation) and owned by the
    /// HNSW index. This avoids lifetime issues without requiring unsafe code.
    #[instrument(skip_all, fields(index_size = self.doc_ids.len()))]
    pub fn add_document(&mut self, doc_id: DocId, embedding: Vec<f32>) -> Result<(), SearchError> {
        validate_dimension(self.dimension, embedding.len())?;

        // Convert Vec<f32> to Box<[f32]> for stable heap allocation
        // This is a zero-copy conversion - just changes the container type
        let boxed_embedding = embedding.into_boxed_slice();

        self.doc_ids.push(doc_id);

        // Insert into HNSW index - index takes ownership of the Box
        self.index.insert(boxed_embedding, &mut self.searcher);
        Ok(())
    }

    /// Add a document embedding without rebuilding the index
    ///
    /// With rust-cv/hnsw, this is identical to add_document() since incremental
    /// insertion is supported natively. This method is kept for API compatibility.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if embedding dimension doesn't match
    /// the engine's configured dimension.
    pub fn add_document_deferred(
        &mut self,
        doc_id: DocId,
        embedding: Vec<f32>,
    ) -> Result<(), SearchError> {
        // rust-cv/hnsw supports incremental insertion, so this is the same as add_document
        self.add_document(doc_id, embedding)
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
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if query embedding dimension doesn't match
    /// the engine's configured dimension.
    ///
    /// Note: Takes `&mut self` because the internal searcher state is mutated during search
    pub fn search(
        &mut self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<(DocId, f32)>, SearchError> {
        validate_dimension(self.dimension, query_embedding.len())?;

        if self.doc_ids.is_empty() {
            return Ok(vec![]); // No documents indexed
        }

        // Allocate neighbor buffer for min(k, index_size) results
        // If index has fewer items than k, we can only return what's available
        let actual_k = std::cmp::min(k, self.doc_ids.len());
        let mut neighbors = vec![
            Neighbor {
                index: !0,
                distance: !0
            };
            actual_k
        ];

        // Search HNSW index
        // ef_search controls search quality (higher = better but slower)
        // We use max(k * 2, MIN_EF_SEARCH) for good quality
        let ef_search = std::cmp::max(k * 2, MIN_EF_SEARCH);

        // Convert query to Box<[f32]> to match the index's data type
        let query_box = query_embedding.to_vec().into_boxed_slice();

        self.index
            .nearest(&query_box, ef_search, &mut self.searcher, &mut neighbors);

        // Convert results to (DocId, score) pairs, filtering out tombstoned entries
        let results = neighbors
            .into_iter()
            .filter(|n| n.index != !0) // Filter out unfilled entries
            .filter(|n| !self.tombstones.contains(&n.index)) // Filter out tombstoned entries
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
            .collect();
        Ok(results)
    }

    /// Get number of indexed documents
    #[allow(dead_code)] // Public API
    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }

    /// Check if index is empty
    #[allow(dead_code)] // Public API
    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    /// Mark an index position as tombstoned (soft-deleted).
    ///
    /// Tombstoned entries remain in the HNSW graph but are filtered from
    /// search results. This provides O(1) "deletion" without expensive
    /// index rebuilding.
    ///
    /// # Arguments
    /// * `idx` - The index position to tombstone (HNSW internal index)
    ///
    /// # Note
    /// This is the internal index position, not the DocId. The caller
    /// (HybridSearchEngine) tracks the mapping from DocId to index.
    pub fn mark_tombstone(&mut self, idx: usize) {
        self.tombstones.insert(idx);
    }

    /// Get all tombstoned indices.
    ///
    /// Returns a copy of the tombstone set for persistence.
    pub fn get_tombstones(&self) -> HashSet<usize> {
        self.tombstones.clone()
    }

    /// Set tombstones from a previously persisted set.
    ///
    /// Used when loading an index from storage to restore tombstone state.
    #[allow(dead_code)] // Will be used in Phase 3 for tombstone persistence
    pub fn set_tombstones(&mut self, tombstones: HashSet<usize>) {
        self.tombstones = tombstones;
    }

    /// Get the number of tombstoned entries.
    #[allow(dead_code)] // Public API
    pub fn tombstone_count(&self) -> usize {
        self.tombstones.len()
    }

    /// Get the DocId at a given index position.
    ///
    /// Returns None if the index is out of bounds.
    #[allow(dead_code)] // Public API
    pub fn get_doc_id_at(&self, idx: usize) -> Option<DocId> {
        self.doc_ids.get(idx).copied()
    }

    /// Find the index position for a DocId.
    ///
    /// Returns None if the DocId is not in the index.
    /// Note: This is O(n) linear search - use sparingly.
    #[allow(dead_code)] // Will be used in Phase 3 for tombstone persistence
    pub fn find_index(&self, doc_id: DocId) -> Option<usize> {
        self.doc_ids.iter().position(|&id| id == doc_id)
    }

    /// Calculate the tombstone ratio (proportion of deleted entries).
    ///
    /// Returns a value in [0.0, 1.0] representing the fraction of entries
    /// that have been soft-deleted via tombstones.
    ///
    /// # Returns
    /// - 0.0 if the index is empty or has no tombstones
    /// - ratio = tombstones / total_entries
    pub fn tombstone_ratio(&self) -> f32 {
        if self.doc_ids.is_empty() {
            return 0.0;
        }
        self.tombstones.len() as f32 / self.doc_ids.len() as f32
    }

    /// Check if compaction is recommended based on tombstone ratio.
    ///
    /// Returns true if tombstone ratio exceeds 30% threshold.
    /// This threshold balances storage overhead vs compaction cost.
    pub fn needs_compaction(&self) -> bool {
        const COMPACTION_THRESHOLD: f32 = 0.30;
        self.tombstone_ratio() > COMPACTION_THRESHOLD
    }

    /// Get live (non-tombstoned) entry count.
    #[allow(dead_code)] // Public API
    pub fn live_count(&self) -> usize {
        self.doc_ids.len() - self.tombstones.len()
    }

    /// Get all live (non-tombstoned) DocIds with their index positions.
    ///
    /// Used during compaction to know which embeddings to reload.
    pub fn get_live_entries(&self) -> Vec<(usize, DocId)> {
        self.doc_ids
            .iter()
            .enumerate()
            .filter(|(idx, _)| !self.tombstones.contains(idx))
            .map(|(idx, &doc_id)| (idx, doc_id))
            .collect()
    }

    /// Rebuild the index from a list of (DocId, embedding) pairs.
    ///
    /// This creates a fresh HNSW index with only the provided entries,
    /// effectively compacting away all tombstoned entries.
    ///
    /// # Arguments
    /// * `entries` - Vec of (DocId, embedding) pairs to include in new index
    ///
    /// # Returns
    /// The number of entries in the compacted index
    ///
    /// # Note
    /// This is an expensive operation that rebuilds the entire HNSW graph.
    /// Should only be called when `needs_compaction()` returns true.
    #[instrument(skip_all, fields(old_size = self.doc_ids.len(), new_size = entries.len()))]
    pub fn compact(&mut self, entries: Vec<(DocId, Vec<f32>)>) -> Result<usize, SearchError> {
        // Validate all embeddings have correct dimension
        for (_, embedding) in &entries {
            validate_dimension(self.dimension, embedding.len())?;
        }

        // Create fresh index and state
        let mut new_index = Hnsw::new(CosineDistance);
        let mut new_searcher = Searcher::default();
        let mut new_doc_ids = Vec::with_capacity(entries.len());

        // Insert all entries into new index
        for (doc_id, embedding) in entries {
            new_doc_ids.push(doc_id);
            let boxed_embedding = embedding.into_boxed_slice();
            new_index.insert(boxed_embedding, &mut new_searcher);
        }

        let compacted_count = new_doc_ids.len();

        // Atomic swap - replace old index with new one
        self.index = new_index;
        self.searcher = new_searcher;
        self.doc_ids = new_doc_ids;
        self.tombstones.clear(); // No tombstones in fresh index

        Ok(compacted_count)
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

        engine.add_document(doc1, vec![1.0, 0.0, 0.0]).unwrap();
        engine.add_document(doc2, vec![0.0, 1.0, 0.0]).unwrap();
        engine.add_document(doc3, vec![1.0, 0.1, 0.0]).unwrap();

        // Search for something close to doc1 and doc3
        let results = engine.search(&[1.0, 0.0, 0.0], 2).unwrap();

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
        engine.add_document(doc1, vec![1.0, 0.0, 0.0]).unwrap();

        assert_eq!(engine.len(), 1);
        assert!(!engine.is_empty());

        let doc2 = DocId::from_u64(2);
        engine.add_document(doc2, vec![0.0, 1.0, 0.0]).unwrap();

        assert_eq!(engine.len(), 2);
    }

    #[test]
    fn test_search_empty_index() {
        let mut engine = VectorSearchEngine::new(3);

        let results = engine.search(&[1.0, 0.0, 0.0], 10).unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn test_incremental_insertion() {
        let mut engine = VectorSearchEngine::new(3);

        // Add documents incrementally (no rebuild needed with rust-cv/hnsw)
        for i in 0..5 {
            let doc_id = DocId::from_u64(i);
            engine
                .add_document(doc_id, vec![i as f32, 0.0, 0.0])
                .unwrap();
        }

        // Index should have all documents
        assert_eq!(engine.len(), 5);

        // Search should work immediately (no rebuild needed)
        let results = engine.search(&[2.0, 0.0, 0.0], 3).unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_deferred_add_is_same_as_regular() {
        let mut engine = VectorSearchEngine::new(3);

        // With rust-cv/hnsw, deferred add is the same as regular add
        for i in 0..5 {
            let doc_id = DocId::from_u64(i);
            engine
                .add_document_deferred(doc_id, vec![i as f32, 0.0, 0.0])
                .unwrap();
        }

        // Documents should be searchable immediately
        assert_eq!(engine.len(), 5);
        let results = engine.search(&[2.0, 0.0, 0.0], 3).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_rebuild_index_is_noop() {
        let mut engine = VectorSearchEngine::new(3);

        // Add documents
        for i in 0..5 {
            let doc_id = DocId::from_u64(i);
            engine
                .add_document_deferred(doc_id, vec![i as f32, 0.0, 0.0])
                .unwrap();
        }

        // rebuild_index should do nothing (no-op)
        engine.rebuild_index();

        // Index should still work
        assert_eq!(engine.len(), 5);
        let results = engine.search(&[2.0, 0.0, 0.0], 3).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_exact_match_returns_high_similarity() {
        let mut engine = VectorSearchEngine::new(3);

        let doc1 = DocId::from_u64(1);
        let embedding = vec![0.5, 0.3, 0.2];

        engine.add_document(doc1, embedding.clone()).unwrap();

        // Search with exact same embedding
        let results = engine.search(&embedding, 1).unwrap();

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
            engine.add_document(doc_id, embedding).unwrap();
        }

        assert_eq!(engine.len(), 100);

        // Search for something in the middle
        let query = vec![0.5, 0.5, 0.0];
        let results = engine.search(&query, 10).unwrap();

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
    fn test_add_document_dimension_mismatch() {
        let mut engine = VectorSearchEngine::new(3);
        let doc_id = DocId::from_u64(1);

        // Try to add 2D embedding to 3D engine
        let result = engine.add_document(doc_id, vec![1.0, 0.0]);
        assert!(matches!(
            result,
            Err(SearchError::DimensionMismatch {
                expected: 3,
                actual: 2
            })
        ));
    }

    #[test]
    fn test_search_dimension_mismatch() {
        let mut engine = VectorSearchEngine::new(3);
        let doc_id = DocId::from_u64(1);
        engine.add_document(doc_id, vec![1.0, 0.0, 0.0]).unwrap();

        // Try to search with 2D query
        let result = engine.search(&[1.0, 0.0], 1);
        assert!(matches!(
            result,
            Err(SearchError::DimensionMismatch {
                expected: 3,
                actual: 2
            })
        ));
    }

    #[test]
    fn test_search_returns_top_k() {
        let mut engine = VectorSearchEngine::new(3);

        // Add 20 documents
        for i in 0..20 {
            let doc_id = DocId::from_u64(i);
            engine
                .add_document(doc_id, vec![i as f32, 0.0, 0.0])
                .unwrap();
        }

        // Request top 5
        let results = engine.search(&[10.0, 0.0, 0.0], 5).unwrap();

        // Should return exactly 5 results
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_similarity_score_range() {
        let mut engine = VectorSearchEngine::new(3);

        let doc1 = DocId::from_u64(1);
        let doc2 = DocId::from_u64(2);

        engine.add_document(doc1, vec![1.0, 0.0, 0.0]).unwrap();
        engine.add_document(doc2, vec![0.0, 1.0, 0.0]).unwrap();

        let results = engine.search(&[1.0, 0.0, 0.0], 2).unwrap();

        // All similarity scores should be in [0, 1] range
        for (_, score) in &results {
            assert!(
                *score >= 0.0 && *score <= 1.0,
                "Score {} out of range [0, 1]",
                score
            );
        }
    }

    #[test]
    fn test_tombstone_filters_results() {
        let mut engine = VectorSearchEngine::new(3);

        let doc1 = DocId::from_u64(1);
        let doc2 = DocId::from_u64(2);
        let doc3 = DocId::from_u64(3);

        engine.add_document(doc1, vec![1.0, 0.0, 0.0]).unwrap();
        engine.add_document(doc2, vec![0.9, 0.1, 0.0]).unwrap();
        engine.add_document(doc3, vec![0.0, 1.0, 0.0]).unwrap();

        // Search should return all 3 documents
        let results = engine.search(&[1.0, 0.0, 0.0], 10).unwrap();
        assert_eq!(results.len(), 3);

        // Tombstone doc1 (index 0)
        engine.mark_tombstone(0);

        // Search should now exclude doc1
        let results = engine.search(&[1.0, 0.0, 0.0], 10).unwrap();
        assert_eq!(results.len(), 2);
        assert!(!results.iter().any(|(id, _)| *id == doc1));
    }

    #[test]
    fn test_tombstone_persistence() {
        let mut engine = VectorSearchEngine::new(3);

        engine
            .add_document(DocId::from_u64(1), vec![1.0, 0.0, 0.0])
            .unwrap();
        engine
            .add_document(DocId::from_u64(2), vec![0.0, 1.0, 0.0])
            .unwrap();

        // Mark some tombstones
        engine.mark_tombstone(0);
        engine.mark_tombstone(1);

        // Get tombstones for persistence
        let tombstones = engine.get_tombstones();
        assert_eq!(tombstones.len(), 2);
        assert!(tombstones.contains(&0));
        assert!(tombstones.contains(&1));

        // Create new engine and restore tombstones
        let mut engine2 = VectorSearchEngine::new(3);
        engine2
            .add_document(DocId::from_u64(1), vec![1.0, 0.0, 0.0])
            .unwrap();
        engine2
            .add_document(DocId::from_u64(2), vec![0.0, 1.0, 0.0])
            .unwrap();
        engine2.set_tombstones(tombstones);

        // Search should return no results (all tombstoned)
        let results = engine2.search(&[1.0, 0.0, 0.0], 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_find_index() {
        let mut engine = VectorSearchEngine::new(3);

        let doc1 = DocId::from_u64(100);
        let doc2 = DocId::from_u64(200);
        let doc3 = DocId::from_u64(300);

        engine.add_document(doc1, vec![1.0, 0.0, 0.0]).unwrap();
        engine.add_document(doc2, vec![0.0, 1.0, 0.0]).unwrap();
        engine.add_document(doc3, vec![0.0, 0.0, 1.0]).unwrap();

        // Find indices
        assert_eq!(engine.find_index(doc1), Some(0));
        assert_eq!(engine.find_index(doc2), Some(1));
        assert_eq!(engine.find_index(doc3), Some(2));

        // Non-existent doc
        assert_eq!(engine.find_index(DocId::from_u64(999)), None);
    }

    #[test]
    fn test_tombstone_count() {
        let mut engine = VectorSearchEngine::new(3);

        engine
            .add_document(DocId::from_u64(1), vec![1.0, 0.0, 0.0])
            .unwrap();
        engine
            .add_document(DocId::from_u64(2), vec![0.0, 1.0, 0.0])
            .unwrap();

        assert_eq!(engine.tombstone_count(), 0);

        engine.mark_tombstone(0);
        assert_eq!(engine.tombstone_count(), 1);

        engine.mark_tombstone(1);
        assert_eq!(engine.tombstone_count(), 2);

        // Marking same index again doesn't increase count
        engine.mark_tombstone(0);
        assert_eq!(engine.tombstone_count(), 2);
    }

    #[test]
    fn test_tombstone_ratio() {
        let mut engine = VectorSearchEngine::new(3);

        // Empty index has 0 ratio
        assert_eq!(engine.tombstone_ratio(), 0.0);

        // Add 10 documents
        for i in 0..10 {
            engine
                .add_document(DocId::from_u64(i), vec![i as f32, 0.0, 0.0])
                .unwrap();
        }

        // No tombstones = 0 ratio
        assert_eq!(engine.tombstone_ratio(), 0.0);

        // Tombstone 3 of 10 = 30%
        engine.mark_tombstone(0);
        engine.mark_tombstone(1);
        engine.mark_tombstone(2);
        assert!((engine.tombstone_ratio() - 0.3).abs() < 0.01);

        // Tombstone 5 of 10 = 50%
        engine.mark_tombstone(3);
        engine.mark_tombstone(4);
        assert!((engine.tombstone_ratio() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_needs_compaction() {
        let mut engine = VectorSearchEngine::new(3);

        // Add 10 documents
        for i in 0..10 {
            engine
                .add_document(DocId::from_u64(i), vec![i as f32, 0.0, 0.0])
                .unwrap();
        }

        // No tombstones - no compaction needed
        assert!(!engine.needs_compaction());

        // 20% tombstones - still no compaction (threshold is 30%)
        engine.mark_tombstone(0);
        engine.mark_tombstone(1);
        assert!(!engine.needs_compaction());

        // 30% tombstones - still no compaction (threshold is >30%)
        engine.mark_tombstone(2);
        assert!(!engine.needs_compaction());

        // 40% tombstones - compaction needed
        engine.mark_tombstone(3);
        assert!(engine.needs_compaction());
    }

    #[test]
    fn test_live_count_and_entries() {
        let mut engine = VectorSearchEngine::new(3);

        let doc1 = DocId::from_u64(1);
        let doc2 = DocId::from_u64(2);
        let doc3 = DocId::from_u64(3);

        engine.add_document(doc1, vec![1.0, 0.0, 0.0]).unwrap();
        engine.add_document(doc2, vec![0.0, 1.0, 0.0]).unwrap();
        engine.add_document(doc3, vec![0.0, 0.0, 1.0]).unwrap();

        // All 3 are live
        assert_eq!(engine.live_count(), 3);
        let live_entries = engine.get_live_entries();
        assert_eq!(live_entries.len(), 3);

        // Tombstone one
        engine.mark_tombstone(1);
        assert_eq!(engine.live_count(), 2);
        let live_entries = engine.get_live_entries();
        assert_eq!(live_entries.len(), 2);
        assert!(live_entries.iter().any(|(_, id)| *id == doc1));
        assert!(!live_entries.iter().any(|(_, id)| *id == doc2));
        assert!(live_entries.iter().any(|(_, id)| *id == doc3));
    }

    #[test]
    fn test_compact() {
        let mut engine = VectorSearchEngine::new(3);

        let doc1 = DocId::from_u64(1);
        let doc2 = DocId::from_u64(2);
        let doc3 = DocId::from_u64(3);

        let emb1 = vec![1.0, 0.0, 0.0];
        let emb2 = vec![0.0, 1.0, 0.0];
        let emb3 = vec![0.0, 0.0, 1.0];

        engine.add_document(doc1, emb1.clone()).unwrap();
        engine.add_document(doc2, emb2.clone()).unwrap();
        engine.add_document(doc3, emb3.clone()).unwrap();

        // Tombstone doc2
        engine.mark_tombstone(1);
        assert_eq!(engine.len(), 3);
        assert_eq!(engine.tombstone_count(), 1);

        // Compact with only live entries
        let entries = vec![(doc1, emb1.clone()), (doc3, emb3.clone())];
        let compacted_count = engine.compact(entries).unwrap();

        assert_eq!(compacted_count, 2);
        assert_eq!(engine.len(), 2);
        assert_eq!(engine.tombstone_count(), 0);
        assert!(!engine.needs_compaction());

        // Search should work on compacted index
        let results = engine.search(&emb1, 2).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|(id, _)| *id == doc1));
        assert!(results.iter().any(|(id, _)| *id == doc3));
        assert!(!results.iter().any(|(id, _)| *id == doc2)); // doc2 was removed
    }

    #[test]
    fn test_compact_empty() {
        let mut engine = VectorSearchEngine::new(3);

        // Add some documents and tombstone all of them
        engine
            .add_document(DocId::from_u64(1), vec![1.0, 0.0, 0.0])
            .unwrap();
        engine
            .add_document(DocId::from_u64(2), vec![0.0, 1.0, 0.0])
            .unwrap();
        engine.mark_tombstone(0);
        engine.mark_tombstone(1);

        // Compact with empty entries (everything was deleted)
        let compacted_count = engine.compact(vec![]).unwrap();

        assert_eq!(compacted_count, 0);
        assert_eq!(engine.len(), 0);
        assert!(engine.is_empty());
    }

    #[test]
    fn test_compact_dimension_mismatch() {
        let mut engine = VectorSearchEngine::new(3);

        // Try to compact with wrong dimension
        let result = engine.compact(vec![(DocId::from_u64(1), vec![1.0, 0.0])]); // 2D instead of 3D

        assert!(matches!(
            result,
            Err(SearchError::DimensionMismatch {
                expected: 3,
                actual: 2
            })
        ));
    }
}
