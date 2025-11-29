//! Index compaction to reclaim space from tombstoned entries.
//!
//! When documents are deleted, they are "tombstoned" in the HNSW index rather than
//! immediately removed. This allows fast deletion without expensive index rebuilding.
//! Compaction removes these tombstoned entries by rebuilding the index.
//!
//! # When to Compact
//!
//! Compaction is recommended when the tombstone ratio exceeds 30% of total entries.
//! Use `needs_compaction()` to check if compaction is advisable.
//!
//! # Performance Considerations
//!
//! Compaction is expensive - it rebuilds the entire HNSW graph. However, search
//! continues to work during compaction (on the old index until swap completes).

use super::HybridSearchEngine;
use crate::search::types::SearchError;
use crate::storage::DocumentStore;
use tracing::{info, instrument, warn};

impl<S: DocumentStore> HybridSearchEngine<S> {
    /// Check if the vector index needs compaction.
    ///
    /// Returns `true` if tombstone ratio exceeds 30% threshold.
    pub fn needs_compaction(&self) -> bool {
        self.vector_engine.needs_compaction()
    }

    /// Get compaction statistics.
    ///
    /// Returns `(tombstone_count, total_count, ratio)` where:
    /// - `tombstone_count`: Number of tombstoned (deleted) entries
    /// - `total_count`: Total entries in the index
    /// - `ratio`: Tombstone ratio (0.0 to 1.0)
    pub fn compaction_stats(&self) -> (usize, usize, f32) {
        let tombstone_count = self.vector_engine.tombstone_count();
        let total_count = self.vector_engine.len();
        let ratio = self.vector_engine.tombstone_ratio();
        (tombstone_count, total_count, ratio)
    }

    /// Run compaction to remove tombstoned entries from the vector index.
    ///
    /// This rebuilds the HNSW index from scratch, loading embeddings from
    /// the document store for all live (non-tombstoned) entries.
    ///
    /// # Returns
    /// The number of entries in the compacted index.
    ///
    /// # Note
    /// This is an expensive operation that:
    /// 1. Loads all live embeddings from storage
    /// 2. Rebuilds the entire HNSW graph
    /// 3. Clears tombstones from storage
    ///
    /// Search continues to work during compaction (on old index until swap).
    #[instrument(skip(self), fields(before_size = self.vector_engine.len()))]
    pub async fn compact(&mut self) -> Result<usize, SearchError> {
        let (tombstone_count, total_count, ratio) = self.compaction_stats();

        if tombstone_count == 0 {
            info!("Compaction skipped: no tombstones");
            return Ok(total_count);
        }

        info!(
            "Starting compaction: {} tombstones / {} total ({:.1}%)",
            tombstone_count,
            total_count,
            ratio * 100.0
        );

        // Get all live entries (non-tombstoned)
        let live_entries = self.vector_engine.get_live_entries();

        // Load embeddings from storage for live entries
        let mut entries_with_embeddings = Vec::with_capacity(live_entries.len());

        for (_idx, chunk_id) in live_entries {
            match self.store.get_embedding(chunk_id).await {
                Ok(Some(embedding)) => {
                    entries_with_embeddings.push((chunk_id, embedding));
                }
                Ok(None) => {
                    warn!(
                        "Embedding not found for chunk_id {} during compaction, skipping",
                        chunk_id.as_u64()
                    );
                }
                Err(e) => {
                    warn!(
                        "Failed to load embedding for chunk_id {} during compaction: {}",
                        chunk_id.as_u64(),
                        e
                    );
                }
            }
        }

        // Rebuild the vector index with only live entries
        // This also clears the in-memory tombstones
        let compacted_count = self.vector_engine.compact(entries_with_embeddings)?;

        info!(
            "Compaction complete: {} → {} entries ({} removed)",
            total_count,
            compacted_count,
            total_count - compacted_count
        );

        Ok(compacted_count)
    }

    /// Conditionally run compaction if needed.
    ///
    /// Only compacts if tombstone ratio exceeds 30% threshold.
    ///
    /// # Returns
    /// - `Ok(Some(count))` if compaction was performed, with the new entry count
    /// - `Ok(None)` if compaction was not needed
    pub async fn compact_if_needed(&mut self) -> Result<Option<usize>, SearchError> {
        if self.needs_compaction() {
            let count = self.compact().await?;
            Ok(Some(count))
        } else {
            Ok(None)
        }
    }
}
