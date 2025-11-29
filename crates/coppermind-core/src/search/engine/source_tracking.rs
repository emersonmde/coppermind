//! Source tracking for re-upload detection.
//!
//! This module provides methods for tracking which sources (files, URLs) have been indexed,
//! allowing the system to detect when a source has been modified and needs re-indexing.
//!
//! # Source ID Conventions
//!
//! Source IDs are platform-specific:
//! - **Desktop**: Full file path (e.g., `/Users/matt/docs/README.md`)
//! - **Web**: `web:{filename}` (e.g., `web:README.md`)
//! - **Crawler**: Full URL (e.g., `https://example.com/docs/intro`)

use super::HybridSearchEngine;
use crate::search::types::{ChunkId, SearchError, SourceRecord};
use crate::storage::DocumentStore;
use tracing::{debug, info, warn};

impl<S: DocumentStore> HybridSearchEngine<S> {
    /// Get a source record by its source_id.
    ///
    /// source_id is platform-specific:
    /// - Desktop: full file path (e.g., "/Users/matt/docs/README.md")
    /// - Web: "web:{filename}" (e.g., "web:README.md")
    /// - Crawler: full URL (e.g., `https://example.com/docs/intro`)
    ///
    /// Returns `Ok(None)` if the source doesn't exist.
    pub async fn get_source(&self, source_id: &str) -> Result<Option<SourceRecord>, SearchError> {
        self.store
            .get_source(source_id)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))
    }

    /// Register a new source before indexing begins.
    ///
    /// Creates an incomplete source record with the given content hash.
    /// Call `add_chunk_to_source` for each chunk, then `complete_source` when done.
    ///
    /// # Arguments
    /// * `source_id` - Unique identifier for the source (path, URL, etc.)
    /// * `content_hash` - SHA-256 hash of the source content
    pub async fn register_source(
        &self,
        source_id: &str,
        content_hash: String,
    ) -> Result<(), SearchError> {
        let record = SourceRecord::new_incomplete(content_hash);
        self.store
            .put_source(source_id, &record)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;
        info!("Registered source: {}", source_id);
        Ok(())
    }

    /// Add a chunk ID to a source's chunk list.
    ///
    /// Call this after adding each chunk from a source.
    pub async fn add_chunk_to_source(
        &self,
        source_id: &str,
        chunk_id: ChunkId,
    ) -> Result<(), SearchError> {
        let mut record = self
            .store
            .get_source(source_id)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?
            .ok_or(SearchError::NotFound)?;

        record.chunk_ids.push(chunk_id);

        self.store
            .put_source(source_id, &record)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;

        Ok(())
    }

    /// Mark a source as completely indexed.
    ///
    /// Call this after all chunks from the source have been added.
    pub async fn complete_source(&self, source_id: &str) -> Result<(), SearchError> {
        let mut record = self
            .store
            .get_source(source_id)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?
            .ok_or(SearchError::NotFound)?;

        record.mark_complete();

        self.store
            .put_source(source_id, &record)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;

        info!(
            "Completed source: {} ({} chunks)",
            source_id,
            record.chunk_ids.len()
        );
        Ok(())
    }

    /// Delete all documents from a source (soft delete via tombstones).
    ///
    /// This marks all chunks from the source as tombstoned in the vector index
    /// and deletes the documents/embeddings from storage. The source record
    /// is also deleted.
    ///
    /// # Returns
    /// Number of chunks tombstoned.
    pub async fn delete_source(&mut self, source_id: &str) -> Result<usize, SearchError> {
        // Get the source record
        let record = match self.store.get_source(source_id).await {
            Ok(Some(r)) => r,
            Ok(None) => {
                warn!("Source not found for deletion: {}", source_id);
                return Ok(0);
            }
            Err(e) => return Err(SearchError::StorageError(e.to_string())),
        };

        let chunk_count = record.chunk_ids.len();

        // Tombstone each chunk in the vector index
        for chunk_id in &record.chunk_ids {
            // Find the index position for this chunk_id
            if let Some(idx) = self.vector_engine.find_index(*chunk_id) {
                self.vector_engine.mark_tombstone(idx);
            }

            // Delete from storage (chunks and embeddings)
            let _ = self.store.delete_chunk(*chunk_id).await;
            let _ = self.store.delete_embedding(*chunk_id).await;
        }

        // Delete the source record
        self.store
            .delete_source(source_id)
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))?;

        // Note: Tombstones are session-local only (not persisted).
        // Documents/embeddings are deleted from storage immediately above,
        // so on restart the index rebuilds without them.

        // Update manifest
        self.manifest.chunk_count = self.manifest.chunk_count.saturating_sub(chunk_count);

        info!(
            "Deleted source: {} ({} chunks tombstoned)",
            source_id, chunk_count
        );
        debug!(
            "Tombstone count after deletion: {}/{}",
            self.vector_engine.tombstone_count(),
            self.vector_engine.len()
        );
        Ok(chunk_count)
    }

    /// List all tracked source IDs.
    pub async fn list_sources(&self) -> Result<Vec<String>, SearchError> {
        self.store
            .list_sources()
            .await
            .map_err(|e| SearchError::StorageError(e.to_string()))
    }

    /// Check if a source has changed by comparing content hashes.
    ///
    /// # Returns
    /// - `Ok(true)` if the source is new or has changed
    /// - `Ok(false)` if the source exists with the same hash
    pub async fn source_needs_update(
        &self,
        source_id: &str,
        new_content_hash: &str,
    ) -> Result<bool, SearchError> {
        match self.get_source(source_id).await? {
            Some(record) => Ok(record.content_hash != new_content_hash),
            None => Ok(true), // New source
        }
    }
}
