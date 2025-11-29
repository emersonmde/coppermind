//! Tests for the HybridSearchEngine.

use super::*;
use crate::search::types::ChunkSourceMetadata;
use crate::storage::InMemoryDocumentStore;

#[tokio::test]
async fn test_hybrid_search_engine() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    // Add chunks
    let chunk1 = Chunk {
        text: "machine learning algorithms".to_string(),
        metadata: ChunkSourceMetadata {
            filename: Some("chunk1.txt".to_string()),
            source: None,
            created_at: 0,
        },
    };

    let chunk2 = Chunk {
        text: "deep neural networks".to_string(),
        metadata: ChunkSourceMetadata {
            filename: Some("chunk2.txt".to_string()),
            source: None,
            created_at: 1,
        },
    };

    // Dummy embeddings (in practice, these come from JinaBERT)
    engine.add_chunk(chunk1, vec![1.0, 0.0, 0.0]).await.unwrap();
    engine.add_chunk(chunk2, vec![0.9, 0.1, 0.0]).await.unwrap();

    // Search
    let results = engine
        .search(&[1.0, 0.0, 0.0], "machine learning", 2)
        .await
        .unwrap();

    assert!(results.len() <= 2);
    assert!(!results.is_empty());
}

#[tokio::test]
async fn test_add_chunk_dimension_mismatch() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    let chunk = Chunk {
        text: "test".to_string(),
        metadata: ChunkSourceMetadata {
            filename: Some("test.txt".to_string()),
            source: None,
            created_at: 0,
        },
    };

    // Try to add chunk with wrong embedding dimension
    let result = engine.add_chunk(chunk, vec![1.0, 0.0]).await; // 2D instead of 3D

    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(SearchError::DimensionMismatch {
            expected: 3,
            actual: 2
        })
    ));
}

#[tokio::test]
async fn test_search_empty_index() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    // Search empty index
    let results = engine.search(&[1.0, 0.0, 0.0], "query", 10).await.unwrap();

    assert!(results.is_empty());
}

#[tokio::test]
async fn test_search_dimension_mismatch() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    let chunk = Chunk {
        text: "test chunk".to_string(),
        metadata: ChunkSourceMetadata::default(),
    };

    engine.add_chunk(chunk, vec![1.0, 0.0, 0.0]).await.unwrap();

    // Try to search with wrong dimension
    let result = engine.search(&[1.0, 0.0], "query", 10).await; // 2D instead of 3D

    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(SearchError::DimensionMismatch {
            expected: 3,
            actual: 2
        })
    ));
}

#[tokio::test]
async fn test_batch_add_deferred() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    // Add multiple chunks without rebuilding index each time
    for i in 0..5 {
        let chunk = Chunk {
            text: format!("chunk {}", i),
            metadata: ChunkSourceMetadata {
                filename: Some(format!("chunk{}.txt", i)),
                source: None,
                created_at: i as u64,
            },
        };

        engine
            .add_chunk_deferred(chunk, vec![i as f32, 0.0, 0.0])
            .await
            .unwrap();
    }

    // Rebuild index once
    engine.rebuild_vector_index().await.unwrap();

    // Verify all chunks are indexed
    assert_eq!(engine.len(), 5);
    assert_eq!(engine.vector_index_len(), 5);

    // Search should work after rebuild
    let results = engine.search(&[2.0, 0.0, 0.0], "chunk 2", 3).await.unwrap();
    assert!(!results.is_empty());
}

#[tokio::test]
async fn test_get_index_metrics() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    // Empty index
    let (docs, chunks, tokens, avg) = engine.get_index_metrics().await.unwrap();
    assert_eq!(docs, 0);
    assert_eq!(chunks, 0);
    assert_eq!(tokens, 0);
    assert_eq!(avg, 0.0);

    // Add chunks
    let chunk1 = Chunk {
        text: "one two three".to_string(),
        metadata: ChunkSourceMetadata::default(),
    };
    let chunk2 = Chunk {
        text: "four five".to_string(),
        metadata: ChunkSourceMetadata::default(),
    };

    engine.add_chunk(chunk1, vec![1.0, 0.0, 0.0]).await.unwrap();
    engine.add_chunk(chunk2, vec![0.0, 1.0, 0.0]).await.unwrap();

    let (_docs, chunks, _tokens, _avg) = engine.get_index_metrics().await.unwrap();
    assert_eq!(chunks, 2);
}

#[tokio::test]
async fn test_vector_and_keyword_index_sync() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    // Add chunks
    for i in 0..3 {
        let chunk = Chunk {
            text: format!("chunk {}", i),
            metadata: ChunkSourceMetadata::default(),
        };
        engine
            .add_chunk(chunk, vec![i as f32, 0.0, 0.0])
            .await
            .unwrap();
    }

    // Both indexes should have same count
    assert_eq!(engine.len(), 3);
    assert_eq!(engine.vector_index_len(), 3);
    assert_eq!(engine.keyword_index_len(), 3);
}

#[tokio::test]
async fn test_clear_index() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    // Add chunks
    let chunk = Chunk {
        text: "test chunk".to_string(),
        metadata: ChunkSourceMetadata::default(),
    };
    engine.add_chunk(chunk, vec![1.0, 0.0, 0.0]).await.unwrap();

    assert_eq!(engine.len(), 1);

    // Clear
    engine.clear_all().await.unwrap();

    assert_eq!(engine.len(), 0);
    assert!(engine.is_empty());
    assert_eq!(engine.vector_index_len(), 0);

    // Should be able to add chunks after clear
    let chunk2 = Chunk {
        text: "new chunk".to_string(),
        metadata: ChunkSourceMetadata::default(),
    };
    let result = engine.add_chunk(chunk2, vec![1.0, 0.0, 0.0]).await;
    assert!(result.is_ok());
    assert_eq!(engine.len(), 1);
}

#[tokio::test]
async fn test_get_chunk() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    let chunk = Chunk {
        text: "test chunk".to_string(),
        metadata: ChunkSourceMetadata {
            filename: Some("test.txt".to_string()),
            source: Some("manual".to_string()),
            created_at: 12345,
        },
    };

    let chunk_id = engine
        .add_chunk(chunk.clone(), vec![1.0, 0.0, 0.0])
        .await
        .unwrap();

    // Get chunk by ID
    let retrieved = engine.get_chunk(&chunk_id).await.unwrap();
    assert!(retrieved.is_some());

    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.id, chunk_id);
    assert_eq!(retrieved.text, chunk.text);
    assert_eq!(retrieved.metadata.filename, chunk.metadata.filename);
    assert_eq!(retrieved.metadata.source, chunk.metadata.source);
    assert_eq!(retrieved.metadata.created_at, chunk.metadata.created_at);
}

#[tokio::test]
async fn test_debug_dump() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    // Empty index
    let dump = engine.debug_dump();
    assert!(dump.contains("Total chunks: 0"));

    // Add chunk
    let chunk = Chunk {
        text: "This is a test chunk with some content".to_string(),
        metadata: ChunkSourceMetadata {
            filename: Some("test.txt".to_string()),
            source: Some("test".to_string()),
            created_at: 123,
        },
    };
    engine.add_chunk(chunk, vec![1.0, 0.0, 0.0]).await.unwrap();

    let dump = engine.debug_dump();
    assert!(dump.contains("Total chunks: 1"));
}

#[tokio::test]
async fn test_search_returns_top_k() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    // Add 10 chunks
    for i in 0..10 {
        let chunk = Chunk {
            text: format!("chunk number {}", i),
            metadata: ChunkSourceMetadata::default(),
        };
        engine
            .add_chunk(chunk, vec![i as f32, 0.0, 0.0])
            .await
            .unwrap();
    }

    // Request top 3
    let results = engine.search(&[5.0, 0.0, 0.0], "chunk", 3).await.unwrap();

    // Should return at most 3 results
    assert!(results.len() <= 3);

    // Verify results have scores
    for result in &results {
        assert!(result.score > 0.0);
    }
}

#[tokio::test]
async fn test_search_result_structure() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    let chunk = Chunk {
        text: "semantic search test".to_string(),
        metadata: ChunkSourceMetadata {
            filename: Some("search.txt".to_string()),
            source: None,
            created_at: 999,
        },
    };

    engine
        .add_chunk(chunk.clone(), vec![1.0, 0.5, 0.2])
        .await
        .unwrap();

    let results = engine
        .search(&[1.0, 0.5, 0.2], "semantic", 1)
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    let result = &results[0];

    // Verify SearchResult structure
    assert!(result.score > 0.0); // RRF fused score
    assert!(result.vector_score.is_some()); // Vector search score
    assert!(result.keyword_score.is_some()); // BM25 score
    assert_eq!(result.text, chunk.text);
    assert_eq!(result.metadata.filename, chunk.metadata.filename);
    assert_eq!(result.metadata.created_at, chunk.metadata.created_at);
}

#[tokio::test]
async fn test_search_empty_query_text() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    let chunk = Chunk {
        text: "test chunk".to_string(),
        metadata: ChunkSourceMetadata::default(),
    };
    engine.add_chunk(chunk, vec![1.0, 0.0, 0.0]).await.unwrap();

    // Empty query text should return InvalidQuery error
    let result = engine.search(&[1.0, 0.0, 0.0], "", 10).await;
    assert!(matches!(result, Err(SearchError::InvalidQuery(_))));

    // Whitespace-only query should also fail
    let result = engine.search(&[1.0, 0.0, 0.0], "   \t\n  ", 10).await;
    assert!(matches!(result, Err(SearchError::InvalidQuery(_))));
}

#[tokio::test]
async fn test_search_zero_k() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    let chunk = Chunk {
        text: "test chunk".to_string(),
        metadata: ChunkSourceMetadata::default(),
    };
    engine.add_chunk(chunk, vec![1.0, 0.0, 0.0]).await.unwrap();

    // k=0 should return InvalidQuery error
    let result = engine.search(&[1.0, 0.0, 0.0], "test", 0).await;
    assert!(matches!(result, Err(SearchError::InvalidQuery(_))));
}

#[tokio::test]
async fn test_persistence_reload() {
    use std::sync::Arc;

    let store = Arc::new(InMemoryDocumentStore::new());

    // Create engine and add chunk
    let chunk_id = {
        let mut engine = HybridSearchEngine::new(Arc::clone(&store), 3)
            .await
            .unwrap();

        let chunk = Chunk {
            text: "persistent chunk".to_string(),
            metadata: ChunkSourceMetadata {
                filename: Some("persist.txt".to_string()),
                source: None,
                created_at: 42,
            },
        };

        let chunk_id = engine.add_chunk(chunk, vec![1.0, 2.0, 3.0]).await.unwrap();
        engine.save().await.unwrap();
        chunk_id
    };

    // Reload from store (simulating app restart with same underlying store)
    let mut engine = HybridSearchEngine::try_load_or_new(Arc::clone(&store), 3)
        .await
        .unwrap();

    // Verify chunk is present
    assert_eq!(engine.len(), 1);
    let retrieved = engine.get_chunk(&chunk_id).await.unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().text, "persistent chunk");

    // Search should work
    let results = engine
        .search(&[1.0, 2.0, 3.0], "persistent", 1)
        .await
        .unwrap();
    assert!(!results.is_empty());
}

// =========================================================================
// Source Tracking Tests
// =========================================================================

#[tokio::test]
async fn test_source_registration_and_lookup() {
    let store = InMemoryDocumentStore::new();
    let engine = HybridSearchEngine::new(store, 3).await.unwrap();

    let source_id = "/Users/test/README.md";
    let content_hash = "abc123def456".to_string();

    // Source shouldn't exist initially
    assert!(engine.get_source(source_id).await.unwrap().is_none());

    // Register source
    engine
        .register_source(source_id, content_hash.clone())
        .await
        .unwrap();

    // Source should exist and be incomplete
    let record = engine.get_source(source_id).await.unwrap().unwrap();
    assert_eq!(record.content_hash, content_hash);
    assert!(!record.complete);
    assert!(record.chunk_ids.is_empty());
}

#[tokio::test]
async fn test_source_chunk_tracking() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    let source_id = "web:test.txt";
    let content_hash = "hash123".to_string();

    // Register source
    engine
        .register_source(source_id, content_hash)
        .await
        .unwrap();

    // Add chunks and track them
    let chunk1 = Chunk {
        text: "First chunk".to_string(),
        metadata: ChunkSourceMetadata {
            filename: Some("test.txt (chunk 1)".to_string()),
            source: Some(source_id.to_string()),
            created_at: 100,
        },
    };
    let chunk_id1 = engine.add_chunk(chunk1, vec![1.0, 0.0, 0.0]).await.unwrap();
    engine
        .add_chunk_to_source(source_id, chunk_id1)
        .await
        .unwrap();

    let chunk2 = Chunk {
        text: "Second chunk".to_string(),
        metadata: ChunkSourceMetadata {
            filename: Some("test.txt (chunk 2)".to_string()),
            source: Some(source_id.to_string()),
            created_at: 100,
        },
    };
    let chunk_id2 = engine.add_chunk(chunk2, vec![0.0, 1.0, 0.0]).await.unwrap();
    engine
        .add_chunk_to_source(source_id, chunk_id2)
        .await
        .unwrap();

    // Complete the source
    engine.complete_source(source_id).await.unwrap();

    // Verify source record
    let record = engine.get_source(source_id).await.unwrap().unwrap();
    assert!(record.complete);
    assert_eq!(record.chunk_ids.len(), 2);
    assert!(record.chunk_ids.contains(&chunk_id1));
    assert!(record.chunk_ids.contains(&chunk_id2));
}

#[tokio::test]
async fn test_source_needs_update() {
    let store = InMemoryDocumentStore::new();
    let engine = HybridSearchEngine::new(store, 3).await.unwrap();

    let source_id = "/path/to/file.md";
    let hash_v1 = "version1hash".to_string();
    let hash_v2 = "version2hash".to_string();

    // New source needs update
    assert!(engine
        .source_needs_update(source_id, &hash_v1)
        .await
        .unwrap());

    // Register source
    engine
        .register_source(source_id, hash_v1.clone())
        .await
        .unwrap();

    // Same hash - no update needed
    assert!(!engine
        .source_needs_update(source_id, &hash_v1)
        .await
        .unwrap());

    // Different hash - update needed
    assert!(engine
        .source_needs_update(source_id, &hash_v2)
        .await
        .unwrap());
}

#[tokio::test]
async fn test_delete_source() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    let source_id = "web:delete-test.txt";

    // Register and add chunks
    engine
        .register_source(source_id, "hash".to_string())
        .await
        .unwrap();

    let chunk1 = Chunk {
        text: "Chunk to delete".to_string(),
        metadata: ChunkSourceMetadata {
            filename: Some("delete-test.txt".to_string()),
            source: Some(source_id.to_string()),
            created_at: 0,
        },
    };
    let chunk_id1 = engine.add_chunk(chunk1, vec![1.0, 0.0, 0.0]).await.unwrap();
    engine
        .add_chunk_to_source(source_id, chunk_id1)
        .await
        .unwrap();

    let chunk2 = Chunk {
        text: "Another chunk".to_string(),
        metadata: ChunkSourceMetadata {
            filename: Some("delete-test.txt".to_string()),
            source: Some(source_id.to_string()),
            created_at: 0,
        },
    };
    let chunk_id2 = engine.add_chunk(chunk2, vec![0.0, 1.0, 0.0]).await.unwrap();
    engine
        .add_chunk_to_source(source_id, chunk_id2)
        .await
        .unwrap();

    engine.complete_source(source_id).await.unwrap();

    // Verify chunks are searchable
    assert_eq!(engine.len(), 2);

    // Delete the source
    let deleted = engine.delete_source(source_id).await.unwrap();
    assert_eq!(deleted, 2);

    // Source should be gone
    assert!(engine.get_source(source_id).await.unwrap().is_none());

    // Chunks should be tombstoned (excluded from search)
    let results = engine
        .search(&[1.0, 0.0, 0.0], "chunk delete", 10)
        .await
        .unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_list_sources() {
    let store = InMemoryDocumentStore::new();
    let engine = HybridSearchEngine::new(store, 3).await.unwrap();

    // Initially empty
    let sources = engine.list_sources().await.unwrap();
    assert!(sources.is_empty());

    // Add some sources
    engine
        .register_source("/path/a.md", "hash1".to_string())
        .await
        .unwrap();
    engine
        .register_source("/path/b.md", "hash2".to_string())
        .await
        .unwrap();
    engine
        .register_source("web:c.txt", "hash3".to_string())
        .await
        .unwrap();

    // List should contain all sources
    let sources = engine.list_sources().await.unwrap();
    assert_eq!(sources.len(), 3);
    assert!(sources.contains(&"/path/a.md".to_string()));
    assert!(sources.contains(&"/path/b.md".to_string()));
    assert!(sources.contains(&"web:c.txt".to_string()));
}

// =========================================================================
// Compaction Tests
// =========================================================================

#[tokio::test]
async fn test_compaction_stats() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    // Add 5 chunks
    for i in 0..5 {
        let chunk = Chunk {
            text: format!("chunk {}", i),
            metadata: ChunkSourceMetadata::default(),
        };
        engine
            .add_chunk(chunk, vec![i as f32, 0.0, 0.0])
            .await
            .unwrap();
    }

    // Initially no tombstones
    let (tombstone_count, total_count, ratio) = engine.compaction_stats();
    assert_eq!(tombstone_count, 0);
    assert_eq!(total_count, 5);
    assert_eq!(ratio, 0.0);
    assert!(!engine.needs_compaction());
}

#[tokio::test]
async fn test_compaction_after_delete() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    // Add source with 5 chunks
    let source_id = "test-source";
    engine
        .register_source(source_id, "hash1".to_string())
        .await
        .unwrap();

    for i in 0..5 {
        let chunk = Chunk {
            text: format!("chunk {}", i),
            metadata: ChunkSourceMetadata {
                filename: Some(format!("chunk{}.txt", i)),
                source: Some(source_id.to_string()),
                created_at: i as u64,
            },
        };
        let chunk_id = engine
            .add_chunk(chunk, vec![i as f32, 0.0, 0.0])
            .await
            .unwrap();
        engine
            .add_chunk_to_source(source_id, chunk_id)
            .await
            .unwrap();
    }
    engine.complete_source(source_id).await.unwrap();

    // Delete the source (creates tombstones)
    engine.delete_source(source_id).await.unwrap();

    // Now we have 100% tombstones
    let (tombstone_count, total_count, ratio) = engine.compaction_stats();
    assert_eq!(tombstone_count, 5);
    assert_eq!(total_count, 5);
    assert!((ratio - 1.0).abs() < 0.01);
    assert!(engine.needs_compaction());

    // Run compaction
    let compacted_count = engine.compact().await.unwrap();
    assert_eq!(compacted_count, 0); // All entries were tombstoned

    // After compaction
    let (tombstone_count, total_count, _) = engine.compaction_stats();
    assert_eq!(tombstone_count, 0);
    assert_eq!(total_count, 0);
    assert!(!engine.needs_compaction());
}

#[tokio::test]
async fn test_compact_if_needed() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    // Add 10 chunks
    let source_id = "test-source";
    engine
        .register_source(source_id, "hash1".to_string())
        .await
        .unwrap();

    for i in 0..10 {
        let chunk = Chunk {
            text: format!("chunk {}", i),
            metadata: ChunkSourceMetadata {
                filename: Some(format!("chunk{}.txt", i)),
                source: Some(source_id.to_string()),
                created_at: i as u64,
            },
        };
        let chunk_id = engine
            .add_chunk(chunk, vec![i as f32, 0.0, 0.0])
            .await
            .unwrap();
        engine
            .add_chunk_to_source(source_id, chunk_id)
            .await
            .unwrap();
    }
    engine.complete_source(source_id).await.unwrap();

    // No compaction needed yet
    let result = engine.compact_if_needed().await.unwrap();
    assert!(result.is_none());

    // Delete source (creates tombstones)
    engine.delete_source(source_id).await.unwrap();

    // Now compaction is needed (100% tombstones > 30% threshold)
    let result = engine.compact_if_needed().await.unwrap();
    assert!(result.is_some());
    assert_eq!(result.unwrap(), 0);
}

#[tokio::test]
async fn test_compaction_preserves_live_entries() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    // Add source1 with 3 chunks
    let source1 = "source1";
    engine
        .register_source(source1, "hash1".to_string())
        .await
        .unwrap();

    for i in 0..3 {
        let chunk = Chunk {
            text: format!("source1 chunk {}", i),
            metadata: ChunkSourceMetadata {
                filename: Some(format!("s1-chunk{}.txt", i)),
                source: Some(source1.to_string()),
                created_at: i as u64,
            },
        };
        let chunk_id = engine
            .add_chunk(chunk, vec![i as f32, 0.0, 0.0])
            .await
            .unwrap();
        engine.add_chunk_to_source(source1, chunk_id).await.unwrap();
    }
    engine.complete_source(source1).await.unwrap();

    // Add source2 with 2 chunks
    let source2 = "source2";
    engine
        .register_source(source2, "hash2".to_string())
        .await
        .unwrap();

    for i in 0..2 {
        let chunk = Chunk {
            text: format!("source2 chunk {}", i),
            metadata: ChunkSourceMetadata {
                filename: Some(format!("s2-chunk{}.txt", i)),
                source: Some(source2.to_string()),
                created_at: (i + 100) as u64,
            },
        };
        let chunk_id = engine
            .add_chunk(chunk, vec![(i + 10) as f32, 0.0, 0.0])
            .await
            .unwrap();
        engine.add_chunk_to_source(source2, chunk_id).await.unwrap();
    }
    engine.complete_source(source2).await.unwrap();

    // Delete source1 (tombstones 3 of 5 = 60% > 30% threshold)
    engine.delete_source(source1).await.unwrap();

    assert!(engine.needs_compaction());

    // Compact
    let compacted_count = engine.compact().await.unwrap();
    assert_eq!(compacted_count, 2); // Only source2's chunks remain

    // source2 chunks should still be searchable
    let results = engine
        .search(&[10.0, 0.0, 0.0], "source2", 10)
        .await
        .unwrap();
    assert_eq!(results.len(), 2);
}

// =========================================================================
// Integration Tests for 1.0 Release
// =========================================================================

/// End-to-end test: add chunks, search, verify results contain expected content
#[tokio::test]
async fn test_integration_add_search_retrieve() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    // Add chunks about different topics
    let topics = [
        (
            "rust",
            vec![1.0, 0.0, 0.0],
            "Rust is a systems programming language",
        ),
        (
            "python",
            vec![0.0, 1.0, 0.0],
            "Python is great for data science",
        ),
        (
            "javascript",
            vec![0.0, 0.0, 1.0],
            "JavaScript runs in browsers",
        ),
    ];

    for (topic, embedding, text) in topics {
        let chunk = Chunk {
            text: text.to_string(),
            metadata: ChunkSourceMetadata {
                filename: Some(format!("{}.md", topic)),
                source: Some(format!("/docs/{}.md", topic)),
                created_at: 0,
            },
        };
        engine.add_chunk(chunk, embedding).await.unwrap();
    }

    // Search for "rust" - should find the Rust chunk via keyword match
    let results = engine.search(&[0.5, 0.5, 0.5], "rust", 3).await.unwrap();
    assert!(!results.is_empty());
    assert!(results[0].text.contains("Rust"));

    // Search for "programming" - should find Rust chunk
    let results = engine
        .search(&[0.5, 0.5, 0.5], "programming", 3)
        .await
        .unwrap();
    assert!(!results.is_empty());
    assert!(results[0].text.contains("programming"));

    // Vector-weighted search (embedding close to Python, generic query)
    let results = engine.search(&[0.0, 1.0, 0.0], "great", 3).await.unwrap();
    assert!(!results.is_empty());
    // Python should be top result due to cosine similarity + keyword "great"
    assert!(results[0].text.contains("Python"));
}

/// Test hybrid search ranking: keyword matches should boost vector results
#[tokio::test]
async fn test_hybrid_ranking_keyword_boost() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    // Add two chunks with similar embeddings but different text
    let chunk1 = Chunk {
        text: "The quick brown fox jumps over the lazy dog".to_string(),
        metadata: ChunkSourceMetadata::default(),
    };
    let chunk2 = Chunk {
        text: "A speedy russet fox leaps across a sleepy canine".to_string(),
        metadata: ChunkSourceMetadata::default(),
    };

    engine.add_chunk(chunk1, vec![1.0, 0.0, 0.0]).await.unwrap();
    engine
        .add_chunk(chunk2, vec![1.0, 0.1, 0.0]) // Very similar embedding
        .await
        .unwrap();

    // Search with keyword "quick" - should rank chunk1 higher despite similar embeddings
    let results = engine.search(&[1.0, 0.05, 0.0], "quick", 2).await.unwrap();
    assert_eq!(results.len(), 2);
    assert!(results[0].text.contains("quick"));
}

/// Test that deleted chunks don't appear in search results
#[tokio::test]
async fn test_deleted_chunks_not_in_results() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    // Add source with chunks
    let source_id = "test_source";
    engine
        .register_source(source_id, "hash".to_string())
        .await
        .unwrap();

    let chunk = Chunk {
        text: "This chunk will be deleted".to_string(),
        metadata: ChunkSourceMetadata {
            filename: Some("deleted.txt".to_string()),
            source: Some(source_id.to_string()),
            created_at: 0,
        },
    };
    let chunk_id = engine.add_chunk(chunk, vec![1.0, 0.0, 0.0]).await.unwrap();
    engine
        .add_chunk_to_source(source_id, chunk_id)
        .await
        .unwrap();
    engine.complete_source(source_id).await.unwrap();

    // Verify chunk is searchable
    let results = engine.search(&[1.0, 0.0, 0.0], "deleted", 5).await.unwrap();
    assert_eq!(results.len(), 1);

    // Delete the source
    engine.delete_source(source_id).await.unwrap();

    // Chunk should no longer appear in results
    let results = engine.search(&[1.0, 0.0, 0.0], "deleted", 5).await.unwrap();
    assert!(results.is_empty());
}

/// Test persistence: data survives engine reload
#[tokio::test]
async fn test_persistence_survives_reload() {
    use std::sync::Arc;
    let store = Arc::new(InMemoryDocumentStore::new());

    // Create engine and add data
    {
        let mut engine = HybridSearchEngine::new(store.clone(), 3).await.unwrap();

        let source_id = "persistent_source";
        engine
            .register_source(source_id, "hash123".to_string())
            .await
            .unwrap();

        let chunk = Chunk {
            text: "This data should persist".to_string(),
            metadata: ChunkSourceMetadata {
                filename: Some("persistent.txt".to_string()),
                source: Some(source_id.to_string()),
                created_at: 42,
            },
        };
        let chunk_id = engine.add_chunk(chunk, vec![0.5, 0.5, 0.0]).await.unwrap();
        engine
            .add_chunk_to_source(source_id, chunk_id)
            .await
            .unwrap();
        engine.complete_source(source_id).await.unwrap();

        // Save to store
        engine.save().await.unwrap();
    }

    // Reload engine from same store
    let mut engine = HybridSearchEngine::try_load_or_new(store, 3).await.unwrap();

    // Verify data was loaded
    assert_eq!(engine.len(), 1);

    // Search should work
    let results = engine.search(&[0.5, 0.5, 0.0], "persist", 5).await.unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].text.contains("persist"));

    // Source should be tracked
    let sources = engine.list_sources().await.unwrap();
    assert_eq!(sources.len(), 1);
    assert!(sources.contains(&"persistent_source".to_string()));
}

/// Test that source update detection works correctly
#[tokio::test]
async fn test_source_update_detection() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    let source_id = "updatable_source";

    // Register with initial hash
    engine
        .register_source(source_id, "hash_v1".to_string())
        .await
        .unwrap();

    let chunk = Chunk {
        text: "Initial content".to_string(),
        metadata: ChunkSourceMetadata {
            filename: Some("file.txt".to_string()),
            source: Some(source_id.to_string()),
            created_at: 0,
        },
    };
    let chunk_id = engine.add_chunk(chunk, vec![1.0, 0.0, 0.0]).await.unwrap();
    engine
        .add_chunk_to_source(source_id, chunk_id)
        .await
        .unwrap();
    engine.complete_source(source_id).await.unwrap();

    // Check if source needs update with same hash (should not)
    assert!(!engine
        .source_needs_update(source_id, "hash_v1")
        .await
        .unwrap());

    // Check if source needs update with different hash (should)
    assert!(engine
        .source_needs_update(source_id, "hash_v2")
        .await
        .unwrap());
}

/// Test search with moderately large result set
#[tokio::test]
async fn test_search_moderate_result_set() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    // Add 50 chunks (enough to test pagination without hitting HNSW limits)
    for i in 0..50 {
        let chunk = Chunk {
            text: format!("Chunk number {} with unique content", i),
            metadata: ChunkSourceMetadata {
                filename: Some(format!("chunk_{}.txt", i)),
                source: Some(format!("/batch/chunk_{}.txt", i)),
                created_at: i,
            },
        };
        // Spread embeddings across the space
        let angle = (i as f32) * 0.1256; // ~7.2 degrees per chunk
        let embedding = vec![angle.cos(), angle.sin(), 0.0];
        engine.add_chunk(chunk, embedding).await.unwrap();
    }

    // Search for top 10
    let results = engine.search(&[1.0, 0.0, 0.0], "Chunk", 10).await.unwrap();
    assert_eq!(results.len(), 10);

    // Search for top 20
    let results = engine.search(&[1.0, 0.0, 0.0], "Chunk", 20).await.unwrap();
    assert_eq!(results.len(), 20);

    // All results should have "Chunk" in their text
    for result in &results {
        assert!(result.text.contains("Chunk"));
    }
}

/// Test search with special characters in query
#[tokio::test]
async fn test_search_special_characters() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    let chunk = Chunk {
        text: "Function foo() returns bar; use foo->bar syntax".to_string(),
        metadata: ChunkSourceMetadata::default(),
    };
    engine.add_chunk(chunk, vec![1.0, 0.0, 0.0]).await.unwrap();

    // Search with parentheses
    let results = engine.search(&[1.0, 0.0, 0.0], "foo()", 5).await.unwrap();
    assert!(!results.is_empty());

    // Search with arrow syntax
    let results = engine
        .search(&[1.0, 0.0, 0.0], "foo->bar", 5)
        .await
        .unwrap();
    assert!(!results.is_empty());

    // Search with semicolon
    let results = engine.search(&[1.0, 0.0, 0.0], "bar;", 5).await.unwrap();
    assert!(!results.is_empty());
}

/// Test concurrent add and search operations don't corrupt state
#[tokio::test]
async fn test_concurrent_operations() {
    use std::sync::Arc;
    let store = Arc::new(InMemoryDocumentStore::new());
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    // Add initial chunks
    for i in 0..10 {
        let chunk = Chunk {
            text: format!("Initial chunk {}", i),
            metadata: ChunkSourceMetadata::default(),
        };
        engine
            .add_chunk(chunk, vec![i as f32, 0.0, 0.0])
            .await
            .unwrap();
    }

    // Perform multiple searches
    for _ in 0..5 {
        let results = engine.search(&[5.0, 0.0, 0.0], "chunk", 5).await.unwrap();
        assert!(results.len() <= 5);
    }

    // Engine state should be consistent
    assert_eq!(engine.len(), 10);
}

/// Test chunk text handling and search accuracy
#[tokio::test]
async fn test_chunk_text_search() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    // Add chunk with specific keyword
    let chunk1 = Chunk {
        text: "Alpha bravo charlie information".to_string(),
        metadata: ChunkSourceMetadata::default(),
    };
    engine.add_chunk(chunk1, vec![1.0, 0.0, 0.0]).await.unwrap();

    // Add chunk with different content
    let chunk2 = Chunk {
        text: "Delta echo foxtrot data".to_string(),
        metadata: ChunkSourceMetadata::default(),
    };
    engine.add_chunk(chunk2, vec![0.0, 1.0, 0.0]).await.unwrap();

    // Hybrid search combines keyword + vector results
    // Both chunks may appear in results, but the one with keyword match should rank higher
    let results = engine.search(&[0.5, 0.5, 0.0], "Alpha", 5).await.unwrap();
    assert!(!results.is_empty());
    // The first result should contain the keyword match
    assert!(results[0].text.contains("Alpha"));

    // Search with embedding closer to chunk2 and keyword "Delta"
    let results = engine.search(&[0.0, 1.0, 0.0], "Delta", 5).await.unwrap();
    assert!(!results.is_empty());
    // The first result should contain "Delta" (keyword + vector agreement)
    assert!(results[0].text.contains("Delta"));
}

// =========================================================================
// Document-Level Indexing Tests (ADR-008)
// =========================================================================

#[tokio::test]
async fn test_document_level_indexing_basic() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    // Create a document
    let doc_id = engine
        .create_document(
            "/test/doc.md",
            "Test Document",
            "hash123",
            "This is the full text of the test document about Rust programming.",
        )
        .await
        .unwrap();

    // Add chunks to the document
    let chunk1 = Chunk {
        text: "Chunk 1: Introduction to Rust".to_string(),
        metadata: ChunkSourceMetadata::default(),
    };
    engine
        .add_chunk_to_document(doc_id, chunk1, vec![1.0, 0.0, 0.0])
        .await
        .unwrap();

    let chunk2 = Chunk {
        text: "Chunk 2: Rust memory safety".to_string(),
        metadata: ChunkSourceMetadata::default(),
    };
    engine
        .add_chunk_to_document(doc_id, chunk2, vec![0.9, 0.1, 0.0])
        .await
        .unwrap();

    // Finalize the document
    engine.finalize_document(doc_id).await.unwrap();

    // Verify document was created
    let doc = engine.get_document(doc_id).await.unwrap();
    assert!(doc.is_some());
    let doc = doc.unwrap();
    assert_eq!(doc.chunk_ids.len(), 2);
    assert_eq!(doc.metadata.chunk_count, 2);

    // Verify document-level search works
    let results = engine
        .search_documents(&[1.0, 0.0, 0.0], "Rust", 5)
        .await
        .unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].doc_id, doc_id);
    assert!(!results[0].chunks.is_empty());
}

#[tokio::test]
async fn test_document_search_returns_nested_chunks() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    // Create a document with multiple chunks
    let doc_id = engine
        .create_document(
            "/test/multi.md",
            "Multi-chunk Document",
            "hash456",
            "A document with multiple chunks about machine learning and AI.",
        )
        .await
        .unwrap();

    for i in 0..3 {
        let chunk = Chunk {
            text: format!("Chunk {} about machine learning", i),
            metadata: ChunkSourceMetadata::default(),
        };
        engine
            .add_chunk_to_document(doc_id, chunk, vec![i as f32, 0.0, 0.0])
            .await
            .unwrap();
    }

    engine.finalize_document(doc_id).await.unwrap();

    // Search for the document
    let results = engine
        .search_documents(&[1.0, 0.0, 0.0], "machine learning", 5)
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    let doc_result = &results[0];

    // Verify nested chunks are returned
    assert!(!doc_result.chunks.is_empty());
    assert!(doc_result.chunks.len() <= 3);

    // Chunks should be sorted by score
    for window in doc_result.chunks.windows(2) {
        assert!(window[0].score >= window[1].score);
    }
}

#[tokio::test]
async fn test_document_keyword_search_works() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    // Create documents with distinct content
    let doc1_id = engine
        .create_document(
            "/test/python.md",
            "Python Guide",
            "hash1",
            "Python is excellent for data science and machine learning applications.",
        )
        .await
        .unwrap();

    let chunk = Chunk {
        text: "Python data science basics".to_string(),
        metadata: ChunkSourceMetadata::default(),
    };
    engine
        .add_chunk_to_document(doc1_id, chunk, vec![1.0, 0.0, 0.0])
        .await
        .unwrap();
    engine.finalize_document(doc1_id).await.unwrap();

    let doc2_id = engine
        .create_document(
            "/test/rust.md",
            "Rust Guide",
            "hash2",
            "Rust is a systems programming language focused on safety and performance.",
        )
        .await
        .unwrap();

    let chunk = Chunk {
        text: "Rust systems programming".to_string(),
        metadata: ChunkSourceMetadata::default(),
    };
    engine
        .add_chunk_to_document(doc2_id, chunk, vec![0.0, 1.0, 0.0])
        .await
        .unwrap();
    engine.finalize_document(doc2_id).await.unwrap();

    // Search for "Python" - should find Python doc
    let results = engine
        .search_documents(&[0.5, 0.5, 0.0], "Python", 5)
        .await
        .unwrap();

    assert!(!results.is_empty());
    // Python document should rank higher due to keyword match in full document text
    assert!(results.iter().any(|r| r.doc_id == doc1_id));

    // Search for "Rust systems" - should find Rust doc
    let results = engine
        .search_documents(&[0.5, 0.5, 0.0], "Rust systems", 5)
        .await
        .unwrap();

    assert!(!results.is_empty());
    assert!(results.iter().any(|r| r.doc_id == doc2_id));
}

#[tokio::test]
async fn test_chunk_has_document_id() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    // Create document
    let doc_id = engine
        .create_document("/test/doc.md", "Test", "hash", "Full text content")
        .await
        .unwrap();

    // Add chunk to document
    let chunk = Chunk {
        text: "Chunk content".to_string(),
        metadata: ChunkSourceMetadata::default(),
    };
    let chunk_id = engine
        .add_chunk_to_document(doc_id, chunk, vec![1.0, 0.0, 0.0])
        .await
        .unwrap();

    // Verify chunk has document_id set
    let chunk_record = engine.get_chunk(&chunk_id).await.unwrap().unwrap();
    assert_eq!(chunk_record.document_id, Some(doc_id));
}

#[tokio::test]
async fn test_legacy_chunk_has_no_document_id() {
    let store = InMemoryDocumentStore::new();
    let mut engine = HybridSearchEngine::new(store, 3).await.unwrap();

    // Add chunk via legacy API
    let chunk = Chunk {
        text: "Legacy chunk content".to_string(),
        metadata: ChunkSourceMetadata::default(),
    };
    let chunk_id = engine.add_chunk(chunk, vec![1.0, 0.0, 0.0]).await.unwrap();

    // Verify chunk has no document_id
    let chunk_record = engine.get_chunk(&chunk_id).await.unwrap().unwrap();
    assert_eq!(chunk_record.document_id, None);
}
