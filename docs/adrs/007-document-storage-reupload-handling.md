# ADR 007: Document Storage and Re-Upload Handling

**Status**: Accepted
**Date**: 2025-11-26
**Context**: Scalable document storage with efficient re-upload detection and update handling

---

## Context and Problem Statement

Coppermind needs persistent storage for indexed documents that:
1. Scales to hundreds of thousands or millions of chunks
2. Detects when documents are re-uploaded (modified content)
3. Efficiently updates indices without full rebuilds
4. Works across platforms (desktop and web)
5. Keeps search responsive during updates

### Current Limitations

The existing architecture has several scalability issues:
- `documents: HashMap<DocId, DocumentRecord>` - entire corpus loaded in memory
- `embeddings: HashMap<DocId, Vec<f32>>` - all embeddings in memory
- No mechanism to detect re-uploaded files
- Full index rebuilds for any structural changes
- JSONL scanning would be O(n) for random access

### Requirements

1. **O(log n) or O(1) document lookup** - no full-file scanning
2. **Incremental writes** - no full-corpus serialization on save
3. **Platform-specific solutions** - different storage APIs for desktop vs web
4. **Source tracking** - know which chunks belong to which file/URL
5. **Soft deletion** - mark deleted without rebuilding HNSW graph
6. **Background compaction** - reclaim space when tombstone ratio high

---

## Decision

### 1. Platform-Specific Embedded KV Stores

Replace blob-based `StorageBackend` with proper KV stores:

| Platform | Store | Characteristics |
|----------|-------|-----------------|
| Desktop | [redb](https://github.com/cberner/redb) | Pure Rust B-tree, ACID, O(log n) |
| Web | [IndexedDB via rexie](https://github.com/devashishdxt/rexie) | Browser-native, O(1) key lookups |

Both implement a common `DocumentStore` trait with tables/stores for:
- `documents` - DocId → DocumentRecord
- `embeddings` - DocId → Vec<f32>
- `sources` - source_id → SourceRecord
- `metadata` - config and tombstone state

### 2. Tombstone-Based HNSW Deletion

Since rust-cv/hnsw doesn't support deletion, we implement soft-delete:

```rust
pub struct VectorSearchEngine {
    index: Hnsw<...>,
    doc_ids: Vec<DocId>,
    doc_id_to_idx: HashMap<DocId, usize>,  // O(1) reverse lookup
    tombstones: HashSet<usize>,            // Deleted indices
}
```

- `mark_deleted(doc_id)` - O(1) tombstone marking
- Search filters tombstones with oversampling
- Background compaction when tombstone ratio > 30%

### 3. Source Identity and Update Detection

**Core logic:**
```
source_id exists + hash matches    → SKIP (unchanged)
source_id exists + hash differs    → UPDATE (delete old, re-add)
source_id not found                → ADD (new source)
```

**Platform-specific source_id:**

| Platform | source_id Format | Example |
|----------|------------------|---------|
| Desktop | Full file path | `/Users/matt/docs/README.md` |
| Web | `web:{filename}` | `web:README.md` |
| Crawler | Full URL | `https://example.com/docs/intro` |

### 4. Web Platform: Under-Index Strategy

Due to Dioxus not exposing `webkitRelativePath` ([issue #3136](https://github.com/DioxusLabs/dioxus/issues/3136)), web uploads only have filename. We chose **under-indexing** (replace on collision):

| Approach | Behavior | Trade-off |
|----------|----------|-----------|
| Under-index (chosen) | Replace existing file with same name | Clean storage, may lose unrelated file |
| Over-index | Keep both files | No data loss, duplicates accumulate |

**Rationale:**
- User expectation: uploading "README.md" should update existing
- Prevents unbounded storage growth
- Recoverable: user can re-upload original if wrong file replaced
- Same filename usually IS the same document

---

## Alternatives Considered

### 1. JSONL with Scanning

**Rejected**: O(n) lookup time unacceptable at scale. A 1M chunk index would require scanning potentially gigabytes of data for each lookup.

### 2. SQLite

**Considered but deferred**: Good cross-platform option, but:
- Web requires sql.js (WASM port) - adds ~500KB
- redb is lighter weight and Rust-native
- IndexedDB is browser-native (zero bundle cost)
- Could revisit if we need relational queries

### 3. Full Rebuilds on Delete

**Rejected**: Original plan proposed rebuilding indices on each delete. At scale (100K+ chunks), this would cause multi-second UI freezes. Tombstone approach is industry standard (Weaviate, Milvus, hnswlib).

### 4. Hash-Only Deduplication for Web

**Rejected**: Would prevent duplicates but couldn't detect updates (same file, different content). Under-indexing provides better UX - users expect "replace" behavior.

### 5. Over-Indexing for Web

**Rejected**: Would preserve all versions but:
- Duplicates clutter search results
- Unbounded storage growth
- Confusing when same file appears multiple times
- Users don't expect "append" behavior for re-uploads

---

## Consequences

### Positive

1. **Scalable storage**: O(log n) lookups via B-tree (redb) or O(1) via IndexedDB
2. **Incremental writes**: No full-corpus serialization
3. **Memory efficient**: Only hot data in memory, text loaded on-demand
4. **Responsive updates**: Tombstones avoid HNSW rebuilds
5. **Clean search results**: Under-indexing prevents stale duplicates
6. **Background maintenance**: Compaction runs without blocking search

### Negative

1. **Added dependencies**: redb (~200KB), rexie (~50KB)
   - Acceptable: essential for scalability

2. **Web filename collisions**: Two different files named "README.md" will collide
   - Mitigation: Document limitation, users can rename files
   - Future: Use full path when Dioxus adds support

3. **Increased complexity**: Two storage implementations to maintain
   - Mitigation: Common trait reduces duplication
   - Both use similar KV semantics

4. **Migration required**: Existing blob-based indices need migration
   - One-time cost, can detect old format and migrate on load

### Neutral

1. **Index drift**: HNSW (tombstoned) vs BM25 (deleted) temporary inconsistency
   - Safe: RRF fusion handles missing DocIds gracefully
   - Resolved during compaction

---

## Implementation Files

| File | Changes |
|------|---------|
| `crates/coppermind-core/src/storage/mod.rs` | Define `DocumentStore` trait |
| `crates/coppermind-core/src/storage/redb_store.rs` | Desktop implementation |
| `crates/coppermind/src/storage/indexeddb_store.rs` | Web implementation |
| `crates/coppermind-core/src/search/engine.rs` | Use DocumentStore, source tracking |
| `crates/coppermind-core/src/search/vector.rs` | Tombstones, filtered search |
| `crates/coppermind-core/src/search/types.rs` | SourceRecord, updated types |
| `crates/coppermind/src/components/batch_processor.rs` | Hash computation, update detection |
| `crates/coppermind/src/components/file_processing.rs` | Platform source_id logic |
| `Cargo.toml` files | Add sha2, redb (native), rexie (wasm) |

---

## References

- **redb**: https://github.com/cberner/redb - Pure Rust embedded database
- **rexie**: https://github.com/devashishdxt/rexie - IndexedDB wrapper for Rust/WASM
- **Weaviate deletion**: https://weaviate.io/developers/weaviate/manage-data/delete
- **Milvus deletion**: https://milvus.io/ai-quick-reference/how-do-you-update-or-delete-vectors-in-an-ai-database
- **hnswlib tombstones**: https://github.com/nmslib/hnswlib/issues/4
- **Dioxus file path issue**: https://github.com/DioxusLabs/dioxus/issues/3136

---

## Decision Record

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-11-26 | Use redb for desktop | Pure Rust, WASM-incompatible but only used on native |
| 2025-11-26 | Use IndexedDB for web | Browser-native, zero bundle cost |
| 2025-11-26 | Tombstone-based deletion | Industry standard, avoids expensive rebuilds |
| 2025-11-26 | Under-index strategy for web | Better UX, cleaner storage, recoverable |
| 2025-11-26 | 30% compaction threshold | Industry standard balance of performance vs space |
