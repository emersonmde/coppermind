# ADR 008: Multi-Level Indexing Architecture

**Status**: Accepted
**Date**: 2025-11-29
**Context**: Proper nomenclature and document-level BM25 for improved search quality

---

## Context and Problem Statement

Coppermind's search system has two fundamental issues that need addressing before adding more indexes:

### 1. Nomenclature Confusion

The codebase uses "document" terminology incorrectly:

| Current Type | Actual Semantics | Problem |
|--------------|------------------|---------|
| `DocId` | Chunk identifier | Misleading - these are chunks, not documents |
| `Document` | Text chunk | A "document" is actually a chunk of a file |
| `DocumentRecord` | Stored chunk | Same issue |
| `DocumentMetadata` | Per-chunk metadata | Confusing when we need real document metadata |
| `DocumentStore` trait | Chunk storage | Stores chunks, not documents |

This makes the code confusing and will cause issues when we add document-level features.

### 2. BM25 at Wrong Level

Current BM25 indexes at the chunk level, which dilutes IDF (Inverse Document Frequency) statistics:

```
File: architecture.md (5 chunks)
  - Chunk 1: "search engine uses BM25..."
  - Chunk 2: "BM25 computes term frequency..."
  - Chunk 3: "...enhanced BM25 algorithm..."
  - Chunk 4: "BM25 parameters include k1..."
  - Chunk 5: "conclusion: BM25 works well..."
```

**Problem**: The term "BM25" appears in 5 chunks. Chunk-level BM25 counts this as appearing in 5 "documents", making IDF(BM25) = log(N/5). But semantically, "BM25" only appears in 1 document - the IDF should reflect document-level frequency.

**Impact**: Common terms within a single long document get incorrectly low IDF scores, reducing their discriminative power.

### 3. Future Index Plans

The user plans to add more indexes:
- Graph-based indexes on document metadata
- Experimentation with recall/precision tuning
- Potentially collection-level indexes

The current single-level architecture doesn't support this well.

---

## Decision

Implement a two-level indexing architecture with proper nomenclature:

### 1. Big Bang Rename

Rename all types to reflect their true semantics:

```rust
// types.rs - Core type renames
DocId → ChunkId
DOC_ID_COUNTER → CHUNK_ID_COUNTER
Document → Chunk
DocumentRecord → ChunkRecord

// engine.rs - Method renames
add_document → add_chunk
add_document_deferred → add_chunk_deferred
add_document_with_tokens → add_chunk_with_tokens
get_document → get_chunk

// vector.rs
doc_ids: Vec<DocId> → chunk_ids: Vec<ChunkId>

// document_store.rs trait - Method renames
get_document → get_chunk
put_document → put_chunk
delete_document → delete_chunk
get_documents_batch → get_chunks_batch
document_count → chunk_count
```

### 2. New Document Type and DocumentId

Add true document-level types:

```rust
/// Unique identifier for a source document (file, URL, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DocumentId(u64);

static DOCUMENT_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

impl DocumentId {
    pub fn new() -> Self { ... }
    pub fn init_counter(max_existing_id: u64) { ... }
    pub fn from_u64(id: u64) -> Self { ... }
    pub fn as_u64(&self) -> u64 { ... }
}

/// Full source document (file, web page, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: DocumentId,
    pub full_text: String,
    pub metadata: DocumentMetadata,
    pub chunk_ids: Vec<ChunkId>,
}

/// Document-level metadata (repurposed from current DocumentMetadata)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub filename: Option<String>,
    pub source: Option<String>,
    pub created_at: u64,
    pub chunk_count: usize,
    pub content_hash: String,
}

/// Updated ChunkMetadata with document reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    pub document_id: DocumentId,  // Parent reference (new)
    pub source_id: String,        // Keep for backward compat during transition
    pub display_name: String,
    pub chunk_index: usize,
    pub chunk_count: usize,
    pub created_at: u64,
}
```

### 3. Document-Level BM25 Index

Create a separate BM25 index for full documents:

```rust
// New file: crates/coppermind-core/src/search/document_keyword.rs

/// Document-level BM25 search engine.
/// Indexes full document text for proper IDF statistics.
pub struct DocumentKeywordEngine {
    search_engine: bm25::SearchEngine<u64>,
    document_count: usize,
}

impl DocumentKeywordEngine {
    pub fn new() -> Self { ... }

    /// Add a full document to the BM25 corpus.
    pub fn add_document(&mut self, doc_id: DocumentId, full_text: String) { ... }

    /// Remove a document from the index.
    pub fn remove_document(&mut self, doc_id: DocumentId) { ... }

    /// Search documents by query.
    pub fn search(&self, query: &str, k: usize) -> Vec<(DocumentId, f32)> { ... }

    pub fn len(&self) -> usize { ... }
}
```

### 4. Updated Engine Structure

```rust
pub struct HybridSearchEngine<S: DocumentStore> {
    // Chunk-level indexes
    vector_engine: VectorSearchEngine,      // HNSW - stays at chunk level
    keyword_engine: KeywordSearchEngine,    // Chunk BM25 - keep for now, may deprecate

    // Document-level indexes (new)
    document_keyword_engine: DocumentKeywordEngine,  // Document BM25

    // Storage
    store: S,
    embedding_dim: usize,
    manifest: IndexManifest,
}
```

### 5. New Search Flow

```
Query → embed(query)
         |
    +─────────────────────────────+
    │ HNSW (chunk-level)          │ → [(ChunkId, similarity), ...]
    │ Returns: k×3 chunk results  │
    +─────────────────────────────+
         |
    Lift to documents: ChunkId → DocumentId (via ChunkMetadata.document_id)
         |
    +─────────────────────────────+
    │ BM25 (document-level)       │ → [(DocumentId, bm25_score), ...]
    │ Query against full documents│
    │ Proper IDF across corpus    │
    +─────────────────────────────+
         |
    RRF Fusion (at document level)
    - Vector score: best chunk score per document
    - Keyword score: document BM25 score
         |
    Return: DocumentSearchResult with best chunk highlights
```

### 6. New Result Types

```rust
/// Document-level search result with best matching chunks.
#[derive(Debug, Clone)]
pub struct DocumentSearchResult {
    pub document_id: DocumentId,
    pub score: f32,  // RRF fused score
    pub vector_score: Option<f32>,  // Best chunk's vector score
    pub keyword_score: Option<f32>, // Document BM25 score
    pub document: Document,
    pub best_chunks: Vec<ChunkSearchResult>,  // Top chunks for highlighting
}

#[derive(Debug, Clone)]
pub struct ChunkSearchResult {
    pub chunk_id: ChunkId,
    pub score: f32,  // Chunk's HNSW similarity score
    pub text: String,
    pub chunk_index: usize,
}
```

### 7. Storage Additions

Add document-level operations to `DocumentStore` trait:

```rust
// Add to DocumentStore trait
async fn get_full_document(&self, id: DocumentId) -> Result<Option<Document>, StoreError>;
async fn put_full_document(&self, id: DocumentId, doc: &Document) -> Result<(), StoreError>;
async fn delete_full_document(&self, id: DocumentId) -> Result<(), StoreError>;
async fn iter_full_documents(&self) -> Result<Vec<(DocumentId, Document)>, StoreError>;
async fn full_document_count(&self) -> Result<usize, StoreError>;
```

New storage tables:
- redb: `FULL_DOCUMENTS_TABLE: TableDefinition<u64, &[u8]>`
- IndexedDB: `full_documents` object store

### 8. Updated Indexing Flow

```rust
// New engine methods
impl<S: DocumentStore> HybridSearchEngine<S> {
    /// Create a new document and return its ID.
    /// Call this before adding chunks.
    pub async fn create_document(
        &mut self,
        full_text: &str,
        metadata: DocumentMetadata,
    ) -> Result<DocumentId, SearchError>;

    /// Add a chunk belonging to a document.
    /// Adds to HNSW index and stores chunk.
    pub async fn add_chunk_to_document(
        &mut self,
        document_id: DocumentId,
        chunk_text: &str,
        chunk_index: usize,
        chunk_count: usize,
        embedding: Vec<f32>,
    ) -> Result<ChunkId, SearchError>;

    /// Finalize document indexing.
    /// Adds full text to document-level BM25.
    pub async fn finalize_document(
        &mut self,
        document_id: DocumentId
    ) -> Result<(), SearchError>;
}
```

**Indexing sequence:**
1. `create_document(full_text, metadata)` → DocumentId
2. For each chunk: `add_chunk_to_document(doc_id, chunk, idx, count, embedding)` → ChunkId
3. `finalize_document(doc_id)` → Adds to document BM25

---

## Implementation Phases

### Phase 1: Big Bang Rename

**Goal**: Rename all types. Pure refactor, no functional changes.

**Files to modify**:
- `crates/coppermind-core/src/search/types.rs` - Type definitions
- `crates/coppermind-core/src/search/engine.rs` - Method names
- `crates/coppermind-core/src/search/vector.rs` - Field names
- `crates/coppermind-core/src/search/keyword.rs` - Type references
- `crates/coppermind-core/src/search/aggregation.rs` - Type references
- `crates/coppermind-core/src/storage/document_store.rs` - Trait methods
- `crates/coppermind-core/src/storage/redb_store.rs` - Implementation
- `crates/coppermind/src/storage/indexeddb_store.rs` - Implementation
- `crates/coppermind/src/components/file_processing.rs` - Usage
- `crates/coppermind/src/components/batch_processor.rs` - Usage
- `crates/coppermind-cli/src/search.rs` - Usage
- `crates/coppermind-cli/src/output.rs` - Usage

**Verification**: All tests pass, cargo clippy clean.

### Phase 2: Add Document Types

**Goal**: Add DocumentId, Document, DocumentMetadata. Update ChunkMetadata.

**Files to modify**:
- `crates/coppermind-core/src/search/types.rs` - Add new types

**No functional changes yet** - just type definitions.

### Phase 3: Document Storage

**Goal**: Add document storage to DocumentStore trait and implementations.

**Files to modify**:
- `crates/coppermind-core/src/storage/document_store.rs` - Add trait methods
- `crates/coppermind-core/src/storage/redb_store.rs` - Implement for redb
- `crates/coppermind/src/storage/indexeddb_store.rs` - Implement for IndexedDB

### Phase 4: Document-Level BM25

**Goal**: Create DocumentKeywordEngine.

**Files to create**:
- `crates/coppermind-core/src/search/document_keyword.rs` (new)

**Files to modify**:
- `crates/coppermind-core/src/search/mod.rs` - Add module, export

### Phase 5: Engine Integration

**Goal**: Add DocumentKeywordEngine to HybridSearchEngine, new indexing methods.

**Files to modify**:
- `crates/coppermind-core/src/search/engine.rs` - Major changes

### Phase 6: Search Flow Update

**Goal**: Implement two-level search with RRF fusion at document level.

**Files to modify**:
- `crates/coppermind-core/src/search/engine.rs` - New search() implementation
- `crates/coppermind-core/src/search/types.rs` - Add DocumentSearchResult
- `crates/coppermind-core/src/search/aggregation.rs` - Simplify or deprecate

### Phase 7: UI and CLI Updates

**Goal**: Update consumers to use new result types.

**Files to modify**:
- `crates/coppermind/src/components/search/search_view.rs`
- `crates/coppermind/src/components/search/result_card.rs`
- `crates/coppermind-cli/src/search.rs`
- `crates/coppermind-cli/src/output.rs`

### Phase 8: Indexing Pipeline Update

**Goal**: Update file processing to use new indexing flow.

**Files to modify**:
- `crates/coppermind/src/components/file_processing.rs`
- `crates/coppermind/src/components/batch_processor.rs`

### Phase 9: Migration and Cleanup

**Goal**: Handle existing indexes, bump schema version.

**Actions**:
1. Bump `CURRENT_SCHEMA_VERSION` to 2 in `types.rs`
2. On load, detect schema v1 → prompt user to re-index
3. Remove deprecated chunk-level BM25 if not needed

---

## Alternatives Considered

### 1. Keep Chunk-Level BM25, Add Document-Level as Optional

**Rejected**: Adds complexity with marginal benefit. Document-level BM25 is strictly better for IDF.

### 2. Gradual Rename (Type Aliases)

```rust
// Gradual approach
pub type DocId = ChunkId;  // Deprecated alias
pub type Document = Chunk; // Deprecated alias
```

**Rejected**: User preferred "big bang" for clean break. Aliases create confusion and tech debt.

### 3. Three-Level Hierarchy (Chunk → Document → Collection)

**Deferred**: User wants to start with two levels. Collection level can be added later.

### 4. Document-Level Vector Search

Index document-level embeddings (average/CLS of chunks).

**Rejected**: Loses fine-grained semantic matching. Chunk-level HNSW with document lifting is industry standard.

### 5. Hybrid BM25 (Both Levels)

Keep chunk BM25 + add document BM25, combine with RRF.

**Considered but deferred**: Adds complexity. Start with document-only BM25, can add chunk BM25 back if needed.

---

## Consequences

### Positive

1. **Correct terminology**: Code reflects actual semantics (chunks vs documents)
2. **Proper IDF statistics**: Document-level BM25 gives correct term importance
3. **Extensible architecture**: Easy to add more document-level indexes (graph, etc.)
4. **Better search quality**: Expected improvement in keyword search precision
5. **Cleaner aggregation**: Search returns documents directly, no post-processing

### Negative

1. **Breaking change**: Requires re-indexing existing data
   - Mitigation: Detect old schema, prompt user
2. **Increased storage**: Store full documents in addition to chunks
   - Mitigation: Already have source tracking, this is incremental
3. **Implementation effort**: Significant refactor across many files
   - Mitigation: Phase-based approach, each phase testable

### Neutral

1. **Memory footprint**: DocumentKeywordEngine adds memory for document BM25
   - Similar to existing chunk BM25, scales with document count
2. **Search latency**: Additional BM25 query + RRF fusion
   - BM25 is fast, should be negligible

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| BM25 level | Document-only | Proper IDF, simpler than hybrid |
| HNSW level | Chunk-only | Fine-grained semantics needed |
| RRF fusion level | Document | Natural aggregation point |
| Chunk highlighting | Best N chunks | Show relevant excerpts |
| Migration | Clear + re-index | Simpler than data migration |
| Rename approach | Big bang | Clean break, no aliases |

---

## Files to Modify (Complete List)

### coppermind-core/src/search/
| File | Phase | Changes |
|------|-------|---------|
| `types.rs` | 1, 2, 6 | Rename types, add DocumentId/Document, add DocumentSearchResult |
| `engine.rs` | 1, 5, 6 | Rename methods, add document engine, new search flow |
| `vector.rs` | 1 | Rename doc_ids to chunk_ids |
| `keyword.rs` | 1 | Update type references |
| `aggregation.rs` | 1, 6 | Update types, possibly deprecate |
| `fusion.rs` | 1 | Minimal changes (generic) |
| `mod.rs` | 4 | Add document_keyword module |
| `document_keyword.rs` | 4 | **New file** |

### coppermind-core/src/storage/
| File | Phase | Changes |
|------|-------|---------|
| `document_store.rs` | 1, 3 | Rename trait methods, add document operations |
| `redb_store.rs` | 1, 3 | Rename implementation, add document table |
| `mod.rs` | 1 | Update exports |

### coppermind/src/storage/
| File | Phase | Changes |
|------|-------|---------|
| `indexeddb_store.rs` | 1, 3 | Rename implementation, add document store |

### coppermind/src/components/
| File | Phase | Changes |
|------|-------|---------|
| `file_processing.rs` | 1, 8 | Update types, new indexing flow |
| `batch_processor.rs` | 1, 8 | Update types, new indexing flow |
| `search/search_view.rs` | 7 | Use DocumentSearchResult |
| `search/result_card.rs` | 7 | Display document + chunks |

### coppermind-cli/src/
| File | Phase | Changes |
|------|-------|---------|
| `search.rs` | 1, 7 | Update types and result handling |
| `output.rs` | 1, 7 | Update formatters |

---

## Testing Strategy

### Unit Tests
- `types.rs`: Test new DocumentId generation, Document creation
- `document_keyword.rs`: Test add/search/remove operations
- `engine.rs`: Test new indexing methods, search flow

### Integration Tests
- Full indexing pipeline with documents and chunks
- Search returning DocumentSearchResult
- Re-indexing after schema migration

### Manual Testing
- Web: `dx serve -p coppermind`
- Desktop: `dx serve -p coppermind --platform desktop`
- CLI: `cargo run -p coppermind-cli -- "query"`

---

## Success Criteria

- [ ] All types renamed (DocId→ChunkId, Document→Chunk, etc.)
- [ ] DocumentId and Document types implemented
- [ ] Document storage working (redb + IndexedDB)
- [ ] DocumentKeywordEngine indexing and searching
- [ ] HybridSearchEngine using document-level BM25
- [ ] Search returns DocumentSearchResult with best chunks
- [ ] UI displays document results with chunk highlights
- [ ] CLI works with new result format
- [ ] Schema v1 indexes trigger re-index prompt
- [ ] All tests pass, clippy clean
- [ ] Pre-commit hook passes

---

## References

- **BM25 IDF**: Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework
- **RRF**: Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). Reciprocal Rank Fusion
- **Chunk vs Document search**: Industry practice in Weaviate, Pinecone, and other vector DBs
- **ADR 007**: Document Storage and Re-Upload Handling (this ADR builds on it)

---

## Decision Record

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-11-29 | Big bang rename | User preference for clean break |
| 2025-11-29 | Document-level BM25 only | Proper IDF, simpler architecture |
| 2025-11-29 | Two-level hierarchy | Start simple, Collection level deferred |
| 2025-11-29 | Clear + re-index migration | Simpler than data migration |
