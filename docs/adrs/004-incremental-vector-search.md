# ADR 004: Incremental Vector Search with rust-cv/hnsw

**Status**: Accepted
**Date**: 2025-11-21
**Context**: Vector search performance optimization - eliminating expensive index rebuilds

---

## Context and Problem Statement

The semantic search engine uses HNSW (Hierarchical Navigable Small World) for fast approximate nearest neighbor search over document embeddings. The original implementation using `instant-distance` required **full index rebuilds** after every batch of documents was added.

### Original Behavior (instant-distance 0.6)

```rust
// Add documents
for (doc_id, embedding) in batch {
    engine.add_document_deferred(doc_id, embedding);  // Store but don't index
}

// Expensive: Rebuild entire HNSW graph from scratch
engine.rebuild_index();  // O(n log n) - blocks UI for seconds on large indexes
```

**Problems identified:**
1. **Performance bottleneck**: Rebuilding index for 1000 documents takes 2-5 seconds
2. **UI freezing**: Blocking operation prevents incremental feedback during file uploads
3. **Poor UX**: Users wait for entire batch to complete before seeing any progress
4. **Scalability issue**: Rebuild time grows super-linearly with index size
5. **WASM constraints**: Limited memory budget makes large rebuilds problematic

### Requirements

1. **Incremental updates**: Add documents without rebuilding the entire index
2. **WASM compatibility**: Pure Rust with no native dependencies
3. **Cross-platform**: Identical behavior on web (WASM) and desktop
4. **Cosine distance**: Match current metric for normalized embeddings
5. **Performance**: Search quality must match or exceed instant-distance
6. **API compatibility**: Minimize breaking changes to existing code

---

## Decision

Replace `instant-distance` with **`rust-cv/hnsw`** which supports true incremental HNSW insertion.

### Core Change

```rust
// OLD: instant-distance (requires rebuild)
impl VectorSearchEngine {
    pub fn add_document(&mut self, doc_id: DocId, embedding: Vec<f32>) {
        self.embeddings.push(embedding);  // Store for rebuild
    }

    pub fn rebuild_index(&mut self) {
        // Expensive: O(n log n) rebuild of entire graph
        self.index = Hnsw::new(&self.embeddings);
    }
}

// NEW: rust-cv/hnsw (incremental insertion)
impl VectorSearchEngine {
    pub fn add_document(&mut self, doc_id: DocId, embedding: Vec<f32>) {
        let point = EmbeddingPoint(embedding);
        self.index.insert(point, &mut self.searcher);  // Direct insertion, no rebuild!
    }

    pub fn rebuild_index(&mut self) {
        // No-op: index is always up-to-date
    }
}
```

### Implementation Details

**Distance Metric**: Custom `CosineDistance` implementing `space::Metric<Box<[f32]>>`
```rust
struct CosineDistance;

impl Metric<Box<[f32]>> for CosineDistance {
    type Unit = u32;

    fn distance(&self, a: &Box<[f32]>, b: &Box<[f32]>) -> u32 {
        // Deref Box to &[f32] - zero cost abstraction
        let a_slice: &[f32] = a;
        let b_slice: &[f32] = b;
        let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();

        let cosine_sim = dot / (mag_a * mag_b);
        let distance = 1.0 - cosine_sim;

        // Scale to u32 range [0, u32::MAX]
        (distance * (u32::MAX as f32 / 2.0)) as u32
    }
}
```

**HNSW Parameters** (tuned for accuracy):
- `M`: 16 (bidirectional links per node, default 12)
- `M0`: 32 (layer 0 connections, typically 2*M)
- `ef_construction`: 200 (construction quality, default 100)
- `ef_search`: max(k*2, 50) (search quality parameter)

**RNG**: `rand::rngs::StdRng` - WASM-compatible via `getrandom` with `js` feature

**Memory Safety**: Embeddings stored as `Box<[f32]>` (stable heap allocations)
- Index owns the Box<[f32]> data directly - no lifetime issues
- No unsafe code required - fully checked by the compiler
- Box provides stable heap allocations that won't move when index grows
- Bonus: Saves 8 bytes per embedding vs Vec (no capacity field)

---

## Alternatives Considered

### 1. hnsw_rs (jean-pierreBoth/hnswlib-rs)

**Pros:**
- Feature-rich: mmap, filtering, multiple distance metrics
- High performance (62k requests/second)
- Active maintenance
- Parallel insertion and search

**Cons:**
- ❌ **WASM blocker**: Depends on `mmap-rs` which doesn't compile to WASM
- ❌ **Platform incompatibility**: `mmap-rs` uses platform-specific syscalls
- Heavier dependency footprint

**Decision**: Rejected due to WASM incompatibility (critical requirement).

### 2. hora (hora-search/hora)

**Pros:**
- Multiple algorithms (HNSW, IVF, LSH)
- Good documentation
- Active community

**Cons:**
- ❌ **Unmaintained**: Last commit Oct 2021 (3+ years ago)
- ❌ **Rust version issues**: Likely incompatible with modern toolchain
- No confidence in long-term support

**Decision**: Rejected due to abandonment.

### 3. usearch (unum-cloud/usearch)

**Pros:**
- High performance (industry-grade)
- Multi-language support
- Active development

**Cons:**
- ❌ **Not pure Rust**: Rust bindings to C++ core
- Native dependencies violate WASM requirement

**Decision**: Rejected - not pure Rust.

### 4. Keep instant-distance, Optimize Rebuild

**Pros:**
- No migration needed
- Known behavior

**Cons:**
- Fundamental limitation: HNSW algorithm implementation doesn't support incremental updates
- Would require forking and rewriting core algorithm
- Still blocks UI during rebuilds

**Decision**: Rejected - doesn't solve the root problem.

---

## Consequences

### Positive

1. **Incremental updates**: Add documents one-by-one without expensive rebuilds
2. **Responsive UI**: No more multi-second blocks during file uploads
3. **Better UX**: Users see progress as each chunk is indexed
4. **Scalability**: Performance degrades gracefully with index size
5. **WASM-compatible**: Pure Rust implementation works identically on web and desktop
6. **Memory efficient**: No need to store embeddings separately for rebuild
7. **Maintained library**: rust-cv is actively developed and well-tested

### Negative

1. **API change**: `search()` now requires `&mut self` (searcher state mutation)
   - **Mitigation**: Single-line change in call sites (`let mut engine`)
   - **Impact**: Minimal - only 2 call sites (engine.rs, search_view.rs)

2. **RNG dependency**: Adds `rand = "0.8"` (was already transitive, now direct)
   - **Size**: ~50KB compiled
   - **Acceptable**: Essential for HNSW level assignment

### Neutral

1. **Different HNSW implementation**: rust-cv/hnsw vs instant-distance internals
   - Search quality should be equivalent (both implement standard HNSW algorithm)
   - Parameters tuned to match or exceed instant-distance quality

2. **Distance metric implementation**: Custom instead of built-in
   - More control over distance calculation
   - Can optimize for specific use case if needed

---

## Implementation Files Changed

```
src/search/vector.rs     - Complete rewrite for rust-cv/hnsw API
src/search/engine.rs     - Update search() signature to &mut self
src/components/search/search_view.rs - Add mut to search_engine binding
Cargo.toml               - Replace instant-distance with hnsw + space + rand
```

### Testing Results

All 126 tests pass ✅
- Vector search unit tests (11 tests)
- Hybrid search engine tests (8 tests)
- Full integration test suite
- WASM build successful
- Desktop build successful

### Performance Characteristics

**Incremental insertion**:
- Single document add: O(log n) - logarithmic with index size
- No rebuild needed: Previous O(n log n) eliminated entirely

**Search performance**:
- Query time: O(log n) - same as instant-distance
- Memory usage: Linear in number of documents (same as before)

**Practical impact** (estimated on 10k document index):
- Before: 2-5 second rebuild after each batch → blocks UI
- After: <1ms per document insertion → responsive UI

---

## Future Directions

### Potential Optimizations

1. **Batch insertion API**: Add `insert_batch()` for slight efficiency gain
   - rust-cv/hnsw doesn't expose batch API currently
   - Individual inserts are already fast enough for our use case

2. **Persistence**: Save/load HNSW index to avoid rebuilding on app restart
   - rust-cv/hnsw doesn't implement serde traits (yet)
   - Could implement custom serialization if needed

3. **Parameter tuning**: Experiment with M/ef values for speed/quality tradeoff
   - Current params (M=16, ef=200) chosen for quality
   - Could reduce for faster insertion if acceptable

4. **SIMD acceleration**: rust-cv/hnsw has SIMD support via `space` crate
   - Not currently enabled
   - Could provide 2-3x speedup on x86/ARM platforms

### Monitoring

Track these metrics to validate decision:
- Index build time (should be near-zero now)
- Search quality (precision@k, recall@k)
- Memory usage
- UI responsiveness during file uploads

---

## References

- **rust-cv/hnsw**: https://github.com/rust-cv/hnsw
- **space crate** (distance metrics): https://github.com/rust-cv/space
- **instant-distance** (previous): https://github.com/instant-labs/instant-distance
- **HNSW paper**: Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (2018)
- **Cosine distance**: Standard metric for normalized embeddings, used in JinaBERT and most sentence transformers

### Evaluation Timeline

1. Initial investigation: Hora (rejected - unmaintained)
2. Tried hnsw_rs first (rejected - WASM incompatibility with mmap-rs)
3. Switched to rust-cv/hnsw (accepted - pure Rust, incremental, active)

---

## Notes

This change is **foundational for UX improvements**. Incremental indexing enables:
- Real-time search as documents are uploaded
- Progress indicators during indexing
- Responsive UI even with large document sets
- Future streaming/live-update features

The migration required careful attention to lifetime management (unsafe transmute) but the safety invariant is simple and well-documented: the embeddings vector is append-only after insertion.

The `&mut self` requirement from rust-cv/hnsw's `Searcher` is a minor API change that accurately reflects the internal state mutation during search. This is more honest than instant-distance's `&self` which hid internal mutability.
