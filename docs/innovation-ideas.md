# Experimental Ideas

This document captures experimental directions for Rust + WASM + ML in browsers.

---

## Project Characteristics

**Full Rust ML Stack:**
- Uses Dioxus + Candle for browser-based semantic search
- Most browser ML uses JavaScript (transformers.js, TensorFlow.js, ONNX Runtime Web)
- Rust compiled to WASM for:
  - UI framework (Dioxus)
  - ML inference (Candle)
  - Tokenization (tokenizers-rs)

**Technical Benefits:**
- Type safety catches bugs at compile time
- Performance closer to native than JavaScript
- No garbage collection pauses during inference
- Same codebase for web, desktop, mobile

**Cross-Platform:**
- Platform-specific optimizations via `cfg` attributes
- Web uses OPFS, desktop uses native filesystem
- Both platforms share core logic

**Privacy:**
- Pure local inference
- No cloud API calls
- Works offline

---

## Implemented Features

### ✅ Hybrid Search: Semantic + BM25 Full-Text
**Status:** Implemented

**Description:**
Successfully implemented hybrid search combining vector similarity (instant-distance HNSW) with keyword matching (BM25), merged using Reciprocal Rank Fusion (RRF). See [Design Document](architecture-design.md) for full implementation details.

**Results:**
- Catches both semantic matches (paraphrases, synonyms) and exact keyword matches
- All-in-Rust implementation using instant-distance and bm25 crates
- RRF fusion operates purely on rank positions, no score normalization needed

---

## Ideas to Explore

### 🚀 High Priority: Novel & Feasible

#### 1. Enable Rayon WASM Parallelism for Candle (3x Speedup)
**Status:** Ready to implement (COI infrastructure already in place)

**Description:**
- We already have Cross-Origin Isolation enabled (COOP/COEP headers via service worker)
- Candle includes Rayon for CPU parallelism, but it's currently single-threaded on WASM
- Adding `wasm-bindgen-rayon` adapter unlocks SharedArrayBuffer parallelism
- **Proven 3x speedup** in Candle PR #3063: 5 → 16 tokens/sec (Phi-1.5 model)

**Current Status:**
```bash
# Rayon is in WASM dependency tree but NOT parallelizing
$ cargo tree --target wasm32-unknown-unknown -p rayon
rayon v1.11.0
├── candle-core v0.8.4  # ← Already using Rayon
├── instant-distance v0.6.1  # ← HNSW also uses Rayon
└── ...

# Missing: web_spin_lock feature + wasm-bindgen-rayon adapter
```

**Requirements:**
1. ✅ COOP/COEP headers (already have via `coi-serviceworker.min.js`)
2. ❌ Enable rayon's `web_spin_lock` feature
3. ❌ Add `wasm-bindgen-rayon` dependency
4. ❌ Switch to nightly Rust (WASM threads not stable yet)

**Implementation Steps:**
1. Add to `Cargo.toml`:
   ```toml
   [target.'cfg(target_arch = "wasm32")'.dependencies]
   wasm-bindgen-rayon = "1.2"
   rayon = { version = "1.11", features = ["web_spin_lock"] }
   ```
2. Initialize rayon in worker context (see Candle PR for pattern)
3. Switch to nightly Rust: `rustup override set nightly`
4. Test embedding throughput before/after
5. Document nightly requirement in README

**Expected Outcome:**
- 3x faster JinaBERT inference on web (proven in similar Candle model)
- Embedding 100 chunks: ~10s → ~3s
- Significantly better UX for large file indexing
- Desktop remains unchanged (already uses native threads)

**Trade-offs:**
- Requires nightly Rust (not stable yet)
- Increases WASM binary size slightly
- More complex build setup

**References:**
- [Candle Rayon WASM PR #3063](https://github.com/huggingface/candle/pull/3063) (3x speedup demo)
- [wasm-bindgen-rayon](https://github.com/GoogleChromeLabs/wasm-bindgen-rayon)
- [ADR 002](adrs/002-web-crawler-desktop-first.md) (documents current COI setup)

---

#### 2. Candle WebGPU Backend for Browser ML
**Status:** In progress by Candle team

**Description:**
- Candle team is actively working on WebGPU backend
- Not worth duplicating this work
- Monitor progress and use prerelease/test branch when available
- Potential 5-20x speedup for ML inference

**Next Steps:**
1. Track Candle GitHub for WebGPU backend progress
2. Test prerelease versions when available
3. Provide feedback to Candle team based on browser ML use case
4. Update Coppermind to use WebGPU backend when stable

**Expected Outcome:**
- Embedding inference: 50-200ms → 5-20ms
- Enables real-time semantic search
- Larger models become feasible (BERT-base, MPNet)

---

### 🧪 Medium Priority: Experimental & High Impact

#### 2. Quantized Models (F16/INT8) for WASM
**Status:** Candle has quantization, but not well-tested for WASM

**Description:**
- Current: F32 weights
- F16: 2x smaller, minimal quality loss
- INT8: 4x smaller, some quality loss

**Challenges:**
- INT8 operations slow on CPU (no SIMD in WASM yet)
- Need to benchmark quality vs size tradeoff

**Benefits:**
- Larger models become feasible
- Faster downloads
- Could fit 2-3 models simultaneously (multi-lingual, domain-specific)

**Note:** Browser support not a concern - fine targeting latest Chrome even with experimental features.

---

#### 3. Turso/libSQL for Native Vector Search in Browser
**Status:** Needs investigation

**Description:**
- [Turso](https://turso.tech/) is a SQLite fork (libSQL) with native vector similarity search
- Built-in `vec_distance_*` functions for cosine, L2, etc.
- Designed for edge computing, could work in browsers via WASM

**Example:**
```sql
-- Native vector search in SQL
SELECT chunk_id, text, vec_distance_cosine(embedding, :query_vector) AS similarity
FROM embeddings
ORDER BY similarity ASC
LIMIT 10;
```

**Benefits:**
- Eliminates need for separate vector index (HNSW, FAISS)
- Native SQL + vector search enables powerful queries
- Single database for everything (documents, chunks, embeddings, metadata)

**Implementation Strategy:**
1. Start with in-memory libSQL (no persistence initially)
2. Test by indexing small set of files then searching
3. Get search working first
4. Add OPFS persistence later

**Next Steps:**
1. Research libSQL WASM compilation (check Turso GitHub)
2. Test in-memory vector search functions in browser
3. Index small test dataset and verify search works
4. Benchmark vs brute-force cosine similarity
5. Add OPFS persistence once search is working

**Challenges:**
- libSQL WASM compilation status unknown
- Vector search performance in WASM vs native
- COOP/COEP compatibility needs verification

**Expected Outcome:**
- Search 10K vectors in <10ms (vs 50-100ms brute force)
- Simpler architecture than separate storage + vector index
- Single database solution

**References:**
- [Turso](https://turso.tech/)
- [libSQL GitHub](https://github.com/tursodatabase/libsql)
- [libSQL Vector Search Docs](https://turso.tech/blog/libsql-vector-search)

---

### 💡 Speculative: Far Future

#### 4. Federated Learning in Browser (Multi-User)
**Status:** Very experimental, privacy-tech research area

**Description:**
- Multiple users process documents locally
- Share only embeddings (not raw text) to build collective knowledge base
- Privacy-preserving semantic search across organizations
- **This is really cool! Didn't even know this was possible**

**How:**
- User A embeds their docs → shares embeddings (vectors only)
- User B does same
- Both can search across combined embedding space
- Neither sees other's raw documents

**Challenges:**
- Trust model (how to verify embeddings are safe?)
- De-duplication
- Malicious user detection
- Data leakage risks

**Note:** Out of scope for near/medium term, but worth revisiting in far future. Keeping in list for future exploration.

---

### 🎯 Infrastructure

#### 5. Candle WASM Benchmarking Suite
**Description:**
- Comprehensive benchmarks for ML ops in WASM
- Compare CPU vs WebGPU vs SIMD
- Identify bottlenecks
- Track progress over time

**Deliverable:**
- Public dashboard showing performance metrics
- Helps identify optimization opportunities

---

#### 6. Dioxus + Candle Example Library
**Description:**
- Coppermind as reference implementation
- Additional examples:
  - Image classification (ConvNext)
  - Audio transcription (Whisper)
  - Text generation (small LLaMA)
- All using Dioxus + Candle + WASM

**Benefits:**
- Makes Rust browser ML more accessible
- Provides working examples for others

---

## Possible Future Directions

### Performance & Optimization

**1. Desktop GPU Acceleration**
- Leverage Candle's CUDA/Metal backends for desktop builds
- Benchmark vs browser WASM performance
- Demonstrate cross-platform power

**2. Multi-Model Support**
- Load different embedding models for different use cases
- Example: code embeddings, multilingual, domain-specific
- Model switching in browser

**3. Export to Standard Formats**
- Export embeddings to Parquet
- Compatible with Python ecosystem (FAISS, Pinecone, Weaviate)
- Enables hybrid workflows (browser → cloud)

---


---

## References

- [Candle GitHub](https://github.com/huggingface/candle) - Rust ML framework
- [Dioxus Docs](https://dioxuslabs.com/) - Rust UI framework
- [WebGPU Spec](https://www.w3.org/TR/webgpu/) - GPU acceleration in browsers
- [WASM Threads Proposal](https://github.com/WebAssembly/threads) - Multi-threading in WASM
- [OPFS Spec](https://fs.spec.whatwg.org/) - Origin Private File System

---

**Note:** This is a living document for future explorations. Implemented features are moved to the "Implemented Features" section.
