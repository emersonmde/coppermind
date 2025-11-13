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
- Web uses IndexedDB, desktop uses SQLite
- Both platforms share core logic

**Privacy:**
- Pure local inference
- No cloud API calls
- Works offline

---

## Ideas to Explore

### 🚀 High Priority: Novel & Feasible

#### 1. Candle WebGPU Backend for Browser ML
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

#### 2. Hybrid Search: Semantic + BM25 Full-Text
**Status:** Common in enterprise, rare in pure browser implementations

**Description:**
- Semantic search: Great for meaning, poor for exact keywords
- BM25: Great for keywords, poor for meaning
- Combine both for better results
- This is exactly what we're after for this project

**Implementation:**
- Store chunk text in IndexedDB/SQLite
- Build BM25 index (term frequency, inverse document frequency)
- For each query:
  1. Run semantic search (top 100 results)
  2. Run BM25 search (top 100 results)
  3. Combine with weighted score (e.g., 0.7 semantic + 0.3 BM25)
  4. Re-rank and return top 10

**Benefits:**
- Better coverage than either approach alone
- All-in-Rust implementation (no JavaScript search libs)

---

### 🧪 Medium Priority: Experimental & High Impact

#### 3. Quantized Models (F16/INT8) for WASM
**Status:** Candle has quantization, but not well-tested for WASM

**Description:**
- Current: F32 weights (262MB)
- F16: 131MB (2x smaller, minimal quality loss)
- INT8: 65MB (4x smaller, some quality loss)

**Challenges:**
- INT8 operations slow on CPU (no SIMD in WASM yet)
- Need to benchmark quality vs size tradeoff

**Benefits:**
- Larger models become feasible
- Faster downloads
- Could fit 2-3 models simultaneously (multi-lingual, domain-specific)

**Note:** Browser support not a concern - fine targeting latest Chrome even with experimental features.

---

#### 4. Turso/libSQL for Native Vector Search in Browser
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

#### 5. Federated Learning in Browser (Multi-User)
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

#### 6. Candle WASM Benchmarking Suite
**Description:**
- Comprehensive benchmarks for ML ops in WASM
- Compare CPU vs WebGPU vs SIMD
- Identify bottlenecks
- Track progress over time

**Deliverable:**
- Public dashboard showing performance metrics
- Helps identify optimization opportunities

---

#### 7. Dioxus + Candle Example Library
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

## Ideas Specifically for This Project

### Near-Term (Next 3-6 Months)

**1. Increase to 4GB WASM Memory + 4096 Token Sequences**
- Unlocks processing longer documents
- Simple config change, high impact
- See: `docs/model-optimization.md`

**2. Implement HNSW Vector Index (Pure Rust)**
- Current: Brute-force cosine similarity
- HNSW: Approximate nearest neighbor search
- Enables >10K document search in <50ms
- Pure Rust implementation (no JavaScript libs)

**3. Desktop App with Native GPU**
- Test Candle with CUDA/Metal on desktop
- Benchmark vs browser WASM
- Demonstrate cross-platform power

### Medium-Term (6-12 Months)

**4. Candle WebGPU Backend**
- Implement WebGPU backend for Candle
- Potential contribution to upstream Candle repo

**5. Multi-Model Support**
- Load different embedding models for different use cases
- Example: code embeddings, multilingual, domain-specific
- Model switching in browser

**6. Export to Standard Formats**
- Export embeddings to Parquet
- Compatible with Python ecosystem (FAISS, Pinecone, Weaviate)
- Enables hybrid workflows (browser → cloud)

---

## Metrics for Success

### Technical Metrics

**Performance:**
- Cold start: <2s
- Embedding (2048 tokens): <50ms (WebGPU) or <200ms (CPU)
- Search (10K docs): <100ms

**Scale:**
- Process 1000 documents in <5 minutes
- Store 10K+ documents in browser
- Support files up to 100MB

---

## References

- [Candle GitHub](https://github.com/huggingface/candle) - Rust ML framework
- [Dioxus Docs](https://dioxuslabs.com/) - Rust UI framework
- [WebGPU Spec](https://www.w3.org/TR/webgpu/) - GPU acceleration in browsers
- [WASM Threads Proposal](https://github.com/WebAssembly/threads) - Multi-threading in WASM
- [OPFS Spec](https://fs.spec.whatwg.org/) - Origin Private File System

---

**Last Updated:** 2025-01-11

**Note:** This is a living document. Ideas will be added, marked completed, and updated based on learnings.
