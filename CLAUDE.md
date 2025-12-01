# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Coppermind is a browser-based semantic search engine using Rust and WASM. It uses Dioxus for the UI and Candle for ML inference, running entirely client-side without server dependencies.

### Core Project Tenets

**1. Rust-First Approach**
- Use Rust for all components where practical
- Only exception: COOP/COEP Service Worker (requires JavaScript for browser header injection)
- Prefer Rust solutions over JavaScript libraries:
  - Dioxus (instead of React/Vue/Svelte)
  - Candle (instead of transformers.js/ONNX Runtime)
  - tokenizers-rs (instead of @xenova/transformers tokenizer)
  - redb/IndexedDB for storage (see `DocumentStore` trait)

**2. Local-First Architecture**
- All processing happens on the user's device
- No cloud API dependencies
- Works offline once loaded
- No telemetry or tracking

## Development Commands

### Serving & Building
```bash
dx serve -p coppermind                           # Development server (web platform)
dx serve -p coppermind --platform desktop        # Desktop platform
dx bundle -p coppermind --release                # Production build for deployment
```

### Quality Checks
```bash
cargo fmt                          # Format code
cargo fmt -- --check               # Check formatting without modifying
cargo clippy --all-targets -- -D warnings  # Lint (must pass with zero warnings)
cargo test --verbose               # Run all tests
cargo doc --no-deps                # Build documentation
cargo audit                        # Security audit (requires cargo-audit)
```

### CLI & Evaluation
```bash
cargo install --path crates/coppermind-cli  # Install CLI tool
cm "search query"                            # Search existing index
cm "query" -n 5 --json                       # JSON output, limit 5

cargo run -p coppermind-eval --release       # Run search quality evaluation
cargo run -p coppermind-eval --release -- --ablation rrf  # RRF ablation study
cargo bench -p coppermind-core               # Run performance benchmarks
```

### Setup
```bash
./download-models.sh               # Download JinaBERT model (65MB + 695KB tokenizer)
git config core.hooksPath .githooks  # Enable pre-commit checks
```

Install required tools:
```bash
cargo install dioxus-cli --locked
cargo install cargo-audit --locked
```

## Architecture

### Workspace Structure

Coppermind uses a Cargo workspace with four crates:

#### `crates/coppermind-core/` - Platform-Independent Core Library
- **src/lib.rs**: Public API exports
- **src/search/**: Hybrid search implementation
  - **engine.rs**: `HybridSearchEngine` orchestrating vector + keyword search
  - **vector.rs**: HNSW semantic search using `hnsw` crate (cosine distance)
  - **keyword.rs**: BM25 full-text search using `bm25` crate
  - **fusion.rs**: Reciprocal Rank Fusion (RRF) algorithm for merging rankings
  - **aggregation.rs**: `aggregate_chunks_by_file()` for file-level result grouping
  - **types.rs**: `DocId`, `Document`, `SearchResult`, `SearchError`
- **src/storage/**: `DocumentStore` trait and implementations
  - **document_store.rs**: `DocumentStore` trait, `InMemoryDocumentStore`
  - **redb_store.rs**: `RedbDocumentStore` for desktop (feature-gated)
- **src/embedding/**: ML model abstractions and implementations (Candle-based)
  - **traits.rs**: `AssetLoader`, `Embedder`, `ModelConfig` traits
  - **config.rs**: `JinaBertConfig` implementation
  - **model.rs**: `JinaBertEmbedder` (Candle-based inference)
  - **tokenizer.rs**: `TokenizerHandle` wrapper for HuggingFace tokenizers
  - **types.rs**: `EmbeddingResult`, `ChunkEmbeddingResult`
- **src/chunking/**: Text chunking strategies
  - **mod.rs**: `ChunkingStrategy` trait, `FileType` enum, `create_chunker()`
  - **text_splitter_adapter.rs**: ICU4X sentence-based chunking
  - **markdown_splitter_adapter.rs**: Markdown-aware chunking (pulldown-cmark)
  - **code_splitter_adapter.rs**: Syntax-aware chunking with tree-sitter (native only)
  - **tokenizer_sizer.rs**: Token-based chunk sizing
- **src/gpu/**: GPU scheduler for thread-safe Metal access (native only)
  - **scheduler.rs**: `GpuScheduler` trait
  - **serial_scheduler.rs**: `SerialScheduler` with dedicated worker thread
  - **types.rs**: `EmbedRequest`, `Priority`, `ModelId`
  - **error.rs**: `GpuError` types
- **src/processing/**: Document processing and indexing pipeline
  - **pipeline.rs**: `IndexingPipeline` for chunking → tokenization → embedding
  - **progress.rs**: `IndexingProgress`, `BatchProgress` for UI feedback
- **src/error.rs**: Error types (`EmbeddingError`, `ChunkingError`, `GpuError`)
- **src/config.rs**: Production configuration constants (`MAX_CHUNK_TOKENS=1024`, `EMBEDDING_DIM=512`)
- **src/metrics.rs**: Performance metrics with rolling averages (indexing, search, scheduler)
- **src/evaluation/**: IR evaluation framework
  - **mod.rs**: Two-tier evaluation (synthetic for CI, real datasets for quality)
  - **metrics.rs**: NDCG, MAP, MRR, Precision@k, Recall@k, F1@k
  - **stats.rs**: Bootstrap CI, paired t-test, Cohen's d effect size
  - **datasets/**: Synthetic and Natural Questions dataset loaders
- **benches/**: Performance benchmarks (Criterion)
  - **indexing.rs**: Indexing throughput, batch processing, concurrent operations
  - **search.rs**: Search latency, hybrid search rebuild, result quality
  - **throughput.rs**: End-to-end throughput measurements

#### `crates/coppermind-eval/` - Standalone Evaluation Tool
- **src/main.rs**: CLI for running evaluation benchmarks
- **src/datasets/**: Custom dataset definitions (coppermind-eval dataset)
- Uses `elinor` crate for additional IR metrics validation

#### `crates/coppermind-cli/` - Command-Line Search Tool
- **src/main.rs**: `cm` CLI entry point with clap argument parsing
- **src/search.rs**: Search execution against existing index
- **src/config.rs**: Platform-specific data directory detection
- **src/output.rs**: Human-readable and JSON output formatting
- Shares index with desktop app (same redb database location)

#### `crates/coppermind/` - Application Crate
- **src/main.rs**: Entry point with platform-specific launch (desktop/mobile/web)
- **src/lib.rs**: Public API surface, module exports
- **src/error.rs**: Error types (`FileProcessingError`)

- **src/embedding/**: App-specific embedding utilities
  - **mod.rs**: High-level API (`compute_embedding`, `embed_text_chunks_auto`)
  - **assets.rs**: Platform-agnostic asset loading (Fetch API on web, tokio::fs on desktop)

- **src/crawler/**: Web page crawling (native only, CORS blocks web)
  - **engine.rs**: BFS crawl with depth limits and cycle detection
  - **fetcher.rs**: HTTP fetching with reqwest
  - **parser.rs**: HTML parsing and text extraction (scraper crate)

- **src/storage/**: Platform-specific `DocumentStore` implementations
  - **indexeddb_store.rs**: `IndexedDbDocumentStore` for web (WASM only)

- **src/workers/**: Web Worker for CPU-intensive embedding (WASM only)
  - **mod.rs**: `EmbeddingWorkerClient`, `start_embedding_worker()` export

- **src/processing/**: File processing pipeline
  - **embedder.rs**: `PlatformEmbedder` (worker on web, direct on desktop)
  - **processor.rs**: High-level `process_file_chunks()` pipeline

- **src/components/**: Dioxus UI components
  - **mod.rs**: `App` component, context providers
  - **app_shell/**: Layout (appbar, footer, metrics_pane)
  - **search/**: Search UI (search_view, search_card, result_card, source_preview)
  - **index/**: Index management (index_view, upload_card, file_row, batch_list)
  - **web_crawler.rs**: Crawler UI (desktop only)
  - **worker.rs**: Platform-specific embedding coordination
  - **batch_processor.rs**: Queue management for file processing
  - **file_processing.rs**: File utilities (binary detection, directory traversal)

- **src/platform/**: Platform abstraction utilities
  - **mod.rs**: `run_blocking()`, `run_async()` (tokio on desktop, direct on web)

- **src/utils/**: General utilities
  - **formatting.rs**: Duration and timestamp formatting
  - **signal_ext.rs**: Dioxus signal utilities

### Critical Technical Details

**WASM Memory Configuration** (`.cargo/config.toml`):
- Initial: 128MB, Max: 512MB
- JinaBERT `max_position_embeddings` set to 2048 tokens (default config)
- ALiBi bias size scales as `heads * seq_len^2` (~134MB for 8 heads at 2048 length)
- Model supports up to 8192 tokens but requires more memory

**Acceleration (Platform-Specific)**:
- **macOS Desktop**: Metal + Accelerate (full GPU acceleration)
- **iOS/iPadOS**: Accelerate only (CPU optimized, no Metal)
  - Reason: Metal has compatibility issues in iOS simulator
  - Accelerate provides optimized CPU inference via Apple's BLAS/LAPACK
  - Future: Could enable Metal for real devices only, but CPU mode ensures universal compatibility
- **Web (WASM)**: CPU only (no GPU acceleration)
  - WebGPU backend in development by Candle team
  - **Note**: WASM threading with Rayon was investigated but abandoned due to Rust WebAssembly atomics being fundamentally broken (see `docs/adrs/003-wasm-threading-workaround.md`)
- **Linux/Windows x86**: Intel MKL (CPU optimized)

**Web Worker Architecture**:
- **JinaBERT Embedding Worker** (`assets/workers/embedding-worker.js`)
  - Runs embedding inference on separate thread to prevent UI freezing
  - Module worker (uses ES6 imports) loads WASM at `/coppermind/wasm/coppermind.js`
  - Downloads and initializes 65MB model in worker context (takes 30-60s)
  - Communicates via postMessage with serialized Rust types (serde-wasm-bindgen)
- Desktop uses `tokio::spawn_blocking` instead (no worker needed)

**Hybrid Search Architecture**:
- **Vector Search**: `hnsw` crate for semantic similarity (cosine distance)
  - Builds index from document embeddings (512D from JinaBERT)
  - HNSW parameters: M=16 (bidirectional links), M0=32 (layer 0 links)
  - Returns nearest neighbors sorted by similarity score
- **Keyword Search**: BM25 for exact keyword matching
  - Term frequency-inverse document frequency scoring
  - Fast full-text search over document corpus
- **RRF Fusion**: Reciprocal Rank Fusion merges vector + keyword rankings
  - Formula: `score = 1/(k + rank)` where k=60 (standard constant)
  - Combines best of semantic understanding + exact keyword matching
  - Robust to score scale differences between algorithms
- **Storage**: Platform-specific persistence via `DocumentStore` trait
  - Web: IndexedDB for O(1) key lookups (browser-native, zero bundle cost)
  - Desktop: redb (Pure Rust B-tree database) for O(log n) lookups
  - Source tracking for re-upload detection (hash-based change detection)
  - Tombstone-based HNSW deletion with background compaction

**Model Loading**:
- JinaBERT weights (65MB) loaded as Dioxus `Asset` from `crates/coppermind/assets/models/`
- Download models with `./download-models.sh`
- Safetensors weights auto-converted F16→F32 by VarBuilder for WASM compatibility
- `Device::cuda_if_available(0)` falls back to CPU in browser

**Text Chunking Strategy** (platform-specific):
- Uses `text-splitter` crate with custom `TokenizerSizer` to avoid onig (C library)
- **File type detection**: Automatic strategy selection based on file extension
  - `.md`, `.markdown` → MarkdownSplitter (structure-aware, uses pulldown-cmark)
  - `.rs`, `.py`, `.js`, `.java`, `.c`, `.cpp`, `.go` → CodeSplitter (syntax-aware, uses tree-sitter) **native only**
  - All others → TextSplitter (sentence-based, uses ICU4X)
- **Platform behavior**:
  - Web (WASM): Markdown + Text chunking (tree-sitter C code doesn't compile to WASM)
  - Desktop/mobile (native): Markdown + Text + Code chunking (tree-sitter works fine)
- Code files on WASM fall back to generic text chunking (still semantic, just not syntax-aware)

**Dioxus Signal Safety** (from `clippy.toml`):
Never hold these across `.await` points (causes deadlocks):
- `generational_box::GenerationalRef` / `GenerationalRefMut`
- `dioxus_signals::Write`

### Extension Points (Traits)

The codebase uses traits for extensibility and testing:

| Trait | Location | Purpose | Implementations |
|-------|----------|---------|-----------------|
| `Embedder` | `crates/coppermind-core/src/embedding/traits.rs` | Abstract ML inference | `JinaBertEmbedder` |
| `ModelConfig` | `crates/coppermind-core/src/embedding/traits.rs` | Model parameters | `JinaBertConfig` |
| `ChunkingStrategy` | `crates/coppermind-core/src/chunking/mod.rs` | Text splitting | `TextSplitterAdapter`, `MarkdownSplitterAdapter`, `CodeSplitterAdapter` |
| `DocumentStore` | `crates/coppermind-core/src/storage/document_store.rs` | Persistence | `RedbDocumentStore`, `IndexedDbDocumentStore`, `InMemoryDocumentStore` |
| `GpuScheduler` | `crates/coppermind-core/src/gpu/scheduler.rs` | GPU access | `SerialScheduler` |

These traits define clean boundaries for:
- Swapping implementations (e.g., different embedding models)
- Testing with mocks
- Platform-specific behavior behind a common interface

### Deployment

**GitHub Pages**:
- Base path: `/coppermind` (set in `Dioxus.toml`)
- Deploys from `target/dx/coppermind/release/web/public` on push to `main`
- CI workflow (`.github/workflows/ci.yml`) runs fmt, clippy, tests, audit, and WASM build
- Multiple jobs require `./download-models.sh` to run before building/testing

## Development Workflow

### Platform Strategy
**Primary development:** Web (faster iteration with `dx serve -p coppermind`)
**Secondary testing:** Desktop (`dx serve -p coppermind --platform desktop`)
**Future:** Mobile support

### Code Style & Architecture

**Idiomatic Dioxus (Preferred):**
- Follow Dioxus recommended patterns and architecture
- Use hooks, components, and signals idiomatically
- Reference: https://dioxuslabs.com/learn/0.5/reference

**Pragmatic Exceptions (Allowed):**
- If a non-idiomatic approach is **significantly simpler, easier, or lower effort**, use it
- Prioritize: Working code > Perfect idioms
- Document why you chose a different approach
- Example: Simple loop vs. complex iterator chain when loop is clearer

**Cross-Platform Development Patterns:**
```rust
// 1. Platform-specific dependencies in Cargo.toml
[target.'cfg(target_arch = "wasm32")'.dependencies]
web-sys = { version = "0.3", features = ["FileSystemDirectoryHandle", ...] }
js-sys = "0.3"

[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
tokio = { version = "1", features = ["fs", "rt", "macros"] }

// 2. Platform-specific launch (IMPORTANT: use feature flags, NOT target_arch)
// Use feature flags to distinguish desktop vs mobile - both are native but have different APIs
#[cfg(feature = "desktop")]
{
    dioxus::LaunchBuilder::desktop().with_cfg(config).launch(App);
}

#[cfg(feature = "mobile")]
{
    dioxus::LaunchBuilder::mobile().launch(App);
}

#[cfg(feature = "web")]
{
    dioxus::launch(App);
}

// 3. Platform-specific imports
#[cfg(target_arch = "wasm32")]
use IndexedDbDocumentStore;  // Web-specific

#[cfg(not(target_arch = "wasm32"))]
use RedbDocumentStore;  // Desktop/mobile

// 4. Logging with dioxus::logger (works on all platforms)
use dioxus::logger::tracing::{info, error};
info!("Message");  // → Browser console on web, stdout on desktop

// 5. CSS loading (asset! macro has issues on desktop)
if cfg!(target_arch = "wasm32") {
    document::Stylesheet { href: MAIN_CSS }  // Web: use asset!
} else {
    style { {include_str!("../assets/coppermind.css")} }  // Desktop: embed directly
}

// 6. Conditional code blocks in rsx!
if cfg!(target_arch = "wasm32") {
    document::Script { src: "/coi-serviceworker.min.js" }  // Web only
}

// 7. Fallback implementations for doc/test builds
// When code requires platform features but must compile for `cargo test` or `cargo doc`,
// provide a fallback cfg that uses InMemoryDocumentStore or stub implementations:
#[cfg(all(
    not(target_arch = "wasm32"),
    not(feature = "desktop"),
    not(feature = "mobile")
))]
type PlatformDocumentStore = InMemoryDocumentStore;

// CI runs tests with --features desktop, so platform code is properly tested.
// The fallback only enables compilation for edge cases like `cargo test` without features.
```

### Milestone Completion Requirements

**Every milestone MUST:**
1. **Pass all quality checks:** Run `.githooks/pre-commit` successfully
   - This covers: fmt, clippy, tests, cargo audit, web build
2. **Work on all platforms:** Test with `dx serve -p coppermind` (web), `dx serve -p coppermind --platform desktop`, and `dx build -p coppermind --platform ios`
3. **Update documentation:** After completing each milestone, you MUST:
   - Update `CLAUDE.md` if architecture or module structure changed
   - Update relevant `docs/*.md` files with new patterns/implementations
   - Update `docs/roadmap.md` to mark milestone complete
   - Add new documentation if introducing new concepts

**Commit only when:**
- `.githooks/pre-commit` passes
- Both platforms tested and working
- Documentation updated

### Testing Strategy

**Automated Testing (Preferred):**
- Write unit tests for business logic
- Write integration tests when possible
- Run tests with `cargo test --verbose`
- Automated tests are preferred over manual testing

**Manual Testing (When Required):**
- **IMPORTANT:** Do NOT run interactive commands like `dx serve -p coppermind` yourself
- Instead, ask the user to run them and provide specific UAT (User Acceptance Testing) steps
- Example: "Please run `dx serve -p coppermind` and verify: [specific checklist]"
- User must manually open browser and perform UAT actions

**Before Committing:**
```bash
# 1. Run pre-commit hook (includes tests)
./.githooks/pre-commit           # Must pass

# 2. If feature needs manual testing, ask user:
# "Please test both platforms with these UAT steps:
#  Web: dx serve -p coppermind
#    - [ ] Step 1
#    - [ ] Step 2
#  Desktop: dx serve -p coppermind --platform desktop
#    - [ ] Step 1
#    - [ ] Step 2"
```

## Documentation

The `docs/` directory contains detailed technical documentation:

### [Roadmap & Implementation Plan](docs/roadmap.md)
Cross-platform development roadmap with completed features and future plans.

**Key Topics:**
- Completed features (hybrid search, crawler, GPU scheduler, persistence)
- Platform strategy (Web → Desktop → Mobile)
- Backlog (WebGPU, quantization, multi-model support)

**When to Read:** Before starting new features, to understand project direction.

### [Architecture Design](docs/architecture-design.md)
Comprehensive technical design document covering the entire system.

**Key Topics:**
- Hybrid search system (HNSW, BM25, RRF fusion)
- Browser ML with Candle (JinaBERT embeddings)
- Web Worker architecture for non-blocking inference
- Cross-platform compilation (web vs desktop)
- Storage & persistence (IndexedDB for web, redb for desktop)

**When to Read:** To understand how the system works and implementation details.

### [Ecosystem & Limitations](docs/ecosystem-and-limitations.md)
Community resources, known limitations, and ecosystem integration details.

**Key Topics:**
- Technology stack integration (Candle + WASM + Dioxus)
- Known limitations and workarounds
- Community implementations (Transformers.js, ONNX Runtime)
- Future directions (WebGPU backend, quantization)

**When to Read:** When evaluating alternatives or troubleshooting limitations.

### [Configuration Options](docs/config-options.md)
Catalog of all configurable options for future preferences implementation.

**When to Read:** When modifying default values or implementing user preferences.

### [Profiling Guide](docs/profiling.md)
How to profile Coppermind to diagnose performance issues.

**When to Read:** When debugging performance problems or UI lag.

---

## Documentation Writing Guidelines

When writing documentation (README, docs/, etc.), follow these principles:

### Tone & Style

**Professional, Not Marketing:**
- Write like a senior engineer presenting technical work to peers
- Be humble—no bragging or self-promotion
- Avoid buzzwords and marketing language
- Be fact-based: make claims you can verify from the codebase
- Position work as "showing off something cool I built" not as a product pitch

**Technical Depth:**
- Assume the reader has general engineering knowledge
- Explain newer/niche concepts that even experienced engineers may not know:
  - Web Workers
  - Specific algorithms (HNSW, RRF, ALiBi)
  - Tombstone-based deletion and compaction
- Don't explain basic concepts like "embeddings" or "BM25" unless necessary
- Technical details are the coolest part—be detailed but concise

**Structure:**
- Use headers to organize content logically
- Link to external resources (algorithms, papers, official docs)
- **DO NOT link to internal docs** when writing top-level documentation

### Accuracy & Verification

**Critical: Verify All Claims**
- Check the actual code before making technical claims
- Don't guess at implementation details
- Avoid including performance numbers/timings unless actually tested
- Don't include file sizes unless necessary (let readers look them up)
- Example corrections from README exercise:
  - ❌ "Candle is for embedded and WASM environments"
  - ✅ "Candle, a minimalist Rust ML framework from Hugging Face"
  - ❌ "100% Rust"
  - ✅ "Rust-First" (acknowledge necessary JavaScript: web worker bootstrap)

**RRF Example:**
- Claim: "without requiring score normalization"
- Verification: Check `src/search/fusion.rs` line 27-28—it uses `(rank, (item, _score))`, ignoring scores
- Conclusion: ✅ Technically correct—RRF is purely rank-based

### What to Include / Exclude

**Include:**
- How the technology works (architecture, algorithms, design decisions)
- Why certain choices were made over alternatives
- Links to external resources (academic papers, library docs, specifications)
- Building/setup instructions for developers
- Current functionality as implemented in code

**Exclude:**
- Future plans, roadmap, or vision (keep in separate roadmap docs)
- "Demo" language—just provide links to live deployments
- Timing/performance data unless rigorously tested
- Installation instructions for end-users (this is a developer project)
- Empty promises or aspirational features

### Formatting Preferences

Based on user edits to README:
- Use simple dashes `-` not em-dashes `—` in prose
- Tech stack: Use dash separator ` - ` not ` — `
- Links: "Try it here:" not "Try it:" or "Demo:"
- Descriptions: Concise and direct
  - ✅ "A local-first cross-platform hybrid search engine"
  - ❌ "A local-first hybrid search engine running ML inference directly in your browser via WebAssembly, built with as much Rust as possible"
- Bullet points: "Fully Local" not "Local-Only"

### Technical Stack Presentation

Present like a senior engineer (from README exercise):
- Organize by logical categories
- Include version numbers
- Add brief descriptions in parentheses (NOT marketing copy)
- Link to project pages
- Example format:
  ```markdown
  ### Core Framework
  - **[Dioxus 0.7](https://dioxuslabs.com/)** - Reactive UI framework (React-like component model, cross-platform rendering)
  - **[Candle 0.8](https://github.com/huggingface/candle)** - ML inference framework (Hugging Face Rust ML library)
  ```

### Common Pitfalls to Avoid

1. **Overclaiming**: Don't say "entirely in Rust" when JavaScript is required
2. **Vague performance**: No "~100-500ms" unless tested on representative hardware
3. **Misleading descriptions**:
   - ❌ Candle is "designed for embedded" (it's a general ML framework)
   - ❌ instant-distance "doesn't use rayon" (it does—check cargo tree)
4. **Normalization confusion**: Verify technical claims about algorithms
5. **Size inflation**: Don't quote file sizes in multiple places inconsistently

### Verification Checklist

Before publishing documentation:
- [ ] All technical claims verified against actual code
- [ ] No performance numbers unless tested
- [ ] No file sizes quoted (or consistent if needed)
- [ ] External links point to authoritative sources
- [ ] Tone is professional and humble
- [ ] Newer concepts (HNSW, tombstone deletion, etc.) are explained
- [ ] Tech stack formatted like senior engineer presentation
