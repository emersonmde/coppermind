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
  - Future: rexie/rusqlite for storage

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

### Setup
```bash
./download-models.sh               # Download JinaBERT model (262MB + 695KB tokenizer)
git config core.hooksPath .githooks  # Enable pre-commit checks
```

Install required tools:
```bash
cargo install dioxus-cli --locked
cargo install cargo-audit --locked
```

## Architecture

### Module Organization

- **main.rs**: Entry point with platform-specific launch (desktop/mobile/web), logging, CSS loading
- **lib.rs**: Public API surface, module exports, crate-level `#![forbid(unsafe_code)]`
- **error.rs**: Error types (`EmbeddingError`, `FileProcessingError`) with `thiserror` derive

- **embedding/**: ML model inference and text processing
  - **mod.rs**: High-level API (`compute_embedding`, `embed_text_chunks_auto`, `get_or_load_model`), WASM bindings
  - **config.rs**: `ModelConfig` trait and `JinaBertConfig` implementation (512-dim, 4 layers, 8 heads)
  - **model.rs**: `Embedder` trait and `JinaBertEmbedder` (Candle-based, mean pooling, L2 normalization)
  - **tokenizer.rs**: Singleton tokenizer initialization with truncation config
  - **assets.rs**: Platform-agnostic asset loading (Fetch API on web, tokio::fs on desktop)
  - **chunking/**: Text chunking strategies
    - **mod.rs**: `ChunkingStrategy` trait, `FileType` enum, `detect_file_type()`
    - **text_splitter_adapter.rs**: ICU4X sentence-based chunking with custom `TokenizerSizer`
    - **markdown_splitter_adapter.rs**: Markdown-aware chunking (pulldown-cmark)
    - **code_splitter_adapter.rs**: Syntax-aware chunking with tree-sitter (native only)

- **search/**: Hybrid search system (vector + keyword + RRF fusion)
  - **mod.rs**: Public exports
  - **types.rs**: `DocId`, `Document`, `DocumentMetadata`, `DocumentRecord`, `SearchResult`, `FileSearchResult`, `SearchError`
  - **engine.rs**: `HybridSearchEngine` orchestrating vector + keyword search
  - **vector.rs**: HNSW semantic search using `hnsw` crate (cosine distance)
  - **keyword.rs**: BM25 full-text search using `bm25` crate
  - **fusion.rs**: Reciprocal Rank Fusion (RRF) algorithm for merging rankings
  - **aggregation.rs**: `aggregate_chunks_by_file()` for file-level result grouping

- **storage/**: Cross-platform persistence layer
  - **mod.rs**: `StorageBackend` trait (save/load/exists/delete/list_keys/clear)
  - **opfs.rs**: OPFS (Origin Private File System) for web (WASM only)
  - **native.rs**: tokio::fs-based storage for desktop

- **workers/**: Web Worker for CPU-intensive embedding (WASM only)
  - **mod.rs**: `EmbeddingWorkerClient` (spawns worker, manages request/response), `start_embedding_worker()` WASM export

- **processing/**: File processing pipeline
  - **mod.rs**: Exports, `ChunkProcessingResult`
  - **embedder.rs**: `PlatformEmbedder` enum (worker on web, direct calls on desktop)
  - **processor.rs**: High-level `process_file_chunks()` pipeline

- **crawler/**: Web page crawling (native only, CORS blocks web)
  - **mod.rs**: `CrawlConfig`, `CrawlResult`, `CrawlProgress`, `CrawlError`
  - **fetcher.rs**: HTTP fetching with reqwest
  - **parser.rs**: HTML parsing and text extraction (scraper crate)
  - **engine.rs**: Recursive crawl logic with depth/page limits

- **platform/**: Platform abstraction utilities
  - **mod.rs**: `run_blocking()` and `run_async()` - abstracts tokio::spawn_blocking on desktop, direct execution on web

- **utils/**: General utilities
  - **mod.rs**: Re-exports
  - **error_ext.rs**: `ResultExt` trait for `.context()` error handling
  - **signal_ext.rs**: Dioxus signal utilities

- **components/**: Dioxus UI components
  - **mod.rs**: `App` component, context providers (SearchEngine, StorageBackend)
  - **app_shell/**: Layout (appbar, footer, metrics_pane)
  - **search/**: Search UI (search_view, search_card, result_card, empty_state, source_preview)
  - **index/**: Index management (index_view, upload_card, file_row, batch, batch_list)
  - **worker.rs**: Platform-specific embedding coordination
  - **batch_processor.rs**: Queue management for file processing
  - **web_crawler.rs**: Crawler UI (desktop only)
  - **file_processing.rs**: File utilities (binary detection, directory traversal)
  - **testing.rs**: Developer testing utilities

### Critical Technical Details

**WASM Memory Configuration** (`.cargo/config.toml`):
- Initial: 128MB, Max: 512MB
- JinaBERT `max_position_embeddings` set to 2048 tokens (default config)
- ALiBi bias size scales as `heads * seq_len^2` (~134MB for 8 heads at 2048 length)
- Model supports up to 8192 tokens but requires more memory (see `docs/model-optimization.md`)

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
- **Storage**: Platform-specific persistence via `StorageBackend` trait
  - Web: OPFS (Origin Private File System) for large binary data
  - Desktop: tokio::fs for native filesystem access

**Model Loading**:
- JinaBERT weights loaded as Dioxus `Asset` from `/assets/models/`
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
| `Embedder` | `embedding/model.rs` | Abstract ML inference | `JinaBertEmbedder` |
| `ModelConfig` | `embedding/config.rs` | Model parameters | `JinaBertConfig` |
| `ChunkingStrategy` | `embedding/chunking/mod.rs` | Text splitting | `TextSplitterAdapter`, `MarkdownSplitterAdapter`, `CodeSplitterAdapter` |
| `StorageBackend` | `storage/mod.rs` | Persistence | `OpfsStorage`, `NativeStorage`, `InMemoryStorage` |

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
use OpfsStorage;  // Web-specific

#[cfg(not(target_arch = "wasm32"))]
use NativeStorage;  // Desktop/mobile

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
Cross-platform development roadmap with detailed milestones.

**Key Topics:**
- Platform strategy (Web → Desktop → Mobile)
- Current implementation status
- Detailed milestones with completion checklists
- Architecture evolution
- Performance targets

**When to Read:** Before starting new features, to understand the project direction and milestone requirements.

### [Model Optimization Guide](docs/model-optimization.md)
Comprehensive guide on WASM memory configuration and JinaBERT sequence length optimization.

**Key Topics:**
- WASM memory limits (4GB vs current 512MB)
- JinaBERT sequence length capabilities (8192 tokens vs current 1024)
- ALiBi memory calculations and tradeoffs
- Recommended configurations for different use cases
- Memory budget breakdowns

**When to Read:** Before optimizing model performance or increasing sequence lengths.

### [Browser ML Architecture](docs/browser-ml-architecture.md)
Deep dive into browser-based ML inference patterns and architecture.

**Key Topics:**
- Cross-Origin Isolation (COOP/COEP) setup via Service Worker
- Web Workers for parallel processing
- WebGPU compute shader implementation
- Model loading patterns (singleton, lazy loading)
- Performance characteristics and memory footprint
- Future architecture directions (worker pools, OPFS)

**When to Read:** When implementing new ML features or debugging WASM/WebGPU issues.

### [Ecosystem & Limitations](docs/ecosystem-and-limitations.md)
Community resources, known limitations, and ecosystem integration details.

**Key Topics:**
- Technology stack integration (Candle + WASM + Dioxus + WebGPU)
- Known limitations and workarounds
- Community implementations (Transformers.js, ONNX Runtime, etc.)
- Blog posts, tutorials, and learning resources
- Performance comparisons across different approaches
- Future directions (WebGPU backend, WASM threads, quantization)

**When to Read:** When evaluating alternatives, troubleshooting limitations, or learning from community patterns.

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
  - OPFS (Origin Private File System)
  - Specific algorithms (HNSW, RRF, ALiBi)
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
- [ ] Newer concepts (OPFS, HNSW, etc.) are explained
- [ ] Tech stack formatted like senior engineer presentation
