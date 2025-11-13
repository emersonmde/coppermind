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
dx serve                           # Development server (web platform)
dx serve --platform desktop        # Desktop platform
dx bundle --release                # Production build for deployment
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
- **main.rs**: Entry point, platform-specific logging and CSS loading, mounts root App component
- **components.rs**: UI components (`TestControls` with file upload and test buttons)
- **embedding.rs**: JinaBERT model loading, tokenization, inference using Candle
- **cpu.rs**: Web Worker spawning for parallel CPU computation
- **wgpu.rs**: WebGPU compute shader setup and execution
- **search/**: Hybrid search system (vector + keyword + RRF fusion)
  - **engine.rs**: `HybridSearchEngine` orchestrating vector and keyword search
  - **vector.rs**: HNSW semantic search using instant-distance (cosine similarity)
  - **keyword.rs**: BM25 full-text search for exact keyword matching
  - **fusion.rs**: Reciprocal Rank Fusion (RRF) algorithm for merging rankings
  - **types.rs**: Shared types (`DocId`, `Document`, `SearchResult`, `SearchError`)
- **storage/**: Cross-platform persistence layer
  - **mod.rs**: `StorageBackend` trait for key-value storage abstraction
  - **opfs.rs**: OPFS (Origin Private File System) for web platform
  - **native.rs**: tokio::fs-based storage for desktop platform

### Critical Technical Details

**WASM Memory Configuration** (`.cargo/config.toml`):
- Initial: 128MB, Max: 512MB
- JinaBERT `max_position_embeddings` limited to 1024 to avoid multi-GB ALiBi tensor allocation
- ALiBi bias size scales as `heads * seq_len^2` (~32MB for 8 heads at 1024 length)

**Cross-Origin Isolation**:
- Service Worker (`public/coi-serviceworker.min.js`) injects COOP/COEP headers
- Required for SharedArrayBuffer support
- Conditionally loaded only on web: `if cfg!(target_arch = "wasm32")`
- Must be unhashed and in `public/` (not bundled via `asset!()`)

**Hybrid Search Architecture**:
- **Vector Search**: instant-distance HNSW for semantic similarity (cosine distance)
  - Builds index from document embeddings (512D from JinaBERT)
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

**Dioxus Signal Safety** (from `clippy.toml`):
Never hold these across `.await` points (causes deadlocks):
- `generational_box::GenerationalRef` / `GenerationalRefMut`
- `dioxus_signals::Write`

### Deployment

**GitHub Pages**:
- Base path: `/coppermind` (set in `Dioxus.toml`)
- Deploys from `target/dx/coppermind/release/web/public` on push to `main`
- CI workflow (`.github/workflows/ci.yml`) runs fmt, clippy, tests, audit, and WASM build
- Multiple jobs require `./download-models.sh` to run before building/testing

## Development Workflow

### Platform Strategy
**Primary development:** Web (faster iteration with `dx serve`)
**Secondary testing:** Desktop (`dx serve --platform desktop`)
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

// 2. Platform-specific imports
#[cfg(target_arch = "wasm32")]
use OpfsStorage;  // Web-specific

#[cfg(not(target_arch = "wasm32"))]
use NativeStorage;  // Desktop/mobile

// 3. Logging with dioxus::logger (works on all platforms)
use dioxus::logger::tracing::{info, error};
info!("Message");  // → Browser console on web, stdout on desktop

// 4. CSS loading (asset! macro has issues on desktop)
if cfg!(target_arch = "wasm32") {
    document::Stylesheet { href: MAIN_CSS }  // Web: use asset!
} else {
    style { {include_str!("../assets/main.css")} }  // Desktop: embed directly
}

// 5. Conditional code blocks in rsx!
if cfg!(target_arch = "wasm32") {
    document::Script { src: "/coi-serviceworker.min.js" }  // Web only
}
```

### Milestone Completion Requirements

**Every milestone MUST:**
1. **Pass all quality checks:** Run `.githooks/pre-commit` successfully
   - This covers: fmt, clippy, tests, cargo audit, web build
2. **Work on both platforms:** Test with both `dx serve` and `dx serve --platform desktop`
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
- **IMPORTANT:** Do NOT run interactive commands like `dx serve` yourself
- Instead, ask the user to run them and provide specific UAT (User Acceptance Testing) steps
- Example: "Please run `dx serve` and verify: [specific checklist]"
- User must manually open browser and perform UAT actions

**Before Committing:**
```bash
# 1. Run pre-commit hook (includes tests)
./.githooks/pre-commit           # Must pass

# 2. If feature needs manual testing, ask user:
# "Please test both platforms with these UAT steps:
#  Web: dx serve
#    - [ ] Step 1
#    - [ ] Step 2
#  Desktop: dx serve --platform desktop
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
