# ADR-005: Workspace Structure for Library Extraction

**Status:** Implemented
**Date:** 2025-11-25
**Supersedes:** None

## Update (2025-11-29): Feature Flags vs Platform Cfg

The original ADR proposed using feature flags (`tree-sitter`, `candle`) to gate optional functionality. After implementation, we revised this approach:

**Original Decision (ADR proposal):**
- `candle` feature flag to optionally enable embedding support
- `tree-sitter` feature flag for syntax-aware code chunking

**Revised Decision (current implementation):**
- **Candle is always-on**: All platforms require embedding capability - there's no use case for coppermind-core without embedding
- **Tree-sitter uses platform cfg**: `#[cfg(not(target_arch = "wasm32"))]` instead of feature flag, because tree-sitter's C code simply doesn't compile to WASM

**Rationale for change:**
1. Feature flags add cognitive overhead when the choice isn't actually optional
2. Candle is required for the core value proposition (semantic search)
3. Tree-sitter availability is determined by platform capability, not user choice
4. Platform cfg is more accurate: "this code physically cannot compile on WASM" vs "user chose not to enable this"

The rest of the ADR remains valid for understanding the workspace structure and module boundaries.

---

## Summary

Extract platform-independent code into a `coppermind-core` library crate to improve testability, enable CLI/MCP development, and establish clear architectural boundaries.

---

## Context

### Current State

Coppermind is a single-crate Dioxus application with platform-specific code mixed throughout:

```
src/
├── embedding/      # ML inference (platform-independent) + asset loading (platform-specific)
├── search/         # Pure algorithms (platform-independent)
├── storage/        # OPFS (web-only) + NativeStorage (desktop/mobile)
├── workers/        # WASM web workers (web-only)
├── platform/       # Execution abstraction (platform-specific)
├── processing/     # UI-aware pipeline (couples to Dioxus signals)
├── crawler/        # HTTP crawling (native-only, blocked by CORS on web)
└── components/     # Dioxus UI (platform-specific)
```

### Problems

1. **Testing friction**: Testing pure algorithms (HNSW, BM25, RRF) requires Dioxus compilation and platform toolchains
2. **No library reuse**: Cannot build a CLI or MCP server without pulling in all UI dependencies
3. **Unclear boundaries**: Platform-specific code is interleaved with platform-independent logic
4. **Extension difficulty**: Adding new embedding models or search backends requires touching UI-coupled code

### Goals

| Priority | Goal | Rationale |
|----------|------|-----------|
| 1 | **Ease of testing** | Core algorithms testable with plain `cargo test` |
| 2 | **CLI/MCP enablement** | Reuse search and embedding for non-GUI interfaces |
| 3 | **Clear extension points** | Traits define where implementations can be swapped |
| 4 | **No performance regression** | Same memory usage, lazy loading, caching behavior |
| 5 | **Incremental migration** | Each step leaves codebase in working state |

### Non-Goals

- Reducing CI costs (not a priority per user)
- Supporting multiple UI frameworks
- Micro-crate architecture (two crates is sufficient)

---

## Decision

Restructure as a Cargo workspace with two crates:

- **`coppermind-core`**: Platform-independent library (no Dioxus, no WASM-specific code)
- **`coppermind`**: Dioxus application consuming the core library

Future addition: **`coppermind-cli`** for command-line and MCP server access.

---

## Detailed Design

### Workspace Structure

```
coppermind/
├── Cargo.toml                    # [workspace]
├── Dioxus.toml                   # Points to crates/coppermind
├── crates/
│   ├── coppermind-core/          # Platform-independent library
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── error.rs
│   │       ├── embedding/
│   │       │   ├── mod.rs        # Core types, no global state
│   │       │   ├── config.rs     # ModelConfig trait, JinaBertConfig
│   │       │   ├── model.rs      # Embedder trait, JinaBertEmbedder
│   │       │   ├── tokenizer.rs  # Tokenizer wrapper
│   │       │   └── chunking/
│   │       │       ├── mod.rs
│   │       │       ├── text_splitter_adapter.rs
│   │       │       ├── markdown_splitter_adapter.rs
│   │       │       └── code_splitter_adapter.rs  # feature = "tree-sitter"
│   │       ├── search/
│   │       │   ├── mod.rs
│   │       │   ├── types.rs
│   │       │   ├── engine.rs
│   │       │   ├── vector.rs
│   │       │   ├── keyword.rs
│   │       │   ├── fusion.rs
│   │       │   └── aggregation.rs
│   │       ├── storage/
│   │       │   ├── mod.rs        # StorageBackend trait
│   │       │   ├── memory.rs     # InMemoryStorage (testing)
│   │       │   └── native.rs     # NativeStorage (tokio::fs)
│   │       ├── crawler/          # feature = "crawler"
│   │       │   ├── mod.rs
│   │       │   ├── fetcher.rs
│   │       │   ├── parser.rs
│   │       │   └── engine.rs
│   │       └── utils/
│   │           ├── mod.rs
│   │           └── error_ext.rs
│   │
│   └── coppermind/               # Dioxus application
│       ├── Cargo.toml
│       ├── assets/
│       └── src/
│           ├── main.rs
│           ├── lib.rs
│           ├── app/              # Application-level coordination
│           │   ├── mod.rs
│           │   ├── model_loader.rs   # Singleton, asset fetching
│           │   └── storage.rs        # Platform storage selection
│           ├── components/
│           │   ├── mod.rs
│           │   ├── app_shell/
│           │   ├── search/
│           │   ├── index/
│           │   └── ...
│           ├── workers/          # WASM web worker client
│           │   └── mod.rs
│           ├── platform/         # run_blocking, run_async
│           │   └── mod.rs
│           ├── storage/          # WASM-specific storage only
│           │   └── opfs.rs       # OpfsStorage
│           ├── processing/       # UI-aware pipeline
│           │   ├── mod.rs
│           │   ├── embedder.rs
│           │   └── processor.rs
│           └── utils/
│               └── signal_ext.rs
```

---

## Key Design Decisions

### 1. Model Loading Boundary

**Decision**: Core provides stateless constructors. Application manages lifecycle (singleton, lazy loading).

**Current code** (problematic for library extraction):
```rust
// embedding/mod.rs - global state in library
static MODEL: OnceCell<Arc<dyn Embedder>> = OnceCell::new();

pub async fn get_or_load_model() -> Result<Arc<dyn Embedder>, EmbeddingError> {
    if let Some(model) = MODEL.get() {
        return Ok(model.clone());
    }
    let weights = fetch_asset_bytes("models/model.safetensors").await?;  // Platform-specific!
    // ...
}
```

**Problems with keeping singleton in core**:
- Core needs to know how to fetch bytes (platform-specific)
- Global state makes testing difficult
- CLI might want different lifecycle (load on demand vs startup)

**Proposed design**:
```rust
// coppermind-core: Pure, stateless, testable
impl JinaBertEmbedder {
    /// Construct from already-loaded bytes. No I/O, no global state.
    pub fn from_bytes(
        weights: &[u8],
        tokenizer_json: &[u8],
        device: &Device,
    ) -> Result<Self, EmbeddingError> {
        let tensors = safetensors::load_buffer(weights)?;
        let tokenizer = Tokenizer::from_bytes(tokenizer_json)?;
        // ... pure construction
    }
}

// coppermind (app): Manages lifecycle
// app/model_loader.rs
static MODEL: OnceCell<Arc<dyn Embedder>> = OnceCell::new();

pub async fn get_or_load_model() -> Result<Arc<dyn Embedder>, EmbeddingError> {
    if let Some(model) = MODEL.get() {
        return Ok(model.clone());
    }

    // Platform-specific fetching
    let weights = fetch_asset_bytes("models/model.safetensors").await?;
    let tokenizer = fetch_asset_bytes("models/tokenizer.json").await?;

    // Delegate to core for pure construction
    let embedder = JinaBertEmbedder::from_bytes(&weights, &tokenizer, &Device::Cpu)?;

    MODEL.set(Arc::new(embedder)).ok();
    Ok(MODEL.get().unwrap().clone())
}
```

**Tradeoffs**:

| Aspect | Singleton in Core | Singleton in App (chosen) |
|--------|-------------------|---------------------------|
| Testability | Harder (global state) | Easier (pure constructors) |
| CLI flexibility | Fixed lifecycle | CLI controls own lifecycle |
| Code location | One place | Split (constructor vs lifecycle) |
| Platform coupling | Core needs asset loading | Clean separation |

**Why this is correct**: Libraries should be pure and stateless. Lifecycle management is an application concern. This follows the pattern used by major libraries (e.g., `reqwest::Client` is constructed by app, not a global).

### 2. Storage Architecture

**Decision**: Trait and native implementation in core. OPFS (web-only) stays in app.

**Platform analysis**:

| Platform | Storage Mechanism | Implementation |
|----------|-------------------|----------------|
| Desktop (macOS, Linux, Windows) | Filesystem | `NativeStorage` (tokio::fs) |
| iOS | Sandboxed filesystem | `NativeStorage` (tokio::fs) |
| Android | Sandboxed filesystem | `NativeStorage` (tokio::fs) |
| Web (WASM) | Origin Private File System | `OpfsStorage` (web-sys) |

**Key insight**: OPFS is the outlier. All native platforms use filesystem APIs. Putting `OpfsStorage` in core would:
- Add `web-sys`, `js-sys`, `wasm-bindgen` as core dependencies (even if cfg-gated)
- Pollute the core library with browser-specific code
- Provide no benefit to CLI or native platforms

**Proposed design**:
```rust
// coppermind-core/storage/mod.rs
#[async_trait]
pub trait StorageBackend: Send + Sync {
    async fn save(&self, key: &str, data: &[u8]) -> Result<(), StorageError>;
    async fn load(&self, key: &str) -> Result<Vec<u8>, StorageError>;
    async fn exists(&self, key: &str) -> Result<bool, StorageError>;
    async fn delete(&self, key: &str) -> Result<(), StorageError>;
    async fn list_keys(&self) -> Result<Vec<String>, StorageError>;
    async fn clear(&self) -> Result<(), StorageError>;
}

// coppermind-core/storage/memory.rs - For testing
pub struct InMemoryStorage { /* ... */ }

// coppermind-core/storage/native.rs - For all native platforms
pub struct NativeStorage {
    base_path: PathBuf,
}

// coppermind (app)/storage/opfs.rs - Web only, uses web-sys
#[cfg(target_arch = "wasm32")]
pub struct OpfsStorage { /* ... */ }
```

**App-level platform selection**:
```rust
// coppermind/app/storage.rs
pub fn create_storage() -> Box<dyn StorageBackend> {
    #[cfg(target_arch = "wasm32")]
    { Box::new(OpfsStorage::new()) }

    #[cfg(not(target_arch = "wasm32"))]
    { Box::new(NativeStorage::new(get_data_dir())) }
}
```

**Tradeoffs**:

| Aspect | All in Core | Split (chosen) |
|--------|-------------|----------------|
| Core dependencies | web-sys, js-sys (cfg-gated) | Pure Rust only |
| CLI usage | Works | Works (uses NativeStorage) |
| Code organization | One location | Split by platform nature |
| Consistency | All storage together | Follows platform boundary |

**Why this is correct**: The trait abstraction is the important part - it enables testing with `InMemoryStorage` and swapping implementations. The web-specific implementation doesn't need to be in core because no non-web consumer will ever use it.

### 3. WASM Bindings Location

**Decision**: All WASM/JS interop code stays in app.

**Affected code**:
- `workers/mod.rs`: `EmbeddingWorkerClient`, `start_embedding_worker()` wasm_bindgen export
- Error type `impl From<EmbeddingError> for JsValue`

**Rationale**: These are UI-layer concerns for web platform:
- Web workers are a browser API for preventing UI freezes
- Desktop uses `tokio::spawn_blocking` instead
- CLI has no UI to freeze

**Error handling approach**:
```rust
// coppermind-core/error.rs - Pure Rust errors
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("Model loading failed: {0}")]
    ModelLoad(String),
    // ...
}

// coppermind (app) - WASM conversion where needed
#[cfg(target_arch = "wasm32")]
impl From<EmbeddingError> for JsValue {
    fn from(e: EmbeddingError) -> JsValue {
        JsValue::from_str(&e.to_string())
    }
}
```

### 4. Crawler Module

**Decision**: Crawler lives in core with feature flag.

**Rationale**:
- Crawler is blocked by CORS on web (can't make cross-origin requests from browser)
- But it's not UI-specific - CLI would want crawling capability
- Pure Rust implementation (reqwest, scraper)

```toml
# coppermind-core/Cargo.toml
[features]
default = []
crawler = ["dep:reqwest", "dep:scraper", "dep:url"]

[dependencies]
reqwest = { version = "0.11", optional = true }
scraper = { version = "0.18", optional = true }
url = { version = "2", optional = true }
```

**Usage**:
```toml
# coppermind/Cargo.toml (app)
[features]
desktop = ["dioxus/desktop", "coppermind-core/crawler"]
web = ["dioxus/web"]  # No crawler on web
```

### 5. Code Chunking (tree-sitter)

**Decision**: Code chunking behind feature flag in core.

**Current situation**:
- `CodeSplitterAdapter` uses tree-sitter for syntax-aware chunking
- tree-sitter has C dependencies that don't compile to WASM
- Currently cfg-gated with `#[cfg(not(target_arch = "wasm32"))]`

**Problem with cfg-gate**: Can't test code chunking on a Linux CI machine targeting WASM.

**Solution**: Feature flag instead of target detection:
```toml
# coppermind-core/Cargo.toml
[features]
default = []
tree-sitter = [
    "dep:tree-sitter",
    "dep:tree-sitter-rust",
    "dep:tree-sitter-python",
    # ...
]
```

```rust
// coppermind-core/embedding/chunking/mod.rs
#[cfg(feature = "tree-sitter")]
pub mod code_splitter_adapter;

pub fn detect_file_type(filename: &str) -> FileType {
    let ext = Path::new(filename).extension();
    match ext.and_then(|e| e.to_str()) {
        Some("md" | "markdown") => FileType::Markdown,
        #[cfg(feature = "tree-sitter")]
        Some("rs" | "py" | "js" | ...) => FileType::Code,
        _ => FileType::Text,
    }
}
```

**Tradeoffs**:

| Aspect | cfg(target_arch) | Feature flag (chosen) |
|--------|------------------|----------------------|
| WASM behavior | Auto-disabled | Explicit opt-out |
| Native testing | Always included | Opt-in |
| CI flexibility | Target-dependent | Feature-dependent |
| Explicit intent | Implicit | Explicit |

---

## Feature Flag Summary

### `coppermind-core`

```toml
[features]
default = []
tree-sitter = [...]  # Syntax-aware code chunking
crawler = [...]      # Web page crawling
```

### `coppermind` (app)

```toml
[features]
default = ["web"]
web = ["dioxus/web"]
desktop = ["dioxus/desktop", "coppermind-core/tree-sitter", "coppermind-core/crawler"]
mobile = ["dioxus/mobile", "coppermind-core/tree-sitter"]
```

---

## Migration Plan

Each PR leaves codebase in working, tested state.

### Phase 1: Workspace Setup + Search (Lowest Risk)

**PR 1.1**: Create workspace structure
- Create `crates/coppermind-core/` with empty `lib.rs`
- Create `crates/coppermind/` mirroring current `src/`
- Update root `Cargo.toml` to workspace
- Update `Dioxus.toml` to point to `crates/coppermind`
- **Verification**: `dx serve` works, all tests pass

**PR 1.2**: Move `search/` module
- Move `src/search/*` to `crates/coppermind-core/src/search/`
- Update imports in app to `use coppermind_core::search::*`
- **Verification**: All search tests pass, app builds

### Phase 2: Embedding Module

**PR 2.1**: Move chunking strategies
- Move `embedding/chunking/*` to core
- Add `tree-sitter` feature flag
- **Verification**: Chunking tests pass

**PR 2.2**: Move embedding core (config, model, tokenizer)
- Move `embedding/{config,model,tokenizer}.rs` to core
- Keep `embedding/mod.rs` and `embedding/assets.rs` in app
- **Verification**: Embedding tests pass

**PR 2.3**: Split embedding/mod.rs
- Extract core types to `coppermind-core/embedding/mod.rs`
- Keep singleton and asset loading in `coppermind/app/model_loader.rs`
- **Verification**: Full embedding pipeline works

### Phase 3: Storage

**PR 3.1**: Move storage trait and native implementation
- Move `StorageBackend` trait to core
- Move `NativeStorage` to core
- Move `InMemoryStorage` to core (or create if doesn't exist)
- Keep `OpfsStorage` in app
- **Verification**: Storage tests pass on native

### Phase 4: Remaining Modules

**PR 4.1**: Move crawler (with feature flag)
- Move `crawler/*` to core behind `crawler` feature
- **Verification**: Crawler tests pass with `--features crawler`

**PR 4.2**: Move error types and utils
- Move `error.rs` to core (remove JsValue conversion)
- Move `utils/error_ext.rs` to core
- Add JsValue conversion in app
- **Verification**: Error handling works on all platforms

### Phase 5: Cleanup and Documentation

**PR 5.1**: Update documentation
- Update CLAUDE.md with new structure
- Update architecture-design.md
- Add migration notes to README

**PR 5.2**: CI updates
- Add test jobs for core crate
- Test feature flag combinations
- Verify WASM build still works

---

## CI Testing Strategy

```yaml
jobs:
  test-core:
    runs-on: ubuntu-latest
    steps:
      - name: Test core (default features)
        run: cargo test -p coppermind-core --verbose

      - name: Test core (tree-sitter)
        run: cargo test -p coppermind-core --features tree-sitter --verbose

      - name: Test core (crawler)
        run: cargo test -p coppermind-core --features crawler --verbose

      - name: Test core (all features)
        run: cargo test -p coppermind-core --all-features --verbose

  test-app-desktop:
    runs-on: macos-latest
    steps:
      - name: Test app (desktop)
        run: cargo test -p coppermind --features desktop --verbose

  build-web:
    runs-on: ubuntu-latest
    steps:
      - name: Build WASM
        run: dx bundle -p coppermind --release
```

---

## Performance Analysis

### Memory

| Operation | Before | After | Impact |
|-----------|--------|-------|--------|
| Model loading | Load bytes → parse | Load bytes → pass to core → parse | **No change** (same data flow) |
| Singleton caching | `OnceCell` in embedding mod | `OnceCell` in app | **No change** |
| Storage | Direct impl | Trait + impl | **Negligible** (vtable indirection) |

### Latency

| Operation | Before | After | Impact |
|-----------|--------|-------|--------|
| First embedding | Fetch + parse + inference | Fetch + parse + inference | **No change** |
| Subsequent embeddings | Cached model | Cached model | **No change** |
| Search | Direct call | Direct call (re-exported) | **No change** |

### Binary Size

| Target | Before | After | Impact |
|--------|--------|-------|--------|
| WASM | Single crate | Core + app | **~Same** (same code, LTO) |
| Desktop | Single crate | Core + app | **~Same** |

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Import path breakage | High | Low | Compiler catches all errors; search-replace |
| Singleton race condition | Low | High | Preserve exact `OnceCell` pattern |
| Feature flag forgotten | Medium | Medium | CI tests all combinations |
| WASM worker breaks | Low | High | Test web platform after each PR |
| Performance regression | Low | Low | Benchmark before/after |

---

## Future: CLI Crate

Once core extraction is complete:

```
crates/
└── coppermind-cli/
    ├── Cargo.toml
    └── src/
        └── main.rs
```

```toml
# coppermind-cli/Cargo.toml
[dependencies]
coppermind-core = { path = "../coppermind-core", features = ["tree-sitter", "crawler"] }
clap = { version = "4", features = ["derive"] }
tokio = { version = "1", features = ["full"] }
```

**CLI capabilities**:
- Index local files/directories
- Search indexed content
- Export search results
- MCP server mode for AI assistant integration

---

## Alternatives Considered

### Alternative A: Keep Single Crate

**Approach**: Leave structure as-is, use feature flags for platform differences.

**Rejected because**:
- Testing core algorithms still requires full Dioxus compilation
- Cannot create CLI without pulling in UI dependencies
- Boundaries remain unclear

### Alternative B: Three-Crate Structure (core + ui + app)

**Approach**: Separate shared UI components into `coppermind-ui` crate.

**Rejected because**:
- UI components rarely need separate testing
- Adds complexity with minimal benefit
- Two crates sufficient for our goals

### Alternative C: Default Dioxus Workspace Template

**Approach**: Use `dx new --template workspace` structure with separate web/desktop/mobile entry points.

**Rejected because**:
- Duplicates views across platforms (3x maintenance)
- Designed for fullstack/SSR apps
- 95%+ of our code is shared across platforms

---

## Decision Outcome

**Chosen approach**: Two-crate workspace (core + app) with:
- Pure, stateless core library
- Application-managed lifecycle
- Feature flags for optional functionality
- Storage trait in core, OPFS implementation in app
- Incremental migration via small PRs

This provides the best balance of testability, extensibility, and migration safety while preserving all existing functionality and performance characteristics.

---

## References

- [ADR-004: Incremental Vector Search](./004-incremental-vector-search.md) - Previous HNSW migration
- [ADR-006: Dioxus Workspace Structure](file:///Users/matthew/workspace/planner/docs/adrs/ADR-006-workspace-structure.md) - Inspiration from planner project
- [Cargo Workspaces](https://doc.rust-lang.org/book/ch14-03-cargo-workspaces.html) - Official documentation
- [Dioxus CLI Workspace Support](https://dioxuslabs.com/learn/0.7/CLI) - Using `-p` flag
