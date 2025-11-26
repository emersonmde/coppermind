# Performance Profiling Guide

This document explains how to profile Coppermind to diagnose performance issues like UI lag during indexing.

## Background: Why Profiling?

During web crawling and embedding, the UI can become laggy or freeze. The embedding model runs on a dedicated GPU thread (see [ADR 006](adrs/006-gpu-scheduler.md)), so it shouldn't block the UI directly. However, other operations on the main tokio runtime can cause lag:

- **Index operations** - Adding documents to HNSW vector index and BM25 keyword index
- **Signal updates** - Dioxus re-renders triggered by progress updates
- **Memory pressure** - Large queues, accumulated embeddings, growing index
- **Channel backpressure** - Bounded channels blocking when full

To diagnose which operation is causing lag, we use the `tracing` crate with Chrome trace output.

## How It Works

### Tracing Infrastructure

The profiling system uses:

1. **`tracing`** - Rust's instrumentation framework for structured logging
2. **`tracing-subscriber`** - Collects and processes trace data
3. **`tracing-chrome`** - Outputs traces in Chrome's trace event format

When profiling is enabled, key functions are annotated with `#[instrument]` which automatically creates spans:

```rust
#[instrument(skip_all, fields(chunks = embedding_results.len()))]
pub async fn index_chunks<S: StorageBackend>(
    engine: Arc<Mutex<HybridSearchEngine<S>>>,
    embedding_results: &[ChunkEmbeddingResult],
    file_label: &str,
) -> Result<usize, FileProcessingError> {
    // ... function body
}
```

This creates a span named `index_chunks` with metadata about the number of chunks being processed. When the function completes, the span duration is recorded.

### Instrumented Functions

The following functions are instrumented for profiling:

| Function | File | Purpose |
|----------|------|---------|
| `HybridSearchEngine::add_document` | `search/engine.rs` | Adds doc to both indexes |
| `HybridSearchEngine::add_document_deferred` | `search/engine.rs` | Batch add variant |
| `VectorSearchEngine::add_document` | `search/vector.rs` | HNSW insertion |
| `KeywordSearchEngine::add_document` | `search/keyword.rs` | BM25 indexing |
| `index_chunks` | `components/file_processing.rs` | High-level indexing loop |

## How to Profile

### Step 1: Run with Profiling Enabled

```bash
dx serve -p coppermind --platform desktop --features profile
```

This:
- Enables the `profile` feature which compiles in tracing-chrome
- Initializes the Chrome trace layer with **filtering to only our crates**
- Creates `./trace.json` which accumulates trace events
- Keeps console logging active so you still see log output

**Important**: The tracing is filtered to only capture spans from `coppermind` and `coppermind_core`.
Without this filtering, dependencies (dioxus, tokio, candle, etc.) would generate gigabyte-sized traces that crash viewers.

### Step 2: Perform the Slow Operation

Trigger the operation you want to profile. For UI lag during crawling:

1. Open the app
2. Start a web crawl (e.g., crawl coppermind.net with 50+ pages)
3. Wait until lag becomes noticeable
4. Stop the crawl (Ctrl+C or stop button)

### Step 3: Stop the Application

Press `Ctrl+C` to gracefully shut down. This flushes the trace buffer to `trace.json`.

**Important**: The trace file is written when the application exits. If you force-kill the app, you may lose trace data.

### Step 4: View in Chrome

1. Open Chrome (or Chromium-based browser)
2. Navigate to `chrome://tracing`
3. Click "Load" and select `./trace.json`
4. Use the timeline to identify slow operations

### Interpreting the Timeline

The Chrome tracing UI shows:

- **Horizontal bars** - Each bar is a span (function call)
- **Width** - Duration of the operation
- **Nesting** - Parent/child relationships between spans
- **Color coding** - Different threads/categories

Look for:
- **Wide bars** - Long-running operations that may block the UI
- **Frequent short bars** - High-frequency operations that add up
- **Gaps** - Time spent outside instrumented code (may indicate other bottlenecks)

### Example Analysis

```
Timeline:
|------ index_chunks (150ms) ------|
  |-- add_document (5ms) --|
    |-- add_document (HNSW) (3ms) --|
    |-- add_document (BM25) (2ms) --|
  |-- add_document (6ms) --|
  ...
```

In this example:
- `index_chunks` takes 150ms total
- Each document addition takes ~5-6ms
- HNSW insertion is slightly slower than BM25
- Processing 25 documents sequentially = 150ms of blocking

This suggests the fix might be:
- Batch HNSW insertions
- Move indexing to a background thread
- Reduce update frequency

## Feature Flag Details

The profiling infrastructure is gated behind the `profile` feature to avoid:
- Compile-time overhead in normal builds
- Runtime overhead from tracing
- Trace file generation

### Cargo.toml Configuration

```toml
[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
tracing = { version = "0.1", optional = true }
tracing-subscriber = { version = "0.3", features = ["fmt", "env-filter"], optional = true }
tracing-chrome = { version = "0.7", optional = true }

[features]
profile = ["tracing", "tracing-subscriber", "tracing-chrome"]
```

### Filtering Configuration

The profiling initialization in `main.rs` filters spans to only our crates:

```rust
// Filter to only capture spans from our crates, not dependencies
let trace_filter = EnvFilter::new("coppermind=trace,coppermind_core=trace");
```

This is critical - without filtering, every tokio task, dioxus render, and candle operation
would be traced, producing 500MB+ trace files in seconds.

### Build Configurations

| Build Command | Includes Profiling? |
|--------------|---------------------|
| `dx serve -p coppermind` (web) | No - wrong target arch |
| `dx serve -p coppermind --platform desktop` | No - feature not enabled |
| `dx serve -p coppermind --platform desktop --features profile` | Yes |
| `dx bundle -p coppermind --release` | No - feature not enabled |

## Adding More Instrumentation

To instrument additional functions:

1. Add the import (in files in coppermind crate):
   ```rust
   #[cfg(feature = "profile")]
   use tracing::instrument;
   ```

2. Add the attribute:
   ```rust
   #[cfg_attr(feature = "profile", instrument(skip_all, fields(my_field = value)))]
   pub fn my_function(...) {
       // ...
   }
   ```

For functions in `coppermind-core` (which has `tracing` as a regular dependency):
```rust
use tracing::instrument;

#[instrument(skip_all, fields(my_field = value))]
pub fn my_function(...) {
    // ...
}
```

### Best Practices

- Use `skip_all` to avoid logging large arguments
- Add `fields()` to capture useful metadata (sizes, counts)
- Focus on I/O-bound or computationally heavy functions
- Don't instrument tight loops - instrument the loop container

## Troubleshooting

### Empty or Missing trace.json

- Ensure you exit the app gracefully (Ctrl+C)
- Check the working directory - file is created at `./trace.json`
- Verify the profile feature is enabled in the build

### Chrome Tracing Won't Load File

- File may be too large (>50MB) - see "Huge Trace Files" below
- File may be corrupted - try a fresh run
- Use `chrome://tracing` not the DevTools Performance tab

### Huge Trace Files (100MB+)

If your trace file is hundreds of megabytes:

1. **Check filtering is working** - You should see this on startup:
   ```
   INFO coppermind: Profiling enabled - trace will be written to ./trace.json
   INFO coppermind: Tracing only: coppermind, coppermind_core (dependencies filtered out)
   ```

2. **Look for hot loop instrumentation** - Don't use `#[instrument]` inside tight loops.
   Instrument the loop container instead:
   ```rust
   // BAD - creates millions of spans
   for item in items {
       #[instrument]
       fn process(item: Item) { ... }
   }

   // GOOD - single span for the batch
   #[instrument(fields(count = items.len()))]
   fn process_batch(items: Vec<Item>) { ... }
   ```

3. **Try speedscope.app** - Handles larger files better than chrome://tracing

4. **Reduce trace duration** - Profile for 5-10 seconds, not minutes

### Spans Not Appearing

- Verify the function has `#[instrument]` or `#[cfg_attr(feature = "profile", instrument(...))]`
- Check that the code path is actually executed
- Ensure tracing subscriber is initialized before the code runs

## Future Improvements

Potential enhancements to the profiling system:

1. **Flame graphs** - Use `tracing-flame` for traditional flame graph output
2. **Remote tracing** - Export to Jaeger/Zipkin for distributed tracing
3. **Metrics** - Add `tracing-subscriber` metrics layer for aggregated stats
4. **Selective instrumentation** - Runtime control over which spans are recorded
