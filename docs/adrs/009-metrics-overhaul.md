# ADR 009: Comprehensive Metrics System Overhaul

**Status**: Implemented
**Date**: 2025-11-29
**Context**: Web metrics bug, missing search/scheduler visibility, ad-hoc metrics architecture

---

## Context and Problem Statement

### Current Issues

1. **Web Platform Metrics Bug**: The WASM version of `embed_text_chunks_auto()` in `crates/coppermind/src/embedding/mod.rs` (lines 607-689) does NOT record any metrics. Unlike the desktop version which calls `global_metrics().record_chunking()`, `record_tokenization()`, and `record_embedding()` at lines 512, 546, and 570 respectively, the WASM path has no equivalent calls.

2. **Ad-hoc Architecture**: Metrics collection is scattered across multiple files:
   - `embedding/mod.rs` - Records chunking, tokenization, embedding (desktop only)
   - `file_processing.rs` - Records HNSW/BM25 indexing with hardcoded 70/30 split
   - `workers/` - Web worker logs timing but doesn't integrate with global metrics
   - `settings_dialog.rs` - Clears metrics

3. **Missing High-Value Metrics**:
   - No search query metrics (latency breakdown, result quality)
   - No GPU scheduler visibility (queue depth, wait times)
   - No score distribution data
   - `record_queue_wait()` exists but is never called

### Current Architecture

```rust
// crates/coppermind/src/metrics.rs (~330 lines)
pub struct PerformanceMetrics {
    inner: Arc<RwLock<MetricsInner>>,
    window: Duration,  // 60 seconds default
}

struct MetricsInner {
    chunking: MetricData,
    tokenization: MetricData,
    embedding: MetricData,
    hnsw_indexing: MetricData,
    bm25_indexing: MetricData,
    queue_wait: MetricData,
}

struct MetricData {
    samples: VecDeque<TimingSample>,  // Rolling window
    total_count: u64,
    total_duration_ms: f64,
}

static GLOBAL_METRICS: Lazy<PerformanceMetrics> = Lazy::new(PerformanceMetrics::new);
```

This uses `instant::Instant` for cross-platform timing (works on WASM).

---

## Decision Drivers

1. **Fix web metrics** - Primary bug, blocking real-time feedback on web
2. **Search visibility** - Users need to see where query time is spent
3. **Scheduler visibility** - GPU queue depth helps diagnose indexing bottlenecks
4. **Score distribution** - Search quality insight for users
5. **No new dependencies** - WASM compatibility, minimal footprint
6. **No performance impact** - Metrics recording must not bottleneck processing

---

## Considered Options

### Option 1: Use metrics-rs Crate

**Approach**: Adopt the `metrics` crate facade with custom recorder.

**Evaluation**:
- Core `metrics` crate: Lightweight (ahash + portable-atomic)
- `metrics-util` (data structures): Heavy deps (quanta uses libc, crossbeam)
- Designed for server telemetry (Prometheus export) - overkill for client UI

**Verdict**: Rejected - adds complexity for features we don't need.

### Option 2: Keep and Extend Custom Implementation (Recommended)

**Approach**: Refactor existing `metrics.rs`, move to core crate, add search/scheduler metrics.

**Pros**:
- Already works, well-tested (~330 lines)
- Uses `instant::Instant` which handles WASM
- Purpose-built for our UI display needs
- No new dependencies
- Rolling window + max samples already prevent memory growth

**Cons**:
- Must manually ensure consistency between platforms
- Not a standard ecosystem pattern

**Verdict**: Accepted - simplest path, meets all requirements.

---

## Decision

Extend the custom metrics implementation with:
1. Move metrics to `coppermind-core` for shared access
2. Add search metrics (query timing breakdown, score distribution)
3. Add scheduler metrics (integrate with existing `SchedulerStats`)
4. Fix web platform recording

### Architecture Changes

#### 1. Move to coppermind-core

Move `crates/coppermind/src/metrics.rs` → `crates/coppermind-core/src/metrics.rs`

This allows the search engine and GPU scheduler (both in core) to record metrics directly.

#### 2. Extended MetricsInner

```rust
struct MetricsInner {
    // Existing pipeline metrics
    chunking: MetricData,
    tokenization: MetricData,
    embedding: MetricData,
    hnsw_indexing: MetricData,
    bm25_indexing: MetricData,

    // NEW: Search metrics
    search_query_embed: MetricData,
    search_vector: MetricData,
    search_keyword: MetricData,
    search_fusion: MetricData,
    search_total: MetricData,

    // NEW: Last search result info (gauges, not rolling)
    last_search: Option<LastSearchInfo>,

    // NEW: Scheduler metrics
    scheduler_queue_wait: MetricData,
    scheduler_inference: MetricData,

    // NEW: Scheduler gauges (updated from scheduler stats)
    scheduler_queue_depth: usize,
    scheduler_requests_completed: u64,
}

struct LastSearchInfo {
    result_count: usize,
    vector_count: usize,
    keyword_count: usize,
    top_score: Option<f32>,
    median_score: Option<f32>,
}
```

#### 3. Extended MetricsSnapshot

```rust
pub struct MetricsSnapshot {
    // Existing pipeline fields...
    pub chunking_avg_ms: Option<f64>,
    pub chunking_count: usize,
    // ... etc

    // NEW: Search section
    pub search: SearchSnapshot,

    // NEW: Scheduler section
    pub scheduler: SchedulerSnapshot,
}

#[derive(Clone, Debug, Default)]
pub struct SearchSnapshot {
    pub query_embed_avg_ms: Option<f64>,
    pub vector_search_avg_ms: Option<f64>,
    pub keyword_search_avg_ms: Option<f64>,
    pub fusion_avg_ms: Option<f64>,
    pub total_latency_avg_ms: Option<f64>,
    pub last_result_count: Option<usize>,
    pub last_vector_count: Option<usize>,
    pub last_keyword_count: Option<usize>,
    pub last_top_score: Option<f32>,
    pub last_median_score: Option<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct SchedulerSnapshot {
    pub queue_depth: usize,
    pub queue_wait_avg_ms: Option<f64>,
    pub inference_avg_ms: Option<f64>,
    pub requests_completed: u64,
}
```

#### 4. New Recording Methods

```rust
impl PerformanceMetrics {
    // Existing methods...

    // NEW: Search metrics
    pub fn record_search(
        &self,
        query_embed_ms: f64,
        vector_ms: f64,
        keyword_ms: f64,
        fusion_ms: f64,
        result_count: usize,
        vector_count: usize,
        keyword_count: usize,
        top_score: Option<f32>,
        median_score: Option<f32>,
    ) {
        if let Ok(mut inner) = self.inner.write() {
            inner.search_query_embed.record(query_embed_ms);
            inner.search_vector.record(vector_ms);
            inner.search_keyword.record(keyword_ms);
            inner.search_fusion.record(fusion_ms);
            inner.search_total.record(query_embed_ms + vector_ms + keyword_ms + fusion_ms);
            inner.last_search = Some(LastSearchInfo {
                result_count,
                vector_count,
                keyword_count,
                top_score,
                median_score,
            });
        }
    }

    // NEW: Scheduler metrics
    pub fn record_scheduler_request(&self, queue_wait_ms: f64, inference_ms: f64) {
        if let Ok(mut inner) = self.inner.write() {
            inner.scheduler_queue_wait.record(queue_wait_ms);
            inner.scheduler_inference.record(inference_ms);
        }
    }

    pub fn update_scheduler_stats(&self, queue_depth: usize, requests_completed: u64) {
        if let Ok(mut inner) = self.inner.write() {
            inner.scheduler_queue_depth = queue_depth;
            inner.scheduler_requests_completed = requests_completed;
        }
    }
}
```

---

## Implementation Details

### Fix Web Platform Metrics

**File**: `crates/coppermind/src/embedding/mod.rs`

In `embed_text_chunks_auto` WASM version (lines 607-689), add timing:

```rust
#[cfg(target_arch = "wasm32")]
pub async fn embed_text_chunks_auto(
    text: &str,
    chunk_tokens: usize,
    filename: Option<&str>,
    worker: Option<EmbeddingWorkerClient>,
) -> Result<Vec<ChunkEmbeddingResult>, EmbeddingError> {
    // Chunking timing
    let chunk_start = instant::Instant::now();
    let text_chunks = chunker.chunk(text)?;
    let chunk_duration_ms = chunk_start.elapsed().as_secs_f64() * 1000.0;
    global_metrics().record_chunking(chunk_duration_ms);

    // Per-chunk embedding timing
    for chunk in text_chunks {
        let embed_start = instant::Instant::now();
        let computation = if let Some(ref w) = worker {
            w.embed(chunk.text.clone()).await?
        } else {
            compute_embedding(&chunk.text).await?
        };
        let embed_duration_ms = embed_start.elapsed().as_secs_f64() * 1000.0;
        global_metrics().record_embedding(embed_duration_ms);
        // ...
    }
}
```

### Search Metrics Integration

**File**: `crates/coppermind-core/src/search/engine.rs`

Add `SearchTimings` struct returned from search:

```rust
#[derive(Debug, Clone)]
pub struct SearchTimings {
    pub vector_ms: f64,
    pub keyword_ms: f64,
    pub fusion_ms: f64,
    pub total_ms: f64,
    pub vector_count: usize,
    pub keyword_count: usize,
}

impl HybridSearchEngine {
    pub fn search_with_timings(
        &self,
        query_embedding: &[f32],
        query_text: &str,
        k: usize,
    ) -> Result<(Vec<SearchResult>, SearchTimings), SearchError> {
        let total_start = instant::Instant::now();

        let vector_start = instant::Instant::now();
        let vector_results = self.vector_engine.search(query_embedding, k * 2);
        let vector_ms = vector_start.elapsed().as_secs_f64() * 1000.0;

        let keyword_start = instant::Instant::now();
        let keyword_results = self.keyword_engine.search(query_text, k * 2);
        let keyword_ms = keyword_start.elapsed().as_secs_f64() * 1000.0;

        let fusion_start = instant::Instant::now();
        let fused = reciprocal_rank_fusion(&vector_results, &keyword_results, RRF_K);
        let fusion_ms = fusion_start.elapsed().as_secs_f64() * 1000.0;

        let timings = SearchTimings {
            vector_ms,
            keyword_ms,
            fusion_ms,
            total_ms: total_start.elapsed().as_secs_f64() * 1000.0,
            vector_count: vector_results.len(),
            keyword_count: keyword_results.len(),
        };

        // Build results...
        Ok((results, timings))
    }
}
```

**File**: `crates/coppermind/src/components/search/search_view.rs`

Record metrics after search:

```rust
let embed_start = instant::Instant::now();
let query_embedding = compute_embedding(&query).await?;
let embed_ms = embed_start.elapsed().as_secs_f64() * 1000.0;

let (results, timings) = engine.search_with_timings(&query_embedding, &query, k)?;

// Record search metrics
let top_score = results.first().map(|r| r.score);
let median_score = results.get(results.len() / 2).map(|r| r.score);

global_metrics().record_search(
    embed_ms,
    timings.vector_ms,
    timings.keyword_ms,
    timings.fusion_ms,
    results.len(),
    timings.vector_count,
    timings.keyword_count,
    top_score,
    median_score,
);
```

### GPU Scheduler Metrics

**File**: `crates/coppermind-core/src/gpu/types.rs`

Extend `EmbedRequest` with submission timestamp:

```rust
pub struct EmbedRequest {
    pub model_id: ModelId,
    pub tokens: Vec<u32>,
    pub priority: Priority,
    pub submitted_at: instant::Instant,  // NEW
}

impl EmbedRequest {
    pub fn new(tokens: Vec<u32>) -> Self {
        Self {
            model_id: ModelId::JinaBert,
            tokens,
            priority: Priority::Interactive,
            submitted_at: instant::Instant::now(),
        }
    }
}
```

**File**: `crates/coppermind-core/src/gpu/serial_scheduler.rs`

In worker loop, record timings:

```rust
fn process_embed_request(&mut self, request: EmbedRequest, response: oneshot::Sender<...>) {
    let queue_wait_ms = request.submitted_at.elapsed().as_secs_f64() * 1000.0;

    let inference_start = instant::Instant::now();
    let result = self.do_embed(&request);
    let inference_ms = inference_start.elapsed().as_secs_f64() * 1000.0;

    // Record to global metrics
    global_metrics().record_scheduler_request(queue_wait_ms, inference_ms);

    let _ = response.send(result);
}
```

Periodically update scheduler gauges:

```rust
// After each request processed:
global_metrics().update_scheduler_stats(
    self.stats.queue_depth.load(Ordering::Relaxed),
    self.stats.requests_completed.load(Ordering::Relaxed),
);
```

### Separate HNSW/BM25 Timing

**File**: `crates/coppermind/src/components/file_processing.rs`

Replace estimated 70/30 split with actual timing:

```rust
// BEFORE (lines 364-365):
global_metrics().record_hnsw_indexing(insert_duration_ms * 0.7);
global_metrics().record_bm25_indexing(insert_duration_ms * 0.3);

// AFTER:
let hnsw_start = instant::Instant::now();
engine.write().insert_embeddings(&embeddings, chunk_ids)?;
let hnsw_ms = hnsw_start.elapsed().as_secs_f64() * 1000.0;
global_metrics().record_hnsw_indexing(hnsw_ms);

let bm25_start = instant::Instant::now();
engine.write().insert_documents(&documents)?;
let bm25_ms = bm25_start.elapsed().as_secs_f64() * 1000.0;
global_metrics().record_bm25_indexing(bm25_ms);
```

### UI Updates

**File**: `crates/coppermind/src/components/app_shell/metrics_pane.rs`

Add new sections for search and scheduler metrics:

```
Search Performance (visible after first search)
+-- Query Latency: 145ms
|   +-- Embed: 85ms
|   +-- Vector: 32ms
|   +-- Keyword: 18ms
|   +-- Fusion: 10ms
+-- Results: 12 (8 vector, 4 keyword)
+-- Top Score: 0.92, Median: 0.61

GPU Scheduler (visible during/after indexing)
+-- Queue Depth: 3 pending
+-- Avg Wait: 45ms
+-- Avg Inference: 23ms
+-- Completed: 1,247
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `crates/coppermind-core/src/metrics.rs` | NEW: Move from app crate, extend with search/scheduler |
| `crates/coppermind-core/src/lib.rs` | Add `pub mod metrics;` |
| `crates/coppermind-core/src/search/engine.rs` | Add `search_with_timings()`, `SearchTimings` |
| `crates/coppermind-core/src/search/types.rs` | Add `SearchTimings` struct |
| `crates/coppermind-core/src/gpu/types.rs` | Add `submitted_at` to `EmbedRequest` |
| `crates/coppermind-core/src/gpu/serial_scheduler.rs` | Record queue wait and inference times |
| `crates/coppermind/src/embedding/mod.rs` | Fix WASM metrics recording |
| `crates/coppermind/src/components/file_processing.rs` | Separate HNSW/BM25 timing |
| `crates/coppermind/src/components/app_shell/metrics_pane.rs` | Add search/scheduler UI sections |
| `crates/coppermind/src/components/search/search_view.rs` | Record search metrics |
| `crates/coppermind/src/metrics.rs` | DELETE (moved to core) |

---

## Consequences

### Positive

1. **Web metrics fixed** - Real-time feedback on WASM platform
2. **Search visibility** - Users see query latency breakdown
3. **Scheduler visibility** - Queue depth helps diagnose indexing issues
4. **Score distribution** - Search quality insight
5. **No new dependencies** - Keeps WASM bundle small
6. **Actual HNSW/BM25 timing** - Removes guesswork of 70/30 split
7. **Centralized in core** - Search engine and scheduler can record directly

### Negative

1. **Breaking API change** - Callers must update imports
   - **Mitigation**: Search-and-replace migration
2. **Global state in core** - `global_metrics()` singleton in library
   - **Mitigation**: Acceptable for this use case; metrics are inherently global

### Neutral

1. **RwLock contention** - Metrics recording acquires write lock
   - Analysis: Lock held briefly (~us), recording happens in ms-scale operations
   - Not a bottleneck for our use case

---

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_search_metrics_recording() {
    let metrics = PerformanceMetrics::new();
    metrics.record_search(10.0, 5.0, 3.0, 2.0, 10, 8, 4, Some(0.9), Some(0.5));

    let snapshot = metrics.snapshot();
    assert!(snapshot.search.query_embed_avg_ms.is_some());
    assert_eq!(snapshot.search.last_result_count, Some(10));
}

#[test]
fn test_scheduler_metrics_recording() {
    let metrics = PerformanceMetrics::new();
    metrics.record_scheduler_request(5.0, 20.0);
    metrics.update_scheduler_stats(3, 100);

    let snapshot = metrics.snapshot();
    assert!(snapshot.scheduler.queue_wait_avg_ms.is_some());
    assert_eq!(snapshot.scheduler.queue_depth, 3);
}
```

### Manual UAT

1. **Web metrics**:
   - `dx serve -p coppermind`
   - Upload files, verify chunking/embedding metrics display in real-time

2. **Desktop metrics**:
   - `dx serve -p coppermind --platform desktop`
   - Same verification

3. **Search metrics**:
   - Perform search, verify latency breakdown appears
   - Verify score distribution shows (top/median)

4. **Scheduler metrics**:
   - During indexing, verify queue depth updates
   - Verify wait/inference times display

---

## References

- [ADR 006: GPU Scheduler](./006-gpu-scheduler.md) - Scheduler architecture
- [metrics-rs crate](https://docs.rs/metrics) - Evaluated but not adopted
- [instant crate](https://docs.rs/instant) - Cross-platform timing (already used)
