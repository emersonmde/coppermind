//! Performance metrics collection with rolling averages.
//!
//! This module provides lightweight metrics collection for tracking
//! performance over time. Metrics are stored in-memory and support
//! rolling averages over configurable time windows.
//!
//! ## Architecture
//!
//! The metrics system uses a global singleton (`global_metrics()`) that can be
//! accessed from anywhere in the codebase. This is intentional for metrics since
//! they are inherently global state - operations across the codebase need to
//! record to the same collector.
//!
//! ## Metrics Categories
//!
//! - **Pipeline metrics**: Chunking, tokenization, embedding, indexing
//! - **Search metrics**: Query timing breakdown, result statistics
//! - **Scheduler metrics**: GPU queue depth, wait times, inference times

use instant::Instant;
use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// Default window size for rolling averages (60 seconds).
/// Used for high-frequency operations like chunking, tokenization, embedding.
const DEFAULT_WINDOW_SECS: u64 = 60;

/// Window size for search metrics (5 minutes).
/// Searches are less frequent than indexing operations, so we use a longer
/// window to capture meaningful averages.
const SEARCH_WINDOW_SECS: u64 = 300;

/// Maximum samples to keep per metric (prevents unbounded growth).
const MAX_SAMPLES: usize = 1000;

/// A single timing sample with timestamp.
#[derive(Clone, Debug)]
struct TimingSample {
    /// When this sample was recorded.
    timestamp: Instant,
    /// Duration of the operation in milliseconds.
    duration_ms: f64,
}

/// Rolling statistics for a single metric.
#[derive(Debug, Default)]
struct MetricData {
    /// Recent samples within the rolling window.
    samples: VecDeque<TimingSample>,
    /// Total count since startup.
    total_count: u64,
    /// Sum of all durations since startup (for lifetime average).
    total_duration_ms: f64,
}

impl MetricData {
    fn new() -> Self {
        Self {
            samples: VecDeque::with_capacity(MAX_SAMPLES),
            total_count: 0,
            total_duration_ms: 0.0,
        }
    }

    /// Record a new sample.
    fn record(&mut self, duration_ms: f64) {
        let now = Instant::now();

        // Add to totals
        self.total_count += 1;
        self.total_duration_ms += duration_ms;

        // Add sample
        self.samples.push_back(TimingSample {
            timestamp: now,
            duration_ms,
        });

        // Trim if over max
        while self.samples.len() > MAX_SAMPLES {
            self.samples.pop_front();
        }
    }

    /// Prune samples older than the window.
    fn prune(&mut self, window: Duration) {
        let now = Instant::now();
        // Use checked_sub to avoid panic on WASM where Instant starts at page load
        let cutoff = match now.checked_sub(window) {
            Some(t) => t,
            None => return, // Window extends before page load, keep all samples
        };

        while let Some(front) = self.samples.front() {
            if front.timestamp < cutoff {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }

    /// Calculate rolling average over the window.
    fn rolling_avg(&self, window: Duration) -> Option<f64> {
        let now = Instant::now();
        // Use checked_sub to avoid panic on WASM where Instant starts at page load
        let cutoff = now.checked_sub(window);

        let mut sum = 0.0;
        let mut count = 0;

        for sample in &self.samples {
            // If cutoff is None (window before page load), include all samples
            let in_window = cutoff.is_none_or(|c| sample.timestamp >= c);
            if in_window {
                sum += sample.duration_ms;
                count += 1;
            }
        }

        if count > 0 {
            Some(sum / count as f64)
        } else {
            None
        }
    }

    /// Get count of samples in the window.
    fn rolling_count(&self, window: Duration) -> usize {
        let now = Instant::now();
        // Use checked_sub to avoid panic on WASM where Instant starts at page load
        let cutoff = now.checked_sub(window);

        self.samples
            .iter()
            .filter(|s| cutoff.is_none_or(|c| s.timestamp >= c))
            .count()
    }

    /// Calculate throughput (samples per second) over the window.
    fn throughput(&self, window: Duration) -> f64 {
        let count = self.rolling_count(window);
        if count == 0 {
            return 0.0;
        }

        let window_secs = window.as_secs_f64();
        count as f64 / window_secs
    }

    /// Get lifetime average.
    #[allow(dead_code)] // Reserved for future use
    fn lifetime_avg(&self) -> Option<f64> {
        if self.total_count > 0 {
            Some(self.total_duration_ms / self.total_count as f64)
        } else {
            None
        }
    }
}

/// Information about the last search result (point-in-time, not rolling).
#[derive(Clone, Debug, Default)]
struct LastSearchInfo {
    result_count: usize,
    vector_count: usize,
    keyword_count: usize,
    top_score: Option<f32>,
    median_score: Option<f32>,
}

/// Collected metrics snapshot for UI display.
#[derive(Clone, Debug, Default)]
pub struct MetricsSnapshot {
    /// Chunking metrics.
    pub chunking_avg_ms: Option<f64>,
    pub chunking_count: usize,
    pub chunking_throughput: f64,

    /// Tokenization metrics.
    pub tokenization_avg_ms: Option<f64>,
    pub tokenization_count: usize,

    /// Embedding (GPU inference) metrics.
    pub embedding_avg_ms: Option<f64>,
    pub embedding_count: usize,
    pub embedding_throughput: f64,

    /// HNSW indexing metrics.
    pub hnsw_avg_ms: Option<f64>,
    pub hnsw_count: usize,

    /// BM25 indexing metrics.
    pub bm25_avg_ms: Option<f64>,
    pub bm25_count: usize,

    /// Lifetime totals.
    pub total_chunks_processed: u64,
    pub total_embeddings_generated: u64,

    /// Search metrics.
    pub search: SearchSnapshot,

    /// Scheduler metrics.
    pub scheduler: SchedulerSnapshot,
}

/// Search-specific metrics snapshot.
#[derive(Clone, Debug, Default)]
pub struct SearchSnapshot {
    /// Average time to embed the query (ms).
    pub query_embed_avg_ms: Option<f64>,
    /// Average HNSW vector search time (ms).
    pub vector_search_avg_ms: Option<f64>,
    /// Average BM25 keyword search time (ms).
    pub keyword_search_avg_ms: Option<f64>,
    /// Average RRF fusion time (ms).
    pub fusion_avg_ms: Option<f64>,
    /// Average total search latency (ms).
    pub total_latency_avg_ms: Option<f64>,
    /// Number of search queries in the rolling window.
    pub query_count: usize,

    /// Last search result info (point-in-time).
    pub last_result_count: Option<usize>,
    pub last_vector_count: Option<usize>,
    pub last_keyword_count: Option<usize>,
    pub last_top_score: Option<f32>,
    pub last_median_score: Option<f32>,
}

/// GPU scheduler metrics snapshot.
#[derive(Clone, Debug, Default)]
pub struct SchedulerSnapshot {
    /// Current queue depth (pending requests).
    pub queue_depth: usize,
    /// Average queue wait time (ms).
    pub queue_wait_avg_ms: Option<f64>,
    /// Average inference time (ms).
    pub inference_avg_ms: Option<f64>,
    /// Total requests completed.
    pub requests_completed: u64,
}

/// Internal metrics storage.
struct MetricsInner {
    // Pipeline metrics
    chunking: MetricData,
    tokenization: MetricData,
    embedding: MetricData,
    hnsw_indexing: MetricData,
    bm25_indexing: MetricData,

    // Search metrics
    search_query_embed: MetricData,
    search_vector: MetricData,
    search_keyword: MetricData,
    search_fusion: MetricData,
    search_total: MetricData,
    last_search: Option<LastSearchInfo>,

    // Scheduler metrics
    scheduler_queue_wait: MetricData,
    scheduler_inference: MetricData,
    scheduler_queue_depth: usize,
    scheduler_requests_completed: u64,
}

impl Default for MetricsInner {
    fn default() -> Self {
        Self {
            chunking: MetricData::new(),
            tokenization: MetricData::new(),
            embedding: MetricData::new(),
            hnsw_indexing: MetricData::new(),
            bm25_indexing: MetricData::new(),

            search_query_embed: MetricData::new(),
            search_vector: MetricData::new(),
            search_keyword: MetricData::new(),
            search_fusion: MetricData::new(),
            search_total: MetricData::new(),
            last_search: None,

            scheduler_queue_wait: MetricData::new(),
            scheduler_inference: MetricData::new(),
            scheduler_queue_depth: 0,
            scheduler_requests_completed: 0,
        }
    }
}

/// Performance metrics collector.
///
/// Thread-safe collector for timing metrics with rolling averages.
/// Use `record_*` methods to log timings, and `snapshot()` to get
/// current statistics for UI display.
///
/// Different metric types use different window sizes:
/// - Pipeline metrics (chunking, tokenization, embedding): 60 seconds
/// - Search metrics: 5 minutes (searches are less frequent)
#[derive(Clone)]
pub struct PerformanceMetrics {
    inner: Arc<RwLock<MetricsInner>>,
    /// Window for high-frequency pipeline operations.
    window: Duration,
    /// Window for search metrics (longer since searches are less frequent).
    search_window: Duration,
}

impl PerformanceMetrics {
    /// Create a new metrics collector with default windows.
    /// - Pipeline metrics: 60 seconds
    /// - Search metrics: 5 minutes
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(MetricsInner::default())),
            window: Duration::from_secs(DEFAULT_WINDOW_SECS),
            search_window: Duration::from_secs(SEARCH_WINDOW_SECS),
        }
    }

    /// Create a new metrics collector with custom window size.
    /// Both pipeline and search use the same window (for testing).
    pub fn with_window(window_secs: u64) -> Self {
        Self {
            inner: Arc::new(RwLock::new(MetricsInner::default())),
            window: Duration::from_secs(window_secs),
            search_window: Duration::from_secs(window_secs),
        }
    }

    // =========================================================================
    // Pipeline metrics
    // =========================================================================

    /// Record chunking time.
    pub fn record_chunking(&self, duration_ms: f64) {
        if let Ok(mut inner) = self.inner.write() {
            inner.chunking.record(duration_ms);
        }
    }

    /// Record tokenization time.
    pub fn record_tokenization(&self, duration_ms: f64) {
        if let Ok(mut inner) = self.inner.write() {
            inner.tokenization.record(duration_ms);
        }
    }

    /// Record embedding (GPU inference) time.
    pub fn record_embedding(&self, duration_ms: f64) {
        if let Ok(mut inner) = self.inner.write() {
            inner.embedding.record(duration_ms);
        }
    }

    /// Record HNSW indexing time.
    pub fn record_hnsw_indexing(&self, duration_ms: f64) {
        if let Ok(mut inner) = self.inner.write() {
            inner.hnsw_indexing.record(duration_ms);
        }
    }

    /// Record BM25 indexing time.
    pub fn record_bm25_indexing(&self, duration_ms: f64) {
        if let Ok(mut inner) = self.inner.write() {
            inner.bm25_indexing.record(duration_ms);
        }
    }

    // =========================================================================
    // Search metrics
    // =========================================================================

    /// Record search metrics from a completed search operation.
    ///
    /// # Arguments
    /// * `query_embed_ms` - Time to embed the query text
    /// * `vector_ms` - Time for HNSW vector search
    /// * `keyword_ms` - Time for BM25 keyword search
    /// * `fusion_ms` - Time for RRF fusion
    /// * `result_count` - Number of results returned
    /// * `vector_count` - Number of vector search results before fusion
    /// * `keyword_count` - Number of keyword search results before fusion
    /// * `top_score` - Score of the top result (if any)
    /// * `median_score` - Median score of results (if any)
    #[allow(clippy::too_many_arguments)]
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
            inner
                .search_total
                .record(query_embed_ms + vector_ms + keyword_ms + fusion_ms);
            inner.last_search = Some(LastSearchInfo {
                result_count,
                vector_count,
                keyword_count,
                top_score,
                median_score,
            });
        }
    }

    // =========================================================================
    // Scheduler metrics
    // =========================================================================

    /// Record scheduler request timing.
    ///
    /// # Arguments
    /// * `queue_wait_ms` - Time spent waiting in queue before processing
    /// * `inference_ms` - Time spent on actual model inference
    pub fn record_scheduler_request(&self, queue_wait_ms: f64, inference_ms: f64) {
        if let Ok(mut inner) = self.inner.write() {
            inner.scheduler_queue_wait.record(queue_wait_ms);
            inner.scheduler_inference.record(inference_ms);
        }
    }

    /// Update scheduler state gauges.
    ///
    /// # Arguments
    /// * `queue_depth` - Current number of pending requests
    /// * `requests_completed` - Total requests completed since startup
    pub fn update_scheduler_stats(&self, queue_depth: usize, requests_completed: u64) {
        if let Ok(mut inner) = self.inner.write() {
            inner.scheduler_queue_depth = queue_depth;
            inner.scheduler_requests_completed = requests_completed;
        }
    }

    // =========================================================================
    // Snapshot and maintenance
    // =========================================================================

    /// Prune old samples outside the window.
    pub fn prune(&self) {
        if let Ok(mut inner) = self.inner.write() {
            // Pipeline metrics use the standard window
            inner.chunking.prune(self.window);
            inner.tokenization.prune(self.window);
            inner.embedding.prune(self.window);
            inner.hnsw_indexing.prune(self.window);
            inner.bm25_indexing.prune(self.window);

            // Search metrics use the longer search window
            inner.search_query_embed.prune(self.search_window);
            inner.search_vector.prune(self.search_window);
            inner.search_keyword.prune(self.search_window);
            inner.search_fusion.prune(self.search_window);
            inner.search_total.prune(self.search_window);

            // Scheduler metrics use the standard window
            inner.scheduler_queue_wait.prune(self.window);
            inner.scheduler_inference.prune(self.window);
        }
    }

    /// Get a snapshot of current metrics for UI display.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let inner = match self.inner.read() {
            Ok(inner) => inner,
            Err(_) => return MetricsSnapshot::default(),
        };

        let last_search = inner.last_search.as_ref();

        MetricsSnapshot {
            // Pipeline metrics
            chunking_avg_ms: inner.chunking.rolling_avg(self.window),
            chunking_count: inner.chunking.rolling_count(self.window),
            chunking_throughput: inner.chunking.throughput(self.window),

            tokenization_avg_ms: inner.tokenization.rolling_avg(self.window),
            tokenization_count: inner.tokenization.rolling_count(self.window),

            embedding_avg_ms: inner.embedding.rolling_avg(self.window),
            embedding_count: inner.embedding.rolling_count(self.window),
            embedding_throughput: inner.embedding.throughput(self.window),

            hnsw_avg_ms: inner.hnsw_indexing.rolling_avg(self.window),
            hnsw_count: inner.hnsw_indexing.rolling_count(self.window),

            bm25_avg_ms: inner.bm25_indexing.rolling_avg(self.window),
            bm25_count: inner.bm25_indexing.rolling_count(self.window),

            total_chunks_processed: inner.chunking.total_count,
            total_embeddings_generated: inner.embedding.total_count,

            // Search metrics (use longer search window)
            search: SearchSnapshot {
                query_embed_avg_ms: inner.search_query_embed.rolling_avg(self.search_window),
                vector_search_avg_ms: inner.search_vector.rolling_avg(self.search_window),
                keyword_search_avg_ms: inner.search_keyword.rolling_avg(self.search_window),
                fusion_avg_ms: inner.search_fusion.rolling_avg(self.search_window),
                total_latency_avg_ms: inner.search_total.rolling_avg(self.search_window),
                query_count: inner.search_total.rolling_count(self.search_window),

                last_result_count: last_search.map(|s| s.result_count),
                last_vector_count: last_search.map(|s| s.vector_count),
                last_keyword_count: last_search.map(|s| s.keyword_count),
                last_top_score: last_search.and_then(|s| s.top_score),
                last_median_score: last_search.and_then(|s| s.median_score),
            },

            // Scheduler metrics
            scheduler: SchedulerSnapshot {
                queue_depth: inner.scheduler_queue_depth,
                queue_wait_avg_ms: inner.scheduler_queue_wait.rolling_avg(self.window),
                inference_avg_ms: inner.scheduler_inference.rolling_avg(self.window),
                requests_completed: inner.scheduler_requests_completed,
            },
        }
    }

    /// Get the window duration.
    pub fn window(&self) -> Duration {
        self.window
    }

    /// Clear all metrics data.
    ///
    /// Resets all counters and samples to their initial state.
    /// Used when clearing storage to ensure metrics reflect the reset state.
    pub fn clear(&self) {
        if let Ok(mut inner) = self.inner.write() {
            *inner = MetricsInner::default();
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// Global metrics instance
use once_cell::sync::Lazy;

static GLOBAL_METRICS: Lazy<PerformanceMetrics> = Lazy::new(PerformanceMetrics::new);

/// Get the global metrics collector.
pub fn global_metrics() -> &'static PerformanceMetrics {
    &GLOBAL_METRICS
}

/// Convenience macro for timing a block and recording to a metric.
#[macro_export]
macro_rules! time_operation {
    ($metric:ident, $block:expr) => {{
        let start = instant::Instant::now();
        let result = $block;
        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        $crate::metrics::global_metrics().$metric(duration_ms);
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_record_and_snapshot() {
        let metrics = PerformanceMetrics::with_window(60);

        metrics.record_chunking(100.0);
        metrics.record_chunking(200.0);
        metrics.record_chunking(150.0);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.chunking_count, 3);
        assert!((snapshot.chunking_avg_ms.unwrap() - 150.0).abs() < 0.1);
    }

    #[test]
    fn test_empty_snapshot() {
        let metrics = PerformanceMetrics::new();
        let snapshot = metrics.snapshot();

        assert_eq!(snapshot.chunking_count, 0);
        assert!(snapshot.chunking_avg_ms.is_none());
    }

    #[test]
    fn test_prune_old_samples() {
        let metrics = PerformanceMetrics::with_window(1); // 1 second window

        metrics.record_embedding(50.0);

        // Wait for sample to age out
        thread::sleep(Duration::from_millis(1100));

        metrics.prune();
        let snapshot = metrics.snapshot();

        assert_eq!(snapshot.embedding_count, 0);
        // But lifetime total should still be there
        assert_eq!(snapshot.total_embeddings_generated, 1);
    }

    #[test]
    fn test_throughput() {
        let metrics = PerformanceMetrics::with_window(10);

        // Record 5 samples
        for _ in 0..5 {
            metrics.record_embedding(10.0);
        }

        let snapshot = metrics.snapshot();
        // 5 samples in 10 second window = 0.5/sec
        assert!(snapshot.embedding_throughput > 0.0);
    }

    #[test]
    fn test_search_metrics_recording() {
        let metrics = PerformanceMetrics::new();
        metrics.record_search(10.0, 5.0, 3.0, 2.0, 10, 8, 4, Some(0.9), Some(0.5));

        let snapshot = metrics.snapshot();
        assert!(snapshot.search.query_embed_avg_ms.is_some());
        assert_eq!(snapshot.search.last_result_count, Some(10));
        assert_eq!(snapshot.search.last_vector_count, Some(8));
        assert_eq!(snapshot.search.last_keyword_count, Some(4));
        assert!((snapshot.search.last_top_score.unwrap() - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_scheduler_metrics_recording() {
        let metrics = PerformanceMetrics::new();
        metrics.record_scheduler_request(5.0, 20.0);
        metrics.update_scheduler_stats(3, 100);

        let snapshot = metrics.snapshot();
        assert!(snapshot.scheduler.queue_wait_avg_ms.is_some());
        assert!(snapshot.scheduler.inference_avg_ms.is_some());
        assert_eq!(snapshot.scheduler.queue_depth, 3);
        assert_eq!(snapshot.scheduler.requests_completed, 100);
    }

    #[test]
    fn test_clear_metrics() {
        let metrics = PerformanceMetrics::new();
        metrics.record_chunking(100.0);
        metrics.record_search(10.0, 5.0, 3.0, 2.0, 10, 8, 4, Some(0.9), Some(0.5));
        metrics.update_scheduler_stats(5, 50);

        metrics.clear();
        let snapshot = metrics.snapshot();

        assert_eq!(snapshot.chunking_count, 0);
        assert!(snapshot.search.last_result_count.is_none());
        assert_eq!(snapshot.scheduler.queue_depth, 0);
    }
}
