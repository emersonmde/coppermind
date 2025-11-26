//! Performance metrics collection with rolling averages.
//!
//! This module provides lightweight metrics collection for tracking
//! performance over time. Metrics are stored in-memory and support
//! rolling averages over configurable time windows.

use instant::Instant;
use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// Default window size for rolling averages (60 seconds).
const DEFAULT_WINDOW_SECS: u64 = 60;

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

    /// Queue wait time (time between request submission and processing start).
    pub queue_wait_avg_ms: Option<f64>,

    /// Lifetime totals.
    pub total_chunks_processed: u64,
    pub total_embeddings_generated: u64,
}

/// Performance metrics collector.
///
/// Thread-safe collector for timing metrics with rolling averages.
/// Use `record_*` methods to log timings, and `snapshot()` to get
/// current statistics for UI display.
#[derive(Clone)]
pub struct PerformanceMetrics {
    inner: Arc<RwLock<MetricsInner>>,
    window: Duration,
}

struct MetricsInner {
    chunking: MetricData,
    tokenization: MetricData,
    embedding: MetricData,
    hnsw_indexing: MetricData,
    bm25_indexing: MetricData,
    queue_wait: MetricData,
}

impl Default for MetricsInner {
    fn default() -> Self {
        Self {
            chunking: MetricData::new(),
            tokenization: MetricData::new(),
            embedding: MetricData::new(),
            hnsw_indexing: MetricData::new(),
            bm25_indexing: MetricData::new(),
            queue_wait: MetricData::new(),
        }
    }
}

impl PerformanceMetrics {
    /// Create a new metrics collector with default 60-second window.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(MetricsInner::default())),
            window: Duration::from_secs(DEFAULT_WINDOW_SECS),
        }
    }

    /// Create a new metrics collector with custom window size.
    pub fn with_window(window_secs: u64) -> Self {
        Self {
            inner: Arc::new(RwLock::new(MetricsInner::default())),
            window: Duration::from_secs(window_secs),
        }
    }

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

    /// Record queue wait time.
    pub fn record_queue_wait(&self, duration_ms: f64) {
        if let Ok(mut inner) = self.inner.write() {
            inner.queue_wait.record(duration_ms);
        }
    }

    /// Prune old samples outside the window.
    pub fn prune(&self) {
        if let Ok(mut inner) = self.inner.write() {
            inner.chunking.prune(self.window);
            inner.tokenization.prune(self.window);
            inner.embedding.prune(self.window);
            inner.hnsw_indexing.prune(self.window);
            inner.bm25_indexing.prune(self.window);
            inner.queue_wait.prune(self.window);
        }
    }

    /// Get a snapshot of current metrics for UI display.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let inner = match self.inner.read() {
            Ok(inner) => inner,
            Err(_) => return MetricsSnapshot::default(),
        };

        MetricsSnapshot {
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

            queue_wait_avg_ms: inner.queue_wait.rolling_avg(self.window),

            total_chunks_processed: inner.chunking.total_count,
            total_embeddings_generated: inner.embedding.total_count,
        }
    }

    /// Get the window duration.
    pub fn window(&self) -> Duration {
        self.window
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
}
