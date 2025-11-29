//! Progress tracking types for indexing operations.
//!
//! These types provide structured progress information that can be used
//! to update UI, log progress, or track batch processing.

use instant::Instant;

/// Progress information for a single file/document indexing operation.
#[derive(Debug, Clone)]
pub struct IndexingProgress {
    /// Number of chunks processed so far
    pub chunks_completed: usize,
    /// Total number of chunks to process
    pub chunks_total: usize,
    /// Total tokens processed so far
    pub tokens_processed: usize,
    /// Time elapsed since start (milliseconds)
    pub elapsed_ms: u64,
}

impl IndexingProgress {
    /// Creates a new progress instance.
    pub fn new(
        chunks_completed: usize,
        chunks_total: usize,
        tokens_processed: usize,
        elapsed_ms: u64,
    ) -> Self {
        Self {
            chunks_completed,
            chunks_total,
            tokens_processed,
            elapsed_ms,
        }
    }

    /// Returns the completion percentage (0.0 to 100.0).
    pub fn percent_complete(&self) -> f64 {
        if self.chunks_total == 0 {
            0.0
        } else {
            (self.chunks_completed as f64 / self.chunks_total as f64) * 100.0
        }
    }

    /// Returns true if processing is complete.
    pub fn is_complete(&self) -> bool {
        self.chunks_completed >= self.chunks_total
    }

    /// Returns estimated time remaining in milliseconds, if computable.
    pub fn estimated_remaining_ms(&self) -> Option<u64> {
        if self.chunks_completed == 0 || self.chunks_completed >= self.chunks_total {
            return None;
        }
        let remaining_chunks = self.chunks_total - self.chunks_completed;
        let ms_per_chunk = self.elapsed_ms / self.chunks_completed as u64;
        Some(remaining_chunks as u64 * ms_per_chunk)
    }

    /// Returns throughput in tokens per second.
    pub fn tokens_per_second(&self) -> f64 {
        if self.elapsed_ms == 0 {
            0.0
        } else {
            self.tokens_processed as f64 / (self.elapsed_ms as f64 / 1000.0)
        }
    }
}

/// Progress information for batch processing of multiple files.
#[derive(Debug, Clone)]
pub struct BatchProgress {
    /// Number of files completed
    pub files_completed: usize,
    /// Total number of files to process
    pub files_total: usize,
    /// Total chunks processed across all files
    pub total_chunks: usize,
    /// Total tokens processed across all files
    pub total_tokens: usize,
    /// Time elapsed since batch start (milliseconds)
    pub elapsed_ms: u64,
    /// Current file being processed (if any)
    pub current_file: Option<String>,
    /// Progress within current file
    pub current_file_progress: Option<IndexingProgress>,
}

impl BatchProgress {
    /// Creates a new batch progress instance.
    pub fn new(files_total: usize) -> Self {
        Self {
            files_completed: 0,
            files_total,
            total_chunks: 0,
            total_tokens: 0,
            elapsed_ms: 0,
            current_file: None,
            current_file_progress: None,
        }
    }

    /// Returns the file completion percentage (0.0 to 100.0).
    pub fn percent_complete(&self) -> f64 {
        if self.files_total == 0 {
            0.0
        } else {
            (self.files_completed as f64 / self.files_total as f64) * 100.0
        }
    }

    /// Returns true if batch processing is complete.
    pub fn is_complete(&self) -> bool {
        self.files_completed >= self.files_total
    }

    /// Returns estimated time remaining in milliseconds, if computable.
    pub fn estimated_remaining_ms(&self) -> Option<u64> {
        if self.files_completed == 0 || self.files_completed >= self.files_total {
            return None;
        }
        let remaining_files = self.files_total - self.files_completed;
        let ms_per_file = self.elapsed_ms / self.files_completed as u64;
        Some(remaining_files as u64 * ms_per_file)
    }
}

/// Helper for tracking elapsed time during processing.
#[allow(dead_code)] // Part of public API for CLI/MCP usage
pub struct ProgressTimer {
    start: Instant,
}

#[allow(dead_code)] // Part of public API for CLI/MCP usage
impl ProgressTimer {
    /// Creates a new timer starting now.
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Returns elapsed time in milliseconds.
    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }
}

impl Default for ProgressTimer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexing_progress_percent() {
        let progress = IndexingProgress::new(50, 100, 5000, 1000);
        assert!((progress.percent_complete() - 50.0).abs() < 0.01);

        let progress = IndexingProgress::new(100, 100, 10000, 2000);
        assert!((progress.percent_complete() - 100.0).abs() < 0.01);
        assert!(progress.is_complete());
    }

    #[test]
    fn test_indexing_progress_empty() {
        let progress = IndexingProgress::new(0, 0, 0, 0);
        assert!((progress.percent_complete() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_estimated_remaining() {
        let progress = IndexingProgress::new(50, 100, 5000, 1000);
        // 50 chunks in 1000ms = 20ms/chunk, 50 remaining = 1000ms
        assert_eq!(progress.estimated_remaining_ms(), Some(1000));
    }

    #[test]
    fn test_tokens_per_second() {
        let progress = IndexingProgress::new(50, 100, 5000, 1000);
        // 5000 tokens in 1000ms = 5000 tokens/sec
        assert!((progress.tokens_per_second() - 5000.0).abs() < 0.01);
    }

    #[test]
    fn test_batch_progress() {
        let mut batch = BatchProgress::new(10);
        assert!((batch.percent_complete() - 0.0).abs() < 0.01);
        assert!(!batch.is_complete());

        batch.files_completed = 5;
        batch.elapsed_ms = 5000;
        assert!((batch.percent_complete() - 50.0).abs() < 0.01);
        assert_eq!(batch.estimated_remaining_ms(), Some(5000));

        batch.files_completed = 10;
        assert!(batch.is_complete());
    }
}
