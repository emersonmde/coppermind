use super::file_row::{FileMetrics, FileStatus};

/// File being processed in a batch
#[derive(Clone, PartialEq)]
pub struct FileInBatch {
    pub name: String,
    pub status: FileStatus,
    pub progress_pct: f64,
    pub metrics: Option<FileMetrics>,
}

/// Status of a batch in the processing queue
#[derive(Clone, PartialEq)]
pub enum BatchStatus {
    /// Batch is queued but not yet processing
    Pending,
    /// Batch is currently being processed
    Processing,
    /// Batch has completed processing
    Completed,
}

/// Aggregate metrics for a completed batch
#[derive(Clone, PartialEq)]
pub struct BatchMetrics {
    pub file_count: usize,
    pub chunk_count: usize,
    pub token_count: usize,
    pub duration_ms: u64,
}

/// A batch of files at various stages of processing
#[derive(Clone, PartialEq)]
pub struct Batch {
    pub batch_number: usize,
    pub status: BatchStatus,
    pub files: Vec<FileInBatch>,
    pub metrics: Option<BatchMetrics>, // Only present when status == Completed
}

impl Batch {
    /// Calculate overall progress percentage for the batch
    pub fn progress_pct(&self) -> f64 {
        match self.status {
            BatchStatus::Pending => 0.0,
            BatchStatus::Completed => 100.0,
            BatchStatus::Processing => {
                if self.files.is_empty() {
                    return 0.0;
                }
                // Average progress across all files
                let total_progress: f64 = self.files.iter().map(|f| f.progress_pct).sum();
                total_progress / self.files.len() as f64
            }
        }
    }

    /// Get a subtitle describing the batch status
    pub fn subtitle(&self) -> String {
        match self.status {
            BatchStatus::Pending => format!("{} files queued", self.files.len()),
            BatchStatus::Processing => {
                let completed = self
                    .files
                    .iter()
                    .filter(|f| matches!(f.status, FileStatus::Completed))
                    .count();

                // If all files are completed, we're in the index rebuild phase
                if completed == self.files.len() {
                    format!("Rebuilding index ({} files)...", self.files.len())
                } else {
                    format!("Processing file {}/{}", completed + 1, self.files.len())
                }
            }
            BatchStatus::Completed => {
                if let Some(metrics) = &self.metrics {
                    let duration_secs = metrics.duration_ms as f64 / 1000.0;

                    let file_word = if metrics.file_count == 1 {
                        "file"
                    } else {
                        "files"
                    };
                    let chunk_word = if metrics.chunk_count == 1 {
                        "chunk"
                    } else {
                        "chunks"
                    };
                    let token_word = if metrics.token_count == 1 {
                        "token"
                    } else {
                        "tokens"
                    };

                    format!(
                        "{} {}, {} {}, {} {} in {:.1}s",
                        metrics.file_count,
                        file_word,
                        metrics.chunk_count,
                        chunk_word,
                        metrics.token_count,
                        token_word,
                        duration_secs
                    )
                } else {
                    "Completed".to_string()
                }
            }
        }
    }
}
