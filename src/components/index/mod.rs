//! Index view components: UploadCard, FileRow, BatchCard, IndexView

mod batch;
mod batch_list;
mod current_batch; // Legacy, will be removed
pub mod file_row; // Make public for parent module access
mod index_view;
mod previous_batches; // Legacy, will be removed
mod upload_card;

// Re-exports for external use
pub use batch::{Batch, BatchMetrics, BatchStatus};
#[allow(unused_imports)]
pub use batch_list::BatchList;
#[allow(unused_imports)]
pub use current_batch::{CurrentBatch, FileInBatch}; // FileInBatch still used by batch.rs
#[allow(unused_imports)]
pub use file_row::{FileMetrics, FileRow, FileStatus};
pub use index_view::IndexView;
#[allow(unused_imports)]
pub use previous_batches::{BatchCard, BatchSummary, PreviousBatches}; // Legacy
#[allow(unused_imports)]
pub use upload_card::{Dropzone, UploadCard};
