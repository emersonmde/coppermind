//! Index view components: UploadCard, FileRow, BatchList, IndexView

mod batch;
mod batch_list;
pub mod file_row; // Make public for parent module access
mod index_view;
mod upload_card;

// Re-exports for external use
pub use batch::{Batch, BatchMetrics, BatchStatus, FileInBatch};
#[allow(unused_imports)]
pub use batch_list::BatchList;
#[allow(unused_imports)]
pub use file_row::{FileMetrics, FileRow, FileStatus};
pub use index_view::IndexView;
#[allow(unused_imports)]
pub use upload_card::UploadCard;
