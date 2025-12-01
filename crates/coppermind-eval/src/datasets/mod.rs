//! Dataset loading for evaluation.

mod coppermind;

pub use coppermind::load_coppermind_dataset;

use std::collections::HashMap;

/// A loaded evaluation dataset ready for use.
#[derive(Debug)]
pub struct EvalDataset {
    /// Dataset name for reporting
    pub name: String,
    /// Documents: (doc_id, title, text)
    pub documents: Vec<(String, String, String)>,
    /// Queries: (query_id, text)
    pub queries: Vec<(String, String)>,
    /// Relevance judgments: query_id -> doc_id -> relevance score
    pub qrels: HashMap<String, HashMap<String, u8>>,
}

impl EvalDataset {
    /// Number of documents in the dataset
    pub fn num_documents(&self) -> usize {
        self.documents.len()
    }

    /// Number of queries in the dataset
    pub fn num_queries(&self) -> usize {
        self.queries.len()
    }

    /// Total number of relevance judgments
    pub fn num_qrels(&self) -> usize {
        self.qrels.values().map(|d| d.len()).sum()
    }
}
