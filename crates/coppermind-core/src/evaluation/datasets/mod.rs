//! Ground truth datasets for evaluation.
//!
//! This module provides dataset abstractions for evaluating retrieval quality
//! using a two-tier approach:
//!
//! ## Tier 1: Synthetic Datasets
//!
//! - [`synthetic`] - Auto-generated queries with known relevant documents
//!
//! Synthetic queries are generated from the corpus itself, where we know
//! by construction which documents should be relevant. This enables:
//!
//! - Automated regression testing in CI
//! - Quick iteration during development
//! - Reproducible benchmarks (seeded RNG)
//!
//! **Use Tier 1 for**: Algorithm correctness verification, CI checks
//!
//! ## Tier 2: Real Datasets
//!
//! - [`natural_questions`] - Real Google search queries with human relevance judgments
//!
//! Real datasets provide human-annotated relevance judgments for measuring
//! actual semantic search quality with real embeddings.
//!
//! **Use Tier 2 for**: Semantic quality evaluation, feature improvement validation
//!
//! # EvalDataset Trait
//!
//! The [`EvalDataset`] trait provides a common interface for loading evaluation
//! datasets, making it easy to add new datasets in the future.
//!
//! # Example
//!
//! ```ignore
//! use coppermind_core::evaluation::datasets::synthetic::SyntheticQueryGenerator;
//!
//! let generator = SyntheticQueryGenerator::new(corpus_texts, 42);
//! let queries = generator.generate_batch(100);
//!
//! for query in queries {
//!     let results = engine.search(&query.embedding, &query.text, 10);
//!     let ndcg = ndcg_at_k(&results, &query.judgments, 10);
//! }
//! ```

pub mod natural_questions;
pub mod synthetic;

use std::collections::HashMap;

pub use natural_questions::{
    load_natural_questions, EvalDocument, EvalQuery, NaturalQuestionsDataset,
};
pub use synthetic::{QueryType, SyntheticQuery, SyntheticQueryGenerator};

// ============================================================================
// EvalDataset Trait
// ============================================================================

/// Common interface for evaluation datasets.
///
/// This trait provides a unified API for loading and accessing evaluation
/// datasets, making it easy to add new datasets (CQADupStack, etc.) in the future.
///
/// # Design
///
/// The trait uses borrowed slices to avoid copying large datasets. Implementations
/// should load data once and cache it internally.
pub trait EvalDataset {
    /// Dataset name for identification and reporting.
    fn name(&self) -> &str;

    /// Documents in the corpus.
    fn documents(&self) -> &[EvalDocument];

    /// Evaluation queries.
    fn queries(&self) -> &[EvalQuery];

    /// Relevance judgments: query_id -> (doc_id -> relevance).
    ///
    /// Relevance levels: 0 = not relevant, 1 = somewhat relevant, 2 = highly relevant
    fn qrels(&self) -> &HashMap<String, HashMap<String, u8>>;

    /// Get relevance judgments for a specific query.
    fn qrels_for_query(&self, query_id: &str) -> HashMap<String, u8> {
        self.qrels().get(query_id).cloned().unwrap_or_default()
    }

    /// Number of documents in the corpus.
    fn num_documents(&self) -> usize {
        self.documents().len()
    }

    /// Number of queries.
    fn num_queries(&self) -> usize {
        self.queries().len()
    }
}
