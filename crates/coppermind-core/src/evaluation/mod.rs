//! Evaluation framework for measuring retrieval quality.
//!
//! This module provides standard Information Retrieval (IR) metrics for evaluating
//! search systems, including NDCG, MAP, MRR, Precision, Recall, and F1.
//!
//! # Two-Tier Evaluation
//!
//! Coppermind uses a two-tier evaluation approach:
//!
//! | Tier | Purpose | Data Source | When to Run |
//! |------|---------|-------------|-------------|
//! | **Tier 1: Synthetic** | Algorithm verification | Generated data | Every commit (CI) |
//! | **Tier 2: Real** | Semantic quality | Natural Questions | Weekly/manual |
//!
//! See [`datasets`] module for available datasets.
//!
//! # Overview
//!
//! The evaluation framework supports:
//! - **Graded relevance**: Documents can have multiple relevance levels (0=not, 1=somewhat, 2=highly relevant)
//! - **Position-aware metrics**: NDCG accounts for where relevant results appear
//! - **Statistical rigor**: Bootstrap confidence intervals and significance testing
//!
//! # Example
//!
//! ```ignore
//! use coppermind_core::evaluation::{RelevanceJudgment, ndcg_at_k, precision_at_k};
//! use coppermind_core::search::types::ChunkId;
//!
//! // Search results (chunk_id, score) in ranked order
//! let results = vec![
//!     (ChunkId::from_u64(1), 0.95),
//!     (ChunkId::from_u64(2), 0.82),
//!     (ChunkId::from_u64(3), 0.71),
//! ];
//!
//! // Ground truth relevance judgments
//! let judgments = vec![
//!     RelevanceJudgment::new(ChunkId::from_u64(1), 2),  // highly relevant
//!     RelevanceJudgment::new(ChunkId::from_u64(3), 1),  // somewhat relevant
//! ];
//!
//! let ndcg = ndcg_at_k(&results, &judgments, 10);
//! let precision = precision_at_k(&results, &judgments, 10);
//! ```
//!
//! # Metrics Reference
//!
//! | Metric | Description | Use Case |
//! |--------|-------------|----------|
//! | NDCG@k | Normalized Discounted Cumulative Gain | Graded relevance, position-aware |
//! | MAP | Mean Average Precision | Overall precision-recall tradeoff |
//! | MRR | Mean Reciprocal Rank | Finding first relevant result |
//! | P@k | Precision at k | Fraction of top-k that are relevant |
//! | R@k | Recall at k | Fraction of relevant found in top-k |
//! | F1@k | F1 score at k | Harmonic mean of P@k and R@k |

pub mod datasets;
pub mod metrics;
pub mod stats;

// Re-export commonly used types and functions
// Tier 1: Synthetic datasets
pub use datasets::{QueryType, SyntheticQuery, SyntheticQueryGenerator};
// Tier 2: Real datasets
pub use datasets::{
    load_natural_questions, EvalDataset, EvalDocument, EvalQuery, NaturalQuestionsDataset,
};
// Metrics
pub use metrics::{
    average_precision, f1_at_k, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank,
    QueryMetrics, RelevanceJudgment,
};
// Statistics
pub use stats::{bootstrap_ci, cohens_d, paired_ttest};
