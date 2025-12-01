//! Standard Information Retrieval metrics for evaluating search quality.
//!
//! This module implements metrics commonly used in IR research:
//! - NDCG (Normalized Discounted Cumulative Gain)
//! - MAP (Mean Average Precision)
//! - MRR (Mean Reciprocal Rank)
//! - Precision@k, Recall@k, F1@k
//!
//! # References
//!
//! - Järvelin & Kekäläinen (2002). "Cumulated gain-based evaluation of IR techniques"
//! - Voorhees & Harman (2005). "TREC: Experiment and Evaluation in Information Retrieval"

use crate::search::types::ChunkId;
use std::collections::{BTreeMap, HashMap};

/// Graded relevance judgment for a document/chunk.
///
/// Relevance is typically graded on a 3-point scale:
/// - 0: Not relevant
/// - 1: Somewhat relevant (partial match)
/// - 2: Highly relevant (perfect match)
///
/// Some datasets use binary relevance (0 or 1) which works fine with these metrics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelevanceJudgment {
    /// The chunk ID being judged
    pub chunk_id: ChunkId,
    /// Relevance grade (0 = not relevant, 1 = somewhat, 2 = highly)
    pub relevance: u8,
}

impl RelevanceJudgment {
    /// Creates a new relevance judgment.
    ///
    /// # Arguments
    ///
    /// * `chunk_id` - The chunk being judged
    /// * `relevance` - Relevance grade (typically 0, 1, or 2)
    pub fn new(chunk_id: ChunkId, relevance: u8) -> Self {
        Self {
            chunk_id,
            relevance,
        }
    }

    /// Creates a binary relevance judgment (relevant = 1).
    pub fn relevant(chunk_id: ChunkId) -> Self {
        Self::new(chunk_id, 1)
    }

    /// Creates a highly relevant judgment (relevance = 2).
    pub fn highly_relevant(chunk_id: ChunkId) -> Self {
        Self::new(chunk_id, 2)
    }

    /// Returns true if the chunk has any relevance (> 0).
    pub fn is_relevant(&self) -> bool {
        self.relevance > 0
    }
}

/// Evaluation metrics for a single query.
///
/// Contains all standard IR metrics computed for one query's results.
/// Use this to aggregate metrics across many queries.
#[derive(Debug, Clone, Default)]
pub struct QueryMetrics {
    /// NDCG@k for various k values
    pub ndcg_at_k: BTreeMap<usize, f64>,
    /// Mean Average Precision
    pub map: f64,
    /// Mean Reciprocal Rank
    pub mrr: f64,
    /// Precision@k for various k values
    pub precision_at_k: BTreeMap<usize, f64>,
    /// Recall@k for various k values
    pub recall_at_k: BTreeMap<usize, f64>,
    /// F1@k for various k values
    pub f1_at_k: BTreeMap<usize, f64>,
}

impl QueryMetrics {
    /// Computes all metrics for given results and judgments at standard k values.
    ///
    /// Standard k values: 1, 3, 5, 10, 20, 100
    ///
    /// # Arguments
    ///
    /// * `results` - Ranked search results as (chunk_id, score) pairs, highest score first
    /// * `judgments` - Ground truth relevance judgments
    pub fn compute(results: &[(ChunkId, f32)], judgments: &[RelevanceJudgment]) -> Self {
        let k_values = [1, 3, 5, 10, 20, 100];

        let mut ndcg_map = BTreeMap::new();
        let mut precision_map = BTreeMap::new();
        let mut recall_map = BTreeMap::new();
        let mut f1_map = BTreeMap::new();

        for k in k_values {
            ndcg_map.insert(k, ndcg_at_k(results, judgments, k));
            precision_map.insert(k, precision_at_k(results, judgments, k));
            recall_map.insert(k, recall_at_k(results, judgments, k));
            f1_map.insert(k, f1_at_k(results, judgments, k));
        }

        Self {
            ndcg_at_k: ndcg_map,
            map: average_precision(results, judgments),
            mrr: reciprocal_rank(results, judgments),
            precision_at_k: precision_map,
            recall_at_k: recall_map,
            f1_at_k: f1_map,
        }
    }
}

// ============================================================================
// NDCG (Normalized Discounted Cumulative Gain)
// ============================================================================

/// Computes NDCG@k (Normalized Discounted Cumulative Gain at k).
///
/// NDCG measures ranking quality with graded relevance judgments. It accounts
/// for both the relevance level of results and their position in the ranking.
///
/// # Formula
///
/// ```text
/// DCG@k = Σ (2^rel_i - 1) / log₂(i + 1)  for i in 1..=k
/// IDCG@k = DCG of ideal (perfectly sorted) ranking
/// NDCG@k = DCG@k / IDCG@k
/// ```
///
/// # Arguments
///
/// * `results` - Ranked search results as (chunk_id, score) pairs, highest score first
/// * `judgments` - Ground truth relevance judgments
/// * `k` - Cutoff position (only considers top k results)
///
/// # Returns
///
/// NDCG score between 0.0 and 1.0. Returns 1.0 if there are no relevant documents.
///
/// # Example
///
/// ```ignore
/// let results = vec![(ChunkId::from_u64(1), 0.9), (ChunkId::from_u64(2), 0.8)];
/// let judgments = vec![RelevanceJudgment::new(ChunkId::from_u64(1), 2)];
/// let ndcg = ndcg_at_k(&results, &judgments, 10);
/// assert!(ndcg > 0.9);  // Result 1 is highly relevant and ranked first
/// ```
pub fn ndcg_at_k(results: &[(ChunkId, f32)], judgments: &[RelevanceJudgment], k: usize) -> f64 {
    // Build lookup map for relevance judgments
    let rel_map: HashMap<ChunkId, u8> = judgments
        .iter()
        .map(|j| (j.chunk_id, j.relevance))
        .collect();

    // Compute DCG for the actual ranking
    let dcg = dcg_at_k(results, &rel_map, k);

    // Compute IDCG (ideal DCG) - sort judgments by relevance descending
    let mut ideal_rels: Vec<u8> = judgments.iter().map(|j| j.relevance).collect();
    ideal_rels.sort_by(|a, b| b.cmp(a));

    let idcg = ideal_rels
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, &rel)| gain(rel) / discount(i + 1))
        .sum::<f64>();

    // Avoid division by zero (no relevant documents)
    if idcg == 0.0 {
        1.0
    } else {
        dcg / idcg
    }
}

/// Computes DCG (Discounted Cumulative Gain) at k.
fn dcg_at_k(results: &[(ChunkId, f32)], rel_map: &HashMap<ChunkId, u8>, k: usize) -> f64 {
    results
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, (chunk_id, _))| {
            let rel = *rel_map.get(chunk_id).unwrap_or(&0);
            gain(rel) / discount(i + 1)
        })
        .sum()
}

/// Computes the gain from a relevance level.
///
/// Uses exponential gain: 2^rel - 1
/// This gives: rel=0 -> 0, rel=1 -> 1, rel=2 -> 3
#[inline]
fn gain(relevance: u8) -> f64 {
    (1u32 << relevance) as f64 - 1.0
}

/// Computes the discount factor for position (1-indexed).
///
/// Uses logarithmic discount: log₂(position + 1)
#[inline]
fn discount(position: usize) -> f64 {
    (position as f64 + 1.0).log2()
}

// ============================================================================
// MAP (Mean Average Precision)
// ============================================================================

/// Computes Average Precision for a single query.
///
/// Average Precision is the mean of precision values at each relevant result position.
/// It summarizes the precision-recall curve into a single value.
///
/// # Formula
///
/// ```text
/// AP = (1 / |relevant|) * Σ P(k) * rel(k)
/// where P(k) is precision at position k
/// and rel(k) is 1 if result at k is relevant, 0 otherwise
/// ```
///
/// # Arguments
///
/// * `results` - Ranked search results as (chunk_id, score) pairs, highest score first
/// * `judgments` - Ground truth relevance judgments (binary: relevant if relevance > 0)
///
/// # Returns
///
/// Average Precision between 0.0 and 1.0. Returns 0.0 if no relevant documents exist.
pub fn average_precision(results: &[(ChunkId, f32)], judgments: &[RelevanceJudgment]) -> f64 {
    let rel_set: HashMap<ChunkId, bool> = judgments
        .iter()
        .filter(|j| j.is_relevant())
        .map(|j| (j.chunk_id, true))
        .collect();

    let total_relevant = rel_set.len();
    if total_relevant == 0 {
        return 0.0;
    }

    let mut precision_sum = 0.0;
    let mut relevant_found = 0;

    for (i, (chunk_id, _)) in results.iter().enumerate() {
        if rel_set.contains_key(chunk_id) {
            relevant_found += 1;
            // Precision at this position
            let precision = relevant_found as f64 / (i + 1) as f64;
            precision_sum += precision;
        }
    }

    precision_sum / total_relevant as f64
}

// ============================================================================
// MRR (Mean Reciprocal Rank)
// ============================================================================

/// Computes Reciprocal Rank for a single query.
///
/// Reciprocal Rank is 1/position of the first relevant result. This metric
/// is particularly useful when users typically stop at the first good result.
///
/// # Formula
///
/// ```text
/// RR = 1 / rank_of_first_relevant_result
/// ```
///
/// # Arguments
///
/// * `results` - Ranked search results as (chunk_id, score) pairs, highest score first
/// * `judgments` - Ground truth relevance judgments (binary: relevant if relevance > 0)
///
/// # Returns
///
/// Reciprocal Rank between 0.0 and 1.0. Returns 0.0 if no relevant document is found.
pub fn reciprocal_rank(results: &[(ChunkId, f32)], judgments: &[RelevanceJudgment]) -> f64 {
    let rel_set: HashMap<ChunkId, bool> = judgments
        .iter()
        .filter(|j| j.is_relevant())
        .map(|j| (j.chunk_id, true))
        .collect();

    for (i, (chunk_id, _)) in results.iter().enumerate() {
        if rel_set.contains_key(chunk_id) {
            return 1.0 / (i + 1) as f64;
        }
    }

    0.0
}

// ============================================================================
// Set-Based Metrics: Precision, Recall, F1
// ============================================================================

/// Computes Precision@k.
///
/// Precision@k is the fraction of the top k results that are relevant.
///
/// # Formula
///
/// ```text
/// P@k = |relevant ∩ top_k| / k
/// ```
///
/// # Arguments
///
/// * `results` - Ranked search results as (chunk_id, score) pairs, highest score first
/// * `judgments` - Ground truth relevance judgments (binary: relevant if relevance > 0)
/// * `k` - Cutoff position
///
/// # Returns
///
/// Precision between 0.0 and 1.0.
pub fn precision_at_k(
    results: &[(ChunkId, f32)],
    judgments: &[RelevanceJudgment],
    k: usize,
) -> f64 {
    let rel_set: HashMap<ChunkId, bool> = judgments
        .iter()
        .filter(|j| j.is_relevant())
        .map(|j| (j.chunk_id, true))
        .collect();

    let top_k = results.iter().take(k);
    let relevant_in_top_k = top_k.filter(|(id, _)| rel_set.contains_key(id)).count();

    if k == 0 {
        0.0
    } else {
        relevant_in_top_k as f64 / k as f64
    }
}

/// Computes Recall@k.
///
/// Recall@k is the fraction of all relevant documents that appear in the top k results.
///
/// # Formula
///
/// ```text
/// R@k = |relevant ∩ top_k| / |relevant|
/// ```
///
/// # Arguments
///
/// * `results` - Ranked search results as (chunk_id, score) pairs, highest score first
/// * `judgments` - Ground truth relevance judgments (binary: relevant if relevance > 0)
/// * `k` - Cutoff position
///
/// # Returns
///
/// Recall between 0.0 and 1.0. Returns 1.0 if no relevant documents exist.
pub fn recall_at_k(results: &[(ChunkId, f32)], judgments: &[RelevanceJudgment], k: usize) -> f64 {
    let rel_set: HashMap<ChunkId, bool> = judgments
        .iter()
        .filter(|j| j.is_relevant())
        .map(|j| (j.chunk_id, true))
        .collect();

    let total_relevant = rel_set.len();
    if total_relevant == 0 {
        return 1.0; // No relevant docs means trivially perfect recall
    }

    let top_k = results.iter().take(k);
    let relevant_in_top_k = top_k.filter(|(id, _)| rel_set.contains_key(id)).count();

    relevant_in_top_k as f64 / total_relevant as f64
}

/// Computes F1@k (harmonic mean of Precision@k and Recall@k).
///
/// F1 balances precision and recall into a single metric.
///
/// # Formula
///
/// ```text
/// F1@k = 2 * P@k * R@k / (P@k + R@k)
/// ```
///
/// # Arguments
///
/// * `results` - Ranked search results as (chunk_id, score) pairs, highest score first
/// * `judgments` - Ground truth relevance judgments
/// * `k` - Cutoff position
///
/// # Returns
///
/// F1 score between 0.0 and 1.0. Returns 0.0 if both precision and recall are 0.
pub fn f1_at_k(results: &[(ChunkId, f32)], judgments: &[RelevanceJudgment], k: usize) -> f64 {
    let precision = precision_at_k(results, judgments, k);
    let recall = recall_at_k(results, judgments, k);

    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk_id(id: u64) -> ChunkId {
        ChunkId::from_u64(id)
    }

    fn results(ids: &[u64]) -> Vec<(ChunkId, f32)> {
        ids.iter()
            .enumerate()
            .map(|(i, &id)| (chunk_id(id), 1.0 - i as f32 * 0.1))
            .collect()
    }

    #[test]
    fn test_ndcg_perfect_ranking() {
        // Results: [1, 2, 3] with 1 highly relevant, 2 somewhat relevant
        let res = results(&[1, 2, 3]);
        let judgments = vec![
            RelevanceJudgment::highly_relevant(chunk_id(1)),
            RelevanceJudgment::relevant(chunk_id(2)),
        ];

        let ndcg = ndcg_at_k(&res, &judgments, 10);
        assert!(
            (ndcg - 1.0).abs() < 0.001,
            "Perfect ranking should have NDCG ≈ 1.0"
        );
    }

    #[test]
    fn test_ndcg_reversed_ranking() {
        // Results: [3, 2, 1] but 1 is highly relevant (at position 3)
        let res = results(&[3, 2, 1]);
        let judgments = vec![
            RelevanceJudgment::highly_relevant(chunk_id(1)),
            RelevanceJudgment::relevant(chunk_id(2)),
        ];

        let ndcg = ndcg_at_k(&res, &judgments, 10);
        // Should be significantly less than 1.0 because relevant docs are ranked lower
        assert!(ndcg < 0.9, "Reversed ranking should have lower NDCG");
        assert!(ndcg > 0.5, "Should still find some relevant docs");
    }

    #[test]
    fn test_ndcg_no_relevant_docs() {
        let res = results(&[1, 2, 3]);
        let judgments: Vec<RelevanceJudgment> = vec![];

        let ndcg = ndcg_at_k(&res, &judgments, 10);
        assert!(
            (ndcg - 1.0).abs() < 0.001,
            "No relevant docs should give NDCG = 1.0"
        );
    }

    #[test]
    fn test_precision_at_k() {
        // Results: [1, 2, 3, 4, 5] with [1, 3] relevant
        let res = results(&[1, 2, 3, 4, 5]);
        let judgments = vec![
            RelevanceJudgment::relevant(chunk_id(1)),
            RelevanceJudgment::relevant(chunk_id(3)),
        ];

        // P@1 = 1/1 = 1.0 (result 1 is relevant)
        assert!((precision_at_k(&res, &judgments, 1) - 1.0).abs() < 0.001);

        // P@2 = 1/2 = 0.5 (only result 1 is relevant in top 2)
        assert!((precision_at_k(&res, &judgments, 2) - 0.5).abs() < 0.001);

        // P@3 = 2/3 ≈ 0.667 (results 1 and 3 are relevant in top 3)
        assert!((precision_at_k(&res, &judgments, 3) - 0.667).abs() < 0.01);

        // P@5 = 2/5 = 0.4
        assert!((precision_at_k(&res, &judgments, 5) - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_recall_at_k() {
        // Results: [1, 2, 3, 4, 5] with [1, 3, 10] relevant (10 not in results)
        let res = results(&[1, 2, 3, 4, 5]);
        let judgments = vec![
            RelevanceJudgment::relevant(chunk_id(1)),
            RelevanceJudgment::relevant(chunk_id(3)),
            RelevanceJudgment::relevant(chunk_id(10)), // Not in results
        ];

        // R@1 = 1/3 ≈ 0.333 (found 1 of 3 relevant)
        assert!((recall_at_k(&res, &judgments, 1) - 0.333).abs() < 0.01);

        // R@3 = 2/3 ≈ 0.667 (found 1 and 3 of 3 relevant)
        assert!((recall_at_k(&res, &judgments, 3) - 0.667).abs() < 0.01);

        // R@5 = 2/3 ≈ 0.667 (still only 2 relevant in results)
        assert!((recall_at_k(&res, &judgments, 5) - 0.667).abs() < 0.01);
    }

    #[test]
    fn test_reciprocal_rank() {
        let judgments = vec![RelevanceJudgment::relevant(chunk_id(3))];

        // Result 3 at position 1 -> RR = 1/1 = 1.0
        let res1 = results(&[3, 1, 2]);
        assert!((reciprocal_rank(&res1, &judgments) - 1.0).abs() < 0.001);

        // Result 3 at position 3 -> RR = 1/3 ≈ 0.333
        let res2 = results(&[1, 2, 3]);
        assert!((reciprocal_rank(&res2, &judgments) - 0.333).abs() < 0.01);

        // Result 3 not in results -> RR = 0
        let res3 = results(&[1, 2, 4]);
        assert!((reciprocal_rank(&res3, &judgments)).abs() < 0.001);
    }

    #[test]
    fn test_average_precision() {
        // Results: [1, 2, 3, 4, 5] with [1, 3] relevant
        let res = results(&[1, 2, 3, 4, 5]);
        let judgments = vec![
            RelevanceJudgment::relevant(chunk_id(1)),
            RelevanceJudgment::relevant(chunk_id(3)),
        ];

        // AP = (P@1 + P@3) / 2 = (1.0 + 0.667) / 2 ≈ 0.833
        // P@1 = 1/1 = 1.0 (first hit at position 1)
        // P@3 = 2/3 ≈ 0.667 (second hit at position 3)
        let ap = average_precision(&res, &judgments);
        assert!((ap - 0.833).abs() < 0.01);
    }

    #[test]
    fn test_f1_at_k() {
        let res = results(&[1, 2, 3, 4, 5]);
        let judgments = vec![
            RelevanceJudgment::relevant(chunk_id(1)),
            RelevanceJudgment::relevant(chunk_id(3)),
            RelevanceJudgment::relevant(chunk_id(10)),
        ];

        // P@5 = 2/5 = 0.4, R@5 = 2/3 ≈ 0.667
        // F1@5 = 2 * 0.4 * 0.667 / (0.4 + 0.667) ≈ 0.5
        let f1 = f1_at_k(&res, &judgments, 5);
        assert!((f1 - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_query_metrics_compute() {
        let res = results(&[1, 2, 3]);
        let judgments = vec![
            RelevanceJudgment::highly_relevant(chunk_id(1)),
            RelevanceJudgment::relevant(chunk_id(2)),
        ];

        let metrics = QueryMetrics::compute(&res, &judgments);

        // Verify all k values are computed
        assert!(metrics.ndcg_at_k.contains_key(&1));
        assert!(metrics.ndcg_at_k.contains_key(&10));
        assert!(metrics.precision_at_k.contains_key(&5));
        assert!(metrics.recall_at_k.contains_key(&20));

        // Perfect ranking should have high scores
        assert!(metrics.mrr > 0.9);
        assert!(metrics.map > 0.9);
    }
}
