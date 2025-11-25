// Reciprocal Rank Fusion (RRF) algorithm

use std::collections::HashMap;
use std::hash::Hash;

/// Standard RRF k parameter value from academic literature.
///
/// This constant (60) is the recommended value from the original RRF paper:
/// "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
/// by Cormack, Clarke, and Buettcher (SIGIR 2009).
///
/// The k parameter controls how much weight is given to top-ranked items:
/// - Smaller k → more emphasis on top results
/// - Larger k → more uniform weighting across ranks
/// - k=60 provides a good balance in most IR scenarios
pub const RRF_K: usize = 60;

/// Combine ranked results from multiple sources using RRF
///
/// RRF Formula: RRF_score(d) = sum_{r} 1 / (k + rank_r(d))
///
/// Where:
/// - d is a document/item
/// - r is a ranker (search method)
/// - rank_r(d) is the rank position of d in ranker r (1-indexed)
/// - k is a constant (typically 60) to reduce impact of high rankings
pub fn reciprocal_rank_fusion<T: Clone + Eq + Hash>(
    results_a: &[(T, f32)],
    results_b: &[(T, f32)],
    k: usize,
) -> Vec<(T, f32)> {
    // k parameter for RRF (typically 60)
    let k_param = k as f32;

    // Calculate RRF scores
    let mut rrf_scores: HashMap<T, f32> = HashMap::new();

    // Add scores from first ranker
    for (rank, (item, _score)) in results_a.iter().enumerate() {
        let rank_position = (rank + 1) as f32; // 1-indexed
        let rrf_contribution = 1.0 / (k_param + rank_position);
        *rrf_scores.entry(item.clone()).or_insert(0.0) += rrf_contribution;
    }

    // Add scores from second ranker
    for (rank, (item, _score)) in results_b.iter().enumerate() {
        let rank_position = (rank + 1) as f32; // 1-indexed
        let rrf_contribution = 1.0 / (k_param + rank_position);
        *rrf_scores.entry(item.clone()).or_insert(0.0) += rrf_contribution;
    }

    // Convert to vec and sort by RRF score descending
    let mut combined: Vec<(T, f32)> = rrf_scores.into_iter().collect();
    combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    combined
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf() {
        // Simulate results from two different search methods
        let vector_results = vec![(1, 0.9), (2, 0.8), (3, 0.7)];
        let keyword_results = vec![(3, 10.0), (1, 8.0), (4, 5.0)];

        let fused = reciprocal_rank_fusion(&vector_results, &keyword_results, RRF_K);

        // Doc 1 appears in both (rank 1 in vector, rank 2 in keyword)
        // Doc 3 appears in both (rank 3 in vector, rank 1 in keyword)
        // Doc 2 appears only in vector (rank 2)
        // Doc 4 appears only in keyword (rank 3)

        // Doc 1 and 3 should be top ranked due to appearing in both
        assert!(fused.len() >= 2);

        let top_ids: Vec<i32> = fused.iter().take(2).map(|(id, _)| *id).collect();
        assert!(top_ids.contains(&1));
        assert!(top_ids.contains(&3));
    }

    #[test]
    fn test_rrf_empty_inputs() {
        let results_a: Vec<(i32, f32)> = vec![];
        let results_b = vec![(1, 1.0), (2, 0.9)];

        let fused = reciprocal_rank_fusion(&results_a, &results_b, RRF_K);

        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].0, 1); // Highest ranked from results_b
    }

    #[test]
    fn test_rrf_both_empty() {
        let results_a: Vec<(i32, f32)> = vec![];
        let results_b: Vec<(i32, f32)> = vec![];

        let fused = reciprocal_rank_fusion(&results_a, &results_b, RRF_K);

        assert!(fused.is_empty());
    }

    #[test]
    fn test_rrf_single_ranker() {
        let results_a = vec![(1, 10.0), (2, 8.0), (3, 5.0)];
        let results_b: Vec<(i32, f32)> = vec![];

        let fused = reciprocal_rank_fusion(&results_a, &results_b, RRF_K);

        // With only one ranker, should preserve original order
        assert_eq!(fused.len(), 3);
        assert_eq!(fused[0].0, 1); // Original rank 1
        assert_eq!(fused[1].0, 2); // Original rank 2
        assert_eq!(fused[2].0, 3); // Original rank 3
    }

    #[test]
    fn test_rrf_k_parameter_impact() {
        let results_a = vec![(1, 0.9), (2, 0.8)];
        let results_b = vec![(2, 10.0), (1, 8.0)];

        // Test with different k values
        let fused_small_k = reciprocal_rank_fusion(&results_a, &results_b, 1);
        let fused_large_k = reciprocal_rank_fusion(&results_a, &results_b, 100);

        // Both should have same items
        assert_eq!(fused_small_k.len(), 2);
        assert_eq!(fused_large_k.len(), 2);

        // With different k values, scores will differ but order might be similar
        // Both doc1 and doc2 appear in both lists, so they should both be present
        let ids_small: Vec<i32> = fused_small_k.iter().map(|(id, _)| *id).collect();
        let ids_large: Vec<i32> = fused_large_k.iter().map(|(id, _)| *id).collect();

        assert!(ids_small.contains(&1) && ids_small.contains(&2));
        assert!(ids_large.contains(&1) && ids_large.contains(&2));
    }

    #[test]
    fn test_rrf_score_independence() {
        // RRF should use ranks only, not original scores
        let results_a = vec![(1, 100.0), (2, 0.01)]; // Very different scores
        let results_b = vec![(2, 0.99), (1, 0.01)]; // Reversed order, similar scores

        let fused = reciprocal_rank_fusion(&results_a, &results_b, RRF_K);

        assert_eq!(fused.len(), 2);

        // Both items appear in both lists:
        // Doc 1: rank 1 in A, rank 2 in B → RRF = 1/(k+1) + 1/(k+2)
        // Doc 2: rank 2 in A, rank 1 in B → RRF = 1/(k+2) + 1/(k+1)
        // RRF scores should be equal (symmetric)

        let score1 = fused
            .iter()
            .find(|(id, _)| *id == 1)
            .map(|(_, s)| s)
            .unwrap();
        let score2 = fused
            .iter()
            .find(|(id, _)| *id == 2)
            .map(|(_, s)| s)
            .unwrap();

        // Scores should be very close (within floating point precision)
        assert!(
            (score1 - score2).abs() < 0.0001,
            "RRF scores should be symmetric when ranks are swapped"
        );
    }

    #[test]
    fn test_rrf_overlapping_results() {
        let results_a = vec![(1, 0.9), (2, 0.8), (3, 0.7)];
        let results_b = vec![(1, 10.0), (3, 8.0), (5, 5.0)];

        let fused = reciprocal_rank_fusion(&results_a, &results_b, RRF_K);

        // Total unique items: 1, 2, 3, 5
        assert_eq!(fused.len(), 4);

        // Items appearing in both should rank higher
        // Doc 1: appears in both (rank 1 in both)
        // Doc 3: appears in both (rank 3 in A, rank 2 in B)
        // Doc 2: appears only in A (rank 2)
        // Doc 5: appears only in B (rank 3)

        let top_item = fused[0].0;
        // Doc 1 should be top (rank 1 in both lists)
        assert_eq!(top_item, 1);
    }

    #[test]
    fn test_rrf_ranking_order() {
        let results_a = vec![(1, 1.0), (2, 0.9), (3, 0.8), (4, 0.7)];
        let results_b = vec![(4, 10.0), (3, 9.0), (2, 8.0), (1, 7.0)];

        let fused = reciprocal_rank_fusion(&results_a, &results_b, RRF_K);

        // All items appear in both, but with reversed rankings
        assert_eq!(fused.len(), 4);

        // Calculate expected RRF scores:
        // Doc 1: 1/(k+1) + 1/(k+4)
        // Doc 2: 1/(k+2) + 1/(k+3)
        // Doc 3: 1/(k+3) + 1/(k+2)
        // Doc 4: 1/(k+4) + 1/(k+1)

        // Doc 1 and Doc 4 should have equal scores (symmetric)
        let score1 = fused
            .iter()
            .find(|(id, _)| *id == 1)
            .map(|(_, s)| *s)
            .unwrap();
        let score4 = fused
            .iter()
            .find(|(id, _)| *id == 4)
            .map(|(_, s)| *s)
            .unwrap();

        assert!(
            (score1 - score4).abs() < 0.0001,
            "Symmetric ranks should have equal RRF scores"
        );

        // Doc 2 and Doc 3 should have equal scores (symmetric)
        let score2 = fused
            .iter()
            .find(|(id, _)| *id == 2)
            .map(|(_, s)| *s)
            .unwrap();
        let score3 = fused
            .iter()
            .find(|(id, _)| *id == 3)
            .map(|(_, s)| *s)
            .unwrap();

        assert!(
            (score2 - score3).abs() < 0.0001,
            "Symmetric ranks should have equal RRF scores"
        );
    }
}
