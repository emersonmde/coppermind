// Reciprocal Rank Fusion (RRF) algorithm

use std::collections::HashMap;
use std::hash::Hash;

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

        let fused = reciprocal_rank_fusion(&vector_results, &keyword_results, 60);

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

        let fused = reciprocal_rank_fusion(&results_a, &results_b, 60);

        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].0, 1); // Highest ranked from results_b
    }
}
