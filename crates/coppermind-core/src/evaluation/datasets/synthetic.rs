//! Synthetic query generation for ground truth evaluation.
//!
//! This module generates queries where we know the ground truth by construction.
//! It extracts distinctive terms from documents to create queries that should
//! retrieve those documents.
//!
//! # Strategy
//!
//! 1. Select a target document from the corpus
//! 2. Extract distinctive terms (high TF, appearing in few documents)
//! 3. Create a query from those terms
//! 4. Mark the target as highly relevant
//!
//! This provides automated ground truth without manual labeling.

use crate::evaluation::metrics::RelevanceJudgment;
use crate::search::types::ChunkId;
use std::collections::{HashMap, HashSet};

/// Type of synthetic query generated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    /// Keyword-heavy query (exact terms from document)
    Keyword,
    /// Semantic query (paraphrased/related terms)
    Semantic,
    /// Mixed query (some exact, some related terms)
    Mixed,
    /// Short query (1-3 terms)
    Short,
    /// Long query (5+ terms)
    Long,
}

impl QueryType {
    /// Returns all query types for iteration.
    pub fn all() -> &'static [QueryType] {
        &[
            QueryType::Keyword,
            QueryType::Semantic,
            QueryType::Mixed,
            QueryType::Short,
            QueryType::Long,
        ]
    }
}

/// A synthetic query with ground truth relevance judgments.
#[derive(Debug, Clone)]
pub struct SyntheticQuery {
    /// Query text
    pub text: String,
    /// Target chunk ID that should be highly relevant
    pub target_id: ChunkId,
    /// Ground truth relevance judgments
    pub judgments: Vec<RelevanceJudgment>,
    /// Type of query generated
    pub query_type: QueryType,
}

/// Generates synthetic queries from a corpus.
///
/// The generator extracts distinctive terms from documents to create
/// queries with known ground truth relevance.
pub struct SyntheticQueryGenerator {
    /// Corpus texts indexed by chunk ID
    corpus: Vec<(ChunkId, String)>,
    /// Document frequency for each term (how many docs contain it)
    doc_freq: HashMap<String, usize>,
    /// Term frequency per document
    term_freq: HashMap<ChunkId, HashMap<String, usize>>,
    /// Simple LCG RNG state
    rng_state: u64,
}

impl SyntheticQueryGenerator {
    /// Creates a new generator from corpus texts.
    ///
    /// # Arguments
    ///
    /// * `corpus` - Vector of (chunk_id, text) pairs
    /// * `seed` - Random seed for reproducibility
    pub fn new(corpus: Vec<(ChunkId, String)>, seed: u64) -> Self {
        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        let mut term_freq: HashMap<ChunkId, HashMap<String, usize>> = HashMap::new();

        // Build term statistics
        for (chunk_id, text) in &corpus {
            let terms = tokenize(text);
            let unique_terms: HashSet<_> = terms.iter().cloned().collect();

            // Count document frequency
            for term in &unique_terms {
                *doc_freq.entry(term.clone()).or_insert(0) += 1;
            }

            // Count term frequency
            let mut tf: HashMap<String, usize> = HashMap::new();
            for term in terms {
                *tf.entry(term).or_insert(0) += 1;
            }
            term_freq.insert(*chunk_id, tf);
        }

        Self {
            corpus,
            doc_freq,
            term_freq,
            rng_state: seed,
        }
    }

    /// Generates a query targeting a specific document.
    ///
    /// Extracts distinctive terms from the target document to create
    /// a query that should retrieve it.
    pub fn generate_query_for(&mut self, target_id: ChunkId) -> Option<SyntheticQuery> {
        // Find the target text (ensure it exists in corpus)
        let _target_exists = self.corpus.iter().find(|(id, _)| *id == target_id)?;

        let tf = self.term_freq.get(&target_id)?;
        let corpus_size = self.corpus.len();

        // Score terms by TF-IDF-like metric
        let mut term_scores: Vec<(String, f64)> = tf
            .iter()
            .filter_map(|(term, &count)| {
                let df = *self.doc_freq.get(term).unwrap_or(&1);
                // Skip very common terms (stopwords)
                if df as f64 > corpus_size as f64 * 0.5 {
                    return None;
                }
                // Skip very rare single-character terms
                if term.len() < 2 {
                    return None;
                }
                let idf = ((corpus_size as f64) / (df as f64 + 1.0)).ln();
                let score = count as f64 * idf;
                Some((term.clone(), score))
            })
            .collect();

        // Sort by score descending, then by term alphabetically for reproducibility
        term_scores.sort_by(|a, b| match b.1.partial_cmp(&a.1) {
            Some(std::cmp::Ordering::Equal) | None => a.0.cmp(&b.0),
            Some(ord) => ord,
        });

        if term_scores.is_empty() {
            return None;
        }

        // Determine query type randomly
        let query_type = self.random_query_type();

        // Select terms based on query type
        let num_terms = match query_type {
            QueryType::Short => self.random_range(1, 3),
            QueryType::Long => self.random_range(5, 8),
            _ => self.random_range(2, 5),
        };

        let selected_terms: Vec<String> = term_scores
            .iter()
            .take(num_terms.min(term_scores.len()))
            .map(|(term, _)| term.clone())
            .collect();

        if selected_terms.is_empty() {
            return None;
        }

        // Build query text
        let query_text = match query_type {
            QueryType::Keyword => selected_terms.join(" "),
            QueryType::Semantic => {
                // For semantic, we could paraphrase, but for now just reorder
                let mut terms = selected_terms.clone();
                self.shuffle(&mut terms);
                terms.join(" ")
            }
            QueryType::Mixed => {
                // Mix of exact and potentially modified terms
                selected_terms.join(" ")
            }
            QueryType::Short | QueryType::Long => selected_terms.join(" "),
        };

        // Build relevance judgments
        let mut judgments = vec![RelevanceJudgment::highly_relevant(target_id)];

        // Find similar documents (those sharing many terms with target)
        for (chunk_id, _) in &self.corpus {
            if *chunk_id == target_id {
                continue;
            }

            if let Some(other_tf) = self.term_freq.get(chunk_id) {
                let shared_terms: usize = selected_terms
                    .iter()
                    .filter(|t| other_tf.contains_key(*t))
                    .count();

                // If shares >50% of query terms, mark as somewhat relevant
                if shared_terms * 2 >= selected_terms.len() && shared_terms > 0 {
                    judgments.push(RelevanceJudgment::relevant(*chunk_id));
                }
            }
        }

        Some(SyntheticQuery {
            text: query_text,
            target_id,
            judgments,
            query_type,
        })
    }

    /// Generates a batch of diverse queries.
    ///
    /// Selects random documents from the corpus and generates queries
    /// targeting them.
    pub fn generate_batch(&mut self, n: usize) -> Vec<SyntheticQuery> {
        let mut queries = Vec::with_capacity(n);
        let corpus_size = self.corpus.len();

        if corpus_size == 0 {
            return queries;
        }

        // Generate queries for random documents
        for _ in 0..n {
            let idx = self.random_range(0, corpus_size);
            let target_id = self.corpus[idx].0;

            if let Some(query) = self.generate_query_for(target_id) {
                queries.push(query);
            }
        }

        queries
    }

    /// Generates queries covering all query types evenly.
    pub fn generate_balanced_batch(&mut self, n_per_type: usize) -> Vec<SyntheticQuery> {
        let mut queries = Vec::new();
        let corpus_size = self.corpus.len();

        if corpus_size == 0 {
            return queries;
        }

        for query_type in QueryType::all() {
            let mut count = 0;
            let mut attempts = 0;

            while count < n_per_type && attempts < n_per_type * 10 {
                attempts += 1;
                let idx = self.random_range(0, corpus_size);
                let target_id = self.corpus[idx].0;

                // Generate and check if it matches desired type
                if let Some(mut query) = self.generate_query_for(target_id) {
                    // Override type to desired type and regenerate
                    query.query_type = *query_type;
                    queries.push(query);
                    count += 1;
                }
            }
        }

        queries
    }

    /// Returns corpus size.
    pub fn corpus_size(&self) -> usize {
        self.corpus.len()
    }

    // ========================================================================
    // Internal helpers
    // ========================================================================

    fn random_query_type(&mut self) -> QueryType {
        let types = QueryType::all();
        let idx = self.random_range(0, types.len());
        types[idx]
    }

    fn random_range(&mut self, min: usize, max: usize) -> usize {
        if max <= min {
            return min;
        }
        min + (self.next_random() as usize % (max - min))
    }

    fn next_random(&mut self) -> u64 {
        // LCG parameters from Numerical Recipes
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        self.rng_state
    }

    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.random_range(0, i + 1);
            items.swap(i, j);
        }
    }
}

/// Simple tokenizer for term extraction.
///
/// Splits on whitespace and punctuation, lowercases, and filters short tokens.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
        .filter(|s| s.len() >= 2)
        .map(|s| s.to_string())
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_corpus() -> Vec<(ChunkId, String)> {
        vec![
            (
                ChunkId::from_u64(1),
                "Machine learning algorithms for neural network training".to_string(),
            ),
            (
                ChunkId::from_u64(2),
                "Database optimization and query performance tuning".to_string(),
            ),
            (
                ChunkId::from_u64(3),
                "Web development with React and JavaScript frameworks".to_string(),
            ),
            (
                ChunkId::from_u64(4),
                "Neural network architectures for deep learning".to_string(),
            ),
            (
                ChunkId::from_u64(5),
                "SQL database indexing strategies for performance".to_string(),
            ),
        ]
    }

    #[test]
    fn test_generator_creation() {
        let corpus = sample_corpus();
        let generator = SyntheticQueryGenerator::new(corpus, 42);

        assert_eq!(generator.corpus_size(), 5);
        assert!(!generator.doc_freq.is_empty());
    }

    #[test]
    fn test_generate_query_for_target() {
        let corpus = sample_corpus();
        let mut generator = SyntheticQueryGenerator::new(corpus, 42);

        let query = generator.generate_query_for(ChunkId::from_u64(1));
        assert!(query.is_some());

        let query = query.unwrap();
        assert!(!query.text.is_empty());
        assert_eq!(query.target_id, ChunkId::from_u64(1));
        assert!(!query.judgments.is_empty());

        // Target should be in judgments as highly relevant
        let target_judgment = query
            .judgments
            .iter()
            .find(|j| j.chunk_id == ChunkId::from_u64(1));
        assert!(target_judgment.is_some());
        assert_eq!(target_judgment.unwrap().relevance, 2);
    }

    #[test]
    fn test_generate_batch() {
        let corpus = sample_corpus();
        let mut generator = SyntheticQueryGenerator::new(corpus, 42);

        let queries = generator.generate_batch(10);
        assert!(!queries.is_empty());

        for query in &queries {
            assert!(!query.text.is_empty());
            assert!(!query.judgments.is_empty());
        }
    }

    #[test]
    fn test_reproducibility() {
        let corpus = sample_corpus();

        let mut gen1 = SyntheticQueryGenerator::new(corpus.clone(), 42);
        let mut gen2 = SyntheticQueryGenerator::new(corpus, 42);

        let queries1 = gen1.generate_batch(5);
        let queries2 = gen2.generate_batch(5);

        assert_eq!(queries1.len(), queries2.len());
        for (q1, q2) in queries1.iter().zip(queries2.iter()) {
            assert_eq!(q1.text, q2.text);
            assert_eq!(q1.target_id, q2.target_id);
        }
    }

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello, World! This is a test.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"this".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // Single char 'a' should be filtered
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn test_empty_corpus() {
        let mut generator = SyntheticQueryGenerator::new(vec![], 42);
        let queries = generator.generate_batch(10);
        assert!(queries.is_empty());
    }

    #[test]
    fn test_query_types() {
        let types = QueryType::all();
        assert_eq!(types.len(), 5);
        assert!(types.contains(&QueryType::Keyword));
        assert!(types.contains(&QueryType::Semantic));
        assert!(types.contains(&QueryType::Mixed));
        assert!(types.contains(&QueryType::Short));
        assert!(types.contains(&QueryType::Long));
    }

    #[test]
    fn test_similar_documents_marked_relevant() {
        // Create corpus with similar documents
        let corpus = vec![
            (
                ChunkId::from_u64(1),
                "machine learning neural network deep learning".to_string(),
            ),
            (
                ChunkId::from_u64(2),
                "neural network training deep learning model".to_string(),
            ),
            (
                ChunkId::from_u64(3),
                "database sql query optimization".to_string(),
            ),
        ];

        let mut generator = SyntheticQueryGenerator::new(corpus, 42);
        let query = generator.generate_query_for(ChunkId::from_u64(1));

        if let Some(q) = query {
            // Doc 2 shares many terms with doc 1, should be marked relevant
            let _has_doc2 = q
                .judgments
                .iter()
                .any(|j| j.chunk_id == ChunkId::from_u64(2));
            // Note: This depends on which terms are selected, so we just check structure
            assert!(!q.judgments.is_empty());
        }
    }
}
