//! Natural Questions dataset loader for Tier 2 evaluation.
//!
//! This module loads the Natural Questions dataset subset for real-world
//! semantic quality evaluation. Unlike Tier 1 synthetic benchmarks, this
//! uses actual Google search queries with human relevance judgments.
//!
//! # Dataset
//!
//! Natural Questions contains:
//! - Real Google search queries
//! - Wikipedia articles as passages
//! - Human-annotated answer spans (used as relevance judgments)
//!
//! # Data Format
//!
//! The dataset should be prepared using `scripts/prepare_nq.py` and stored in
//! `data/natural-questions/`:
//!
//! ```text
//! data/natural-questions/
//! ├── corpus.jsonl              # {"doc_id": "...", "title": "...", "text": "..."}
//! ├── queries.jsonl             # {"query_id": "...", "text": "..."}
//! ├── qrels.tsv                 # query_id \t doc_id \t relevance
//! └── embeddings.safetensors    # Pre-computed JinaBERT embeddings (optional)
//! ```
//!
//! # License
//!
//! Natural Questions is released under CC BY-SA 3.0, which is compatible with
//! MIT and allows commercial use with attribution.

use super::EvalDataset;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

// ============================================================================
// Data Structures
// ============================================================================

/// A document from an evaluation dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalDocument {
    /// Unique document identifier.
    pub doc_id: String,
    /// Document title (e.g., Wikipedia article title).
    pub title: String,
    /// Document text content.
    pub text: String,
    /// Pre-computed embedding (loaded separately from safetensors).
    #[serde(skip)]
    pub embedding: Option<Vec<f32>>,
}

/// A query from an evaluation dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalQuery {
    /// Unique query identifier.
    pub query_id: String,
    /// Query text.
    pub text: String,
    /// Pre-computed embedding (loaded separately from safetensors).
    #[serde(skip)]
    pub embedding: Option<Vec<f32>>,
}

/// Natural Questions dataset for Tier 2 evaluation.
///
/// This dataset contains real Google search queries with human-annotated
/// relevance judgments on Wikipedia passages.
#[derive(Debug)]
pub struct NaturalQuestionsDataset {
    /// Dataset name.
    name: String,
    /// Corpus documents.
    documents: Vec<EvalDocument>,
    /// Evaluation queries.
    queries: Vec<EvalQuery>,
    /// Relevance judgments: query_id -> (doc_id -> relevance).
    qrels: HashMap<String, HashMap<String, u8>>,
}

impl NaturalQuestionsDataset {
    /// Creates a new dataset from loaded components.
    pub fn new(
        documents: Vec<EvalDocument>,
        queries: Vec<EvalQuery>,
        qrels: HashMap<String, HashMap<String, u8>>,
    ) -> Self {
        Self {
            name: "natural-questions".to_string(),
            documents,
            queries,
            qrels,
        }
    }

    /// Sets embeddings for all documents.
    ///
    /// The embeddings vector should be indexed by document order in `documents()`.
    pub fn set_document_embeddings(&mut self, embeddings: Vec<Vec<f32>>) {
        for (doc, emb) in self.documents.iter_mut().zip(embeddings) {
            doc.embedding = Some(emb);
        }
    }

    /// Sets embeddings for all queries.
    ///
    /// The embeddings vector should be indexed by query order in `queries()`.
    pub fn set_query_embeddings(&mut self, embeddings: Vec<Vec<f32>>) {
        for (query, emb) in self.queries.iter_mut().zip(embeddings) {
            query.embedding = Some(emb);
        }
    }

    /// Returns documents with embeddings (filters out any without).
    pub fn documents_with_embeddings(&self) -> Vec<&EvalDocument> {
        self.documents
            .iter()
            .filter(|d| d.embedding.is_some())
            .collect()
    }

    /// Returns queries with embeddings (filters out any without).
    pub fn queries_with_embeddings(&self) -> Vec<&EvalQuery> {
        self.queries
            .iter()
            .filter(|q| q.embedding.is_some())
            .collect()
    }
}

impl EvalDataset for NaturalQuestionsDataset {
    fn name(&self) -> &str {
        &self.name
    }

    fn documents(&self) -> &[EvalDocument] {
        &self.documents
    }

    fn queries(&self) -> &[EvalQuery] {
        &self.queries
    }

    fn qrels(&self) -> &HashMap<String, HashMap<String, u8>> {
        &self.qrels
    }
}

// ============================================================================
// Loading Functions
// ============================================================================

/// Error type for dataset loading.
#[derive(Debug)]
pub enum DatasetError {
    /// IO error reading files.
    Io(std::io::Error),
    /// JSON parsing error.
    Json(serde_json::Error),
    /// Missing required file.
    MissingFile(String),
    /// Invalid data format.
    InvalidFormat(String),
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatasetError::Io(e) => write!(f, "IO error: {}", e),
            DatasetError::Json(e) => write!(f, "JSON error: {}", e),
            DatasetError::MissingFile(path) => write!(f, "Missing file: {}", path),
            DatasetError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
        }
    }
}

impl std::error::Error for DatasetError {}

impl From<std::io::Error> for DatasetError {
    fn from(e: std::io::Error) -> Self {
        DatasetError::Io(e)
    }
}

impl From<serde_json::Error> for DatasetError {
    fn from(e: serde_json::Error) -> Self {
        DatasetError::Json(e)
    }
}

/// Loads the Natural Questions dataset from a directory.
///
/// # Arguments
///
/// * `data_dir` - Path to the dataset directory containing corpus.jsonl,
///   queries.jsonl, and qrels.tsv.
///
/// # Returns
///
/// A `NaturalQuestionsDataset` with documents, queries, and relevance judgments
/// loaded. Embeddings are NOT loaded by this function - call
/// `load_embeddings_from_safetensors` separately.
///
/// # Example
///
/// ```ignore
/// let dataset = load_natural_questions(Path::new("data/natural-questions"))?;
/// println!("Loaded {} docs, {} queries", dataset.num_documents(), dataset.num_queries());
/// ```
pub fn load_natural_questions(data_dir: &Path) -> Result<NaturalQuestionsDataset, DatasetError> {
    let corpus_path = data_dir.join("corpus.jsonl");
    let queries_path = data_dir.join("queries.jsonl");
    let qrels_path = data_dir.join("qrels.tsv");

    // Check required files exist
    if !corpus_path.exists() {
        return Err(DatasetError::MissingFile(corpus_path.display().to_string()));
    }
    if !queries_path.exists() {
        return Err(DatasetError::MissingFile(
            queries_path.display().to_string(),
        ));
    }
    if !qrels_path.exists() {
        return Err(DatasetError::MissingFile(qrels_path.display().to_string()));
    }

    // Load corpus
    let documents = load_jsonl::<EvalDocument>(&corpus_path)?;

    // Load queries
    let queries = load_jsonl::<EvalQuery>(&queries_path)?;

    // Load qrels
    let qrels = load_qrels(&qrels_path)?;

    Ok(NaturalQuestionsDataset::new(documents, queries, qrels))
}

/// Loads a JSONL file into a vector of deserialized items.
fn load_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>, DatasetError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut items = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str(&line) {
            Ok(item) => items.push(item),
            Err(e) => {
                return Err(DatasetError::InvalidFormat(format!(
                    "Line {}: {}",
                    line_num + 1,
                    e
                )));
            }
        }
    }

    Ok(items)
}

/// Loads qrels from a TSV file.
///
/// Format: query_id \t doc_id \t relevance
fn load_qrels(path: &Path) -> Result<HashMap<String, HashMap<String, u8>>, DatasetError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut qrels: HashMap<String, HashMap<String, u8>> = HashMap::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 3 {
            return Err(DatasetError::InvalidFormat(format!(
                "Line {}: expected 3 tab-separated fields, got {}",
                line_num + 1,
                parts.len()
            )));
        }

        let query_id = parts[0].to_string();
        let doc_id = parts[1].to_string();
        let relevance: u8 = parts[2].parse().map_err(|_| {
            DatasetError::InvalidFormat(format!(
                "Line {}: invalid relevance value '{}'",
                line_num + 1,
                parts[2]
            ))
        })?;

        qrels.entry(query_id).or_default().insert(doc_id, relevance);
    }

    Ok(qrels)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_dataset() -> TempDir {
        let dir = TempDir::new().unwrap();

        // Write corpus.jsonl
        let mut corpus = File::create(dir.path().join("corpus.jsonl")).unwrap();
        writeln!(
            corpus,
            r#"{{"doc_id": "d1", "title": "Test Doc 1", "text": "This is test document one."}}"#
        )
        .unwrap();
        writeln!(
            corpus,
            r#"{{"doc_id": "d2", "title": "Test Doc 2", "text": "This is test document two."}}"#
        )
        .unwrap();

        // Write queries.jsonl
        let mut queries = File::create(dir.path().join("queries.jsonl")).unwrap();
        writeln!(queries, r#"{{"query_id": "q1", "text": "test document"}}"#).unwrap();
        writeln!(queries, r#"{{"query_id": "q2", "text": "another query"}}"#).unwrap();

        // Write qrels.tsv
        let mut qrels = File::create(dir.path().join("qrels.tsv")).unwrap();
        writeln!(qrels, "q1\td1\t2").unwrap();
        writeln!(qrels, "q1\td2\t1").unwrap();
        writeln!(qrels, "q2\td2\t2").unwrap();

        dir
    }

    #[test]
    fn test_load_natural_questions() {
        let dir = create_test_dataset();
        let dataset = load_natural_questions(dir.path()).unwrap();

        assert_eq!(dataset.name(), "natural-questions");
        assert_eq!(dataset.num_documents(), 2);
        assert_eq!(dataset.num_queries(), 2);

        // Check documents
        let docs = dataset.documents();
        assert_eq!(docs[0].doc_id, "d1");
        assert_eq!(docs[0].title, "Test Doc 1");
        assert_eq!(docs[1].doc_id, "d2");

        // Check queries
        let queries = dataset.queries();
        assert_eq!(queries[0].query_id, "q1");
        assert_eq!(queries[0].text, "test document");

        // Check qrels
        let qrels = dataset.qrels_for_query("q1");
        assert_eq!(qrels.get("d1"), Some(&2));
        assert_eq!(qrels.get("d2"), Some(&1));
    }

    #[test]
    fn test_missing_file() {
        let dir = TempDir::new().unwrap();
        let result = load_natural_questions(dir.path());
        assert!(matches!(result, Err(DatasetError::MissingFile(_))));
    }

    #[test]
    fn test_set_embeddings() {
        let dir = create_test_dataset();
        let mut dataset = load_natural_questions(dir.path()).unwrap();

        // Set document embeddings
        let doc_embeddings = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        dataset.set_document_embeddings(doc_embeddings);

        assert!(dataset.documents()[0].embedding.is_some());
        assert_eq!(
            dataset.documents()[0].embedding.as_ref().unwrap(),
            &vec![1.0, 2.0, 3.0]
        );

        // Set query embeddings
        let query_embeddings = vec![vec![7.0, 8.0, 9.0], vec![10.0, 11.0, 12.0]];
        dataset.set_query_embeddings(query_embeddings);

        assert!(dataset.queries()[0].embedding.is_some());
        assert_eq!(
            dataset.queries()[0].embedding.as_ref().unwrap(),
            &vec![7.0, 8.0, 9.0]
        );
    }

    #[test]
    fn test_documents_with_embeddings() {
        let dir = create_test_dataset();
        let mut dataset = load_natural_questions(dir.path()).unwrap();

        // Initially no embeddings
        assert_eq!(dataset.documents_with_embeddings().len(), 0);

        // Set embeddings for one document
        dataset.documents[0].embedding = Some(vec![1.0, 2.0, 3.0]);

        assert_eq!(dataset.documents_with_embeddings().len(), 1);
        assert_eq!(dataset.documents_with_embeddings()[0].doc_id, "d1");
    }
}
