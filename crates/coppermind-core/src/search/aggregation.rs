//! File-level aggregation for search results.
//!
//! Groups chunk-level search results by source file for cleaner UX.

use super::types::{FileSearchResult, SearchResult};
use std::collections::HashMap;

/// Aggregates chunk-level search results into file-level results.
///
/// Groups chunks by their `metadata.source` field (file path/URL) and
/// uses the highest-scoring chunk to represent each file's relevance.
///
/// # Scoring Strategy (Max Aggregation)
///
/// **Why max score?** Chunks are already globally ranked by RRF. A file is
/// relevant if it contains at least one highly relevant passage. This matches
/// user intent: "Which files answer my question?"
///
/// Alternative considered: RRF re-ranking at file level. Not implemented because:
/// - Adds complexity without clear UX benefit
/// - Max score already preserves semantic correctness
/// - Chunk count provides secondary relevance signal
///
/// **Trade-off:** A file with one great chunk (0.9) ranks higher than a file
/// with multiple good chunks (0.8, 0.75, 0.7). This is intentional - the best
/// match matters most for search UX.
///
/// # Algorithm
///
/// 1. Group chunks by source field
/// 2. Sort chunks within each file by score (descending)
/// 3. Use best chunk's score as file score (max aggregation)
/// 4. Extract file name from path for display
/// 5. Sort files by best chunk score (preserves global ranking)
///
/// # Arguments
///
/// * `chunk_results` - Vector of chunk-level search results from hybrid search
///   (already globally ranked by RRF)
///
/// # Returns
///
/// Vector of file-level results, sorted by relevance (descending).
/// Each file contains all its chunks sorted by score for UI display.
///
/// # Examples
///
/// ```ignore
/// let chunk_results = vec![
///     SearchResult { metadata: { source: Some("foo.md") }, score: 0.9, ... },
///     SearchResult { metadata: { source: Some("foo.md") }, score: 0.7, ... },
///     SearchResult { metadata: { source: Some("bar.md") }, score: 0.8, ... },
/// ];
///
/// let file_results = aggregate_chunks_by_file(chunk_results);
/// // Returns: [
/// //   FileSearchResult { file_path: "foo.md", score: 0.9, chunks: [0.9, 0.7] },
/// //   FileSearchResult { file_path: "bar.md", score: 0.8, chunks: [0.8] }
/// // ]
/// ```
pub fn aggregate_chunks_by_file(chunk_results: Vec<SearchResult>) -> Vec<FileSearchResult> {
    // Group chunks by source path
    let mut file_groups: HashMap<String, Vec<SearchResult>> = HashMap::new();

    for chunk in chunk_results {
        // Use source field as grouping key (falls back to "Unknown" if missing)
        let source = chunk
            .metadata
            .source
            .clone()
            .unwrap_or_else(|| "Unknown".to_string());

        file_groups.entry(source).or_default().push(chunk);
    }

    // Build FileSearchResult for each file
    let mut file_results: Vec<FileSearchResult> = file_groups
        .into_iter()
        .map(|(file_path, mut chunks)| {
            // Sort chunks by score (descending) - best chunk first
            chunks.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Best chunk determines file relevance
            // Safety: chunks is guaranteed non-empty because file_groups only contains
            // entries that had at least one chunk pushed to them. Using .first().unwrap()
            // instead of [0] to make the invariant explicit and produce a clearer panic
            // message if the invariant is ever violated.
            let best_chunk = chunks.first().expect("file group cannot be empty");

            // Extract file name from path
            let file_name = extract_file_name(&file_path);

            FileSearchResult {
                file_path: file_path.clone(),
                file_name,
                score: best_chunk.score,
                vector_score: best_chunk.vector_score,
                keyword_score: best_chunk.keyword_score,
                created_at: best_chunk.metadata.created_at,
                chunks,
            }
        })
        .collect();

    // Sort files by score (descending)
    file_results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    file_results
}

/// Extracts file name from a path or URL.
///
/// Handles various formats:
/// - Unix paths: "/path/to/file.txt" → "file.txt"
/// - Windows paths: "C:\path\to\file.txt" → "file.txt"
/// - URLs: "https://example.com/docs/page.html" → "page.html"
/// - Plain names: "file.txt" → "file.txt"
///
/// # Arguments
///
/// * `path` - File path or URL
///
/// # Returns
///
/// File name extracted from path, or the full path if extraction fails.
fn extract_file_name(path: &str) -> String {
    // Try URL parsing first
    if path.starts_with("http://") || path.starts_with("https://") {
        if let Some(last_segment) = path.rsplit('/').next() {
            if !last_segment.is_empty() {
                return last_segment.to_string();
            }
        }
        // Fallback: use domain
        return path.to_string();
    }

    // File path: try both / and \ separators
    path.rsplit(&['/', '\\'])
        .next()
        .filter(|s| !s.is_empty())
        .unwrap_or(path)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::types::{DocId, DocumentMetadata};

    #[test]
    fn test_aggregate_chunks_by_file() {
        let chunks = vec![
            SearchResult {
                doc_id: DocId::from_u64(1),
                score: 0.9,
                vector_score: Some(0.85),
                keyword_score: Some(0.75),
                text: "High scoring chunk from foo.md".to_string(),
                metadata: DocumentMetadata {
                    filename: Some("foo.md (chunk 2)".to_string()),
                    source: Some("foo.md".to_string()),
                    created_at: 100,
                },
            },
            SearchResult {
                doc_id: DocId::from_u64(2),
                score: 0.7,
                vector_score: Some(0.65),
                keyword_score: Some(0.55),
                text: "Lower scoring chunk from foo.md".to_string(),
                metadata: DocumentMetadata {
                    filename: Some("foo.md (chunk 1)".to_string()),
                    source: Some("foo.md".to_string()),
                    created_at: 100,
                },
            },
            SearchResult {
                doc_id: DocId::from_u64(3),
                score: 0.8,
                vector_score: Some(0.75),
                keyword_score: Some(0.65),
                text: "Chunk from bar.md".to_string(),
                metadata: DocumentMetadata {
                    filename: Some("bar.md (chunk 1)".to_string()),
                    source: Some("bar.md".to_string()),
                    created_at: 200,
                },
            },
        ];

        let file_results = aggregate_chunks_by_file(chunks);

        // Should have 2 files
        assert_eq!(file_results.len(), 2);

        // First file should be foo.md (score 0.9)
        assert_eq!(file_results[0].file_path, "foo.md");
        assert_eq!(file_results[0].file_name, "foo.md");
        assert_eq!(file_results[0].score, 0.9);
        assert_eq!(file_results[0].chunks.len(), 2);
        assert_eq!(file_results[0].chunks[0].score, 0.9); // Best chunk first

        // Second file should be bar.md (score 0.8)
        assert_eq!(file_results[1].file_path, "bar.md");
        assert_eq!(file_results[1].score, 0.8);
        assert_eq!(file_results[1].chunks.len(), 1);
    }

    #[test]
    fn test_extract_file_name() {
        assert_eq!(extract_file_name("file.txt"), "file.txt");
        assert_eq!(extract_file_name("/path/to/file.txt"), "file.txt");
        assert_eq!(extract_file_name("C:\\path\\to\\file.txt"), "file.txt");
        assert_eq!(
            extract_file_name("https://example.com/docs/page.html"),
            "page.html"
        );
        assert_eq!(
            extract_file_name("http://example.com/path/to/doc.md"),
            "doc.md"
        );
        assert_eq!(
            extract_file_name("https://example.com/"),
            "https://example.com/"
        ); // Domain fallback
    }

    #[test]
    fn test_empty_input_returns_empty() {
        let chunks: Vec<SearchResult> = vec![];
        let file_results = aggregate_chunks_by_file(chunks);

        // Empty input should produce empty output (no panic)
        assert!(file_results.is_empty());
    }

    #[test]
    fn test_chunks_without_source() {
        let chunks = vec![SearchResult {
            doc_id: DocId::from_u64(1),
            score: 0.9,
            vector_score: Some(0.85),
            keyword_score: Some(0.75),
            text: "Chunk without source".to_string(),
            metadata: DocumentMetadata {
                filename: Some("test.txt".to_string()),
                source: None, // No source
                created_at: 100,
            },
        }];

        let file_results = aggregate_chunks_by_file(chunks);

        // Should group under "Unknown"
        assert_eq!(file_results.len(), 1);
        assert_eq!(file_results[0].file_path, "Unknown");
    }
}
