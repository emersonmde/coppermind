//! Output formatting for search results.
//!
//! Supports both human-readable terminal output and JSON for scripting.
//! Results are file-level (chunks aggregated by source), matching the UI behavior.

use coppermind_core::search::FileSearchResult;
use serde::Serialize;

/// Maximum characters to show in text snippet
const SNIPPET_MAX_LEN: usize = 200;

/// JSON output structure for search results
#[derive(Serialize)]
pub struct JsonOutput {
    pub query: String,
    pub results: Vec<JsonFileResult>,
}

/// File-level result in JSON format
#[derive(Serialize)]
pub struct JsonFileResult {
    /// File path or URL
    pub file_path: String,
    /// Display name (extracted from path)
    pub file_name: String,
    /// Best chunk's RRF score
    pub score: f32,
    /// Best chunk's vector/semantic score
    pub vector_score: Option<f32>,
    /// Best chunk's keyword/BM25 score
    pub keyword_score: Option<f32>,
    /// Number of matching chunks in this file
    pub chunk_count: usize,
    /// All matching chunks with their scores and snippets
    pub chunks: Vec<JsonChunk>,
}

/// Individual chunk within a file result
#[derive(Serialize)]
pub struct JsonChunk {
    pub score: f32,
    pub vector_score: Option<f32>,
    pub keyword_score: Option<f32>,
    pub snippet: String,
}

impl From<&FileSearchResult> for JsonFileResult {
    fn from(result: &FileSearchResult) -> Self {
        Self {
            file_path: result.file_path.clone(),
            file_name: result.file_name.clone(),
            score: result.score,
            vector_score: result.vector_score,
            keyword_score: result.keyword_score,
            chunk_count: result.chunks.len(),
            chunks: result
                .chunks
                .iter()
                .map(|chunk| JsonChunk {
                    score: chunk.score,
                    vector_score: chunk.vector_score,
                    keyword_score: chunk.keyword_score,
                    snippet: truncate_text(&chunk.text, SNIPPET_MAX_LEN),
                })
                .collect(),
        }
    }
}

/// Formats search results as JSON.
pub fn format_json(query: &str, results: &[FileSearchResult]) -> String {
    let output = JsonOutput {
        query: query.to_string(),
        results: results.iter().map(JsonFileResult::from).collect(),
    };
    serde_json::to_string_pretty(&output).unwrap_or_else(|_| "{}".to_string())
}

/// Formats search results for human-readable terminal output.
pub fn format_human(query: &str, results: &[FileSearchResult]) -> String {
    if results.is_empty() {
        return format!("No results found for \"{}\"", query);
    }

    let mut output = String::new();
    output.push_str(&format!(
        "Found {} file{} for \"{}\":\n\n",
        results.len(),
        if results.len() == 1 { "" } else { "s" },
        query
    ));

    for (i, result) in results.iter().enumerate() {
        // File header with score
        output.push_str(&format!(
            "{}. {} (score: {:.2})\n",
            i + 1,
            result.file_name,
            result.score
        ));

        // Show individual scores for best chunk
        let mut score_parts = Vec::new();
        if let Some(vs) = result.vector_score {
            score_parts.push(format!("semantic: {:.2}", vs));
        }
        if let Some(ks) = result.keyword_score {
            score_parts.push(format!("keyword: {:.2}", ks));
        }
        if !score_parts.is_empty() {
            output.push_str(&format!("   [{}]\n", score_parts.join(", ")));
        }

        // Show full path if different from file name
        if result.file_path != result.file_name {
            output.push_str(&format!("   Path: {}\n", result.file_path));
        }

        // Show chunk count if more than one
        if result.chunks.len() > 1 {
            output.push_str(&format!("   {} matching chunks\n", result.chunks.len()));
        }

        // Show best chunk snippet
        if let Some(best_chunk) = result.chunks.first() {
            let snippet = truncate_text(&best_chunk.text, SNIPPET_MAX_LEN);
            output.push_str(&format!("   {}\n", indent_text(&snippet, "   ")));
        }

        output.push('\n');
    }

    output.trim_end().to_string()
}

/// Truncates text to a maximum length, adding ellipsis if needed.
fn truncate_text(text: &str, max_len: usize) -> String {
    let text = text.trim();
    if text.len() <= max_len {
        text.to_string()
    } else {
        // Find a word boundary near max_len
        let truncated = &text[..max_len];
        if let Some(last_space) = truncated.rfind(' ') {
            format!("{}...", &truncated[..last_space])
        } else {
            format!("{}...", truncated)
        }
    }
}

/// Indents all lines of text after the first line.
fn indent_text(text: &str, indent: &str) -> String {
    text.lines()
        .enumerate()
        .map(|(i, line)| {
            if i == 0 {
                line.to_string()
            } else {
                format!("{}{}", indent, line)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use coppermind_core::search::{ChunkId, ChunkSourceMetadata, SearchResult};

    fn make_file_result(file_path: &str, text: &str, score: f32) -> FileSearchResult {
        FileSearchResult {
            file_path: file_path.to_string(),
            file_name: file_path
                .rsplit('/')
                .next()
                .unwrap_or(file_path)
                .to_string(),
            score,
            vector_score: Some(score * 0.9),
            keyword_score: Some(score * 0.8),
            created_at: 0,
            chunks: vec![SearchResult {
                chunk_id: ChunkId::from_u64(1),
                score,
                vector_score: Some(score * 0.9),
                keyword_score: Some(score * 0.8),
                text: text.to_string(),
                metadata: ChunkSourceMetadata {
                    filename: Some(format!("{} (chunk 1)", file_path)),
                    source: Some(file_path.to_string()),
                    created_at: 0,
                },
            }],
        }
    }

    #[test]
    fn test_format_human_empty() {
        let output = format_human("test query", &[]);
        assert!(output.contains("No results found"));
    }

    #[test]
    fn test_format_human_single() {
        let results = vec![make_file_result("test.md", "This is test content", 0.85)];
        let output = format_human("test", &results);
        assert!(output.contains("1 file"));
        assert!(output.contains("test.md"));
        assert!(output.contains("0.85"));
    }

    #[test]
    fn test_format_json() {
        let results = vec![make_file_result("/path/to/doc.txt", "Content here", 0.9)];
        let output = format_json("query", &results);
        assert!(output.contains("\"query\": \"query\""));
        assert!(output.contains("\"file_path\": \"/path/to/doc.txt\""));
        assert!(output.contains("\"file_name\": \"doc.txt\""));
        assert!(output.contains("\"score\": 0.9"));
        assert!(output.contains("\"chunk_count\": 1"));
    }

    #[test]
    fn test_truncate_text() {
        let short = "Short text";
        assert_eq!(truncate_text(short, 50), short);

        let long = "This is a much longer text that should be truncated at a reasonable point";
        let truncated = truncate_text(long, 30);
        assert!(truncated.ends_with("..."));
        assert!(truncated.len() <= 33); // 30 + "..."
    }
}
