//! Output formatting for search results.
//!
//! Supports both human-readable terminal output and JSON for scripting.
//! Results are document-level, using the ADR-008 document-centric architecture.

use coppermind_core::search::DocumentSearchResult;
use serde::Serialize;

/// Maximum characters to show in text snippet for human output
const SNIPPET_MAX_LEN: usize = 200;

/// JSON output structure for search results
#[derive(Serialize)]
pub struct JsonOutput {
    pub query: String,
    pub results: Vec<JsonDocumentResult>,
}

/// Document-level result in JSON format
#[derive(Serialize)]
pub struct JsonDocumentResult {
    /// Document identifier
    pub doc_id: u64,
    /// Source path or URL
    pub source: String,
    /// Document title/display name
    pub title: String,
    /// Final fused relevance score (RRF)
    pub score: f32,
    /// Document-level keyword/BM25 score
    pub doc_keyword_score: Option<f32>,
    /// Best chunk's vector/semantic score
    pub best_chunk_score: Option<f32>,
    /// Number of matching chunks in this document
    pub chunk_count: usize,
    /// Total character count of original document
    pub char_count: usize,
    /// All matching chunks with their scores and full text
    pub chunks: Vec<JsonChunk>,
}

/// Individual chunk within a document result
#[derive(Serialize)]
pub struct JsonChunk {
    pub chunk_id: u64,
    pub score: f32,
    pub vector_score: Option<f32>,
    pub keyword_score: Option<f32>,
    /// Full chunk text (not truncated)
    pub text: String,
}

impl From<&DocumentSearchResult> for JsonDocumentResult {
    fn from(result: &DocumentSearchResult) -> Self {
        Self {
            doc_id: result.doc_id.as_u64(),
            source: result.metadata.source_id.clone(),
            title: result.metadata.title.clone(),
            score: result.score,
            doc_keyword_score: result.doc_keyword_score,
            best_chunk_score: result.best_chunk_score,
            chunk_count: result.chunks.len(),
            char_count: result.metadata.char_count,
            chunks: result
                .chunks
                .iter()
                .map(|chunk| JsonChunk {
                    chunk_id: chunk.chunk_id.as_u64(),
                    score: chunk.score,
                    vector_score: chunk.vector_score,
                    keyword_score: chunk.keyword_score,
                    text: chunk.text.clone(),
                })
                .collect(),
        }
    }
}

/// Formats search results as JSON.
///
/// Returns full chunk text for RAG/LLM consumption.
pub fn format_json(query: &str, results: &[DocumentSearchResult]) -> String {
    let output = JsonOutput {
        query: query.to_string(),
        results: results.iter().map(JsonDocumentResult::from).collect(),
    };
    serde_json::to_string_pretty(&output).unwrap_or_else(|_| "{}".to_string())
}

/// Formats search results for human-readable terminal output.
///
/// Truncates text snippets for readability.
pub fn format_human(query: &str, results: &[DocumentSearchResult]) -> String {
    if results.is_empty() {
        return format!("No results found for \"{}\"", query);
    }

    let mut output = String::new();
    output.push_str(&format!(
        "Found {} document{} for \"{}\":\n\n",
        results.len(),
        if results.len() == 1 { "" } else { "s" },
        query
    ));

    for (i, result) in results.iter().enumerate() {
        // Document header with score
        output.push_str(&format!(
            "{}. {} (score: {:.2})\n",
            i + 1,
            result.metadata.title,
            result.score
        ));

        // Show individual scores
        let mut score_parts = Vec::new();
        if let Some(vs) = result.best_chunk_score {
            score_parts.push(format!("semantic: {:.2}", vs));
        }
        if let Some(ks) = result.doc_keyword_score {
            score_parts.push(format!("keyword: {:.2}", ks));
        }
        if !score_parts.is_empty() {
            output.push_str(&format!("   [{}]\n", score_parts.join(", ")));
        }

        // Show full path if different from title
        if result.metadata.source_id != result.metadata.title {
            output.push_str(&format!("   Path: {}\n", result.metadata.source_id));
        }

        // Show chunk count and document size
        output.push_str(&format!(
            "   {} chunk{}, {} chars total\n",
            result.chunks.len(),
            if result.chunks.len() == 1 { "" } else { "s" },
            result.metadata.char_count
        ));

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
    use coppermind_core::search::{
        ChunkId, ChunkSourceMetadata, DocumentId, DocumentMetainfo, SearchResult,
    };

    fn make_doc_result(source: &str, text: &str, score: f32) -> DocumentSearchResult {
        let title = source.rsplit('/').next().unwrap_or(source).to_string();
        DocumentSearchResult {
            doc_id: DocumentId::from_u64(1),
            score,
            doc_keyword_score: Some(score * 0.8),
            best_chunk_score: Some(score * 0.9),
            metadata: DocumentMetainfo {
                source_id: source.to_string(),
                title,
                mime_type: None,
                content_hash: "abc123".to_string(),
                created_at: 0,
                updated_at: 0,
                char_count: text.len(),
                chunk_count: 1,
            },
            chunks: vec![SearchResult {
                chunk_id: ChunkId::from_u64(1),
                score,
                vector_score: Some(score * 0.9),
                keyword_score: Some(score * 0.8),
                text: text.to_string(),
                metadata: ChunkSourceMetadata {
                    filename: Some(source.to_string()),
                    source: Some(source.to_string()),
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
        let results = vec![make_doc_result("test.md", "This is test content", 0.85)];
        let output = format_human("test", &results);
        assert!(output.contains("1 document"));
        assert!(output.contains("test.md"));
        assert!(output.contains("0.85"));
    }

    #[test]
    fn test_format_json() {
        let results = vec![make_doc_result("/path/to/doc.txt", "Content here", 0.9)];
        let output = format_json("query", &results);
        assert!(output.contains("\"query\": \"query\""));
        assert!(output.contains("\"source\": \"/path/to/doc.txt\""));
        assert!(output.contains("\"title\": \"doc.txt\""));
        assert!(output.contains("\"score\": 0.9"));
        assert!(output.contains("\"chunk_count\": 1"));
        // Full text should be present (not truncated)
        assert!(output.contains("\"text\": \"Content here\""));
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
