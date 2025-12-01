//! Search command implementation.
//!
//! Handles loading the search engine and executing queries against existing indexes.

use crate::config;
use anyhow::{anyhow, Context, Result};
use coppermind_core::embedding::{Embedder, JinaBertConfig, JinaBertEmbedder, TokenizerHandle};
use coppermind_core::search::{DocumentSearchResult, HybridSearchEngine};
use coppermind_core::storage::RedbDocumentStore;
use std::path::PathBuf;
use tracing::info;

/// Performs a search against the existing index.
///
/// This function:
/// 1. Opens the existing document store (read-only effectively)
/// 2. Loads the search engine with existing index data
/// 3. Loads the embedding model and tokenizer
/// 4. Embeds the query
/// 5. Performs document-level hybrid search (vector + keyword with RRF fusion)
///
/// # Arguments
///
/// * `query` - The search query text
/// * `limit` - Maximum number of document results to return
/// * `data_dir` - Optional custom data directory
///
/// # Returns
///
/// Vector of document-level search results, sorted by relevance.
/// Each document contains its matching chunks with full text.
pub async fn execute_search(
    query: &str,
    limit: usize,
    data_dir: Option<&PathBuf>,
) -> Result<Vec<DocumentSearchResult>> {
    // 1. Open existing store
    let db_path = config::database_path(data_dir)?;

    if !db_path.exists() {
        return Err(anyhow!(
            "No index found at {}.\n\
             Please index some files using the desktop app first.",
            db_path.display()
        ));
    }

    info!("Opening database: {}", db_path.display());
    let store = RedbDocumentStore::open(&db_path)
        .with_context(|| format!("Failed to open database: {}", db_path.display()))?;

    // 2. Load search engine with existing index
    let embedding_dim = JinaBertConfig::default().hidden_size;
    info!("Loading search engine (embedding dim: {})", embedding_dim);
    let mut engine = HybridSearchEngine::try_load_or_new(store, embedding_dim)
        .await
        .context("Failed to load search engine")?;

    let chunk_count = engine.len();
    if chunk_count == 0 {
        return Err(anyhow!(
            "Index is empty. Please index some files using the desktop app first."
        ));
    }
    info!("Loaded index with {} chunks", chunk_count);

    // 3. Load model and tokenizer
    info!("Loading embedding model...");
    let model_bytes = config::load_model_bytes()?;
    let tokenizer_bytes = config::load_tokenizer_bytes()?;

    let config = JinaBertConfig::default();
    let tokenizer = TokenizerHandle::from_bytes(tokenizer_bytes, config.max_position_embeddings)
        .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

    let embedder = JinaBertEmbedder::from_bytes(model_bytes, tokenizer.vocab_size(), config)
        .map_err(|e| anyhow!("Failed to load embedding model: {}", e))?;

    info!("Model loaded successfully");

    // 4. Embed query
    let tokens = tokenizer
        .tokenize(query)
        .map_err(|e| anyhow!("Failed to tokenize query: {}", e))?;

    let query_embedding = embedder
        .embed_tokens(tokens)
        .map_err(|e| anyhow!("Failed to embed query: {}", e))?;

    // 5. Search using document-level search (ADR-008)
    info!("Searching for: \"{}\"", query);
    let results = engine
        .search_documents(&query_embedding, query, limit)
        .await
        .map_err(|e| anyhow!("Search failed: {}", e))?;

    info!("Found {} document results", results.len());
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_search_missing_database() {
        let result = execute_search("test", 10, Some(&PathBuf::from("/nonexistent/path"))).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("No index found") || err.contains("Failed to open"));
    }
}
