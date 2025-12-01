//! MCP (Model Context Protocol) server implementation.
//!
//! Exposes Coppermind search as an MCP tool for AI assistants.
//! Uses the ADR-008 document-centric search architecture.

use crate::config;
use anyhow::{Context, Result};
use coppermind_core::embedding::{Embedder, JinaBertConfig, JinaBertEmbedder, TokenizerHandle};
use coppermind_core::search::{DocumentSearchResult, HybridSearchEngine};
use coppermind_core::storage::RedbDocumentStore;
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{
        CallToolResult, Content, ErrorCode, ErrorData, Implementation, ProtocolVersion,
        ServerCapabilities, ServerInfo,
    },
    schemars, tool, tool_handler, tool_router, ServerHandler, ServiceExt,
};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;

/// Search request parameters.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchParams {
    /// The search query text to find relevant documents.
    query: String,
    /// Maximum number of document results to return (default: 10).
    #[serde(default)]
    limit: Option<usize>,
}

/// Search response containing document-level results.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct SearchResponse {
    /// The query that was searched.
    query: String,
    /// Number of documents found.
    result_count: usize,
    /// Search results at the document level.
    results: Vec<DocumentResult>,
}

/// A single document result in the search response.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct DocumentResult {
    /// Document identifier.
    doc_id: u64,
    /// Source path or URL.
    source: String,
    /// Document title/display name.
    title: String,
    /// Relevance score (higher is better).
    score: f32,
    /// Document-level keyword score.
    doc_keyword_score: Option<f32>,
    /// Best chunk's semantic score.
    best_chunk_score: Option<f32>,
    /// Total character count of original document.
    char_count: usize,
    /// Matching chunks from this document with full text.
    chunks: Vec<ChunkResult>,
}

/// A single chunk within a document result.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct ChunkResult {
    /// Chunk identifier.
    chunk_id: u64,
    /// Chunk relevance score.
    score: f32,
    /// Full chunk text (not truncated).
    text: String,
}

/// Holds the initialized search components.
struct SearchComponents {
    engine: HybridSearchEngine<RedbDocumentStore>,
    embedder: JinaBertEmbedder,
    tokenizer: TokenizerHandle,
}

fn make_error(message: impl Into<String>) -> ErrorData {
    ErrorData {
        code: ErrorCode::INTERNAL_ERROR,
        message: Cow::from(message.into()),
        data: None,
    }
}

/// Initialize search components.
async fn init_components(data_dir: Option<&PathBuf>) -> Result<SearchComponents> {
    // Open database
    let db_path = config::database_path(data_dir)?;

    if !db_path.exists() {
        anyhow::bail!(
            "No index found at {}. Please index files using the desktop app first.",
            db_path.display()
        );
    }

    let store = RedbDocumentStore::open(&db_path)
        .with_context(|| format!("Failed to open database: {}", db_path.display()))?;

    // Load search engine
    let embedding_dim = JinaBertConfig::default().hidden_size;
    let engine = HybridSearchEngine::try_load_or_new(store, embedding_dim)
        .await
        .context("Failed to load search engine")?;

    if engine.len() == 0 {
        anyhow::bail!("Index is empty. Please index files using the desktop app first.");
    }

    // Load model and tokenizer
    let model_bytes = config::load_model_bytes().context("Failed to load model")?;
    let tokenizer_bytes = config::load_tokenizer_bytes().context("Failed to load tokenizer")?;

    let model_config = JinaBertConfig::default();
    let tokenizer =
        TokenizerHandle::from_bytes(tokenizer_bytes, model_config.max_position_embeddings)
            .map_err(|e| anyhow::anyhow!("Failed to create tokenizer: {}", e))?;

    let embedder = JinaBertEmbedder::from_bytes(model_bytes, tokenizer.vocab_size(), model_config)
        .map_err(|e| anyhow::anyhow!("Failed to create embedder: {}", e))?;

    Ok(SearchComponents {
        engine,
        embedder,
        tokenizer,
    })
}

/// Perform a search using block_in_place to safely run async code.
/// This avoids deadlocks that would occur with block_on inside a tokio runtime.
fn do_search_sync(
    components: &mut SearchComponents,
    query: &str,
    limit: usize,
) -> Result<SearchResponse, String> {
    // Embed query (synchronous)
    let tokens = components
        .tokenizer
        .tokenize(query)
        .map_err(|e| format!("Failed to tokenize query: {}", e))?;

    let query_embedding = components
        .embedder
        .embed_tokens(tokens)
        .map_err(|e| format!("Failed to embed query: {}", e))?;

    // Search using document-level search (ADR-008)
    // Use block_in_place to safely run async code from a sync context
    let doc_results: Vec<DocumentSearchResult> = tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(components.engine.search_documents(
            &query_embedding,
            query,
            limit,
        ))
    })
    .map_err(|e| format!("Search failed: {}", e))?;

    // Convert to response format with full chunk text
    let results: Vec<DocumentResult> = doc_results
        .iter()
        .map(|doc| DocumentResult {
            doc_id: doc.doc_id.as_u64(),
            source: doc.metadata.source_id.clone(),
            title: doc.metadata.title.clone(),
            score: doc.score,
            doc_keyword_score: doc.doc_keyword_score,
            best_chunk_score: doc.best_chunk_score,
            char_count: doc.metadata.char_count,
            chunks: doc
                .chunks
                .iter()
                .map(|chunk| ChunkResult {
                    chunk_id: chunk.chunk_id.as_u64(),
                    score: chunk.score,
                    text: chunk.text.clone(), // Full text, not truncated
                })
                .collect(),
        })
        .collect();

    Ok(SearchResponse {
        query: query.to_string(),
        result_count: results.len(),
        results,
    })
}

/// MCP server that exposes Coppermind semantic search.
#[derive(Clone)]
pub struct CoppermindMcpServer {
    /// Pre-initialized search components (shared across clones).
    /// Using std::sync::Mutex to avoid Send issues with tokio::sync::Mutex.
    components: Arc<StdMutex<SearchComponents>>,
    /// Tool router generated by the macro.
    tool_router: ToolRouter<Self>,
}

impl CoppermindMcpServer {
    /// Creates a new MCP server instance with pre-initialized components.
    fn new(components: SearchComponents) -> Self {
        Self {
            components: Arc::new(StdMutex::new(components)),
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router]
impl CoppermindMcpServer {
    #[tool(
        description = "Search indexed documents using hybrid semantic and keyword search. Returns document-level results with full chunk text for RAG/LLM consumption. Documents must be indexed using the Coppermind desktop app first."
    )]
    fn search(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let limit = params.limit.unwrap_or(10);

        let mut guard = self
            .components
            .lock()
            .map_err(|_| make_error("Failed to acquire lock on search components"))?;

        let response = do_search_sync(&mut guard, &params.query, limit).map_err(make_error)?;

        let json = serde_json::to_string_pretty(&response)
            .map_err(|e| make_error(format!("Failed to serialize response: {}", e)))?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }
}

#[tool_handler]
impl ServerHandler for CoppermindMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "coppermind".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                title: Some("Coppermind Semantic Search".to_string()),
                icons: None,
                website_url: Some("https://github.com/emersonmde/coppermind".to_string()),
            },
            instructions: Some(
                "Coppermind is a local-first semantic search engine. Use the 'search' tool to \
                 find relevant documents from your indexed files. Returns full document content \
                 for RAG/LLM consumption. Documents must first be indexed using the Coppermind \
                 desktop app."
                    .to_string(),
            ),
        }
    }
}

/// Runs the MCP server on stdio.
pub async fn run_mcp_server(data_dir: Option<PathBuf>) -> Result<()> {
    use rmcp::transport::stdio;

    // Initialize components before starting MCP server
    // This avoids Send issues with async initialization during tool calls
    eprintln!("Initializing Coppermind search engine...");
    let components = init_components(data_dir.as_ref()).await?;
    eprintln!("Search engine ready. Starting MCP server...");

    let server = CoppermindMcpServer::new(components);

    // Serve on stdio
    let service = server
        .serve(stdio())
        .await
        .context("Failed to start MCP server")?;

    // Wait for client disconnect
    service.waiting().await?;

    Ok(())
}
