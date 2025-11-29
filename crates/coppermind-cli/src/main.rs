//! Coppermind CLI - Command-line interface for semantic search.
//!
//! # Usage
//!
//! ```bash
//! # Search existing index
//! cm "search query"
//! cm "rust embeddings" -n 5
//! cm "query" --json
//!
//! # Show help
//! cm --help
//! ```

mod config;
mod output;
mod search;

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

/// Coppermind semantic search CLI.
///
/// Search your indexed documents using hybrid semantic + keyword search.
/// Indexes are shared with the Coppermind desktop app.
#[derive(Parser)]
#[command(name = "cm", version, about)]
struct Cli {
    /// Search query
    query: Option<String>,

    /// Maximum number of results to return
    #[arg(short = 'n', long, default_value = "10")]
    limit: usize,

    /// Output results as JSON
    #[arg(long)]
    json: bool,

    /// Custom data directory (default: platform standard location)
    #[arg(long)]
    data_dir: Option<PathBuf>,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let filter = if cli.verbose {
        EnvFilter::new("info")
    } else {
        EnvFilter::new("warn")
    };
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();

    // Handle search query
    match &cli.query {
        Some(query) => {
            let results = search::execute_search(query, cli.limit, cli.data_dir.as_ref()).await?;

            let output = if cli.json {
                output::format_json(query, &results)
            } else {
                output::format_human(query, &results)
            };

            println!("{}", output);
        }
        None => {
            // No query provided - show help
            eprintln!("No search query provided. Use --help for usage information.");
            std::process::exit(1);
        }
    }

    Ok(())
}
