# Coppermind CLI

Command-line interface for [Coppermind](https://github.com/emersonmde/coppermind) semantic search. Search your indexed documents directly from the terminal or integrate with AI assistants via MCP.

## Installation

```bash
cargo install --path crates/coppermind-cli
```

This installs the `cm` binary.

## Prerequisites

Before using the CLI, you need:

1. **JinaBERT model files** - Run `./download-models.sh` from the workspace root
2. **An existing index** - Use the Coppermind desktop app to index documents first

The CLI shares the same index as the desktop app, stored at the platform-standard location:
- macOS: `~/Library/Application Support/dev.errorsignal.Coppermind/`
- Linux: `~/.local/share/dev.errorsignal.Coppermind/`
- Windows: `%APPDATA%\errorsignal\Coppermind\data\`

## Usage

### Command-Line Search

```bash
# Basic search
cm "your search query"

# Limit results
cm "rust embeddings" -n 5

# JSON output (for scripting)
cm "semantic search" --json

# Custom data directory
cm "query" --data-dir /path/to/index

# Verbose logging
cm "query" -v
```

### MCP Server Mode

Run Coppermind as a [Model Context Protocol](https://modelcontextprotocol.io/) server for integration with AI assistants like Claude:

```bash
cm --mcp
```

#### Claude Desktop Configuration

Add to your Claude Desktop config (`~/.config/claude/claude_desktop_config.json` on Linux/macOS):

```json
{
  "mcpServers": {
    "coppermind": {
      "command": "cm",
      "args": ["--mcp"]
    }
  }
}
```

Or with a custom data directory:

```json
{
  "mcpServers": {
    "coppermind": {
      "command": "cm",
      "args": ["--mcp", "--data-dir", "/path/to/index"]
    }
  }
}
```

#### Available MCP Tools

**search** - Search indexed documents using hybrid semantic and keyword search

Parameters:
- `query` (string, required): The search query text
- `limit` (number, optional): Maximum results to return (default: 10)

Returns file-level results with relevance scores and text snippets.

## Environment Variables

- `COPPERMIND_MODEL_DIR` - Override the model files location

## Output Formats

### Human-readable (default)

```
Found 3 files for "semantic search":

1. architecture.md (score: 0.85)
   [semantic: 0.82, keyword: 0.79]
   Path: docs/architecture.md
   2 matching chunks
   Hybrid search combines vector similarity with keyword matching...
```

### JSON (`--json`)

```json
{
  "query": "semantic search",
  "results": [
    {
      "file_path": "docs/architecture.md",
      "file_name": "architecture.md",
      "score": 0.85,
      "vector_score": 0.82,
      "keyword_score": 0.79,
      "chunk_count": 2,
      "chunks": [...]
    }
  ]
}
```
