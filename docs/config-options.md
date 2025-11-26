# Configuration Options

This document catalogs all configurable options in Coppermind for future preference storage implementation.

## Overview

Currently, configuration values are hardcoded as constants throughout the codebase. This document serves as a reference for implementing a unified preferences system.

---

## Crawler Options

| Option | Location | Default | Description |
|--------|----------|---------|-------------|
| `max_pages` | `crates/coppermind/src/components/web_crawler.rs` | `100` | Maximum pages to crawl per session |
| `delay_ms` | `crates/coppermind/src/components/web_crawler.rs` | `500` | Politeness delay between requests (ms) |
| `parallel_requests` | `crates/coppermind/src/components/web_crawler.rs` | `2` | Concurrent HTTP requests |
| `same_origin_only` | `crates/coppermind/src/components/web_crawler.rs` | `true` | Restrict crawl to same origin |
| `max_depth` | UI selectable | `999` | Maximum link depth to follow |

### User-Facing Options (via UI)

These are already configurable via the crawler UI:
- `max_depth`: Dropdown (0, 1, 2, 3, 5, unlimited)
- `parallel_requests`: Dropdown (1, 2, 4, 8, 16)

---

## Embedding Options

| Option | Location | Default | Description |
|--------|----------|---------|-------------|
| `max_tokens` | `crates/coppermind/src/embedding/mod.rs` | `512` | Max tokens per chunk (model limit: 8192) |

### GPU Scheduler (Desktop)

GPU access is serialized via `SerialScheduler` to work around Candle's Metal threading bug:
- Location: `crates/coppermind/src/gpu/`
- Priority queue: search queries (P0) before background work (P2)
- Batch processing for efficient background embedding

**Status**: Using Candle 0.9.1 stable with Metal support.

**References:**
- [Candle Issue #2637](https://github.com/huggingface/candle/issues/2637) - Metal tensor assertion failure
- [ADR 006](adrs/006-gpu-scheduler.md) - GPU scheduler design

**Future**: Upgrade to stable 0.9.2+ when released with Metal fixes.

---

## Search Options

| Option | Location | Default | Description |
|--------|----------|---------|-------------|
| `MIN_EF_SEARCH` | `crates/coppermind-core/src/search/vector.rs` | `50` | Minimum HNSW ef_search parameter |
| `k` (RRF constant) | `crates/coppermind-core/src/search/fusion.rs` | `60` | Reciprocal Rank Fusion constant |
| `top_k` | `crates/coppermind/src/components/search/search_view.rs` | `20` | Number of search results to return |

---

## HTTP Client Options

| Option | Location | Default | Description |
|--------|----------|---------|-------------|
| `timeout` | `crates/coppermind/src/crawler/fetcher.rs` | `30s` | HTTP request timeout |
| `pool_max_idle_per_host` | `crates/coppermind/src/crawler/fetcher.rs` | `10` | Max idle connections per host |
| `user_agent` | `crates/coppermind/src/crawler/fetcher.rs` | `Coppermind/0.1.0...` | HTTP User-Agent header |

---

## Model Configuration

| Option | Location | Default | Description |
|--------|----------|---------|-------------|
| `hidden_size` | `crates/coppermind/src/embedding/config.rs` | `512` | JinaBERT hidden dimension |
| `num_hidden_layers` | `crates/coppermind/src/embedding/config.rs` | `4` | Transformer layers |
| `num_attention_heads` | `crates/coppermind/src/embedding/config.rs` | `8` | Attention heads |
| `max_position_embeddings` | `crates/coppermind/src/embedding/config.rs` | `8192` | Max sequence length |
| `intermediate_size` | `crates/coppermind/src/embedding/config.rs` | `2048` | FFN intermediate size |

**Note**: Model config is fixed at compile time and tied to the model weights. These are not user-configurable.

---

## Future Preferences System

### Storage Strategy

1. **Web (WASM)**: Use `localStorage` or OPFS for JSON config
2. **Desktop**: Use platform config directory (`~/.config/coppermind/` on Linux, `~/Library/Application Support/` on macOS)

### Suggested Implementation

```rust
/// User preferences (persisted across sessions)
#[derive(Serialize, Deserialize, Default)]
pub struct UserPreferences {
    // Crawler
    pub crawler_max_pages: usize,
    pub crawler_delay_ms: u64,
    pub crawler_parallel_requests: usize,

    // Search
    pub search_top_k: usize,
}

/// Application config (not user-facing, may vary by platform)
pub struct AppConfig {
    pub http_timeout_secs: u64,
    pub http_pool_size: usize,
}
```

### Priority

1. **High**: `crawler_parallel_requests` - performance impact
2. **Medium**: `crawler_max_pages`, `crawler_delay_ms` - user convenience
3. **Low**: HTTP settings - rarely need changing

---

## Related Files

- `crates/coppermind/src/components/web_crawler.rs` - Crawler UI and config
- `crates/coppermind/src/crawler/fetcher.rs` - HTTP client config
- `crates/coppermind/src/embedding/config.rs` - Model configuration
- `crates/coppermind/src/gpu/` - GPU scheduler
- `crates/coppermind-core/src/search/` - Search engine constants
