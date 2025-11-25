# Configuration Options

This document catalogs all configurable options in Coppermind for future preference storage implementation.

## Overview

Currently, configuration values are hardcoded as constants throughout the codebase. This document serves as a reference for implementing a unified preferences system.

---

## Crawler Options

| Option | Location | Default | Description |
|--------|----------|---------|-------------|
| `max_pages` | `web_crawler.rs:62` | `100` | Maximum pages to crawl per session |
| `delay_ms` | `web_crawler.rs:63` | `500` | Politeness delay between requests (ms) |
| `parallel_requests` | `web_crawler.rs:64` | `2` | Concurrent HTTP requests |
| `same_origin_only` | `web_crawler.rs:61` | `true` | Restrict crawl to same origin |
| `max_depth` | UI selectable | `999` | Maximum link depth to follow |

### User-Facing Options (via UI)

These are already configurable via the crawler UI:
- `max_depth`: Dropdown (0, 1, 2, 3, 5, unlimited)
- `parallel_requests`: Dropdown (1, 2, 4, 8, 16)

---

## Embedding Options

| Option | Location | Default | Description |
|--------|----------|---------|-------------|
| `EMBEDDING_BATCH_SIZE` | `web_crawler.rs:26` | `10` | Pages per batch for streaming embedding |
| `MAX_CONCURRENT_EMBEDDINGS` | `embedding/mod.rs:87` | `1` | Max concurrent GPU embedding operations |
| `max_tokens` | `embedding/mod.rs` | `512` | Max tokens per chunk (model limit: 8192) |

### GPU Thread Safety (Candle-specific)

`MAX_CONCURRENT_EMBEDDINGS` controls the semaphore that serializes GPU access:
- **Current value**: 1 (workaround for Candle bug)
- **Reason**: Candle's Metal backend doesn't properly isolate command buffers between threads
- **Note**: This is NOT an inherent Metal/CUDA limitation - Metal command queues are thread-safe by design

This workaround prevents the Candle Metal crash:
```
-[AGXG15XFamilyCommandBuffer tryCoalescingPreviousComputeCommandEncoderWithConfig:...]:
failed assertion 'A command encoder is already encoding to this command buffer'
```

**References:**
- [Candle Issue #2637](https://github.com/huggingface/candle/issues/2637) - Metal tensor assertion failure
- [PR #3079/3090](https://github.com/huggingface/candle/pull/3090) - Fix via thread-isolated command buffers

**Future**: May be removable when upgrading Candle to a version with the thread-isolated command buffer fix.

### Batch Processing

The `EMBEDDING_BATCH_SIZE` controls the streaming batch threshold:
- **Smaller values (5)**: More responsive progress, slightly more overhead
- **Larger values (20)**: More efficient embedding, longer waits between updates
- **Trade-off**: Memory usage vs. responsiveness

---

## Search Options

| Option | Location | Default | Description |
|--------|----------|---------|-------------|
| `MIN_EF_SEARCH` | `vector.rs` | `50` | Minimum HNSW ef_search parameter |
| `DEBUG_TEXT_PREVIEW_LEN` | `engine.rs` | `100` | Characters to show in debug output |
| `k` (RRF constant) | `fusion.rs` | `60` | Reciprocal Rank Fusion constant |
| `top_k` | `search_view.rs` | `20` | Number of search results to return |

---

## HTTP Client Options

| Option | Location | Default | Description |
|--------|----------|---------|-------------|
| `timeout` | `fetcher.rs:28` | `30s` | HTTP request timeout |
| `pool_max_idle_per_host` | `fetcher.rs:29` | `10` | Max idle connections per host |
| `user_agent` | `fetcher.rs:27` | `Coppermind/0.1.0...` | HTTP User-Agent header |

---

## Model Configuration

| Option | Location | Default | Description |
|--------|----------|---------|-------------|
| `hidden_size` | `config.rs` | `512` | JinaBERT hidden dimension |
| `num_hidden_layers` | `config.rs` | `4` | Transformer layers |
| `num_attention_heads` | `config.rs` | `8` | Attention heads |
| `max_position_embeddings` | `config.rs` | `8192` | Max sequence length |
| `intermediate_size` | `config.rs` | `2048` | FFN intermediate size |

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

    // Embedding
    pub embedding_batch_size: usize,

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

1. **High**: `embedding_batch_size`, `crawler_parallel_requests` - performance impact
2. **Medium**: `crawler_max_pages`, `crawler_delay_ms` - user convenience
3. **Low**: HTTP settings - rarely need changing

---

## Related Files

- `crates/coppermind/src/components/web_crawler.rs` - Crawler UI and config
- `crates/coppermind/src/crawler/fetcher.rs` - HTTP client config
- `crates/coppermind/src/embedding/config.rs` - Model configuration
- `crates/coppermind-core/src/search/` - Search engine constants
