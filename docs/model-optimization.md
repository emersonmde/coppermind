# Model Optimization Guide

This document covers optimization strategies for running JinaBERT embeddings in WebAssembly, including memory configuration and sequence length tuning.

## WASM Memory Configuration

### Current Status ❌
```toml
# .cargo/config.toml
[target.wasm32-unknown-unknown]
rustflags = [
  "--cfg", "getrandom_backend=\"wasm_js\"",
  "-C", "link-arg=--initial-memory=134217728",  # 128MB initial
  "-C", "link-arg=--max-memory=536870912",      # 512MB max
]
```

### Problem
The 512MB limit is overly conservative. WebAssembly (wasm32) supports up to **4GB** of memory as of 2020.

**Historical Context:**
- Pre-2020: 2GB limit
- 2020+: 4GB limit (standard across Chrome, Firefox, Safari)
- Future: Memory64 proposal enables >4GB (wasm64)

**Source:** [V8 Blog - Up to 4GB of memory in WebAssembly](https://v8.dev/blog/4gb-wasm-memory)

### Recommended Configuration ✅
```toml
# .cargo/config.toml
[target.wasm32-unknown-unknown]
rustflags = [
  "--cfg", "getrandom_backend=\"wasm_js\"",
  "-C", "link-arg=--initial-memory=268435456",   # 256MB initial
  "-C", "link-arg=--max-memory=4294967296",      # 4GB max (full wasm32 support)
]
```

**Rationale:**
- 256MB initial: Enough for model loading without excessive upfront allocation
- 4GB max: Full wasm32 capability, allows model to grow as needed
- Enables longer sequence lengths (see below)

---

## JinaBERT Sequence Length Optimization

### Current Status ❌
```rust
// src/embedding.rs - JinaBertConfig::default()
impl Default for JinaBertConfig {
    fn default() -> Self {
        Self {
            model_id: "jinaai/jina-embeddings-v2-small-en".to_string(),
            normalize_embeddings: true,
            hidden_size: 512,
            num_hidden_layers: 4,
            num_attention_heads: 8,
            intermediate_size: 2048,
            max_position_embeddings: 1024,  // ❌ Severely underutilized
        }
    }
}
```

### Problem
**Model Capability:** JinaBERT v2 supports **8192 tokens** natively
**Current Config:** 1024 tokens (8x underutilization)

From the [official HuggingFace model card](https://huggingface.co/jinaai/jina-embeddings-v2-small-en):
> "jina-embeddings-v2-small-en is an English, monolingual embedding model supporting **8192 sequence length**. It is based on a BERT architecture (JinaBERT) that supports the symmetric bidirectional variant of ALiBi to allow longer sequence length."

**Key Technical Detail:**
- Trained at 512 tokens
- Extrapolates to 8192 via ALiBi (Attention with Linear Biases)
- ALiBi enables length generalization without position embedding retraining

**Sources:**
- Model capability: [JinaBERT v2 Model Card](https://huggingface.co/jinaai/jina-embeddings-v2-small-en)
- Model architecture: [Jina Embeddings 2 Paper (arXiv:2310.19923)](https://arxiv.org/abs/2310.19923)
- ALiBi mechanism: [Train Short, Test Long (arXiv:2108.12409)](https://arxiv.org/abs/2108.12409) - Original ALiBi paper

---

## ALiBi Memory Calculation

The limiting factor for sequence length is **ALiBi bias memory**, which scales quadratically with sequence length.

### Formula
```
ALiBi memory = num_heads × seq_len² × sizeof(f32)
             = 8 × seq_len² × 4 bytes
```

### Memory Requirements by Sequence Length

| Sequence Length | ALiBi Memory | Total Estimated* | WASM Limit | Feasible? |
|-----------------|--------------|------------------|------------|-----------|
| 1024 (current)  | ~32 MB       | ~400 MB          | 512 MB     | ✅ Yes    |
| 2048            | ~128 MB      | ~650 MB          | 4 GB       | ✅ Yes    |
| 4096            | ~512 MB      | ~1.2 GB          | 4 GB       | ✅ Yes    |
| 8192 (max)      | ~2 GB        | ~2.8 GB          | 4 GB       | ⚠️  Tight |

\* *Total estimated includes: model weights (~250MB), ALiBi bias, activations (~100-200MB), working memory*

### Detailed Calculations

```python
# 1024 tokens
8 heads × 1024² = 8,388,608 elements × 4 bytes = 33,554,432 bytes ≈ 32 MB

# 2048 tokens
8 heads × 2048² = 33,554,432 elements × 4 bytes = 134,217,728 bytes ≈ 128 MB

# 4096 tokens
8 heads × 4096² = 134,217,728 elements × 4 bytes = 536,870,912 bytes ≈ 512 MB

# 8192 tokens
8 heads × 8192² = 536,870,912 elements × 4 bytes = 2,147,483,648 bytes ≈ 2 GB
```

---

## Recommended Configurations

### Option 1: Balanced (Recommended for Most Use Cases)
```rust
impl Default for JinaBertConfig {
    fn default() -> Self {
        // 2048 tokens: 4x improvement, ~650MB total memory
        Self {
            model_id: "jinaai/jina-embeddings-v2-small-en".to_string(),
            normalize_embeddings: true,
            hidden_size: 512,
            num_hidden_layers: 4,
            num_attention_heads: 8,
            intermediate_size: 2048,
            max_position_embeddings: 2048,  // ✅ 2x current, safe margin
        }
    }
}
```

**Pros:**
- 2x the context of current config
- Comfortable memory headroom (~650MB / 4GB)
- Handles most document chunks well
- ALiBi bias only ~128MB

**Use Cases:**
- Standard document chunks (1-2 pages)
- Code snippets and functions
- Article paragraphs

### Option 2: Long Documents
```rust
impl Default for JinaBertConfig {
    fn default() -> Self {
        // 4096 tokens: 8x improvement, ~1.2GB total memory
        Self {
            model_id: "jinaai/jina-embeddings-v2-small-en".to_string(),
            normalize_embeddings: true,
            hidden_size: 512,
            num_hidden_layers: 4,
            num_attention_heads: 8,
            intermediate_size: 2048,
            max_position_embeddings: 4096,  // ✅ 4x current, good for long docs
        }
    }
}
```

**Pros:**
- 4x the context of current config
- Still has 3GB headroom
- Handles long documents
- ALiBi bias ~512MB

**Use Cases:**
- Long articles and papers
- Complete code files
- Book chapters

### Option 3: Maximum Length (Experimental)
```rust
impl Default for JinaBertConfig {
    fn default() -> Self {
        // 8192 tokens: Full model capability, ~2.8GB total memory
        Self {
            model_id: "jinaai/jina-embeddings-v2-small-en".to_string(),
            normalize_embeddings: true,
            hidden_size: 512,
            num_hidden_layers: 4,
            num_attention_heads: 8,
            intermediate_size: 2048,
            max_position_embeddings: 8192,  // ⚠️ Full capability, tight on memory
        }
    }
}
```

**Pros:**
- Full model capability
- No chunking needed for most documents
- Maximum context preservation

**Cons:**
- ~2.8GB memory usage (70% of 4GB limit)
- Less headroom for browser overhead
- Potential OOM on memory-constrained devices

**Use Cases:**
- Very long documents
- Entire codebase files
- Research papers

---

## Making Sequence Length Configurable

For a production semantic search application, consider making this user-configurable:

```rust
pub enum SequenceLengthPreset {
    Short,   // 1024 tokens - fast, low memory
    Medium,  // 2048 tokens - balanced (recommended)
    Long,    // 4096 tokens - long documents
    Maximum, // 8192 tokens - full capability
}

impl SequenceLengthPreset {
    pub fn max_positions(&self) -> usize {
        match self {
            Self::Short => 1024,
            Self::Medium => 2048,
            Self::Long => 4096,
            Self::Maximum => 8192,
        }
    }

    pub fn estimated_memory_mb(&self) -> usize {
        match self {
            Self::Short => 400,
            Self::Medium => 650,
            Self::Long => 1200,
            Self::Maximum => 2800,
        }
    }
}
```

This allows users to trade off context length vs. memory usage based on their needs.

---

## Memory Budget Breakdown

### With 2048 Token Configuration (Recommended)
```
Model weights (F32):              ~250 MB
ALiBi bias matrix:                ~128 MB
Intermediate activations:         ~100 MB
Token embeddings (working):       ~50 MB
Working memory/overhead:          ~150 MB
Browser overhead:                 ~200 MB
─────────────────────────────────────────
Total estimated:                  ~878 MB
Available (4GB WASM):             4096 MB
Headroom:                         ~3.2 GB (78%)
```

### With 4096 Token Configuration
```
Model weights (F32):              ~250 MB
ALiBi bias matrix:                ~512 MB
Intermediate activations:         ~200 MB
Token embeddings (working):       ~100 MB
Working memory/overhead:          ~200 MB
Browser overhead:                 ~200 MB
─────────────────────────────────────────
Total estimated:                  ~1.46 GB
Available (4GB WASM):             4096 MB
Headroom:                         ~2.6 GB (63%)
```

---

## Implementation Checklist

- [ ] Update `.cargo/config.toml` to increase WASM memory to 4GB
- [ ] Update `JinaBertConfig::default()` to use 2048 or 4096 `max_position_embeddings`
- [ ] Test memory usage with browser DevTools Performance monitor
- [ ] Verify model outputs are still correct with longer sequences
- [ ] Update UI to show current sequence length limit
- [ ] (Optional) Add sequence length preset selector in UI
- [ ] Update `CLAUDE.md` and documentation with new limits
- [ ] Test on lower-end devices to ensure compatibility

---

## References

1. [V8 Blog: Up to 4GB of memory in WebAssembly](https://v8.dev/blog/4gb-wasm-memory)
2. [JinaBERT v2 Model Card - HuggingFace](https://huggingface.co/jinaai/jina-embeddings-v2-small-en)
3. [Jina Embeddings 2 Paper (arXiv:2310.19923)](https://arxiv.org/abs/2310.19923)
4. [WebAssembly Memory64 Proposal](https://github.com/WebAssembly/memory64)
5. [ALiBi: Train Short, Test Long (arXiv:2108.12409)](https://arxiv.org/abs/2108.12409)
