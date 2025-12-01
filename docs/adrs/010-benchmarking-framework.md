# ADR 010: Two-Tier Evaluation Framework

**Status**: Superseded (restructured)
**Date**: 2025-11-29
**Updated**: 2025-11-30
**Context**: Need benchmarks that measure both algorithmic correctness AND real-world search quality

> **Note**: This ADR documents the original design. The implementation was restructured to:
> - Move quality evaluation to a dedicated `coppermind-eval` binary using the `elinor` crate
> - Keep only essential performance benchmarks (`search.rs`, `indexing.rs`, `throughput.rs`) for CI
> - Remove ablation, scalability, load_test, memory, quality, and tier2_quality benchmarks
>
> See `docs/evaluation-guide.md` for current usage.

---

## Context and Problem Statement

### Current State

Coppermind has existing Criterion-based benchmarks in `crates/coppermind-core/benches/`:

| Benchmark | Purpose | Coverage |
|-----------|---------|----------|
| `search.rs` | Search latency at various k values, modalities | Good |
| `scalability.rs` | O(log n) complexity validation, up to 100K docs | Limited scale |
| `throughput.rs` | QPS (sequential + concurrent), percentiles | Basic |
| `recall.rs` | HNSW recall@k vs brute-force | Vector only |
| `indexing.rs` | Insert/batch/rebuild/compaction | Good |

### Gaps for Publication-Quality Research

1. **No retrieval quality metrics** - No NDCG, MAP, MRR, Precision@k with ground truth relevance labels
2. **Recall only measures HNSW** - Does not evaluate hybrid search quality (RRF fusion effectiveness)
3. **No real-world workload benchmarks** - Synthetic data only, no realistic document/query distributions
4. **No load/stress testing** - `bench_sustained_load` runs 10K queries but not for fixed duration (e.g., 5 minutes)
5. **No memory profiling** - Only rough estimates (`size * dim * 4 * 3`), no actual RSS measurements
6. **No per-stage timing breakdown** - Cannot see where search time is spent (vector vs BM25 vs fusion)
7. **No ablation studies** - Cannot isolate contribution of vector vs keyword vs fusion components
8. **Scale limited to 100K** - Need desktop-class workloads up to 500K documents

### Research Requirements

For publication in top-tier venues, we need:

- **Standard IR metrics**: NDCG@k, MAP, MRR, Precision@k, Recall@k (the field's lingua franca)
- **Ground truth datasets**: Relevance judgments for meaningful quality evaluation
- **Statistical rigor**: Confidence intervals, significance testing, effect sizes
- **Ablation studies**: Isolate contribution of each component
- **Reproducibility**: Fixed seeds, documented methodology
- **Scalability analysis**: Performance characterization across orders of magnitude

---

## Decision Drivers

1. **Publication standards** - Must meet expectations of SIGIR, NeurIPS, ACL reviewers
2. **Actionable insights** - Benchmarks should guide optimization decisions
3. **CI integration** - Regression testing without external dependencies
4. **Cross-platform** - Benchmarks must work on macOS, Linux, and (where feasible) WASM
5. **Minimal dependencies** - Avoid heavy crates that bloat WASM bundle
6. **Leverage existing infrastructure** - Build on Criterion, existing benchmark patterns

---

## Decision

Implement a **two-tier benchmarking framework** that separates algorithmic correctness from semantic quality:

| Tier | Purpose | Data Source | When to Run | What It Measures |
|------|---------|-------------|-------------|------------------|
| **Tier 1: Synthetic** | Algorithm verification | Generated embeddings/queries | Every commit (CI) | HNSW/BM25/RRF mechanics |
| **Tier 2: Real** | Semantic quality | Natural Questions dataset | Weekly/manual | Actual search quality |

### Why Two Tiers?

**Problem with current synthetic-only approach:**
- Synthetic embeddings are random - "machine learning" and "deep learning" have no semantic relationship
- TF-IDF generated queries are artificially easy - created from target document's own terms
- Ground truth is circular - brute-force k-NN on random embeddings just tests HNSW algorithm
- Results don't indicate real-world quality improvements

**Solution:**
- **Tier 1** (fast, CI): Keep synthetic benchmarks for regression detection on algorithm mechanics
- **Tier 2** (slow, manual): Add real dataset evaluation for semantic quality measurement

This approach lets us answer both "did we break something?" (Tier 1) and "did we improve quality?" (Tier 2).

### Dataset Choice: Natural Questions

After evaluating MS MARCO, HotpotQA, CQADupStack, and others:

| Criteria | Natural Questions | Why It Fits |
|----------|-------------------|-------------|
| **Content** | Wikipedia articles | Matches wiki/docs use case |
| **Queries** | Real Google searches | Actual user information needs |
| **Ground truth** | Human annotations | Not circular/synthetic |
| **License** | CC BY-SA 3.0 | Compatible with MIT (commercial OK) |
| **Size** | Subset to ~3K docs, 500 queries | Fast enough for iteration |

**Rejected alternatives:**
- **MS MARCO**: Non-commercial license, incompatible with MIT project
- **HotpotQA**: Multi-hop questions don't match keyword search use case
- **CQADupStack**: Good for technical content, can add later if needed

```
crates/coppermind-core/
├── src/
│   └── evaluation/              # Evaluation framework
│       ├── mod.rs               # Public API
│       ├── metrics.rs           # NDCG, MAP, MRR, P@k, R@k, F1@k
│       ├── stats.rs             # Bootstrap CI, t-tests, Cohen's d
│       └── datasets/
│           ├── mod.rs           # EvalDataset trait
│           ├── synthetic.rs     # Synthetic query generation (Tier 1)
│           └── natural_questions.rs  # NQ loader (Tier 2)
├── benches/
│   ├── quality.rs               # Tier 1: Synthetic quality metrics
│   ├── ablation.rs              # Tier 1: Component ablation
│   ├── tier2_quality.rs         # Tier 2: Real semantic quality (NEW)
│   ├── search.rs                # Performance: latency
│   ├── scalability.rs           # Performance: scaling
│   ├── throughput.rs            # Performance: QPS
│   ├── load_test.rs             # Performance: sustained load
│   ├── memory.rs                # Performance: memory profiling
│   └── indexing.rs              # Performance: indexing

data/natural-questions/          # Tier 2 dataset (Git LFS)
├── corpus.jsonl                 # Wikipedia paragraphs
├── queries.jsonl                # Google search queries
├── qrels.tsv                    # Human relevance judgments
└── embeddings.safetensors       # Pre-computed JinaBERT embeddings

scripts/
└── prepare_nq.py                # Download and prepare NQ subset
```

---

## Pillar 1: Retrieval Quality Evaluation

### IR Metrics (`src/evaluation/metrics.rs`)

Implement standard Information Retrieval metrics:

```rust
/// Graded relevance judgment for a document
#[derive(Debug, Clone)]
pub struct RelevanceJudgment {
    pub chunk_id: ChunkId,
    pub relevance: u8,  // 0=not relevant, 1=somewhat relevant, 2=highly relevant
}

/// Evaluation metrics for a single query
#[derive(Debug, Clone)]
pub struct QueryMetrics {
    pub ndcg_at_k: BTreeMap<usize, f64>,      // k -> NDCG@k
    pub map: f64,                              // Mean Average Precision
    pub mrr: f64,                              // Mean Reciprocal Rank
    pub precision_at_k: BTreeMap<usize, f64>,
    pub recall_at_k: BTreeMap<usize, f64>,
    pub f1_at_k: BTreeMap<usize, f64>,
}

// Core computation functions
pub fn ndcg_at_k(results: &[(ChunkId, f32)], judgments: &[RelevanceJudgment], k: usize) -> f64;
pub fn average_precision(results: &[(ChunkId, f32)], judgments: &[RelevanceJudgment]) -> f64;
pub fn reciprocal_rank(results: &[(ChunkId, f32)], judgments: &[RelevanceJudgment]) -> f64;
pub fn precision_at_k(results: &[(ChunkId, f32)], judgments: &[RelevanceJudgment], k: usize) -> f64;
pub fn recall_at_k(results: &[(ChunkId, f32)], judgments: &[RelevanceJudgment], k: usize) -> f64;
```

**Metric Definitions:**

| Metric | Formula | Purpose |
|--------|---------|---------|
| **NDCG@k** | `DCG@k / IDCG@k` where `DCG = Σ (2^rel - 1) / log₂(rank + 1)` | Graded relevance, position-aware |
| **MAP** | `(1/\|rel\|) * Σ P(k) * rel(k)` | Classic IR, holistic precision-recall |
| **MRR** | `1 / rank` of first relevant result | First good result quality |
| **P@k** | `\|relevant ∩ top-k\| / k` | Set-based precision |
| **R@k** | `\|relevant ∩ top-k\| / \|relevant\|` | Set-based recall |
| **F1@k** | `2 * P@k * R@k / (P@k + R@k)` | Precision-recall harmonic mean |

### Statistical Utilities (`src/evaluation/stats.rs`)

```rust
/// Bootstrap 95% confidence interval
/// Returns (mean, lower_bound, upper_bound)
pub fn bootstrap_ci(values: &[f64], n_bootstrap: usize, seed: u64) -> (f64, f64, f64);

/// Paired t-test for comparing two systems on same queries
/// Returns (t_statistic, p_value)
pub fn paired_ttest(system_a: &[f64], system_b: &[f64]) -> (f64, f64);

/// Cohen's d effect size
pub fn cohens_d(group_a: &[f64], group_b: &[f64]) -> f64;
```

**Statistical Methodology:**

- **Bootstrap CI**: 1000 resamples, 95% confidence interval (α = 0.05)
- **Paired t-test**: For comparing systems on same query set (same variance)
- **Bonferroni correction**: When comparing k systems, use α' = α/k threshold
- **Cohen's d**: Effect size interpretation (0.2=small, 0.5=medium, 0.8=large)

### Ground Truth Datasets (`src/evaluation/datasets/`)

#### Synthetic Queries (Primary, CI-friendly)

Generate queries where ground truth is known by construction:

```rust
pub struct SyntheticQueryGenerator {
    corpus: Vec<ChunkRecord>,
    rng: StdRng,
}

impl SyntheticQueryGenerator {
    /// Generate query targeting specific chunk as highly relevant
    pub fn generate_query(&mut self, target_id: ChunkId) -> SyntheticQuery;

    /// Generate batch of diverse queries covering corpus
    pub fn generate_batch(&mut self, n: usize) -> Vec<SyntheticQuery>;
}

pub struct SyntheticQuery {
    pub text: String,
    pub judgments: Vec<RelevanceJudgment>,
    pub query_type: QueryType,  // Semantic, Keyword, Mixed
}
```

**Generation Strategy:**
1. Select target chunk from corpus
2. Extract distinctive terms (TF-IDF or BM25 term weights)
3. Optionally paraphrase for semantic queries
4. Mark target as highly relevant (2), similar chunks as somewhat relevant (1)

**Advantages:**
- Fully automated, deterministic (seeded RNG)
- Runs in CI without external data
- Tests both keyword and semantic retrieval paths

#### BEIR Subset (Optional, for publication)

For publication-quality comparison to established baselines:

```rust
pub struct BeirDataset {
    pub name: String,           // "scifact", "nfcorpus", etc.
    pub passages: Vec<Passage>,
    pub queries: Vec<BeirQuery>,
}

pub struct BeirQuery {
    pub qid: String,
    pub text: String,
    pub relevant_passages: Vec<(String, u8)>,  // (passage_id, relevance)
}

impl BeirDataset {
    pub async fn load(name: &str, data_dir: &Path) -> Result<Self, DatasetError>;
}
```

**Recommended Datasets:**

| Dataset | Docs | Queries | Domain | Size |
|---------|------|---------|--------|------|
| SciFact | 5,183 | 300 | Scientific claims | ~15MB |
| NFCorpus | 3,633 | 323 | Medical/nutrition | ~10MB |
| FiQA | 57,638 | 648 | Financial QA | ~150MB |

**Why BEIR over MS MARCO:**
- MS MARCO is 8.8M passages (too large for Coppermind's use case)
- BEIR subsets are appropriately sized (1K-50K docs)
- Zero-shot evaluation matches Coppermind's use case (no training data)
- MIT licensed, suitable for open-source

---

## Pillar 2: Performance Characterization

### Load Testing (`benches/load_test.rs`)

#### Time-Bounded Tests

Run at maximum sustainable throughput for fixed duration:

```rust
fn bench_sustained_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("load/sustained");

    for duration_secs in [60, 120, 300] {
        group.bench_function(format!("{}s", duration_secs), |b| {
            b.iter_custom(|_iters| {
                let start = Instant::now();
                let deadline = start + Duration::from_secs(duration_secs);
                let mut query_count = 0u64;
                let mut latencies = Vec::new();

                while Instant::now() < deadline {
                    let query_start = Instant::now();
                    let _ = engine.search(&query, k);
                    latencies.push(query_start.elapsed());
                    query_count += 1;
                }

                // Report: QPS, p50, p95, p99
                let qps = query_count as f64 / duration_secs as f64;
                report_metrics(qps, &latencies);

                start.elapsed()
            });
        });
    }
}
```

**Metrics Captured:**
- **QPS**: Queries per second sustained over duration
- **Latency percentiles**: p50, p95, p99 under load
- **Degradation**: QPS trend over time (first minute vs last minute)

#### Request-Bounded Tests

Complete N requests as fast as possible:

```rust
fn bench_bulk_queries(c: &mut Criterion) {
    for query_count in [10_000, 100_000, 1_000_000] {
        group.bench_function(format!("{}queries", query_count), |b| {
            b.iter(|| {
                for query in &queries[..query_count] {
                    let _ = engine.search(query, k);
                }
            });
        });
    }
}
```

**Metrics Captured:**
- **Total time**: Wall-clock time to complete all queries
- **Effective QPS**: query_count / total_time

### Scalability Extension (`benches/scalability.rs`)

Extend corpus sizes to 500K with adaptive skip:

```rust
const CORPUS_SIZES: &[usize] = &[
    100, 500, 1_000, 5_000, 10_000, 25_000, 50_000,
    100_000, 200_000, 500_000,  // NEW: desktop-class
];

fn should_skip_size(size: usize) -> bool {
    // Skip if:
    // 1. Memory constrained (check available RAM)
    // 2. Build time exceeds threshold (use timing from previous run)
    // 3. Feature flag not set for large-scale benchmarks
    if size > 100_000 && !cfg!(feature = "large_scale_bench") {
        return true;
    }
    false
}
```

### Memory Profiling (`benches/memory.rs`)

#### Cross-Platform Estimation (CI-friendly)

```rust
fn bench_memory_estimate(c: &mut Criterion) {
    for size in CORPUS_SIZES {
        let vector_bytes = size * EMBEDDING_DIM * 4;  // f32 per dimension
        let hnsw_overhead = vector_bytes * 2;         // ~2x for graph structure
        let bm25_overhead = size * 200;               // ~200 bytes per doc average
        let total_estimate = vector_bytes + hnsw_overhead + bm25_overhead;

        group.bench_function(format!("estimate_{}", size), |b| {
            b.iter(|| total_estimate);
        });
    }
}
```

#### Actual RSS Measurement (Platform-specific)

```rust
#[cfg(target_os = "macos")]
fn get_rss_bytes() -> usize {
    use mach::mach_types::task_t;
    use mach::task_info::{task_basic_info, TASK_BASIC_INFO};

    // ... mach API calls
}

#[cfg(target_os = "linux")]
fn get_rss_bytes() -> usize {
    // Parse /proc/self/statm
    let statm = std::fs::read_to_string("/proc/self/statm").unwrap();
    let rss_pages: usize = statm.split_whitespace().nth(1).unwrap().parse().unwrap();
    rss_pages * 4096  // Page size
}

fn bench_memory_actual(c: &mut Criterion) {
    for size in CORPUS_SIZES {
        let baseline_rss = get_rss_bytes();

        // Build index
        let engine = build_engine(size);

        let after_rss = get_rss_bytes();
        let index_memory = after_rss - baseline_rss;

        group.bench_function(format!("actual_{}", size), |b| {
            b.iter(|| index_memory);
        });
    }
}
```

**Metrics Captured:**
- **Base memory**: Before index construction
- **Index memory**: After construction minus base
- **Per-1K increment**: Memory growth rate
- **Peak during rebuild**: Maximum RSS during cold start

### Per-Stage Timing (`benches/search_stages.rs`)

Leverage existing `SearchTimings` from ADR-009:

```rust
fn bench_stage_breakdown(c: &mut Criterion) {
    for corpus_size in [1_000, 10_000, 100_000] {
        let engine = build_engine(corpus_size);

        group.bench_function(format!("stages_{}", corpus_size), |b| {
            b.iter_custom(|iters| {
                let mut total_timings = SearchTimings::default();

                for _ in 0..iters {
                    let (_, timings) = engine.search_with_timings(&query, &text, k).unwrap();
                    total_timings.accumulate(&timings);
                }

                // Report breakdown as percentages
                let total = total_timings.total_ms;
                println!("Vector: {:.1}%, BM25: {:.1}%, Fusion: {:.1}%, Hydrate: {:.1}%",
                    total_timings.vector_ms / total * 100.0,
                    total_timings.keyword_ms / total * 100.0,
                    total_timings.fusion_ms / total * 100.0,
                    total_timings.hydration_ms / total * 100.0,
                );

                Duration::from_secs_f64(total / 1000.0)
            });
        });
    }
}
```

**Expected Findings:**
- At small corpus: Hydration dominates (store lookups)
- At large corpus: BM25 dominates (O(n) term scanning)
- Vector search grows slowly (O(log n) HNSW)

---

## Pillar 3: Ablation Studies

### Component Contribution (`benches/ablation/component.rs`)

Compare search modalities:

```rust
enum SearchMode {
    VectorOnly,
    KeywordOnly,
    Hybrid,
}

fn bench_component_contribution(c: &mut Criterion) {
    let queries = SyntheticQueryGenerator::new(&corpus).generate_batch(100);

    for mode in [SearchMode::VectorOnly, SearchMode::KeywordOnly, SearchMode::Hybrid] {
        group.bench_function(format!("{:?}", mode), |b| {
            b.iter(|| {
                let mut ndcg_scores = Vec::new();

                for query in &queries {
                    let results = match mode {
                        SearchMode::VectorOnly => engine.search_vector_only(&query.embedding, k),
                        SearchMode::KeywordOnly => engine.search_keyword_only(&query.text, k),
                        SearchMode::Hybrid => engine.search(&query.embedding, &query.text, k),
                    };
                    ndcg_scores.push(ndcg_at_k(&results, &query.judgments, 10));
                }

                let (mean, lower, upper) = bootstrap_ci(&ndcg_scores, 1000, 42);
                println!("{:?}: NDCG@10 = {:.4} [{:.4}, {:.4}]", mode, mean, lower, upper);
            });
        });
    }
}
```

**Required API Extensions:**

```rust
impl<S: DocumentStore> HybridSearchEngine<S> {
    /// BM25-only search (keyword baseline)
    pub async fn search_keyword_only(&mut self, query_text: &str, k: usize)
        -> Result<Vec<SearchResult>, SearchError>;

    /// Vector-only search (semantic baseline)
    pub async fn search_vector_only(&mut self, query_embedding: &[f32], k: usize)
        -> Result<Vec<SearchResult>, SearchError>;
}
```

### RRF Parameter Sensitivity (`benches/ablation/rrf.rs`)

```rust
const RRF_K_VALUES: &[usize] = &[10, 30, 60, 100, 200];

fn bench_rrf_sensitivity(c: &mut Criterion) {
    for rrf_k in RRF_K_VALUES {
        group.bench_function(format!("rrf_k={}", rrf_k), |b| {
            b.iter(|| {
                let mut ndcg_scores = Vec::new();

                for query in &queries {
                    let results = engine.search_with_rrf_k(&query.embedding, &query.text, k, *rrf_k);
                    ndcg_scores.push(ndcg_at_k(&results, &query.judgments, 10));
                }

                let (mean, _, _) = bootstrap_ci(&ndcg_scores, 1000, 42);
                mean
            });
        });
    }
}
```

**Required API Extension:**

```rust
impl<S: DocumentStore> HybridSearchEngine<S> {
    /// Hybrid search with custom RRF k parameter
    pub async fn search_with_rrf_k(
        &mut self,
        query_embedding: &[f32],
        query_text: &str,
        k: usize,
        rrf_k: usize,  // Custom RRF parameter instead of default 60
    ) -> Result<Vec<SearchResult>, SearchError>;
}
```

### HNSW Parameter Grid (`benches/ablation/hnsw.rs`)

```rust
const M_VALUES: &[usize] = &[8, 12, 16, 24, 32];
const EF_CONSTRUCTION: &[usize] = &[100, 200, 400];
const EF_SEARCH_MULT: &[usize] = &[1, 2, 4, 8];

fn bench_hnsw_pareto(c: &mut Criterion) {
    for m in M_VALUES {
        for ef_c in EF_CONSTRUCTION {
            for ef_mult in EF_SEARCH_MULT {
                let config = HnswConfig { m: *m, ef_construction: *ef_c, ef_search: k * ef_mult };
                let engine = build_engine_with_config(corpus_size, config);

                group.bench_function(format!("M{}_efc{}_efs{}", m, ef_c, ef_mult), |b| {
                    b.iter(|| {
                        // Measure recall and latency
                        let recall = measure_recall(&engine, &queries, k);
                        let latency = measure_p95_latency(&engine, &queries, k);
                        (recall, latency)
                    });
                });
            }
        }
    }
}
```

**Output:** Pareto frontier of recall vs latency vs memory

---

## Implementation Plan

### Phase 1: Evaluation Module (Priority: High)

1. Create `src/evaluation/mod.rs` - Module structure
2. Create `src/evaluation/metrics.rs` - IR metrics (NDCG, MAP, MRR, P@k, R@k)
3. Create `src/evaluation/stats.rs` - Statistical utilities
4. Create `src/evaluation/datasets/synthetic.rs` - Query generation
5. Add unit tests with hand-computed examples

**Estimated effort:** ~500 lines of code

### Phase 2: Performance Benchmarks (Priority: High)

1. Create `benches/load_test.rs` - Time/request bounded tests
2. Create `benches/memory.rs` - RSS profiling
3. Create `benches/search_stages.rs` - Stage breakdown
4. Modify `benches/scalability.rs` - Extend to 500K

**Estimated effort:** ~400 lines of code

### Phase 3: API Extensions (Priority: Medium)

1. Add `search_keyword_only()` to HybridSearchEngine
2. Add `search_vector_only()` to HybridSearchEngine
3. Add `search_with_rrf_k()` to HybridSearchEngine
4. Expose HNSW configuration parameters

**Estimated effort:** ~200 lines of code

### Phase 4: Ablation Benchmarks (Priority: Medium)

1. Create `benches/ablation/component.rs`
2. Create `benches/ablation/rrf.rs`
3. Create `benches/ablation/hnsw.rs`

**Estimated effort:** ~300 lines of code

### Phase 5: Retrieval Quality Benchmark (Priority: Medium)

1. Create `benches/retrieval_quality.rs`
2. Integrate with evaluation module
3. Generate baseline results

**Estimated effort:** ~200 lines of code

### Phase 6: BEIR Integration (Priority: Low, Optional)

1. Create `scripts/download-beir-subset.sh`
2. Create `src/evaluation/datasets/beir.rs`
3. Integration tests with real data

**Estimated effort:** ~300 lines of code

---

## Files to Modify

| File | Action | Purpose |
|------|--------|---------|
| `crates/coppermind-core/src/lib.rs` | Modify | Add `pub mod evaluation;` |
| `crates/coppermind-core/src/evaluation/mod.rs` | Create | Module structure |
| `crates/coppermind-core/src/evaluation/metrics.rs` | Create | IR metrics |
| `crates/coppermind-core/src/evaluation/stats.rs` | Create | Statistical utilities |
| `crates/coppermind-core/src/evaluation/datasets/mod.rs` | Create | Dataset module |
| `crates/coppermind-core/src/evaluation/datasets/synthetic.rs` | Create | Query generation |
| `crates/coppermind-core/src/search/engine/mod.rs` | Modify | Add ablation methods |
| `crates/coppermind-core/src/search/fusion.rs` | Modify | Expose RRF k parameter |
| `crates/coppermind-core/benches/scalability.rs` | Modify | Extend to 500K |
| `crates/coppermind-core/benches/load_test.rs` | Create | Load testing |
| `crates/coppermind-core/benches/memory.rs` | Create | Memory profiling |
| `crates/coppermind-core/benches/search_stages.rs` | Create | Stage breakdown |
| `crates/coppermind-core/benches/retrieval_quality.rs` | Create | IR metrics eval |
| `crates/coppermind-core/benches/ablation/mod.rs` | Create | Ablation module |
| `crates/coppermind-core/benches/ablation/component.rs` | Create | Component ablation |
| `crates/coppermind-core/benches/ablation/rrf.rs` | Create | RRF sensitivity |
| `crates/coppermind-core/benches/ablation/hnsw.rs` | Create | HNSW parameter grid |
| `crates/coppermind-core/Cargo.toml` | Modify | Add dev-dependencies |

---

## Expected Outputs

### Tables for Publication

**Table 1: Retrieval Quality**
| System | NDCG@10 | MAP | MRR | P@10 | R@100 |
|--------|---------|-----|-----|------|-------|
| BM25-only | 0.XXX ± 0.YYY | ... | ... | ... | ... |
| Vector-only | 0.XXX ± 0.YYY | ... | ... | ... | ... |
| Hybrid (k=60) | **0.XXX ± 0.YYY** | ... | ... | ... | ... |

**Table 2: Scalability**
| Corpus | HNSW (ms) | BM25 (ms) | Hybrid (ms) | Memory (MB) |
|--------|-----------|-----------|-------------|-------------|
| 1K | ... | ... | ... | ... |
| 10K | ... | ... | ... | ... |
| 100K | ... | ... | ... | ... |
| 500K | ... | ... | ... | ... |

**Table 3: Load Test Results**
| Duration | QPS | p50 (ms) | p95 (ms) | p99 (ms) |
|----------|-----|----------|----------|----------|
| 60s | ... | ... | ... | ... |
| 300s | ... | ... | ... | ... |

### Figures

1. **Latency vs Corpus Size** - Log-log plot showing O(log n) for HNSW
2. **Memory Scaling** - Measured RSS vs corpus size
3. **Stage Breakdown** - Stacked bar chart: % vector/BM25/fusion/hydration
4. **RRF Sensitivity** - NDCG@10 vs RRF k parameter
5. **HNSW Pareto Frontier** - Recall vs latency scatter plot

---

## Consequences

### Positive

1. **Publication-ready evaluation** - Meets standards for SIGIR, NeurIPS, ACL
2. **Actionable insights** - Data-driven optimization decisions
3. **Regression detection** - CI catches quality degradation
4. **Reproducibility** - Fixed seeds, documented methodology
5. **Community contribution** - Benchmark methodology applicable to other hybrid search systems

### Negative

1. **Increased benchmark runtime** - Full suite takes longer to run
   - **Mitigation:** Feature flags for CI vs full benchmarks
2. **API surface expansion** - New methods on HybridSearchEngine
   - **Mitigation:** Keep ablation methods `#[doc(hidden)]` or in benchmark-only module
3. **Optional BEIR dependency** - External data download
   - **Mitigation:** Download script, not required for CI

### Risks

1. **Memory constraints at 500K scale** - May OOM on CI runners
   - **Mitigation:** Adaptive skip based on available memory
2. **Criterion time limits** - Long benchmarks may timeout
   - **Mitigation:** Use `iter_custom` with explicit duration control

---

## Success Criteria

After implementation, we can answer:

**Quality:**
- "What is NDCG@10 for hybrid search?" → `0.XXX ± 0.YYY` (95% CI)
- "Does hybrid outperform BM25-only?" → `t=X.XX, p<0.05, Cohen's d=0.X`
- "What query types benefit most from hybrid?" → "Semantic: +15% NDCG"

**Performance:**
- "Max sustainable QPS at 100K?" → `XXX QPS for 5 minutes`
- "Memory at 500K?" → `XXX MB RSS`
- "Where does search time go?" → "40% vector, 30% BM25, 5% fusion, 25% hydration"

**Ablation:**
- "Is RRF k=60 optimal?" → "Yes/No, k=XX gives +Y% NDCG"
- "HNSW recommendations?" → "M=16, ef_search=2x is Pareto-optimal"

---

## References

**IR Metrics:**
- Järvelin & Kekäläinen (2002). "Cumulated gain-based evaluation of IR techniques" - NDCG
- Voorhees & Harman (2005). "TREC: Experiment and Evaluation in IR" - MAP

**Statistical Methods:**
- Efron & Tibshirani (1993). "An Introduction to the Bootstrap"
- Smucker et al. (2007). "A comparison of statistical significance tests for IR evaluation"

**Benchmarks:**
- Thakur et al. (2021). "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation"
- Nguyen et al. (2016). "MS MARCO: A Human Generated MAchine Reading COmprehension Dataset"

**Hybrid Search:**
- Cormack et al. (2009). "Reciprocal Rank Fusion outperforms Condorcet and individual methods"
