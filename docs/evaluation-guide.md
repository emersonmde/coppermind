# Evaluation Guide

This guide explains the performance benchmarking and quality evaluation framework used in Coppermind. It covers both performance metrics (latency, throughput) and information retrieval quality metrics (NDCG, MAP, MRR). It's written for software engineers who want to understand how search systems are evaluated, even without a background in information retrieval or applied science.

## Table of Contents

1. [Why Benchmark?](#why-benchmark)
2. [Two-Tier Evaluation Framework](#two-tier-evaluation-framework)
   - [Tier 1: Performance Benchmarks (Criterion)](#tier-1-performance-benchmarks-criterion)
   - [Tier 2: Quality Evaluation (coppermind-eval)](#tier-2-quality-evaluation-coppermind-eval)
   - [When to Use Each Tier](#when-to-use-each-tier)
3. [Experimental Data and Methodology](#experimental-data-and-methodology)
   - [Why Synthetic Data?](#why-synthetic-data)
   - [Corpus Generation](#corpus-generation)
   - [Query Generation](#query-generation)
   - [Ground Truth Computation](#ground-truth-computation)
   - [Experimental Constants](#experimental-constants)
   - [Limitations and Caveats](#limitations-and-caveats)
4. [Performance Benchmarks](#performance-benchmarks)
   - [Search Latency](#search-latency)
   - [Indexing Performance](#indexing-performance)
   - [Throughput](#throughput)
5. [Retrieval Quality Metrics](#retrieval-quality-metrics)
   - [The Ground Truth Problem](#the-ground-truth-problem)
   - [Precision and Recall](#precision-and-recall)
   - [NDCG: Normalized Discounted Cumulative Gain](#ndcg-normalized-discounted-cumulative-gain)
   - [Mean Average Precision (MAP)](#mean-average-precision-map)
   - [Mean Reciprocal Rank (MRR)](#mean-reciprocal-rank-mrr)
   - [F1 Score](#f1-score)
6. [Statistical Rigor](#statistical-rigor)
   - [Why Statistics Matter](#why-statistics-matter)
   - [Bootstrap Confidence Intervals](#bootstrap-confidence-intervals)
   - [Paired t-Tests](#paired-t-tests)
   - [Effect Size (Cohen's d)](#effect-size-cohens-d)
7. [Interpreting Results](#interpreting-results)
8. [Running the Benchmarks](#running-the-benchmarks)

---

## Why Benchmark?

Benchmarking serves two distinct purposes that are often conflated: measuring **speed** and measuring **quality**. A search system that returns results in 1 millisecond is useless if those results are wrong. Conversely, a system with perfect accuracy is useless if users abandon it before results appear.

Coppermind's benchmarking framework addresses both concerns. Performance benchmarks tell you how fast the system operates under various conditions. Quality benchmarks tell you whether the system returns the right documents. Statistical analysis tells you whether observed differences are meaningful or just noise.

This separation matters because optimizations often involve tradeoffs. Making search faster might reduce accuracy. Adding a new ranking signal might improve quality but slow things down. Good benchmarks help you make informed decisions about these tradeoffs rather than guessing.

---

## Two-Tier Evaluation Framework

Coppermind uses a two-tier evaluation approach that separates **performance** from **quality**:

| Tier | Purpose | Tool | When to Run | What It Measures |
|------|---------|------|-------------|------------------|
| **Tier 1: Performance** | Latency, throughput, scalability | Criterion benchmarks | Every commit (CI) | Speed and resource usage |
| **Tier 2: Quality** | Search accuracy and ranking | `coppermind-eval` binary | Manual evaluation | NDCG, MAP, MRR with real embeddings |

### Tier 1: Performance Benchmarks (Criterion)

**Location**: `crates/coppermind-core/benches/`

Tier 1 uses Criterion for micro-benchmarking performance. These benchmarks run fast and are suitable for CI regression detection.

**Benchmark files:**

| File | Purpose | Key Measurements |
|------|---------|------------------|
| `search.rs` | Search latency | HNSW/BM25/hybrid latency by k and corpus size |
| `indexing.rs` | Index operations | Single/batch insert, rebuild, compaction |
| `throughput.rs` | Sustained performance | QPS, latency percentiles, concurrent access |

**What Tier 1 benchmarks measure:**
- Search latency at various k values and corpus sizes
- Index build and rebuild time
- Queries per second (throughput)
- Latency percentiles under load

**What Tier 1 benchmarks do NOT measure:**
- Search result quality (embeddings are synthetic)
- Whether results are semantically relevant
- Real-world ranking effectiveness

### Tier 2: Quality Evaluation (coppermind-eval)

**Location**: `crates/coppermind-eval/`

Tier 2 uses a dedicated evaluation binary with the [elinor](https://crates.io/crates/elinor) crate for scientific IR evaluation. This measures actual search quality using real JinaBERT embeddings.

**What Tier 2 evaluation measures:**
- NDCG@k (Normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)
- MRR (Mean Reciprocal Rank)
- Precision@k and Recall@k
- Statistical comparisons between search modes (vector vs keyword vs hybrid)

**Features:**
- Embedded evaluation dataset with human relevance judgments
- Embedding cache for fast re-runs
- Paired t-tests for statistical significance
- JSON output for analysis

### When to Use Each Tier

| Scenario | Tier 1 (Criterion) | Tier 2 (coppermind-eval) |
|----------|-------------------|--------------------------|
| PR check / CI | ✅ | ❌ |
| Performance regression | ✅ | ❌ |
| Algorithm refactoring | ✅ | ✅ |
| Evaluating new ranking feature | ❌ | ✅ |
| Answering "does hybrid help?" | ❌ | ✅ |
| Tuning fusion parameters | ❌ | ✅ |

**Key insight**: A change that improves Tier 1 metrics (faster) may not improve Tier 2 metrics (better quality) and vice versa. Both matter, but they measure different things.

---

## Experimental Data and Methodology

> **Note**: This section describes **Tier 1 synthetic data** methodology. For Tier 2, see the Natural Questions dataset documentation.

Benchmarks are only as good as the data they run against. This section explains how Coppermind generates test corpora, queries, and ground truth - the foundational elements that make quality measurements meaningful.

### Why Synthetic Data?

Ideally, benchmarks would use real user queries against real document collections with human-labeled relevance judgments. This is how academic IR benchmarks like TREC work - hundreds of human annotators read documents and judge their relevance to queries.

This approach doesn't scale for development benchmarking. Human annotation is expensive, slow, and creates fixed test sets that can't adapt to new features. Instead, Coppermind uses synthetic data generation with carefully designed properties:

**Reproducibility**: All randomness uses seeded pseudo-random number generators. Running benchmarks with the same code produces identical results, enabling reliable regression detection.

**Controllability**: Synthetic generation lets us create corpora of any size with known properties. We can test 100 documents for quick iteration, then 100,000 for scalability analysis.

**Known ground truth**: By construction, we know exactly which documents should match which queries. This eliminates annotation noise and disagreement.

The tradeoff is realism. Synthetic queries don't capture the full complexity of real user information needs. The benchmarks measure how well the system handles the synthetic workload, which approximates but doesn't guarantee real-world performance.

### Corpus Generation

#### Document Embeddings

Each synthetic document has a 512-dimensional embedding vector generated deterministically from its ID:

```rust
fn seeded_embedding(seed: u64) -> Vec<f32> {
    // Hash-based generation: each dimension derived from hash(seed, dimension_index)
    let raw: Vec<f32> = (0..512)
        .map(|i| {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            i.hash(&mut hasher);
            ((hasher.finish() as f32 / u64::MAX as f32) * 2.0) - 1.0
        })
        .collect();

    // L2-normalize to match real JinaBERT output
    let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    raw.into_iter().map(|x| x / norm).collect()
}
```

Key properties:
- **512 dimensions**: Matches JinaBERT's output, ensuring benchmarks reflect production embedding costs
- **L2-normalized**: Real embeddings from JinaBERT are always unit vectors; synthetic ones match this property
- **Deterministic**: Document 42 always gets the same embedding across runs
- **Pseudo-random distribution**: Hash-based generation creates embeddings spread throughout the vector space, avoiding pathological clustering

#### Document Text

Each document has synthetic text content that varies by topic:

```rust
fn sample_text(id: u64) -> String {
    let topics = [
        "machine learning algorithms neural network deep learning",
        "semantic search information retrieval embeddings",
        "natural language processing text understanding NLP",
        "vector embeddings similarity metrics distance",
        "transformer attention mechanisms BERT GPT",
        "approximate nearest neighbor HNSW indexing",
        "text classification sentiment analysis NER",
        "knowledge graphs entity recognition relations",
    ];
    let topic = topics[(id % 8) as usize];

    format!("Document {id} discusses {topic}. This content covers ...")
}
```

Key properties:
- **8 rotating topics**: Creates realistic term distribution for BM25 testing
- **~2000 characters**: Matches production chunk size (TARGET_CHUNK_CHARS = 2048)
- **Domain-relevant vocabulary**: ML/search/NLP terms reflect Coppermind's target use case
- **Deterministic**: Document 42 always discusses the same topic

#### Corpus Size Constants

Different benchmarks use different corpus sizes depending on what they measure:

| Benchmark Type | Sizes Used | Rationale |
|----------------|------------|-----------|
| Quality metrics | 1K, 5K, 10K | Large enough for meaningful statistics, small enough for tractable ground truth computation |
| Scalability | 100 to 100K+ | Wide range to observe algorithmic complexity |
| Indexing | 100 to 1K | Focused on per-document costs |
| Ablation | 2K-5K | Balanced between speed and representativeness |

### Query Generation

#### Vector Search Queries

For vector search evaluation, queries are embeddings generated with a separate seed range to avoid overlap with corpus documents:

```rust
const QUERY_SEED_BASE: u64 = 1_000_000;

// Document seeds: 0, 1, 2, ..., N-1
// Query seeds: 1,000,000, 1,000,001, ...
let query_embedding = seeded_embedding(QUERY_SEED_BASE + query_index);
```

This separation ensures queries are genuinely "unseen" - they don't accidentally match a corpus document exactly. The hash-based generation still produces embeddings in the same space, so similarity search works naturally.

#### Keyword Search Queries

For BM25 evaluation, the `SyntheticQueryGenerator` creates queries with known relevant documents:

**Algorithm:**
1. Select a target document from the corpus
2. Tokenize the document text (lowercase, 2+ character tokens)
3. Compute TF-IDF scores for each term:
   - Term Frequency (TF): How often the term appears in this document
   - Inverse Document Frequency (IDF): `log(corpus_size / documents_containing_term)`
   - Score = TF × IDF (high score = distinctive to this document)
4. Select top-scoring terms as query words
5. Mark the target document as relevant ground truth

**Query types generated:**
- **Keyword**: 2-5 top terms joined with spaces
- **Short**: 1-3 terms (tests minimal queries)
- **Long**: 5-8 terms (tests verbose queries)

**Example:**
For a document about "machine learning neural networks", TF-IDF might identify "neural" and "networks" as distinctive (common in this doc, rare elsewhere). The generated query "neural networks" has known ground truth: this document should rank highly.

### Ground Truth Computation

#### Vector Search Ground Truth

For vector search, ground truth is computed by **brute-force k-nearest neighbors**:

```rust
fn brute_force_knn(query: &[f32], corpus: &[(ChunkId, Vec<f32>)], k: usize) -> Vec<ChunkId> {
    // Compute cosine similarity to every document
    let mut scored: Vec<_> = corpus
        .iter()
        .map(|(id, emb)| (*id, cosine_similarity(query, emb)))
        .collect();

    // Sort by similarity descending, return top k
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}
```

This is O(n) per query - slow, but guaranteed correct. It computes the exact answer against which HNSW's approximate results are compared. The recall metric directly measures what fraction of these true nearest neighbors HNSW finds.

#### Relevance Grading

Quality metrics like NDCG require graded relevance (not just relevant/not-relevant). The benchmarks convert brute-force k-NN results to grades:

```rust
fn knn_to_judgments(knn_results: &[ChunkId]) -> Vec<RelevanceJudgment> {
    knn_results.iter().enumerate().map(|(rank, &id)| {
        let relevance = if rank < 5 {
            2  // Highly relevant (positions 0-4)
        } else if rank < 20 {
            1  // Somewhat relevant (positions 5-19)
        } else {
            0  // Not relevant (positions 20+)
        };
        RelevanceJudgment::new(id, relevance)
    }).collect()
}
```

This position-based grading creates a conservative relevance scale:
- **Grade 2**: Top 5 nearest neighbors (should definitely be retrieved)
- **Grade 1**: Positions 5-19 (good to retrieve, but less critical)
- **Grade 0**: Everything else

The grading thresholds (5, 20) are arbitrary but consistent. What matters is that the same grading applies to all systems being compared.

### Experimental Constants

Key constants that affect benchmark behavior:

```rust
// Embedding configuration (matches production)
const EMBEDDING_DIM: usize = 512;        // JinaBERT hidden size
const EMBEDDINGS_NORMALIZED: bool = true; // L2-normalized vectors

// Query generation
const NUM_QUERIES: usize = 100;          // Queries per benchmark
const QUERY_SEED_BASE: u64 = 1_000_000;  // Separates query/doc seeds
const SEED: u64 = 42;                    // Master seed for RNG

// Ground truth
const GROUND_TRUTH_K: usize = 50;        // Top-50 for relevance judgments
const HIGHLY_RELEVANT_CUTOFF: usize = 5; // Grade 2 threshold
const SOMEWHAT_RELEVANT_CUTOFF: usize = 20; // Grade 1 threshold

// Evaluation
const K_VALUES: &[usize] = &[1, 5, 10, 20, 50]; // Precision/Recall@k cutoffs
```

### Limitations and Caveats

**Synthetic embeddings don't capture semantic relationships**: Real JinaBERT embeddings place semantically similar documents closer together. Synthetic hash-based embeddings are essentially random - "machine learning" and "deep learning" documents aren't closer than unrelated topics. This means the benchmarks test retrieval mechanics, not semantic quality.

**Topic cycling creates artificial structure**: With 8 topics rotating by document ID, documents 0, 8, 16, 24... all share the same topic template. This regularity wouldn't exist in real corpora.

**TF-IDF queries are optimistic**: Synthetic queries are constructed from the target document's most distinctive terms. Real users don't always use optimal query terms; they might use synonyms, misspellings, or vague descriptions.

**Every query has relevant documents**: In real search, users sometimes search for content that doesn't exist in the corpus, or their query is so vague that nothing truly matches. Synthetic benchmarks always have at least one relevant document (the target used to generate the query). This doesn't defeat precision/recall - the system can still return wrong documents (hurting precision) or miss relevant ones (hurting recall) - but it does mean we never test the "no good answer exists" scenario.

These limitations don't invalidate the benchmarks - they still measure whether the system correctly implements its algorithms and how performance scales. But they do mean benchmark scores are upper bounds on real-world quality.

### Reproducibility Checklist

To reproduce benchmark results exactly:

1. **Same code version**: Benchmark logic is in `crates/coppermind-core/benches/`
2. **Same Rust toolchain**: Check `rust-toolchain.toml` or use `rustup show`
3. **Same random seeds**: Default SEED = 42 used throughout
4. **Same corpus sizes**: Check constants at top of benchmark files
5. **Clean baseline**: Run `rm -rf target/criterion` before first run

Criterion saves results to `target/criterion/` for regression comparison. The first run after clearing establishes a fresh baseline.

---

## Performance Benchmarks

The performance benchmarks are located in `crates/coppermind-core/benches/` and use Criterion for statistical measurement.

### Search Latency

**Benchmark file**: `search.rs`

Search latency measures the time between receiving a query and returning results. This is the most user-visible performance metric because it directly affects the perceived responsiveness of the application.

Coppermind uses a hybrid search architecture with three components, each with different performance characteristics:

**Vector Search (HNSW)** finds documents with similar semantic meaning to the query. Under the hood, it converts the query text into a 512-dimensional embedding vector and searches for nearest neighbors in a graph structure called HNSW (Hierarchical Navigable Small World). The key insight of HNSW is that it achieves O(log n) search complexity rather than O(n), meaning search time grows very slowly as the corpus grows.

**Keyword Search (BM25)** finds documents containing the query terms. BM25 is a classical information retrieval algorithm that scores documents based on term frequency (how often query words appear in the document) and inverse document frequency (how rare those words are across all documents). It's fast because it uses inverted indices—precomputed mappings from words to documents containing them.

**Fusion (RRF)** combines the rankings from vector and keyword search. Reciprocal Rank Fusion works by converting each system's rankings into scores (1/(k + rank)) and summing them. This approach is remarkably robust because it only uses rank positions, not raw scores, making it immune to score scale differences between systems.

**Key benchmarks in `search.rs`:**
- `hnsw/search_by_k`: Vector search latency with varying k values
- `hnsw/search_by_size`: Vector search latency with varying corpus sizes
- `bm25/search_by_size`: Keyword search latency with varying corpus sizes
- `hybrid/search`: Full hybrid pipeline (vector + keyword + RRF)
- `hybrid/modality_comparison`: Side-by-side comparison of vector-only, keyword-only, and hybrid

**What good looks like:**
- p50 (median): < 10ms
- p90: < 50ms
- p99: < 100ms

### Indexing Performance

**Benchmark file**: `indexing.rs`

Indexing performance determines how quickly new documents become searchable and how long users wait when the application starts with existing data.

**Key benchmarks in `indexing.rs`:**
- `hnsw/insert_single`: Time to add one document to an existing index
- `hnsw/insert_batch`: Throughput when building an index from scratch
- `bm25/insert_batch`: BM25 index build throughput (baseline comparison)
- `hybrid/add_document`: Full document insertion (storage + HNSW + BM25)
- `hybrid/rebuild_from_store`: Application startup time with existing data
- `hnsw/compaction`: Time to compact index after deletions

**Index rebuild from storage** is perhaps the most critical indexing benchmark because it determines application startup time. When a user opens Coppermind with existing indexed documents, the system must reconstruct the in-memory search indices from persisted storage.

**What good looks like:**
- Single insertion: < 10ms
- Startup with 1,000 docs: < 2 seconds
- Startup with 10,000 docs: < 20 seconds

### Throughput

**Benchmark file**: `throughput.rs`

Throughput benchmarks measure sustained query performance and reveal issues that only manifest during continuous operation:

- **Cache effects**: The first query might be slow while subsequent queries benefit from warmed caches.
- **Memory pressure**: Temporary allocations during query processing might accumulate.

**Key benchmarks in `throughput.rs`:**
- `throughput/hnsw_sequential`: Vector search queries per second (single-threaded)
- `throughput/bm25_sequential`: Keyword search queries per second
- `throughput/hnsw_concurrent`: Vector search with multiple threads
- `throughput/hnsw_latency`: Latency percentile measurement (p50, p95, p99)
- `throughput/hybrid`: Hybrid search QPS

**What good looks like:**
- Stable QPS throughout the test (no degradation over time)
- p99 latency < 5x p50 latency (low variance)
- No out-of-memory errors or crashes

---

## Retrieval Quality Metrics

### The Ground Truth Problem

Quality metrics answer the question: "Did we return the right documents?" But this requires knowing which documents are right—the **ground truth** or **relevance judgments**.

In academic research, ground truth comes from human annotators who read queries and documents, then judge relevance. This is expensive and doesn't scale.

Coppermind uses **synthetic query generation** as a practical alternative. The process works as follows:

1. Select a target document from the corpus
2. Extract distinctive terms using TF-IDF (terms that are frequent in this document but rare elsewhere)
3. Form a query from these terms
4. Mark the target document as relevant (and optionally, similar documents)

This approach has limitations—synthetic queries don't capture the full complexity of real user information needs—but it provides consistent, reproducible ground truth for benchmarking.

### Precision and Recall

Precision and recall are foundational metrics that capture two different types of errors.

**Precision** measures the fraction of returned documents that are relevant. If you search for "machine learning" and get back 10 documents, but only 7 are actually about machine learning, your precision is 0.7 (70%). Low precision means users wade through irrelevant results.

**Recall** measures the fraction of relevant documents that were returned. If there are 20 documents about machine learning in your corpus but you only returned 7 of them, your recall is 0.35 (35%). Low recall means users miss relevant information.

These metrics trade off against each other. Returning more documents tends to increase recall (you're less likely to miss relevant ones) but decrease precision (you're more likely to include irrelevant ones). A system that returns every document has perfect recall but terrible precision.

**Precision@k and Recall@k** measure these values considering only the top k results. This reflects real user behavior—people rarely look beyond the first page of results. Precision@10 and Recall@10 are common choices.

**What good looks like:**
- Precision@10 > 0.7: Most returned results are relevant
- Recall@10 > 0.5: Finding at least half of relevant documents in top 10

### NDCG: Normalized Discounted Cumulative Gain

Precision and recall treat all positions equally—a relevant document at position 1 counts the same as one at position 10. But users care deeply about ranking. The first result gets the most attention; by position 10, many users have given up.

**NDCG (Normalized Discounted Cumulative Gain)** addresses this by applying position-based discounting. The formula has three components:

**Gain**: Each relevant document contributes a gain value. Binary relevance uses 1 for relevant, 0 for not relevant. Graded relevance might use scores like 0, 1, 2, 3 for increasing relevance levels.

**Discounted**: Gain is divided by log₂(position + 1). A relevant document at position 1 contributes gain/1 = full value. At position 2, it contributes gain/1.58. At position 10, only gain/3.46. This mathematical discount models the decreasing attention users pay to lower-ranked results.

**Cumulative**: Discounted gains are summed across all positions up to k.

**Normalized**: The cumulative discounted gain (DCG) is divided by the ideal DCG—what you'd get with perfect ranking. This normalization produces a score between 0 and 1, where 1 means optimal ranking.

The practical interpretation is intuitive: NDCG rewards putting relevant documents at the top. A system that finds all relevant documents but ranks them at positions 8, 9, 10 will score lower than one that puts them at positions 1, 2, 3.

**What good looks like:**
- NDCG@10 > 0.8: Excellent ranking
- NDCG@10 0.6-0.8: Good ranking
- NDCG@10 < 0.5: Significant room for improvement

### Mean Average Precision (MAP)

**Average Precision (AP)** captures the entire precision-recall curve in a single number. The idea is to measure precision at every position where a relevant document appears, then average these values.

Consider a ranked list where R marks relevant documents:

```
Position: 1  2  3  4  5  6  7  8  9  10
Relevant: R  -  R  -  -  R  -  -  -  R
```

Precision at each relevant position:
- Position 1: 1/1 = 1.0 (1 relevant out of 1 seen)
- Position 3: 2/3 = 0.67 (2 relevant out of 3 seen)
- Position 6: 3/6 = 0.5 (3 relevant out of 6 seen)
- Position 10: 4/10 = 0.4 (4 relevant out of 10 seen)

Average Precision = (1.0 + 0.67 + 0.5 + 0.4) / 4 = 0.64

**Mean Average Precision (MAP)** averages AP across multiple queries. This single number summarizes system performance across diverse information needs.

MAP rewards systems that rank relevant documents early. If all four relevant documents appeared at positions 1, 2, 3, 4, the precisions would be 1.0, 1.0, 1.0, 1.0, giving AP = 1.0.

**What good looks like:**
- MAP > 0.7: Excellent overall performance
- MAP 0.5-0.7: Good performance
- MAP < 0.4: Needs improvement

### Mean Reciprocal Rank (MRR)

Sometimes users just want one good answer. They search, click the first relevant result, and they're done. **Mean Reciprocal Rank** measures how well a system serves this use case.

**Reciprocal Rank (RR)** is simply 1 divided by the position of the first relevant result:

- First relevant at position 1: RR = 1/1 = 1.0
- First relevant at position 2: RR = 1/2 = 0.5
- First relevant at position 5: RR = 1/5 = 0.2
- No relevant in top k: RR = 0

**Mean Reciprocal Rank** averages RR across queries. High MRR means users typically find a relevant document near the top.

MRR is especially relevant for question-answering scenarios where there's often one best answer. It's less useful when users need to find multiple relevant documents.

**What good looks like:**
- MRR > 0.8: First relevant result usually in top 2
- MRR 0.5-0.8: First relevant result usually in top 5
- MRR < 0.3: Users often scroll to find relevant content

### F1 Score

The **F1 score** combines precision and recall into a single number using the harmonic mean:

```
F1 = 2 × (precision × recall) / (precision + recall)
```

The harmonic mean has an important property: it's low if either component is low. A system with 0.9 precision but 0.1 recall gets F1 = 0.18, not 0.5 as an arithmetic mean would give. This makes F1 useful when you care about both precision and recall and want to penalize severe imbalance.

**F1@k** applies this at a specific cutoff, measuring the balance of precision@k and recall@k.

**What good looks like:**
- F1@10 > 0.6: Good balance of precision and recall
- F1@10 < 0.4: Either precision or recall (or both) needs work

---

## Statistical Rigor

### Why Statistics Matter

Performance benchmarks are inherently noisy. Run the same search 100 times and you'll get 100 different latencies due to CPU scheduling, cache states, background processes, and other factors.

Without statistical analysis, you might observe System A at 10ms and System B at 11ms and conclude A is faster. But if the true distributions overlap significantly, this 1ms difference could be noise. Tomorrow's run might show B faster than A.

Statistical methods help distinguish signal from noise, ensuring that reported improvements are real and reproducible.

### Bootstrap Confidence Intervals

**Bootstrapping** is a technique for estimating the uncertainty of a measurement. The procedure is:

1. Collect n measurements (e.g., 100 latency values)
2. Randomly sample n values with replacement (some values will be picked multiple times, others not at all)
3. Compute the statistic of interest (e.g., mean) on this sample
4. Repeat steps 2-3 many times (e.g., 1,000 times)
5. The distribution of these computed statistics estimates uncertainty

A **95% confidence interval** spans from the 2.5th to 97.5th percentile of the bootstrap distribution. If you report "mean latency = 10ms, 95% CI [9.2, 10.8]", you're saying that the true mean is likely between 9.2ms and 10.8ms.

Confidence intervals enable meaningful comparisons. If System A has CI [9.2, 10.8] and System B has CI [12.1, 13.5], the intervals don't overlap, suggesting a real difference. If the intervals overlap substantially, the difference might be noise.

### Paired t-Tests

The **paired t-test** formally tests whether two systems differ. "Paired" means each query is run on both systems, and we analyze the per-query differences rather than overall averages.

Pairing is important because some queries are inherently harder than others. If Query 1 takes 50ms on both systems and Query 2 takes 5ms on both, comparing averages would obscure whether there's any per-query difference.

The t-test produces a **p-value**: the probability of observing a difference this large (or larger) if the systems were actually identical. By convention:

- p < 0.05: "Statistically significant" (less than 5% chance it's noise)
- p < 0.01: "Highly significant" (less than 1% chance)
- p > 0.05: "Not significant" (could plausibly be noise)

Important caveats: Statistical significance doesn't imply practical significance. With enough data, you can achieve p < 0.001 for a 0.1ms difference that no user would notice. Conversely, non-significance doesn't prove equality—it might just mean you need more data.

### Effect Size (Cohen's d)

**Cohen's d** measures the magnitude of a difference in standard deviation units:

```
d = (mean_A - mean_B) / pooled_standard_deviation
```

This answers "how big is the difference?" independent of statistical significance. Guidelines for interpretation:

- |d| < 0.2: Negligible effect (differences exist but are tiny)
- |d| = 0.2-0.5: Small effect
- |d| = 0.5-0.8: Medium effect
- |d| > 0.8: Large effect

Combining p-values and effect sizes gives a complete picture:

| p-value | Cohen's d | Interpretation |
|---------|-----------|----------------|
| < 0.05 | > 0.5 | Real and meaningful difference |
| < 0.05 | < 0.2 | Statistically significant but practically negligible |
| > 0.05 | > 0.5 | Might be meaningful, need more data |
| > 0.05 | < 0.2 | No evidence of meaningful difference |

---

## Interpreting Results

When reviewing benchmark results, consider multiple dimensions:

**Absolute values**: Is the system fast enough? Is quality high enough? Compare against user requirements and expectations.

**Relative comparisons**: Is version 2 better than version 1? Is configuration A better than B? Use statistical tests to validate differences.

**Tradeoffs**: Does improvement in one metric come at the cost of another? Faster search might mean lower quality. More accuracy might require more memory.

**Scalability trends**: How do metrics change with corpus size? A system that works well at 1,000 documents might struggle at 100,000.

**Variance**: Are results consistent or highly variable? High variance (wide confidence intervals, high p99/p50 ratio) suggests unpredictable user experience.

A complete evaluation considers all these factors. A system with good average performance but high variance and poor scaling might be worse than one with moderate average performance but consistent behavior at scale.

---

## Running the Benchmarks

### Tier 1: Performance Benchmarks (Criterion)

Run all performance benchmarks:
```bash
cargo bench -p coppermind-core --features candle-metal
```

Run specific benchmark files:
```bash
cargo bench -p coppermind-core --features candle-metal --bench search
cargo bench -p coppermind-core --features candle-metal --bench indexing
cargo bench -p coppermind-core --features candle-metal --bench throughput
```

**Results location:** Criterion saves results to `target/criterion/`. Open `target/criterion/report/index.html` in a browser for interactive visualizations including:
- Time series showing performance over benchmark history
- Violin plots showing latency distributions
- Comparison reports when baselines exist

**Feature flags:** For optimal performance on Apple Silicon, include the `candle-metal` feature to enable Metal GPU acceleration and Accelerate CPU optimization.

### Tier 2: Quality Evaluation (coppermind-eval)

Run quality evaluation:
```bash
# Build and run the evaluation binary (uses default cache at target/eval-cache)
cargo run -p coppermind-eval --release

# Output as JSON
cargo run -p coppermind-eval --release -- --json

# Custom cache directory
cargo run -p coppermind-eval --release -- --cache-dir /path/to/cache
```

**First run:** The evaluation tool automatically:
1. Creates an evaluation dataset (20 documents, 20 queries with known relevance)
2. Loads JinaBERT and computes embeddings (~1-2 minutes)
3. Caches embeddings to `target/eval-cache/` for fast subsequent runs

**Subsequent runs:** Load from cache, evaluation runs immediately.

**To regenerate the cache** (e.g., after model changes):
```bash
rm -rf target/eval-cache
cargo run -p coppermind-eval --release
```

**Output includes:**
- Per-system metrics (NDCG@10, MAP, MRR, Precision@k, Recall@k)
- Statistical comparisons (paired t-test, p-value, Cohen's d)
- 95% confidence intervals via bootstrap

### Adjusting Corpus Sizes

Performance benchmark corpus sizes are controlled by constants at the top of each file. For quick iteration, the defaults use smaller sizes (100-2000 documents). For thorough analysis, edit the constants.

Quality evaluation uses an embedded dataset with 20 documents and 20 queries. This is intentionally small for fast iteration; for larger-scale evaluation, modify the dataset generation in `crates/coppermind-eval/src/main.rs`.

---

## Further Reading

- [HNSW Paper](https://arxiv.org/abs/1603.09320): Original paper on Hierarchical Navigable Small World graphs
- [BM25 Explanation](https://en.wikipedia.org/wiki/Okapi_BM25): Overview of the BM25 ranking function
- [NDCG in Practice](https://en.wikipedia.org/wiki/Discounted_cumulative_gain): Detailed explanation of DCG and NDCG
- [Reciprocal Rank Fusion Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf): Original RRF paper by Cormack et al.
- [Bootstrap Methods](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)): Overview of bootstrap statistical techniques
