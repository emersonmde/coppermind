# coppermind-eval

Scientific evaluation tool for measuring Coppermind's search quality using standard IR metrics.

## Overview

This crate provides a rigorous evaluation framework for comparing vector search, keyword search (BM25), and hybrid search (RRF fusion) strategies. It uses a custom dataset generated from Wikipedia articles with diverse query types designed to stress-test hybrid search.

## Prerequisites

```bash
# 1. Download the embedding model
./download-models.sh

# 2. Generate the evaluation dataset (requires ANTHROPIC_API_KEY)
cd crates/coppermind-eval
./scripts/generate-dataset.sh
```

The dataset generator fetches ~200 Wikipedia articles and uses Claude to generate 7 query types per article:

| Query Type | Description | Example |
|------------|-------------|---------|
| `SHORT_KEYWORD` | 2-3 word keyword query | "dna structure" |
| `QUESTION` | Natural language question | "What is the structure of DNA?" |
| `CONCEPTUAL` | Describes concept without using title | "molecule carrying genetic instructions" |
| `SYNONYM` | Uses related/scientific terms | "deoxyribonucleic acid double helix" |
| `TYPO` | Realistic misspelling | "chromosone" |
| `PARTIAL` | Incomplete query (mid-typing) | "dna doub" |
| `QUESTION_DETAIL` | Question about specific detail | "Who discovered the double helix?" |

## Usage

```bash
# Run standard evaluation
cargo run -p coppermind-eval --release

# Output JSON for analysis
cargo run -p coppermind-eval --release -- --json

# Run RRF k-parameter ablation study
cargo run -p coppermind-eval --release -- --ablation rrf

# Show per-query breakdown
cargo run -p coppermind-eval --release -- --per-query

# Custom k values
cargo run -p coppermind-eval --release -- --k-values 1,5,10,20,50

# Skip specific search types
cargo run -p coppermind-eval --release -- --skip-keyword
cargo run -p coppermind-eval --release -- --skip-vector

# Force recompute embeddings (clears cache)
cargo run -p coppermind-eval --release -- --force-recompute
```

## Metrics

The evaluation computes standard IR metrics at multiple cutoffs (k=1, 5, 10, 20):

| Metric | Description |
|--------|-------------|
| **NDCG@k** | Normalized Discounted Cumulative Gain - measures ranking quality with graded relevance |
| **MAP@k** | Mean Average Precision - average precision across all queries |
| **MRR@k** | Mean Reciprocal Rank - position of first relevant result |
| **P@k** | Precision at k - fraction of top-k results that are relevant |
| **R@k** | Recall at k - fraction of relevant documents in top-k |

Statistical significance testing uses paired t-tests with effect size (Cohen's d) to compare systems.

## Ablation Studies

### RRF k-parameter sweep

The `--ablation rrf` flag tests different RRF k values (default: 10, 30, 60, 100, 200) to find optimal fusion parameters. RRF score formula: `score = 1/(k + rank)`.

Lower k values weight top ranks more heavily; higher k values produce smoother score distributions.

## Output Format

### Human-readable (default)

```
================================================================================
COPPERMIND SEARCH QUALITY EVALUATION
================================================================================

Dataset: Coppermind Custom (200 docs, 1400 queries, 1400 qrels)

----------------------------------------------------------------------
RESULTS @ k=10
System          NDCG      MAP      MRR     Prec   Recall
vector        0.7234   0.6891   0.8123   0.1000   0.9856
keyword       0.6543   0.6234   0.7456   0.1000   0.9234
hybrid        0.7512   0.7123   0.8345   0.1000   0.9912

----------------------------------------------------------------------
STATISTICAL COMPARISONS (* = p < 0.05)
hybrid vs vector (NDCG@10): p=0.0234* effect=0.156
hybrid vs keyword (NDCG@10): p=0.0001* effect=0.423
```

### JSON (`--json`)

```json
{
  "dataset": {
    "name": "Coppermind Custom",
    "num_documents": 200,
    "num_queries": 1400,
    "num_qrels": 1400
  },
  "k_values": [1, 5, 10, 20],
  "systems": [...],
  "comparisons": [...],
  "ablations": {...}
}
```

## Caching

Embeddings are cached in `target/eval-cache/coppermind-embeddings.bin` to avoid recomputing on subsequent runs. Use `--force-recompute` to regenerate.

## Dependencies

- **coppermind-core**: Core search engine library
- **elinor**: IR evaluation metrics and statistical tests
- **clap**: CLI argument parsing
- **indicatif**: Progress bars

## Current Evaluation Results


```
================================================================================
COPPERMIND SEARCH QUALITY EVALUATION
================================================================================

Dataset: Coppermind Custom (234 docs, 1160 queries, 1160 qrels)

----------------------------------------------------------------------
RESULTS @ k=1
System           NDCG      MAP      MRR     Prec   Recall
vector         0.8681   0.8681   0.8681   0.8681   0.8681
keyword        0.8733   0.8733   0.8733   0.8733   0.8733
hybrid         0.8733   0.8733   0.8733   0.8733   0.8733

----------------------------------------------------------------------
RESULTS @ k=5
System           NDCG      MAP      MRR     Prec   Recall
vector         0.9322   0.9163   0.9163   0.1957   0.9784
keyword        0.9327   0.9172   0.9172   0.1957   0.9784
hybrid         0.9399   0.9234   0.9234   0.1976   0.9879

----------------------------------------------------------------------
RESULTS @ k=10
System           NDCG      MAP      MRR     Prec   Recall
vector         0.9357   0.9177   0.9177   0.0990   0.9897
keyword        0.9384   0.9196   0.9196   0.0996   0.9957
hybrid         0.9424   0.9245   0.9245   0.0996   0.9957

----------------------------------------------------------------------
RESULTS @ k=20
System           NDCG      MAP      MRR     Prec   Recall
vector         0.9374   0.9182   0.9182   0.0498   0.9966
keyword        0.9393   0.9199   0.9199   0.0500   0.9991
hybrid         0.9435   0.9248   0.9248   0.0500   1.0000

----------------------------------------------------------------------
STATISTICAL COMPARISONS (* = p < 0.05)
hybrid vs vector (NDCG@10): p=0.0332* effect=0.063
hybrid vs keyword (NDCG@10): p=0.2224 effect=0.036
================================================================================
```
