# IR Evaluation Dataset Analysis for Coppermind

## Executive Summary

This document analyzes IR evaluation datasets for Coppermind, a local-first semantic search engine running in-browser via WASM. The analysis focuses on dataset suitability for evaluating hybrid search (vector + BM25 with RRF fusion) on personal documents, notes, code, and markdown files.

**Key Findings:**
- **MS MARCO**: Non-commercial license (research only), very large corpus (8.8M passages), web search queries
- **Natural Questions**: CC BY-SA 3.0 (commercial use allowed), large corpus (3.2M Wikipedia passages), natural language questions
- **BEIR Benchmark**: Mixed licenses per dataset, comprehensive evaluation across 18 diverse tasks
- **Recommended for Coppermind**: Small BEIR datasets (SciFact, NFCorpus, ArguAna, FiQA) due to manageable size and diverse document types

---

## Dataset Analysis

### 1. Natural Questions (NQ)

#### Dataset Characteristics
- **Size**: 307,373 training queries + 7,830 dev + 7,842 test
- **Corpus**: 3.2M Wikipedia passages (union of candidate passages shown to annotators)
- **Document Type**: Wikipedia articles (full pages)
- **Document Length**: Long-form encyclopedia articles
- **Query Type**: Real user questions from Google Search (natural language)
- **Relevance Judgments**:
  - Long answers (paragraph-level)
  - Short answers (entity-level)
  - Yes/No answers
  - 5-way annotations for dev/test sets
  - 90% annotation accuracy
- **Relevance Density**: Sparse (not all documents contain answers)

#### Licensing
- **License**: CC BY-SA 3.0 (Creative Commons Attribution-ShareAlike 3.0)
- **Commercial Use**: ✅ **Allowed** (with attribution and share-alike requirements)
- **Attribution**: Required
- **Share-Alike**: Derivative works must use same license
- **Source**: [Natural Questions GitHub](https://github.com/google-research-datasets/natural-questions), [License File](https://github.com/google-research-datasets/natural-questions/blob/master/LICENSE)

#### Alignment with Coppermind

**Document Type Match**: ⚠️ **Moderate**
- Wikipedia articles are structured, informative text (similar to documentation)
- Not representative of personal notes, markdown files, or code
- More formal than typical personal documents

**Query Type Match**: ✅ **Good**
- Natural language questions match expected user behavior
- Real user queries (not artificial)
- Diverse information needs

**Corpus Size**: ❌ **Challenging**
- 3.2M passages requires significant embedding computation
- At ~1s per chunk (web worker), would take ~37 days of continuous processing
- Desktop with Metal acceleration: ~10-50ms per chunk = 9-44 hours
- Not practical for rapid iteration during development

**Practical Considerations**:
- ✅ Available via BEIR, ir_datasets, Hugging Face, TensorFlow Datasets
- ✅ Well-documented with baseline results
- ❌ Very large download size (exact GB not specified, but includes full Wikipedia dump)
- ⚠️ Subset availability: Can use dev set (7,830 queries) for faster evaluation

#### Formats Available
- Original format: JSONL with HTML annotations
- BEIR format: corpus.jsonl, queries.jsonl, qrels.tsv
- ir_datasets: Python library access
- Simplified format: Available via `simplify_nq_example()` function

---

### 2. MS MARCO

#### Dataset Characteristics
- **Size**:
  - Passage Ranking: 8,841,823 passages
  - Queries: 1,010,916 unique queries (from Bing search logs)
  - Training: ~500k query-passage pairs
  - Dev set: 6,980 queries
  - V2: 138M passages (much larger)
- **Document Type**: Web passages (short text snippets)
- **Document Length**: Average 55.98 words (short passages)
- **Query Type**: Real Bing search queries (keyword + natural language mix)
- **Relevance Judgments**:
  - Binary (relevant/non-relevant)
  - Derived from QnA dataset (which passages contained answers)
  - Sparse judgments (avg 1.1 relevant docs per query)
  - One of the largest relevance datasets ever created
- **Relevance Density**: Very sparse (millions of candidates, few relevant per query)

#### Licensing
- **License**: ⚠️ **Non-commercial research only**
- **Commercial Use**: ❌ **NOT Allowed**
- **Restriction**: "Intended for non-commercial research purposes only"
- **Trademark**: MS MARCO name/logo not licensed for use
- **Source**: [MS MARCO Official Site](https://microsoft.github.io/msmarco/), [GitHub Repository](https://github.com/microsoft/MSMARCO-Passage-Ranking)

**Impact on Coppermind**:
- Cannot use for benchmarking if Coppermind is considered a commercial product
- Research/academic use is fine
- Would need Microsoft permission for commercial evaluation

#### Alignment with Coppermind

**Document Type Match**: ⚠️ **Moderate**
- Web passages are diverse but short
- Not representative of personal documents, long-form notes, or code
- More representative of search engine snippets than personal knowledge base

**Query Type Match**: ✅ **Good**
- Real user search queries
- Mix of keyword and natural language
- Representative of actual search behavior

**Corpus Size**: ❌ **Very Challenging**
- 8.8M passages is extremely large for local embedding
- V2 (138M passages) is completely impractical
- Even dev set (6,980 queries) requires searching entire 8.8M corpus
- Estimated embedding time (web): ~102 days continuous
- Estimated embedding time (desktop Metal): 24-122 hours

**Practical Considerations**:
- ✅ Available via BEIR, Hugging Face, ir_datasets
- ✅ Industry-standard benchmark (widely used)
- ❌ Download size: Large (exact GB not specified)
- ⚠️ Subset availability: Dev set available but still requires full corpus
- ⚠️ Non-commercial license restriction

#### Formats Available
- BEIR format: corpus.jsonl, queries.jsonl, qrels.tsv
- Official format: .tar.gz archives from Azure blob storage
- ir_datasets: Python library access
- Hugging Face: `microsoft/ms_marco`, `BeIR/msmarco`

---

### 3. BEIR Benchmark

BEIR (Benchmarking Information Retrieval) is a heterogeneous benchmark containing 18 datasets across 9 IR tasks, designed for zero-shot evaluation.

#### Overall Characteristics
- **Framework License**: Apache 2.0 (code/library)
- **Dataset Licenses**: ⚠️ **Varies by individual dataset** (user responsibility to verify)
- **Tasks**: 9 diverse IR tasks
- **Datasets**: 18 total (14 public, 4 private requiring email access)
- **Format**: Standardized (corpus.jsonl, queries.jsonl, qrels.tsv)
- **Focus**: Zero-shot evaluation (no training data provided)

#### BEIR Dataset Inventory

Based on research, here are the 18 BEIR datasets organized by task:

| Task | Datasets | Size Category |
|------|----------|---------------|
| **Fact-Checking** | FEVER, Climate-FEVER, SciFact | 1K-10M |
| **Question-Answering** | Natural Questions, HotpotQA, FiQA | 50K-5M |
| **Bio-Medical IR** | TREC-COVID, BioASQ, NFCorpus | 3K-171K |
| **News Retrieval** | TREC-NEWS, Robust04 | (sizes not specified) |
| **Argument Retrieval** | Touche-2020, ArguAna | <10K |
| **Duplicate Question** | Quora, CQADupStack | 100K-500K |
| **Citation Prediction** | SCIDOCS | 25K |
| **Tweet Retrieval** | Signal-1M | 1M |
| **Entity Retrieval** | DBPedia | 4.6M |

#### Detailed Statistics for Key Datasets

| Dataset | Corpus Size | Test Queries | Avg Rel/Query | License | Commercial Use |
|---------|-------------|--------------|---------------|---------|----------------|
| **SciFact** | 5,183 | 300 | ~4.9 | CC BY-NC 2.0 | ❌ No |
| **NFCorpus** | 3,633 | 323 | 38.2 | CC BY-SA 4.0 | ✅ Yes |
| **ArguAna** | 8,674 | 1,406 | - | CC BY-SA 4.0 | ✅ Yes |
| **FiQA** | 57,638 | 648 | 2.6 | CC BY 3.0 | ✅ Yes |
| **TREC-COVID** | 171,332 | 50 | 493.5 | CC BY-NC-SA 4.0 | ❌ No |
| **SCIDOCS** | 25,000 | 1,000 | 4.9 | - | Unknown |
| **Touche-2020** | 382,545 | 49 | 19.0 | - | Unknown |
| **MS MARCO** | 8,841,823 | 6,980 | 1.1 | Research only | ❌ No |
| **Natural Questions** | 2,681,468 | 3,452 | 1.2 | CC BY-SA 3.0 | ✅ Yes |
| **HotpotQA** | 5,233,329 | 7,405 | 2.0 | - | Unknown |
| **DBPedia** | 4,635,922 | 400 | 38.2 | - | Unknown |
| **FEVER** | 5,416,568 | 6,666 | 1.2 | - | Unknown |
| **Quora** | 522,931 | 10,000 | 1.6 | - | Unknown |
| **Climate-FEVER** | 5,416,593 | 1,535 | - | - | Unknown |
| **CQADupStack** | ~22,998 (per subforum) | ~699 (per subforum) | - | - | Unknown |

**Note**: Sizes vary between sources; these are approximate based on BEIR paper and Hugging Face metadata.

#### License Analysis

**Creative Commons License Types**:
- **CC BY-SA** (Attribution-ShareAlike): ✅ Commercial use allowed, must share derivatives under same license
- **CC BY-NC** (Attribution-NonCommercial): ❌ Commercial use NOT allowed
- **CC BY** (Attribution): ✅ Commercial use allowed, attribution required

**BEIR Framework Disclaimer**:
> "We only distribute these datasets in a specific format, but we do not vouch for their quality or fairness, or claim that you have license to use the dataset. It remains the user's responsibility to determine whether you as a user have permission to use the dataset under the dataset's license."

#### Alignment with Coppermind

**Small Datasets (Recommended for Coppermind)**:

1. **NFCorpus** (3.6K docs, 323 queries)
   - ✅ Medical/nutrition information retrieval
   - ✅ Natural language queries from NutritionFacts.org
   - ✅ CC BY-SA 4.0 (commercial use allowed)
   - ✅ Manageable size: ~1-2 minutes embedding (web), ~2-18 seconds (desktop)
   - ⚠️ Domain-specific (nutrition/medical), not general personal docs

2. **SciFact** (5.2K docs, 300 queries)
   - ✅ Scientific claim verification
   - ✅ Expert-written claims + evidence abstracts
   - ❌ CC BY-NC 2.0 (non-commercial only)
   - ✅ Manageable size: ~1-2 minutes embedding (web), ~2-26 seconds (desktop)
   - ⚠️ Highly specialized (scientific papers)

3. **ArguAna** (8.7K docs, 1.4K queries)
   - ✅ Argument retrieval (counterargument finding)
   - ✅ CC BY-SA 4.0 (commercial use allowed)
   - ✅ Manageable size: ~2-3 minutes embedding (web), ~4-43 seconds (desktop)
   - ⚠️ Specialized task (debate arguments from idebate.org)

4. **FiQA** (57K docs, 648 queries)
   - ✅ Financial question answering
   - ✅ CC BY 3.0 (commercial use allowed)
   - ✅ Moderate size: ~16 minutes embedding (web), ~28-285 seconds (desktop)
   - ⚠️ Domain-specific (finance)

**Medium Datasets (Possible with patience)**:

5. **TREC-COVID** (171K docs, 50 queries)
   - ✅ Scientific articles (COVID-19 research)
   - ❌ CC BY-NC-SA 4.0 (non-commercial only)
   - ⚠️ Larger: ~3 hours embedding (web), ~8.5-85 minutes (desktop)
   - ⚠️ Only 50 queries (very small test set)

**Document Type Match**:
- ⚠️ **Moderate overall**: Most BEIR datasets are domain-specific (medical, scientific, financial)
- None are representative of personal notes, markdown files, or general code
- NFCorpus and ArguAna are closest to informational text retrieval
- No markdown or code-specific datasets in BEIR

**Query Type Match**:
- ✅ **Good**: Natural language queries across most datasets
- Mix of questions, claims, and information needs
- Representative of how users search their personal knowledge

**Practical Considerations**:
- ✅ Standardized format (easy integration)
- ✅ Well-documented with baseline results
- ✅ Python library (`pip install beir`)
- ✅ Zero-shot evaluation (no training needed)
- ✅ Hugging Face integration
- ⚠️ Must verify each dataset's license individually

#### Formats Available
- BEIR standardized format: `corpus.jsonl`, `queries.jsonl`, `qrels.tsv`
- Python library: `from beir import util; util.download_and_unzip(url, out_dir)`
- Hugging Face: `BeIR/{dataset_name}` (e.g., `BeIR/scifact`)
- ir_datasets: `ir_datasets.load("beir/{dataset_name}")`

---

## Alternative Datasets Considered

### CoIR (Code Information Retrieval Benchmark)

- **Overview**: Comprehensive benchmark for code retrieval across 10 datasets, 8 tasks, 7 domains
- **Tasks**: Text-to-Code, Code-to-Code, Code-to-Text, Hybrid Code Retrieval
- **Relevance to Coppermind**: ✅ **Highly relevant** for code search use case
- **Status**: Recent (2024 publication)
- **License**: Unknown (requires further investigation)
- **Challenge**: Specialized for code; not representative of notes/markdown/general docs

**Recommendation**: Worth investigating for code-specific evaluation, but should be combined with general document retrieval datasets.

### OmniDocBench

- **Overview**: Comprehensive benchmark for document parsing across 9 sources
- **Sources**: Academic papers, textbooks, handwritten notes, newspapers
- **Relevance to Coppermind**: ⚠️ **Partial** - focuses on parsing/OCR, not retrieval
- **Unique Value**: Includes handwritten notes (closest to personal docs)
- **License**: Unknown
- **Challenge**: Designed for parsing evaluation, not search/retrieval

**Recommendation**: Not suitable for IR evaluation, but interesting for document processing pipeline.

### CRUMB (Complex Retrieval with Unified Markdown Benchmark)

- **Overview**: Benchmark for complex retrieval with unified markdown format
- **Format**: ✅ Documents in markdown with contextualized chunks
- **Relevance to Coppermind**: ✅ **High** - markdown format matches Coppermind use case
- **Status**: Relatively new
- **License**: Unknown
- **Challenge**: Limited information available; need to investigate further

**Recommendation**: Promising for markdown document evaluation; requires more research.

---

## Recommendations

### For Immediate Use (Development & Testing)

**1. Start with Small BEIR Datasets**:
- **NFCorpus** (3.6K docs): Fast to embed, CC BY-SA 4.0 (commercial OK)
- **ArguAna** (8.7K docs): Slightly larger, argument retrieval, CC BY-SA 4.0
- **FiQA** (57K docs): Medium size, financial QA, CC BY 3.0

**Rationale**:
- Manageable embedding time (minutes to ~30 minutes)
- Diverse query types
- Multiple datasets provide robustness
- Mix of dense (NFCorpus: 38.2 rel/query) and sparse (FiQA: 2.6 rel/query) relevance
- Commercial-use-friendly licenses

**2. Use BEIR Framework**:
```python
from beir import util
from beir.datasets.data_loader import GenericDataLoader

# Download datasets
datasets = ["nfcorpus", "arguana", "fiqa"]
for dataset in datasets:
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    data_path = util.download_and_unzip(url, "datasets")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
```

**3. Evaluation Metrics**:
- NDCG@10 (primary metric for BEIR)
- Recall@100
- MAP (Mean Average Precision)
- Compare against BM25 baseline (BEIR provides reference scores)

### For Comprehensive Evaluation (Post-Development)

**4. Add Medium BEIR Datasets** (if time permits):
- **SCIDOCS** (25K docs): Citation prediction, 1K queries
  - ⚠️ License unclear, verify before use
- **Touche-2020** (382K docs): Argument retrieval, larger scale
  - ⚠️ License unclear, verify before use

**5. Consider Natural Questions Dev Set**:
- Use NQ dev set (7,830 queries) if willing to embed full corpus
- Provides real user queries and Wikipedia content
- ✅ CC BY-SA 3.0 (commercial OK)
- ⚠️ Requires ~9-44 hours embedding time (desktop) or subset corpus

**6. Investigate Code-Specific Evaluation**:
- Research **CoIR** benchmark for code retrieval
- Verify license and dataset availability
- Provides targeted evaluation for Coppermind's code search feature

### For Production Benchmarking (Post-Launch)

**7. Create Custom Evaluation Dataset**:
- Collect anonymized real user queries (with consent)
- Use actual personal documents, notes, markdown, code from volunteers
- Build qrels through manual relevance judgments
- Most representative of actual Coppermind use cases
- Can be open-sourced to contribute to IR community

**Why custom dataset**:
- No existing dataset matches "personal document search" domain
- BEIR datasets are domain-specific (medical, scientific, financial)
- Real user data would validate performance on actual use case

### Licensing Considerations

**Commercial Use Safe** (verified licenses):
- ✅ NFCorpus (CC BY-SA 4.0)
- ✅ ArguAna (CC BY-SA 4.0)
- ✅ FiQA (CC BY 3.0)
- ✅ Natural Questions (CC BY-SA 3.0)

**Non-Commercial Only** (avoid for product benchmarking):
- ❌ MS MARCO (research only)
- ❌ SciFact (CC BY-NC 2.0)
- ❌ TREC-COVID (CC BY-NC-SA 4.0)

**Unknown/Verify** (research before use):
- ⚠️ SCIDOCS, Touche-2020, HotpotQA, DBPedia, FEVER, Quora, CoIR, CRUMB
- Check individual dataset licenses in BEIR paper Appendix D
- Contact dataset maintainers if unclear

### Practical Implementation Plan

**Phase 1: Quick Validation** (1-2 hours embedding + evaluation)
1. Download NFCorpus, ArguAna, FiQA from BEIR
2. Embed corpora using Coppermind indexing pipeline
3. Run test queries through hybrid search
4. Compute NDCG@10, Recall@100, MAP
5. Compare against BM25 baseline (BEIR provides reference)

**Phase 2: Comprehensive Testing** (1-2 days)
1. Add SCIDOCS (verify license first)
2. Consider NQ dev set if corpus subset available
3. Investigate CoIR for code retrieval
4. Document performance across diverse tasks

**Phase 3: Real-World Validation** (ongoing)
1. Design custom dataset collection process
2. Recruit volunteers for document donation + relevance judgments
3. Build representative test collection
4. Benchmark and iterate on real use cases

---

## Dataset Comparison Matrix

| Dataset | Corpus Size | Queries | License | Commercial | Embed Time (Web) | Embed Time (Desktop) | Document Type Match | Query Type Match | Overall Fit |
|---------|-------------|---------|---------|------------|------------------|----------------------|---------------------|------------------|-------------|
| **NFCorpus** | 3.6K | 323 | CC BY-SA 4.0 | ✅ Yes | ~2 min | ~2-18 sec | ⚠️ Medical | ✅ Natural | ⭐⭐⭐⭐ |
| **ArguAna** | 8.7K | 1.4K | CC BY-SA 4.0 | ✅ Yes | ~3 min | ~4-43 sec | ⚠️ Arguments | ✅ Natural | ⭐⭐⭐⭐ |
| **FiQA** | 57K | 648 | CC BY 3.0 | ✅ Yes | ~16 min | ~28-285 sec | ⚠️ Financial | ✅ Questions | ⭐⭐⭐⭐ |
| **SciFact** | 5.2K | 300 | CC BY-NC 2.0 | ❌ No | ~2 min | ~2-26 sec | ⚠️ Scientific | ✅ Claims | ⭐⭐⭐ |
| **SCIDOCS** | 25K | 1K | Unknown | ❓ Unknown | ~7 min | ~12-125 sec | ⚠️ Scientific | ✅ Citations | ⭐⭐⭐ |
| **TREC-COVID** | 171K | 50 | CC BY-NC-SA 4.0 | ❌ No | ~3 hours | ~8.5-85 min | ⚠️ Scientific | ✅ Natural | ⭐⭐ |
| **Natural Questions** | 3.2M | 7.8K | CC BY-SA 3.0 | ✅ Yes | ~37 days | ~9-44 hours | ⚠️ Wikipedia | ✅ Natural | ⭐⭐⭐ |
| **MS MARCO** | 8.8M | 6.9K | Research only | ❌ No | ~102 days | ~24-122 hours | ⚠️ Web snippets | ✅ Search | ⭐⭐ |

**Embed Time Calculations**:
- Web (WASM): ~1 second per chunk (single-threaded worker)
- Desktop (Metal): ~10-50ms per chunk (GPU accelerated)
- Assumes 1 chunk per document (conservative; actual chunking may vary)

**Overall Fit Rating**:
- ⭐⭐⭐⭐⭐: Perfect match for Coppermind use case
- ⭐⭐⭐⭐: Good match, recommended for use
- ⭐⭐⭐: Acceptable, consider with caveats
- ⭐⭐: Poor match, use only if necessary
- ⭐: Not suitable

---

## Conclusion

**For Coppermind evaluation, prioritize**:
1. **NFCorpus**, **ArguAna**, **FiQA**: Small, fast, commercially viable, diverse tasks
2. **BEIR framework**: Standardized, well-documented, industry-accepted
3. **Custom dataset** (long-term): Most representative of actual use case

**Avoid for commercial benchmarking**:
- MS MARCO (non-commercial license)
- SciFact (CC BY-NC)
- TREC-COVID (CC BY-NC-SA)

**Key limitations**:
- No existing dataset matches "personal document search" domain perfectly
- All evaluated datasets are domain-specific (medical, scientific, financial, encyclopedic)
- Code retrieval datasets (CoIR) are specialized; need investigation
- Markdown-specific benchmarks (CRUMB) are emerging; need more research

**Next steps**:
1. Implement BEIR dataset loading in Coppermind
2. Run initial evaluation on NFCorpus, ArguAna, FiQA
3. Compare hybrid search vs pure vector vs pure BM25
4. Investigate CoIR and CRUMB for specialized evaluation
5. Design custom dataset collection process for real-world validation

---

## Sources

- [Natural Questions - ir_datasets](https://ir-datasets.com/natural-questions.html)
- [Natural Questions GitHub Repository](https://github.com/google-research-datasets/natural-questions)
- [Natural Questions License](https://github.com/google-research-datasets/natural-questions/blob/master/LICENSE)
- [MS MARCO Official Website](https://microsoft.github.io/msmarco/)
- [MS MARCO GitHub - Web Search](https://github.com/microsoft/MS-MARCO-Web-Search)
- [MS MARCO Datasets Page](https://microsoft.github.io/msmarco/Datasets.html)
- [BEIR GitHub Repository](https://github.com/beir-cellar/beir)
- [BEIR Paper (arXiv:2104.08663)](https://arxiv.org/abs/2104.08663)
- [BEIR License (Apache 2.0)](https://github.com/beir-cellar/beir/blob/main/LICENSE)
- [BEIR Datasets Wiki](https://github.com/beir-cellar/beir/wiki/Datasets-available)
- [BEIR - ir_datasets](https://ir-datasets.com/beir.html)
- [SciFact - Hugging Face](https://huggingface.co/datasets/allenai/scifact)
- [SciFact GitHub](https://github.com/allenai/scifact)
- [NFCorpus - StatNLP Heidelberg](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)
- [NFCorpus - ir_datasets](https://ir-datasets.com/nfcorpus.html)
- [FiQA Challenge Website](https://sites.google.com/view/fiqa/home)
- [ArguAna - Hugging Face](https://huggingface.co/datasets/BeIR/arguana)
- [ArguAna - Webis Data](https://webis.de/data/arguana-counterargs.html)
- [Creative Commons Licenses Overview](https://creativecommons.org/share-your-work/cclicenses/)
- [CC License Comparison - Wikipedia](https://en.wikipedia.org/wiki/Creative_Commons_license)
- [CoIR Benchmark (arXiv:2407.02883)](https://arxiv.org/html/2407.02883v1)
- [OmniDocBench GitHub](https://github.com/opendatalab/OmniDocBench)
