//! Coppermind Evaluation Tool
//!
//! Scientific evaluation of search quality using a custom dataset designed
//! to test hybrid search with diverse query types.
//!
//! # Prerequisites
//!
//! ```bash
//! # Generate the evaluation dataset (requires ANTHROPIC_API_KEY)
//! cd crates/coppermind-eval && ./scripts/generate-dataset.sh
//!
//! # Download the embedding model
//! ./download-models.sh
//! ```
//!
//! # Usage
//!
//! ```bash
//! # Run evaluation
//! cargo run -p coppermind-eval --release
//!
//! # Output JSON for analysis
//! cargo run -p coppermind-eval --release -- --json
//!
//! # Run RRF ablation study
//! cargo run -p coppermind-eval --release -- --ablation rrf
//!
//! # Show per-query breakdown
//! cargo run -p coppermind-eval --release -- --per-query
//! ```

mod datasets;

use clap::{Parser, ValueEnum};
use coppermind_core::config::EMBEDDING_DIM;
use coppermind_core::embedding::{Embedder, JinaBertConfig, JinaBertEmbedder, TokenizerHandle};
use coppermind_core::search::fusion::reciprocal_rank_fusion;
use coppermind_core::search::keyword::KeywordSearchEngine;
use coppermind_core::search::types::ChunkId;
use coppermind_core::search::vector::VectorSearchEngine;
use datasets::{load_coppermind_dataset, EvalDataset};
use elinor::statistical_tests::{pairs_from_maps, StudentTTest};
use elinor::{Metric, PredRelStoreBuilder, TrueRelStoreBuilder};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

const MAX_POSITION_EMBEDDINGS: usize = 2048;
const DEFAULT_K_VALUES: &[usize] = &[1, 5, 10, 20];
const DEFAULT_RRF_K_VALUES: &[usize] = &[10, 30, 60, 100, 200];

// =============================================================================
// CLI
// =============================================================================

#[derive(Debug, Clone, ValueEnum)]
enum AblationType {
    Rrf,
    All,
}

#[derive(Parser, Debug)]
#[command(name = "coppermind-eval")]
#[command(about = "Evaluate Coppermind search quality")]
struct Args {
    /// Output results as JSON
    #[arg(long)]
    json: bool,

    /// Cache directory for embeddings
    #[arg(long, default_value = "target/eval-cache")]
    cache_dir: PathBuf,

    /// k values for evaluation (comma-separated)
    #[arg(long, value_delimiter = ',')]
    k_values: Option<Vec<usize>>,

    /// Run ablation study
    #[arg(long, value_enum)]
    ablation: Option<AblationType>,

    /// RRF k values for ablation (comma-separated)
    #[arg(long, value_delimiter = ',')]
    rrf_k_values: Option<Vec<usize>>,

    /// Skip vector search
    #[arg(long)]
    skip_vector: bool,

    /// Skip keyword search
    #[arg(long)]
    skip_keyword: bool,

    /// Skip hybrid search
    #[arg(long)]
    skip_hybrid: bool,

    /// Force recomputation of embeddings
    #[arg(long)]
    force_recompute: bool,

    /// Show per-query breakdown
    #[arg(long)]
    per_query: bool,
}

// =============================================================================
// Output Types
// =============================================================================

#[derive(Debug, Serialize)]
struct EvalReport {
    dataset: DatasetInfo,
    k_values: Vec<usize>,
    systems: Vec<SystemResult>,
    comparisons: Vec<Comparison>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ablations: Option<AblationResults>,
    #[serde(skip_serializing_if = "Option::is_none")]
    per_query: Option<PerQueryAnalysis>,
}

#[derive(Debug, Serialize)]
struct DatasetInfo {
    name: String,
    num_documents: usize,
    num_queries: usize,
    num_qrels: usize,
}

#[derive(Debug, Serialize)]
struct SystemResult {
    name: String,
    metrics_by_k: BTreeMap<usize, MetricScores>,
}

#[derive(Debug, Clone, Serialize)]
struct MetricScores {
    ndcg: f64,
    map: f64,
    mrr: f64,
    precision: f64,
    recall: f64,
}

#[derive(Debug, Serialize)]
struct Comparison {
    system_a: String,
    system_b: String,
    metric: String,
    k: usize,
    p_value: f64,
    effect_size: f64,
    significant: bool,
}

#[derive(Debug, Serialize)]
struct AblationResults {
    rrf_k_sweep: Vec<RrfKResult>,
}

#[derive(Debug, Serialize)]
struct RrfKResult {
    rrf_k: usize,
    metrics_by_k: BTreeMap<usize, MetricScores>,
}

#[derive(Debug, Serialize)]
struct PerQueryAnalysis {
    query_results: Vec<QueryResult>,
    queries_where_hybrid_wins: usize,
    queries_where_hybrid_loses: usize,
}

#[derive(Debug, Serialize)]
struct QueryResult {
    query_id: String,
    query_text: String,
    scores: BTreeMap<String, f64>,
}

// =============================================================================
// Model Loading
// =============================================================================

static TOKENIZER: OnceLock<Arc<TokenizerHandle>> = OnceLock::new();
static EMBEDDER: OnceLock<Arc<JinaBertEmbedder>> = OnceLock::new();

fn load_tokenizer() -> Arc<TokenizerHandle> {
    TOKENIZER
        .get_or_init(|| {
            let path = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../coppermind/assets/models/jina-bert-tokenizer.json"
            );
            let bytes = std::fs::read(path).expect("Failed to read tokenizer");
            Arc::new(
                TokenizerHandle::from_bytes(bytes, MAX_POSITION_EMBEDDINGS)
                    .expect("Failed to load tokenizer"),
            )
        })
        .clone()
}

fn load_embedder() -> Arc<JinaBertEmbedder> {
    EMBEDDER
        .get_or_init(|| {
            let tokenizer = load_tokenizer();
            let path = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../coppermind/assets/models/jina-bert.safetensors"
            );
            let bytes = std::fs::read(path).expect("Failed to read model");
            Arc::new(
                JinaBertEmbedder::from_bytes(
                    bytes,
                    tokenizer.vocab_size(),
                    JinaBertConfig::default(),
                )
                .expect("Failed to load model"),
            )
        })
        .clone()
}

// =============================================================================
// Evaluation Setup
// =============================================================================

struct EvalSetup {
    keyword_engine: KeywordSearchEngine,
    doc_id_to_chunk: HashMap<String, ChunkId>,
    doc_embeddings: HashMap<ChunkId, Vec<f32>>,
    query_embeddings: HashMap<String, Vec<f32>>,
    query_texts: HashMap<String, String>,
    qrels: HashMap<String, HashMap<String, u8>>,
    query_ids: Vec<String>,
    dataset_info: DatasetInfo,
}

impl EvalSetup {
    fn build_vector_engine(&self) -> VectorSearchEngine {
        let mut engine = VectorSearchEngine::new(EMBEDDING_DIM);
        for (&chunk_id, emb) in &self.doc_embeddings {
            let _ = engine.add_chunk(chunk_id, emb.clone());
        }
        engine
    }
}

fn prepare_eval(args: &Args) -> Result<EvalSetup, String> {
    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data");
    let dataset = load_coppermind_dataset(&data_dir)?;

    let dataset_info = DatasetInfo {
        name: dataset.name.clone(),
        num_documents: dataset.num_documents(),
        num_queries: dataset.num_queries(),
        num_qrels: dataset.num_qrels(),
    };

    // Check cache
    std::fs::create_dir_all(&args.cache_dir)
        .map_err(|e| format!("Failed to create cache dir: {}", e))?;
    let cache_path = args.cache_dir.join("coppermind-embeddings.bin");

    if !args.force_recompute && cache_path.exists() {
        eprintln!("Loading embeddings from cache...");
        return load_from_cache(&cache_path, dataset);
    }

    // Compute embeddings
    eprintln!("Computing embeddings...");
    let (doc_embs, query_embs) = compute_embeddings(&dataset)?;

    // Save cache
    save_cache(&cache_path, &doc_embs, &query_embs)?;

    build_setup(dataset, doc_embs, query_embs, dataset_info)
}

fn compute_embeddings(dataset: &EvalDataset) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), String> {
    let tokenizer = load_tokenizer();
    let embedder = load_embedder();

    let pb = ProgressBar::new(dataset.documents.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40}] {pos}/{len}")
            .unwrap(),
    );
    pb.set_message("Documents");

    let mut doc_embs = Vec::with_capacity(dataset.documents.len());
    for (_, title, text) in &dataset.documents {
        let full = format!("{} {}", title, text);
        let tokens = tokenizer.tokenize(&full).map_err(|e| e.to_string())?;
        let emb = embedder.embed_tokens(tokens).map_err(|e| e.to_string())?;
        doc_embs.push(emb);
        pb.inc(1);
    }
    pb.finish();

    let pb = ProgressBar::new(dataset.queries.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40}] {pos}/{len}")
            .unwrap(),
    );
    pb.set_message("Queries");

    let mut query_embs = Vec::with_capacity(dataset.queries.len());
    for (_, text) in &dataset.queries {
        let tokens = tokenizer.tokenize(text).map_err(|e| e.to_string())?;
        let emb = embedder.embed_tokens(tokens).map_err(|e| e.to_string())?;
        query_embs.push(emb);
        pb.inc(1);
    }
    pb.finish();

    Ok((doc_embs, query_embs))
}

fn build_setup(
    dataset: EvalDataset,
    doc_embs: Vec<Vec<f32>>,
    query_embs: Vec<Vec<f32>>,
    dataset_info: DatasetInfo,
) -> Result<EvalSetup, String> {
    let mut doc_id_to_chunk = HashMap::new();
    let mut doc_embeddings = HashMap::new();
    let mut keyword_engine = KeywordSearchEngine::new();

    for (idx, (doc_id, title, text)) in dataset.documents.iter().enumerate() {
        let chunk_id = ChunkId::from_u64(idx as u64);
        doc_id_to_chunk.insert(doc_id.clone(), chunk_id);
        if let Some(emb) = doc_embs.get(idx) {
            doc_embeddings.insert(chunk_id, emb.clone());
        }
        keyword_engine.add_chunk(chunk_id, format!("{} {}", title, text));
    }

    let mut query_embeddings = HashMap::new();
    let mut query_texts = HashMap::new();
    let mut query_ids = Vec::new();

    for (idx, (query_id, text)) in dataset.queries.iter().enumerate() {
        if let Some(emb) = query_embs.get(idx) {
            query_embeddings.insert(query_id.clone(), emb.clone());
            query_texts.insert(query_id.clone(), text.clone());
            query_ids.push(query_id.clone());
        }
    }

    Ok(EvalSetup {
        keyword_engine,
        doc_id_to_chunk,
        doc_embeddings,
        query_embeddings,
        query_texts,
        qrels: dataset.qrels,
        query_ids,
        dataset_info,
    })
}

fn save_cache(
    path: &PathBuf,
    doc_embs: &[Vec<f32>],
    query_embs: &[Vec<f32>],
) -> Result<(), String> {
    use std::io::Write;
    let mut f = std::fs::File::create(path).map_err(|e| e.to_string())?;

    let header = [
        doc_embs.len() as u64,
        query_embs.len() as u64,
        EMBEDDING_DIM as u64,
    ];
    for h in &header {
        f.write_all(&h.to_le_bytes()).map_err(|e| e.to_string())?;
    }
    for emb in doc_embs {
        for &v in emb {
            f.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
        }
    }
    for emb in query_embs {
        for &v in emb {
            f.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
        }
    }
    eprintln!("Cached embeddings to {}", path.display());
    Ok(())
}

fn load_from_cache(path: &PathBuf, dataset: EvalDataset) -> Result<EvalSetup, String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;

    let mut buf = [0u8; 8];
    f.read_exact(&mut buf).map_err(|e| e.to_string())?;
    let num_docs = u64::from_le_bytes(buf) as usize;
    f.read_exact(&mut buf).map_err(|e| e.to_string())?;
    let num_queries = u64::from_le_bytes(buf) as usize;
    f.read_exact(&mut buf).map_err(|e| e.to_string())?;
    let dim = u64::from_le_bytes(buf) as usize;

    let mut doc_embs = Vec::with_capacity(num_docs);
    for _ in 0..num_docs {
        let mut emb = vec![0f32; dim];
        for v in &mut emb {
            let mut b = [0u8; 4];
            f.read_exact(&mut b).map_err(|e| e.to_string())?;
            *v = f32::from_le_bytes(b);
        }
        doc_embs.push(emb);
    }

    let mut query_embs = Vec::with_capacity(num_queries);
    for _ in 0..num_queries {
        let mut emb = vec![0f32; dim];
        for v in &mut emb {
            let mut b = [0u8; 4];
            f.read_exact(&mut b).map_err(|e| e.to_string())?;
            *v = f32::from_le_bytes(b);
        }
        query_embs.push(emb);
    }

    let dataset_info = DatasetInfo {
        name: format!("{} (cached)", dataset.name),
        num_documents: dataset.num_documents(),
        num_queries: dataset.num_queries(),
        num_qrels: dataset.num_qrels(),
    };

    build_setup(dataset, doc_embs, query_embs, dataset_info)
}

// =============================================================================
// Evaluation
// =============================================================================

fn evaluate_system(
    setup: &EvalSetup,
    name: &str,
    mut search: impl FnMut(&str) -> Vec<(ChunkId, f32)>,
    k_values: &[usize],
) -> Result<(SystemResult, HashMap<usize, BTreeMap<String, f64>>), String> {
    let max_k = *k_values.iter().max().unwrap_or(&10);

    let mut true_builder = TrueRelStoreBuilder::new();
    let mut pred_builder = PredRelStoreBuilder::new();

    for qid in &setup.query_ids {
        if let Some(qrels) = setup.qrels.get(qid) {
            for (doc_id, &rel) in qrels {
                true_builder.add_record(qid, doc_id, rel as u32).ok();
            }
        }

        let results = search(qid);
        for (rank, (chunk_id, _)) in results.iter().take(max_k).enumerate() {
            if let Some((doc_id, _)) = setup.doc_id_to_chunk.iter().find(|(_, &c)| c == *chunk_id) {
                pred_builder
                    .add_record(qid, doc_id, (1.0 / (rank + 1) as f64).into())
                    .ok();
            }
        }
    }

    let true_rels = true_builder.build();
    let pred_rels = pred_builder.build();

    let mut metrics_by_k = BTreeMap::new();
    let mut scores_by_k = HashMap::new();

    for &k in k_values {
        let ndcg = elinor::evaluate(&true_rels, &pred_rels, Metric::NDCG { k })
            .map_err(|e| e.to_string())?;
        let map = elinor::evaluate(&true_rels, &pred_rels, Metric::AP { k })
            .map_err(|e| e.to_string())?;
        let mrr = elinor::evaluate(&true_rels, &pred_rels, Metric::RR { k })
            .map_err(|e| e.to_string())?;
        let prec = elinor::evaluate(&true_rels, &pred_rels, Metric::Precision { k })
            .map_err(|e| e.to_string())?;
        let rec = elinor::evaluate(&true_rels, &pred_rels, Metric::Recall { k })
            .map_err(|e| e.to_string())?;

        metrics_by_k.insert(
            k,
            MetricScores {
                ndcg: ndcg.mean(),
                map: map.mean(),
                mrr: mrr.mean(),
                precision: prec.mean(),
                recall: rec.mean(),
            },
        );

        scores_by_k.insert(
            k,
            ndcg.scores()
                .iter()
                .map(|(k, v)| (k.to_string(), *v))
                .collect(),
        );
    }

    Ok((
        SystemResult {
            name: name.to_string(),
            metrics_by_k,
        },
        scores_by_k,
    ))
}

fn compare_systems(
    a: &str,
    scores_a: &BTreeMap<String, f64>,
    b: &str,
    scores_b: &BTreeMap<String, f64>,
    k: usize,
) -> Result<Comparison, String> {
    let pairs = pairs_from_maps(scores_a, scores_b).map_err(|e| e.to_string())?;
    let stat =
        StudentTTest::from_paired_samples(pairs.iter().copied()).map_err(|e| e.to_string())?;

    Ok(Comparison {
        system_a: a.to_string(),
        system_b: b.to_string(),
        metric: format!("NDCG@{}", k),
        k,
        p_value: stat.p_value(),
        effect_size: stat.effect_size(),
        significant: stat.p_value() < 0.05,
    })
}

fn run_ablation(
    setup: &EvalSetup,
    rrf_ks: &[usize],
    k_values: &[usize],
) -> Result<Vec<RrfKResult>, String> {
    let mut results = Vec::new();

    for &rrf_k in rrf_ks {
        eprintln!("  RRF k={}", rrf_k);
        let mut vec_engine = setup.build_vector_engine();

        let (sys, _) = evaluate_system(
            setup,
            &format!("hybrid_rrf{}", rrf_k),
            |qid| {
                let vec_res = setup
                    .query_embeddings
                    .get(qid)
                    .and_then(|e| vec_engine.search(e, 100).ok())
                    .unwrap_or_default();
                let kw_res = setup
                    .query_texts
                    .get(qid)
                    .map(|t| setup.keyword_engine.search(t, 100))
                    .unwrap_or_default();
                reciprocal_rank_fusion(&vec_res, &kw_res, rrf_k)
            },
            k_values,
        )?;

        results.push(RrfKResult {
            rrf_k,
            metrics_by_k: sys.metrics_by_k,
        });
    }

    Ok(results)
}

fn analyze_queries(
    setup: &EvalSetup,
    scores: &BTreeMap<String, HashMap<usize, BTreeMap<String, f64>>>,
    k: usize,
) -> PerQueryAnalysis {
    let mut results = Vec::new();
    let mut wins = 0;
    let mut loses = 0;

    for qid in &setup.query_ids {
        let text = setup.query_texts.get(qid).cloned().unwrap_or_default();
        let mut q_scores = BTreeMap::new();

        for (sys, sys_scores) in scores {
            if let Some(k_scores) = sys_scores.get(&k) {
                if let Some(&s) = k_scores.get(qid) {
                    q_scores.insert(sys.clone(), s);
                }
            }
        }

        let h = q_scores.get("hybrid").copied().unwrap_or(0.0);
        let v = q_scores.get("vector").copied().unwrap_or(0.0);
        let kw = q_scores.get("keyword").copied().unwrap_or(0.0);

        if h > v && h > kw {
            wins += 1;
        }
        if h < v && h < kw {
            loses += 1;
        }

        results.push(QueryResult {
            query_id: qid.clone(),
            query_text: text,
            scores: q_scores,
        });
    }

    PerQueryAnalysis {
        query_results: results,
        queries_where_hybrid_wins: wins,
        queries_where_hybrid_loses: loses,
    }
}

// =============================================================================
// Output
// =============================================================================

fn print_report(report: &EvalReport) {
    println!("\n{}", "=".repeat(80));
    println!("COPPERMIND SEARCH QUALITY EVALUATION");
    println!("{}", "=".repeat(80));
    println!(
        "\nDataset: {} ({} docs, {} queries, {} qrels)",
        report.dataset.name,
        report.dataset.num_documents,
        report.dataset.num_queries,
        report.dataset.num_qrels
    );

    for &k in &report.k_values {
        println!("\n{}", "-".repeat(70));
        println!("RESULTS @ k={}", k);
        println!(
            "{:<12} {:>8} {:>8} {:>8} {:>8} {:>8}",
            "System", "NDCG", "MAP", "MRR", "Prec", "Recall"
        );
        for sys in &report.systems {
            if let Some(m) = sys.metrics_by_k.get(&k) {
                println!(
                    "{:<12} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4}",
                    sys.name, m.ndcg, m.map, m.mrr, m.precision, m.recall
                );
            }
        }
    }

    if !report.comparisons.is_empty() {
        println!("\n{}", "-".repeat(70));
        println!("STATISTICAL COMPARISONS (* = p < 0.05)");
        for c in &report.comparisons {
            let sig = if c.significant { "*" } else { "" };
            println!(
                "{} vs {} ({}): p={:.4}{} effect={:.3}",
                c.system_a, c.system_b, c.metric, c.p_value, sig, c.effect_size
            );
        }
    }

    if let Some(abl) = &report.ablations {
        println!("\n{}", "-".repeat(70));
        println!("RRF ABLATION (NDCG@10)");
        for r in &abl.rrf_k_sweep {
            if let Some(m) = r.metrics_by_k.get(&10) {
                println!("  k={:<3} NDCG={:.4}", r.rrf_k, m.ndcg);
            }
        }
    }

    if let Some(pq) = &report.per_query {
        println!("\n{}", "-".repeat(70));
        println!("PER-QUERY ANALYSIS");
        println!(
            "Hybrid wins: {}/{}",
            pq.queries_where_hybrid_wins,
            pq.query_results.len()
        );
        println!(
            "Hybrid loses: {}/{}",
            pq.queries_where_hybrid_loses,
            pq.query_results.len()
        );
    }

    println!("{}\n", "=".repeat(80));
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<(), String> {
    let args = Args::parse();
    let setup = prepare_eval(&args)?;

    let k_values = args
        .k_values
        .clone()
        .unwrap_or_else(|| DEFAULT_K_VALUES.to_vec());
    let mut systems = Vec::new();
    let mut all_scores: BTreeMap<String, HashMap<usize, BTreeMap<String, f64>>> = BTreeMap::new();

    // Vector search
    if !args.skip_vector {
        eprintln!("\nEvaluating vector search...");
        let mut engine = setup.build_vector_engine();
        let (sys, scores) = evaluate_system(
            &setup,
            "vector",
            |qid| {
                setup
                    .query_embeddings
                    .get(qid)
                    .and_then(|e| engine.search(e, 100).ok())
                    .unwrap_or_default()
            },
            &k_values,
        )?;
        all_scores.insert("vector".into(), scores);
        systems.push(sys);
    }

    // Keyword search
    if !args.skip_keyword {
        eprintln!("Evaluating keyword search...");
        let (sys, scores) = evaluate_system(
            &setup,
            "keyword",
            |qid| {
                setup
                    .query_texts
                    .get(qid)
                    .map(|t| setup.keyword_engine.search(t, 100))
                    .unwrap_or_default()
            },
            &k_values,
        )?;
        all_scores.insert("keyword".into(), scores);
        systems.push(sys);
    }

    // Hybrid search
    if !args.skip_hybrid {
        eprintln!("Evaluating hybrid search...");
        let mut engine = setup.build_vector_engine();
        let (sys, scores) = evaluate_system(
            &setup,
            "hybrid",
            |qid| {
                let vec = setup
                    .query_embeddings
                    .get(qid)
                    .and_then(|e| engine.search(e, 100).ok())
                    .unwrap_or_default();
                let kw = setup
                    .query_texts
                    .get(qid)
                    .map(|t| setup.keyword_engine.search(t, 100))
                    .unwrap_or_default();
                reciprocal_rank_fusion(&vec, &kw, 60)
            },
            &k_values,
        )?;
        all_scores.insert("hybrid".into(), scores);
        systems.push(sys);
    }

    // Comparisons
    let cmp_k = if k_values.contains(&10) {
        10
    } else {
        k_values[0]
    };
    let mut comparisons = Vec::new();

    if let (Some(h), Some(v)) = (all_scores.get("hybrid"), all_scores.get("vector")) {
        if let (Some(hs), Some(vs)) = (h.get(&cmp_k), v.get(&cmp_k)) {
            comparisons.push(compare_systems("hybrid", hs, "vector", vs, cmp_k)?);
        }
    }
    if let (Some(h), Some(k)) = (all_scores.get("hybrid"), all_scores.get("keyword")) {
        if let (Some(hs), Some(ks)) = (h.get(&cmp_k), k.get(&cmp_k)) {
            comparisons.push(compare_systems("hybrid", hs, "keyword", ks, cmp_k)?);
        }
    }

    // Ablation
    let ablations = match args.ablation {
        Some(AblationType::Rrf | AblationType::All) => {
            eprintln!("\nRunning RRF ablation...");
            let rrf_ks = args
                .rrf_k_values
                .clone()
                .unwrap_or_else(|| DEFAULT_RRF_K_VALUES.to_vec());
            Some(AblationResults {
                rrf_k_sweep: run_ablation(&setup, &rrf_ks, &k_values)?,
            })
        }
        None => None,
    };

    // Per-query
    let per_query = if args.per_query {
        Some(analyze_queries(&setup, &all_scores, cmp_k))
    } else {
        None
    };

    let report = EvalReport {
        dataset: setup.dataset_info,
        k_values,
        systems,
        comparisons,
        ablations,
        per_query,
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    } else {
        print_report(&report);
    }

    Ok(())
}
