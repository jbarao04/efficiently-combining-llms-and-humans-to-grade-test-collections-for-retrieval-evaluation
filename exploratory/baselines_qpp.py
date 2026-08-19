# Classical QPP baselines (NQC, WIG, Clarity) using BM25 retrieval via Pyserini
import os
os.environ["OPENAI_API_KEY"] = "dummy"

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from tqdm import tqdm

import traceback
try:
    from pyserini.search.lucene import LuceneSearcher
    from pyserini.index.lucene import LuceneIndexReader
except Exception:
    print("Pyserini import failed.")
    print("Actual traceback:")
    traceback.print_exc()
    sys.exit(1)

# Pre-built index for MS MARCO v1 passage collection.
# This downloads ~2 GB on first use and caches locally.
PREBUILT_INDEX = "msmarco-v1-passage"

# QPP feature cutoffs
CUTOFFS = [10, 50, 100]

# nDCG cutoff
NDCG_K = 10

# Number of documents to retrieve per query
RETRIEVAL_DEPTH = 1000


# -----------------------------------------------------------------------
# Data loading (reused from verify_and_load.py)
# -----------------------------------------------------------------------
def load_qrels(path: Path) -> dict:
    """Load qrels: {qid: {pid: grade}}"""
    qrels = defaultdict(dict)
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                qid, _, pid, grade = parts
                qrels[qid][pid] = int(grade)
    return dict(qrels)


def load_queries(path: Path) -> dict:
    """Load queries: {qid: text}"""
    queries = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                queries[parts[0]] = parts[1]
    return queries


# -----------------------------------------------------------------------
# nDCG computation
# -----------------------------------------------------------------------
def dcg_at_k(relevances: list, k: int) -> float:
    """Compute DCG@k from a list of relevance grades in rank order."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2**rel - 1) / math.log2(i + 2)  # i+2 because rank starts at 1
    return dcg


def ndcg_at_k(ranked_pids: list, qrels: dict, k: int) -> float:
    """
    Compute nDCG@k for a ranked list of passage IDs against qrels.

    Parameters:
        ranked_pids: passage IDs in rank order (best first)
        qrels: {pid: relevance_grade} for this query
        k: cutoff
    """
    # Actual relevances in the given ranking
    relevances = [qrels.get(pid, 0) for pid in ranked_pids[:k]]
    actual_dcg = dcg_at_k(relevances, k)

    # Ideal relevances: all qrels sorted descending
    ideal_relevances = sorted(qrels.values(), reverse=True)
    ideal_dcg = dcg_at_k(ideal_relevances, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


# -----------------------------------------------------------------------
# QPP features
# -----------------------------------------------------------------------
def compute_nqc(scores: list, k: int, corpus_score: float) -> float:
    """
    Normalized Query Commitment (Shtok et al., 2012).

    NQC measures the spread of retrieval scores in the top-k.
    A narrow spread suggests the system cannot distinguish relevant
    from non-relevant documents, predicting poor performance.

    NQC(q) = std(scores[:k]) / |corpus_score|
    """
    if k > len(scores) or corpus_score == 0:
        return 0.0
    top_scores = np.array(scores[:k])
    return float(np.std(top_scores) / abs(corpus_score))


def compute_wig(scores: list, k: int, corpus_score: float) -> float:
    """
    Weighted Information Gain (Zhou and Croft, 2007).

    WIG measures how much the top-k scores stand above the corpus average.
    Higher values predict better retrieval performance.

    WIG(q) = (1/sqrt(k)) * mean(scores[:k] - corpus_score)
    """
    if k > len(scores):
        return 0.0
    top_scores = np.array(scores[:k])
    return float((np.mean(top_scores) - corpus_score) / math.sqrt(k))


def compute_clarity(
    searcher: LuceneSearcher,
    index_reader: LuceneIndexReader,
    query_text: str,
    hit_pids: list,
    k: int,
    mu: float = 1000.0,
) -> float:
    """
    Clarity Score (Cronen-Townsend et al., 2002).

    Estimates the KL divergence between a query language model
    (built from top-k retrieved documents) and the collection
    language model. Higher divergence predicts better performance
    because the query is more "specific" relative to the collection.

    Uses Dirichlet-smoothed document language models with parameter mu.

    Clarity(q) = sum_w P(w|theta_q) * log2(P(w|theta_q) / P(w|C))

    Parameters:
        searcher: Pyserini searcher for fetching document text
        index_reader: Pyserini index reader for collection statistics
        query_text: raw query string
        hit_pids: list of retrieved passage IDs
        k: number of top documents to use
        mu: Dirichlet smoothing parameter
    """
    top_pids = hit_pids[:k]
    if not top_pids:
        return 0.0

    total_terms_in_collection = index_reader.stats()["total_terms"]
    if total_terms_in_collection == 0:
        return 0.0

    # Build term frequencies from top-k documents
    query_model = Counter()
    total_tokens = 0

    for pid in top_pids:
        doc = searcher.doc(pid)
        if doc is None:
            continue
        # Get raw text and tokenize simply (lowercase, split on non-alpha)
        raw = doc.raw()
        if isinstance(raw, str):
            try:
                raw_obj = json.loads(raw)
                text = raw_obj.get("contents", raw_obj.get("body", ""))
            except (json.JSONDecodeError, AttributeError):
                text = raw
        else:
            text = str(raw)

        # Simple whitespace tokenization (matches Lucene's basic behaviour)
        tokens = text.lower().split()
        query_model.update(tokens)
        total_tokens += len(tokens)

    if total_tokens == 0:
        return 0.0

    # Compute KL divergence: P(w|theta_q) vs P(w|C)
    clarity = 0.0
    for term, tf in query_model.items():
        # P(w|theta_q): MLE from top-k documents
        p_w_query = tf / total_tokens

        # P(w|C): collection probability
        # Use index reader to get collection term frequency
        try:
            cf = index_reader.get_term_counts(term, analyzer=None)
            if cf is None or cf == 0:
                cf = 0.5  # smoothing for unseen terms
            elif isinstance(cf, tuple):
                cf = cf[1]  # (df, cf) tuple
        except Exception:
            cf = 0.5

        p_w_collection = cf / total_terms_in_collection

        if p_w_query > 0 and p_w_collection > 0:
            clarity += p_w_query * math.log2(p_w_query / p_w_collection)

    return float(clarity)


# -----------------------------------------------------------------------
# Correlation and confidence intervals
# -----------------------------------------------------------------------
def kendall_tau_with_ci(x: list, y: list, confidence: float = 0.95):
    """
    Compute Kendall's tau-b with confidence interval.

    Uses the asymptotic normal approximation for the CI:
    SE(tau) ≈ sqrt(2(2n+5) / 9n(n-1))
    """
    n = len(x)
    tau, p_value = stats.kendalltau(x, y)

    # Standard error (asymptotic approximation)
    se = math.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1)))
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    ci_low = tau - z * se
    ci_high = tau + z * se

    return {
        "tau": round(tau, 4),
        "p_value": round(p_value, 6),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "n": n,
    }


# -----------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Classical QPP baselines via BM25")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data_prep/data/trec-dl",
        help="TREC DL v1 data directory (default: ./data_prep/data/trec-dl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/baselines",
        help="Output directory for results (default: ./results/baselines)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # Step 1: Load queries and qrels
    # -------------------------------------------------------------------
    print("=== Step 1: Loading queries and qrels ===\n")

    all_queries = {}
    all_qrels = {}

    for year in ["2019", "2020"]:
        year_dir = data_dir / year
        queries = load_queries(year_dir / "queries.tsv")
        qrels = load_qrels(year_dir / "qrels.txt")

        # Keep only queries that have qrels
        for qid in qrels:
            if qid in queries:
                all_queries[qid] = queries[qid]
                all_qrels[qid] = qrels[qid]

        print(f"  {year}: {len(qrels)} judged queries loaded")

    print(f"  Total: {len(all_queries)} queries\n")

    # -------------------------------------------------------------------
    # Step 2: Initialize Pyserini
    # -------------------------------------------------------------------
    print("=== Step 2: Initializing Pyserini ===\n")
    print(f"  Loading pre-built index: {PREBUILT_INDEX}")
    print("  (first run downloads ~2 GB, subsequent runs use cache)\n")

    searcher = LuceneSearcher.from_prebuilt_index(PREBUILT_INDEX)
    index_reader = LuceneIndexReader.from_prebuilt_index(PREBUILT_INDEX)

    index_stats = index_reader.stats()
    print(f"  Index loaded: {index_stats['documents']:,} documents")
    print(f"  Total terms:  {index_stats['total_terms']:,}\n")

    # -------------------------------------------------------------------
    # Step 3: BM25 retrieval
    # -------------------------------------------------------------------
    print(f"=== Step 3: BM25 retrieval (top {RETRIEVAL_DEPTH}) ===\n")

    retrieval_results = {}  # {qid: [(pid, score), ...]}

    for qid in tqdm(sorted(all_queries.keys()), desc="Retrieving"):
        query_text = all_queries[qid]
        hits = searcher.search(query_text, k=RETRIEVAL_DEPTH)
        retrieval_results[qid] = [(hit.docid, hit.score) for hit in hits]

    print(f"\n  Retrieved results for {len(retrieval_results)} queries\n")

    # -------------------------------------------------------------------
    # Step 4: Compute nDCG@10 for BM25
    # -------------------------------------------------------------------
    print(f"=== Step 4: Computing nDCG@{NDCG_K} ===\n")

    ndcg_scores = {}
    for qid, results in retrieval_results.items():
        ranked_pids = [pid for pid, _ in results]
        ndcg_scores[qid] = ndcg_at_k(ranked_pids, all_qrels[qid], NDCG_K)

    ndcg_values = list(ndcg_scores.values())
    print(f"  Mean nDCG@{NDCG_K}: {np.mean(ndcg_values):.4f}")
    print(f"  Std nDCG@{NDCG_K}:  {np.std(ndcg_values):.4f}")
    print(f"  Min: {np.min(ndcg_values):.4f}  Max: {np.max(ndcg_values):.4f}\n")

    # -------------------------------------------------------------------
    # Step 5: Compute QPP features
    # -------------------------------------------------------------------
    print("=== Step 5: Computing QPP features ===\n")

    # Compute corpus score estimate per query (mean score at deep cutoff)
    qpp_features = {}

    for qid in tqdm(sorted(all_queries.keys()), desc="QPP features"):
        results = retrieval_results[qid]
        scores = [score for _, score in results]

        if not scores:
            continue

        # Corpus score estimate: mean of all retrieved scores
        corpus_score = np.mean(scores)

        features = {}

        # NQC and WIG at multiple cutoffs
        for k in CUTOFFS:
            features[f"nqc_{k}"] = compute_nqc(scores, k, corpus_score)
            features[f"wig_{k}"] = compute_wig(scores, k, corpus_score)

        # Clarity at k=10 (most expensive, only at one cutoff)
        hit_pids = [pid for pid, _ in results]
        features["clarity_10"] = compute_clarity(
            searcher, index_reader, all_queries[qid], hit_pids, k=10
        )

        # Additional simple distribution features (useful later for comparison)
        top_10_scores = np.array(scores[:10])
        features["max_score"] = float(scores[0]) if scores else 0.0
        features["score_range_10"] = float(top_10_scores.max() - top_10_scores.min())
        features["score_entropy_10"] = float(
            stats.entropy(top_10_scores - top_10_scores.min() + 1e-10)
        )

        qpp_features[qid] = features

    print(f"\n  Computed features for {len(qpp_features)} queries\n")

    # -------------------------------------------------------------------
    # Step 6: Correlation analysis
    # -------------------------------------------------------------------
    print("=== Step 6: Correlation analysis (Kendall's tau) ===\n")

    # Get aligned arrays
    qids = sorted(qpp_features.keys())
    y = [ndcg_scores[qid] for qid in qids]

    feature_names = sorted(qpp_features[qids[0]].keys())
    correlations = {}

    print(f"  {'Feature':<20} {'tau':>8} {'p-value':>10} {'95% CI':>20}")
    print(f"  {'-'*60}")

    for fname in feature_names:
        x = [qpp_features[qid][fname] for qid in qids]
        result = kendall_tau_with_ci(x, y)
        correlations[fname] = result

        ci_str = f"[{result['ci_low']:.4f}, {result['ci_high']:.4f}]"
        sig = "*" if result["p_value"] < 0.05 else " "
        print(
            f"  {fname:<20} {result['tau']:>8.4f} {result['p_value']:>10.6f} "
            f"{ci_str:>20} {sig}"
        )

    # -------------------------------------------------------------------
    # Step 7: Save results
    # -------------------------------------------------------------------
    print(f"\n=== Step 7: Saving results ===\n")

    # Per-query CSV
    csv_path = output_dir / "bm25_qpp_per_query.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["query_id", f"ndcg_{NDCG_K}"] + feature_names
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for qid in qids:
            row = {"query_id": qid, f"ndcg_{NDCG_K}": round(ndcg_scores[qid], 6)}
            row.update(
                {k: round(v, 6) for k, v in qpp_features[qid].items()}
            )
            writer.writerow(row)

    print(f"  Per-query results: {csv_path}")

    # Summary JSON
    summary = {
        "description": "Classical QPP baselines: BM25 + NQC/WIG/Clarity on TREC DL 2019+2020",
        "index": PREBUILT_INDEX,
        "retrieval_depth": RETRIEVAL_DEPTH,
        "ndcg_cutoff": NDCG_K,
        "num_queries": len(qids),
        "ndcg_summary": {
            "mean": round(float(np.mean(y)), 4),
            "std": round(float(np.std(y)), 4),
            "min": round(float(np.min(y)), 4),
            "max": round(float(np.max(y)), 4),
        },
        "correlations": correlations,
    }

    summary_path = output_dir / "bm25_qpp_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Summary:           {summary_path}")

    # BM25 run file in TREC format (useful for later comparison)
    run_path = output_dir / "bm25_run.txt"
    with open(run_path, "w", encoding="utf-8") as f:
        for qid in sorted(retrieval_results.keys()):
            for rank, (pid, score) in enumerate(retrieval_results[qid], 1):
                f.write(f"{qid} Q0 {pid} {rank} {score:.6f} BM25\n")

    print(f"  BM25 run file:     {run_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
