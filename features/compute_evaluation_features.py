# Compute primary evaluation features: per-query Spearman and nDCG with tie-shuffle averaging

import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# -----------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SCORES_V1 = BASE_DIR / "results" / "scoring" / "normal_scores" / "scores_v1.jsonl"
SCORES_V2 = BASE_DIR / "results" / "scoring" / "normal_scores" / "scores_v2.jsonl"

QRELS_FILES = {
    2019: BASE_DIR / "data_prep" / "data" / "trec-dl" / "2019" / "qrels.txt",
    2020: BASE_DIR / "data_prep" / "data" / "trec-dl" / "2020" / "qrels.txt",
    2021: BASE_DIR / "data_prep" / "data" / "trec-dl-v2" / "2021" / "qrels_dedup.txt",
    2022: BASE_DIR / "data_prep" / "data" / "trec-dl-v2" / "2022" / "qrels_dedup.txt",
    2023: BASE_DIR / "data_prep" / "data" / "trec-dl-v2" / "2023" / "qrels_dedup.txt",
}

BM25_BASELINES = {"NQC": 0.338, "WIG": 0.300}

OUTPUT_DIR = BASE_DIR / "results" / "level2"
NDCG_CUTOFFS = [10, 20, 50, 100]
N_SHUFFLES = 100
SEED_BASE = 42
N_BOOTSTRAP = 10000

FEATURES = [
    "score_entropy", "score_std", "frac_zero", "top_heaviness",
    "discriminability_gap", "mean_max_prob", "mean_token_entropy", "mean_prob_margin",
]

FEATURE_META = {
    "score_entropy":        {"family": "A"},
    "score_std":            {"family": "A"},
    "frac_zero":            {"family": "A"},
    "top_heaviness":        {"family": "A"},
    "discriminability_gap": {"family": "A"},
    "mean_max_prob":        {"family": "C"},
    "mean_token_entropy":   {"family": "C"},
    "mean_prob_margin":     {"family": "C"},
}


# -----------------------------------------------------------------------
# Part 1: Data loading
# -----------------------------------------------------------------------
def load_qrels(path: Path) -> dict:
    """Load qrels: {qid: {pid: grade}}."""
    qrels = defaultdict(dict)
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                qid, _, pid, grade = parts
                qrels[str(qid)][str(pid)] = int(grade)
    return dict(qrels)


def load_scores(path: Path) -> list[dict]:
    """Load a JSONL scores file into a list of dicts."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_data():
    """Load and merge all scores with qrels. Returns (DataFrame, qid_to_year)."""
    print("Part 1: Loading data...")

    scores = load_scores(SCORES_V1) + load_scores(SCORES_V2)
    print(f"  Loaded {len(scores)} score rows ({SCORES_V1.name} + {SCORES_V2.name})")

    all_qrels = {}
    qid_to_year = {}
    for year, path in QRELS_FILES.items():
        qrels = load_qrels(path)
        for qid, pids in qrels.items():
            qid_to_year[str(qid)] = year
            for pid, grade in pids.items():
                all_qrels[(str(qid), str(pid))] = grade
        print(f"  Loaded {year} qrels: {len(qrels)} queries")

    rows = []
    for s in scores:
        qid = str(s["query_id"])
        pid = str(s["passage_id"])
        key = (qid, pid)
        if key in all_qrels:
            rows.append({
                "query_id": qid,
                "passage_id": pid,
                "score": int(s["score"]),
                "prob_0": float(s["probs"]["0"]),
                "prob_1": float(s["probs"]["1"]),
                "prob_2": float(s["probs"]["2"]),
                "prob_3": float(s["probs"]["3"]),
                "human_grade": all_qrels[key],
            })

    df = pd.DataFrame(rows)
    n_queries = df["query_id"].nunique()
    n_unmatched = len(scores) - len(df)

    print(f"  Merged: {len(df)} pairs, {n_queries} queries, {n_unmatched} unmatched")
    assert n_queries == 308, f"Expected 308 queries, got {n_queries}"
    assert n_unmatched == 0, f"Expected 0 unmatched, got {n_unmatched}"

    return df, qid_to_year


# -----------------------------------------------------------------------
# Part 2: Primary target — per-query Spearman
# -----------------------------------------------------------------------
def compute_spearman(df: pd.DataFrame) -> dict:
    """Compute per-query Spearman correlation. Returns {qid: rho}."""
    print("Part 2: Computing per-query Spearman...")
    results = {}
    for qid, group in df.groupby("query_id"):
        if group["score"].nunique() < 2 or group["human_grade"].nunique() < 2:
            results[qid] = 0.0
        else:
            rho, _ = stats.spearmanr(group["score"].values, group["human_grade"].values)
            results[qid] = float(rho) if not np.isnan(rho) else 0.0
    return results


# -----------------------------------------------------------------------
# Part 3: Secondary target — nDCG@k at multiple cutoffs
# -----------------------------------------------------------------------
def dcg_at_k(relevances: list, k: int) -> float:
    """DCG@k from a list of relevance grades in rank order."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2 ** rel - 1) / math.log2(i + 2)
    return dcg


def ndcg_at_k(ranked_rels: list, all_rels: list, k: int) -> float:
    """nDCG@k given ranked relevances and all relevances for ideal."""
    ideal_rels = sorted(all_rels, reverse=True)
    idcg = dcg_at_k(ideal_rels, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked_rels, k) / idcg


def expected_score(prob_0, prob_1, prob_2, prob_3):
    """Logit-weighted expected score: E[s] = sum(g * p_g)."""
    return 0 * prob_0 + 1 * prob_1 + 2 * prob_2 + 3 * prob_3


def compute_ndcg_all_cutoffs(df: pd.DataFrame) -> dict:
    """Compute per-query nDCG@k for all cutoffs using tie-shuffle averaging.

    Returns {qid: {10: val, 20: val, 50: val, 100: val}}.
    Also returns per-query within-shuffle std for nDCG@10 (for tie-dominance).
    """
    print("Part 3: Computing nDCG@k (tie-shuffle averaging, 100 shuffles)...")
    grouped = df.groupby("query_id")
    total = len(grouped)

    ndcg_results = {}
    ndcg10_within_std = {}

    for idx, (qid, group) in enumerate(grouped, 1):
        if idx % 50 == 0 or idx == total:
            print(f"  Query {idx}/{total}", end="\r")

        scores_arr = group["score"].values
        grades_arr = group["human_grade"].values
        all_rels = grades_arr.tolist()
        n = len(scores_arr)

        # Collect nDCG per shuffle per cutoff
        per_shuffle = {k: [] for k in NDCG_CUTOFFS}
        for i in range(N_SHUFFLES):
            rng = np.random.RandomState(SEED_BASE + i)
            noise = rng.random(n)
            order = np.lexsort((noise, -scores_arr))
            ranked_rels = grades_arr[order].tolist()
            for k in NDCG_CUTOFFS:
                per_shuffle[k].append(ndcg_at_k(ranked_rels, all_rels, k))

        ndcg_results[qid] = {k: float(np.mean(per_shuffle[k])) for k in NDCG_CUTOFFS}
        ndcg10_within_std[qid] = float(np.std(per_shuffle[10]))

    print()
    return ndcg_results, ndcg10_within_std


# -----------------------------------------------------------------------
# Part 4: Tie-dominance analysis
# -----------------------------------------------------------------------
def compute_tie_dominance(df: pd.DataFrame, ndcg10_within_std: dict, ndcg10_cross_std: float):
    """Analyze how ties dominate the ranking. Returns summary dict for JSON output."""
    print("Part 4: Analyzing tie dominance...")

    query_stats = []
    for qid, group in df.groupby("query_id"):
        scores = group["score"].values
        n = len(scores)
        score_counts = np.array([np.sum(scores == g) for g in range(4)])
        largest_tie = int(score_counts.max())
        n_distinct = len(np.unique(scores))

        # Top-10 tie analysis
        rng = np.random.RandomState(42)
        noise = rng.random(n)
        order = np.lexsort((noise, -scores))
        top10_scores = scores[order[:min(10, n)]]
        if len(top10_scores) >= 10:
            _, counts_top10 = np.unique(top10_scores, return_counts=True)
            top10_in_ties = int(sum(c for c in counts_top10 if c > 1))
        else:
            top10_in_ties = 0

        query_stats.append({
            "n_passages": n,
            "largest_tie_group": largest_tie,
            "n_distinct_scores": n_distinct,
            "top10_in_ties": top10_in_ties,
        })

    qs = pd.DataFrame(query_stats)
    n_all_tied = int((qs["top10_in_ties"] == 10).sum())
    pct_all_tied = n_all_tied / len(qs) * 100

    within_std_vals = np.array(list(ndcg10_within_std.values()))
    noise_ratio = within_std_vals.mean() / ndcg10_cross_std if ndcg10_cross_std > 0 else float("inf")

    # Print
    print(f"\n  Tie structure across {len(qs)} queries:")
    for col in ["n_passages", "largest_tie_group", "n_distinct_scores"]:
        vals = qs[col]
        print(f"    {col:<25} mean={vals.mean():.1f}  median={vals.median():.1f}  "
              f"min={vals.min():.0f}  max={vals.max():.0f}")

    print(f"\n  Top-10 tie analysis:")
    print(f"    Mean positions determined by tie-breaking: {qs['top10_in_ties'].mean():.1f}/10")
    print(f"    Queries with ALL 10 positions tied: {n_all_tied}/{len(qs)} ({pct_all_tied:.1f}%)")

    print(f"\n  nDCG@10 noise analysis:")
    print(f"    Mean within-query std (across 100 shuffles): {within_std_vals.mean():.4f}")
    print(f"    Cross-query std of nDCG@10:                  {ndcg10_cross_std:.4f}")
    print(f"    Noise ratio (within / cross):                {noise_ratio:.4f}")
    print(f"    => nDCG@10 is {'dominated by' if noise_ratio > 0.3 else 'partially affected by'} tie-breaking noise")

    return {
        "mean_largest_tie_group": round(float(qs["largest_tie_group"].mean()), 1),
        "mean_distinct_scores": round(float(qs["n_distinct_scores"].mean()), 2),
        "mean_top10_tied_positions": round(float(qs["top10_in_ties"].mean()), 1),
        "queries_all_10_tied": n_all_tied,
        "pct_all_10_tied": round(pct_all_tied, 1),
        "mean_within_query_std_ndcg10": round(float(within_std_vals.mean()), 4),
        "cross_query_std_ndcg10": round(ndcg10_cross_std, 4),
        "noise_ratio": round(noise_ratio, 4),
    }


# -----------------------------------------------------------------------
# Part 5: Family A features (score-based)
# -----------------------------------------------------------------------
def entropy_from_counts(counts: np.ndarray) -> float:
    """Entropy (base 2) from a count array."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def compute_family_a(df: pd.DataFrame) -> dict:
    """Compute Family A features per query. Returns {qid: dict of features}."""
    print("Part 5: Computing Family A features...")
    results = {}

    for qid, group in df.groupby("query_id"):
        scores = group["score"].values
        counts = np.array([np.sum(scores == g) for g in range(4)])
        n = len(scores)

        score_entropy = entropy_from_counts(counts)
        score_std = float(np.std(scores))
        frac_zero = float(counts[0] / n)
        top_heaviness = float(counts[3] / n)

        # Discriminability gap: use logit-weighted expected score for tie-breaking
        g = group.copy()
        g["expected_score"] = expected_score(g["prob_0"], g["prob_1"], g["prob_2"], g["prob_3"])
        g = g.sort_values(["score", "expected_score"], ascending=[False, False])
        if n >= 10:
            top_scores = g["score"].values[:10]
            rest_scores = g["score"].values[10:]
            discriminability_gap = float(np.mean(top_scores) - np.mean(rest_scores))
        else:
            discriminability_gap = 0.0

        results[qid] = {
            "score_entropy": score_entropy,
            "score_std": score_std,
            "frac_zero": frac_zero,
            "top_heaviness": top_heaviness,
            "discriminability_gap": discriminability_gap,
        }

    return results


# -----------------------------------------------------------------------
# Part 6: Family C features (probability-based)
# -----------------------------------------------------------------------
def compute_family_c(df: pd.DataFrame) -> dict:
    """Compute Family C features per query. Returns {qid: dict of features}."""
    print("Part 6: Computing Family C features...")
    results = {}

    for qid, group in df.groupby("query_id"):
        probs = group[["prob_0", "prob_1", "prob_2", "prob_3"]].values  # (n, 4)

        # mean_max_prob
        max_probs = np.max(probs, axis=1)
        mean_max_prob = float(np.mean(max_probs))

        # mean_token_entropy
        log_probs = np.zeros_like(probs)
        mask = probs > 0
        log_probs[mask] = np.log2(probs[mask])
        per_passage_entropy = -np.sum(probs * log_probs, axis=1)
        mean_token_entropy = float(np.mean(per_passage_entropy))

        # mean_prob_margin
        sorted_probs = np.sort(probs, axis=1)[:, ::-1]
        margins = sorted_probs[:, 0] - sorted_probs[:, 1]
        mean_prob_margin = float(np.mean(margins))

        results[qid] = {
            "mean_max_prob": mean_max_prob,
            "mean_token_entropy": mean_token_entropy,
            "mean_prob_margin": mean_prob_margin,
        }

    return results


# -----------------------------------------------------------------------
# Part 7: Level 2 evaluation — Kendall tau
# -----------------------------------------------------------------------
def kendall_tau_with_bootstrap(feature_vals: np.ndarray, target_vals: np.ndarray,
                                n_bootstrap: int = N_BOOTSTRAP) -> dict:
    """Compute Kendall tau-b with bootstrap 95% CI."""
    tau, pvalue = stats.kendalltau(feature_vals, target_vals)

    rng = np.random.RandomState(42)
    n = len(feature_vals)
    boot_taus = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        bt, _ = stats.kendalltau(feature_vals[idx], target_vals[idx])
        boot_taus[b] = bt

    ci_low = float(np.percentile(boot_taus, 2.5))
    ci_high = float(np.percentile(boot_taus, 97.5))

    return {
        "tau": float(tau),
        "p_value": float(pvalue),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def evaluate_features(query_df: pd.DataFrame):
    """Compute Kendall tau for each feature against primary and secondary targets.

    Returns (primary_results, secondary_results).
    """
    print("Part 7: Computing Kendall tau...")
    spearman_vals = query_df["spearman"].values

    # Primary: features vs Spearman (with bootstrap CIs)
    print("  Primary evaluation (vs Spearman, with bootstrap CIs)...")
    primary_results = []
    for i, feat in enumerate(FEATURES, 1):
        print(f"    Feature {i}/{len(FEATURES)}: {feat}", end="\r")
        feat_vals = query_df[feat].values
        r = kendall_tau_with_bootstrap(feat_vals, spearman_vals)
        r["feature"] = feat
        r["family"] = FEATURE_META[feat]["family"]
        primary_results.append(r)
    print()

    # Secondary: features vs nDCG@k (point estimates only)
    print("  Secondary evaluation (vs nDCG@k)...")
    secondary_results = []
    for feat in FEATURES:
        feat_vals = query_df[feat].values
        row = {"feature": feat, "family": FEATURE_META[feat]["family"]}
        for k in NDCG_CUTOFFS:
            col = f"ndcg_{k}"
            tau, pval = stats.kendalltau(feat_vals, query_df[col].values)
            row[f"tau_{k}"] = float(tau)
            row[f"p_{k}"] = float(pval)
        secondary_results.append(row)

    return primary_results, secondary_results


# -----------------------------------------------------------------------
# Part 8: Output
# -----------------------------------------------------------------------
def print_results(query_df: pd.DataFrame, primary_results: list[dict],
                  secondary_results: list[dict], tie_stats: dict):
    """Print all results to stdout."""

    # --- Spearman summary ---
    print("\n" + "=" * 70)
    print("  PRIMARY TARGET: Per-query Spearman correlation")
    print("=" * 70)
    sp = query_df["spearman"]
    print(f"  Mean:   {sp.mean():.4f}")
    print(f"  Std:    {sp.std():.4f}")
    print(f"  Min:    {sp.min():.4f}")
    print(f"  Max:    {sp.max():.4f}")
    print(f"  Median: {sp.median():.4f}")

    print("\n  Per-year breakdown:")
    for year in sorted(query_df["year"].unique()):
        subset = query_df[query_df["year"] == year]
        print(f"    {year}: mean={subset['spearman'].mean():.4f}  (n={len(subset)})")

    # --- nDCG summary ---
    print("\n" + "=" * 70)
    print("  SECONDARY TARGET: nDCG@k (tie-shuffle averaging)")
    print("=" * 70)
    for k in NDCG_CUTOFFS:
        col = f"ndcg_{k}"
        vals = query_df[col]
        r_sp, _ = stats.pearsonr(query_df["spearman"].values, vals.values)
        print(f"  nDCG@{k:<4} mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"Pearson r(Spearman, nDCG@{k})={r_sp:.4f}")

    # --- Tie dominance ---
    print("\n" + "=" * 70)
    print("  TIE-DOMINANCE ANALYSIS (why nDCG@10 fails as target)")
    print("=" * 70)
    print(f"  Mean top-10 positions determined by tie-breaking: "
          f"{tie_stats['mean_top10_tied_positions']}/10")
    print(f"  Queries with ALL 10 positions tied: "
          f"{tie_stats['queries_all_10_tied']}/308 ({tie_stats['pct_all_10_tied']}%)")
    print(f"  Mean within-query std of nDCG@10:   {tie_stats['mean_within_query_std_ndcg10']}")
    print(f"  Cross-query std of nDCG@10:         {tie_stats['cross_query_std_ndcg10']}")
    print(f"  Noise ratio (within/cross):         {tie_stats['noise_ratio']}")

    # --- Table 1: Primary (vs Spearman) ---
    print("\n" + "=" * 70)
    print("  TABLE 1: Kendall tau vs per-query Spearman (PRIMARY)")
    print("=" * 70)

    sorted_primary = sorted(primary_results, key=lambda x: abs(x["tau"]), reverse=True)

    header = f"{'Feature':<25} {'Family':>6} {'Tau':>8} {'CI_low':>8} {'CI_high':>8} {'p-value':>12}"
    print(header)
    print("-" * len(header))

    for r in sorted_primary:
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
        print(f"{r['feature']:<25} {r['family']:>6} {r['tau']:>8.4f} "
              f"{r['ci_low']:>8.4f} {r['ci_high']:>8.4f} {r['p_value']:>12.2e} {sig}")

    print("-" * len(header))
    print(f"{'BM25 NQC (ref, v1 only)':<25} {'BM25':>6} {0.338:>8.4f} {'--':>8} {'--':>8} {'--':>12}")
    print(f"{'BM25 WIG (ref, v1 only)':<25} {'BM25':>6} {0.300:>8.4f} {'--':>8} {'--':>8} {'--':>12}")

    # --- Table 2: Secondary (vs nDCG@k) ---
    print("\n" + "=" * 70)
    print("  TABLE 2: Kendall tau vs nDCG@k (SECONDARY)")
    print("=" * 70)

    sorted_secondary = sorted(secondary_results, key=lambda x: abs(x["tau_100"]), reverse=True)

    header2 = (f"{'Feature':<25} {'Family':>6} {'tau@10':>8} {'tau@20':>8} "
               f"{'tau@50':>8} {'tau@100':>8}")
    print(header2)
    print("-" * len(header2))

    for r in sorted_secondary:
        print(f"{r['feature']:<25} {r['family']:>6} {r['tau_10']:>8.4f} {r['tau_20']:>8.4f} "
              f"{r['tau_50']:>8.4f} {r['tau_100']:>8.4f}")

    print()


def save_results(query_df: pd.DataFrame, primary_results: list[dict],
                 secondary_results: list[dict], tie_stats: dict):
    """Save per-query CSV and summary JSON."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Per-query CSV
    csv_cols = [
        "query_id", "year", "spearman",
        "ndcg_10", "ndcg_20", "ndcg_50", "ndcg_100",
        "score_entropy", "score_std", "frac_zero", "top_heaviness",
        "discriminability_gap", "mean_max_prob", "mean_token_entropy", "mean_prob_margin",
    ]
    csv_path = OUTPUT_DIR / "per_query_results.csv"
    query_df[csv_cols].to_csv(csv_path, index=False, float_format="%.6f")
    print(f"Saved per-query results to {csv_path}")

    # Summary JSON
    sp_vals = query_df["spearman"]
    summary = {
        "description": "Level 2 QPP evaluation: LLM-as-judge features. "
                       "Primary target: per-query Spearman. Secondary: nDCG@k.",
        "num_queries": len(query_df),
        "primary_target": {
            "metric": "spearman",
            "mean": round(float(sp_vals.mean()), 4),
            "std": round(float(sp_vals.std()), 4),
            "min": round(float(sp_vals.min()), 4),
            "max": round(float(sp_vals.max()), 4),
            "median": round(float(sp_vals.median()), 4),
        },
        "secondary_targets": {},
        "per_year": {},
        "tie_dominance": tie_stats,
        "primary_correlations": {},
        "secondary_correlations": {},
        "bm25_baselines": BM25_BASELINES,
    }

    # nDCG per cutoff
    for k in NDCG_CUTOFFS:
        col = f"ndcg_{k}"
        vals = query_df[col]
        r_sp, _ = stats.pearsonr(query_df["spearman"].values, vals.values)
        summary["secondary_targets"][f"ndcg_{k}"] = {
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
            "pearson_r_with_spearman": round(r_sp, 4),
        }

    # Per-year
    for year in sorted(query_df["year"].unique()):
        subset = query_df[query_df["year"] == year]
        summary["per_year"][int(year)] = {
            "n": int(len(subset)),
            "mean_spearman": round(float(subset["spearman"].mean()), 4),
            "mean_ndcg_10": round(float(subset["ndcg_10"].mean()), 4),
        }

    # Primary correlations
    for r in sorted(primary_results, key=lambda x: abs(x["tau"]), reverse=True):
        summary["primary_correlations"][r["feature"]] = {
            "family": r["family"],
            "tau": round(r["tau"], 4),
            "p_value": r["p_value"],
            "ci_low": round(r["ci_low"], 4),
            "ci_high": round(r["ci_high"], 4),
        }

    # Secondary correlations
    for r in sorted(secondary_results, key=lambda x: abs(x["tau_100"]), reverse=True):
        summary["secondary_correlations"][r["feature"]] = {
            "family": r["family"],
            "tau_10": round(r["tau_10"], 4),
            "p_10": r["p_10"],
            "tau_20": round(r["tau_20"], 4),
            "p_20": r["p_20"],
            "tau_50": round(r["tau_50"], 4),
            "p_50": r["p_50"],
            "tau_100": round(r["tau_100"], 4),
            "p_100": r["p_100"],
        }

    json_path = OUTPUT_DIR / "level2_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {json_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    # Part 1: Load data
    df, qid_to_year = load_data()

    # Part 2: Primary target — per-query Spearman
    spearman_results = compute_spearman(df)

    # Part 3: Secondary target — nDCG@k
    ndcg_results, ndcg10_within_std = compute_ndcg_all_cutoffs(df)

    # Part 5 & 6: Compute features
    family_a = compute_family_a(df)
    family_c = compute_family_c(df)

    # Assemble per-query DataFrame
    query_rows = []
    for qid in sorted(spearman_results.keys()):
        row = {
            "query_id": qid,
            "year": qid_to_year[qid],
            "spearman": spearman_results[qid],
        }
        for k in NDCG_CUTOFFS:
            row[f"ndcg_{k}"] = ndcg_results[qid][k]
        row.update(family_a[qid])
        row.update(family_c[qid])
        query_rows.append(row)

    query_df = pd.DataFrame(query_rows)

    # Part 4: Tie-dominance analysis
    ndcg10_cross_std = float(query_df["ndcg_10"].std())
    tie_stats = compute_tie_dominance(df, ndcg10_within_std, ndcg10_cross_std)

    # Part 7: Evaluate features
    primary_results, secondary_results = evaluate_features(query_df)

    # Part 8: Output
    print_results(query_df, primary_results, secondary_results, tie_stats)
    save_results(query_df, primary_results, secondary_results, tie_stats)

    print("Done.")


if __name__ == "__main__":
    main()
