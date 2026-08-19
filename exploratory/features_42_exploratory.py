# Exploratory sweep of 42 candidate features for per-query reliability prediction

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

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

OUTPUT_DIR = BASE_DIR / "results" / "level2"


# -----------------------------------------------------------------------
# Data loading (same as compute_level2.py)
# -----------------------------------------------------------------------
def load_qrels(path: Path) -> dict:
    qrels = defaultdict(dict)
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                qid, _, pid, grade = parts
                qrels[str(qid)][str(pid)] = int(grade)
    return dict(qrels)


def load_scores(path: Path) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_data():
    print("Loading data...")
    scores = load_scores(SCORES_V1) + load_scores(SCORES_V2)
    print(f"  Loaded {len(scores)} score rows")

    all_qrels = {}
    qid_to_year = {}
    for year, path in QRELS_FILES.items():
        qrels = load_qrels(path)
        for qid, pids in qrels.items():
            qid_to_year[str(qid)] = year
            for pid, grade in pids.items():
                all_qrels[(str(qid), str(pid))] = grade
        print(f"  {year}: {len(qrels)} queries")

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
# Helpers
# -----------------------------------------------------------------------
def entropy_base2(counts_or_probs: np.ndarray, is_counts: bool = True) -> float:
    """Entropy in bits from counts or probability array."""
    if is_counts:
        total = counts_or_probs.sum()
        if total == 0:
            return 0.0
        p = counts_or_probs / total
    else:
        p = counts_or_probs
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def safe_spearmanr(x, y):
    """Spearman r handling constant inputs."""
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    r, _ = sp_stats.spearmanr(x, y)
    return float(r) if not np.isnan(r) else 0.0


def passage_token_entropy(probs: np.ndarray) -> np.ndarray:
    """Per-passage entropy (bits) from (n, 4) probability matrix."""
    log_p = np.zeros_like(probs)
    mask = probs > 0
    log_p[mask] = np.log2(probs[mask])
    return -np.sum(probs * log_p, axis=1)


def expected_score_vec(probs: np.ndarray) -> np.ndarray:
    """Expected score E[s] = sum(g * p_g) for (n, 4) prob matrix."""
    return probs[:, 0] * 0 + probs[:, 1] * 1 + probs[:, 2] * 2 + probs[:, 3] * 3


def safe_mean(arr):
    """Mean of array; 0.0 if empty."""
    return float(np.mean(arr)) if len(arr) > 0 else 0.0


# -----------------------------------------------------------------------
# Feature computation — all 42 features per query
# -----------------------------------------------------------------------
def compute_all_features(df: pd.DataFrame) -> dict:
    """Compute all 42 features per query. Returns {qid: {feature_name: value}}."""
    print("Computing 42 features per query...")
    grouped = df.groupby("query_id")
    total = len(grouped)
    results = {}

    for idx, (qid, group) in enumerate(grouped, 1):
        if idx % 50 == 0 or idx == total:
            print(f"  Query {idx}/{total}", end="\r")

        scores = group["score"].values
        probs = group[["prob_0", "prob_1", "prob_2", "prob_3"]].values  # (n, 4)
        n = len(scores)

        # Precompute common quantities
        counts = np.array([np.sum(scores == g) for g in range(4)])
        max_probs = np.max(probs, axis=1)
        tok_entropy = passage_token_entropy(probs)
        sorted_p = np.sort(probs, axis=1)[:, ::-1]
        margins = sorted_p[:, 0] - sorted_p[:, 1]
        exp_scores = expected_score_vec(probs)

        # Ranked order: score desc, expected_score desc (for top-k features)
        order = np.lexsort((-exp_scores, -scores))
        scores_ranked = scores[order]
        probs_ranked = probs[order]
        max_probs_ranked = max_probs[order]
        tok_entropy_ranked = tok_entropy[order]

        # ---- GROUP 1: Original Family A ----
        A_score_entropy = entropy_base2(counts)
        A_score_std = float(np.std(scores))
        A_frac_zero = float(counts[0] / n)
        A_top_heaviness = float(counts[3] / n)
        if n >= 10:
            A_discriminability_gap = float(np.mean(scores_ranked[:10]) - np.mean(scores_ranked[10:]))
        else:
            A_discriminability_gap = 0.0

        # ---- GROUP 2: New Family A ----
        A_compression_ratio = float(counts.max() / n)
        A_frac_nonzero = float(1.0 - counts[0] / n)
        A_frac_grade2 = float(counts[2] / n)
        A_score_range = float(scores.max() - scores.min()) if n > 0 else 0.0
        q75, q25 = np.percentile(scores, [75, 25])
        A_score_iqr = float(q75 - q25)

        top20 = scores_ranked[:min(20, n)]
        A_top20_distinct_scores = float(len(np.unique(top20)))

        mean_score = np.mean(scores)
        if mean_score > 0:
            diffs = np.abs(scores[:, None].astype(float) - scores[None, :].astype(float))
            A_gini_coefficient = float(diffs.mean() / (2 * mean_score))
        else:
            A_gini_coefficient = 0.0

        A_score_kurtosis = float(sp_stats.kurtosis(scores, fisher=True))
        A_score_skewness = float(sp_stats.skew(scores))

        binary = (scores >= 2).astype(int)
        b_counts = np.array([np.sum(binary == 0), np.sum(binary == 1)])
        A_binary_entropy = entropy_base2(b_counts)
        A_mean_score = float(mean_score)

        # ---- GROUP 3: Original Family C ----
        C_mean_max_prob = float(np.mean(max_probs))
        C_mean_token_entropy = float(np.mean(tok_entropy))
        C_mean_prob_margin = float(np.mean(margins))

        # ---- GROUP 4: Conditional aggregation ----
        mask_top = scores >= 2
        mask_g3 = scores == 3
        mask_g0 = scores == 0
        mask_g2 = scores == 2

        C_conf_top_scored = safe_mean(max_probs[mask_top])
        C_entropy_top_scored = safe_mean(tok_entropy[mask_top])
        C_margin_top_scored = safe_mean(margins[mask_top])
        C_conf_grade3 = safe_mean(max_probs[mask_g3])
        C_conf_grade0 = safe_mean(max_probs[mask_g0])
        C_conf_grade2 = safe_mean(max_probs[mask_g2])
        C_entropy_grade2 = safe_mean(tok_entropy[mask_g2])

        # ---- GROUP 5: Confidence distribution ----
        C_conf_std = float(np.std(max_probs))
        C_conf_cv = float(np.std(max_probs) / np.mean(max_probs)) if np.mean(max_probs) > 0 else 0.0
        C_frac_low_conf = float(np.mean(max_probs < 0.5))
        C_frac_high_conf = float(np.mean(max_probs > 0.8))

        # ---- GROUP 6: Uncertainty directionality ----
        # Adjacent: prob mass on scores within +/- 1 of assigned score
        adj_mass = np.zeros(n)
        dist_mass = np.zeros(n)
        for i in range(n):
            assigned = scores[i]
            for g in range(4):
                if abs(g - assigned) <= 1:
                    adj_mass[i] += probs[i, g]
                else:
                    dist_mass[i] += probs[i, g]
        C_mean_prob_on_adjacent = float(np.mean(adj_mass))
        C_mean_prob_on_distant = float(np.mean(dist_mass))

        C_expected_score_std = float(np.std(exp_scores))

        # Expected score entropy: bin into 20 equal-width bins [0, 3]
        bin_edges = np.linspace(0, 3, 21)
        hist, _ = np.histogram(exp_scores, bins=bin_edges)
        C_expected_score_entropy = entropy_base2(hist)

        # ---- GROUP 7: Score-confidence interaction ----
        C_weighted_score_mean = float(np.mean(scores * max_probs))
        C_conf_score_correlation = safe_spearmanr(max_probs, scores)
        C_entropy_score_correlation = safe_spearmanr(tok_entropy, scores)

        # ---- GROUP 8: Mixed features ----
        # Grade boundary sharpness: gaps at 3->2 boundaries in expected score ranking
        exp_ranked = exp_scores[order]
        scores_by_exp = scores[order]
        boundary_gaps = []
        for i in range(len(scores_by_exp) - 1):
            if scores_by_exp[i] == 3 and scores_by_exp[i + 1] == 2:
                boundary_gaps.append(exp_ranked[i] - exp_ranked[i + 1])
        M_grade_boundary_sharpness = float(np.mean(boundary_gaps)) if boundary_gaps else 0.0

        top10 = min(10, n)
        M_top10_confidence = float(np.mean(max_probs_ranked[:top10]))
        M_top10_entropy = float(np.mean(tok_entropy_ranked[:top10]))

        bot10 = min(10, n)
        M_bottom10_confidence = float(np.mean(max_probs_ranked[-bot10:]))
        M_conf_range_top_bottom = M_top10_confidence - M_bottom10_confidence

        results[qid] = {
            # Group 1: Original Family A
            "A_score_entropy": A_score_entropy,
            "A_score_std": A_score_std,
            "A_frac_zero": A_frac_zero,
            "A_top_heaviness": A_top_heaviness,
            "A_discriminability_gap": A_discriminability_gap,
            # Group 2: New Family A
            "A_compression_ratio": A_compression_ratio,
            "A_frac_nonzero": A_frac_nonzero,
            "A_frac_grade2": A_frac_grade2,
            "A_score_range": A_score_range,
            "A_score_iqr": A_score_iqr,
            "A_top20_distinct_scores": A_top20_distinct_scores,
            "A_gini_coefficient": A_gini_coefficient,
            "A_score_kurtosis": A_score_kurtosis,
            "A_score_skewness": A_score_skewness,
            "A_binary_entropy": A_binary_entropy,
            "A_mean_score": A_mean_score,
            # Group 3: Original Family C
            "C_mean_max_prob": C_mean_max_prob,
            "C_mean_token_entropy": C_mean_token_entropy,
            "C_mean_prob_margin": C_mean_prob_margin,
            # Group 4: Conditional C
            "C_conf_top_scored": C_conf_top_scored,
            "C_entropy_top_scored": C_entropy_top_scored,
            "C_margin_top_scored": C_margin_top_scored,
            "C_conf_grade3": C_conf_grade3,
            "C_conf_grade0": C_conf_grade0,
            "C_conf_grade2": C_conf_grade2,
            "C_entropy_grade2": C_entropy_grade2,
            # Group 5: Confidence distribution
            "C_conf_std": C_conf_std,
            "C_conf_cv": C_conf_cv,
            "C_frac_low_conf": C_frac_low_conf,
            "C_frac_high_conf": C_frac_high_conf,
            # Group 6: Uncertainty directionality
            "C_mean_prob_on_adjacent": C_mean_prob_on_adjacent,
            "C_mean_prob_on_distant": C_mean_prob_on_distant,
            "C_expected_score_std": C_expected_score_std,
            "C_expected_score_entropy": C_expected_score_entropy,
            # Group 7: Score-confidence interaction
            "C_weighted_score_mean": C_weighted_score_mean,
            "C_conf_score_correlation": C_conf_score_correlation,
            "C_entropy_score_correlation": C_entropy_score_correlation,
            # Group 8: Mixed
            "M_grade_boundary_sharpness": M_grade_boundary_sharpness,
            "M_top10_confidence": M_top10_confidence,
            "M_top10_entropy": M_top10_entropy,
            "M_bottom10_confidence": M_bottom10_confidence,
            "M_conf_range_top_bottom": M_conf_range_top_bottom,
        }

    print()
    return results


# -----------------------------------------------------------------------
# Spearman target
# -----------------------------------------------------------------------
def compute_spearman(df: pd.DataFrame) -> dict:
    print("Computing per-query Spearman...")
    results = {}
    for qid, group in df.groupby("query_id"):
        results[qid] = safe_spearmanr(group["score"].values, group["human_grade"].values)
    return results


# -----------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------
def get_group(feat_name: str) -> str:
    if feat_name.startswith("A_"):
        return "A"
    elif feat_name.startswith("C_"):
        return "C"
    elif feat_name.startswith("M_"):
        return "M"
    return "?"


def evaluate_and_print(query_df: pd.DataFrame, feature_names: list[str]):
    print("\nComputing Kendall tau for 42 features vs per-query Spearman...")
    spearman_vals = query_df["spearman"].values

    results = []
    for feat in feature_names:
        feat_vals = query_df[feat].values
        tau, pval = sp_stats.kendalltau(feat_vals, spearman_vals)
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        results.append({
            "feature": feat,
            "group": get_group(feat),
            "tau": float(tau),
            "p_value": float(pval),
            "sig": sig,
        })

    results.sort(key=lambda x: abs(x["tau"]), reverse=True)

    # Print table
    print("\n" + "=" * 72)
    print("  KENDALL TAU vs PER-QUERY SPEARMAN (42 features)")
    print("=" * 72)
    header = f"{'Feature':<35} {'Group':>5} {'Tau':>8} {'p-value':>12} {'Sig':>4}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['feature']:<35} {r['group']:>5} {r['tau']:>8.4f} {r['p_value']:>12.2e} {r['sig']:>4}")

    # Summary
    n_05 = sum(1 for r in results if r["p_value"] < 0.05)
    n_01 = sum(1 for r in results if r["p_value"] < 0.01)
    n_001 = sum(1 for r in results if r["p_value"] < 0.001)

    print(f"\n  Significant features: {n_05} at p<0.05, {n_01} at p<0.01, {n_001} at p<0.001")

    # Best per group
    for g in ["A", "C", "M"]:
        group_results = [r for r in results if r["group"] == g]
        if group_results:
            best = group_results[0]  # already sorted by |tau|
            n_sig = sum(1 for r in group_results if r["p_value"] < 0.05)
            print(f"  Group {g}: best = {best['feature']} (tau={best['tau']:.4f}), "
                  f"{n_sig}/{len(group_results)} significant at p<0.05")

    return results


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    df, qid_to_year = load_data()

    # Target
    spearman_results = compute_spearman(df)

    # Features
    features = compute_all_features(df)

    # Assemble DataFrame
    feature_names = list(next(iter(features.values())).keys())
    query_rows = []
    for qid in sorted(features.keys()):
        row = {"query_id": qid, "year": qid_to_year[qid], "spearman": spearman_results[qid]}
        row.update(features[qid])
        query_rows.append(row)

    query_df = pd.DataFrame(query_rows)

    # Evaluate
    tau_results = evaluate_and_print(query_df, feature_names)

    # Save CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_cols = ["query_id", "year", "spearman"] + feature_names
    csv_path = OUTPUT_DIR / "per_query_features_v2.csv"
    query_df[csv_cols].to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\nSaved per-query features to {csv_path}")

    print("Done.")


if __name__ == "__main__":
    main()
