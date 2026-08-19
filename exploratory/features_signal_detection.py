# Signal-detection features (d-prime, Fisher ratio) and information-geometry features for reliability prediction

import json
import math
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
CUTOFFS = [10, 20, 50, 100]
EPS = 1e-12


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
def expected_score_vec(probs: np.ndarray) -> np.ndarray:
    return probs[:, 1] + 2 * probs[:, 2] + 3 * probs[:, 3]


def safe_spearmanr(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    r, _ = sp_stats.spearmanr(x, y)
    return float(r) if not np.isnan(r) else 0.0


def jsd(p, q):
    """Jensen-Shannon divergence in bits between two distributions."""
    p = p + EPS
    q = q + EPS
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)))


def entropy_base2(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


# -----------------------------------------------------------------------
# Feature computation
# -----------------------------------------------------------------------
def compute_all_features(df: pd.DataFrame) -> dict:
    """Compute all individual features per query."""
    print("Computing features per query...")
    grouped = df.groupby("query_id")
    total = len(grouped)
    results = {}

    for idx, (qid, group) in enumerate(grouped, 1):
        if idx % 50 == 0 or idx == total:
            print(f"  Query {idx}/{total}", end="\r")

        scores = group["score"].values
        probs = group[["prob_0", "prob_1", "prob_2", "prob_3"]].values  # (n, 4)
        n = len(scores)
        es = expected_score_vec(probs)

        # Sort by E[s] descending
        es_order = np.argsort(-es)
        es_sorted = es[es_order]
        mean_es_all = float(np.mean(es))

        feats = {}

        # ---- GROUP A: Classical QPP on expected scores ----
        for k in CUTOFFS:
            actual_k = min(k, n)
            es_topk = es_sorted[:actual_k]
            mean_topk = float(np.mean(es_topk))
            std_topk = float(np.std(es_topk))

            # NQC
            if mean_es_all > 0:
                feats[f"NQC_k{k}"] = std_topk / mean_es_all
            else:
                feats[f"NQC_k{k}"] = 0.0

            # WIG
            feats[f"WIG_k{k}"] = (mean_topk - mean_es_all) / math.sqrt(k)

            # SMV
            if mean_es_all > 0:
                feats[f"SMV_k{k}"] = mean_topk * std_topk / mean_es_all
            else:
                feats[f"SMV_k{k}"] = 0.0

        # sigma_max (= C_expected_score_std from v2)
        feats["sigma_max"] = float(np.std(es))

        # ---- GROUP B: Signal detection ----
        # Group expected scores by integer score
        groups_by_score = {}
        for g in range(4):
            mask = scores == g
            if mask.sum() >= 2:
                groups_by_score[g] = es[mask]

        def dprime(g1, g2):
            if g1 not in groups_by_score or g2 not in groups_by_score:
                return 0.0
            a, b = groups_by_score[g1], groups_by_score[g2]
            pooled = math.sqrt((np.var(a) + np.var(b)) / 2)
            if pooled == 0:
                return 0.0
            return float(abs(np.mean(a) - np.mean(b)) / pooled)

        feats["dprime_0v2"] = dprime(0, 2)
        feats["dprime_2v3"] = dprime(2, 3)

        # dprime 0 vs 2+3 combined
        if 0 in groups_by_score and (2 in groups_by_score or 3 in groups_by_score):
            combined_23 = []
            if 2 in groups_by_score:
                combined_23.append(groups_by_score[2])
            if 3 in groups_by_score:
                combined_23.append(groups_by_score[3])
            combined_23 = np.concatenate(combined_23)
            g0 = groups_by_score[0]
            pooled = math.sqrt((np.var(g0) + np.var(combined_23)) / 2)
            if pooled > 0 and len(combined_23) >= 2:
                feats["dprime_0v23"] = float(abs(np.mean(g0) - np.mean(combined_23)) / pooled)
            else:
                feats["dprime_0v23"] = 0.0
        else:
            feats["dprime_0v23"] = 0.0

        # Fisher ratio
        active_groups = {g: v for g, v in groups_by_score.items()}
        if len(active_groups) >= 2:
            total_n = sum(len(v) for v in active_groups.values())
            grand_mean = np.mean(es)
            between_var = sum(len(v) * (np.mean(v) - grand_mean) ** 2 for v in active_groups.values()) / total_n
            within_var = sum(len(v) * np.var(v) for v in active_groups.values()) / total_n
            feats["fisher_ratio"] = float(between_var / within_var) if within_var > 0 else 0.0
        else:
            feats["fisher_ratio"] = 0.0

        # Silhouette proxy
        if len(active_groups) >= 2:
            group_means = {g: np.mean(v) for g, v in active_groups.items()}
            sil_scores = []
            for g, vals in active_groups.items():
                own_mean = group_means[g]
                other_means = [group_means[og] for og in active_groups if og != g]
                for val in vals:
                    within_dist = abs(val - own_mean)
                    nearest_dist = min(abs(val - om) for om in other_means)
                    denom = max(within_dist, nearest_dist)
                    if denom > 0:
                        sil_scores.append((nearest_dist - within_dist) / denom)
                    else:
                        sil_scores.append(0.0)
            feats["silhouette_proxy"] = float(np.mean(sil_scores))
        else:
            feats["silhouette_proxy"] = 0.0

        # ---- GROUP C: Information geometry ----
        # Mean pairwise JSD (sample 200 pairs)
        rng = np.random.RandomState(42)
        if n < 20:
            # All pairs
            jsd_vals = []
            for i in range(n):
                for j in range(i + 1, n):
                    jsd_vals.append(jsd(probs[i], probs[j]))
        else:
            jsd_vals = []
            for _ in range(200):
                i, j = rng.choice(n, size=2, replace=False)
                jsd_vals.append(jsd(probs[i], probs[j]))
        feats["mean_pairwise_JSD"] = float(np.mean(jsd_vals)) if jsd_vals else 0.0

        # Effective rank of probability matrix
        _, s, _ = np.linalg.svd(probs, full_matrices=False)
        s_norm = s / s.sum() if s.sum() > 0 else s
        s_norm = s_norm[s_norm > 0]
        feats["prob_matrix_effective_rank"] = float(np.exp(-np.sum(s_norm * np.log(s_norm))))

        # Total variance
        feats["prob_total_variance"] = float(np.sum(np.var(probs, axis=0)))

        # Component range sum
        feats["prob_component_range_sum"] = float(np.sum(np.ptp(probs, axis=0)))

        # ---- GROUP D: Distributional shape of expected scores ----
        feats["es_iqr"] = float(np.percentile(es, 75) - np.percentile(es, 25))
        feats["es_range"] = float(es.max() - es.min())

        skew_es = float(sp_stats.skew(es))
        kurt_es = float(sp_stats.kurtosis(es, fisher=True))
        denom_bim = kurt_es + 3
        feats["es_bimodality"] = float((skew_es ** 2 + 1) / denom_bim) if denom_bim > 0 else 0.0

        p10 = np.percentile(es, 10)
        feats["es_q90_q10_ratio"] = float(np.percentile(es, 90) / max(p10, 0.01))

        feats["es_above_mean_frac"] = float(np.mean(es > np.mean(es)))

        top20_k = min(20, n)
        es_top20 = es_sorted[:top20_k]
        feats["es_top20_std"] = float(np.std(es_top20))
        feats["es_top20_range"] = float(es_top20[0] - es_top20[-1]) if top20_k > 1 else 0.0

        results[qid] = feats

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
# Combinations (Group E)
# -----------------------------------------------------------------------
def minmax_normalise(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def build_combos(query_df: pd.DataFrame, individual_features: list[str],
                 tau_results: list[dict]) -> pd.DataFrame:
    """Build Group E combination features. Returns DataFrame with combo columns added."""
    print("Building combination features (Group E)...")

    # Build tau lookup
    tau_lookup = {r["feature"]: r["tau"] for r in tau_results}

    # Sort individual features by |tau|
    ranked = sorted(individual_features, key=lambda f: abs(tau_lookup.get(f, 0)), reverse=True)

    # Normalise all individual features, flipping sign if tau is negative
    normed = {}
    for feat in individual_features:
        tau_sign = 1 if tau_lookup.get(feat, 0) >= 0 else -1
        vals = query_df[feat] * tau_sign
        normed[feat] = minmax_normalise(vals)

    def combo(feat_list: list[str]) -> pd.Series:
        return pd.concat([normed[f] for f in feat_list], axis=1).mean(axis=1)

    # COMBO_top2, top3, top5
    query_df["COMBO_top2"] = combo(ranked[:2])
    query_df["COMBO_top3"] = combo(ranked[:3])
    query_df["COMBO_top5"] = combo(ranked[:5])

    # COMBO_bestA_bestC: A_score_entropy + best from this script
    best_this = ranked[0]
    query_df["COMBO_bestA_bestC"] = combo(["A_score_entropy", best_this]) if "A_score_entropy" in normed else combo(ranked[:2])

    # COMBO_NQC_sigma: best NQC + sigma_max
    nqc_feats = [f for f in ranked if f.startswith("NQC_")]
    best_nqc = nqc_feats[0] if nqc_feats else ranked[0]
    query_df["COMBO_NQC_sigma"] = combo([best_nqc, "sigma_max"])

    # COMBO_NQC_fisher
    query_df["COMBO_NQC_fisher"] = combo([best_nqc, "fisher_ratio"])

    # COMBO_all_significant: all features with p < 0.01
    sig_feats = [r["feature"] for r in tau_results if r["p_value"] < 0.01 and r["feature"] in normed]
    if len(sig_feats) >= 2:
        query_df["COMBO_all_significant"] = combo(sig_feats)
    else:
        query_df["COMBO_all_significant"] = combo(ranked[:3])

    combo_names = ["COMBO_top2", "COMBO_top3", "COMBO_top5", "COMBO_bestA_bestC",
                   "COMBO_NQC_sigma", "COMBO_NQC_fisher", "COMBO_all_significant"]

    # Print what went into each combo
    print(f"  COMBO_top2: {ranked[:2]}")
    print(f"  COMBO_top3: {ranked[:3]}")
    print(f"  COMBO_top5: {ranked[:5]}")
    print(f"  COMBO_bestA_bestC: A_score_entropy + {best_this}")
    print(f"  COMBO_NQC_sigma: {best_nqc} + sigma_max")
    print(f"  COMBO_NQC_fisher: {best_nqc} + fisher_ratio")
    print(f"  COMBO_all_significant ({len(sig_feats)} feats): {sig_feats}")

    return query_df, combo_names


# -----------------------------------------------------------------------
# We also need A_score_entropy from v2 for the combo
# -----------------------------------------------------------------------
def compute_a_score_entropy(df: pd.DataFrame) -> dict:
    """Compute A_score_entropy per query for use in combos."""
    results = {}
    for qid, group in df.groupby("query_id"):
        scores = group["score"].values
        counts = np.array([np.sum(scores == g) for g in range(4)])
        results[qid] = entropy_base2(counts)
    return results


# -----------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------
def get_group(feat_name: str) -> str:
    for prefix, group in [("NQC_", "A"), ("WIG_", "A"), ("SMV_", "A"), ("sigma_", "A"),
                           ("dprime_", "B"), ("fisher_", "B"), ("silhouette_", "B"),
                           ("mean_pairwise_", "C"), ("prob_", "C"),
                           ("es_", "D"), ("COMBO_", "E"), ("A_", "A")]:
        if feat_name.startswith(prefix):
            return group
    return "?"


def evaluate_features(query_df, feature_names):
    """Compute Kendall tau for features vs Spearman. Returns sorted list."""
    spearman_vals = query_df["spearman"].values
    results = []
    for feat in feature_names:
        feat_vals = query_df[feat].values
        tau, pval = sp_stats.kendalltau(feat_vals, spearman_vals)
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        results.append({"feature": feat, "group": get_group(feat),
                         "tau": float(tau), "p_value": float(pval), "sig": sig})
    results.sort(key=lambda x: abs(x["tau"]), reverse=True)
    return results


def print_table(results, title, n_features=None):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)
    header = f"{'Feature':<35} {'Group':>5} {'Tau':>8} {'p-value':>12} {'Sig':>4}"
    print(header)
    print("-" * len(header))
    items = results[:n_features] if n_features else results
    for r in items:
        print(f"{r['feature']:<35} {r['group']:>5} {r['tau']:>8.4f} {r['p_value']:>12.2e} {r['sig']:>4}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    df, qid_to_year = load_data()
    spearman_results = compute_spearman(df)
    a_entropy = compute_a_score_entropy(df)
    features = compute_all_features(df)

    # Assemble DataFrame
    individual_feature_names = list(next(iter(features.values())).keys())
    query_rows = []
    for qid in sorted(features.keys()):
        row = {"query_id": qid, "year": qid_to_year[qid], "spearman": spearman_results[qid]}
        row["A_score_entropy"] = a_entropy[qid]
        row.update(features[qid])
        query_rows.append(row)

    query_df = pd.DataFrame(query_rows)

    # Add A_score_entropy to individual features list
    all_individual = ["A_score_entropy"] + individual_feature_names

    # Evaluate individual features
    print("\nEvaluating individual features...")
    individual_results = evaluate_features(query_df, all_individual)

    # Print individual features
    print_table(individual_results, f"INDIVIDUAL FEATURES ({len(all_individual)} features) vs Spearman")

    # Build combos
    query_df, combo_names = build_combos(query_df, all_individual, individual_results)

    # Evaluate combos
    combo_results = evaluate_features(query_df, combo_names)
    print_table(combo_results, "COMBINATION FEATURES (Group E) vs Spearman")

    # Combined table
    all_results = individual_results + combo_results
    all_results.sort(key=lambda x: abs(x["tau"]), reverse=True)
    print_table(all_results, "ALL FEATURES — sorted by |tau|")

    # Summary sections
    n_05 = sum(1 for r in all_results if r["p_value"] < 0.05)
    n_01 = sum(1 for r in all_results if r["p_value"] < 0.01)
    n_001 = sum(1 for r in all_results if r["p_value"] < 0.001)
    print(f"\n  Significant: {n_05} at p<0.05, {n_01} at p<0.01, {n_001} at p<0.001")

    for g in ["A", "B", "C", "D", "E"]:
        gr = [r for r in all_results if r["group"] == g]
        if gr:
            best = gr[0]
            print(f"  Group {g}: best = {best['feature']} (tau={best['tau']:.4f})")

    # Features beating current best (0.28)
    beating = [r for r in all_results if abs(r["tau"]) > 0.28]
    if beating:
        print(f"\n  Features beating current best (|tau| > 0.28): {len(beating)}")
        for r in beating:
            print(f"    {r['feature']:<35} tau={r['tau']:.4f}")
    else:
        print("\n  No features beat current best (|tau| > 0.28)")

    # Top 10
    print_table(all_results, "TOP 10 FEATURES", n_features=10)

    # Save CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_cols = ["query_id", "year", "spearman"] + all_individual + combo_names
    csv_path = OUTPUT_DIR / "per_query_features_v3.csv"
    query_df[csv_cols].to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\nSaved per-query features to {csv_path}")

    print("Done.")


if __name__ == "__main__":
    main()
