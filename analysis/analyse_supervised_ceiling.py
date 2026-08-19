# Supervised ceiling: ridge regression with oracle vs training-free features under cross-validation

import argparse
import json
import math
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(
        description="Supervised ceiling experiment for per-query LLM judge reliability prediction."
    )
    p.add_argument("--scores-v1", type=Path,
                    default=BASE_DIR / "results" / "scoring" / "normal_scores" / "scores_v1.jsonl",
                    help="Path to scores_v1.jsonl (TREC DL 2019-2020)")
    p.add_argument("--scores-v2", type=Path,
                    default=BASE_DIR / "results" / "scoring" / "normal_scores" / "scores_v2.jsonl",
                    help="Path to scores_v2.jsonl (TREC DL 2021-2023)")
    p.add_argument("--qrels-v1", type=Path, nargs="+",
                    default=[
                        BASE_DIR / "data_prep" / "data" / "trec-dl" / "2019" / "qrels.txt",
                        BASE_DIR / "data_prep" / "data" / "trec-dl" / "2020" / "qrels.txt",
                    ],
                    help="Paths to v1 qrels files")
    p.add_argument("--qrels-v2", type=Path, nargs="+",
                    default=[
                        BASE_DIR / "data_prep" / "data" / "trec-dl-v2" / "2021" / "qrels_dedup.txt",
                        BASE_DIR / "data_prep" / "data" / "trec-dl-v2" / "2022" / "qrels_dedup.txt",
                        BASE_DIR / "data_prep" / "data" / "trec-dl-v2" / "2023" / "qrels_dedup.txt",
                    ],
                    help="Paths to v2 qrels files")
    p.add_argument("--b1b-features", type=Path,
                    default=BASE_DIR / "results" / "level2" / "b1b_features.csv",
                    help="Path to b1b_features.csv")
    p.add_argument("--output-dir", type=Path,
                    default=BASE_DIR / "results" / "supervised_ceiling",
                    help="Output directory")
    p.add_argument("--n-repeats", type=int, default=20,
                    help="Number of CV repeats (default: 20)")
    p.add_argument("--seed", type=int, default=42, help="Master random seed")
    return p.parse_args()


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------
def load_qrels(path: Path) -> dict:
    """Load TREC-format qrels. Returns {qid: {pid: grade}}."""
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


def load_all_data(args):
    """Load scores and qrels, merge into a single DataFrame."""
    print("Loading data...")

    # Load scores
    scores = []
    for p in [args.scores_v1, args.scores_v2]:
        if not p.exists():
            print(f"  WARNING: {p} not found, skipping")
            continue
        scores.extend(load_scores(p))
        print(f"  Loaded {p.name}")
    print(f"  Total score rows: {len(scores)}")

    # Load qrels
    all_qrels = {}  # (qid, pid) -> grade
    for p in list(args.qrels_v1) + list(args.qrels_v2):
        if not p.exists():
            print(f"  WARNING: {p} not found, skipping")
            continue
        qrels = load_qrels(p)
        for qid, pids in qrels.items():
            for pid, grade in pids.items():
                all_qrels[(str(qid), str(pid))] = grade
        print(f"  Qrels from {p.name}: {len(qrels)} queries")

    # Merge
    rows = []
    for s in scores:
        qid = str(s["query_id"])
        pid = str(s["passage_id"])
        key = (qid, pid)
        if key in all_qrels:
            probs = s["probs"]
            p = np.array([float(probs[str(g)]) for g in range(4)])
            es = float(p[0] * 0 + p[1] * 1 + p[2] * 2 + p[3] * 3)
            rows.append({
                "query_id": qid,
                "passage_id": pid,
                "score": int(s["score"]),
                "es": es,
                "prob_0": p[0], "prob_1": p[1], "prob_2": p[2], "prob_3": p[3],
                "human_grade": all_qrels[key],
            })

    df = pd.DataFrame(rows)
    n_queries = df["query_id"].nunique()
    print(f"  Merged: {len(df)} pairs, {n_queries} queries")
    return df


# -----------------------------------------------------------------------
# Target variable computation
# -----------------------------------------------------------------------
def compute_targets(df: pd.DataFrame, min_passages=10):
    """Compute per-query Pearson and Spearman correlations."""
    targets = []
    skipped = {"too_few": 0, "const_human": 0, "const_llm": 0}

    for qid, g in df.groupby("query_id"):
        if len(g) < min_passages:
            skipped["too_few"] += 1
            continue
        es = g["es"].values
        hg = g["human_grade"].values
        sc = g["score"].values

        if np.std(hg) == 0:
            skipped["const_human"] += 1
            continue
        if np.std(es) == 0:
            skipped["const_llm"] += 1
            continue

        pearson_r, _ = sp_stats.pearsonr(es, hg)
        spearman_rho, _ = sp_stats.spearmanr(sc, hg)

        targets.append({
            "query_id": qid,
            "pearson": pearson_r,
            "spearman": spearman_rho,
            "n_passages": len(g),
        })

    if skipped["too_few"] or skipped["const_human"] or skipped["const_llm"]:
        print(f"  Skipped queries: {skipped}")

    return pd.DataFrame(targets)


# -----------------------------------------------------------------------
# Feature computation
# -----------------------------------------------------------------------
def compute_features(df: pd.DataFrame, b1b_path: Path):
    """Compute all features per query."""
    features = []

    for qid, g in df.groupby("query_id"):
        es = g["es"].values
        hg = g["human_grade"].values
        sc = g["score"].values
        feat = {"query_id": qid}

        # --- Training-free features ---

        # 1. fisher_ratio
        groups = defaultdict(list)
        for s, e in zip(sc, es):
            groups[s].append(e)
        if len(groups) >= 2:
            grand_mean = np.mean(es)
            between = sum(len(v) * (np.mean(v) - grand_mean) ** 2 for v in groups.values()) / len(es)
            within = sum(np.var(v) * len(v) for v in groups.values()) / len(es)
            feat["fisher_ratio"] = between / within if within > 1e-12 else 0.0
        else:
            feat["fisher_ratio"] = 0.0

        # 3. score3_mean_es
        s3_es = es[sc == 3]
        feat["score3_mean_es"] = float(np.mean(s3_es)) if len(s3_es) > 0 else np.nan

        # --- Oracle features ---

        # 4. extreme_sep
        es_rel = es[hg >= 1]
        es_irrel = es[hg == 0]
        feat["extreme_sep"] = (float(np.mean(es_rel)) - float(np.mean(es_irrel))
                               if len(es_rel) > 0 and len(es_irrel) > 0 else np.nan)

        # 5. es_h0
        feat["es_h0"] = float(np.mean(es_irrel)) if len(es_irrel) > 0 else np.nan

        # 6. es_h3
        es_h3 = es[hg == 3]
        feat["es_h3"] = float(np.mean(es_h3)) if len(es_h3) > 0 else np.nan

        # 7. sep_neg
        es_h1 = es[hg == 1]
        feat["sep_neg"] = (float(np.mean(es_h1)) - float(np.mean(es_irrel))
                           if len(es_h1) > 0 and len(es_irrel) > 0 else np.nan)

        # 8. fp_h1_share
        fp_mask = (sc >= 2) & (hg <= 1)
        fp_grades = hg[fp_mask]
        if len(fp_grades) > 0:
            feat["fp_h1_share"] = float(np.sum(fp_grades == 1)) / len(fp_grades)
        else:
            feat["fp_h1_share"] = np.nan

        # 9. dE_fp_ar
        fp_es = es[fp_mask]
        ar_mask = (sc >= 2) & (hg >= 2)
        ar_es = es[ar_mask]
        feat["dE_fp_ar"] = (float(np.mean(fp_es)) - float(np.mean(ar_es))
                            if len(fp_es) > 0 and len(ar_es) > 0 else np.nan)

        # --- Pure-qrels features ---

        # 10. human_grade_entropy
        counts = np.array([np.sum(hg == g) for g in range(4)], dtype=float)
        freqs = counts / counts.sum()
        ent = -sum(f * math.log2(f) for f in freqs if f > 0)
        feat["human_grade_entropy"] = ent

        # 11. frac_grade0_human
        feat["frac_grade0_human"] = float(np.mean(hg == 0))

        # 12. frac_grade3_human
        feat["frac_grade3_human"] = float(np.mean(hg == 3))

        # 13. n_relevant (log-transformed)
        feat["n_relevant"] = math.log(1 + int(np.sum(hg >= 1)))

        features.append(feat)

    feat_df = pd.DataFrame(features)

    # Load b1b_stability_tau
    b1b_loaded = False
    if b1b_path.exists():
        b1b = pd.read_csv(b1b_path)
        b1b["query_id"] = b1b["query_id"].astype(str)
        feat_df = feat_df.merge(b1b[["query_id", "b1b_stability_tau"]], on="query_id", how="left")
        b1b_loaded = True
        print(f"  B1b features loaded: {b1b['query_id'].nunique()} queries")
    else:
        feat_df["b1b_stability_tau"] = np.nan
        print(f"  WARNING: B1b features not found at {b1b_path}")

    return feat_df, b1b_loaded


# -----------------------------------------------------------------------
# Model definitions
# -----------------------------------------------------------------------
MODELS = {
    "M1": {
        "name": "Training-free (2)",
        "features": ["fisher_ratio", "b1b_stability_tau"],
    },
    "M2": {
        "name": "Training-free (3)",
        "features": ["fisher_ratio", "b1b_stability_tau", "score3_mean_es"],
    },
    "M3": {
        "name": "Core oracle (8)",
        "features": ["extreme_sep", "es_h0", "fp_h1_share", "dE_fp_ar",
                      "human_grade_entropy", "frac_grade0_human", "frac_grade3_human", "n_relevant"],
    },
    "M4": {
        "name": "Full oracle (10)",
        "features": ["extreme_sep", "es_h0", "es_h3", "fp_h1_share", "dE_fp_ar",
                      "sep_neg", "human_grade_entropy", "frac_grade0_human",
                      "frac_grade3_human", "n_relevant"],
    },
    "M5": {
        "name": "Combined (10)",
        "features": ["extreme_sep", "es_h0", "fp_h1_share", "dE_fp_ar",
                      "human_grade_entropy", "frac_grade0_human", "frac_grade3_human",
                      "n_relevant", "fisher_ratio", "b1b_stability_tau"],
    },
    "M6": {
        "name": "extreme_sep alone",
        "features": ["extreme_sep"],
    },
}


# -----------------------------------------------------------------------
# Cross-validation
# -----------------------------------------------------------------------
def run_cv(X: np.ndarray, y: np.ndarray, n_repeats: int, master_seed: int):
    """Repeated stratified 5-fold CV with RidgeCV.

    Returns dict with per-repeat metrics and out-of-fold predictions.
    """
    n = len(y)
    alphas = np.logspace(-3, 3, 50)

    # Bin target into quintiles for stratification
    bins = pd.qcut(y, q=5, labels=False, duplicates="drop")

    oof_preds_sum = np.zeros(n)
    oof_counts = np.zeros(n)
    repeat_metrics = {"r2": [], "pearson_r": [], "kendall_tau": []}

    for rep in range(n_repeats):
        seed = master_seed + rep
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        preds = np.full(n, np.nan)
        for train_idx, test_idx in skf.split(X, bins):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            ridge = RidgeCV(alphas=alphas, scoring="r2")
            ridge.fit(X_train_s, y_train)
            preds[test_idx] = ridge.predict(X_test_s)

        # Compute metrics for this repeat
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        pr, _ = sp_stats.pearsonr(preds, y)
        kt, _ = sp_stats.kendalltau(preds, y)

        repeat_metrics["r2"].append(r2)
        repeat_metrics["pearson_r"].append(pr)
        repeat_metrics["kendall_tau"].append(kt)

        oof_preds_sum += preds
        oof_counts += 1

    oof_preds_avg = oof_preds_sum / oof_counts

    # In-sample R²
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    ridge_full = RidgeCV(alphas=alphas, scoring="r2")
    ridge_full.fit(X_s, y)
    y_hat = ridge_full.predict(X_s)
    ss_res_full = np.sum((y - y_hat) ** 2)
    ss_tot_full = np.sum((y - np.mean(y)) ** 2)
    insample_r2 = 1 - ss_res_full / ss_tot_full

    return {
        "insample_r2": insample_r2,
        "cv_r2_mean": np.mean(repeat_metrics["r2"]),
        "cv_r2_std": np.std(repeat_metrics["r2"]),
        "cv_r_mean": np.mean(repeat_metrics["pearson_r"]),
        "cv_r_std": np.std(repeat_metrics["pearson_r"]),
        "cv_tau_mean": np.mean(repeat_metrics["kendall_tau"]),
        "cv_tau_std": np.std(repeat_metrics["kendall_tau"]),
        "oof_preds": oof_preds_avg,
        "repeat_metrics": repeat_metrics,
    }


# -----------------------------------------------------------------------
# Split-half reliability
# -----------------------------------------------------------------------
def split_half_reliability(df: pd.DataFrame, n_iter=100, master_seed=42):
    """Estimate passage-sampling reliability of per-query Pearson."""
    rng = np.random.RandomState(master_seed)
    corrs = []

    query_groups = {qid: g for qid, g in df.groupby("query_id")}

    for it in range(n_iter):
        pearson_a = {}
        pearson_b = {}
        for qid, g in query_groups.items():
            n = len(g)
            if n < 4:  # need at least 2 per half
                continue
            idx = rng.permutation(n)
            half = n // 2
            a = g.iloc[idx[:half]]
            b = g.iloc[idx[half:half * 2]]

            # Check variance
            if np.std(a["human_grade"].values) == 0 or np.std(a["es"].values) == 0:
                continue
            if np.std(b["human_grade"].values) == 0 or np.std(b["es"].values) == 0:
                continue

            r_a, _ = sp_stats.pearsonr(a["es"].values, a["human_grade"].values)
            r_b, _ = sp_stats.pearsonr(b["es"].values, b["human_grade"].values)
            pearson_a[qid] = r_a
            pearson_b[qid] = r_b

        common = sorted(set(pearson_a) & set(pearson_b))
        if len(common) < 10:
            continue
        va = np.array([pearson_a[q] for q in common])
        vb = np.array([pearson_b[q] for q in common])
        r, _ = sp_stats.pearsonr(va, vb)
        corrs.append(r)

    r_hh = np.mean(corrs)
    reliability = 2 * r_hh / (1 + r_hh)
    return r_hh, reliability, len(corrs)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Load data
    df = load_all_data(args)
    if len(df) == 0:
        print("ERROR: No data loaded. Check file paths.")
        sys.exit(1)

    # Compute targets
    print("\nComputing targets...")
    targets = compute_targets(df)
    print(f"  Queries with valid targets: {len(targets)}")

    # Compute features
    print("\nComputing features...")
    feat_df, b1b_loaded = compute_features(df, args.b1b_features)

    # Merge targets and features
    data = targets.merge(feat_df, on="query_id", how="inner")
    print(f"  Final dataset: {len(data)} queries")

    # ---- Data summary ----
    print("\n" + "=" * 77)
    print("DATA SUMMARY")
    print("=" * 77)
    print(f"  Queries:       {len(data)}")
    print(f"  Total pairs:   {df.shape[0]}")
    for col, label in [("pearson", "Pearson target"), ("spearman", "Spearman target")]:
        v = data[col]
        print(f"  {label}: mean={v.mean():.4f}  std={v.std():.4f}  "
              f"min={v.min():.4f}  max={v.max():.4f}  median={v.median():.4f}")

    # ---- Feature availability ----
    all_features = sorted(set(f for m in MODELS.values() for f in m["features"]))
    print(f"\n{'FEATURE AVAILABILITY':^77}")
    print("-" * 77)
    print(f"  {'Feature':<25s} {'n_avail':>8s} {'n_miss':>8s} {'mean':>10s} {'std':>10s}")
    print("-" * 77)
    for feat_name in all_features:
        if feat_name in data.columns:
            valid = data[feat_name].dropna()
            print(f"  {feat_name:<25s} {len(valid):>8d} {len(data)-len(valid):>8d} "
                  f"{valid.mean():>10.4f} {valid.std():>10.4f}")
        else:
            print(f"  {feat_name:<25s} {'N/A':>8s} {'N/A':>8s} {'N/A':>10s} {'N/A':>10s}")

    # ---- Run models ----
    for target_col, target_label in [("pearson", "per-query Pearson"), ("spearman", "per-query Spearman")]:
        print(f"\n{'=' * 77}")
        print(f"MODEL COMPARISON (target: {target_label})")
        print(f"{'=' * 77}")
        print(f"  {'Model':<22s} {'n':>5s} {'Feats':>5s} {'In-samp R²':>11s} "
              f"{'CV R² (mean±std)':>18s} {'CV r':>10s} {'CV tau':>10s}")
        print("-" * 77)

        results = {}
        pred_cols = {}

        for mid, mdef in MODELS.items():
            feat_cols = mdef["features"]

            # Check if all features exist
            missing_cols = [f for f in feat_cols if f not in data.columns]
            if missing_cols:
                print(f"  {mdef['name']:<22s}  SKIPPED (missing columns: {missing_cols})")
                continue

            # Skip models requiring b1b if not loaded
            if not b1b_loaded and "b1b_stability_tau" in feat_cols:
                print(f"  {mdef['name']:<22s}  SKIPPED (b1b features unavailable)")
                continue

            # Restrict to complete cases
            subset = data.dropna(subset=feat_cols + [target_col])
            n = len(subset)
            if n < 20:
                print(f"  {mdef['name']:<22s}  SKIPPED (n={n} too small)")
                continue

            X = subset[feat_cols].values
            y = subset[target_col].values
            qids = subset["query_id"].values

            print(f"  Running {mdef['name']} (n={n}, {len(feat_cols)} features)...", end="", flush=True)

            res = run_cv(X, y, args.n_repeats, args.seed)

            print(f"\r  {mdef['name']:<22s} {n:>5d} {len(feat_cols):>5d} "
                  f"{res['insample_r2']:>11.4f} "
                  f"{res['cv_r2_mean']:>8.4f}±{res['cv_r2_std']:<7.4f} "
                  f"{res['cv_r_mean']:>10.4f} {res['cv_tau_mean']:>10.4f}")

            results[mid] = {
                "name": mdef["name"],
                "n": n,
                "n_features": len(feat_cols),
                "features": feat_cols,
                **{k: v for k, v in res.items() if k != "oof_preds" and k != "repeat_metrics"},
                "repeat_metrics": res["repeat_metrics"],
            }

            # Store predictions for primary target only
            if target_col == "pearson":
                for qid_val, pred_val in zip(qids, res["oof_preds"]):
                    if qid_val not in pred_cols:
                        pred_cols[qid_val] = {}
                    pred_cols[qid_val][f"pred_{mid}"] = pred_val

        print("=" * 77)

        # Store results for output
        if target_col == "pearson":
            primary_results = results
            primary_preds = pred_cols
        else:
            secondary_results = results

    # ---- Interpretive summary ----
    print(f"\n{'=' * 77}")
    print("INTERPRETIVE SUMMARY")
    print("=" * 77)

    tf_key = "M1" if "M1" in primary_results else ("M2" if "M2" in primary_results else None)
    oracle_key = "M3" if "M3" in primary_results else None

    if tf_key and oracle_key:
        tf_r2 = primary_results[tf_key]["cv_r2_mean"]
        oracle_r2 = primary_results[oracle_key]["cv_r2_mean"]
        ratio = tf_r2 / oracle_r2 if oracle_r2 > 0 else float("nan")
        print(f"  Training-free ({tf_key}) CV R²:     {tf_r2:.4f}")
        print(f"  Core oracle (M3) CV R²:             {oracle_r2:.4f}")
        print(f"  Fraction of ceiling captured:        {ratio:.4f}  ({ratio*100:.1f}%)")
        print(f"  Unexplained by oracle (noise+inac.): {1 - oracle_r2:.4f}  ({(1-oracle_r2)*100:.1f}%)")
    else:
        print("  Could not compute summary (missing model results).")

    # ---- Split-half reliability ----
    print(f"\n{'=' * 77}")
    print("SPLIT-HALF RELIABILITY OF TARGET")
    print("=" * 77)
    r_hh, reliability, n_iters = split_half_reliability(df, n_iter=100, master_seed=args.seed)
    print(f"  Split-half correlation (mean over {n_iters} iterations): {r_hh:.4f}")
    print(f"  Spearman-Brown reliability:                              {reliability:.4f}")
    print(f"  Upper bound on R² for any predictor:                     {reliability:.4f}")
    print(f"\n  CAVEAT: This captures passage-sampling noise only, not assessor")
    print(f"  disagreement. True reliability may be lower if assessors disagree.")

    # ---- Save outputs ----
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. results_summary.json
    summary = {
        "n_queries": len(data),
        "n_pairs": int(df.shape[0]),
        "target_stats": {
            "pearson": {"mean": float(data["pearson"].mean()), "std": float(data["pearson"].std()),
                        "min": float(data["pearson"].min()), "max": float(data["pearson"].max())},
            "spearman": {"mean": float(data["spearman"].mean()), "std": float(data["spearman"].std()),
                         "min": float(data["spearman"].min()), "max": float(data["spearman"].max())},
        },
        "split_half": {"r_hh": float(r_hh), "reliability": float(reliability)},
        "primary_results": {},
        "secondary_results": {},
    }
    for mid, res in primary_results.items():
        summary["primary_results"][mid] = {
            k: (float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in res.items() if k != "repeat_metrics"
        }
        summary["primary_results"][mid]["repeat_r2"] = [float(x) for x in res["repeat_metrics"]["r2"]]
    for mid, res in secondary_results.items():
        summary["secondary_results"][mid] = {
            k: (float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in res.items() if k != "repeat_metrics"
        }

    with open(args.output_dir / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved results_summary.json")

    # 2. per_query_predictions.csv
    pred_rows = []
    for qid in data["query_id"].values:
        row = {
            "query_id": qid,
            "true_pearson": float(data.loc[data["query_id"] == qid, "pearson"].values[0]),
            "true_spearman": float(data.loc[data["query_id"] == qid, "spearman"].values[0]),
        }
        if qid in primary_preds:
            row.update(primary_preds[qid])
        pred_rows.append(row)
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(args.output_dir / "per_query_predictions.csv", index=False)
    print(f"  Saved per_query_predictions.csv")

    # 3. feature_matrix.csv
    data.to_csv(args.output_dir / "feature_matrix.csv", index=False)
    print(f"  Saved feature_matrix.csv")

    # ---- Verification checks ----
    print(f"\n{'=' * 77}")
    print("VERIFICATION CHECKS")
    print("=" * 77)
    checks = []

    if "M1" in primary_results:
        r2 = primary_results["M1"]["cv_r2_mean"]
        ok = 0.10 <= r2 <= 0.30
        checks.append(("M1 CV R² in [0.10, 0.30]", r2, ok))

    if "M6" in primary_results and tf_key and tf_key in primary_results:
        m6_r2 = primary_results["M6"]["cv_r2_mean"]
        tf_r2 = primary_results[tf_key]["cv_r2_mean"]
        ok = m6_r2 > tf_r2
        checks.append(("M6 > Training-free", f"{m6_r2:.4f} > {tf_r2:.4f}", ok))

    if "M3" in primary_results and "M6" in primary_results:
        m3_r2 = primary_results["M3"]["cv_r2_mean"]
        m6_r2 = primary_results["M6"]["cv_r2_mean"]
        ok = m3_r2 >= m6_r2 - 0.01  # small tolerance
        checks.append(("M3 >= M6", f"{m3_r2:.4f} >= {m6_r2:.4f}", ok))

    if "M5" in primary_results and "M3" in primary_results:
        m5_r2 = primary_results["M5"]["cv_r2_mean"]
        m3_r2 = primary_results["M3"]["cv_r2_mean"]
        ok = m5_r2 >= m3_r2 - 0.01
        checks.append(("M5 >= M3", f"{m5_r2:.4f} >= {m3_r2:.4f}", ok))

    ok_rel = reliability > 0.80
    checks.append(("Split-half reliability > 0.80", f"{reliability:.4f}", ok_rel))

    for desc, val, ok in checks:
        status = "PASS" if ok else "WARNING"
        print(f"  [{status}] {desc}: {val}")

    print(f"\nDone. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
