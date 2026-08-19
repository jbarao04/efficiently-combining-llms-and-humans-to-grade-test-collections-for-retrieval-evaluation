# Compute sentence-permutation stability features from reordered-passage scoring runs

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

SCORES_V1 = BASE_DIR / "results" / "scoring" / "normal_scores" / "scores_v1.jsonl"
SCORES_V2 = BASE_DIR / "results" / "scoring" / "normal_scores" / "scores_v2.jsonl"

B1B_DIR = BASE_DIR / "results" / "scoring" / "feature_family_b"

RUN_FILES = {
    0: (B1B_DIR / "b1b_sent_run0_v1.jsonl", B1B_DIR / "b1b_sent_run0_v2.jsonl"),
    1: (B1B_DIR / "b1b_sent_run1_v1.jsonl", B1B_DIR / "b1b_sent_run1_v2.jsonl"),
    2: (B1B_DIR / "b1b_sent_run2_v1.jsonl", B1B_DIR / "b1b_sent_run2_v2.jsonl"),
}

SPEARMAN_CSV = BASE_DIR / "results" / "level2" / "per_query_results.csv"
OUTPUT_DIR   = BASE_DIR / "results" / "level2"

N_BOOTSTRAP = 10000
SEED        = 42

RUN_DESCRIPTIONS = {
    0: "Sentence permutation (seed 0)",
    1: "Sentence permutation (seed 1)",
    2: "Sentence permutation (seed 2)",
}


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------
def load_scores_jsonl(paths) -> dict:
    """Load one or more JSONL score files.
    Returns: {qid: {pid: {"score": int, "probs": {0: f, ...}}}}
    """
    data = {}
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Score file not found: {path}")
        with open(path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                qid = str(obj["query_id"])
                pid = str(obj["passage_id"])
                probs = {int(k): float(v) for k, v in obj["probs"].items()}
                data.setdefault(qid, {})[pid] = {
                    "score": int(obj["score"]),
                    "probs": probs,
                }
    return data


def expected_score(probs: dict) -> float:
    return sum(g * probs[g] for g in range(4))


# -----------------------------------------------------------------------
# Feature computation
# -----------------------------------------------------------------------
def compute_b1b_features(original: dict, runs: dict) -> pd.DataFrame:
    """Compute per-query B1b stability features."""
    all_qids = sorted(original.keys())
    rows = []

    for qid in all_qids:
        orig_passages = original[qid]

        common_pids = set(orig_passages.keys())
        skip = False
        for r_data in runs.values():
            if qid not in r_data:
                print(f"  WARNING: query {qid} missing from a run, skipping.")
                skip = True
                break
            common_pids &= set(r_data[qid].keys())

        if skip or len(common_pids) < 5:
            rows.append({
                "query_id": qid,
                "n_passages": len(common_pids),
                "b1b_instability_var": np.nan,
                "b1b_stability_tau": np.nan,
                "b1b_mean_score_shift": np.nan,
            })
            continue

        pids = sorted(common_pids)
        n = len(pids)

        es_orig = np.array([expected_score(orig_passages[pid]["probs"]) for pid in pids])

        es_runs = []
        for r_idx in sorted(runs.keys()):
            es_r = np.array([
                expected_score(runs[r_idx][qid][pid]["probs"]) for pid in pids
            ])
            es_runs.append(es_r)

        # Measure 1: instability_var
        all_es = np.vstack([es_orig] + es_runs)
        instability_var = float(np.mean(np.var(all_es, axis=0, ddof=0)))

        # Measure 2: stability_tau
        taus = []
        for es_r in es_runs:
            tau, _ = stats.kendalltau(es_orig, es_r)
            if not np.isnan(tau):
                taus.append(float(tau))
        stability_tau = float(np.mean(taus)) if taus else np.nan

        # Diagnostic: mean absolute integer score shift
        int_orig = np.array([orig_passages[pid]["score"] for pid in pids], dtype=float)
        shifts = []
        for r_idx in sorted(runs.keys()):
            int_r = np.array([runs[r_idx][qid][pid]["score"] for pid in pids], dtype=float)
            shifts.append(float(np.mean(np.abs(int_orig - int_r))))
        mean_score_shift = float(np.mean(shifts))

        rows.append({
            "query_id": qid,
            "n_passages": n,
            "b1b_instability_var":  round(instability_var, 6),
            "b1b_stability_tau":    round(stability_tau, 6),
            "b1b_mean_score_shift": round(mean_score_shift, 4),
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------
def kendall_tau_with_ci(x: np.ndarray, y: np.ndarray,
                         n_bootstrap: int = N_BOOTSTRAP,
                         seed: int = SEED) -> dict:
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)

    tau, p_value = stats.kendalltau(x, y)

    rng = np.random.RandomState(seed)
    boot_taus = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        t, _ = stats.kendalltau(x[idx], y[idx])
        if not np.isnan(t):
            boot_taus.append(t)

    return {
        "tau":     float(tau),
        "p_value": float(p_value),
        "ci_low":  float(np.percentile(boot_taus, 2.5)),
        "ci_high": float(np.percentile(boot_taus, 97.5)),
        "n":       int(n),
    }


def evaluate_features(merged_df: pd.DataFrame) -> list:
    spearman = merged_df["spearman"].values
    features = [
        ("b1b_instability_var",  "B", "Mean per-passage variance of E[s] across sentence permutations", "negative"),
        ("b1b_stability_tau",    "B", "Mean Kendall tau between original and permuted E[s] rankings", "positive"),
        ("b1b_mean_score_shift", "B (diag)", "Mean absolute integer score shift across permutations", "negative"),
    ]
    results = []
    for feat_name, family, description, expected_sign in features:
        x = merged_df[feat_name].values
        r = kendall_tau_with_ci(x, spearman)
        r["feature"] = feat_name
        r["family"] = family
        r["description"] = description
        r["expected_sign"] = expected_sign
        results.append(r)
    return results


# -----------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------
def print_results(merged_df: pd.DataFrame, results: list):
    print("\n" + "=" * 70)
    print("  FAMILY B1b: SENTENCE PERMUTATION FEATURES")
    print("  Stability measures vs per-query Spearman (Kendall tau-b)")
    print("=" * 70)
    print(f"\n  Queries evaluated: {len(merged_df)}")

    for col in ["b1b_instability_var", "b1b_stability_tau", "b1b_mean_score_shift"]:
        vals = merged_df[col].dropna()
        print(f"\n  {col}:")
        print(f"    mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"min={vals.min():.4f}  max={vals.max():.4f}")

    print("\n" + "=" * 70)
    print("  KENDALL TAU vs PER-QUERY SPEARMAN")
    print("=" * 70)

    header = (f"{'Feature':<25} {'Family':>8} {'Tau':>8} "
              f"{'CI_low':>8} {'CI_high':>8} {'p-value':>12}  Sign")
    print(header)
    print("-" * len(header))

    for r in sorted(results, key=lambda x: abs(x["tau"]), reverse=True):
        sig = ("***" if r["p_value"] < 0.001
               else "**" if r["p_value"] < 0.01
               else "*"  if r["p_value"] < 0.05
               else "ns")
        sign_ok = ("OK" if (r["expected_sign"] == "positive" and r["tau"] > 0)
                       or  (r["expected_sign"] == "negative" and r["tau"] < 0)
                   else "INVERTED")
        print(f"{r['feature']:<25} {r['family']:>8} {r['tau']:>8.4f} "
              f"{r['ci_low']:>8.4f} {r['ci_high']:>8.4f} "
              f"{r['p_value']:>12.2e}  {sig:<4} {sign_ok}")

    print("-" * len(header))
    print("\n  Reference features:")
    print("    B3 stability_tau (instruction perturbation): tau = 0.259")
    print("    Fisher discriminant ratio (Family A):        tau = 0.297")
    print("    sigma_max (expected score std, Family A):    tau = 0.281")


def save_results(merged_df: pd.DataFrame, results: list):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_cols = ["query_id", "year", "spearman",
                "b1b_instability_var", "b1b_stability_tau",
                "b1b_mean_score_shift", "n_passages"]
    csv_cols = [c for c in csv_cols if c in merged_df.columns]
    csv_path = OUTPUT_DIR / "b1b_features.csv"
    merged_df[csv_cols].to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\nSaved: {csv_path}")

    summary = {
        "description": "Family B1b sentence-permutation stability features.",
        "n_queries": len(merged_df),
        "n_runs": 3,
        "runs": {str(k): v for k, v in RUN_DESCRIPTIONS.items()},
        "feature_results": {},
    }
    for r in results:
        summary["feature_results"][r["feature"]] = {
            "family":        r["family"],
            "description":   r["description"],
            "expected_sign": r["expected_sign"],
            "tau":           round(r["tau"], 4),
            "p_value":       r["p_value"],
            "ci_low":        round(r["ci_low"], 4),
            "ci_high":       round(r["ci_high"], 4),
            "n":             r["n"],
        }

    json_path = OUTPUT_DIR / "b1b_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  compute_b1b_features.py — Family B1b: Sentence Permutation")
    print("=" * 70)

    print("\nLoading original scores...")
    original = load_scores_jsonl([SCORES_V1, SCORES_V2])
    print(f"  {len(original)} queries")

    runs = {}
    for r_idx, (f_v1, f_v2) in RUN_FILES.items():
        print(f"Loading run {r_idx}: {RUN_DESCRIPTIONS[r_idx]}")
        runs[r_idx] = load_scores_jsonl([f_v1, f_v2])
        print(f"  {len(runs[r_idx])} queries")

    print("\nComputing stability features...")
    b1b_df = compute_b1b_features(original, runs)
    print(f"  Done: {b1b_df.dropna(subset=['b1b_instability_var']).shape[0]}/{len(b1b_df)} queries valid")

    print(f"\nLoading per-query Spearman from {SPEARMAN_CSV.name}...")
    if not SPEARMAN_CSV.exists():
        raise FileNotFoundError(f"Not found: {SPEARMAN_CSV}\nRun compute_level2.py first.")
    spearman_df = pd.read_csv(SPEARMAN_CSV)[["query_id", "year", "spearman"]]
    spearman_df["query_id"] = spearman_df["query_id"].astype(str)
    print(f"  {len(spearman_df)} queries")

    b1b_df["query_id"] = b1b_df["query_id"].astype(str)
    merged_df = spearman_df.merge(b1b_df, on="query_id", how="inner")
    print(f"  After merge: {len(merged_df)} queries")
    assert len(merged_df) == 308, f"Expected 308, got {len(merged_df)}"

    print("\nComputing Kendall tau (with 10,000-sample bootstrap CI)...")
    results = evaluate_features(merged_df)

    print_results(merged_df, results)
    save_results(merged_df, results)

    print("\nDone.")


if __name__ == "__main__":
    main()
