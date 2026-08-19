# Compute synonym-substitution stability features from perturbed-passage scoring runs

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

B1A_DIR = BASE_DIR / "results" / "scoring" / "feature_family_b"

RUN_FILES = {
    0: (B1A_DIR / "b1a_syn_run0_v1.jsonl", B1A_DIR / "b1a_syn_run0_v2.jsonl"),
    1: (B1A_DIR / "b1a_syn_run1_v1.jsonl", B1A_DIR / "b1a_syn_run1_v2.jsonl"),
    2: (B1A_DIR / "b1a_syn_run2_v1.jsonl", B1A_DIR / "b1a_syn_run2_v2.jsonl"),
}

SPEARMAN_CSV = BASE_DIR / "results" / "level2" / "per_query_results.csv"
OUTPUT_DIR   = BASE_DIR / "results" / "level2"

N_BOOTSTRAP = 10000
SEED        = 42


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------
def load_scores_jsonl(paths) -> dict:
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
def compute_features(original: dict, runs: dict) -> pd.DataFrame:
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
                "b1a_instability_var": np.nan,
                "b1a_stability_tau": np.nan,
                "b1a_mean_score_shift": np.nan,
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

        all_es = np.vstack([es_orig] + es_runs)
        instability_var = float(np.mean(np.var(all_es, axis=0, ddof=0)))

        taus = []
        for es_r in es_runs:
            tau, _ = stats.kendalltau(es_orig, es_r)
            if not np.isnan(tau):
                taus.append(float(tau))
        stability_tau = float(np.mean(taus)) if taus else np.nan

        int_orig = np.array([orig_passages[pid]["score"] for pid in pids], dtype=float)
        shifts = []
        for r_idx in sorted(runs.keys()):
            int_r = np.array([runs[r_idx][qid][pid]["score"] for pid in pids], dtype=float)
            shifts.append(float(np.mean(np.abs(int_orig - int_r))))
        mean_score_shift = float(np.mean(shifts))

        rows.append({
            "query_id": qid,
            "n_passages": n,
            "b1a_instability_var":  round(instability_var, 6),
            "b1a_stability_tau":    round(stability_tau, 6),
            "b1a_mean_score_shift": round(mean_score_shift, 4),
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------
def kendall_tau_with_ci(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    tau, p_value = stats.kendalltau(x, y)

    rng = np.random.RandomState(SEED)
    boot_taus = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(n, size=n, replace=True)
        t, _ = stats.kendalltau(x[idx], y[idx])
        if not np.isnan(t):
            boot_taus.append(t)

    return {
        "tau": float(tau), "p_value": float(p_value),
        "ci_low": float(np.percentile(boot_taus, 2.5)),
        "ci_high": float(np.percentile(boot_taus, 97.5)),
        "n": int(n),
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  compute_b1a_features.py — Family B1a: Synonym Substitution")
    print("=" * 70)

    print("\nLoading original scores...")
    original = load_scores_jsonl([SCORES_V1, SCORES_V2])
    print(f"  {len(original)} queries")

    runs = {}
    for r_idx, (f_v1, f_v2) in RUN_FILES.items():
        print(f"Loading run {r_idx}...")
        runs[r_idx] = load_scores_jsonl([f_v1, f_v2])
        print(f"  {len(runs[r_idx])} queries")

    print("\nComputing stability features...")
    df = compute_features(original, runs)
    print(f"  Done: {df.dropna(subset=['b1a_stability_tau']).shape[0]}/{len(df)} queries valid")

    print(f"\nLoading per-query Spearman...")
    spearman_df = pd.read_csv(SPEARMAN_CSV)[["query_id", "year", "spearman"]]
    spearman_df["query_id"] = spearman_df["query_id"].astype(str)
    df["query_id"] = df["query_id"].astype(str)
    merged = spearman_df.merge(df, on="query_id", how="inner")
    assert len(merged) == 308, f"Expected 308, got {len(merged)}"

    print("\nComputing Kendall tau (with bootstrap CI)...\n")

    print("=" * 70)
    print("  KENDALL TAU vs PER-QUERY SPEARMAN")
    print("=" * 70)

    header = f"{'Feature':<25} {'Tau':>8} {'CI_low':>8} {'CI_high':>8} {'p-value':>12}"
    print(header)
    print("-" * len(header))

    features = [
        ("b1a_stability_tau", "positive"),
        ("b1a_instability_var", "negative"),
        ("b1a_mean_score_shift", "negative"),
    ]
    for feat, expected in features:
        r = kendall_tau_with_ci(merged[feat].values, merged["spearman"].values)
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
        sign_ok = "OK" if (expected == "positive" and r["tau"] > 0) or (expected == "negative" and r["tau"] < 0) else "INVERTED"
        print(f"{feat:<25} {r['tau']:>8.4f} {r['ci_low']:>8.4f} {r['ci_high']:>8.4f} {r['p_value']:>12.2e}  {sig:<4} {sign_ok}")

    print("-" * len(header))
    print("\n  Reference:")
    print("    b1b_stability_tau (sentence permutation): tau = 0.338")
    print("    b3_stability_tau  (instruction perturb):  tau = 0.259")
    print("    fisher_ratio      (Family A):             tau = 0.297")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_cols = [c for c in ["query_id", "year", "spearman", "b1a_stability_tau",
                "b1a_instability_var", "b1a_mean_score_shift", "n_passages"]
                if c in merged.columns]
    merged[csv_cols].to_csv(OUTPUT_DIR / "b1a_features.csv", index=False, float_format="%.6f")
    print(f"\nSaved: {OUTPUT_DIR / 'b1a_features.csv'}")


if __name__ == "__main__":
    main()
