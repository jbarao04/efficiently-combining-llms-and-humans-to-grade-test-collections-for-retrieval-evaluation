# Structural diagnostics: error distribution, retrieval heterogeneity, and error-frequency alignment

import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from v2_id_mapping import V2_YEARS, load_canonical_map, canonicalize_runs

# ── Paths (identical layout to run_impact_oracle.py) ────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent

YEARS = {
    2019: {
        "qrels":    BASE_DIR / "data_prep" / "data" / "trec-dl" / "2019" / "qrels.txt",
        "scores":   BASE_DIR / "results" / "scoring" / "normal_scores" / "scores_v1.jsonl",
        "runs_dir": BASE_DIR / "data" / "system_runs" / "2019",
    },
    2020: {
        "qrels":    BASE_DIR / "data_prep" / "data" / "trec-dl" / "2020" / "qrels.txt",
        "scores":   BASE_DIR / "results" / "scoring" / "normal_scores" / "scores_v1.jsonl",
        "runs_dir": BASE_DIR / "data" / "system_runs" / "2020",
    },
    2021: {
        "qrels":    BASE_DIR / "data_prep" / "data" / "trec-dl-v2" / "2021" / "qrels_dedup.txt",
        "scores":   BASE_DIR / "results" / "scoring" / "normal_scores" / "scores_v2.jsonl",
        "runs_dir": BASE_DIR / "data" / "system_runs" / "2021",
    },
    2022: {
        "qrels":    BASE_DIR / "data_prep" / "data" / "trec-dl-v2" / "2022" / "qrels_dedup.txt",
        "scores":   BASE_DIR / "results" / "scoring" / "normal_scores" / "scores_v2.jsonl",
        "runs_dir": BASE_DIR / "data" / "system_runs" / "2022",
    },
    2023: {
        "qrels":    BASE_DIR / "data_prep" / "data" / "trec-dl-v2" / "2023" / "qrels_dedup.txt",
        "scores":   BASE_DIR / "results" / "scoring" / "normal_scores" / "scores_v2.jsonl",
        "runs_dir": BASE_DIR / "data" / "system_runs" / "2023",
    },
}

FEATURES_CSV = BASE_DIR / "results" / "level2" / "per_query_results.csv"
FISHER_CSV   = BASE_DIR / "results" / "level2" / "per_query_features_v3.csv"
B1B_CSV      = BASE_DIR / "results" / "level2" / "b1b_features.csv"
OUTPUT_DIR   = BASE_DIR / "results" / "diagnostics"


# ── Data loaders (identical to run_impact_oracle.py) ────────────────────────

def load_qrels(path):
    qrels = defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qrels[parts[0]][parts[2]] = int(parts[3])
    return dict(qrels)


def load_llm_qrels(jsonl_path, year_queries):
    qrels = defaultdict(dict)
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            qid = str(rec["query_id"])
            if qid in year_queries:
                qrels[qid][str(rec["passage_id"])] = int(rec["score"])
    return dict(qrels)


def load_system_runs(runs_dir):
    runs = {}
    for fname in sorted(os.listdir(runs_dir)):
        if not fname.endswith(".txt"):
            continue
        sys_name = fname[:-4]
        sys_runs = defaultdict(list)
        with open(os.path.join(runs_dir, fname)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                sys_runs[parts[0]].append((int(parts[3]), parts[2]))
        for qid in sys_runs:
            sys_runs[qid].sort()
            sys_runs[qid] = [pid for _, pid in sys_runs[qid][:1000]]
        runs[sys_name] = dict(sys_runs)
    return runs


def load_feature_queries():
    """Return the intersection of query_ids present across all three feature CSVs."""
    df_main  = pd.read_csv(FEATURES_CSV, dtype={"query_id": str})
    df_fish  = pd.read_csv(FISHER_CSV,   dtype={"query_id": str})
    df_b1b   = pd.read_csv(B1B_CSV,      dtype={"query_id": str})
    return set(df_main["query_id"]) & set(df_fish["query_id"]) & set(df_b1b["query_id"])


# ── Helpers ─────────────────────────────────────────────────────────────────

def exp_gain(grade):
    """trec_eval exponential gain: 2^g - 1."""
    return (2 ** grade) - 1


def gini(x):
    """
    Gini coefficient of array x (all non-negative).
    Formula: G = (2 * Σ i*x[i] - (n+1)*S) / (n*S)  where x is sorted ascending (1-indexed i).
    Returns 0.0 when all values are zero.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0:
        return 0.0
    S = x.sum()
    if S == 0.0:
        return 0.0
    indices = np.arange(1, n + 1)
    return float((2.0 * np.dot(indices, x) - (n + 1) * S) / (n * S))


# ── Measurement 1: LLM error distribution within each query ─────────────────

def compute_eps_distribution(queries, human_qrels, llm_qrels):
    """
    For each query, compute per-passage gain errors and summarise their distribution.

    Returns
    -------
    rows : list of dicts  (one per query, for CSV)
    per_year_abs_eps : list of floats  (all |ε| values pooled, for CDF plot)
    """
    rows = []
    per_year_abs_eps = []

    for qid in queries:
        hq = human_qrels.get(qid, {})
        lq = llm_qrels.get(qid, {})
        pool = set(hq.keys()) | set(lq.keys())

        if not pool:
            continue

        abs_eps = np.array([
            abs(exp_gain(lq.get(pid, 0)) - exp_gain(hq.get(pid, 0)))
            for pid in pool
        ])
        per_year_abs_eps.extend(abs_eps.tolist())

        n_pool = len(abs_eps)
        total  = abs_eps.sum()

        if total == 0.0:
            gini_val      = 0.0
            share_top10pct = 0.0
            share_top5    = 0.0
        else:
            gini_val = gini(abs_eps)
            sorted_desc = np.sort(abs_eps)[::-1]
            k10 = max(1, int(np.ceil(n_pool * 0.1)))
            share_top10pct = float(sorted_desc[:k10].sum() / total)
            share_top5     = float(sorted_desc[:5].sum()  / total)

        n_nonzero = int((abs_eps > 0).sum())

        rows.append({
            "query_id":        qid,
            "n_pool":          n_pool,
            "mean_abs_eps":    float(abs_eps.mean()),
            "std_abs_eps":     float(abs_eps.std()),
            "max_abs_eps":     float(abs_eps.max()),
            "gini_abs_eps":    float(gini_val),
            "share_top10pct":  float(share_top10pct),
            "share_top5":      float(share_top5),
            "n_nonzero_eps":   n_nonzero,
            "frac_nonzero_eps": float(n_nonzero / n_pool),
        })

    return rows, per_year_abs_eps


# ── Measurement 2: Top-10 retrieval heterogeneity across systems ─────────────

def compute_retrieval_heterogeneity(queries, runs, human_qrels, llm_qrels):
    """
    For each query, measure pairwise Jaccard overlap of top-10 lists and pool coverage.
    """
    system_names = sorted(runs.keys())
    rows = []

    for qid in queries:
        hq   = human_qrels.get(qid, {})
        lq   = llm_qrels.get(qid, {})
        pool = set(hq.keys()) | set(lq.keys())

        top10s = [set(runs[s].get(qid, [])[:10]) for s in system_names]
        n_sys  = len(top10s)

        # Pairwise Jaccard
        jaccards = []
        for i in range(n_sys):
            for j in range(i + 1, n_sys):
                inter = len(top10s[i] & top10s[j])
                union = len(top10s[i] | top10s[j])
                jaccards.append(inter / union if union > 0 else 1.0)
        jaccards = np.array(jaccards)

        # Pool coverage
        all_top10 = set()
        for t in top10s:
            all_top10 |= t
        pool_coverage = len(all_top10 & pool) / len(pool) if pool else 0.0
        n_unique = len(all_top10)

        # Mean retrieval count (over passages appearing in ≥1 top-10)
        ret_cnt = defaultdict(int)
        for t in top10s:
            for pid in t:
                ret_cnt[pid] += 1
        mean_ret_count = (sum(ret_cnt.values()) / n_unique) if n_unique > 0 else 0.0

        rows.append({
            "query_id":               qid,
            "n_systems":              n_sys,
            "mean_jaccard":           float(jaccards.mean())   if len(jaccards) else 0.0,
            "median_jaccard":         float(np.median(jaccards)) if len(jaccards) else 0.0,
            "min_jaccard":            float(jaccards.min())    if len(jaccards) else 0.0,
            "max_jaccard":            float(jaccards.max())    if len(jaccards) else 0.0,
            "pool_coverage":          float(pool_coverage),
            "n_unique_top10_passages": n_unique,
            "mean_retrieval_count":   float(mean_ret_count),
        })

    return rows


# ── Measurement 3: Alignment between LLM errors and retrieval frequency ──────

def compute_eps_retrieval_alignment(queries, runs, human_qrels, llm_qrels):
    """
    For each query, compute Kendall tau between |ε(p)| and retrieval_freq(p),
    and summary fractions for the high-|ε| subset.

    Returns
    -------
    rows        : list of dicts (one per query)
    scatter_data : list of (abs_eps, retrieval_freq) tuples pooled across queries
    """
    system_names = sorted(runs.keys())
    N = len(system_names)
    rows = []
    scatter_data = []   # list of (abs_eps, retrieval_freq)

    for qid in queries:
        hq   = human_qrels.get(qid, {})
        lq   = llm_qrels.get(qid, {})
        pool = set(hq.keys()) | set(lq.keys())

        if not pool:
            continue

        # Retrieval frequency per passage
        ret_freq = defaultdict(int)
        for s in system_names:
            for pid in runs[s].get(qid, [])[:10]:
                ret_freq[pid] += 1

        pids     = sorted(pool)
        abs_eps  = np.array([
            abs(exp_gain(lq.get(p, 0)) - exp_gain(hq.get(p, 0)))
            for p in pids
        ])
        freq_arr = np.array([ret_freq.get(p, 0) for p in pids])

        for ae, fr in zip(abs_eps, freq_arr):
            scatter_data.append((float(ae), int(fr)))

        n_pool = len(abs_eps)

        # Kendall tau
        if n_pool > 1:
            tau, _ = kendalltau(abs_eps, freq_arr)
            tau = 0.0 if np.isnan(tau) else float(tau)
        else:
            tau = 0.0

        # Top/bottom 10% by retrieval frequency
        k10 = max(1, int(np.ceil(n_pool * 0.1)))
        top_freq_idx    = np.argsort(freq_arr)[::-1][:k10]
        bottom_freq_idx = np.argsort(freq_arr)[:k10]

        mean_eps_top_freq    = float(abs_eps[top_freq_idx].mean())
        mean_eps_bottom_freq = float(abs_eps[bottom_freq_idx].mean())

        # Top 10% by |ε| — fraction unretrieved vs widely retrieved
        high_eps_idx   = np.argsort(abs_eps)[::-1][:k10]
        high_eps_freqs = freq_arr[high_eps_idx]

        frac_unretrieved = float((high_eps_freqs == 0).sum()        / k10)
        frac_widely      = float((high_eps_freqs >= N / 2).sum()    / k10) if N > 0 else 0.0

        rows.append({
            "query_id":                        qid,
            "kendall_tau_eps_freq":            tau,
            "mean_eps_top10pct_retrieval":     mean_eps_top_freq,
            "mean_eps_bottom10pct_retrieval":  mean_eps_bottom_freq,
            "fraction_high_eps_unretrieved":   frac_unretrieved,
            "fraction_high_eps_widely_retrieved": frac_widely,
        })

    return rows, scatter_data


# ── Plotting helpers ─────────────────────────────────────────────────────────

YEARS_SORTED = sorted(YEARS.keys())


def _setup_year_axes(figsize):
    fig, axes = plt.subplots(1, 5, figsize=figsize)
    return fig, axes


def plot_eps_cdf(per_year_abs_eps, out_path):
    """Empirical CDF of |ε(p)| per year (log x-axis)."""
    fig, axes = _setup_year_axes((20, 4))
    for yi, year in enumerate(YEARS_SORTED):
        ax  = axes[yi]
        vals = np.array(per_year_abs_eps[year])
        pos  = vals[vals > 0]  # log scale: discard exact zeros
        if len(pos) == 0:
            ax.set_title(f"DL {year}")
            continue
        sorted_vals = np.sort(pos)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(vals)   # denominator = all passages
        ax.step(sorted_vals, cdf, where="post", linewidth=1.0, color="tab:blue")
        ax.set_xscale("log")
        ax.set_xlabel(r"$|\varepsilon(p)|$", fontsize=9)
        ax.set_ylabel("Empirical CDF", fontsize=9)
        ax.set_title(f"DL {year}\n(n={len(vals):,} passages)", fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(True, which="both", alpha=0.3)
    plt.suptitle(r"Empirical CDF of $|\varepsilon(p)|$ (log x)", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_gini_hist(eps_df, out_path):
    """Histogram of gini_abs_eps across queries per year."""
    fig, axes = _setup_year_axes((20, 4))
    for yi, year in enumerate(YEARS_SORTED):
        ax   = axes[yi]
        vals = eps_df[eps_df["year"] == year]["gini_abs_eps"].dropna().values
        if len(vals) == 0:
            ax.set_title(f"DL {year}")
            continue
        ax.hist(vals, bins=20, color="tab:orange", edgecolor="white", linewidth=0.4)
        med = np.median(vals)
        ax.axvline(med, color="black", linestyle="--", linewidth=1.2,
                   label=f"median={med:.2f}")
        ax.set_xlabel("Gini(|ε|)", fontsize=9)
        ax.set_ylabel("Queries", fontsize=9)
        ax.set_title(f"DL {year}  (n={len(vals)} queries)", fontsize=9)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
    plt.suptitle("Gini coefficient of |ε(p)| per query", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_jaccard_hist(het_df, out_path):
    """Histogram of mean_jaccard across queries per year, median marked."""
    fig, axes = _setup_year_axes((20, 4))
    for yi, year in enumerate(YEARS_SORTED):
        ax   = axes[yi]
        vals = het_df[het_df["year"] == year]["mean_jaccard"].dropna().values
        if len(vals) == 0:
            ax.set_title(f"DL {year}")
            continue
        ax.hist(vals, bins=20, color="tab:green", edgecolor="white", linewidth=0.4)
        med = np.median(vals)
        ax.axvline(med, color="black", linestyle="--", linewidth=1.2,
                   label=f"median={med:.3f}")
        ax.set_xlabel("Mean pairwise Jaccard (top-10)", fontsize=9)
        ax.set_ylabel("Queries", fontsize=9)
        ax.set_title(f"DL {year}  (n={len(vals)} queries)", fontsize=9)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
    plt.suptitle("Top-10 retrieval heterogeneity (mean pairwise Jaccard)", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_pool_coverage_hist(het_df, out_path):
    """Histogram of pool_coverage across queries per year."""
    fig, axes = _setup_year_axes((20, 4))
    for yi, year in enumerate(YEARS_SORTED):
        ax   = axes[yi]
        vals = het_df[het_df["year"] == year]["pool_coverage"].dropna().values
        if len(vals) == 0:
            ax.set_title(f"DL {year}")
            continue
        ax.hist(vals, bins=20, color="tab:purple", edgecolor="white", linewidth=0.4)
        med = np.median(vals)
        ax.axvline(med, color="black", linestyle="--", linewidth=1.2,
                   label=f"median={med:.3f}")
        ax.set_xlabel("Pool coverage (fraction in ≥1 top-10)", fontsize=9)
        ax.set_ylabel("Queries", fontsize=9)
        ax.set_title(f"DL {year}  (n={len(vals)} queries)", fontsize=9)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
    plt.suptitle("Fraction of judged pool covered by system top-10 lists", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_eps_vs_freq_scatter(per_year_scatter, out_path):
    """Scatter of |ε(p)| vs retrieval_freq per year, alpha=0.3."""
    fig, axes = _setup_year_axes((20, 4))
    for yi, year in enumerate(YEARS_SORTED):
        ax   = axes[yi]
        data = per_year_scatter[year]
        if not data:
            ax.set_title(f"DL {year}")
            continue
        ae  = np.array([d[0] for d in data])
        fr  = np.array([d[1] for d in data])
        ax.scatter(fr, ae, alpha=0.3, s=3, color="tab:red", rasterized=True)
        ax.set_xlabel("Retrieval frequency (# systems, top-10)", fontsize=9)
        ax.set_ylabel(r"$|\varepsilon(p)|$", fontsize=9)
        ax.set_title(f"DL {year}  (n={len(ae):,} passages)", fontsize=9)
        ax.grid(True, alpha=0.2)
    plt.suptitle(r"$|\varepsilon(p)|$ vs retrieval frequency", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_tau_hist(align_df, out_path):
    """Histogram of kendall_tau_eps_freq across queries per year."""
    fig, axes = _setup_year_axes((20, 4))
    for yi, year in enumerate(YEARS_SORTED):
        ax   = axes[yi]
        vals = align_df[align_df["year"] == year]["kendall_tau_eps_freq"].dropna().values
        if len(vals) == 0:
            ax.set_title(f"DL {year}")
            continue
        ax.hist(vals, bins=20, color="tab:cyan", edgecolor="white", linewidth=0.4)
        med = np.median(vals)
        ax.axvline(med, color="black", linestyle="--", linewidth=1.2,
                   label=f"median={med:.3f}")
        ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel(r"Kendall $\tau$(|ε|, retrieval_freq)", fontsize=9)
        ax.set_ylabel("Queries", fontsize=9)
        ax.set_title(f"DL {year}  (n={len(vals)} queries)", fontsize=9)
        ax.legend(fontsize=8)
    plt.suptitle(r"Kendall $\tau$ between $|\varepsilon(p)|$ and retrieval frequency", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  DIAGNOSTIC MEASUREMENTS: WHY QUERY-LEVEL TRIAGE HAS A SMALL CEILING")
    print("=" * 70)

    # Feature query set (same filter as run_impact_oracle.py)
    feature_queries = load_feature_queries()

    # Accumulators
    all_eps_rows   = []
    all_het_rows   = []
    all_align_rows = []
    per_year_abs_eps = {}   # year -> list of floats
    per_year_scatter = {}   # year -> list of (abs_eps, freq)
    summary_rows   = []

    for year, cfg in sorted(YEARS.items()):
        print(f"\n{'─'*60}")
        print(f"  Year {year}")
        print(f"{'─'*60}")

        # ── Load data (same filtering as run_impact_oracle.py) ──────────────
        human_qrels  = load_qrels(cfg["qrels"])
        year_queries = set(human_qrels.keys())
        llm_qrels    = load_llm_qrels(cfg["scores"], year_queries)
        year_queries &= set(llm_qrels.keys())
        year_queries &= feature_queries
        runs         = load_system_runs(cfg["runs_dir"])
        if year in V2_YEARS:
            canonicalize_runs(runs, load_canonical_map())

        queries      = sorted(year_queries)
        system_names = sorted(runs.keys())
        n_q          = len(queries)
        n_sys        = len(system_names)
        print(f"  {n_q} queries, {n_sys} systems")

        # ── Measurement 1 ───────────────────────────────────────────────────
        print("  [M1] Computing epsilon distributions...")
        eps_rows, abs_eps_pool = compute_eps_distribution(queries, human_qrels, llm_qrels)
        for r in eps_rows:
            r["year"] = year
        all_eps_rows.extend(eps_rows)
        per_year_abs_eps[year] = abs_eps_pool

        # ── Measurement 2 ───────────────────────────────────────────────────
        print("  [M2] Computing retrieval heterogeneity...")
        het_rows = compute_retrieval_heterogeneity(queries, runs, human_qrels, llm_qrels)
        for r in het_rows:
            r["year"] = year
        all_het_rows.extend(het_rows)

        # ── Measurement 3 ───────────────────────────────────────────────────
        print("  [M3] Computing epsilon-retrieval alignment...")
        align_rows, scatter = compute_eps_retrieval_alignment(
            queries, runs, human_qrels, llm_qrels
        )
        for r in align_rows:
            r["year"] = year
        all_align_rows.extend(align_rows)
        per_year_scatter[year] = scatter

        # ── Per-year summary ────────────────────────────────────────────────
        eps_df   = pd.DataFrame(eps_rows)
        het_df   = pd.DataFrame(het_rows)
        align_df = pd.DataFrame(align_rows)

        med_gini      = float(eps_df["gini_abs_eps"].median())
        med_top10pct  = float(eps_df["share_top10pct"].median())
        med_jaccard   = float(het_df["mean_jaccard"].median())
        med_pool_cov  = float(het_df["pool_coverage"].median())
        med_tau       = float(align_df["kendall_tau_eps_freq"].median())
        med_unretr    = float(align_df["fraction_high_eps_unretrieved"].median())
        med_widely    = float(align_df["fraction_high_eps_widely_retrieved"].median())

        summary_rows.append({
            "year":                               year,
            "n_queries":                          n_q,
            "n_systems":                          n_sys,
            "median_gini_eps":                    med_gini,
            "median_share_top10pct":              med_top10pct,
            "median_mean_jaccard":                med_jaccard,
            "median_pool_coverage":               med_pool_cov,
            "median_kendall_tau_eps_freq":        med_tau,
            "median_fraction_high_eps_unretrieved":      med_unretr,
            "median_fraction_high_eps_widely_retrieved": med_widely,
        })

        print(f"\n  Summary statistics for DL {year}:")
        print(f"    median_gini_eps                          = {med_gini:.4f}")
        print(f"    median_share_top10pct                    = {med_top10pct:.4f}")
        print(f"    median_mean_jaccard                      = {med_jaccard:.4f}")
        print(f"    median_pool_coverage                     = {med_pool_cov:.4f}")
        print(f"    median_kendall_tau_eps_freq              = {med_tau:.4f}")
        print(f"    median_fraction_high_eps_unretrieved     = {med_unretr:.4f}")
        print(f"    median_fraction_high_eps_widely_retrieved= {med_widely:.4f}")

        # Interpretation (two sentences)
        p2_strong = med_gini > 0.7 and med_top10pct > 0.8
        p2_mod    = med_gini > 0.5 or med_top10pct > 0.6
        p2 = ("strongly" if p2_strong else "moderately" if p2_mod else "weakly")

        p3_strong = med_jaccard < 0.3
        p3_mod    = med_jaccard < 0.5
        p3 = ("strongly" if p3_strong else "moderately" if p3_mod else "weakly")

        print(f"\n  Interpretation: DL {year} {p2} supports Pillar 2 "
              f"(LLM errors concentrated on ~10% of pool passages; "
              f"aggregating all ~{int(eps_df['n_pool'].median())} pool passages into one scalar discards this signal).")
        print(f"  DL {year} {p3} supports Pillar 3 "
              f"(systems share only {med_jaccard:.1%} of their top-10 passages on average, "
              f"so the same LLM error hits different system windows).")

    # ── Save CSVs ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SAVING CSV OUTPUTS")
    print("=" * 70)

    eps_all_df   = pd.DataFrame(all_eps_rows)
    het_all_df   = pd.DataFrame(all_het_rows)
    align_all_df = pd.DataFrame(all_align_rows)
    summary_df   = pd.DataFrame(summary_rows)

    eps_cols = [
        "year", "query_id", "n_pool", "mean_abs_eps", "std_abs_eps", "max_abs_eps",
        "gini_abs_eps", "share_top10pct", "share_top5", "n_nonzero_eps", "frac_nonzero_eps",
    ]
    het_cols = [
        "year", "query_id", "n_systems", "mean_jaccard", "median_jaccard",
        "min_jaccard", "max_jaccard", "pool_coverage",
        "n_unique_top10_passages", "mean_retrieval_count",
    ]
    align_cols = [
        "year", "query_id", "kendall_tau_eps_freq",
        "mean_eps_top10pct_retrieval", "mean_eps_bottom10pct_retrieval",
        "fraction_high_eps_unretrieved", "fraction_high_eps_widely_retrieved",
    ]

    eps_path    = OUTPUT_DIR / "eps_distribution.csv"
    het_path    = OUTPUT_DIR / "retrieval_heterogeneity.csv"
    align_path  = OUTPUT_DIR / "eps_retrieval_alignment.csv"
    summary_path = OUTPUT_DIR / "summary.csv"

    eps_all_df[eps_cols].to_csv(eps_path,     index=False)
    het_all_df[het_cols].to_csv(het_path,     index=False)
    align_all_df[align_cols].to_csv(align_path, index=False)
    summary_df.to_csv(summary_path,            index=False)

    for p in [eps_path, het_path, align_path, summary_path]:
        print(f"  Saved {p}")

    # ── Generate plots ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  GENERATING FIGURES")
    print("=" * 70)

    p = OUTPUT_DIR / "eps_distribution.pdf"
    plot_eps_cdf(per_year_abs_eps, p)
    print(f"  Saved {p}")

    p = OUTPUT_DIR / "gini_distribution.pdf"
    plot_gini_hist(eps_all_df, p)
    print(f"  Saved {p}")

    p = OUTPUT_DIR / "jaccard_distribution.pdf"
    plot_jaccard_hist(het_all_df, p)
    print(f"  Saved {p}")

    p = OUTPUT_DIR / "pool_coverage_distribution.pdf"
    plot_pool_coverage_hist(het_all_df, p)
    print(f"  Saved {p}")

    p = OUTPUT_DIR / "eps_vs_retrieval_freq.pdf"
    plot_eps_vs_freq_scatter(per_year_scatter, p)
    print(f"  Saved {p}")

    p = OUTPUT_DIR / "eps_retrieval_tau.pdf"
    plot_tau_hist(align_all_df, p)
    print(f"  Saved {p}")

    print(f"\nDone. All outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
