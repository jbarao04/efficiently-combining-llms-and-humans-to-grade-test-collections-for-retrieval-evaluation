# Within-grade permutation null test for year 2019

import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=FutureWarning)

from triage.experiment_correction_resampling import (
    rank_systems, compute_tau, compute_tau_at_k,
)


def compute_displacement(epsilon_map, runs_top10, queries, system_names, k=10,
                         normalize=False, qrels_for_idcg=None):
    m = len(system_names)
    result = np.zeros(m)
    for si, sys_name in enumerate(system_names):
        sys_run = runs_top10.get(sys_name, {})
        total = 0.0
        for qid in queries:
            ranked = sys_run.get(qid, [])
            dcg_shift = sum(
                epsilon_map.get((qid, pid), 0) / math.log2(i + 2)
                for i, pid in enumerate(ranked[:k])
            )
            if normalize and qrels_for_idcg is not None:
                hq = qrels_for_idcg.get(qid, {})
                ideal_gains = sorted(hq.values(), reverse=True)[:k]
                idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
                dcg_shift = dcg_shift / idcg if idcg > 0 else 0.0
            total += dcg_shift
        result[si] = total
    return result


def compute_score_bias(epsilon_map, runs_top10, queries, system_names, k=10):
    m = len(system_names)
    result = np.zeros(m)
    for si, sys_name in enumerate(system_names):
        errs = []
        sys_run = runs_top10.get(sys_name, {})
        for qid in queries:
            ranked = sys_run.get(qid, [])
            for pid in ranked[:k]:
                errs.append(epsilon_map.get((qid, pid), 0))
        result[si] = np.mean(errs) if errs else 0.0
    return result

BASE_DIR = Path(__file__).resolve().parent.parent
INT_DIR  = BASE_DIR / "results" / "spectral" / "intermediates"
HARD_DIR = BASE_DIR / "results" / "spectral" / "hardening"
OUT_DIR  = BASE_DIR / "results" / "thesis_verification" / "t15_permutation"

YEARS   = [2019, 2020, 2021, 2022, 2023]
K       = 10
SEED    = 42       # same as existing test
N_PERMS = 1000     # same as existing test

W    = np.array([1.0 / math.log2(r + 1) for r in range(1, K + 1)])


# ── Loading ────────────────────────────────────────────────────────────────

def load_year(year):
    """Load parquets for a year (mirrors run_hardening.load_year)."""
    ydir = INT_DIR / str(year)
    df_eps = pd.read_parquet(ydir / "epsilon.parquet")
    df_rt  = pd.read_parquet(ydir / "runs_top10.parquet")
    df_sys = pd.read_parquet(ydir / "systems.parquet")

    # Validate eps == g_llm - g_human
    bad = (df_eps["eps"] != df_eps["g_llm"] - df_eps["g_human"]).sum()
    assert bad == 0, f"{year}: {bad} rows where eps != g_llm - g_human"

    epsilon_map   = {(r.qid, r.pid): r.eps
                     for r in df_eps.itertuples(index=False)}
    g_human_map   = {(r.qid, r.pid): r.g_human
                     for r in df_eps.itertuples(index=False)}

    runs_top10 = {}
    for sys_name, grp in df_rt.groupby("system"):
        runs_top10[sys_name] = {}
        for qid, qgrp in grp.groupby("qid"):
            runs_top10[sys_name][qid] = qgrp.sort_values("rank")["pid"].tolist()

    system_names = sorted(df_sys["system"].tolist())
    queries      = sorted(df_rt["qid"].unique().tolist())

    # Compute DCG per system
    dcg_human = np.zeros(len(system_names))
    for si, s in enumerate(system_names):
        for q in queries:
            ranked = runs_top10[s].get(q, [])
            for ri, pid in enumerate(ranked[:K]):
                w = 1.0 / math.log2(ri + 2)
                dcg_human[si] += g_human_map.get((q, pid), 0) * w

    disp_dcg  = compute_displacement(epsilon_map, runs_top10, queries,
                                     system_names, k=K)
    score_bias = compute_score_bias(epsilon_map, runs_top10, queries,
                                    system_names, k=K)

    return {
        "epsilon_map":   epsilon_map,
        "g_human_map":   g_human_map,
        "runs_top10":    runs_top10,
        "system_names":  system_names,
        "queries":       queries,
        "dcg_human":     dcg_human,
        "disp_dcg":      disp_dcg,
        "score_bias":    score_bias,
    }


# ── Exposure sets ──────────────────────────────────────────────────────────

def build_free_exposure(data):
    """Return per-query dict: qid -> (pids_sorted, eps_array)."""
    queries, runs_top10, epsilon_map = (
        data["queries"], data["runs_top10"], data["epsilon_map"])
    system_names = data["system_names"]
    per_q = {}
    for q in queries:
        pids = set()
        for s in system_names:
            pids.update(runs_top10[s].get(q, []))
        pids_list = sorted(pids)
        eps_vals  = np.array([epsilon_map.get((q, pid), 0)
                              for pid in pids_list])
        per_q[q] = (pids_list, eps_vals)
    return per_q


def build_within_grade_exposure(data):
    """
    Return per-query dict: qid -> list of (pids_in_grade, eps_in_grade) per grade.

    Passages are partitioned by g_human (human relevance grade, 0-3).
    Only grades with >= 2 passages can be shuffled.
    Grades with < 2 passages are returned with a 'fixed' flag.

    Also computes unshufflable statistics.
    """
    queries, runs_top10, epsilon_map, g_human_map = (
        data["queries"], data["runs_top10"],
        data["epsilon_map"], data["g_human_map"])
    system_names = data["system_names"]

    per_q = {}
    # grade_counts[grade] = (n_groups_size_lt2, n_passages_fixed)
    grade_counts = defaultdict(lambda: [0, 0])

    for q in queries:
        pids = set()
        for s in system_names:
            pids.update(runs_top10[s].get(q, []))
        pids_list = sorted(pids)

        # Partition by human grade (missing = 0)
        grade_groups = defaultdict(list)
        for pid in pids_list:
            g = int(g_human_map.get((q, pid), 0))
            grade_groups[g].append(pid)

        # Build per-grade (pids, eps, shufflable) tuples
        q_groups = []
        for grade in sorted(grade_groups.keys()):
            gpids = grade_groups[grade]
            geps  = np.array([epsilon_map.get((q, pid), 0) for pid in gpids])
            shufflable = len(gpids) >= 2
            if not shufflable:
                grade_counts[grade][0] += 1
                grade_counts[grade][1] += len(gpids)
            q_groups.append((gpids, geps, shufflable))

        per_q[q] = q_groups

    return per_q, grade_counts


# ── Shufflers ─────────────────────────────────────────────────────────────

def shuffle_free(per_q_exposure, queries, rng):
    """Standard within-query free shuffle (mirrors run_hardening)."""
    eps_null = {}
    for q in queries:
        pids_list, eps_vals = per_q_exposure[q]
        shuffled = eps_vals.copy()
        rng.shuffle(shuffled)
        for pid, ev in zip(pids_list, shuffled):
            eps_null[(q, pid)] = int(ev)
    return eps_null


def shuffle_within_grade(per_q_wg, queries, rng):
    """Within-grade shuffle: permute ε only among passages sharing the same human grade."""
    eps_null = {}
    for q in queries:
        for gpids, geps, shufflable in per_q_wg[q]:
            if shufflable:
                shuffled = geps.copy()
                rng.shuffle(shuffled)
            else:
                shuffled = geps  # leave in place (size 0 or 1)
            for pid, ev in zip(gpids, shuffled):
                eps_null[(q, pid)] = int(ev)
    return eps_null


# ── Core permutation runner ────────────────────────────────────────────────

def run_permutation(data, null_type, n_perms=N_PERMS):
    """
    Run n_perms permutations under a given null type.

    null_type: 'free' or 'within_grade'

    Returns dict of observed/null statistics and a distributions DataFrame.
    """
    system_names = data["system_names"]
    queries      = data["queries"]
    runs_top10   = data["runs_top10"]
    dcg_human    = data["dcg_human"]
    disp_dcg     = data["disp_dcg"]
    score_bias   = data["score_bias"]

    # Observed statistics
    obs_var_disp  = float(np.var(disp_dcg))
    obs_corr_comp, _ = spearmanr(disp_dcg, dcg_human)
    obs_tau20     = compute_tau_at_k(
        rank_systems(dcg_human, system_names),
        rank_systems(dcg_human + disp_dcg, system_names),
        20)

    # Build exposure structures
    if null_type == "free":
        per_q_free = build_free_exposure(data)
        per_q_wg   = None
    elif null_type == "within_grade":
        per_q_free = None
        per_q_wg, _ = build_within_grade_exposure(data)
    else:
        raise ValueError(f"Unknown null_type: {null_type}")

    rng = np.random.RandomState(SEED)

    null_var_disp  = np.zeros(n_perms)
    null_corr_comp = np.zeros(n_perms)
    null_tau20     = np.zeros(n_perms)

    for pi in range(n_perms):
        if null_type == "free":
            eps_null = shuffle_free(per_q_free, queries, rng)
        else:
            eps_null = shuffle_within_grade(per_q_wg, queries, rng)

        disp_null = compute_displacement(eps_null, runs_top10, queries,
                                         system_names, k=K)
        bias_null = compute_score_bias(eps_null, runs_top10, queries,
                                        system_names, k=K)

        null_var_disp[pi]  = float(np.var(disp_null))
        rho, _ = spearmanr(disp_null, dcg_human)
        null_corr_comp[pi] = float(rho) if not np.isnan(rho) else 0.0
        null_tau20[pi]     = compute_tau_at_k(
            rank_systems(dcg_human, system_names),
            rank_systems(dcg_human + disp_null, system_names),
            20)

    def p_upper(obs, null_arr):
        """One-sided upper: fraction of null >= obs."""
        return float(np.mean(null_arr >= obs))

    def p_two_sided(obs, null_arr):
        return float(np.mean(np.abs(null_arr) >= np.abs(obs)))

    def p_lower(obs, null_arr):
        """One-sided lower: fraction of null <= obs."""
        return float(np.mean(null_arr <= obs))

    stats = {
        # displacement spread
        "var_disp_obs":       obs_var_disp,
        "var_disp_null_mean": float(null_var_disp.mean()),
        "var_disp_null_sd":   float(null_var_disp.std()),
        "var_disp_ratio":     obs_var_disp / null_var_disp.mean() if null_var_disp.mean() > 0 else np.nan,
        "var_disp_p":         p_upper(obs_var_disp, null_var_disp),
        # compression corr
        "corr_comp_obs":       float(obs_corr_comp),
        "corr_comp_null_mean": float(null_corr_comp.mean()),
        "corr_comp_null_sd":   float(null_corr_comp.std()),
        "corr_comp_ratio":     (abs(obs_corr_comp) / abs(null_corr_comp).mean()
                                if abs(null_corr_comp).mean() > 0 else np.nan),
        "corr_comp_p":         p_two_sided(obs_corr_comp, null_corr_comp),
        # tau@20
        "tau20_obs":       obs_tau20,
        "tau20_null_mean": float(null_tau20.mean()),
        "tau20_null_sd":   float(null_tau20.std()),
        "tau20_p":         p_lower(obs_tau20, null_tau20),
    }

    dist_df = pd.DataFrame({
        "null_type":       null_type,
        "perm":            range(n_perms),
        "var_disp":        null_var_disp,
        "corr_comp":       null_corr_comp,
        "tau20":           null_tau20,
    })

    return stats, dist_df


# ── Unshufflable count reporter ────────────────────────────────────────────

def compute_unshufflable_counts(data, year):
    """Count passages in groups of size < 2 per (year, grade)."""
    _, grade_counts = build_within_grade_exposure(data)
    rows = []
    for grade in sorted(grade_counts.keys()):
        n_grp_lt2, n_pass_fixed = grade_counts[grade]
        rows.append({
            "year":              year,
            "human_grade":       grade,
            "n_groups_size_lt2": n_grp_lt2,
            "n_passages_fixed":  n_pass_fixed,
        })
    return rows


# ── Per-year runner ────────────────────────────────────────────────────────

def process_year(year):
    print(f"\n{'=' * 70}")
    print(f"YEAR {year}")
    print(f"{'=' * 70}")
    data = load_year(year)
    print(f"  Loaded: {len(data['system_names'])} systems, "
          f"{len(data['queries'])} queries")

    # Unshufflable counts
    unc_rows = compute_unshufflable_counts(data, year)
    total_fixed = sum(r["n_passages_fixed"] for r in unc_rows)
    print(f"  Unshufflable passages (size-1 grade groups): {total_fixed}")
    for r in unc_rows:
        if r["n_passages_fixed"] > 0:
            print(f"    grade {r['human_grade']}: "
                  f"{r['n_groups_size_lt2']} groups, "
                  f"{r['n_passages_fixed']} passages fixed")

    # Free shuffle (replicate existing test for comparison)
    print(f"\n  Running FREE shuffle ({N_PERMS} perms)...")
    stats_free, dist_free = run_permutation(data, "free")
    ratio_free = stats_free["var_disp_ratio"]
    print(f"  FREE  var_disp: obs={stats_free['var_disp_obs']:.2f}, "
          f"null_mean={stats_free['var_disp_null_mean']:.2f}, "
          f"ratio={ratio_free:.2f}×, p={stats_free['var_disp_p']:.4f}")
    print(f"  FREE  corr_comp: obs={stats_free['corr_comp_obs']:.4f}, "
          f"null_mean={stats_free['corr_comp_null_mean']:.4f}, "
          f"p={stats_free['corr_comp_p']:.4f}")
    print(f"  FREE  tau@20: obs={stats_free['tau20_obs']:.4f}, "
          f"null_mean={stats_free['tau20_null_mean']:.4f}, "
          f"p(null≤obs)={stats_free['tau20_p']:.4f}")

    # Within-grade shuffle
    print(f"\n  Running WITHIN-GRADE shuffle ({N_PERMS} perms)...")
    stats_wg, dist_wg = run_permutation(data, "within_grade")
    ratio_wg = stats_wg["var_disp_ratio"]
    print(f"  WG    var_disp: obs={stats_wg['var_disp_obs']:.2f}, "
          f"null_mean={stats_wg['var_disp_null_mean']:.2f}, "
          f"ratio={ratio_wg:.2f}×, p={stats_wg['var_disp_p']:.4f}")
    print(f"  WG    corr_comp: obs={stats_wg['corr_comp_obs']:.4f}, "
          f"null_mean={stats_wg['corr_comp_null_mean']:.4f}, "
          f"p={stats_wg['corr_comp_p']:.4f}")
    print(f"  WG    tau@20: obs={stats_wg['tau20_obs']:.4f}, "
          f"null_mean={stats_wg['tau20_null_mean']:.4f}, "
          f"p(null≤obs)={stats_wg['tau20_p']:.4f}")

    attenuation = ratio_wg / ratio_free if ratio_free > 0 else np.nan
    print(f"\n  Ratio WG/free: {attenuation:.3f} "
          f"({'SURVIVES' if ratio_wg >= 2.0 else 'WEAKENED (<2x)'})")

    return {
        "stats_free": stats_free,
        "stats_wg":   stats_wg,
        "dist_free":  dist_free,
        "dist_wg":    dist_wg,
        "unc_rows":   unc_rows,
        "attenuation": attenuation,
        "ratio_free":  ratio_free,
        "ratio_wg":    ratio_wg,
    }


# ── Output builders ────────────────────────────────────────────────────────

def build_permutation_comparison_row(year, null_type, stats):
    """Build rows for permutation_comparison.csv."""
    rows = []
    for stat_key, label in [
        ("var_disp",   "displacement_spread"),
        ("corr_comp",  "compression_corr"),
        ("tau20",      "tau_at20_degradation"),
    ]:
        obs       = stats[f"{stat_key}_obs"]
        null_mean = stats[f"{stat_key}_null_mean"]
        null_sd   = stats[f"{stat_key}_null_sd"]
        p         = stats[f"{stat_key}_p"]
        if null_mean != 0:
            ratio = abs(obs) / abs(null_mean) if label != "displacement_spread" else obs / null_mean
        else:
            ratio = np.nan
        rows.append({
            "year":                  year,
            "null_type":             null_type,
            "statistic":             label,
            "observed":              obs,
            "null_mean":             null_mean,
            "null_sd":               null_sd,
            "ratio_obs_to_null_mean": ratio,
            "p_one_sided":           p,
            "n_permutations":        N_PERMS,
        })
    return rows


def build_signature_comparison_row(year, null_type, stats):
    """Build rows for signature_comparison.csv."""
    return [
        {
            "year":       year,
            "null_type":  null_type,
            "signature":  "displacement_spread",
            "observed":   stats["var_disp_obs"],
            "null_mean":  stats["var_disp_null_mean"],
            "p_value":    stats["var_disp_p"],
        },
        {
            "year":       year,
            "null_type":  null_type,
            "signature":  "compression_corr",
            "observed":   stats["corr_comp_obs"],
            "null_mean":  stats["corr_comp_null_mean"],
            "p_value":    stats["corr_comp_p"],
        },
        {
            "year":       year,
            "null_type":  null_type,
            "signature":  "tau_at20_degradation",
            "observed":   stats["tau20_obs"],
            "null_mean":  stats["tau20_null_mean"],
            "p_value":    stats["tau20_p"],
        },
    ]


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which years to run
    if len(sys.argv) > 1:
        years_to_run = [int(sys.argv[1])]
    else:
        years_to_run = YEARS

    perm_comp_rows  = []
    sig_comp_rows   = []
    unc_all_rows    = []
    results_by_year = {}

    for year in years_to_run:
        result = process_year(year)
        results_by_year[year] = result

        # Check 2019 stopping condition
        if year == 2019 and 2019 in years_to_run and len(years_to_run) == 1:
            ratio_wg = result["ratio_wg"]
            print(f"\n{'=' * 70}")
            print(f"STOP CHECK (2019 only run)")
            print(f"Within-grade ratio = {ratio_wg:.2f}×")
            if ratio_wg < 2.0:
                print("WARNING: ratio < 2×. Thesis text needs rethinking "
                      "before running remaining years.")
            else:
                print("PROCEED: ratio >= 2×. Safe to run remaining years.")
            print(f"{'=' * 70}")

        # Accumulate output rows
        for null_type, stats in [("free", result["stats_free"]),
                                  ("within_grade", result["stats_wg"])]:
            perm_comp_rows.extend(
                build_permutation_comparison_row(year, null_type, stats))
            sig_comp_rows.extend(
                build_signature_comparison_row(year, null_type, stats))

        unc_all_rows.extend(result["unc_rows"])

    # Write CSVs
    pd.DataFrame(perm_comp_rows).to_csv(
        OUT_DIR / "permutation_comparison.csv", index=False)
    pd.DataFrame(sig_comp_rows).to_csv(
        OUT_DIR / "signature_comparison.csv", index=False)

    # Unshufflable counts (only for years actually run)
    unc_df = pd.DataFrame(unc_all_rows)
    if not unc_df.empty:
        unc_df.to_csv(OUT_DIR / "unshufflable_counts.csv", index=False)
        print(f"\nUnshufflable counts:")
        print(unc_df.to_string(index=False))

    print(f"\nCSVs written to {OUT_DIR}")

    # Print summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY TABLE — var_displacement ratio (obs / null_mean)")
    print(f"{'=' * 70}")
    print(f"{'Year':<6} {'Free':>8} {'WG':>8} {'WG/Free':>10} {'Survives?':>12}")
    print("-" * 50)
    for year in years_to_run:
        if year not in results_by_year:
            continue
        r = results_by_year[year]
        surv = "YES" if r["ratio_wg"] >= 2.0 else "NO (<2×)"
        print(f"{year:<6} {r['ratio_free']:>8.2f} {r['ratio_wg']:>8.2f} "
              f"{r['attenuation']:>10.3f} {surv:>12}")

    # If all years done, write REPORT.md
    if set(years_to_run) == set(YEARS):
        write_report(results_by_year)
        print(f"\nREPORT.md written to {OUT_DIR}")


def write_report(results_by_year):
    """Write REPORT.md summarising findings."""
    lines = []
    lines.append("# Within-Grade Permutation Null — Report (Task t15)")
    lines.append("")
    lines.append("## Summary")
    lines.append("")

    all_survive = all(results_by_year[y]["ratio_wg"] >= 2.0
                      for y in YEARS if y in results_by_year)
    min_ratio_wg = min(results_by_year[y]["ratio_wg"]
                       for y in YEARS if y in results_by_year)
    max_ratio_wg = max(results_by_year[y]["ratio_wg"]
                       for y in YEARS if y in results_by_year)
    mean_atten   = np.mean([results_by_year[y]["attenuation"]
                            for y in YEARS if y in results_by_year])

    if all_survive:
        lines.append(
            f"**The alignment claim survives the within-grade null in all five years.**")
        lines.append("")
        lines.append(
            f"The ratio of observed displacement variance to null mean ranges from "
            f"{min_ratio_wg:.1f}× to {max_ratio_wg:.1f}× under the within-grade null, "
            f"compared to the free-shuffle null which produced 7–18×. "
            f"The mean attenuation factor is {mean_atten:.2f} "
            f"(within-grade ratio / free-shuffle ratio).")
        lines.append("")
        lines.append(
            "This means the effect cannot be explained by the judge's tendency to "
            "over-score passages of particular grade classes. The surviving signal "
            "is attributable to the alignment of errors with the retrieval-weight "
            "covariance structure — exactly the mechanistic claim the thesis makes.")
    else:
        lines.append(
            "**The within-grade null substantially weakens the effect in one or more years.**")
        lines.append("")
        failed = [y for y in YEARS if y in results_by_year
                  and results_by_year[y]["ratio_wg"] < 2.0]
        lines.append(
            f"Years where ratio < 2× under within-grade null: {failed}. "
            "Two sections of the thesis should be softened to match.")

    lines.append("")
    lines.append("## Per-year results")
    lines.append("")
    lines.append("### Displacement variance ratio (obs / null_mean)")
    lines.append("")
    lines.append("| Year | Free (orig) | Within-grade | WG/Free | p (WG) | Survives? |")
    lines.append("|------|------------|--------------|---------|--------|-----------|")
    for year in YEARS:
        if year not in results_by_year:
            continue
        r = results_by_year[year]
        surv = "YES" if r["ratio_wg"] >= 2.0 else "NO"
        lines.append(f"| {year} | {r['ratio_free']:.2f}× | {r['ratio_wg']:.2f}× | "
                     f"{r['attenuation']:.3f} | "
                     f"{r['stats_wg']['var_disp_p']:.4f} | {surv} |")

    lines.append("")
    lines.append("### Compression correlation (Spearman ρ, displacement vs human DCG)")
    lines.append("")
    lines.append("| Year | Observed | Free null_mean | WG null_mean | p (Free) | p (WG) |")
    lines.append("|------|---------|----------------|--------------|----------|--------|")
    for year in YEARS:
        if year not in results_by_year:
            continue
        r = results_by_year[year]
        sf = r["stats_free"]
        sw = r["stats_wg"]
        lines.append(f"| {year} | {sf['corr_comp_obs']:.4f} | "
                     f"{sf['corr_comp_null_mean']:.4f} | "
                     f"{sw['corr_comp_null_mean']:.4f} | "
                     f"{sf['corr_comp_p']:.4f} | "
                     f"{sw['corr_comp_p']:.4f} |")

    lines.append("")
    lines.append("### τ@20 degradation (frac null ≤ obs)")
    lines.append("")
    lines.append("| Year | Observed | Free null_mean | WG null_mean | p (Free) | p (WG) |")
    lines.append("|------|---------|----------------|--------------|----------|--------|")
    for year in YEARS:
        if year not in results_by_year:
            continue
        r = results_by_year[year]
        sf = r["stats_free"]
        sw = r["stats_wg"]
        lines.append(f"| {year} | {sf['tau20_obs']:.4f} | "
                     f"{sf['tau20_null_mean']:.4f} | "
                     f"{sw['tau20_null_mean']:.4f} | "
                     f"{sf['tau20_p']:.4f} | "
                     f"{sw['tau20_p']:.4f} |")

    lines.append("")
    lines.append("## Methodology notes")
    lines.append("")
    lines.append("- **Partition variable**: human relevance grade (g_human ∈ {0, 1, 2, 3}). "
                 "Missing grades in the exposure set are treated as 0.")
    lines.append("- **Why human grade**: partitioning by human grade holds fixed the "
                 "association between error magnitude and relevance tier, which is the "
                 "confound identified by the reviewer. Only within-grade placement is randomised.")
    lines.append("- **Unshufflable groups**: groups of size < 2 are left in place. "
                 "See `unshufflable_counts.csv` for counts per year and grade.")
    lines.append("- **n_perms = 1000**, **SEED = 42** — identical to existing free-shuffle test.")
    lines.append("- **p-values**: upper one-sided for displacement variance "
                 "(frac null ≥ obs); two-sided for compression correlation; "
                 "lower one-sided for τ@20 (frac null ≤ obs).")
    lines.append("- Files written alongside existing results without overwriting "
                 "`results/spectral/hardening/permutation_null.csv` or "
                 "`permutation_null_distributions.parquet`.")

    (OUT_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
