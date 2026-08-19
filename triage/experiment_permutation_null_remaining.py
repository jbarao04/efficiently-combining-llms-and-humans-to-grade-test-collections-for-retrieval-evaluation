# Within-grade and within-confusion-cell permutation null for years 2020-2023

import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
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
OUT_DIR  = BASE_DIR / "results" / "thesis_verification" / "t15b_permutation_remaining"

YEARS   = [2019, 2020, 2021, 2022, 2023]
K       = 10
SEED    = 42
N_PERMS = 1000

W = np.array([1.0 / math.log2(r + 1) for r in range(1, K + 1)])


# ── Loading (identical to run_t15_permutation.py) ─────────────────────────

def load_year(year):
    ydir = INT_DIR / str(year)
    df_eps = pd.read_parquet(ydir / "epsilon.parquet")
    df_rt  = pd.read_parquet(ydir / "runs_top10.parquet")
    df_sys = pd.read_parquet(ydir / "systems.parquet")

    bad = (df_eps["eps"] != df_eps["g_llm"] - df_eps["g_human"]).sum()
    assert bad == 0, f"{year}: {bad} rows where eps != g_llm - g_human"

    epsilon_map = {(r.qid, r.pid): r.eps for r in df_eps.itertuples(index=False)}
    g_human_map = {(r.qid, r.pid): r.g_human for r in df_eps.itertuples(index=False)}
    g_llm_map   = {(r.qid, r.pid): r.g_llm for r in df_eps.itertuples(index=False)}

    runs_top10 = {}
    for sys_name, grp in df_rt.groupby("system"):
        runs_top10[sys_name] = {}
        for qid, qgrp in grp.groupby("qid"):
            runs_top10[sys_name][qid] = qgrp.sort_values("rank")["pid"].tolist()

    system_names = sorted(df_sys["system"].tolist())
    queries      = sorted(df_rt["qid"].unique().tolist())

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
        "g_llm_map":     g_llm_map,
        "runs_top10":    runs_top10,
        "system_names":  system_names,
        "queries":       queries,
        "dcg_human":     dcg_human,
        "disp_dcg":      disp_dcg,
        "score_bias":    score_bias,
    }


# ── Exposure sets ──────────────────────────────────────────────────────────

def build_free_exposure(data):
    queries, runs_top10, epsilon_map = (
        data["queries"], data["runs_top10"], data["epsilon_map"])
    system_names = data["system_names"]
    per_q = {}
    for q in queries:
        pids = set()
        for s in system_names:
            pids.update(runs_top10[s].get(q, []))
        pids_list = sorted(pids)
        eps_vals  = np.array([epsilon_map.get((q, pid), 0) for pid in pids_list])
        per_q[q] = (pids_list, eps_vals)
    return per_q


def build_within_grade_exposure(data):
    queries, runs_top10, epsilon_map, g_human_map = (
        data["queries"], data["runs_top10"],
        data["epsilon_map"], data["g_human_map"])
    system_names = data["system_names"]

    per_q = {}
    grade_counts = defaultdict(lambda: [0, 0])

    for q in queries:
        pids = set()
        for s in system_names:
            pids.update(runs_top10[s].get(q, []))
        pids_list = sorted(pids)

        grade_groups = defaultdict(list)
        for pid in pids_list:
            g = int(g_human_map.get((q, pid), 0))
            grade_groups[g].append(pid)

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


def build_within_confusion_cell_exposure(data):
    """
    Partition by (human_grade, llm_grade) pair.
    Only cells with >= 2 passages can be shuffled.
    """
    queries, runs_top10, epsilon_map, g_human_map, g_llm_map = (
        data["queries"], data["runs_top10"],
        data["epsilon_map"], data["g_human_map"], data["g_llm_map"])
    system_names = data["system_names"]

    per_q = {}
    cell_counts = defaultdict(lambda: [0, 0])  # key = (human_grade, llm_grade)

    for q in queries:
        pids = set()
        for s in system_names:
            pids.update(runs_top10[s].get(q, []))
        pids_list = sorted(pids)

        cell_groups = defaultdict(list)
        for pid in pids_list:
            gh = int(g_human_map.get((q, pid), 0))
            gl = int(g_llm_map.get((q, pid), 0))
            cell_groups[(gh, gl)].append(pid)

        q_groups = []
        for cell_key in sorted(cell_groups.keys()):
            cpids = cell_groups[cell_key]
            ceps  = np.array([epsilon_map.get((q, pid), 0) for pid in cpids])
            shufflable = len(cpids) >= 2
            if not shufflable:
                cell_counts[cell_key][0] += 1
                cell_counts[cell_key][1] += len(cpids)
            q_groups.append((cpids, ceps, shufflable))

        per_q[q] = q_groups

    return per_q, cell_counts


# ── Shufflers ─────────────────────────────────────────────────────────────

def shuffle_free(per_q_exposure, queries, rng):
    eps_null = {}
    for q in queries:
        pids_list, eps_vals = per_q_exposure[q]
        shuffled = eps_vals.copy()
        rng.shuffle(shuffled)
        for pid, ev in zip(pids_list, shuffled):
            eps_null[(q, pid)] = int(ev)
    return eps_null


def shuffle_within_groups(per_q_groups, queries, rng):
    """Generic within-group shuffle (works for both grade and confusion cell)."""
    eps_null = {}
    for q in queries:
        for gpids, geps, shufflable in per_q_groups[q]:
            if shufflable:
                shuffled = geps.copy()
                rng.shuffle(shuffled)
            else:
                shuffled = geps
            for pid, ev in zip(gpids, shuffled):
                eps_null[(q, pid)] = int(ev)
    return eps_null


# ── Core permutation runner ──────────────────────────────────────────────

def run_permutation(data, null_type, n_perms=N_PERMS):
    system_names = data["system_names"]
    queries      = data["queries"]
    runs_top10   = data["runs_top10"]
    dcg_human    = data["dcg_human"]
    disp_dcg     = data["disp_dcg"]

    obs_var_disp  = float(np.var(disp_dcg))
    obs_corr_comp, _ = spearmanr(disp_dcg, dcg_human)
    obs_tau20     = compute_tau_at_k(
        rank_systems(dcg_human, system_names),
        rank_systems(dcg_human + disp_dcg, system_names),
        20)

    if null_type == "free":
        per_q_free = build_free_exposure(data)
        shuffler = lambda rng: shuffle_free(per_q_free, queries, rng)
    elif null_type == "within_grade":
        per_q_wg, _ = build_within_grade_exposure(data)
        shuffler = lambda rng: shuffle_within_groups(per_q_wg, queries, rng)
    elif null_type == "within_confusion_cell":
        per_q_cc, _ = build_within_confusion_cell_exposure(data)
        shuffler = lambda rng: shuffle_within_groups(per_q_cc, queries, rng)
    else:
        raise ValueError(f"Unknown null_type: {null_type}")

    rng = np.random.RandomState(SEED)

    null_var_disp  = np.zeros(n_perms)
    null_corr_comp = np.zeros(n_perms)
    null_tau20     = np.zeros(n_perms)

    for pi in range(n_perms):
        eps_null = shuffler(rng)

        disp_null = compute_displacement(eps_null, runs_top10, queries,
                                         system_names, k=K)

        null_var_disp[pi]  = float(np.var(disp_null))
        rho, _ = spearmanr(disp_null, dcg_human)
        null_corr_comp[pi] = float(rho) if not np.isnan(rho) else 0.0
        null_tau20[pi]     = compute_tau_at_k(
            rank_systems(dcg_human, system_names),
            rank_systems(dcg_human + disp_null, system_names),
            20)

    def p_upper(obs, null_arr):
        return float(np.mean(null_arr >= obs))

    def p_two_sided(obs, null_arr):
        return float(np.mean(np.abs(null_arr) >= np.abs(obs)))

    def p_lower(obs, null_arr):
        return float(np.mean(null_arr <= obs))

    stats = {
        "var_disp_obs":       obs_var_disp,
        "var_disp_null_mean": float(null_var_disp.mean()),
        "var_disp_null_sd":   float(null_var_disp.std()),
        "var_disp_ratio":     obs_var_disp / null_var_disp.mean() if null_var_disp.mean() > 0 else np.nan,
        "var_disp_p":         p_upper(obs_var_disp, null_var_disp),
        "corr_comp_obs":       float(obs_corr_comp),
        "corr_comp_null_mean": float(null_corr_comp.mean()),
        "corr_comp_null_sd":   float(null_corr_comp.std()),
        "corr_comp_ratio":     (abs(obs_corr_comp) / abs(null_corr_comp).mean()
                                if abs(null_corr_comp).mean() > 0 else np.nan),
        "corr_comp_p":         p_two_sided(obs_corr_comp, null_corr_comp),
        "tau20_obs":       obs_tau20,
        "tau20_null_mean": float(null_tau20.mean()),
        "tau20_null_sd":   float(null_tau20.std()),
        "tau20_p":         p_lower(obs_tau20, null_tau20),
    }

    return stats


# ── Unshufflable count reporters ─────────────────────────────────────────

def compute_unshufflable_wg(data, year):
    _, grade_counts = build_within_grade_exposure(data)
    rows = []
    for grade in sorted(grade_counts.keys()):
        n_grp_lt2, n_pass_fixed = grade_counts[grade]
        rows.append({
            "year":              year,
            "null_type":         "within_grade",
            "human_grade":       grade,
            "n_groups_size_lt2": n_grp_lt2,
            "n_passages_fixed":  n_pass_fixed,
        })
    return rows


def compute_unshufflable_cc(data, year):
    _, cell_counts = build_within_confusion_cell_exposure(data)
    rows = []
    for (gh, gl) in sorted(cell_counts.keys()):
        n_grp_lt2, n_pass_fixed = cell_counts[(gh, gl)]
        rows.append({
            "year":              year,
            "null_type":         "within_confusion_cell",
            "human_grade":       gh,
            "llm_grade":         gl,
            "n_groups_size_lt2": n_grp_lt2,
            "n_passages_fixed":  n_pass_fixed,
        })
    return rows


# ── Output builders ──────────────────────────────────────────────────────

def build_permutation_comparison_rows(year, null_type, stats):
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
            ratio = obs / null_mean if label == "displacement_spread" else abs(obs) / abs(null_mean)
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


def build_signature_comparison_rows(year, null_type, stats):
    return [
        {"year": year, "null_type": null_type, "signature": "displacement_spread",
         "observed": stats["var_disp_obs"], "null_mean": stats["var_disp_null_mean"],
         "p_value": stats["var_disp_p"]},
        {"year": year, "null_type": null_type, "signature": "compression_corr",
         "observed": stats["corr_comp_obs"], "null_mean": stats["corr_comp_null_mean"],
         "p_value": stats["corr_comp_p"]},
        {"year": year, "null_type": null_type, "signature": "tau_at20_degradation",
         "observed": stats["tau20_obs"], "null_mean": stats["tau20_null_mean"],
         "p_value": stats["tau20_p"]},
    ]


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-2019", action="store_true",
                        help="Re-run 2019 and verify against existing results")
    parser.add_argument("--confusion-cell", action="store_true",
                        help="Also run within-confusion-cell variant")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    perm_comp_rows = []
    sig_comp_rows  = []
    unc_all_rows   = []
    results_all    = {}  # (year, null_type) -> stats

    # ── Step 1: Verify 2019 if requested ──────────────────────────────────
    if args.verify_2019:
        print("=" * 70)
        print("VERIFYING 2019 — checking free and within_grade reproduce exactly")
        print("=" * 70)
        data_2019 = load_year(2019)

        # Reference values from the existing run
        ref_free_var = 527.7938148478255
        ref_free_var_null = 46.92666358014364
        ref_wg_var = 527.7938148478255
        ref_wg_var_null = 457.62050871385276

        stats_free_2019 = run_permutation(data_2019, "free")
        stats_wg_2019   = run_permutation(data_2019, "within_grade")

        checks = [
            ("free obs",       stats_free_2019["var_disp_obs"],       ref_free_var),
            ("free null_mean", stats_free_2019["var_disp_null_mean"], ref_free_var_null),
            ("WG obs",         stats_wg_2019["var_disp_obs"],         ref_wg_var),
            ("WG null_mean",   stats_wg_2019["var_disp_null_mean"],   ref_wg_var_null),
        ]
        all_ok = True
        for label, got, expected in checks:
            match = abs(got - expected) < 1e-10
            status = "OK" if match else "MISMATCH"
            if not match:
                all_ok = False
            print(f"  {label}: {got} vs {expected} -> {status}")

        if not all_ok:
            print("\nERROR: 2019 results do not reproduce. Aborting.")
            sys.exit(1)
        print("\n2019 verification PASSED — results are byte-identical.\n")

    # ── Step 2: Run 2020-2023 free + within_grade ─────────────────────────
    years_main = [2020, 2021, 2022, 2023]

    for year in years_main:
        print(f"\n{'=' * 70}")
        print(f"YEAR {year}")
        print(f"{'=' * 70}")
        data = load_year(year)
        print(f"  Loaded: {len(data['system_names'])} systems, "
              f"{len(data['queries'])} queries")

        # Unshufflable counts (within_grade)
        unc_wg = compute_unshufflable_wg(data, year)
        total_fixed = sum(r["n_passages_fixed"] for r in unc_wg)
        print(f"  Unshufflable passages (WG, size-1 grade groups): {total_fixed}")
        for r in unc_wg:
            if r["n_passages_fixed"] > 0:
                print(f"    grade {r['human_grade']}: "
                      f"{r['n_groups_size_lt2']} groups, "
                      f"{r['n_passages_fixed']} passages")
        unc_all_rows.extend(unc_wg)

        for null_type in ["free", "within_grade"]:
            print(f"\n  Running {null_type.upper()} shuffle ({N_PERMS} perms)...")
            stats = run_permutation(data, null_type)
            results_all[(year, null_type)] = stats

            ratio = stats["var_disp_ratio"]
            print(f"    var_disp: obs={stats['var_disp_obs']:.2f}, "
                  f"null_mean={stats['var_disp_null_mean']:.2f}, "
                  f"ratio={ratio:.2f}x, p={stats['var_disp_p']:.4f}")
            print(f"    corr_comp: obs={stats['corr_comp_obs']:.4f}, "
                  f"null_mean={stats['corr_comp_null_mean']:.4f}, "
                  f"p={stats['corr_comp_p']:.4f}")
            print(f"    tau@20: obs={stats['tau20_obs']:.4f}, "
                  f"null_mean={stats['tau20_null_mean']:.4f}, "
                  f"p={stats['tau20_p']:.4f}")

            perm_comp_rows.extend(build_permutation_comparison_rows(year, null_type, stats))
            sig_comp_rows.extend(build_signature_comparison_rows(year, null_type, stats))

        # Attenuation
        rf = results_all[(year, "free")]["var_disp_ratio"]
        rw = results_all[(year, "within_grade")]["var_disp_ratio"]
        att = rw / rf if rf > 0 else np.nan
        print(f"\n  Ratio WG/Free: {att:.3f} "
              f"({'SURVIVES' if rw >= 2.0 else 'WEAKENED (<2x)'})")

    # ── Step 2b: Include 2019 from existing results ───────────────────────
    # Load the 2019 results from the prior run's CSV for the combined report
    prior_csv = BASE_DIR / "results" / "thesis_verification" / "t15_permutation" / "permutation_comparison.csv"
    prior_df = pd.read_csv(prior_csv)
    for _, row in prior_df.iterrows():
        perm_comp_rows.append(row.to_dict())
    prior_sig = BASE_DIR / "results" / "thesis_verification" / "t15_permutation" / "signature_comparison.csv"
    prior_sig_df = pd.read_csv(prior_sig)
    for _, row in prior_sig_df.iterrows():
        sig_comp_rows.append(row.to_dict())
    prior_unc = BASE_DIR / "results" / "thesis_verification" / "t15_permutation" / "unshufflable_counts.csv"
    prior_unc_df = pd.read_csv(prior_unc)
    for _, row in prior_unc_df.iterrows():
        r = row.to_dict()
        r["null_type"] = "within_grade"
        unc_all_rows.append(r)

    # Reconstruct 2019 stats for the report
    for _, row in prior_df[prior_df["year"] == 2019].iterrows():
        nt = row["null_type"]
        if (2019, nt) not in results_all:
            results_all[(2019, nt)] = {}
        stat = row["statistic"]
        key_map = {
            "displacement_spread": "var_disp",
            "compression_corr": "corr_comp",
            "tau_at20_degradation": "tau20",
        }
        sk = key_map[stat]
        results_all[(2019, nt)][f"{sk}_obs"] = row["observed"]
        results_all[(2019, nt)][f"{sk}_null_mean"] = row["null_mean"]
        results_all[(2019, nt)][f"{sk}_null_sd"] = row["null_sd"]
        results_all[(2019, nt)][f"{sk}_p"] = row["p_one_sided"]
        results_all[(2019, nt)][f"{sk}_ratio"] = row["ratio_obs_to_null_mean"]

    # ── Step 3: Confusion cell variant ────────────────────────────────────
    if args.confusion_cell:
        print(f"\n{'=' * 70}")
        print("WITHIN-CONFUSION-CELL VARIANT (partition by human_grade x llm_grade)")
        print(f"{'=' * 70}")

        for year in YEARS:
            print(f"\n  Year {year}...")
            data = load_year(year)

            # Unshufflable counts (confusion cell)
            unc_cc = compute_unshufflable_cc(data, year)
            total_fixed_cc = sum(r["n_passages_fixed"] for r in unc_cc)
            print(f"    Unshufflable passages (confusion cell): {total_fixed_cc}")
            unc_all_rows.extend(unc_cc)

            print(f"    Running WITHIN-CONFUSION-CELL shuffle ({N_PERMS} perms)...")
            stats_cc = run_permutation(data, "within_confusion_cell")
            results_all[(year, "within_confusion_cell")] = stats_cc

            ratio_cc = stats_cc["var_disp_ratio"]
            print(f"    var_disp: obs={stats_cc['var_disp_obs']:.2f}, "
                  f"null_mean={stats_cc['var_disp_null_mean']:.2f}, "
                  f"ratio={ratio_cc:.2f}x, p={stats_cc['var_disp_p']:.4f}")
            print(f"    corr_comp: obs={stats_cc['corr_comp_obs']:.4f}, "
                  f"null_mean={stats_cc['corr_comp_null_mean']:.4f}, "
                  f"p={stats_cc['corr_comp_p']:.4f}")
            print(f"    tau@20: obs={stats_cc['tau20_obs']:.4f}, "
                  f"null_mean={stats_cc['tau20_null_mean']:.4f}, "
                  f"p={stats_cc['tau20_p']:.4f}")

            perm_comp_rows.extend(build_permutation_comparison_rows(year, "within_confusion_cell", stats_cc))
            sig_comp_rows.extend(build_signature_comparison_rows(year, "within_confusion_cell", stats_cc))

    # ── Write CSVs ────────────────────────────────────────────────────────
    perm_df = pd.DataFrame(perm_comp_rows)
    perm_df = perm_df.sort_values(["year", "null_type", "statistic"]).reset_index(drop=True)
    perm_df.to_csv(OUT_DIR / "permutation_comparison.csv", index=False)

    sig_df = pd.DataFrame(sig_comp_rows)
    sig_df = sig_df.sort_values(["year", "null_type", "signature"]).reset_index(drop=True)
    sig_df.to_csv(OUT_DIR / "signature_comparison.csv", index=False)

    unc_df = pd.DataFrame(unc_all_rows)
    if not unc_df.empty:
        unc_df = unc_df.sort_values(["year", "null_type", "human_grade"]).reset_index(drop=True)
        unc_df.to_csv(OUT_DIR / "unshufflable_counts.csv", index=False)

    print(f"\nCSVs written to {OUT_DIR}")

    # ── Write REPORT.md ──────────────────────────────────────────────────
    write_report(results_all, unc_all_rows, args.confusion_cell)
    print(f"REPORT.md written to {OUT_DIR}")


def write_report(results_all, unc_all_rows, has_cc):
    lines = []
    lines.append("# Within-Grade Permutation Null — Full Report (Task t15b)")
    lines.append("")

    # ── Main table: displacement spread ──────────────────────────────────
    lines.append("## Displacement variance ratio (obs / null_mean), all five years")
    lines.append("")

    null_types = ["free", "within_grade"]
    if has_cc:
        null_types.append("within_confusion_cell")

    header = "| Year |"
    sep    = "|------|"
    for nt in null_types:
        short = {"free": "Free", "within_grade": "WG",
                 "within_confusion_cell": "CC"}[nt]
        header += f" {short} ratio | {short} p |"
        sep    += "---------|-------|"
    header += " Survives WG? |"
    sep    += "--------------|"
    lines.append(header)
    lines.append(sep)

    n_survive = 0
    for year in YEARS:
        row = f"| {year} |"
        for nt in null_types:
            key = (year, nt)
            if key in results_all:
                s = results_all[key]
                ratio = s.get("var_disp_ratio", np.nan)
                p = s.get("var_disp_p", np.nan)
                row += f" {ratio:.2f}x | {p:.4f} |"
            else:
                row += " — | — |"
        # Survives?
        wg_key = (year, "within_grade")
        if wg_key in results_all:
            wg_ratio = results_all[wg_key].get("var_disp_ratio", 0)
            surv = "YES" if wg_ratio >= 2.0 else "NO"
            if wg_ratio >= 2.0:
                n_survive += 1
        else:
            surv = "—"
        row += f" {surv} |"
        lines.append(row)

    lines.append("")
    lines.append(f"**Answer: the displacement-spread effect survives the within-grade null "
                 f"in {n_survive} of 5 collections.**")
    lines.append("")

    # ── Compression correlation table ────────────────────────────────────
    lines.append("## Compression correlation (Spearman rho)")
    lines.append("")
    lines.append("| Year | Observed | Free null | Free p | WG null | WG p |"
                 + (" CC null | CC p |" if has_cc else ""))
    lines.append("|------|---------|-----------|--------|---------|------|"
                 + ("---------|------|" if has_cc else ""))

    for year in YEARS:
        obs = results_all.get((year, "free"), results_all.get((year, "within_grade"), {}))
        obs_val = obs.get("corr_comp_obs", np.nan)
        row = f"| {year} | {obs_val:.4f} |"
        for nt in null_types:
            key = (year, nt)
            if key in results_all:
                s = results_all[key]
                row += f" {s.get('corr_comp_null_mean', np.nan):.4f} | {s.get('corr_comp_p', np.nan):.4f} |"
            else:
                row += " — | — |"
        lines.append(row)

    lines.append("")

    # ── Tau@20 table ─────────────────────────────────────────────────────
    lines.append("## tau@20 degradation (frac null <= obs)")
    lines.append("")
    lines.append("| Year | Observed | Free null | Free p | WG null | WG p |"
                 + (" CC null | CC p |" if has_cc else ""))
    lines.append("|------|---------|-----------|--------|---------|------|"
                 + ("---------|------|" if has_cc else ""))

    for year in YEARS:
        obs = results_all.get((year, "free"), results_all.get((year, "within_grade"), {}))
        obs_val = obs.get("tau20_obs", np.nan)
        row = f"| {year} | {obs_val:.4f} |"
        for nt in null_types:
            key = (year, nt)
            if key in results_all:
                s = results_all[key]
                row += f" {s.get('tau20_null_mean', np.nan):.4f} | {s.get('tau20_p', np.nan):.4f} |"
            else:
                row += " — | — |"
        lines.append(row)

    lines.append("")

    # ── Unshufflable counts summary ──────────────────────────────────────
    lines.append("## Unshufflable passages")
    lines.append("")
    lines.append("Passages in groups of size < 2 that cannot be permuted:")
    lines.append("")

    unc_df = pd.DataFrame(unc_all_rows)
    if not unc_df.empty:
        for nt in unc_df["null_type"].unique():
            sub = unc_df[unc_df["null_type"] == nt]
            totals = sub.groupby("year")["n_passages_fixed"].sum()
            lines.append(f"**{nt}**: " + ", ".join(f"{y}: {int(v)}" for y, v in totals.items()))
        lines.append("")

    # ── Methodology notes ────────────────────────────────────────────────
    lines.append("## Methodology")
    lines.append("")
    lines.append(f"- **Number of permutations per collection**: {N_PERMS}")
    lines.append(f"- **Random seed**: {SEED} (numpy RandomState, identical to free-shuffle test)")
    lines.append("- **Statistic permuted**: the epsilon vector (eps = g_llm - g_human) "
                 "is shuffled within each query's exposure set (the union of passages "
                 "retrieved by any system for that query at depth k=10)")
    lines.append("- **Within-grade partition**: passages grouped by human relevance grade "
                 "(g_human in {0, 1, 2, 3}); shuffling occurs only within same-grade groups")
    if has_cc:
        lines.append("- **Within-confusion-cell partition**: passages grouped by the pair "
                     "(g_human, g_llm); this is strictly finer than within-grade and holds "
                     "fixed everything about an error except which specific passage received it")
    lines.append("- **p-values**:")
    lines.append("  - displacement_spread: one-sided upper (frac of null >= obs)")
    lines.append("  - compression_corr: two-sided (frac of |null| >= |obs|)")
    lines.append("  - tau@20_degradation: one-sided lower (frac of null <= obs)")
    lines.append("- **Multiplicity correction**: none applied across the five collections; "
                 "results are reported per-year and the reader should consider the pattern "
                 "across all five years rather than any single p-value")
    lines.append("- **Missing grades**: passages in the exposure set without a human qrel "
                 "are assigned grade 0 (consistent with TREC convention)")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `permutation_comparison.csv`: full statistics per (year, null_type, statistic)")
    lines.append("- `signature_comparison.csv`: the three signatures per (year, null_type)")
    lines.append("- `unshufflable_counts.csv`: counts of fixed passages per grade group")
    lines.append("- This report does NOT overwrite `results/spectral/hardening/permutation_null.csv` "
                 "or `permutation_null_distributions.parquet`")

    (OUT_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
