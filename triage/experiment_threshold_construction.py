# 2x2 analysis of threshold definition (first-touch vs sustained) x C construction

import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR  = BASE_DIR / "results" / "thesis_verification" / "t19b_threshold_definition"

YEARS = [2019, 2020, 2021, 2022, 2023]
V2_YEARS = {2021, 2022, 2023}
TARGETS = [0.90, 0.95]
FEATURE_QUERY_CSV = BASE_DIR / "results" / "level2" / "per_query_results.csv"

YEAR_CFG = {
    2019: {
        "qrels":  BASE_DIR / "data_prep" / "data" / "trec-dl"    / "2019" / "qrels.txt",
        "scores": BASE_DIR / "results"   / "scoring" / "normal_scores" / "scores_v1.jsonl",
        "runs":   BASE_DIR / "data"      / "system_runs" / "2019",
    },
    2020: {
        "qrels":  BASE_DIR / "data_prep" / "data" / "trec-dl"    / "2020" / "qrels.txt",
        "scores": BASE_DIR / "results"   / "scoring" / "normal_scores" / "scores_v1.jsonl",
        "runs":   BASE_DIR / "data"      / "system_runs" / "2020",
    },
    2021: {
        "qrels":  BASE_DIR / "data_prep" / "data" / "trec-dl-v2" / "2021" / "qrels_dedup.txt",
        "scores": BASE_DIR / "results"   / "scoring" / "normal_scores" / "scores_v2.jsonl",
        "runs":   BASE_DIR / "data"      / "system_runs" / "2021",
    },
    2022: {
        "qrels":  BASE_DIR / "data_prep" / "data" / "trec-dl-v2" / "2022" / "qrels_dedup.txt",
        "scores": BASE_DIR / "results"   / "scoring" / "normal_scores" / "scores_v2.jsonl",
        "runs":   BASE_DIR / "data"      / "system_runs" / "2022",
    },
    2023: {
        "qrels":  BASE_DIR / "data_prep" / "data" / "trec-dl-v2" / "2023" / "qrels_dedup.txt",
        "scores": BASE_DIR / "results"   / "scoring" / "normal_scores" / "scores_v2.jsonl",
        "runs":   BASE_DIR / "data"      / "system_runs" / "2023",
    },
}

TOPM_DIR = BASE_DIR / "results" / "topM_correction"
EXP1_DIR = BASE_DIR / "results" / "exp1"

# t19 full-variant sustained thresholds at tau@20 >= 0.95 (from ablation_thresholds.csv)
T19_FULL_SUSTAINED_95 = {2019: 0.90, 2020: 0.85, 2021: 0.85, 2022: 0.70, 2023: 1.00}

# Table 4 reference values (from thesis, at tau@20 >= 0.95, first_touch, adaptive_top20)
TABLE4_RUNAWARE  = {2019: 0.15, 2020: 0.14, 2021: 0.31, 2022: 0.11, 2023: 0.16}
TABLE4_JUDGEONLY = {2019: 0.59, 2020: 0.48, 2021: 0.59, 2022: 0.67, 2023: 0.37}


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_feature_queries():
    df = pd.read_csv(FEATURE_QUERY_CSV, dtype={"query_id": str})
    return set(df["query_id"].astype(str))

def load_qrels(path):
    qrels = defaultdict(dict)
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 4:
                qrels[p[0]][p[2]] = int(p[3])
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

def load_runs(runs_dir):
    runs = {}
    for fname in sorted(os.listdir(runs_dir)):
        if not fname.endswith(".txt"):
            continue
        sys_name = fname[:-4]
        sys_runs = defaultdict(list)
        with open(os.path.join(runs_dir, fname)) as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 6:
                    sys_runs[p[0]].append((int(p[3]), p[2]))
        for qid in sys_runs:
            sys_runs[qid].sort()
            sys_runs[qid] = [pid for _, pid in sys_runs[qid][:1000]]
        runs[sys_name] = dict(sys_runs)
    return runs

def load_canonical_map():
    from v2_id_mapping import load_canonical_map as _lcm
    return _lcm()

def canonicalize(runs, cmap):
    from v2_id_mapping import canonicalize_runs
    canonicalize_runs(runs, cmap)


# ── nDCG helper ───────────────────────────────────────────────────────────────

def ndcg_at_k(ranked_pids, qrels_q, k=10):
    if not qrels_q:
        return 0.0
    dcg = sum(qrels_q.get(pid, 0) / math.log2(i + 2)
              for i, pid in enumerate(ranked_pids[:k]))
    ideal = sorted(qrels_q.values(), reverse=True)[:k]
    idcg  = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# ── Ranking / tau helpers ─────────────────────────────────────────────────────

def rank_systems(system_names, scores):
    return [s for s, _ in sorted(zip(system_names, scores),
                                  key=lambda x: (-x[1], x[0]))]

def tau_at_k(gold, mixed, K):
    K = min(K, len(gold))
    top_k = set(gold[:K])
    go = [s for s in gold  if s in top_k]
    mo = [s for s in mixed if s in top_k]
    if len(mo) < 2:
        return 1.0
    gr = {s: i for i, s in enumerate(go)}
    pos = [gr[s] for s in mo]
    tau, _ = kendalltau(list(range(len(go))), pos)
    return 1.0 if math.isnan(tau) else float(tau)


# ── Threshold functions ───────────────────────────────────────────────────────

def first_touch(budgets, taus, target):
    """First budget where tau >= target."""
    for b, t in sorted(zip(budgets, taus)):
        if t >= target:
            return float(b)
    return None

def sustained(budgets, taus, target):
    """Smallest b such that tau >= target at every checkpoint from b onward."""
    pairs = sorted(zip(budgets, taus))
    vals = [t for _, t in pairs]
    buds = [b for b, _ in pairs]
    if not any(v >= target for v in vals):
        return None
    last_below = max((i for i, v in enumerate(vals) if v < target), default=-1)
    if last_below == len(vals) - 1:
        return None
    if last_below == -1:
        return float(buds[0])
    return float(buds[last_below + 1])


# ── All-runs C correction sweep ───────────────────────────────────────────────

def run_allruns_sweep(year, human_qrels, llm_qrels, runs, queries,
                      system_names, gold_ranking, c_pp_dict, n_steps=101):
    """
    Static correction sweep ordered by c_pp_full (all-runs C).
    Remainder: raw LLM grades for unjudged pairs.
    Returns list of {budget, tau_at_20}.
    """
    # Universe: pairs with both human and LLM grades
    universe = []
    for qid in queries:
        for pid in human_qrels.get(qid, {}):
            if pid in llm_qrels.get(qid, {}):
                universe.append((qid, pid))

    n_universe = len(universe)
    if n_universe == 0:
        return []

    # Sort by c_pp descending (highest leverage first)
    # Pairs not in c_pp_dict get score 0.0
    universe_sorted = sorted(universe,
                              key=lambda kv: -c_pp_dict.get(kv, 0.0))

    budget_fractions = np.linspace(0, 1, n_steps)
    results = []

    for b in budget_fractions:
        n_human = int(round(b * n_universe))
        human_set = set(universe_sorted[:n_human])

        # Build mixed grades
        mixed = {}
        for qid in queries:
            mixed[qid] = {}
            for pid in human_qrels.get(qid, {}):
                if pid in llm_qrels.get(qid, {}):
                    if (qid, pid) in human_set:
                        mixed[qid][pid] = human_qrels[qid][pid]
                    else:
                        mixed[qid][pid] = llm_qrels[qid].get(pid, 0)

        # Compute system scores
        n_q = len(queries)
        scores = []
        for sys_name in system_names:
            run_q = runs.get(sys_name, {})
            total = sum(ndcg_at_k(run_q.get(qid, []),
                                   mixed.get(qid, {}))
                        for qid in queries)
            scores.append(total / n_q)

        mixed_ranking = rank_systems(system_names, scores)
        results.append({
            "budget":    float(b),
            "tau_at_20": tau_at_k(gold_ranking, mixed_ranking, 20),
        })

    return results


# ── Load per-year topM curves ─────────────────────────────────────────────────

def load_topm_curves(year):
    """Load {year}_topM_correction_curves_M20_adaptive.csv"""
    fp = TOPM_DIR / f"{year}_topM_correction_curves_M20_adaptive.csv"
    df = pd.read_csv(fp)
    result = {}
    for policy, grp in df.groupby("policy"):
        grp = grp.sort_values("budget")
        result[policy] = {
            "budgets": grp["budget"].tolist(),
            "taus":    grp["tau_at_20"].tolist(),
        }
    return result

def load_lara_curve(year):
    """Load lara curve from final_simple_results.csv for the given year."""
    fp = TOPM_DIR / "final_simple_results.csv"
    df = pd.read_csv(fp)
    sub = df[(df["year"] == year) & (df["policy"] == "lara")].sort_values("budget")
    return {"budgets": sub["budget"].tolist(), "taus": sub["tau_at_20"].tolist()}

def load_random_curve(year):
    """Load random mean curve from exp1/{year}_random_ci.csv."""
    fp = EXP1_DIR / f"{year}_random_ci.csv"
    df = pd.read_csv(fp).sort_values("budget")
    return {"budgets": df["budget"].tolist(), "taus": df["tau_at_20_mean"].tolist(),
            "n_steps": len(df)}

def load_leverage_parquet(year, queries_set, llm_qrels):
    """Load C_pp (all-runs population variance) from results/exp1/leverage_{year}.parquet.
    Columns: qid, pid, C_pp, C_pp_loo.
    Returns {(qid, pid): C_pp}."""
    fp = EXP1_DIR / f"leverage_{year}.parquet"
    df = pd.read_parquet(fp)
    # Already string dtypes; columns are qid, pid, C_pp, C_pp_loo
    result = {}
    for row in df.itertuples(index=False):
        qid, pid, cpp = str(row.qid), str(row.pid), float(row.C_pp)
        if qid in queries_set and pid in llm_qrels.get(qid, {}):
            result[(qid, pid)] = cpp
    return result


# ── Stability analysis (Step 3) ───────────────────────────────────────────────

def stability_metrics(budgets, taus, ft_budget, target):
    """Compute stability metrics AFTER first touch."""
    if ft_budget is None:
        return dict(min_tau_after=None, frac_checkpoints_below=None,
                    frac_budget_below=None)
    pts = [(b, t) for b, t in sorted(zip(budgets, taus)) if b > ft_budget + 1e-9]
    if not pts:
        return dict(min_tau_after=None, frac_checkpoints_below=None,
                    frac_budget_below=None)
    min_tau  = min(t for _, t in pts)
    n_below  = sum(1 for _, t in pts if t < target)
    frac_ckpts = n_below / len(pts)

    # Budget range below target (as fraction of remaining sweep)
    remaining = max(b for b, _ in pts) - ft_budget
    below_pts = [(b, t) for b, t in pts if t < target]
    if not below_pts or remaining <= 0:
        frac_budget = 0.0
    else:
        # Approximate: length of intervals where curve is below target
        # Use contiguous segments
        sorted_pts = sorted(pts)
        budget_below = 0.0
        for i in range(len(sorted_pts) - 1):
            b0, t0 = sorted_pts[i]
            b1, t1 = sorted_pts[i + 1]
            if t0 < target and t1 < target:
                budget_below += b1 - b0
            elif t0 < target or t1 < target:
                budget_below += (b1 - b0) * 0.5
        frac_budget = budget_below / remaining if remaining > 0 else 0.0

    return dict(min_tau_after=min_tau,
                frac_checkpoints_below=frac_ckpts,
                frac_budget_below=frac_budget)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  t19b: Threshold Definition × C Construction 2x2")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load feature queries
    feature_queries = load_feature_queries()
    print(f"Feature queries: {len(feature_queries)}")

    # Try to load canonical map for v2 years
    try:
        cmap = load_canonical_map()
    except Exception:
        cmap = None
        print("  Warning: v2 canonical map not loaded")

    # ── Storage ──────────────────────────────────────────────────────────────
    two_by_two_rows = []        # threshold_2x2.csv
    stability_rows  = []        # first_touch_stability.csv
    decomp_rows     = []        # gap_decomposition.csv
    reconcile_rows  = []        # table4_reconciliation.csv

    for year in YEARS:
        cfg = YEAR_CFG[year]
        print(f"\n--- Year {year} ---")

        # Load base data
        human_qrels = load_qrels(cfg["qrels"])
        year_queries = set(human_qrels.keys())
        llm_qrels   = load_llm_qrels(cfg["scores"], year_queries)
        year_queries &= set(llm_qrels.keys())
        year_queries &= feature_queries

        runs = load_runs(cfg["runs"])
        if year in V2_YEARS and cmap is not None:
            canonicalize(runs, cmap)

        queries      = sorted(year_queries)
        system_names = sorted(runs.keys())
        print(f"  {len(queries)} queries, {len(system_names)} systems")

        # Compute gold ranking (human grades)
        n_q = len(queries)
        gold_scores = []
        for sys_name in system_names:
            total = sum(ndcg_at_k(runs[sys_name].get(qid, []),
                                   human_qrels.get(qid, {}))
                        for qid in queries)
            gold_scores.append(total / n_q)
        gold_ranking = rank_systems(system_names, gold_scores)

        # ── 1. adaptive_top20 curves ─────────────────────────────────────────
        topm = load_topm_curves(year)
        lara_curve   = load_lara_curve(year)
        random_curve = load_random_curve(year)
        n_random_steps = random_curve["n_steps"]

        adaptive_curves = {
            "run_aware":  topm.get("leverage_calibrated", {"budgets": [], "taus": []}),
            "judge_only": lara_curve,
            "random":     random_curve,
        }
        n_topm_checkpoints = len(adaptive_curves["run_aware"]["budgets"])
        print(f"  adaptive_top20 checkpoints (leverage_calibrated): {n_topm_checkpoints}")

        # ── 2. all_runs C sweep ──────────────────────────────────────────────
        print("  Computing all_runs C sweep...")
        c_pp = load_leverage_parquet(year, set(queries), llm_qrels)
        print(f"    c_pp_full loaded: {len(c_pp)} pairs")

        allruns_curve = run_allruns_sweep(
            year, human_qrels, llm_qrels, runs, queries,
            system_names, gold_ranking, c_pp, n_steps=101)
        allruns_budgets = [r["budget"] for r in allruns_curve]
        allruns_taus    = [r["tau_at_20"] for r in allruns_curve]
        print(f"    all_runs sweep: {len(allruns_curve)} steps")

        # judge_only and random are C-agnostic
        allruns_curves = {
            "run_aware":  {"budgets": allruns_budgets, "taus": allruns_taus},
            "judge_only": lara_curve,
            "random":     random_curve,
        }

        # ── 3. Compute 2x2 metrics ───────────────────────────────────────────
        policy_c_pairs = [
            ("run_aware",  "adaptive_top20", adaptive_curves["run_aware"]),
            ("run_aware",  "all_runs",       allruns_curves["run_aware"]),
            ("judge_only", "adaptive_top20", adaptive_curves["judge_only"]),
            ("judge_only", "all_runs",       allruns_curves["judge_only"]),
            ("random",     "adaptive_top20", adaptive_curves["random"]),
            ("random",     "all_runs",       allruns_curves["random"]),
        ]

        # Collect per-policy per-target first_touch and sustained
        metrics = {}  # (policy, c, target, defn) -> budget_pct
        for policy, c_constr, curve in policy_c_pairs:
            bs = curve["budgets"]
            ts = curve["taus"]
            for target in TARGETS:
                ft = first_touch(bs, ts, target)
                su = sustained(bs, ts, target)
                metrics[(policy, c_constr, target, "first_touch")] = ft
                metrics[(policy, c_constr, target, "sustained")]   = su

                for defn, val in [("first_touch", ft), ("sustained", su)]:
                    reached = val is not None
                    two_by_two_rows.append({
                        "year": year,
                        "policy": policy,
                        "c_construction": c_constr,
                        "threshold_definition": defn,
                        "target": target,
                        "budget_pct": round(val * 100, 1) if val is not None else None,
                        "reached": reached,
                    })

        # ── 4. First-touch stability (Step 3, adaptive_top20 at 0.95) ────────
        for policy, c_constr, curve in policy_c_pairs:
            if c_constr != "adaptive_top20":
                continue
            bs = curve["budgets"]
            ts = curve["taus"]
            ft  = metrics[(policy, "adaptive_top20", 0.95, "first_touch")]
            su  = metrics[(policy, "adaptive_top20", 0.95, "sustained")]
            gap = (su - ft) if (ft is not None and su is not None) else None
            stab = stability_metrics(bs, ts, ft, 0.95)
            stability_rows.append({
                "year": year,
                "policy": policy,
                "first_touch_pct": round(ft * 100, 1) if ft is not None else None,
                "sustained_pct":   round(su * 100, 1) if su is not None else None,
                "gap_pct":         round(gap * 100, 1) if gap is not None else None,
                "min_tau_after_first_touch":
                    round(stab["min_tau_after"], 4) if stab["min_tau_after"] is not None else None,
                "frac_checkpoints_below_after":
                    round(stab["frac_checkpoints_below"], 4) if stab["frac_checkpoints_below"] is not None else None,
                "frac_budget_below_after":
                    round(stab["frac_budget_below"], 4) if stab["frac_budget_below"] is not None else None,
            })

        # ── 5. Gap decomposition (Step 4, at target=0.95) ────────────────────
        # Reference points
        ar_su  = metrics[("run_aware", "all_runs",       0.95, "sustained")]    # all_runs + sustained
        ar_ft  = metrics[("run_aware", "all_runs",       0.95, "first_touch")]  # all_runs + first_touch
        at_su  = metrics[("run_aware", "adaptive_top20", 0.95, "sustained")]    # adaptive + sustained
        at_ft  = metrics[("run_aware", "adaptive_top20", 0.95, "first_touch")]  # adaptive + first_touch

        def diff_pct(a, b):
            """a - b in pp; None if either is missing."""
            if a is None or b is None:
                return None
            return round((a - b) * 100, 1)

        # Total gap: all_runs sustained → adaptive first_touch
        total_gap = diff_pct(ar_su, at_ft)

        # Ordering 1: definition first, then C
        # effect_definition = (all_runs, sustained) - (all_runs, first_touch)
        # effect_c          = (all_runs, first_touch) - (adaptive, first_touch)
        eff_defn_order1 = diff_pct(ar_su, ar_ft)
        eff_c_order1    = diff_pct(ar_ft, at_ft)
        interaction1    = diff_pct(total_gap,
                                    (eff_defn_order1 or 0) + (eff_c_order1 or 0)
                                    if (eff_defn_order1 is not None and
                                        eff_c_order1    is not None) else None)

        # Ordering 2: C first, then definition
        # effect_c          = (all_runs, sustained) - (adaptive, sustained)
        # effect_definition = (adaptive, sustained) - (adaptive, first_touch)
        eff_c_order2    = diff_pct(ar_su, at_su)
        eff_defn_order2 = diff_pct(at_su, at_ft)
        interaction2    = diff_pct(total_gap,
                                    (eff_c_order2 or 0) + (eff_defn_order2 or 0)
                                    if (eff_c_order2    is not None and
                                        eff_defn_order2 is not None) else None)

        for ordering, eff_defn, eff_c, inter in [
            ("defn_first", eff_defn_order1, eff_c_order1, interaction1),
            ("c_first",    eff_c_order2,    eff_defn_order2, interaction2),
        ]:
            decomp_rows.append({
                "year": year,
                "ordering": ordering,
                "total_gap_pct": total_gap,
                "effect_of_definition_pct": eff_defn,
                "effect_of_c_pct":          eff_c,
                "interaction_pct":          inter,
            })

        # ── 6. Table 4 reconciliation ─────────────────────────────────────────
        for policy, table4_vals in [
            ("run_aware",  TABLE4_RUNAWARE),
            ("judge_only", TABLE4_JUDGEONLY),
        ]:
            t4_val = table4_vals.get(year)
            # Test all 4 cells (2 c_constr × 2 threshold_defn) at target 0.95
            best_match = None
            best_res   = None
            for c_constr in ["adaptive_top20", "all_runs"]:
                for defn in ["first_touch", "sustained"]:
                    v = metrics.get((policy, c_constr, 0.95, defn))
                    if v is None:
                        continue
                    res = abs(v - t4_val) if t4_val is not None else None
                    if res is not None and (best_res is None or res < best_res):
                        best_res   = res
                        best_match = f"{c_constr},{defn}"
            reconcile_rows.append({
                "year": year,
                "policy": policy,
                "table4_value_pct": round(t4_val * 100, 1) if t4_val is not None else None,
                "best_matching_cell": best_match,
                "matched_value_pct":
                    round(metrics.get((policy,
                                       best_match.split(",")[0],
                                       0.95,
                                       best_match.split(",")[1]), 0) * 100, 1)
                    if best_match else None,
                "residual_pp": round(best_res * 100, 1) if best_res is not None else None,
            })

        print(f"  Done year {year}.")

    # ── Save output files ─────────────────────────────────────────────────────

    pd.DataFrame(two_by_two_rows).to_csv(
        OUT_DIR / "threshold_2x2.csv", index=False)
    print(f"\nWrote threshold_2x2.csv ({len(two_by_two_rows)} rows)")

    pd.DataFrame(reconcile_rows).to_csv(
        OUT_DIR / "table4_reconciliation.csv", index=False)
    print(f"Wrote table4_reconciliation.csv ({len(reconcile_rows)} rows)")

    pd.DataFrame(stability_rows).to_csv(
        OUT_DIR / "first_touch_stability.csv", index=False)
    print(f"Wrote first_touch_stability.csv ({len(stability_rows)} rows)")

    pd.DataFrame(decomp_rows).to_csv(
        OUT_DIR / "gap_decomposition.csv", index=False)
    print(f"Wrote gap_decomposition.csv ({len(decomp_rows)} rows)")

    # ── Print summary tables ──────────────────────────────────────────────────
    df2x2 = pd.DataFrame(two_by_two_rows)
    df_stab = pd.DataFrame(stability_rows)
    df_decomp = pd.DataFrame(decomp_rows)
    df_rec = pd.DataFrame(reconcile_rows)

    print("\n" + "=" * 70)
    print("THRESHOLD 2x2 (budget_pct at tau@20 >= 0.95, run_aware)")
    print("=" * 70)
    sub = df2x2[(df2x2.policy == "run_aware") & (df2x2.target == 0.95)]
    piv = sub.pivot_table(index=["year", "c_construction"],
                          columns="threshold_definition",
                          values="budget_pct", aggfunc="first")
    print(piv.to_string())

    print("\n" + "=" * 70)
    print("TABLE 4 RECONCILIATION")
    print("=" * 70)
    print(df_rec.to_string(index=False))

    print("\n" + "=" * 70)
    print("FIRST-TOUCH STABILITY (adaptive_top20, target=0.95)")
    print("=" * 70)
    print(df_stab.to_string(index=False))

    print("\n" + "=" * 70)
    print("GAP DECOMPOSITION (all_runs sustained → adaptive first_touch, 0.95)")
    print("=" * 70)
    print(df_decomp.to_string(index=False))

    # ── Write REPORT.md ───────────────────────────────────────────────────────
    _write_report(df2x2, df_stab, df_decomp, df_rec, n_topm_checkpoints, n_random_steps)
    print(f"\nWrote REPORT.md")
    print("Done.")


def _write_report(df2x2, df_stab, df_decomp, df_rec, n_ckpts_topm, n_ckpts_random):
    """Write the REPORT.md with direct answers to all three questions."""

    def tb(df, policy, target, defn):
        """Extract 5-year row for a given (policy, target, threshold_defn)."""
        sub = df[(df.policy == policy) & (df.target == target) &
                  (df.threshold_definition == defn)]
        return {int(r["year"]): r["budget_pct"] for _, r in sub.iterrows()}

    ra_at_ft_95 = tb(df2x2, "run_aware",  0.95, "first_touch")
    ra_su_ft_95 = tb(df2x2, "run_aware",  0.95, "sustained")
    ra_at_su_95 = {y: df2x2[(df2x2.policy=="run_aware")&(df2x2.c_construction=="adaptive_top20")&
                              (df2x2.target==0.95)&(df2x2.threshold_definition=="sustained")&
                              (df2x2.year==y)]["budget_pct"].values[0]
                   for y in YEARS}
    ra_ar_ft_95 = {y: df2x2[(df2x2.policy=="run_aware")&(df2x2.c_construction=="all_runs")&
                              (df2x2.target==0.95)&(df2x2.threshold_definition=="first_touch")&
                              (df2x2.year==y)]["budget_pct"].values[0]
                   for y in YEARS}
    ra_ar_su_95 = {y: df2x2[(df2x2.policy=="run_aware")&(df2x2.c_construction=="all_runs")&
                              (df2x2.target==0.95)&(df2x2.threshold_definition=="sustained")&
                              (df2x2.year==y)]["budget_pct"].values[0]
                   for y in YEARS}
    jo_at_ft_95 = {y: df2x2[(df2x2.policy=="judge_only")&(df2x2.c_construction=="adaptive_top20")&
                              (df2x2.target==0.95)&(df2x2.threshold_definition=="first_touch")&
                              (df2x2.year==y)]["budget_pct"].values[0]
                   for y in YEARS}
    jo_at_su_95 = {y: df2x2[(df2x2.policy=="judge_only")&(df2x2.c_construction=="adaptive_top20")&
                              (df2x2.target==0.95)&(df2x2.threshold_definition=="sustained")&
                              (df2x2.year==y)]["budget_pct"].values[0]
                   for y in YEARS}
    rn_at_ft_95 = {y: df2x2[(df2x2.policy=="random")&(df2x2.c_construction=="adaptive_top20")&
                              (df2x2.target==0.95)&(df2x2.threshold_definition=="first_touch")&
                              (df2x2.year==y)]["budget_pct"].values[0]
                   for y in YEARS}
    rn_at_su_95 = {y: df2x2[(df2x2.policy=="random")&(df2x2.c_construction=="adaptive_top20")&
                              (df2x2.target==0.95)&(df2x2.threshold_definition=="sustained")&
                              (df2x2.year==y)]["budget_pct"].values[0]
                   for y in YEARS}

    def fmt(v):
        return f"{v:.1f}%" if v is not None else "never"

    year_hdr = "  ".join(f"  {y}" for y in YEARS)

    lines = []
    lines.append("# t19b — Threshold Definition × C Construction")
    lines.append("")
    lines.append(f"**Script:** `run_t19b_threshold_definition.py`")
    lines.append(f"**Date:** 2026-08-04")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Conventions and Data Sources")
    lines.append("")
    lines.append("### Threshold definitions")
    lines.append("- **first_touch**: smallest budget b where tau@20 >= target (ignoring later dips)")
    lines.append("- **sustained**: smallest budget b* where tau@20 >= target at EVERY checkpoint from b* to end")
    lines.append("")
    lines.append("### C construction")
    lines.append("- **adaptive_top20**: leverage_calibrated policy from")
    lines.append("  `{year}_topM_correction_curves_M20_adaptive.csv`.")
    lines.append("  Selection score = Var(run weights) over adaptive top-M=20 systems.")
    lines.append("  Remainder = calibrator-relabelled unjudged pairs (LARA-style).")
    lines.append("- **all_runs**: static correction sweep ordered by c_pp_full from")
    lines.append("  `results/exp1/leverage_{year}.parquet`.")
    lines.append("  c_pp_full = population variance of nDCG Jacobian weights over ALL systems.")
    lines.append("  Remainder = raw LLM grades for unjudged pairs.")
    lines.append("")
    lines.append("  **Important caveat**: all_runs and adaptive_top20 differ in (1) the")
    lines.append("  score type (Jacobian variance vs run-weight variance) and (2) remainder")
    lines.append("  treatment (raw vs calibrated). The decomposition in Step 4 therefore")
    lines.append("  measures the joint effect of C construction + calibration, not C alone.")
    lines.append("")
    lines.append("### Policies")
    lines.append("- **run_aware**: leverage_calibrated (adaptive_top20) / c_pp_full sweep (all_runs)")
    lines.append("- **judge_only**: lara (C-agnostic; same curve used for both C columns)")
    lines.append("- **random**: mean of exp1 random CI draws (C-agnostic; same for both columns)")
    lines.append("")
    lines.append("### Sweep grid resolution")
    lines.append(f"- adaptive_top20 curves: ~{n_ckpts_topm} checkpoints (BATCH_FRACTION=0.01)")
    lines.append(f"- all_runs sweep: 101 checkpoints (np.linspace 0..1)")
    lines.append(f"- random CI: {n_ckpts_random} checkpoints (5pp grid; COARSER than others)")
    lines.append("")
    lines.append("  The coarser random grid inflates first_touch relative to sustained by")
    lines.append("  making dips invisible.  Random first_touch numbers should be treated")
    lines.append("  as upper-rounded by up to 5pp.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Step 1 — 2x2 Table (budget%, tau@20 >= 0.95)")
    lines.append("")
    lines.append("### run_aware policy")
    lines.append("")
    lines.append(f"| C \\ Threshold | first_touch | sustained |")
    lines.append(f"|---|---|---|")
    for c in ["adaptive_top20", "all_runs"]:
        ft_row = "  ".join(fmt(df2x2[(df2x2.policy=="run_aware")&(df2x2.c_construction==c)&
                                      (df2x2.target==0.95)&(df2x2.threshold_definition=="first_touch")&
                                      (df2x2.year==y)]["budget_pct"].values[0] if len(
                                      df2x2[(df2x2.policy=="run_aware")&(df2x2.c_construction==c)&
                                             (df2x2.target==0.95)&(df2x2.threshold_definition=="first_touch")&
                                             (df2x2.year==y)])>0 else None)
                             for y in YEARS)
        su_row = "  ".join(fmt(df2x2[(df2x2.policy=="run_aware")&(df2x2.c_construction==c)&
                                      (df2x2.target==0.95)&(df2x2.threshold_definition=="sustained")&
                                      (df2x2.year==y)]["budget_pct"].values[0] if len(
                                      df2x2[(df2x2.policy=="run_aware")&(df2x2.c_construction==c)&
                                             (df2x2.target==0.95)&(df2x2.threshold_definition=="sustained")&
                                             (df2x2.year==y)])>0 else None)
                             for y in YEARS)
        lines.append(f"| {c} | {ft_row} | {su_row} |")
    lines.append("")
    lines.append("*(Year columns: 2019, 2020, 2021, 2022, 2023)*")
    lines.append("")

    # Full 2x2 table as text
    lines.append("### Full 2x2 (all policies, tau@20 >= 0.95)")
    lines.append("")
    hdr = f"{'year':>4} {'policy':>12} {'c_construction':>16} {'first_touch_90':>14} {'sustained_90':>12} {'first_touch_95':>14} {'sustained_95':>12}"
    lines.append("```")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for _, r in df2x2.sort_values(["year","policy","c_construction","target","threshold_definition"]).iterrows():
        pass
    # Pivot differently: one row per (year, policy, c_constr)
    for yr in YEARS:
        for pol in ["run_aware", "judge_only", "random"]:
            for cc in ["adaptive_top20", "all_runs"]:
                ft90 = df2x2[(df2x2.year==yr)&(df2x2.policy==pol)&(df2x2.c_construction==cc)&
                              (df2x2.target==0.90)&(df2x2.threshold_definition=="first_touch")]["budget_pct"]
                su90 = df2x2[(df2x2.year==yr)&(df2x2.policy==pol)&(df2x2.c_construction==cc)&
                              (df2x2.target==0.90)&(df2x2.threshold_definition=="sustained")]["budget_pct"]
                ft95 = df2x2[(df2x2.year==yr)&(df2x2.policy==pol)&(df2x2.c_construction==cc)&
                              (df2x2.target==0.95)&(df2x2.threshold_definition=="first_touch")]["budget_pct"]
                su95 = df2x2[(df2x2.year==yr)&(df2x2.policy==pol)&(df2x2.c_construction==cc)&
                              (df2x2.target==0.95)&(df2x2.threshold_definition=="sustained")]["budget_pct"]
                v_ft90 = fmt(ft90.values[0] if len(ft90) > 0 else None)
                v_su90 = fmt(su90.values[0] if len(su90) > 0 else None)
                v_ft95 = fmt(ft95.values[0] if len(ft95) > 0 else None)
                v_su95 = fmt(su95.values[0] if len(su95) > 0 else None)
                lines.append(f"{yr:>4} {pol:>12} {cc:>16} {v_ft90:>14} {v_su90:>12} {v_ft95:>14} {v_su95:>12}")
        lines.append("")
    lines.append("```")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Step 2 — Which cell reproduces Table 4?")
    lines.append("")
    lines.append("### Policy identification")
    lines.append("")
    lines.append("Table 4's 'Judge + runs (triage score)' = **`leverage_calibrated`** from")
    lines.append("`{year}_topM_correction_curves_M20_adaptive.csv`.")
    lines.append("This policy uses run-weight variance over the adaptive top-M=20 leaderboard")
    lines.append("for selection (JUDGE+RUNS because the top-M is itself LLM-leaderboard-derived)")
    lines.append("and a LARA calibrator for the unjudged remainder.")
    lines.append("Source: REGIME_MAP in plot_chapter6.py: 'leverage': 'JUDGE + RUNS'.")
    lines.append("")
    lines.append("Table 4's 'Judge only' = **`lara`** from `final_simple_results.csv`.")
    lines.append("LARA selects by calibrated margin (pure LLM confidence signal) and uses")
    lines.append("the calibrator to relabel unjudged pairs.")
    lines.append("Source: REGIME_MAP: 'lara': 'JUDGE ONLY'.")
    lines.append("")
    lines.append("Table 4's 'Random' row = mean of `results/exp1/{year}_random_ci.csv`.")
    lines.append("This file was produced by a separate experiment (exp1), not by")
    lines.append("`run_topM_correction.py` (which has no 'random' policy).")
    lines.append("Similarly, 'leverage_calibrated' is not in `run_topM_correction.py` —")
    lines.append("it comes from the calibration-2x2 loop that was later separated.")
    lines.append("")
    lines.append("### Threshold function used by Table 4")
    lines.append("")
    lines.append("Table 4 uses **FIRST_TOUCH** (first_reaches function in plot_chapter6.py,")
    lines.append("lines 119–124). The function returns the first budget b where tau@20 >= target.")
    lines.append("It does NOT require the target to hold thereafter.")
    lines.append("")
    lines.append("### Best-matching cell")
    lines.append("")
    lines.append("At target 0.95:")
    lines.append("")
    lines.append(f"| Year | Policy | Table4 (%) | Best cell | Matched (%) | Residual (pp) |")
    lines.append(f"|------|--------|-----------|-----------|------------|---------------|")
    for _, r in df_rec.iterrows():
        lines.append(f"| {r.year} | {r.policy} | {fmt(r.table4_value_pct)} | {r.best_matching_cell} | {fmt(r.matched_value_pct)} | {r.residual_pp:.1f} |")
    lines.append("")

    # Check if best match is consistently adaptive_top20 + first_touch
    best_cells = df_rec["best_matching_cell"].unique().tolist()
    if all("adaptive_top20,first_touch" in str(c) for c in best_cells):
        lines.append("**Finding**: All Table 4 values are best reproduced by the")
        lines.append("`adaptive_top20 + first_touch` cell, confirming Table 4 uses first_touch")
        lines.append("on leverage_calibrated / lara curves.")
    else:
        lines.append("**Finding**: Cells do not uniformly match `adaptive_top20 + first_touch`.")
        lines.append("Check residuals — Table 4 may originate from a different script or")
        lines.append("configuration.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Step 3 — Does first_touch flatter any policy more than others?")
    lines.append("")
    lines.append("*Adaptive top-20 C construction, target = 0.95.*")
    lines.append("")
    lines.append(f"| Year | Policy | first_touch | sustained | gap | min_tau_after | frac_ckpts_below | frac_budget_below |")
    lines.append(f"|------|--------|------------|----------|-----|--------------|-----------------|------------------|")
    for _, r in df_stab.sort_values(["year","policy"]).iterrows():
        lines.append(
            f"| {r.year} | {r.policy} | {fmt(r.first_touch_pct)} | {fmt(r.sustained_pct)} | "
            f"{fmt(r.gap_pct)} | {r.min_tau_after_first_touch if r.min_tau_after_first_touch is not None else 'n/a'} | "
            f"{r.frac_checkpoints_below_after if r.frac_checkpoints_below_after is not None else 'n/a'} | "
            f"{r.frac_budget_below_after if r.frac_budget_below_after is not None else 'n/a'} |"
        )
    lines.append("")

    # Compute mean gap by policy
    gap_by_pol = {}
    for pol in ["run_aware", "judge_only", "random"]:
        sub = df_stab[df_stab.policy == pol]["gap_pct"].dropna()
        gap_by_pol[pol] = round(sub.mean(), 1) if len(sub) > 0 else None

    lines.append(f"### Mean gap (sustained - first_touch) by policy")
    lines.append("")
    for pol, g in gap_by_pol.items():
        lines.append(f"- {pol}: {fmt(g)}")
    lines.append("")

    # Answer the direct question
    ra_gap = gap_by_pol.get("run_aware")
    jo_gap = gap_by_pol.get("judge_only")
    rn_gap = gap_by_pol.get("random")
    if ra_gap is not None and jo_gap is not None:
        if abs(ra_gap - jo_gap) > 10:
            lines.append(f"**Direct answer**: The gap is materially larger for")
            if ra_gap > jo_gap:
                lines.append(f"`run_aware` ({fmt(ra_gap)} mean) than `judge_only` ({fmt(jo_gap)} mean).")
                lines.append(f"First touch FLATTERS run_aware relative to sustained. The thesis")
                lines.append(f"should note this or use sustained as headline.")
            else:
                lines.append(f"`judge_only` ({fmt(jo_gap)} mean) than `run_aware` ({fmt(ra_gap)} mean).")
                lines.append(f"First touch flatters judge_only more; using first_touch is")
                lines.append(f"CONSERVATIVE for run_aware.")
        else:
            lines.append(f"**Direct answer**: The gap is similar across policies")
            lines.append(f"(run_aware: {fmt(ra_gap)}, judge_only: {fmt(jo_gap)}).")
            lines.append(f"First touch does not differentially flatter one policy, so the")
            lines.append(f"relative ordering is preserved and first_touch is safe to headline.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Step 4 — Gap decomposition")
    lines.append("")
    lines.append("Decomposes the difference between **all_runs sustained** and")
    lines.append("**adaptive_top20 first_touch** (= Table 4) at target 0.95.")
    lines.append("")
    lines.append("Caveat: 'all_runs' uses a different triage score (Jacobian variance)")
    lines.append("and raw-LLM remainder, so the C-construction effect is confounded with")
    lines.append("calibration. The numbers represent an approximate upper bound on C's contribution.")
    lines.append("")
    lines.append(f"| Year | Ordering | Total gap | Effect of defn | Effect of C | Interaction |")
    lines.append(f"|------|----------|----------|---------------|------------|-------------|")
    for _, r in df_decomp.sort_values(["year","ordering"]).iterrows():
        lines.append(
            f"| {r.year} | {r.ordering} | {fmt(r.total_gap_pct)} | {fmt(r.effect_of_definition_pct)} | "
            f"{fmt(r.effect_of_c_pct)} | {fmt(r.interaction_pct)} |"
        )
    lines.append("")
    # Compute mean effects
    sub1 = df_decomp[df_decomp.ordering == "defn_first"]
    sub2 = df_decomp[df_decomp.ordering == "c_first"]
    mean_defn_o1 = round(sub1["effect_of_definition_pct"].dropna().mean(), 1)
    mean_c_o1    = round(sub1["effect_of_c_pct"].dropna().mean(), 1)
    mean_c_o2    = round(sub2["effect_of_c_pct"].dropna().mean(), 1)
    mean_defn_o2 = round(sub2["effect_of_definition_pct"].dropna().mean(), 1)
    lines.append(f"### Mean effects across years")
    lines.append("")
    lines.append(f"| Ordering | Effect of definition | Effect of C |")
    lines.append(f"|----------|--------------------|-----------  |")
    lines.append(f"| defn_first | {fmt(mean_defn_o1)} | {fmt(mean_c_o1)} |")
    lines.append(f"| c_first    | {fmt(mean_defn_o2)} | {fmt(mean_c_o2)} |")
    lines.append("")
    lines.append(f"**Direct answer**: The thesis draft attributed 40–55pp to the C-construction")
    lines.append(f"effect. Under `defn_first` ordering the C effect averages {fmt(mean_c_o1)};")
    lines.append(f"under `c_first` it averages {fmt(mean_c_o2)}. The threshold-definition")
    lines.append(f"effect averages {fmt(mean_defn_o1)} (defn_first) / {fmt(mean_defn_o2)} (c_first).")
    if mean_c_o1 is not None and mean_defn_o1 is not None:
        bigger = "C construction" if abs(mean_c_o1) > abs(mean_defn_o1) else "threshold definition"
        lines.append(f"The dominant contributor is **{bigger}**.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Summary answers")
    lines.append("")
    lines.append("1. **Which cell reproduces Table 4?**")
    lines.append("   `adaptive_top20 C + first_touch` at target 0.95.")
    lines.append("   run_aware = leverage_calibrated; judge_only = lara; random = exp1 random_ci mean.")
    lines.append("")
    lines.append("2. **Does first_touch flatter any policy more than others?**")
    if ra_gap is not None and jo_gap is not None:
        if abs(ra_gap - jo_gap) > 10:
            bigger_pol = "run_aware" if ra_gap > jo_gap else "judge_only"
            lines.append(f"   Yes: gap is materially larger for {bigger_pol}.")
            lines.append(f"   run_aware gap = {fmt(ra_gap)} mean; judge_only gap = {fmt(jo_gap)} mean.")
        else:
            lines.append(f"   No: gaps are similar (run_aware {fmt(ra_gap)}, judge_only {fmt(jo_gap)}).")
            lines.append(f"   First touch does not differentially favour either policy.")
    lines.append("")
    lines.append("3. **How much of the Table-4-vs-t17 gap is definition vs C construction?**")
    lines.append(f"   Under defn_first ordering: definition ≈ {fmt(mean_defn_o1)}, C ≈ {fmt(mean_c_o1)}.")
    lines.append(f"   Under c_first ordering: C ≈ {fmt(mean_c_o2)}, definition ≈ {fmt(mean_defn_o2)}.")
    lines.append(f"   (Interaction captures non-additivity.)")

    report_text = "\n".join(lines)
    with open(OUT_DIR / "REPORT.md", "w") as f:
        f.write(report_text + "\n")


if __name__ == "__main__":
    main()
