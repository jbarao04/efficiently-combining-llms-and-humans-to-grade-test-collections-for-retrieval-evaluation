# Decompose run-aware advantage into eligibility filter vs fine-grained leverage ordering

import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from v2_id_mapping import V2_YEARS, load_canonical_map, canonicalize_runs

# ── Constants ──────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "results" / "thesis_verification" / "t18_eligibility"
EXP1_DIR   = BASE_DIR / "results" / "exp1"

TAU_THRESHOLD     = 0.95
K_TOP             = 20
N_BUDGET_STEPS    = 101      # 0 %, 1 %, …, 100 % of pool
N_RANDOM_DRAWS    = 200
SEED              = 42
SMALL_CPP_PCTILE  = 10       # 10th percentile of positive C_pp as near-zero threshold

POLICIES = ["random", "judge_margin", "retrieval_count",
            "leverage_only", "triage_calibrated"]

YEARS_CFG = {
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


# ── Data loading ───────────────────────────────────────────────────────────

def load_qrels(path):
    qrels = defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qrels[parts[0]][parts[2]] = int(parts[3])
    return dict(qrels)


def load_llm_data(jsonl_path, year_queries):
    """Return (llm_qrels, margins) where margins[(qid,pid)] = top1_prob - top2_prob."""
    llm_qrels = defaultdict(dict)
    margins   = {}
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            qid = str(rec["query_id"])
            if qid not in year_queries:
                continue
            pid  = str(rec["passage_id"])
            llm_qrels[qid][pid] = int(rec["score"])
            p    = sorted([float(rec["probs"].get(str(g), 0.0)) for g in range(4)],
                          reverse=True)
            margins[(qid, pid)] = p[0] - p[1]
    return dict(llm_qrels), margins


def load_system_runs(runs_dir):
    runs = {}
    for fname in sorted(os.listdir(str(runs_dir))):
        if not fname.endswith(".txt"):
            continue
        sname = fname[:-4]
        sr    = defaultdict(list)
        with open(os.path.join(str(runs_dir), fname)) as f:
            for line in f:
                p = line.strip().split()
                if len(p) < 6:
                    continue
                sr[p[0]].append((int(p[3]), p[2]))
        for qid in sr:
            sr[qid].sort()
            sr[qid] = [pid for _, pid in sr[qid][:1000]]
        runs[sname] = dict(sr)
    return runs


def load_parquet_or_csv(path_stem):
    for suf in [".parquet", ".csv"]:
        p = path_stem.with_suffix(suf)
        if p.exists():
            return pd.read_parquet(p) if suf == ".parquet" else pd.read_csv(p, dtype=str)
    return None


# ── nDCG and ranking ──────────────────────────────────────────────────────

def ndcg_at_k(ranked_pids, qrels_q, k=10):
    if not qrels_q:
        return 0.0
    dcg  = sum(qrels_q.get(p, 0) / math.log2(i + 2)
               for i, p in enumerate(ranked_pids[:k]))
    idcg = sum(g / math.log2(i + 2)
               for i, g in enumerate(sorted(qrels_q.values(), reverse=True)[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def rank_systems(scores, system_names):
    return [n for n, _ in sorted(zip(system_names, scores),
                                  key=lambda x: (-x[1], x[0]))]


def compute_tau(gold, pred):
    rg  = {n: i for i, n in enumerate(gold)}
    tau, _ = kendalltau(list(range(len(gold))), [rg[n] for n in pred])
    return float(tau) if not np.isnan(tau) else 1.0


def compute_tau_at_k(gold, pred, K=K_TOP):
    K   = min(K, len(gold))
    top = set(gold[:K])
    go  = [s for s in gold if s in top]
    pr  = [s for s in pred  if s in top]
    if len(pr) < 2:
        return 1.0
    gr  = {n: i for i, n in enumerate(go)}
    tau, _ = kendalltau(list(range(K)), [gr[n] for n in pr])
    return float(tau) if not np.isnan(tau) else 1.0


def max_drop(gold, pred):
    gr = {n: i for i, n in enumerate(gold)}
    pr = {n: i for i, n in enumerate(pred)}
    return max(pr[n] - gr[n] for n in gold)


def compute_ndcg_mat(runs, qrels, queries, system_names):
    mat = np.zeros((len(system_names), len(queries)))
    for si, sn in enumerate(system_names):
        sr = runs.get(sn, {})
        for qi, qid in enumerate(queries):
            mat[si, qi] = ndcg_at_k(sr.get(qid, []), qrels.get(qid, {}))
    return mat


# ── Per-query ndcg table ──────────────────────────────────────────────────

def build_per_query_table(pair_order, human_qrels, llm_qrels,
                          queries, system_names, sys_top10):
    """
    Precompute per-query ndcg tables for the given correction order.

    pair_order  : list of (qid, pid) in correction order
    sys_top10   : dict {si: {qi: [pid_list]}}
    Returns:
      ndcg_qk     : list[n_q] of np.ndarray shape (K_qi+1, n_sys)
                    ndcg_qk[qi][k] = nDCG array (n_sys,) after k pairs from qi corrected
      pairs_per_q : list[n_q] of pid lists (correction order for each query)
      qi_map      : {qid: qi}
    """
    n_sys  = len(system_names)
    n_q    = len(queries)
    qi_map = {qid: i for i, qid in enumerate(queries)}

    pairs_per_q = [[] for _ in range(n_q)]
    for qid, pid in pair_order:
        qi = qi_map.get(qid)
        if qi is not None:
            pairs_per_q[qi].append(pid)

    ndcg_qk = []
    for qi, qid in enumerate(queries):
        pids_in_order = pairs_per_q[qi]
        K_qi = len(pids_in_order)

        # Start: all-LLM grades for this query (fall back to human for unpaired passages)
        grades_q = {}
        for pid, g_h in human_qrels.get(qid, {}).items():
            grades_q[pid] = llm_qrels.get(qid, {}).get(pid, g_h)

        top10_by_si = [sys_top10[si][qi] for si in range(n_sys)]

        tab = np.zeros((K_qi + 1, n_sys))
        for si in range(n_sys):
            tab[0, si] = ndcg_at_k(top10_by_si[si], grades_q)

        for k, pid in enumerate(pids_in_order, start=1):
            grades_q[pid] = human_qrels[qid][pid]
            for si in range(n_sys):
                tab[k, si] = ndcg_at_k(top10_by_si[si], grades_q)

        ndcg_qk.append(tab)

    return ndcg_qk, pairs_per_q, qi_map


# ── Correction sweep ──────────────────────────────────────────────────────

def correction_sweep(pair_order, ndcg_qk, qi_map,
                     n_pool, system_names, gold_ranking, n_q, budget_fracs):
    """
    Run correction sweep using pre-built per-query tables.

    pair_order   : global ordering of (qid, pid) to correct
    n_pool       : total pool size (denominator for budget_fracs)
    n_q          : number of queries (mean divisor for system scores)
    budget_fracs : array of fractions [0, 1]
    Returns list of dicts {budget, tau_all, tau_at_20, max_drop}.
    """
    n_sys    = len(system_names)
    qi_order = [(qi_map[qid], pid)
                for qid, pid in pair_order if qid in qi_map]
    n_restr  = len(qi_order)

    k_per_q = np.zeros(n_q, dtype=int)
    ptr     = 0
    results = []

    for b in budget_fracs:
        n_target = int(round(b * n_pool))

        while ptr < n_target and ptr < n_restr:
            qi, _ = qi_order[ptr]
            k_per_q[qi] = min(k_per_q[qi] + 1, ndcg_qk[qi].shape[0] - 1)
            ptr += 1

        sys_scores = np.zeros(n_sys)
        for qi in range(n_q):
            sys_scores += ndcg_qk[qi][k_per_q[qi]]
        sys_scores /= n_q

        ranking = rank_systems(sys_scores, system_names)
        results.append({
            "budget":    float(b),
            "tau_all":   compute_tau(gold_ranking, ranking),
            "tau_at_20": compute_tau_at_k(gold_ranking, ranking, K_TOP),
            "max_drop":  max_drop(gold_ranking, ranking),
        })

    return results


# ── Random baseline ───────────────────────────────────────────────────────

def random_baseline(eligible_pairs, human_qrels, llm_qrels,
                    runs, queries, system_names, gold_ranking,
                    n_draws, budget_fracs, seed=SEED):
    """
    Restricted random baseline: shuffle eligible pairs, correct in random order.
    Non-eligible pairs keep LLM grades (implicit: they're never in pair_order).
    Uses the sensitive-pairs optimisation for speed.
    """
    rng    = np.random.RandomState(seed)
    n_q    = len(queries)
    n_sys  = len(system_names)
    n_elig = len(eligible_pairs)

    ndcg_llm_mat   = compute_ndcg_mat(runs, llm_qrels,   queries, system_names)
    ndcg_human_mat = compute_ndcg_mat(runs, human_qrels,  queries, system_names)

    # sys_top10[si][qi] = top-10 pid list
    sys_top10 = []
    for sn in system_names:
        sr = runs.get(sn, {})
        sys_top10.append([sr.get(qid, [])[:10] for qid in queries])

    pair_to_idx = {k: i for i, k in enumerate(eligible_pairs)}

    # sensitive[si][qi] = set of eligible-pair indices that are in top-10
    #                     with different human vs llm grade
    sensitive = [[set() for _ in range(n_q)] for _ in range(n_sys)]
    for si in range(n_sys):
        for qi, qid in enumerate(queries):
            for pid in sys_top10[si][qi]:
                k   = (qid, pid)
                idx = pair_to_idx.get(k)
                if idx is None:
                    continue
                h = human_qrels.get(qid, {}).get(pid)
                l = llm_qrels.get(qid,   {}).get(pid)
                if h != l:
                    sensitive[si][qi].add(idx)

    # Precompute per-query mixed-grade dict for quick partial recompute
    def recompute_ndcg(si, qi, qid, human_idx_set):
        hq = human_qrels.get(qid, {})
        lq = llm_qrels.get(qid,   {})
        mixed = {}
        for pid in hq:
            idx = pair_to_idx.get((qid, pid))
            mixed[pid] = hq[pid] if (idx is not None and idx in human_idx_set) \
                         else lq.get(pid, 0)
        return ndcg_at_k(sys_top10[si][qi], mixed)

    agg = {b: {"tau_all": [], "tau_at_20": []} for b in budget_fracs}

    for _ in range(n_draws):
        perm = rng.permutation(n_elig)

        for b in budget_fracs:
            n_human = int(round(b * n_elig))
            h_set   = set(perm[:n_human])

            sys_scores = np.zeros(n_sys)
            for si in range(n_sys):
                total = 0.0
                for qi, qid in enumerate(queries):
                    sens = sensitive[si][qi]
                    if not sens or sens.issubset(h_set):
                        total += ndcg_human_mat[si, qi]
                    elif sens.isdisjoint(h_set):
                        total += ndcg_llm_mat[si, qi]
                    else:
                        total += recompute_ndcg(si, qi, qid, h_set)
                sys_scores[si] = total / n_q

            ranking = rank_systems(sys_scores, system_names)
            agg[b]["tau_all"].append(compute_tau(gold_ranking, ranking))
            agg[b]["tau_at_20"].append(compute_tau_at_k(gold_ranking, ranking, K_TOP))

    results = []
    for b in budget_fracs:
        ta = agg[b]["tau_all"]
        t2 = agg[b]["tau_at_20"]
        results.append({
            "budget":         float(b),
            "tau_all_mean":   float(np.mean(ta)),
            "tau_all_lo":     float(np.percentile(ta, 2.5)),
            "tau_all_hi":     float(np.percentile(ta, 97.5)),
            "tau_at_20_mean": float(np.mean(t2)),
            "tau_at_20_lo":   float(np.percentile(t2, 2.5)),
            "tau_at_20_hi":   float(np.percentile(t2, 97.5)),
        })
    return results


# ── Sustained threshold ───────────────────────────────────────────────────

def sustained_threshold(curve, metric="tau_at_20", thr=TAU_THRESHOLD):
    """
    Smallest budget b* such that metric >= thr for all subsequent budget steps.
    Returns (b_star, True) or (None, False).
    """
    vals = [(r["budget"], r[metric]) for r in curve]
    if not any(v >= thr for _, v in vals):
        return None, False
    last_below = max((i for i, (_, v) in enumerate(vals) if v < thr), default=-1)
    if last_below == len(vals) - 1:
        return None, False
    if last_below == -1:
        return vals[0][0], True
    return vals[last_below + 1][0], True


# ── Per-year computation ───────────────────────────────────────────────────

def run_year(year, cfg, composition_rows, sweep_rows, threshold_rows, gap_rows):
    print(f"\n{'='*65}")
    print(f"  YEAR {year}")
    print(f"{'='*65}")

    # ── Load ──────────────────────────────────────────────────────────
    human_qrels = load_qrels(cfg["qrels"])
    year_qs     = set(human_qrels.keys())
    llm_qrels, margins = load_llm_data(cfg["scores"], year_qs)
    year_qs    &= set(llm_qrels.keys())

    runs = load_system_runs(cfg["runs_dir"])
    if year in V2_YEARS:
        canonicalize_runs(runs, load_canonical_map())

    queries      = sorted(year_qs)
    system_names = sorted(runs.keys())
    n_q          = len(queries)
    n_sys        = len(system_names)
    qi_map       = {qid: i for i, qid in enumerate(queries)}

    print(f"  {n_q} queries, {n_sys} systems")

    # Universe: all (qid, pid) with both human and LLM grades
    universe = [(qid, pid)
                for qid in queries
                for pid in human_qrels.get(qid, {})
                if pid in llm_qrels.get(qid, {})]
    n_universe = len(universe)
    print(f"  Universe: {n_universe:,} pairs")

    # ── Leverage and error maps ────────────────────────────────────────
    lev_df = load_parquet_or_csv(EXP1_DIR / f"leverage_{year}")
    if lev_df is None:
        raise FileNotFoundError(f"leverage_{year} not found in {EXP1_DIR}")
    lev_df["qid"] = lev_df["qid"].astype(str)
    lev_df["pid"] = lev_df["pid"].astype(str)
    lev_df["C_pp"] = lev_df["C_pp"].astype(float)
    lev_df = lev_df[lev_df["qid"].isin(set(queries))]
    lev_map = {(row["qid"], row["pid"]): row["C_pp"]
               for _, row in lev_df.iterrows()}

    err_df = load_parquet_or_csv(EXP1_DIR / f"error_{year}")
    if err_df is None:
        raise FileNotFoundError(f"error_{year} not found in {EXP1_DIR}")
    err_df["qid"] = err_df["qid"].astype(str)
    err_df["pid"] = err_df["pid"].astype(str)
    err_df["E_eps2_cal"] = err_df["E_eps2_cal"].astype(float)
    err_df = err_df[err_df["qid"].isin(set(queries))]
    err_map = {(row["qid"], row["pid"]): row["E_eps2_cal"]
               for _, row in err_df.iterrows()}

    # ── STEP 1: Pool composition ───────────────────────────────────────
    print("\n  STEP 1: Pool composition")

    eligible_pairs = [(qid, pid) for qid, pid in universe
                      if (qid, pid) in lev_map]
    n_eligible  = len(eligible_pairs)
    n_zero_lev  = n_universe - n_eligible
    share_zero  = n_zero_lev / n_universe if n_universe > 0 else 0.0
    share_elig  = n_eligible / n_universe if n_universe > 0 else 0.0
    ratio_elig  = share_elig  # used for original-denominator conversion

    cpp_vals   = np.array([lev_map[k] for k in eligible_pairs], dtype=float)
    small_thr  = float(np.percentile(cpp_vals, SMALL_CPP_PCTILE)) if len(cpp_vals) > 0 else 0.0
    n_near_zero = int(np.sum(cpp_vals < small_thr))

    print(f"  n_total:         {n_universe:>8,}")
    print(f"  n_zero_leverage: {n_zero_lev:>8,}  ({100*share_zero:.1f}%)")
    print(f"  n_near_zero:     {n_near_zero:>8,}  (C_pp < {small_thr:.4g}, P{SMALL_CPP_PCTILE})")
    print(f"  n_eligible:      {n_eligible:>8,}  ({100*share_elig:.1f}%)")

    composition_rows.append({
        "year":                year,
        "n_pairs_total":       n_universe,
        "n_zero_leverage":     n_zero_lev,
        "share_zero_leverage": round(share_zero, 4),
        "n_near_zero":         n_near_zero,
        "small_cpp_threshold": round(small_thr, 6),
        "small_cpp_pctile":    SMALL_CPP_PCTILE,
        "n_eligible":          n_eligible,
        "share_eligible":      round(share_elig, 4),
        "threshold_used":      "C_pp = 0 (pair not in top-10 of any system)",
    })

    if share_zero > 0.60:
        print(f"\n  STOP: share_zero = {100*share_zero:.1f}% > 60%. "
              "Report Step 1 result before continuing.")
        return False   # signal to caller

    concern = "SMALL" if share_zero < 0.20 else "MODERATE"
    print(f"  Concern level: {concern}  (threshold: <20% small, >60% stop)")

    # ── Auxiliary structures ───────────────────────────────────────────
    # Retrieval count per eligible pair
    ret_count: dict = defaultdict(int)
    for sn in system_names:
        sr = runs.get(sn, {})
        for qid in queries:
            for pid in sr.get(qid, [])[:10]:
                ret_count[(qid, pid)] += 1

    # sys_top10[si][qi] = list of top-10 pids (integer-indexed)
    sys_top10 = {}
    for si, sn in enumerate(system_names):
        sr = runs.get(sn, {})
        sys_top10[si] = {qi: sr.get(qid, [])[:10]
                         for qi, qid in enumerate(queries)}

    # Gold ranking: full human reference
    ndcg_h = compute_ndcg_mat(runs, human_qrels, queries, system_names)
    gold_ranking = rank_systems(ndcg_h.mean(axis=1), system_names)

    budget_fracs = np.linspace(0, 1, N_BUDGET_STEPS)

    # ── Policy ordering factories ──────────────────────────────────────
    def order_restricted(pol):
        if pol == "judge_margin":
            return sorted(eligible_pairs, key=lambda k: margins.get(k, 0.5))
        elif pol == "retrieval_count":
            return sorted(eligible_pairs, key=lambda k: -ret_count.get(k, 0))
        elif pol == "leverage_only":
            return sorted(eligible_pairs, key=lambda k: -lev_map.get(k, 0.0))
        elif pol == "triage_calibrated":
            return sorted(eligible_pairs,
                          key=lambda k: -(err_map.get(k, 0.0) * lev_map.get(k, 0.0)))
        raise ValueError(pol)

    def order_unrestricted(pol):
        if pol == "judge_margin":
            return sorted(universe, key=lambda k: margins.get(k, 0.5))
        elif pol == "retrieval_count":
            return sorted(universe, key=lambda k: -ret_count.get(k, 0))
        elif pol == "leverage_only":
            return sorted(universe, key=lambda k: -lev_map.get(k, 0.0))
        elif pol == "triage_calibrated":
            return sorted(universe,
                          key=lambda k: -(err_map.get(k, 0.0) * lev_map.get(k, 0.0)))
        raise ValueError(pol)

    # ── STEP 2: Restricted sweeps ──────────────────────────────────────
    print("\n  STEP 2: Restricted and unrestricted sweeps")

    curves_r = {}   # restricted:   budget in [0,1] of eligible pairs
    curves_u = {}   # unrestricted: budget in [0,1] of all universe pairs

    for pol in [p for p in POLICIES if p != "random"]:
        # Restricted
        print(f"    restricted   {pol:<22} ...", end="", flush=True)
        ord_r = order_restricted(pol)
        qk_r, _, _ = build_per_query_table(
            ord_r, human_qrels, llm_qrels, queries, system_names, sys_top10)
        c_r = correction_sweep(ord_r, qk_r, qi_map,
                               n_eligible, system_names, gold_ranking, n_q, budget_fracs)
        curves_r[pol] = c_r
        b_r, ok_r = sustained_threshold(c_r, "tau_at_20")
        print(f" thr={b_r*100:.1f}%" if ok_r else " never")

        # Unrestricted
        print(f"    unrestricted {pol:<22} ...", end="", flush=True)
        ord_u = order_unrestricted(pol)
        qk_u, _, _ = build_per_query_table(
            ord_u, human_qrels, llm_qrels, queries, system_names, sys_top10)
        c_u = correction_sweep(ord_u, qk_u, qi_map,
                               n_universe, system_names, gold_ranking, n_q, budget_fracs)
        curves_u[pol] = c_u
        b_u, ok_u = sustained_threshold(c_u, "tau_at_20")
        print(f" thr={b_u*100:.1f}%" if ok_u else " never")

    # Random restricted baseline
    print(f"    random ({N_RANDOM_DRAWS} draws) ...", end="", flush=True)
    rand_curve = random_baseline(
        eligible_pairs, human_qrels, llm_qrels, runs, queries, system_names,
        gold_ranking, N_RANDOM_DRAWS, budget_fracs, seed=SEED)
    curves_r["random"] = rand_curve
    rand_det  = [{"budget": r["budget"], "tau_at_20": r["tau_at_20_mean"]}
                 for r in rand_curve]
    b_rand, ok_rand = sustained_threshold(rand_det, "tau_at_20")
    print(f" thr={b_rand*100:.1f}%" if ok_rand else " never")

    # ── Collect sweep rows ─────────────────────────────────────────────
    for pol in POLICIES:
        if pol == "random":
            for r in rand_curve:
                sweep_rows.append({
                    "year":                  year,
                    "policy":                pol,
                    "budget_restricted_pct": round(r["budget"] * 100, 1),
                    "budget_original_pct":   round(r["budget"] * ratio_elig * 100, 2),
                    "budget_pairs":          int(round(r["budget"] * n_eligible)),
                    "tau_all":               round(r["tau_all_mean"], 4),
                    "tau_at20":              round(r["tau_at_20_mean"], 4),
                    "max_drop":              None,
                    "draw_id":               -1,
                })
        else:
            for r in curves_r[pol]:
                sweep_rows.append({
                    "year":                  year,
                    "policy":                pol,
                    "budget_restricted_pct": round(r["budget"] * 100, 1),
                    "budget_original_pct":   round(r["budget"] * ratio_elig * 100, 2),
                    "budget_pairs":          int(round(r["budget"] * n_eligible)),
                    "tau_all":               round(r["tau_all"], 4),
                    "tau_at20":              round(r["tau_at_20"], 4),
                    "max_drop":              r["max_drop"],
                    "draw_id":               -1,
                })

    # ── STEP 3: Threshold table ────────────────────────────────────────
    print("\n  STEP 3: Threshold table")

    def get_thr(pol, pool="restricted"):
        """Return (b_star_fraction, ok) for the given pool."""
        if pool == "restricted":
            if pol == "random":
                return sustained_threshold(rand_det, "tau_at_20")
            return sustained_threshold(curves_r[pol], "tau_at_20")
        else:  # unrestricted
            if pol == "random":
                return None, False
            return sustained_threshold(curves_u[pol], "tau_at_20")

    print(f"\n  {'Policy':<22} {'Restr%':>7} {'Orig%':>7} {'Unrest%':>9} {'Delta(O-U)':>11}")
    print(f"  {'-'*60}")

    for pol in POLICIES:
        b_r, ok_r = get_thr(pol, "restricted")
        b_u, ok_u = get_thr(pol, "unrestricted")

        thr_r = round(b_r * 100, 1)              if ok_r else None
        thr_o = round(b_r * ratio_elig * 100, 1) if ok_r else None
        thr_u = round(b_u * 100, 1)              if ok_u else None
        delta  = round(thr_u - thr_o, 1)          if (thr_o is not None and thr_u is not None) else None

        sr = str(thr_r) if thr_r is not None else "N/A"
        so = str(thr_o) if thr_o is not None else "N/A"
        su = str(thr_u) if thr_u is not None else "N/A"
        sd = str(delta)  if delta  is not None else "N/A"
        print(f"  {pol:<22} {sr:>7} {so:>7} {su:>9} {sd:>11}")

        for denom, thr_val in [("restricted", thr_r), ("original", thr_o)]:
            threshold_rows.append({
                "year":                       year,
                "policy":                     pol,
                "denominator":                denom,
                "sustained_threshold_pct":    thr_val,
                "unrestricted_threshold_pct": thr_u,
                "delta":                      delta,
            })

    # ── STEP 4: Policy gaps ────────────────────────────────────────────
    print("\n  STEP 4: Policy gaps")

    def thr_pct(pol, setting):
        """Return threshold in percentage for a given setting."""
        if setting == "1_unrestricted_original":
            b, ok = get_thr(pol, "unrestricted")
            return round(b * 100, 2) if ok else None
        elif setting == "2_restricted_restricted":
            b, ok = get_thr(pol, "restricted")
            return round(b * 100, 2) if ok else None
        elif setting == "3_restricted_original":
            b, ok = get_thr(pol, "restricted")
            return round(b * ratio_elig * 100, 2) if ok else None
        return None

    comparisons = [
        ("triage_calibrated", "judge_margin",
         "triage_calibrated vs judge_margin"),
        ("triage_calibrated", "leverage_only",
         "triage_calibrated vs leverage_only"),
        ("leverage_only", "retrieval_count",
         "leverage_only vs retrieval_count"),
        ("judge_margin", "retrieval_count",
         "judge_margin vs retrieval_count"),
    ]
    settings = [
        "1_unrestricted_original",
        "2_restricted_restricted",
        "3_restricted_original",
    ]

    print(f"  {'Comparison':<38} {'S1(pp)':>7} {'S2(pp)':>7} {'S3(pp)':>7}")
    print(f"  {'-'*62}")

    for pol_a, pol_b, desc in comparisons:
        parts = []
        for s in settings:
            t_a = thr_pct(pol_a, s)
            t_b = thr_pct(pol_b, s)
            gap = round(t_b - t_a, 2) if (t_a is not None and t_b is not None) else None
            parts.append(gap)

            gap_rows.append({
                "year":               year,
                "comparison":         f"{pol_a}_vs_{pol_b}",
                "setting":            s,
                "gap_pct_points":     gap,
                "threshold_pol_a_pct": t_a,
                "threshold_pol_b_pct": t_b,
            })

        row_str = "  " + "  ".join(
            [f"{desc:<38}"] +
            [f"{(f'{p:+.1f}' if p is not None else 'N/A'):>7}" for p in parts]
        )
        print(row_str)

    return True  # success


# ── REPORT writer ──────────────────────────────────────────────────────────

def write_report(composition_rows, threshold_rows, gap_rows):
    comp_df = pd.DataFrame(composition_rows)
    thr_df  = pd.DataFrame(threshold_rows)
    gap_df  = pd.DataFrame(gap_rows)

    pooled_total = comp_df["n_pairs_total"].sum()
    pooled_zero  = comp_df["n_zero_leverage"].sum()
    pooled_elig  = comp_df["n_eligible"].sum()
    avg_zero_pct = comp_df["share_zero_leverage"].mean() * 100
    avg_elig_pct = comp_df["share_eligible"].mean() * 100

    lines = [
        "# T18 Eligibility Baseline — Report",
        "",
        "## Question",
        "",
        "How much of the run-aware triage advantage is due to the eligibility filter",
        "(triage score = 0 for pairs not retrieved in any system's top-10) versus",
        "fine-grained ordering within the eligible pool?",
        "",
        "## Definitions",
        "",
        "- **Zero-leverage pair**: a judged passage that appears in no system's top-10.",
        "  Its grade cannot change any system's nDCG@10 regardless of whether it is",
        "  corrected. C_pp = 0 exactly.",
        "- **Eligible pair**: C_pp > 0. At least one system places this passage in",
        "  its top-10, so its grade affects the evaluation outcome.",
        "- **triage_calibrated** (this script): static sort by descending",
        "  E[eps^2]_cal * C_pp. This is the offline equivalent of the LARA policy",
        "  in Table 4 (same selection signal, no online calibration updates).",
        "- **Threshold C_pp = 0**: a pair is ineligible iff it is absent from the",
        "  leverage parquet, which only stores pairs retrieved in at least one top-10.",
        "- **Small C_pp threshold**: P10 of positive C_pp values per year, flagging",
        "  technically eligible pairs with negligible positional weight.",
        "",
        "## Step 1: Pool Composition",
        "",
        "| Year | Total pairs | Zero-leverage | Share (%) | Near-zero (P10) | Eligible | Share (%) |",
        "|------|-------------|---------------|-----------|-----------------|----------|-----------|",
    ]

    for _, row in comp_df.iterrows():
        lines.append(
            f"| {int(row['year'])} "
            f"| {int(row['n_pairs_total']):,} "
            f"| {int(row['n_zero_leverage']):,} "
            f"| {row['share_zero_leverage']*100:.1f} "
            f"| {int(row['n_near_zero']):,} "
            f"| {int(row['n_eligible']):,} "
            f"| {row['share_eligible']*100:.1f} |"
        )
    lines += [
        f"| **All** | **{pooled_total:,}** | **{pooled_zero:,}** "
        f"| **{100*pooled_zero/pooled_total:.1f}** | — "
        f"| **{pooled_elig:,}** | **{100*pooled_elig/pooled_total:.1f}** |",
        "",
    ]

    if avg_zero_pct < 20:
        concern = (
            f"The zero-leverage share averages {avg_zero_pct:.1f}% across years "
            f"(< 20%). The eligibility filter is a small fraction of the candidate "
            f"space. The rest of this analysis is confirmatory."
        )
    elif avg_zero_pct > 60:
        concern = (
            f"The zero-leverage share averages {avg_zero_pct:.1f}% across years "
            f"(> 60% threshold). **Analysis stopped.** The eligibility filter "
            f"accounts for the majority of the pool; the thesis framing needs revisiting."
        )
    else:
        concern = (
            f"The zero-leverage share averages {avg_zero_pct:.1f}% across years "
            f"(between 20% and 60%). The filter is a moderate factor. "
            f"The restricted-pool analysis below isolates how much advantage remains "
            f"after removing it."
        )
    lines += [concern, ""]

    # Threshold table (restricted denom columns: R, O, U)
    lines += [
        "## Step 3: Sustained Thresholds (tau@20 >= 0.95)",
        "",
        "R% = budget as % of **eligible** pairs (restricted denominator).",
        "O% = budget as % of **all universe** pairs (original denominator; R% × share_eligible).",
        "U% = unrestricted sweep, original denominator (all universe pairs, ineligible last).",
        "",
    ]

    years = sorted(comp_df["year"].unique().tolist())
    # Header
    header = "| Policy |" + "".join(f" {y} R% | {y} O% | {y} U% |" for y in years)
    sep    = "|--------|" + "".join("|------|----|-----|" for _ in years)
    lines += [header, sep]

    for pol in POLICIES:
        row_parts = [f"| {pol} |"]
        for y in years:
            sub = thr_df[(thr_df["year"] == y) & (thr_df["policy"] == pol)]
            r_v = sub[sub["denominator"] == "restricted"]["sustained_threshold_pct"]
            o_v = sub[sub["denominator"] == "original"]["sustained_threshold_pct"]
            u_v = sub[sub["denominator"] == "restricted"]["unrestricted_threshold_pct"]
            rv = f"{r_v.values[0]:.0f}" if (len(r_v) > 0 and r_v.values[0] is not None) else "N/A"
            ov = f"{o_v.values[0]:.0f}" if (len(o_v) > 0 and o_v.values[0] is not None) else "N/A"
            uv = f"{u_v.values[0]:.0f}" if (len(u_v) > 0 and u_v.values[0] is not None
                                              and not pd.isna(u_v.values[0])) else "N/A"
            row_parts.append(f" {rv} | {ov} | {uv} |")
        lines.append("".join(row_parts))
    lines.append("")

    # Gap tables
    lines += [
        "## Step 4: Policy Gaps (percentage points)",
        "",
        "**Positive gap** = pol_b reaches 0.95 later than pol_a → pol_a is more efficient.",
        "",
        "Three settings:",
        "- S1: Unrestricted pool, original denominator (thesis baseline)",
        "- S2: Restricted pool, restricted denominator (eligibility filter removed)",
        "- S3: Restricted pool, original denominator (apples-to-apples with Table 4)",
        "",
        "Key interpretations:",
        "- If S1 gap >> S2 gap: advantage is mainly the eligibility filter.",
        "- If S1 gap ≈ S2 gap: advantage is mainly fine ordering within eligible pairs.",
        "- triage_calibrated vs leverage_only (S2/S3): judge signal contribution within eligible pool.",
        "",
    ]

    key_comps = [
        "triage_calibrated_vs_judge_margin",
        "triage_calibrated_vs_leverage_only",
        "leverage_only_vs_retrieval_count",
        "judge_margin_vs_retrieval_count",
    ]

    for comp in key_comps:
        sub = gap_df[gap_df["comparison"] == comp]
        if sub.empty:
            continue
        lines.append(f"### {comp.replace('_vs_', ' vs ')}")
        lines.append("")
        lines.append("| Year | S1 (unrest, orig) | S2 (restr, restr) | S3 (restr, orig) |")
        lines.append("|------|-------------------|-------------------|------------------|")
        for y in sorted(gap_df["year"].unique()):
            ry = sub[sub["year"] == y]
            def gv(s):
                c = ry[ry["setting"] == s]["gap_pct_points"]
                return f"{c.values[0]:+.1f}" if len(c) > 0 and c.values[0] is not None else "N/A"
            lines.append(f"| {y} | {gv('1_unrestricted_original')} "
                         f"| {gv('2_restricted_restricted')} "
                         f"| {gv('3_restricted_original')} |")
        lines.append("")

    # Pooled means
    lines += ["### Pooled means across years", ""]
    lines.append("| Comparison | S1 | S2 | S3 |")
    lines.append("|------------|----|----|-----|")
    for comp in key_comps:
        sub = gap_df[gap_df["comparison"] == comp]
        if sub.empty:
            continue
        means = []
        for s in ["1_unrestricted_original", "2_restricted_restricted", "3_restricted_original"]:
            vals = sub[sub["setting"] == s]["gap_pct_points"].dropna()
            means.append(f"{vals.mean():+.1f}" if len(vals) > 0 else "N/A")
        lines.append(f"| {comp} | {means[0]} | {means[1]} | {means[2]} |")
    lines.append("")

    # Answer
    lines += [
        "## Answer: How Much of the Advantage is the Eligibility Filter?",
        "",
    ]

    # Compute the answer from data
    tc_jm_sub = gap_df[gap_df["comparison"] == "triage_calibrated_vs_judge_margin"]
    if not tc_jm_sub.empty:
        s1_vals = tc_jm_sub[tc_jm_sub["setting"] == "1_unrestricted_original"]["gap_pct_points"].dropna()
        s3_vals = tc_jm_sub[tc_jm_sub["setting"] == "3_restricted_original"]["gap_pct_points"].dropna()
        s2_vals = tc_jm_sub[tc_jm_sub["setting"] == "2_restricted_restricted"]["gap_pct_points"].dropna()

        s1_mean = s1_vals.mean() if len(s1_vals) > 0 else float("nan")
        s3_mean = s3_vals.mean() if len(s3_vals) > 0 else float("nan")
        s2_mean = s2_vals.mean() if len(s2_vals) > 0 else float("nan")

        filter_contribution = s1_mean - s3_mean if not (np.isnan(s1_mean) or np.isnan(s3_mean)) else float("nan")
        ordering_contribution = s3_mean if not np.isnan(s3_mean) else float("nan")

        tc_lo_sub = gap_df[gap_df["comparison"] == "triage_calibrated_vs_leverage_only"]
        lo_s2 = tc_lo_sub[tc_lo_sub["setting"] == "2_restricted_restricted"]["gap_pct_points"].dropna()
        lo_s2_mean = lo_s2.mean() if len(lo_s2) > 0 else float("nan")

        lines += [
            f"**triage_calibrated vs judge_margin:**",
            f"- S1 (unrestricted): {s1_mean:+.1f} pp average gap" if not np.isnan(s1_mean) else "- S1: N/A",
            f"- S3 (restricted, original denom): {s3_mean:+.1f} pp average gap" if not np.isnan(s3_mean) else "- S3: N/A",
            f"- Filter contribution (S1 − S3): {filter_contribution:+.1f} pp" if not np.isnan(filter_contribution) else "- Filter contribution: N/A",
            f"- Ordering contribution within eligible pool (S3): {ordering_contribution:+.1f} pp" if not np.isnan(ordering_contribution) else "- Ordering: N/A",
            "",
            f"**triage_calibrated vs leverage_only (S2, restricted denom):**",
            f"- Gap: {lo_s2_mean:+.1f} pp (contribution of judge error estimate on top of run leverage)" if not np.isnan(lo_s2_mean) else "- Gap: N/A",
            "",
        ]

        if abs(s1_mean) < 1 and abs(s3_mean) < 1:
            lines.append(
                "**Conclusion**: Both the unrestricted and restricted gaps are near zero. "
                "Neither the eligibility filter nor the fine ordering within the eligible pool "
                "explains the advantage — the advantage itself is small."
            )
        elif not np.isnan(filter_contribution) and filter_contribution > 0.5 * abs(s1_mean):
            lines.append(
                f"**Conclusion**: The eligibility filter ({filter_contribution:+.1f} pp) accounts for "
                f"a substantial share of the total unrestricted advantage ({s1_mean:+.1f} pp). "
                f"The remaining ordering advantage within the eligible pool is "
                f"{ordering_contribution:+.1f} pp in original-denominator terms."
            )
        elif not np.isnan(s1_mean) and not np.isnan(s3_mean):
            lines.append(
                f"**Conclusion**: The eligibility filter contributes {filter_contribution:+.1f} pp "
                f"of the {s1_mean:+.1f} pp total advantage. "
                f"Most of the advantage ({ordering_contribution:+.1f} pp) comes from fine ordering "
                f"within the eligible pool. The triage advantage is real even after restricting "
                f"to pairs that actually matter for nDCG@10."
            )
    else:
        lines.append("(Gap data unavailable — no years completed Step 4.)")

    lines += [
        "",
        "---",
        "These results should be presented in the thesis as a control experiment, "
        "not as a replacement for Table 4. Both the unrestricted and restricted numbers "
        "belong in the text. The unrestricted comparison shows the full practical advantage; "
        "the restricted comparison isolates the source of that advantage.",
    ]

    report_path = OUTPUT_DIR / "REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved {report_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  T18: NON-ZERO-LEVERAGE BASELINE TEST")
    print("=" * 70)

    composition_rows = []
    sweep_rows       = []
    threshold_rows   = []
    gap_rows         = []

    any_stopped = False
    for year in sorted(YEARS_CFG.keys()):
        ok = run_year(year, YEARS_CFG[year],
                      composition_rows, sweep_rows,
                      threshold_rows, gap_rows)
        if ok is False:
            any_stopped = True

    # ── Step 1 summary before Step 2 output ───────────────────────────
    print("\n\n" + "=" * 70)
    print("  STEP 1 SUMMARY (pool composition)")
    print("=" * 70)
    comp_df = pd.DataFrame(composition_rows)
    print(comp_df.to_string(index=False))

    avg_zero = comp_df["share_zero_leverage"].mean() * 100
    if avg_zero < 20:
        print(f"\n  Average zero-leverage share: {avg_zero:.1f}% < 20%")
        print("  Concern is SMALL — analysis is confirmatory.")
    elif avg_zero > 60:
        print(f"\n  Average zero-leverage share: {avg_zero:.1f}% > 60%")
        print("  STOP: Thesis framing needs rethinking before further compute.")
    else:
        print(f"\n  Average zero-leverage share: {avg_zero:.1f}% (20-60% range)")
        print("  MODERATE concern — see restricted-pool results.")

    # ── Save outputs ───────────────────────────────────────────────────
    print("\n\nSaving outputs...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(composition_rows).to_csv(
        OUTPUT_DIR / "pool_composition.csv", index=False)
    print("  pool_composition.csv")

    if sweep_rows:
        try:
            pd.DataFrame(sweep_rows).to_parquet(
                OUTPUT_DIR / "restricted_sweeps.parquet", index=False)
            print("  restricted_sweeps.parquet")
        except Exception:
            pd.DataFrame(sweep_rows).to_csv(
                OUTPUT_DIR / "restricted_sweeps.csv", index=False)
            print("  restricted_sweeps.csv (parquet unavailable)")

    pd.DataFrame(threshold_rows).to_csv(
        OUTPUT_DIR / "restricted_thresholds.csv", index=False)
    print("  restricted_thresholds.csv")

    pd.DataFrame(gap_rows).to_csv(
        OUTPUT_DIR / "policy_gaps.csv", index=False)
    print("  policy_gaps.csv")

    write_report(composition_rows, threshold_rows, gap_rows)

    print("\nDone.")


if __name__ == "__main__":
    main()
