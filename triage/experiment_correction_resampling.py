# Budget sweep for all correction policies with paired-query bootstrap confidence intervals

import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import json
import math
import os
import time
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import kendalltau, spearmanr, pearsonr

try:
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from v2_id_mapping import V2_YEARS, load_canonical_map, canonicalize_runs


# ---------------------------------------------------------------------------
#  COMPAT
# ---------------------------------------------------------------------------

_TRAPZ = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
#  CONSTANTS
# ---------------------------------------------------------------------------

BASE_DIR   = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "results" / "thesis_verification" / "t12_resampling"

SEED             = 42
B_INITIAL        = 200
B_FULL           = 1000
N_BUDGET_STEPS   = 101      # 0%, 1%, ..., 100%  -> the 1% grid of the thesis
TAU_THRESHOLD    = 0.95
K_TOP            = 20       # tau@K and the size of the adaptive target set
BATCH_FRACTION   = 0.01     # adaptive policies re-score every 1% of the pool
N_RAND_TABLES    = 20       # distinct random orderings cycled over bootstraps

# MTF / MM-NS treat a passage as relevant at human grade >= 2.  TREC DL
# convention, and LARA's l/2 rule with l = 3 gives the same threshold.
RELEVANCE_THRESHOLD = 2

# Calibrated error estimator
CAL_MIN_CELL = 20
CAL_BIN_TRY  = (3, 2, 1)    # 1 means "emitted grade only"

# MaxMean non-stationary (optional, not in the thesis table)
MM_NS_WINDOW  = 50
MM_NS_EPSILON = 0.1

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

CONFIDENCE_CSV = BASE_DIR / "results" / "spectral" / "confidence_passage_linear.csv"
DAMAGE_DIR     = BASE_DIR / "results" / "triage_impact"
SPECTRAL_INT   = BASE_DIR / "results" / "spectral" / "intermediates"
DDEC_CSV       = BASE_DIR / "results" / "spectral" / "damage_decoupling_linear.csv"

# Policy identifier -> the name used in the thesis table and figures.
DISPLAY_NAME = {
    "random":       "Random",
    "naive":        "Confidence, raw margin (Naive)",
    "lara":         "Confidence, calibrated margin (LARA)",
    "mtf":          "Move-to-front pooling",
    "leverage":     "Leverage",
    "product_raw":  "Product, raw",
    "product_cal":  "Product, calibrated",
    "oracle":       "Oracle (error magnitude)",
    "mm_ns":        "MaxMean non-stationary",
    "depth_k":      "Depth-k pooling",
    "retrieval_count": "Retrieval count",
}

READS = {
    "random": "neither", "naive": "judge", "lara": "judge", "mtf": "runs",
    "leverage": "runs", "product_raw": "both", "product_cal": "both",
    "oracle": "labels", "mm_ns": "runs", "depth_k": "runs",
    "retrieval_count": "runs",
}

# Rows of the thesis table, in order.
TABLE_POLICIES = ["random", "naive", "lara", "mtf", "leverage",
                  "product_raw", "product_cal", "oracle"]

# Extra judge-side signals, used only for the oracle-share numbers.
EXTRA_POLICIES = ["max_prob", "entropy"]


# ---------------------------------------------------------------------------
#  LINEAR GAIN
# ---------------------------------------------------------------------------

def gain(g):
    """Linear gain, per Eq. (4.1).  NOT 2^g - 1."""
    return float(g)


# ---------------------------------------------------------------------------
#  LOADING
# ---------------------------------------------------------------------------

def load_qrels(path):
    qrels = defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            g = int(parts[3])
            qrels[parts[0]][parts[2]] = max(g, 0)   # TREC uses -1 for "not judged"
    return dict(qrels)


def load_llm_data(jsonl_path, year_queries):
    """Return (grades, probs).  Probabilities are renormalised to sum to one,
    which is LARA Eq. (1)."""
    grades = defaultdict(dict)
    probs  = {}
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            qid = str(rec["query_id"])
            if qid not in year_queries:
                continue
            pid = str(rec["passage_id"])
            grades[qid][pid] = int(rec["score"])
            p = np.array([float(rec["probs"].get(str(k), 0.0)) for k in range(4)],
                         dtype=np.float64)
            s = p.sum()
            probs[(qid, pid)] = p / s if s > 0 else np.full(4, 0.25)
    return dict(grades), probs


def load_system_runs(runs_dir):
    runs = {}
    for fname in sorted(os.listdir(runs_dir)):
        if not fname.endswith(".txt"):
            continue
        sname = fname[:-4]
        sr = defaultdict(list)
        with open(os.path.join(runs_dir, fname)) as f:
            for line in f:
                p = line.strip().split()
                if len(p) < 6:
                    continue
                sr[p[0]].append((int(p[3]), p[2]))
        for qid in sr:
            sr[qid].sort()
            seen, out = set(), []
            for _, pid in sr[qid]:
                if pid in seen:          # duplicate ids can survive v2 mapping
                    continue
                seen.add(pid)
                out.append(pid)
                if len(out) >= 1000:
                    break
            sr[qid] = out
        runs[sname] = dict(sr)
    return runs


# ---------------------------------------------------------------------------
#  nDCG, LINEAR GAIN, IDCG SHARED ACROSS SYSTEMS
# ---------------------------------------------------------------------------

_DISCOUNT = np.array([1.0 / math.log2(i + 2) for i in range(10)])


def idcg_from_counter(cnt, k=10):
    """Ideal DCG at cutoff k from a Counter of grades.  Linear gain."""
    out, i = 0.0, 0
    for g in (3, 2, 1, 0):
        m = cnt.get(g, 0)
        while m > 0 and i < k:
            out += gain(g) * _DISCOUNT[i]
            i += 1
            m -= 1
        if i >= k:
            break
    return out


def dcg_at_k(ranked_pids, grades_q, k=10):
    out = 0.0
    for i, p in enumerate(ranked_pids[:k]):
        out += gain(grades_q.get(p, 0)) * _DISCOUNT[i]
    return out


def ndcg_matrix_from_grades(grades_by_q, queries, system_names, sys_top10,
                            out=None, dirty=None):
    """nDCG@10 for every (system, query).  IDCG computed once per query.

    `dirty` is an iterable of query indices to refresh; everything else is
    left alone in `out`.  Pass dirty=None to refresh all.
    """
    n_sys, n_q = len(system_names), len(queries)
    if out is None:
        out = np.zeros((n_sys, n_q))
        dirty = range(n_q)
    if dirty is None:
        dirty = range(n_q)
    for qi in dirty:
        gq = grades_by_q.get(queries[qi], {})
        idcg = idcg_from_counter(Counter(gq.values()))
        if idcg <= 0:
            out[:, qi] = 0.0
            continue
        for si in range(n_sys):
            out[si, qi] = dcg_at_k(sys_top10[si][qi], gq) / idcg
    return out


def rank_systems(scores, system_names):
    return [n for n, _ in sorted(zip(system_names, scores), key=lambda x: (-x[1], x[0]))]


def compute_tau(gold, pred):
    rank_g = {n: i for i, n in enumerate(gold)}
    tau, _ = kendalltau(list(range(len(gold))), [rank_g[n] for n in pred])
    return float(tau) if not np.isnan(tau) else 1.0


def compute_tau_at_k(gold, pred, K=K_TOP):
    K = min(K, len(gold))
    top = set(gold[:K])
    go = [s for s in gold if s in top]
    pr = [s for s in pred if s in top]
    if len(pr) < 2:
        return 1.0
    gr = {n: i for i, n in enumerate(go)}
    tau, _ = kendalltau(list(range(K)), [gr[n] for n in pr])
    return float(tau) if not np.isnan(tau) else 1.0


def compute_max_drop(gold, pred):
    gr = {n: i for i, n in enumerate(gold)}
    pr = {n: i for i, n in enumerate(pred)}
    return max(pr[n] - gr[n] for n in gold)


# ---------------------------------------------------------------------------
#  AUXILIARY STRUCTURES
# ---------------------------------------------------------------------------

def build_sys_top10(runs, queries, system_names):
    return [
        {qi: runs.get(sn, {}).get(qid, [])[:10] for qi, qid in enumerate(queries)}
        for sn in system_names
    ]


def build_pair_weight_matrix(runs, queries, system_names, universe):
    """Sparse (n_pairs x n_sys) matrix of nDCG position weights, Eq. (4.4).

    w[p, s] = 1 / log2(rank_s(p) + 1) if p is in s's top ten, else 0.
    """
    pair_index = {k: i for i, k in enumerate(universe)}
    rows, cols, vals = [], [], []
    for si, sn in enumerate(system_names):
        sr = runs.get(sn, {})
        for qid in queries:
            for r, pid in enumerate(sr.get(qid, [])[:10]):
                j = pair_index.get((qid, pid))
                if j is None:
                    continue
                rows.append(j)
                cols.append(si)
                vals.append(_DISCOUNT[r])
    W = sp.csr_matrix((vals, (rows, cols)),
                      shape=(len(universe), len(system_names)))
    return W.tocsc(), pair_index


def leverage_over(Wc, sys_idx):
    """C_pp = population variance of the position weight across the given
    systems, Eq. (4.5) diagonal.  Zeros count."""
    M = len(sys_idx)
    if M == 0:
        return np.zeros(Wc.shape[0])
    sub = Wc[:, list(sys_idx)]
    s1 = np.asarray(sub.sum(axis=1)).ravel()
    s2 = np.asarray(sub.multiply(sub).sum(axis=1)).ravel()
    return np.maximum(s2 / M - (s1 / M) ** 2, 0.0)


def build_pool_depth(runs, queries, system_names, universe_set):
    depth, n_sys = {}, {}
    for sn in system_names:
        sr = runs.get(sn, {})
        for qid in queries:
            for ri, pid in enumerate(sr.get(qid, [])):
                k = (qid, pid)
                if k not in universe_set:
                    continue
                depth[k] = min(depth.get(k, 10 ** 9), ri + 1)
                n_sys[k] = n_sys.get(k, 0) + 1
    return depth, n_sys


# ---------------------------------------------------------------------------
#  EXPECTED SQUARED ERROR,  Eq. (4.13)
# ---------------------------------------------------------------------------

def expected_sq_error_raw(universe, llm_grades, softmax_probs):
    """E[eps_p^2]_raw = sum_h pi_h(p) (g_L(p) - h)^2.  Linear gain.

    Label free.  Assumes the judge is calibrated, which Sec. 3.2 shows it
    is not: on a passage confidently graded 2, almost no mass sits on
    grade 1, so a human-grade-1 passage gets a near-zero error term.
    """
    out = np.empty(len(universe))
    grades_g = np.arange(4, dtype=np.float64)
    for i, (qid, pid) in enumerate(universe):
        pi = softmax_probs.get((qid, pid))
        if pi is None:
            out[i] = 0.0
            continue
        gl = float(llm_grades[qid][pid])
        out[i] = float(np.sum(pi * (gl - grades_g) ** 2))
    return out


def _quantile_bin_edges(x, n_bins):
    if n_bins <= 1:
        return np.array([])
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(x, qs))
    return edges


def expected_sq_error_calibrated(universe, llm_grades, human_qrels, softmax_probs,
                                 min_cell=CAL_MIN_CELL, bin_try=CAL_BIN_TRY,
                                 verbose=True):
    """E[eps_p^2]_cal = sum_h P(g_h = h | g_L(p), bin(p)) (g_L(p) - h)^2.

    The table is fitted LEAVE-ONE-QUERY-OUT within the year, and never
    across the v1 / v2 boundary because each year is processed alone.

    Bin edges come from max_prob quantiles.  max_prob uses no human labels,
    so the edges leak nothing.  Leave-one-query-out is exact and cheap
    because counts are additive: subtract the held-out query's counts.

    Backoff, per Sec. 5.1: three bins, then two, then the emitted grade
    alone, whenever an occupied cell holds fewer than `min_cell` examples.
    A cell with no training mass at all falls back to the raw softmax.

    NOT label free.  It uses human grades on the other queries of the same
    year.  It is training free at inference only.
    """
    n = len(universe)
    maxp = np.array([float(np.max(softmax_probs.get(k, np.full(4, .25))))
                     for k in universe])
    gl   = np.array([llm_grades[q][p] for q, p in universe], dtype=int)
    gh   = np.array([human_qrels[q].get(p, 0) for q, p in universe], dtype=int)
    qid_of = np.array([q for q, _ in universe])

    chosen_bins, edges, bin_of, N = None, None, None, None
    for nb in bin_try:
        e = _quantile_bin_edges(maxp, nb)
        b = np.digitize(maxp, e) if nb > 1 else np.zeros(n, dtype=int)
        nb_eff = int(b.max()) + 1
        cnt = np.zeros((4, nb_eff, 4))
        np.add.at(cnt, (gl, b, gh), 1.0)
        occupied = cnt.sum(axis=2)
        ok = np.all((occupied == 0) | (occupied >= min_cell))
        if ok or nb == bin_try[-1]:
            chosen_bins, edges, bin_of, N = nb_eff, e, b, cnt
            if not ok and verbose:
                print(f"      calibration: forced to {nb_eff} bin(s); "
                      f"smallest occupied cell = {occupied[occupied > 0].min():.0f}")
            break

    if verbose:
        print(f"      calibration table: {chosen_bins} confidence bin(s), "
              f"min occupied cell = "
              f"{N.sum(axis=2)[N.sum(axis=2) > 0].min():.0f}")

    # Per-query counts, for the leave-one-out subtraction.
    per_q = defaultdict(lambda: np.zeros((4, chosen_bins, 4)))
    for i in range(n):
        per_q[qid_of[i]][gl[i], bin_of[i], gh[i]] += 1.0

    raw = expected_sq_error_raw(universe, llm_grades, softmax_probs)
    grades_g = np.arange(4, dtype=np.float64)
    out = np.empty(n)
    n_fallback = 0

    order = defaultdict(list)
    for i in range(n):
        order[qid_of[i]].append(i)

    for q, idxs in order.items():
        T = N - per_q[q]
        for i in idxs:
            row = T[gl[i], bin_of[i]]
            tot = row.sum()
            if tot < min_cell:
                row = T[gl[i]].sum(axis=0)      # grade-only backoff
                tot = row.sum()
            if tot < min_cell:
                out[i] = raw[i]                 # nothing to learn from
                n_fallback += 1
                continue
            P = row / tot
            out[i] = float(np.sum(P * (gl[i] - grades_g) ** 2))

    if verbose and n_fallback:
        print(f"      calibration: {n_fallback} pair(s) fell back to the raw "
              f"softmax ({100 * n_fallback / n:.2f}%)")
    return out, {"n_bins": chosen_bins, "n_fallback": int(n_fallback)}


# ---------------------------------------------------------------------------
#  STATIC SCORERS   (higher score = bought earlier)
# ---------------------------------------------------------------------------

def scores_naive_margin(universe, softmax_probs):
    """LARA's Naive method, their Sec. 3.2 and Eq. (2).

    m' = pi^{k'} - pi^{s'} on the RAW normalised probabilities.  Smallest
    margin first, so the score is the negated margin.
    """
    out = np.empty(len(universe))
    for i, k in enumerate(universe):
        pi = np.sort(softmax_probs.get(k, np.full(4, .25)))[::-1]
        out[i] = -(pi[0] - pi[1])
    return out


def scores_max_prob(universe, softmax_probs):
    return np.array([-float(np.max(softmax_probs.get(k, np.full(4, .25))))
                     for k in universe])


def scores_entropy(universe, softmax_probs):
    out = np.empty(len(universe))
    for i, k in enumerate(universe):
        p = np.clip(softmax_probs.get(k, np.full(4, .25)), 1e-12, 1.0)
        out[i] = float(-np.sum(p * np.log(p)))
    return out


def scores_oracle_error(universe, human_qrels, llm_grades):
    """|eps_p| under LINEAR gain.  eps_p = g_L(p) - g_h(p).

    Not a policy.  It is the denominator of the oracle-share numbers.
    """
    return np.array([abs(gain(llm_grades[q][p]) - gain(human_qrels[q].get(p, 0)))
                     for q, p in universe])


def scores_depth_k(universe, pool_depth, pool_nsys):
    return np.array([-pool_depth.get(k, 10 ** 9) + 1e-6 * pool_nsys.get(k, 0)
                     for k in universe])


def scores_retrieval_count(universe, pool_nsys):
    return np.array([float(pool_nsys.get(k, 0)) for k in universe])


def order_from_scores(universe, scores, rng):
    """Descending sort with a fixed random tie-break.

    Roughly 85-90 percent of pairs have C_pp = 0.  Without this, the tie
    group would be resolved by passage-id order, which is neither random
    nor meaningful.
    """
    jitter = rng.permutation(len(universe))
    idx = np.lexsort((jitter, -np.asarray(scores, dtype=np.float64)))
    return [universe[i] for i in idx]


# ---------------------------------------------------------------------------
#  PER-QUERY nDCG SCHEDULE FOR A FIXED ACQUISITION ORDER
# ---------------------------------------------------------------------------

def build_per_query_ndcg_table(ordering_qi, grades_start, human_qrels,
                               queries, system_names, sys_top10):
    """ndcg_qk[qi][k, si] = nDCG of system si on query qi after the first k
    pairs OF THAT QUERY (in acquisition order) have been bought.

    IDCG depends only on (query, step), so it is computed once and shared
    across systems.  DCG is updated incrementally: a correction moves a
    system only if the corrected passage sits in that system's top ten.
    """
    n_sys, n_q = len(system_names), len(queries)
    pairs_per_q = [[] for _ in range(n_q)]
    for qi, pid in ordering_qi:
        pairs_per_q[qi].append(pid)

    ndcg_qk = []
    for qi, qid in enumerate(queries):
        pids = pairs_per_q[qi]
        K = len(pids)
        g0 = dict(grades_start.get(qid, {}))
        hq = human_qrels.get(qid, {})

        cnt = Counter(g0.values())
        idcg = np.empty(K + 1)
        idcg[0] = idcg_from_counter(cnt)
        for k, pid in enumerate(pids, 1):
            old, new = g0[pid], hq[pid]
            cnt[old] -= 1
            cnt[new] += 1
            g0[pid] = new
            idcg[k] = idcg_from_counter(cnt)
        idcg_safe = np.where(idcg > 0, idcg, 1.0)

        step_of = {pid: k for k, pid in enumerate(pids, 1)}
        delta   = {pid: gain(hq[pid]) - gain(grades_start[qid][pid]) for pid in pids}
        base_g  = grades_start.get(qid, {})

        tab = np.zeros((K + 1, n_sys))
        for si in range(n_sys):
            top = sys_top10[si][qi]
            dcg = np.full(K + 1, sum(gain(base_g.get(p, 0)) * _DISCOUNT[r]
                                     for r, p in enumerate(top)))
            for r, pid in enumerate(top):
                k = step_of.get(pid)
                if k is not None:
                    dcg[k:] += _DISCOUNT[r] * delta[pid]
            tab[:, si] = np.where(idcg > 0, dcg / idcg_safe, 0.0)
        ndcg_qk.append(tab)

    return ndcg_qk, pairs_per_q


# ---------------------------------------------------------------------------
#  ADAPTIVE RUN-AWARE POLICIES:  leverage, product_raw, product_cal
# ---------------------------------------------------------------------------

def run_adaptive_run_aware(universe, pair_index, Wc, grades_start, human_qrels,
                           queries, system_names, sys_top10,
                           error_term=None, M=K_TOP, batch_size=100,
                           rng=None, verbose=False):
    """Buy in descending order of C_pp, optionally times E[eps^2].

    The target set is the M systems currently ranked highest under the
    grades available at that moment.  It starts from the ALL-LLM
    leaderboard, so no human label enters the selection.  Purchases change
    the estimated ranking, which changes the target set, which changes the
    leverage used for the next batch.  This is the adaptive rule described
    in Sec. 4.5, not one fixed sorting.
    """
    rng = rng or np.random.RandomState(0)
    qi_map = {q: i for i, q in enumerate(queries)}
    grades = {q: dict(v) for q, v in grades_start.items()}

    ndcg = ndcg_matrix_from_grades(grades, queries, system_names, sys_top10)
    n_q = len(queries)

    unbought = np.ones(len(universe), dtype=bool)
    jitter = rng.permutation(len(universe)).astype(np.float64)
    jitter /= (jitter.max() + 1.0)          # strictly inside [0, 1)
    ordering = []

    while unbought.any():
        top_idx = np.argsort(-ndcg.mean(axis=1), kind="stable")[:M]
        C = leverage_over(Wc, top_idx)
        score = C if error_term is None else C * error_term

        cand = np.flatnonzero(unbought)
        # Rank by score, ties broken by the fixed jitter.
        key = np.lexsort((jitter[cand], -score[cand]))
        take = cand[key[:min(batch_size, cand.size)]]

        dirty = set()
        for j in take:
            qid, pid = universe[j]
            grades[qid][pid] = human_qrels[qid][pid]
            unbought[j] = False
            ordering.append((qi_map[qid], pid))
            dirty.add(qi_map[qid])

        ndcg = ndcg_matrix_from_grades(grades, queries, system_names,
                                       sys_top10, out=ndcg, dirty=dirty)
        if verbose and len(ordering) % (20 * batch_size) < batch_size:
            print(".", end="", flush=True)

    return ordering


# ---------------------------------------------------------------------------
#  MOVE-TO-FRONT POOLING  (Cormack, Palmer and Clarke, SIGIR 1998, Sec. 6)
# ---------------------------------------------------------------------------

def run_mtf_policy(universe, human_qrels, queries, system_names, sys_top10,
                   runs, relevance_threshold=RELEVANCE_THRESHOLD):
    """Local MTF with numeric priorities.

    Cormack et al., Sec. 6: "The submissions themselves are prioritized,
    and the top-ranked document from the submission with the top priority
    is judged.  If it is judged relevant (or has been previously judged
    relevant because it appeared in some other submission) its priority is
    set to the maximum.  Otherwise, its priority is reduced."

    Three points the previous implementation missed.

      1. A miss REDUCES the priority by one.  It does not send the run to
         the back of the queue.  A run far ahead of the field stays ahead
         after a single miss.
      2. A hit sets the priority to the current maximum, so the run
         returns to the front.
      3. A document already judged still updates the priority, and costs
         no budget.  That is explicit in the paper.

    Local, not global.  Cormack's local variant "ensures that each topic
    receives a comparable number of judgements".  Global MTF would exhaust
    one topic before touching the next, which destroys a per-query
    leaderboard sweep at any budget below 100 percent.  Topics are visited
    round-robin, one judgement per visit.
    """
    universe_set = set(universe)
    qi_map = {q: i for i, q in enumerate(queries)}
    pids_of_q = defaultdict(set)
    for qid, pid in universe:
        pids_of_q[qid].add(pid)

    prio    = {qid: {sn: 0.0 for sn in system_names} for qid in queries}
    cursor  = {qid: {sn: 0 for sn in system_names} for qid in queries}
    judged  = {qid: {} for qid in queries}          # pid -> human grade
    remaining = defaultdict(int)
    for qid, pid in universe:
        remaining[qid] += 1

    ordering = []
    active = [q for q in queries if remaining[q] > 0]

    while active:
        still = []
        for qid in active:
            sr_prio = prio[qid]
            bought = False
            # Bounded: each inner pass advances at least one cursor.
            for _ in range(len(system_names) * 4):
                if remaining[qid] == 0:
                    break
                sn = max(system_names, key=lambda s: (sr_prio[s], s))
                pids = runs.get(sn, {}).get(qid, [])
                c = cursor[qid][sn]
                # Advance past pairs that are not part of the judged pool.
                while c < len(pids) and (qid, pids[c]) not in universe_set:
                    c += 1
                cursor[qid][sn] = c
                if c >= len(pids):
                    sr_prio[sn] = -1e18          # this run is exhausted here
                    continue
                pid = pids[c]
                cursor[qid][sn] = c + 1
                if pid in judged[qid]:
                    # Already judged elsewhere.  Update priority, spend nothing.
                    g = judged[qid][pid]
                    if g >= relevance_threshold:
                        sr_prio[sn] = max(sr_prio.values())
                    else:
                        sr_prio[sn] -= 1.0
                    continue
                g = human_qrels[qid][pid]
                judged[qid][pid] = g
                remaining[qid] -= 1
                ordering.append((qi_map[qid], pid))
                if g >= relevance_threshold:
                    sr_prio[sn] = max(sr_prio.values())
                else:
                    sr_prio[sn] -= 1.0
                bought = True
                break
            if remaining[qid] > 0:
                if bought or any(v > -1e17 for v in sr_prio.values()):
                    still.append(qid)
                else:
                    # No run reaches the leftover pairs.  Append them in a
                    # fixed order so the ordering covers the whole universe.
                    for pid in sorted(pids_of_q[qid] - set(judged[qid])):
                        judged[qid][pid] = human_qrels[qid][pid]
                        ordering.append((qi_map[qid], pid))
                        remaining[qid] -= 1
        active = still

    return ordering


# ---------------------------------------------------------------------------
#  MaxMean non-stationary  (Losada et al. 2016).  Optional.
# ---------------------------------------------------------------------------

def run_mm_ns_policy(universe, human_qrels, queries, system_names, runs,
                     window=MM_NS_WINDOW, epsilon=MM_NS_EPSILON, seed=SEED,
                     relevance_threshold=RELEVANCE_THRESHOLD):
    from collections import deque as _deque
    rng = np.random.RandomState(seed)
    universe_set = set(universe)
    qi_map = {q: i for i, q in enumerate(queries)}
    pids_of_q = defaultdict(set)
    for qid, pid in universe:
        pids_of_q[qid].add(pid)

    rw = {qid: {sn: _deque([0.5], maxlen=window) for sn in system_names}
          for qid in queries}
    cursor = {qid: {sn: 0 for sn in system_names} for qid in queries}
    judged = {qid: set() for qid in queries}
    remaining = defaultdict(int)
    for qid, pid in universe:
        remaining[qid] += 1

    ordering = []
    active = [q for q in queries if remaining[q] > 0]
    while active:
        still = []
        for qid in active:
            arms = [s for s in system_names if cursor[qid][s] is not None]
            if not arms or remaining[qid] == 0:
                continue
            if rng.random() < epsilon:
                sn = arms[rng.randint(len(arms))]
            else:
                sn = max(arms, key=lambda s: (sum(rw[qid][s]) / len(rw[qid][s]), s))
            pids = runs.get(sn, {}).get(qid, [])
            c = cursor[qid][sn]
            while c < len(pids) and ((qid, pids[c]) not in universe_set
                                     or pids[c] in judged[qid]):
                c += 1
            if c >= len(pids):
                cursor[qid][sn] = None
                still.append(qid)
                continue
            pid = pids[c]
            cursor[qid][sn] = c + 1
            judged[qid].add(pid)
            remaining[qid] -= 1
            ordering.append((qi_map[qid], pid))
            rw[qid][sn].append(1.0 if human_qrels[qid][pid] >= relevance_threshold
                               else 0.0)
            if remaining[qid] > 0:
                still.append(qid)
        active = [q for q in still if remaining[q] > 0
                  and any(cursor[q][s] is not None for s in system_names)]

    # Anything no run reaches, appended last.
    for qid in queries:
        for pid in sorted(pids_of_q[qid] - judged[qid]):
            ordering.append((qi_map[qid], pid))
    return ordering


# ---------------------------------------------------------------------------
#  LARA  (Takehi, Voorhees, Sakai and Soboroff, SIGIR 2025, Algorithm 1)
# ---------------------------------------------------------------------------

class LaraCalibrator:
    """Per-level calibration of the LLM probability vector.

    Their Sec. 3.3: "we propose to learn this calibration mapping for each
    relevance level j".  Their Sec. 4.2: "Logistic regression is used as
    the calibration model for LARA."

    One one-vs-rest logistic regression per relevance level, taking the
    four LLM probabilities as features, refit on all labels acquired so
    far, then normalised across levels.  Until at least two distinct human
    grades have been seen, the calibrator is the identity, which is
    Algorithm 1 line 3.
    """

    def __init__(self, n_cls=4, C=1.0, seed=SEED):
        self.n_cls = n_cls
        self.C = C
        self.seed = seed
        self.X, self.y = [], []
        self.models = None

    @property
    def active(self):
        return self.models is not None

    def add(self, x, label):
        self.X.append(np.asarray(x, dtype=np.float64))
        self.y.append(int(label))

    def refit(self):
        if len(self.y) < 2 or len(set(self.y)) < 2 or not SKLEARN_AVAILABLE:
            return
        X = np.vstack(self.X)
        y = np.asarray(self.y)
        models = {}
        for j in range(self.n_cls):
            t = (y == j).astype(int)
            if t.sum() == 0 or t.sum() == len(t):
                models[j] = float(t.mean())          # degenerate: constant rate
                continue
            m = LogisticRegression(C=self.C, max_iter=1000,
                                   random_state=self.seed)
            m.fit(X, t)
            models[j] = m
        self.models = models

    def predict_proba(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if not self.active:
            return X / np.clip(X.sum(axis=1, keepdims=True), 1e-12, None)
        P = np.zeros((X.shape[0], self.n_cls))
        for j, m in self.models.items():
            P[:, j] = m if isinstance(m, float) else m.predict_proba(X)[:, 1]
        P = np.clip(P, 1e-12, None)
        return P / P.sum(axis=1, keepdims=True)

    def margin(self, X):
        P = self.predict_proba(X)
        s = np.sort(P, axis=1)[:, ::-1]
        return s[:, 0] - s[:, 1]

    def argmax(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def run_lara_policy(universe, grades_start, human_qrels, softmax_probs,
                    queries, system_names, sys_top10, batch_size,
                    n_groups=1, seed=SEED, verbose=False):
    """LARA, Algorithm 1.

    Selection is by the CALIBRATED margin, not the raw one.  The remainder
    is imputed with argmax of the calibrated distribution.  That imputation
    is why LARA needs its own leaderboard schedule: unbought grades move
    every time the calibrator is refit, so a per-query correction table
    cannot represent it.

    `n_groups` is their LARA(n).  n = 1 is a single pool over all topics.
    n = N gives every topic its own budget share.  They report n = N as
    marginally best.  Default 1, which is what the thesis reproduces.

    Returns the acquisition ordering plus the checkpoint schedule of
    system-by-query nDCG under the mixed human / calibrated grades.
    """
    rng = np.random.RandomState(seed)
    qi_map = {q: i for i, q in enumerate(queries)}
    n_q, n_sys = len(queries), len(system_names)
    n_pairs = len(universe)

    X_all = np.vstack([softmax_probs.get(k, np.full(4, .25)) for k in universe])
    jitter = rng.permutation(n_pairs).astype(np.float64)
    jitter /= (jitter.max() + 1.0)

    if n_groups <= 1:
        groups = [np.arange(n_pairs)]
    else:
        gq = {q: i % n_groups for i, q in enumerate(queries)}
        groups = [np.array([i for i, (q, _) in enumerate(universe) if gq[q] == g])
                  for g in range(n_groups)]
        groups = [g for g in groups if g.size]

    cal = LaraCalibrator(seed=seed)
    grades = {q: dict(v) for q, v in grades_start.items()}
    unbought = np.ones(n_pairs, dtype=bool)

    ndcg = ndcg_matrix_from_grades(grades, queries, system_names, sys_top10)
    sched_ndcg = [ndcg.copy()]
    sched_cum  = [np.zeros(n_q, dtype=int)]
    cum = np.zeros(n_q, dtype=int)
    ordering = []

    for g in groups:
        while unbought[g].any():
            cand = g[unbought[g]]
            m = cal.margin(X_all[cand])
            key = np.lexsort((jitter[cand], m))       # smallest margin first
            take = cand[key[:min(batch_size, cand.size)]]

            for j in take:
                qid, pid = universe[j]
                y = human_qrels[qid][pid]
                grades[qid][pid] = y
                cal.add(X_all[j], y)
                unbought[j] = False
                ordering.append((qi_map[qid], pid))
                cum[qi_map[qid]] += 1
            cal.refit()

            # Impute every unbought grade with the calibrated argmax.
            mixed = {q: dict(v) for q, v in grades.items()}
            rest = np.flatnonzero(unbought)
            if rest.size and cal.active:
                pred = cal.argmax(X_all[rest])
                for t, j in enumerate(rest):
                    qid, pid = universe[j]
                    mixed[qid][pid] = int(pred[t])

            ndcg = ndcg_matrix_from_grades(mixed, queries, system_names, sys_top10)
            sched_ndcg.append(ndcg.copy())
            sched_cum.append(cum.copy())
            if verbose:
                print(".", end="", flush=True)

    return (ordering, np.array(sched_ndcg), np.array(sched_cum, dtype=int))


# ---------------------------------------------------------------------------
#  SWEEPS
# ---------------------------------------------------------------------------

def boot_correction_sweep(ordering_qi, ndcg_qk, query_counts, system_names,
                          gold_ranking, n_q, budget_fracs):
    """Sweep for a policy whose unbought pairs keep their starting grade."""
    n_sys = len(system_names)
    boot_qs = np.flatnonzero(query_counts > 0)
    restr = [(qi, pid) for qi, pid in ordering_qi if query_counts[qi] > 0]
    n_restr = max(len(restr), 1)

    out, k_per_q, ptr = [], np.zeros(n_q, dtype=int), 0
    for b in budget_fracs:
        target = int(round(b * n_restr))
        while ptr < target:
            k_per_q[restr[ptr][0]] += 1
            ptr += 1
        s = np.zeros(n_sys)
        for qi in boot_qs:
            k = min(k_per_q[qi], ndcg_qk[qi].shape[0] - 1)
            s += query_counts[qi] * ndcg_qk[qi][k]
        s /= n_q
        r = rank_systems(s, system_names)
        out.append({"budget": float(b),
                    "tau_all": compute_tau(gold_ranking, r),
                    "tau_at_20": compute_tau_at_k(gold_ranking, r, K_TOP),
                    "max_drop": compute_max_drop(gold_ranking, r)})
    return out


def boot_schedule_sweep(sched_ndcg, sched_cum, query_counts, system_names,
                        gold_ranking, n_q, budget_fracs):
    """Sweep for a policy that also rewrites its unbought grades, i.e. LARA."""
    boot_qs = np.flatnonzero(query_counts > 0)
    n_boot = int(sched_cum[-1][boot_qs].sum())
    if n_boot == 0:
        d = {"budget": 0.0, "tau_all": 1.0, "tau_at_20": 1.0, "max_drop": 0}
        return [dict(d, budget=float(b)) for b in budget_fracs]
    b_at = sched_cum[:, boot_qs].sum(axis=1) / n_boot

    out = []
    for b in budget_fracs:
        i = max(0, min(int(np.searchsorted(b_at, b, side="right")) - 1,
                       len(sched_ndcg) - 1))
        s = (sched_ndcg[i] * query_counts[None, :]).sum(axis=1) / n_q
        r = rank_systems(s, system_names)
        out.append({"budget": float(b),
                    "tau_all": compute_tau(gold_ranking, r),
                    "tau_at_20": compute_tau_at_k(gold_ranking, r, K_TOP),
                    "max_drop": compute_max_drop(gold_ranking, r)})
    return out


def ql_triage_sweep(ndcg_h, ndcg_l, order_qi, query_counts, system_names,
                    gold_ranking, n_q):
    """Query-level triage, budget expressed as the fraction of pairs bought.

    Emitted ASCENDING in budget.  The previous version emitted descending,
    which silently negated every area and broke np.interp.
    """
    n_sys = len(system_names)
    boot_qs = set(np.flatnonzero(query_counts > 0).tolist())
    restr = [qi for qi in order_qi if qi in boot_qs]
    n_restr = max(len(restr), 1)

    human_total = (ndcg_h * query_counts[None, :]).sum(axis=1)
    delta = ndcg_l - ndcg_h

    rows = []
    switch = np.zeros(n_sys)
    for k in range(n_restr + 1):
        s = (human_total + switch) / n_q
        r = rank_systems(s, system_names)
        rows.append({"budget": (n_restr - k) / n_restr,
                     "tau_all": compute_tau(gold_ranking, r),
                     "tau_at_20": compute_tau_at_k(gold_ranking, r, K_TOP),
                     "max_drop": compute_max_drop(gold_ranking, r)})
        if k < n_restr:
            qi = restr[k]
            switch += delta[:, qi] * query_counts[qi]
    rows.sort(key=lambda d: d["budget"])
    return rows


# ---------------------------------------------------------------------------
#  THRESHOLDS AND AREAS
# ---------------------------------------------------------------------------

def threshold_first_touch(curve, metric="tau_at_20", thr=TAU_THRESHOLD):
    """Smallest budget at which the metric FIRST reaches thr.  The table."""
    for r in sorted(curve, key=lambda d: d["budget"]):
        if r[metric] >= thr:
            return r["budget"], True
    return None, False


def threshold_sustained(curve, metric="tau_at_20", thr=TAU_THRESHOLD):
    """Smallest budget b such that the metric is at or above thr at EVERY
    budget from b to the end of the sweep.  The appendix."""
    c = sorted(curve, key=lambda d: d["budget"])
    v = [r[metric] for r in c]
    idx = len(v)
    for i in range(len(v) - 1, -1, -1):
        if v[i] >= thr:
            idx = i
        else:
            break
    return (c[idx]["budget"], True) if idx < len(v) else (None, False)


def area_between(pol_curve, rand_curve, metric="tau_at_20"):
    """Trapezoidal area between a policy curve and the random mean curve.

    Both curves are sorted ascending in budget first.  np.interp needs an
    increasing xp, and the trapezoid of a descending axis is negated.
    """
    p = sorted(pol_curve, key=lambda d: d["budget"])
    r = sorted(rand_curve, key=lambda d: d["budget"])
    pb = np.array([x["budget"] for x in p])
    pv = np.array([x[metric] for x in p])
    rb = np.array([x["budget"] for x in r])
    rv = np.array([x[metric] for x in r])
    span = pb[-1] - pb[0]
    if span <= 0:
        return 0.0
    return float(_TRAPZ(pv - np.interp(pb, rb, rv), pb))


# ---------------------------------------------------------------------------
#  ONE YEAR
# ---------------------------------------------------------------------------

def run_year(year, cfg, B, rng_master, verbose=True, include_mm_ns=False,
             lara_groups=1):
    t0 = time.time()
    rng = np.random.RandomState(rng_master.randint(0, 2 ** 31))

    human_qrels = load_qrels(cfg["qrels"])
    year_qs = set(human_qrels)
    llm_grades, softmax_probs = load_llm_data(cfg["scores"], year_qs)
    year_qs &= set(llm_grades)
    queries = sorted(year_qs)

    runs = load_system_runs(cfg["runs_dir"])
    if year in V2_YEARS:
        canonicalize_runs(runs, load_canonical_map())
    system_names = sorted(runs)
    n_q, n_sys = len(queries), len(system_names)

    universe = [(q, p) for q in queries for p in sorted(human_qrels[q])
                if p in llm_grades.get(q, {})]
    n_universe = len(universe)
    n_missing = sum(len(human_qrels[q]) for q in queries) - n_universe
    batch_size = max(1, int(round(BATCH_FRACTION * n_universe)))

    # Pairs the judge never scored keep their human grade in BOTH the gold
    # and the mixed leaderboards, so the two use the same passage set.
    grades_start = {q: {p: (llm_grades[q][p] if p in llm_grades.get(q, {})
                            else human_qrels[q][p])
                        for p in human_qrels[q]} for q in queries}

    if verbose:
        print(f"  {n_q} queries, {n_sys} systems, {n_universe} pairs, "
              f"batch {batch_size}, {n_missing} pair(s) without an LLM grade")

    sys_top10 = build_sys_top10(runs, queries, system_names)
    Wc, pair_index = build_pair_weight_matrix(runs, queries, system_names, universe)
    pool_depth, pool_nsys = build_pool_depth(runs, queries, system_names,
                                             set(universe))

    zero_lev = float((leverage_over(Wc, range(n_sys)) == 0).mean())
    if verbose:
        print(f"  {100 * zero_lev:.1f}% of pairs have C_pp = 0 over all runs "
              f"(the eligibility share)")

    ndcg_h = ndcg_matrix_from_grades(human_qrels, queries, system_names, sys_top10)
    ndcg_l = ndcg_matrix_from_grades(grades_start, queries, system_names, sys_top10)
    gold_full = rank_systems(ndcg_h.mean(axis=1), system_names)
    llm_full  = rank_systems(ndcg_l.mean(axis=1), system_names)
    point = {"tau_all": compute_tau(gold_full, llm_full),
             "tau_at_20": compute_tau_at_k(gold_full, llm_full, K_TOP),
             "max_drop": compute_max_drop(gold_full, llm_full)}
    if verbose:
        print(f"  all-LLM endpoint: tau_all={point['tau_all']:.3f}  "
              f"tau@20={point['tau_at_20']:.3f}  max_drop={point['max_drop']}")

    # ---- error terms --------------------------------------------------
    if verbose:
        print("  error estimators...")
    e_raw = expected_sq_error_raw(universe, llm_grades, softmax_probs)
    e_cal, cal_info = expected_sq_error_calibrated(
        universe, llm_grades, human_qrels, softmax_probs, verbose=verbose)

    # ---- orderings ----------------------------------------------------
    if verbose:
        print("  policy orderings...")
    qi_map = {q: i for i, q in enumerate(queries)}
    orderings, schedules = {}, {}

    def _static(name, s):
        t = time.time()
        o = order_from_scores(universe, s, np.random.RandomState(
            rng.randint(0, 2 ** 31)))
        orderings[name] = [(qi_map[q], p) for q, p in o]
        if verbose:
            print(f"    {name:<16s} {time.time() - t:5.1f}s")

    _static("random", rng.rand(n_universe))
    _static("naive", scores_naive_margin(universe, softmax_probs))
    _static("max_prob", scores_max_prob(universe, softmax_probs))
    _static("entropy", scores_entropy(universe, softmax_probs))
    _static("oracle", scores_oracle_error(universe, human_qrels, llm_grades))
    _static("depth_k", scores_depth_k(universe, pool_depth, pool_nsys))
    _static("retrieval_count", scores_retrieval_count(universe, pool_nsys))

    for name, term in [("leverage", None), ("product_raw", e_raw),
                       ("product_cal", e_cal)]:
        t = time.time()
        orderings[name] = run_adaptive_run_aware(
            universe, pair_index, Wc, grades_start, human_qrels, queries,
            system_names, sys_top10, error_term=term, M=K_TOP,
            batch_size=batch_size,
            rng=np.random.RandomState(rng.randint(0, 2 ** 31)))
        if verbose:
            print(f"    {name:<16s} {time.time() - t:5.1f}s")

    t = time.time()
    orderings["mtf"] = run_mtf_policy(universe, human_qrels, queries,
                                      system_names, sys_top10, runs)
    if verbose:
        print(f"    {'mtf':<16s} {time.time() - t:5.1f}s")

    if include_mm_ns:
        t = time.time()
        orderings["mm_ns"] = run_mm_ns_policy(universe, human_qrels, queries,
                                              system_names, runs,
                                              seed=SEED + year)
        if verbose:
            print(f"    {'mm_ns':<16s} {time.time() - t:5.1f}s")

    t = time.time()
    lara_ord, lara_sched, lara_cum = run_lara_policy(
        universe, grades_start, human_qrels, softmax_probs, queries,
        system_names, sys_top10, batch_size, n_groups=lara_groups, seed=SEED)
    orderings["lara"] = lara_ord
    schedules["lara"] = (lara_sched, lara_cum)
    if verbose:
        print(f"    {'lara':<16s} {time.time() - t:5.1f}s")

    for name, o in orderings.items():
        assert len(o) == n_universe, f"{name} ordering covers {len(o)}/{n_universe}"

    # ---- per-query nDCG schedules -------------------------------------
    if verbose:
        print("  per-query nDCG tables...")
    tables = {}
    for name, o in orderings.items():
        if name == "random":
            continue          # random gets its own bank of tables below
        tables[name] = build_per_query_ndcg_table(
            o, grades_start, human_qrels, queries, system_names, sys_top10)[0]

    rand_tables = [(orderings["random"], build_per_query_ndcg_table(
        orderings["random"], grades_start, human_qrels, queries, system_names,
        sys_top10)[0])]
    for _ in range(N_RAND_TABLES - 1):
        o = list(orderings["random"])
        rng.shuffle(o)
        rand_tables.append((o, build_per_query_ndcg_table(
            o, grades_start, human_qrels, queries, system_names, sys_top10)[0]))
    tables["random"] = rand_tables[0][1]

    precomp = time.time() - t0
    if verbose:
        print(f"  precompute {precomp:.1f}s")

    # ---- full-data curves ---------------------------------------------
    budgets = np.linspace(0, 1, N_BUDGET_STEPS)
    ones = np.ones(n_q)
    full_curves = {}

    # The random baseline is the MEAN over draws, per Sec. 4.4.  Averaging a
    # single draw would make the comparison depend on one shuffle.
    draws = [boot_correction_sweep(o, t, ones, system_names, gold_full, n_q,
                                   budgets) for o, t in rand_tables]
    rand_full = []
    for i, b in enumerate(budgets):
        rand_full.append({
            "budget": float(b),
            "tau_all": float(np.mean([d[i]["tau_all"] for d in draws])),
            "tau_at_20": float(np.mean([d[i]["tau_at_20"] for d in draws])),
            "max_drop": float(np.mean([d[i]["max_drop"] for d in draws])),
        })
    full_curves["random"] = rand_full

    if point["tau_at_20"] >= TAU_THRESHOLD:
        print(f"  WARNING: the all-LLM tau@20 for {year} is already at or "
              f"above {TAU_THRESHOLD}. Every policy will report a first-touch "
              f"budget of 0 and the table row is uninformative for this year.")
    for name in orderings:
        if name == "random":
            continue
        if name in schedules:
            full_curves[name] = boot_schedule_sweep(*schedules[name], ones,
                                                    system_names, gold_full,
                                                    n_q, budgets)
        else:
            full_curves[name] = boot_correction_sweep(
                orderings[name], tables[name], ones, system_names, gold_full,
                n_q, budgets)

    # ---- query-level orders -------------------------------------------
    dmg_path = DAMAGE_DIR / f"{year}_query_damage.csv"
    impact_order, reliab_order = [], []
    if dmg_path.exists():
        d = pd.read_csv(dmg_path, dtype={"query_id": str})
        col = "pearson" if "pearson" in d.columns else "spearman"
        m = {r["query_id"]: (r["damage_all"], r[col]) for _, r in d.iterrows()}
        ql = [q for q in queries if q in m]
        impact_order = [qi_map[q] for q in sorted(ql, key=lambda x: m[x][0])]
        reliab_order = [qi_map[q] for q in sorted(ql, key=lambda x: -m[x][1])]
    elif verbose:
        print(f"  no {dmg_path.name}; query-level oracles skipped")

    # ---- bootstrap -----------------------------------------------------
    if verbose:
        print(f"  bootstrap B={B} ", end="", flush=True)
    tb = time.time()

    pol_names = [p for p in TABLE_POLICIES if p in orderings] + \
                [p for p in EXTRA_POLICIES if p in orderings]
    stor = {p: defaultdict(list) for p in pol_names + ["all_llm"]}
    stor_ql = {p: defaultdict(list) for p in ["damage_oracle", "reliability_oracle"]}
    paired = defaultdict(list)
    bystep = {p: defaultdict(list) for p in pol_names}

    PAIRS = [("leverage", "lara"), ("leverage", "naive"), ("mtf", "lara"),
             ("mtf", "naive"), ("product_cal", "leverage"),
             ("product_raw", "leverage"), ("product_cal", "lara")]

    for it in range(B):
        idx = rng.choice(n_q, size=n_q, replace=True)
        cnt = np.bincount(idx, minlength=n_q).astype(float)

        gold_b = rank_systems(ndcg_h @ cnt / n_q, system_names)
        llm_b  = rank_systems(ndcg_l @ cnt / n_q, system_names)
        stor["all_llm"]["tau_all"].append(compute_tau(gold_b, llm_b))
        stor["all_llm"]["tau_at_20"].append(compute_tau_at_k(gold_b, llm_b, K_TOP))
        stor["all_llm"]["max_drop"].append(compute_max_drop(gold_b, llm_b))

        ro, rt = rand_tables[it % N_RAND_TABLES]
        rand_curve = boot_correction_sweep(ro, rt, cnt, system_names, gold_b,
                                           n_q, budgets)
        curves = {"random": rand_curve}
        for name in pol_names:
            if name == "random":
                continue
            if name in schedules:
                curves[name] = boot_schedule_sweep(*schedules[name], cnt,
                                                   system_names, gold_b, n_q,
                                                   budgets)
            else:
                curves[name] = boot_correction_sweep(orderings[name],
                                                     tables[name], cnt,
                                                     system_names, gold_b, n_q,
                                                     budgets)

        rv20 = np.array([r["tau_at_20"] for r in rand_curve])
        for name, c in curves.items():
            if name not in stor:
                continue
            ft, ok_ft = threshold_first_touch(c)
            su, ok_su = threshold_sustained(c)
            stor[name]["first_touch"].append(ft)
            stor[name]["first_touch_ok"].append(ok_ft)
            stor[name]["sustained"].append(su)
            stor[name]["sustained_ok"].append(ok_su)
            stor[name]["area_tau20"].append(area_between(c, rand_curve, "tau_at_20"))
            stor[name]["area_tau_all"].append(area_between(c, rand_curve, "tau_all"))
            v20 = np.array([r["tau_at_20"] for r in c])
            for ti, bf in enumerate(budgets):
                bystep[name][bf].append(v20[ti] - rv20[ti])

        for a, b_ in PAIRS:
            if a not in curves or b_ not in curves:
                continue
            fa, oa = threshold_first_touch(curves[a])
            fb, ob = threshold_first_touch(curves[b_])
            if oa and ob:
                paired[(a, b_, "first_touch_tau20")].append(fa - fb)
            paired[(a, b_, "area_tau20")].append(
                stor[a]["area_tau20"][-1] - stor[b_]["area_tau20"][-1])

        if impact_order:
            qlc = np.where(cnt > 0, cnt, 0.0)
            gold_ql = rank_systems(ndcg_h @ qlc / max(qlc.sum(), 1), system_names)
            rnd = list(range(n_q))
            rng.shuffle(rnd)
            rand_ql = ql_triage_sweep(ndcg_h, ndcg_l, rnd, qlc, system_names,
                                      gold_ql, n_q)
            for nm, order in [("damage_oracle", impact_order),
                              ("reliability_oracle", reliab_order)]:
                c = ql_triage_sweep(ndcg_h, ndcg_l, order, qlc, system_names,
                                    gold_ql, n_q)
                stor_ql[nm]["area_tau20"].append(area_between(c, rand_ql, "tau_at_20"))
                stor_ql[nm]["area_tau_all"].append(area_between(c, rand_ql, "tau_all"))

        if verbose and (it + 1) % 50 == 0:
            print(f"{it + 1} ", end="", flush=True)

    boot_time = time.time() - tb
    if verbose:
        print(f"\n  bootstrap {boot_time:.1f}s ({boot_time / B:.2f}s/iter)")

    return {
        "year": year, "n_q": n_q, "n_sys": n_sys, "n_universe": n_universe,
        "n_missing": n_missing, "batch_size": batch_size,
        "zero_leverage_share": zero_lev, "cal_info": cal_info,
        "point": point, "full_curves": full_curves,
        "stor": stor, "stor_ql": stor_ql, "paired": dict(paired),
        "bystep": {k: dict(v) for k, v in bystep.items()},
        "precomp_time": precomp, "boot_time": boot_time,
        "policies": pol_names,
    }


# ---------------------------------------------------------------------------
#  STRUCTURAL CORRELATIONS
# ---------------------------------------------------------------------------

def run_structural_bootstrap(B, rng_master, years):
    rows = []
    rng = np.random.RandomState(rng_master.randint(0, 2 ** 31))
    ddec = pd.read_csv(DDEC_CSV, dtype={"query_id": str}) if DDEC_CSV.exists() else None

    for year in years:
        if ddec is not None:
            y = ddec[ddec["year"] == year]
            for col, label in [("R_q_pearson", "agreement_pearson"),
                               ("R_q_spearman", "agreement_spearman")]:
                if col not in y.columns:
                    continue
                d = y[[col, "D_q"]].dropna()
                R, D = d[col].values, d["D_q"].values
                if len(R) < 3:
                    continue
                boot = [spearmanr(R[s], D[s])[0] for s in
                        (rng.choice(len(R), len(R), True) for _ in range(B))]
                rows.append({"year": year,
                             "quantity": f"spearman({label}, damage)",
                             "resample_unit": "query",
                             "point": float(spearmanr(R, D)[0]),
                             "lo_2p5": np.nanpercentile(boot, 2.5),
                             "median": np.nanpercentile(boot, 50),
                             "hi_97p5": np.nanpercentile(boot, 97.5)})

        sys_path = SPECTRAL_INT / str(year) / "systems.parquet"
        if not sys_path.exists():
            continue
        sdf = pd.read_parquet(sys_path)
        n_s = len(sdf)
        pairs = [("score_bias", "displacement_dcg", "score_bias_vs_score_shift"),
                 ("score_bias", "baseline_quality", "score_bias_vs_system_quality")]
        for xa, xb, label in pairs:
            if xa not in sdf.columns or xb not in sdf.columns:
                continue
            a, b = sdf[xa].values, sdf[xb].values
            samples = [rng.choice(n_s, n_s, True) for _ in range(B)]
            # Pearson is what the thesis quotes.  Spearman is reported beside
            # it; the previous version computed Spearman under a Pearson label.
            for fn, tag in [(pearsonr, "pearson"), (spearmanr, "spearman")]:
                boot = [fn(a[s], b[s])[0] for s in samples]
                rows.append({"year": year, "quantity": f"{label}_{tag}",
                             "resample_unit": "system",
                             "point": float(fn(a, b)[0]),
                             "lo_2p5": np.nanpercentile(boot, 2.5),
                             "median": np.nanpercentile(boot, 50),
                             "hi_97p5": np.nanpercentile(boot, 97.5)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  CI HELPERS
# ---------------------------------------------------------------------------

def pct(arr, p):
    a = [x for x in arr if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.percentile(a, p)) if a else np.nan


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=B_INITIAL)
    ap.add_argument("--years", type=int, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--include-mm-ns", action="store_true")
    ap.add_argument("--lara-groups", type=int, default=1,
                    help="LARA(n).  1 = single pool.  Use the number of "
                         "queries for LARA(n=N).")
    args = ap.parse_args()

    if not SKLEARN_AVAILABLE:
        raise SystemExit("scikit-learn is required for the LARA calibrator. "
                         "pip install scikit-learn")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng_master = np.random.RandomState(args.seed)
    years = args.years or sorted(YEARS_CFG)
    B = args.B

    print(f"=== t12_resampling  B={B}  years={years}  seed={args.seed} ===")
    print("    gain = LINEAR (g), per Eq. (4.1)\n")

    res_by_year, wall = {}, {}
    for i, year in enumerate(years):
        print(f"\n{'-' * 62}\n  {year}\n{'-' * 62}")
        r = run_year(year, YEARS_CFG[year], B, rng_master,
                     include_mm_ns=args.include_mm_ns,
                     lara_groups=args.lara_groups)
        res_by_year[year] = r
        wall[year] = r["boot_time"] + r["precomp_time"]
        if i == 0:
            mins = wall[year] / 60
            print(f"\n  *** TIMING: {mins:.1f} min for B={B}. "
                  f"Five years at B=1000 projects to "
                  f"{5 * (r['boot_time'] / B) * 1000 / 60 + 5 * r['precomp_time'] / 60:.0f} min ***")
            if mins < 30 and B < B_FULL and len(years) == 1:
                print(f"  Under 30 min. Re-run with --B {B_FULL}.")

    print(f"\n{'-' * 62}\n  structural correlations\n{'-' * 62}")
    struct = run_structural_bootstrap(B, rng_master, years)

    # ---- endpoints -----------------------------------------------------
    endpoints = []
    for y, r in res_by_year.items():
        for m in ("tau_all", "tau_at_20", "max_drop"):
            a = r["stor"]["all_llm"][m]
            endpoints.append({"year": y, "metric": m, "point": r["point"][m],
                              "lo_2p5": pct(a, 2.5), "median": pct(a, 50),
                              "hi_97p5": pct(a, 97.5)})
    pd.DataFrame(endpoints).to_csv(OUTPUT_DIR / "endpoints_ci.csv", index=False)

    # ---- thresholds ----------------------------------------------------
    thresholds, table_cells = [], {}
    for y, r in res_by_year.items():
        for pol in r["policies"]:
            row = {"year": y, "policy": pol, "display": DISPLAY_NAME.get(pol, pol),
                   "reads": READS.get(pol, "")}
            for kind in ("first_touch", "sustained"):
                a = r["stor"][pol][kind]
                ok = r["stor"][pol][kind + "_ok"]
                ft_full, ft_ok = (threshold_first_touch(r["full_curves"][pol])
                                  if kind == "first_touch"
                                  else threshold_sustained(r["full_curves"][pol]))
                span = pct(a, 97.5) - pct(a, 2.5)
                row.update({
                    f"{kind}_point": None if ft_full is None else 100 * ft_full,
                    f"{kind}_reached": bool(ft_ok),
                    f"{kind}_lo_2p5": 100 * pct(a, 2.5),
                    f"{kind}_median": 100 * pct(a, 50),
                    f"{kind}_hi_97p5": 100 * pct(a, 97.5),
                    f"{kind}_frac_never": 1 - (sum(ok) / max(len(ok), 1)),
                    f"{kind}_span_pp": 100 * span if not np.isnan(span) else np.nan,
                })
            thresholds.append(row)
            table_cells[(y, pol)] = row["first_touch_point"]
    thr_df = pd.DataFrame(thresholds)
    thr_df.to_csv(OUTPUT_DIR / "thresholds_ci.csv", index=False)

    # ---- the table -----------------------------------------------------
    tab_rows = []
    for pol in TABLE_POLICIES:
        row = {"policy": DISPLAY_NAME.get(pol, pol), "reads": READS.get(pol, "")}
        for y in years:
            v = table_cells.get((y, pol))
            row[str(y)] = "---" if v is None else f"{v:.0f}"
        tab_rows.append(row)
    tab_df = pd.DataFrame(tab_rows)
    tab_df.to_csv(OUTPUT_DIR / "budget_table.csv", index=False)

    tex = ["\\begin{table}[h]", "\\centering",
           "\\begin{tabular}{ll" + "c" * len(years) + "}", "\\hline",
           "Policy & Reads & " + " & ".join(str(y) for y in years) + " \\\\",
           "\\hline"]
    for r_ in tab_rows:
        tex.append(f"{r_['policy']} & {r_['reads']} & "
                   + " & ".join(r_[str(y)] for y in years) + " \\\\")
    tex += ["\\hline", "\\end{tabular}",
            "\\caption{Percentage of judged pairs verified before $\\tau@20$ "
            "first reaches 0.95.}",
            "\\label{tab:budget-first-touch}", "\\end{table}"]
    (OUTPUT_DIR / "budget_table.tex").write_text("\n".join(tex), encoding="utf-8")

    # ---- areas ---------------------------------------------------------
    areas = []
    for y, r in res_by_year.items():
        orc = {m: r["stor"]["oracle"].get(f"area_{m}", []) for m in
               ("tau20", "tau_all")}
        for pol in r["policies"]:
            for metric, key in (("tau_at_20", "area_tau20"),
                                ("tau_all", "area_tau_all")):
                a = r["stor"][pol][key]
                if not a:
                    continue
                o = orc["tau20" if metric == "tau_at_20" else "tau_all"]
                ratios = [a[i] / o[i] for i in range(min(len(a), len(o)))
                          if o[i] not in (0, None) and abs(o[i]) > 1e-12]
                areas.append({"year": y, "policy": pol, "metric": metric,
                              "area_vs_random": pct(a, 50),
                              "lo_2p5": pct(a, 2.5), "hi_97p5": pct(a, 97.5),
                              "ratio_to_oracle": pct(ratios, 50),
                              "ratio_lo": pct(ratios, 2.5),
                              "ratio_hi": pct(ratios, 97.5)})
        for nm in ("damage_oracle", "reliability_oracle"):
            for metric, key in (("tau_at_20", "area_tau20"),
                                ("tau_all", "area_tau_all")):
                a = r["stor_ql"][nm][key]
                if not a:
                    continue
                areas.append({"year": y, "policy": nm, "metric": metric,
                              "area_vs_random": pct(a, 50),
                              "lo_2p5": pct(a, 2.5), "hi_97p5": pct(a, 97.5),
                              "ratio_to_oracle": np.nan,
                              "ratio_lo": np.nan, "ratio_hi": np.nan})
    pd.DataFrame(areas).to_csv(OUTPUT_DIR / "area_ci.csv", index=False)

    # ---- paired differences --------------------------------------------
    pairs_out = []
    for y, r in res_by_year.items():
        for (a, b_, q), d in r["paired"].items():
            if not d:
                continue
            lo, hi = pct(d, 2.5), pct(d, 97.5)
            pairs_out.append({"year": y, "policy_a": a, "policy_b": b_,
                              "quantity": q, "mean_diff": float(np.mean(d)),
                              "lo_2p5": lo, "hi_97p5": hi,
                              "excludes_zero": bool(lo > 0 or hi < 0)})
    pd.DataFrame(pairs_out).to_csv(OUTPUT_DIR / "paired_differences.csv", index=False)

    # ---- difference by budget ------------------------------------------
    bystep_out = []
    for y, r in res_by_year.items():
        for pol, d in r["bystep"].items():
            for bf, arr in d.items():
                if not arr:
                    continue
                lo, hi = pct(arr, 2.5), pct(arr, 97.5)
                bystep_out.append({"year": y, "policy": pol, "metric": "tau_at_20",
                                   "budget_pct": round(100 * float(bf), 1),
                                   "mean_diff_vs_random": float(np.mean(arr)),
                                   "lo_2p5": lo, "hi_97p5": hi,
                                   "excludes_zero": bool(lo > 0 or hi < 0)})
    pd.DataFrame(bystep_out).to_csv(OUTPUT_DIR / "difference_by_budget.csv",
                                    index=False)

    # ---- full-data curves ----------------------------------------------
    cur = []
    for y, r in res_by_year.items():
        for pol, c in r["full_curves"].items():
            for row in c:
                cur.append({"year": y, "policy": pol,
                            "budget_pct": round(100 * row["budget"], 1),
                            "tau_all": row["tau_all"],
                            "tau_at_20": row["tau_at_20"],
                            "max_drop": row["max_drop"]})
    pd.DataFrame(cur).to_csv(OUTPUT_DIR / "curves_full.csv", index=False)

    struct.to_csv(OUTPUT_DIR / "structural_ci.csv", index=False)

    # ---- config --------------------------------------------------------
    wide = thr_df[thr_df["first_touch_span_pp"] > 30][
        ["year", "policy", "first_touch_span_pp"]]

    cfg_md = [
        "# bootstrap_config.md", "",
        f"- B: {B}", f"- seed: {args.seed}", f"- budget grid: {N_BUDGET_STEPS} steps",
        f"- threshold: tau@20 >= {TAU_THRESHOLD}", f"- K: {K_TOP}",
        f"- adaptive batch: {BATCH_FRACTION:.0%} of the pool",
        f"- random orderings cycled: {N_RAND_TABLES}",
        f"- LARA(n): {args.lara_groups}",
        "- gain: LINEAR, gain(g) = g, per Eq. (4.1)", "",
        "## Wall clock", ""]
    for y, t in wall.items():
        cfg_md.append(f"- {y}: {t / 60:.1f} min")
    cfg_md += ["", "## Per year", "",
               "| year | queries | systems | pairs | no LLM grade | C_pp = 0 | cal bins |",
               "|---|---|---|---|---|---|---|"]
    for y, r in res_by_year.items():
        cfg_md.append(f"| {y} | {r['n_q']} | {r['n_sys']} | {r['n_universe']} | "
                      f"{r['n_missing']} | {100 * r['zero_leverage_share']:.1f}% | "
                      f"{r['cal_info']['n_bins']} |")

    cfg_md += ["", "## Design decision (a): duplicated queries", "",
               "A query drawn k times contributes k copies of its nDCG to the",
               "mean.  Its pairs appear once in the correction universe, so the",
               "budget denominator counts unique pairs.  The mean nDCG",
               "denominator stays at the full query count.", "",
               "## Design decision (b): policy ordering", "",
               "RESTRICT throughout.  The acquisition ordering is computed once",
               "on the full data and restricted to the sampled pairs.", "",
               "Exact for random, naive, max_prob, entropy, oracle, depth_k and",
               "retrieval_count, whose scores are per-pair quantities.", "",
               "NOT exact, and therefore UNDERSTATING uncertainty, for:", "",
               "- leverage, product_raw, product_cal: the target top-20 set is",
               "  recomputed from the current leaderboard, which spans queries.",
               "- mtf, mm_ns: the run priority queue carries cross-query state.",
               "- lara: the calibrator is fitted on labels from every query and",
               "  its imputation rewrites every unbought grade.", "",
               "## Design decision (c): B", "",
               f"Started at {B_INITIAL}.  Raise to {B_FULL} once year one fits in",
               "thirty minutes.", ""]

    if len(wide):
        cfg_md += ["## FLAG: first-touch intervals wider than 30 points", ""]
        for _, w in wide.iterrows():
            cfg_md.append(f"- {int(w['year'])}, {w['policy']}: "
                          f"{w['first_touch_span_pp']:.0f} points")
        cfg_md += ["", "These cells are not stable enough to quote as integers.", ""]

    (OUTPUT_DIR / "bootstrap_config.md").write_text("\n".join(cfg_md), encoding="utf-8")

    # ---- report --------------------------------------------------------
    rep = ["# REPORT.md, t12_resampling", "",
           f"B = {B}, seed = {args.seed}, linear gain, tau@20 threshold "
           f"{TAU_THRESHOLD}.", "",
           "## The table, first touch, percent of judged pairs", "",
           "| Policy | Reads | " + " | ".join(str(y) for y in years) + " |",
           "|" + "---|" * (2 + len(years))]
    for r_ in tab_rows:
        rep.append(f"| {r_['policy']} | {r_['reads']} | "
                   + " | ".join(r_[str(y)] for y in years) + " |")

    rep += ["", "## First touch against sustained, with intervals", "",
            "| year | policy | first touch | 95% CI | sustained | never reached |",
            "|---|---|---|---|---|---|"]
    for _, r_ in thr_df.iterrows():
        ft = "---" if r_["first_touch_point"] is None else f"{r_['first_touch_point']:.0f}"
        su = "---" if r_["sustained_point"] is None else f"{r_['sustained_point']:.0f}"
        rep.append(f"| {int(r_['year'])} | {r_['policy']} | {ft} | "
                   f"[{r_['first_touch_lo_2p5']:.0f}, {r_['first_touch_hi_97p5']:.0f}] | "
                   f"{su} | {r_['first_touch_frac_never']:.2f} |")

    rep += ["", "## All-LLM endpoints", "",
            "| year | metric | point | 95% CI |", "|---|---|---|---|"]
    for e in endpoints:
        rep.append(f"| {e['year']} | {e['metric']} | {e['point']:.3f} | "
                   f"[{e['lo_2p5']:.3f}, {e['hi_97p5']:.3f}] |")

    rep += ["", "## Share of the oracle recovered, tau@20", "",
            "Sections 4.5 and 5.2 say judge confidence recovers roughly 40 to 50",
            "percent of the oracle's area.  The measured ratios:", "",
            "| year | policy | area | ratio to oracle | 95% CI |",
            "|---|---|---|---|---|"]
    for a in areas:
        if a["metric"] != "tau_at_20" or a["policy"] not in (
                "naive", "lara", "max_prob", "entropy", "leverage",
                "product_raw", "product_cal", "mtf"):
            continue
        rep.append(f"| {a['year']} | {a['policy']} | {a['area_vs_random']:.4f} | "
                   f"{a['ratio_to_oracle']:.3f} | "
                   f"[{a['ratio_lo']:.3f}, {a['ratio_hi']:.3f}] |")

    rep += ["", "## Paired differences, computed within each sample", "",
            "| year | A | B | quantity | mean | 95% CI | excludes zero |",
            "|---|---|---|---|---|---|---|"]
    for p in pairs_out:
        rep.append(f"| {p['year']} | {p['policy_a']} | {p['policy_b']} | "
                   f"{p['quantity']} | {p['mean_diff']:.4f} | "
                   f"[{p['lo_2p5']:.4f}, {p['hi_97p5']:.4f}] | "
                   f"{p['excludes_zero']} |")

    if len(struct):
        rep += ["", "## Structural correlations", "",
                "| year | quantity | unit | point | 95% CI |",
                "|---|---|---|---|---|"]
        for _, s in struct.iterrows():
            rep.append(f"| {int(s['year'])} | {s['quantity']} | "
                       f"{s['resample_unit']} | {s['point']:.3f} | "
                       f"[{s['lo_2p5']:.3f}, {s['hi_97p5']:.3f}] |")

    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(rep), encoding="utf-8")

    print(f"\nDone.  Outputs in {OUTPUT_DIR}")
    if len(wide):
        print(f"WARNING: {len(wide)} (year, policy) cells have a first-touch "
              f"interval wider than 30 points.  See bootstrap_config.md.")


if __name__ == "__main__":
    main()