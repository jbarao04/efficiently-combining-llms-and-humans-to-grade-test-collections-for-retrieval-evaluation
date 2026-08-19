# Top-M targeted human correction: iterative passage-level correction with seven policies

import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import json
import math
import os
from collections import defaultdict, deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from contextlib import nullcontext

from v2_id_mapping import V2_YEARS, load_canonical_map, canonicalize_runs

# ── Configuration ────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent

YEARS = {
    2019: {
        "qrels": BASE_DIR / "data_prep" / "data" / "trec-dl" / "2019" / "qrels.txt",
        "scores": BASE_DIR / "results" / "scoring" / "normal_scores" / "scores_v1.jsonl",
        "runs_dir": BASE_DIR / "data" / "system_runs" / "2019",
    },
    2020: {
        "qrels": BASE_DIR / "data_prep" / "data" / "trec-dl" / "2020" / "qrels.txt",
        "scores": BASE_DIR / "results" / "scoring" / "normal_scores" / "scores_v1.jsonl",
        "runs_dir": BASE_DIR / "data" / "system_runs" / "2020",
    },
    2021: {
        "qrels": BASE_DIR / "data_prep" / "data" / "trec-dl-v2" / "2021" / "qrels_dedup.txt",
        "scores": BASE_DIR / "results" / "scoring" / "normal_scores" / "scores_v2.jsonl",
        "runs_dir": BASE_DIR / "data" / "system_runs" / "2021",
    },
    2022: {
        "qrels": BASE_DIR / "data_prep" / "data" / "trec-dl-v2" / "2022" / "qrels_dedup.txt",
        "scores": BASE_DIR / "results" / "scoring" / "normal_scores" / "scores_v2.jsonl",
        "runs_dir": BASE_DIR / "data" / "system_runs" / "2022",
    },
    2023: {
        "qrels": BASE_DIR / "data_prep" / "data" / "trec-dl-v2" / "2023" / "qrels_dedup.txt",
        "scores": BASE_DIR / "results" / "scoring" / "normal_scores" / "scores_v2.jsonl",
        "runs_dir": BASE_DIR / "data" / "system_runs" / "2023",
    },
}

CONFIDENCE_CSV = BASE_DIR / "results" / "spectral" / "confidence_passage_linear.csv"
OUTPUT_DIR = BASE_DIR / "results" / "topM_correction"

SEED = 42
BATCH_FRACTION = 0.01  # 1% of universe per round
RELEVANCE_THRESHOLD = 2  # binarize at grade >= 2 for MTF/MM-NS reward


# ── Data Loading ─────────────────────────────────────────────────────────

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


def load_confidence(year):
    df = pd.read_csv(CONFIDENCE_CSV, dtype={"query_id": str, "passage_id": str})
    df = df[df["year"] == year]
    conf = {}
    for _, row in df.iterrows():
        conf[(row["query_id"], row["passage_id"])] = row["margin"]
    return conf


def load_softmax_probs(jsonl_path, year_queries):
    """Load full softmax prob_0..prob_3 from scores JSONL.
    Returns: {(qid, pid): np.array([p0, p1, p2, p3])}"""
    probs = {}
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            qid = str(rec["query_id"])
            if qid not in year_queries:
                continue
            pid = str(rec["passage_id"])
            p = rec["probs"]
            # probs dict has keys "0","1","2","3"
            n_classes = len(p)
            vec = np.array([float(p.get(str(k), 0.0)) for k in range(n_classes)],
                           dtype=np.float64)
            probs[(qid, pid)] = vec
    return probs


# ── nDCG@10 with mutable grades ─────────────────────────────────────────

def ndcg_at_k(ranked_pids, qrels_for_query, k=10):
    if not qrels_for_query:
        return 0.0
    dcg = sum(qrels_for_query.get(pid, 0) / math.log2(i + 2)
              for i, pid in enumerate(ranked_pids[:k]))
    ideal_gains = sorted(qrels_for_query.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0


# ── Ranking & Tau (identical to run_passage_triage.py) ───────────────────

def rank_systems(mean_scores, system_names):
    paired = sorted(zip(system_names, mean_scores), key=lambda x: (-x[1], x[0]))
    return [name for name, _ in paired]


def compute_tau(ranking_a, ranking_b):
    rank_a = {name: i for i, name in enumerate(ranking_a)}
    pos_b = [rank_a[name] for name in ranking_b]
    tau, _ = kendalltau(list(range(len(ranking_a))), pos_b)
    return tau if not np.isnan(tau) else 1.0


def compute_tau_at_k(gold_ranking, mixed_ranking, K):
    if K > len(gold_ranking):
        K = len(gold_ranking)
    top_k = set(gold_ranking[:K])
    gold_order = [s for s in gold_ranking if s in top_k]
    mixed_order = [s for s in mixed_ranking if s in top_k]
    if len(mixed_order) < 2:
        return 1.0
    gold_rank = {name: i for i, name in enumerate(gold_order)}
    mixed_positions = [gold_rank[name] for name in mixed_order]
    tau, _ = kendalltau(list(range(len(gold_order))), mixed_positions)
    return tau if not np.isnan(tau) else 1.0


# ── Max Drop metric ─────────────────────────────────────────────────────

def compute_max_drop(gold_ranking, eval_ranking):
    """Largest single-system rank fall: max over s of (eval_rank - gold_rank).
    Positive means the system is ranked lower (worse) than in the gold.
    Lower max_drop is better."""
    gold_rank = {name: i for i, name in enumerate(gold_ranking)}
    eval_rank = {name: i for i, name in enumerate(eval_ranking)}
    max_d = 0
    for name in gold_ranking:
        drop = eval_rank[name] - gold_rank[name]
        if drop > max_d:
            max_d = drop
    return max_d


# ── Precompute run weights ───────────────────────────────────────────────

def build_run_weights(runs, queries, system_names):
    """
    For each (system, query, passage), compute w = 1/log2(rank+1) if rank<=10.
    Returns: {(qid, pid): {sys_name: weight}}
    Also returns: sys_top10[sys_name][qid] = [pid list] (ordered)
    """
    pair_weights = defaultdict(dict)  # (qid, pid) -> {sys: w}
    sys_top10 = {}
    for sys_name in system_names:
        sys_run = runs.get(sys_name, {})
        top10 = {}
        for qid in queries:
            pids = sys_run.get(qid, [])[:10]
            top10[qid] = pids
            for rank_idx, pid in enumerate(pids):
                rank = rank_idx + 1  # 1-based
                w = 1.0 / math.log2(rank + 1)
                pair_weights[(qid, pid)][sys_name] = w
        sys_top10[sys_name] = top10
    return dict(pair_weights), sys_top10


def build_pool_depth(runs, queries, system_names, universe_set):
    """For each pooled (qid, pid), compute:
      - pool_depth: min rank across all systems (shallowest retrieval)
      - n_systems:  number of systems that retrieve the pair (in top 1000)
    Only considers pairs in universe_set."""
    depth = {}   # (qid, pid) -> min rank (1-based)
    n_sys = {}   # (qid, pid) -> count
    for sys_name in system_names:
        sys_run = runs.get(sys_name, {})
        for qid in queries:
            pids = sys_run.get(qid, [])
            for rank_idx, pid in enumerate(pids):
                k = (qid, pid)
                if k not in universe_set:
                    continue
                r = rank_idx + 1
                if k not in depth or r < depth[k]:
                    depth[k] = r
                n_sys[k] = n_sys.get(k, 0) + 1
    return depth, n_sys


# ── Core: Leaderboard from current grades ────────────────────────────────

class Leaderboard:
    """Maintains system nDCG scores with incremental updates."""

    def __init__(self, runs, current_grades, queries, system_names, sys_top10):
        self.runs = runs
        self.grades = current_grades  # {qid: {pid: grade}} — mutable reference
        self.queries = queries
        self.system_names = system_names
        self.sys_top10 = sys_top10
        self.n_q = len(queries)
        self.n_sys = len(system_names)
        self.sys_idx = {s: i for i, s in enumerate(system_names)}

        # Compute full nDCG matrix: shape (n_sys, n_q)
        self.ndcg_mat = np.zeros((self.n_sys, self.n_q))
        for si, sys_name in enumerate(system_names):
            for qi, qid in enumerate(queries):
                self.ndcg_mat[si, qi] = ndcg_at_k(
                    self.sys_top10[sys_name][qid],
                    self.grades.get(qid, {}))

        self.sys_means = self.ndcg_mat.mean(axis=1)
        self._ranking = None

    def get_ranking(self):
        if self._ranking is None:
            self._ranking = rank_systems(self.sys_means, self.system_names)
        return self._ranking

    def update_pairs(self, corrected):
        """
        corrected: list of (qid, pid, old_grade, new_grade)
        Recompute only affected (system, query) nDCG cells.
        """
        affected_queries = set()
        for qid, pid, old_g, new_g in corrected:
            if old_g != new_g:
                affected_queries.add(qid)

        q_idx_map = {qid: qi for qi, qid in enumerate(self.queries)}
        for qid in affected_queries:
            qi = q_idx_map[qid]
            q_grades = self.grades.get(qid, {})
            for si, sys_name in enumerate(self.system_names):
                self.ndcg_mat[si, qi] = ndcg_at_k(
                    self.sys_top10[sys_name][qid], q_grades)

        self.sys_means = self.ndcg_mat.mean(axis=1)
        self._ranking = None

    def top_M(self, M):
        ranking = self.get_ranking()
        return set(ranking[:M])


# ── Record metrics helper ────────────────────────────────────────────────

def _record_metrics(board, gold_ranking, judged_count, n_universe):
    ranking = board.get_ranking()
    return {
        "budget": judged_count / n_universe,
        "tau_all": compute_tau(gold_ranking, ranking),
        "tau_at_20": compute_tau_at_k(gold_ranking, ranking, 20),
        "max_drop": compute_max_drop(gold_ranking, ranking),
    }


# ── Policy scoring functions ─────────────────────────────────────────────
# Feasible policies: NEVER read human grades (h). Only see runs, current
# grades (g), and confidence (margin). The only variation is acquisition order.

def score_naive(pair_keys, pair_weights, confidence, target_systems, rng):
    """LARA uncertainty baseline: 1 - margin. Smallest margin first."""
    return {k: 1.0 - confidence.get(k, 1.0) for k in pair_keys}


def score_popularity(pair_keys, pair_weights, confidence, target_systems, rng):
    scores = {}
    for k in pair_keys:
        ws = pair_weights.get(k, {})
        scores[k] = sum(ws.values())  # sum over ALL systems
    return scores


def score_leverage(pair_keys, pair_weights, confidence, target_systems, rng):
    scores = {}
    t_list = list(target_systems)
    for k in pair_keys:
        ws = pair_weights.get(k, {})
        vals = [ws.get(s, 0.0) for s in t_list]
        scores[k] = float(np.var(vals)) if vals else 0.0
    return scores


def score_depth_k(pool_depth, pool_nsys):
    """Factory: static ordering by shallowest retrieval depth, then by
    number of systems retrieving (more first). Score is negated so that
    highest score = shallowest depth."""
    def _score(pair_keys, pair_weights, confidence, target_systems, rng):
        scores = {}
        for k in pair_keys:
            d = pool_depth.get(k, 9999)
            n = pool_nsys.get(k, 0)
            # Negate depth (shallower = higher score), add n_sys as tiebreak
            scores[k] = -d + n * 1e-6
        return scores
    return _score


# ── LARA Calibrator (ported from RikiyaT/LARA graded_algo.py) ────────────
# LogisticOrdinalNet: ordinal logistic regression with n_classes-1 cumulative
# logits, two epochs per batch, 0.1 * MSE transition loss.
#
# LogisticOrdinalPredictor: wraps the net. After warmup (n_samples >= warmup),
# the calibrator is fully trusted (no blend with raw softmax). Before warmup,
# falls back to raw LLM softmax. This avoids the slow-blend inertness where
# confidence_rate=0.0001 suppresses the calibrator across the entire budget
# range that matters (blend weight = 0.05 at 5% of a 9k pool).

CALIBRATOR_WARMUP = 50  # trust calibrator fully once this many labels are collected

class LogisticOrdinalNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.logits = nn.Linear(n_classes, n_classes - 1)

    def forward(self, x):
        return self.logits(x)


class LogisticOrdinalPredictor:
    def __init__(self, n_classes, learning_rate=0.01,
                 batch_size=32, warmup=CALIBRATOR_WARMUP,
                 random_state=None, device=None):
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.warmup = warmup
        self.device = device if device is not None else torch.device('cpu')
        self.model = LogisticOrdinalNet(n_classes).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.X_train = []
        self.y_train = []
        self.n_samples = 0
        self._trained = False  # has the model been trained at least once?
        if random_state is not None:
            torch.manual_seed(random_state)

    @property
    def is_active(self):
        """True once the calibrator has enough labels to be trusted."""
        return self._trained and self.n_samples >= self.warmup

    def _to_ordinal(self, y):
        y_ord = torch.zeros((len(y), self.n_classes - 1), device=self.device)
        for i, yi in enumerate(y):
            y_ord[i, :yi] = 1
        return y_ord

    def _from_ordinal(self, cumulative_probs):
        probs = torch.zeros((len(cumulative_probs), self.n_classes),
                            device=self.device)
        probs[:, 0] = 1 - cumulative_probs[:, 0]
        for k in range(1, self.n_classes - 1):
            probs[:, k] = cumulative_probs[:, k-1] - cumulative_probs[:, k]
        probs[:, -1] = cumulative_probs[:, -1]
        return F.softmax(probs, dim=1)

    def fit_batch(self, X_batch, y_batch):
        X_batch = np.asarray(X_batch, dtype=np.float64)
        y_batch = np.asarray(y_batch, dtype=np.int64)
        X_tensor = torch.FloatTensor(X_batch).to(self.device)
        y_tensor = torch.LongTensor(y_batch).to(self.device)
        self.model.train()
        for _ in range(2):  # two epochs per batch, per LARA
            self.optimizer.zero_grad()
            logits = self.model(X_tensor)
            y_ord = self._to_ordinal(y_tensor)
            loss = F.binary_cross_entropy_with_logits(logits, y_ord)
            cumulative_probs = torch.sigmoid(logits)
            pred_probs = self._from_ordinal(cumulative_probs)
            transition_loss = 0.1 * F.mse_loss(pred_probs, X_tensor)
            total_loss = loss + transition_loss
            total_loss.backward()
            self.optimizer.step()
        self._trained = True

    def refit_all(self, n_epochs=5):
        """Refit on ALL collected training data. Called at each checkpoint
        so the calibrator sees the full label set, not just recent batches."""
        if self.n_samples < self.warmup:
            return
        X_all = np.vstack(self.X_train)
        y_all = np.array(self.y_train, dtype=np.int64)
        X_tensor = torch.FloatTensor(X_all).to(self.device)
        y_tensor = torch.LongTensor(y_all).to(self.device)
        self.model.train()
        for _ in range(n_epochs):
            self.optimizer.zero_grad()
            logits = self.model(X_tensor)
            y_ord = self._to_ordinal(y_tensor)
            loss = F.binary_cross_entropy_with_logits(logits, y_ord)
            cumulative_probs = torch.sigmoid(logits)
            pred_probs = self._from_ordinal(cumulative_probs)
            transition_loss = 0.1 * F.mse_loss(pred_probs, X_tensor)
            total_loss = loss + transition_loss
            total_loss.backward()
            self.optimizer.step()
        self._trained = True

    def _learned_proba(self, X):
        """Pure model output, no blending."""
        X = np.asarray(X, dtype=np.float64)
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            cumulative_probs = torch.sigmoid(logits)
            return self._from_ordinal(cumulative_probs).cpu().numpy()

    def predict_proba(self, X):
        """Return calibrated probabilities. Before warmup: raw input.
        After warmup: fully trust the learned calibrator."""
        X = np.asarray(X, dtype=np.float64)
        if not self.is_active:
            return X.copy()
        return self._learned_proba(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def add_training_sample(self, X, y):
        self.X_train.append(X)
        self.y_train.append(int(y))
        self.n_samples += 1
        if self.n_samples % self.batch_size == 0:
            X_batch = np.vstack(self.X_train[-self.batch_size:])
            y_batch = np.array(self.y_train[-self.batch_size:], dtype=np.int64)
            self.fit_batch(X_batch, y_batch)

    def calibrated_margin(self, X):
        """Return calibrated margin: top1_prob - top2_prob from calibrated probs."""
        probs = self.predict_proba(X)
        sorted_p = np.sort(probs, axis=1)[:, ::-1]
        return sorted_p[:, 0] - sorted_p[:, 1]


# ── 2x2 Calibration Loop ────────────────────────────────────────────────
# selection x remainder: {confidence, leverage} x {raw, calibrated}

def _build_calibrated_grades(calibrator, softmax_probs, universe, judged,
                             human_grades, llm_grades):
    """Build grade dict: corrected pairs get human grade, uncorrected get
    calibrator argmax. This is the 'calibrated remainder' treatment."""
    grades = {}
    # Collect unjudged pairs for batch prediction
    unjudged_keys = []
    unjudged_probs = []
    for qid, pid in universe:
        if qid not in grades:
            grades[qid] = {}
        if (qid, pid) in judged:
            grades[qid][pid] = human_grades[qid][pid]
        else:
            unjudged_keys.append((qid, pid))
            unjudged_probs.append(softmax_probs.get((qid, pid),
                                  np.array([0.25, 0.25, 0.25, 0.25])))

    if unjudged_probs:
        X = np.vstack(unjudged_probs)
        predicted = calibrator.predict(X)
        for i, (qid, pid) in enumerate(unjudged_keys):
            grades[qid][pid] = int(predicted[i])

    return grades


def _run_lara(universe, human_qrels, llm_qrels, init_grades,
              softmax_probs, runs, queries, system_names, sys_top10,
              pair_weights, confidence, gold_ranking, M, adaptive,
              batch_size, seed=42):
    """Run LARA proper: calibrated margin selects, calibrator cleans remainder.

    Returns curve for 'confidence_calibrated' (LARA).
    The calibrator is topic-agnostic (one model over all queries), trained
    online on corrected pairs. After warmup (CALIBRATOR_WARMUP samples) the
    calibrator is fully trusted. Refit on ALL collected labels each checkpoint.
    """
    n_universe = len(universe)
    n_classes = 4  # grades 0-3

    raw_llm_argmax = {}
    for qid, pid in universe:
        sx = softmax_probs.get((qid, pid), np.array([0.25, 0.25, 0.25, 0.25]))
        raw_llm_argmax[(qid, pid)] = int(np.argmax(sx))

    calibrator = LogisticOrdinalPredictor(
        n_classes=n_classes, learning_rate=0.01,
        batch_size=32, warmup=CALIBRATOR_WARMUP,
        random_state=seed, device=torch.device('cpu'))

    grades_raw = {}
    for qid in init_grades:
        grades_raw[qid] = dict(init_grades[qid])

    board_raw = Leaderboard(runs, grades_raw, queries, system_names, sys_top10)
    judged = set()
    unjudged = set(universe)

    curve_cal = [_record_metrics(board_raw, gold_ranking, 0, n_universe)]
    n_checkpoints_active = 0

    while len(judged) < n_universe:
        this_batch = min(batch_size, len(unjudged))
        if this_batch == 0:
            break

        # Selection: LARA calibrated margin (smallest margin first)
        unjudged_list = list(unjudged)
        X_unjudged = np.vstack([
            softmax_probs.get(k, np.array([0.25, 0.25, 0.25, 0.25]))
            for k in unjudged_list])
        margins = calibrator.calibrated_margin(X_unjudged)
        scores = {k: -margins[i] for i, k in enumerate(unjudged_list)}

        unjudged_list.sort(key=lambda k: -scores.get(k, 0.0))
        batch = unjudged_list[:this_batch]

        corrections = []
        for qid, pid in batch:
            old_g = grades_raw[qid].get(pid, 0)
            new_g = human_qrels[qid][pid]
            grades_raw[qid][pid] = new_g
            corrections.append((qid, pid, old_g, new_g))
            judged.add((qid, pid))
            unjudged.discard((qid, pid))
            sx = softmax_probs.get((qid, pid),
                                   np.array([0.25, 0.25, 0.25, 0.25]))
            calibrator.add_training_sample(sx.reshape(1, -1), new_g)

        board_raw.update_pairs(corrections)
        calibrator.refit_all(n_epochs=5)

        # Remainder: calibrated grades for unjudged pairs
        if calibrator.is_active:
            n_checkpoints_active += 1
            cal_grades = _build_calibrated_grades(
                calibrator, softmax_probs, universe, judged,
                human_qrels, init_grades)
            board_cal = Leaderboard(runs, cal_grades, queries,
                                    system_names, sys_top10)
            curve_cal.append(_record_metrics(
                board_cal, gold_ranking, len(judged), n_universe))
        else:
            # Before warmup: use raw LLM grades
            curve_cal.append(_record_metrics(
                board_raw, gold_ranking, len(judged), n_universe))

    # Guard: verify calibrator was active and made a difference
    assert calibrator.is_active, (
        f"LARA calibrator never activated "
        f"(n_samples={calibrator.n_samples}, warmup={calibrator.warmup})")
    # Count relabeled at midpoint for diagnostics
    n_unjudged_final = n_universe - len(judged)
    print(f"    LARA: calibrator active for {n_checkpoints_active} checkpoints, "
          f"{n_unjudged_final} unjudged remain")

    return curve_cal


# ── Oracle policy ────────────────────────────────────────────────────────

def run_oracle_tau(universe, human_qrels, current_grades, leaderboard,
                   gold_ranking, batch_size, K=20):
    """
    Greedy oracle: for each unjudged pair, simulate correction and pick
    the one that maximally raises tau@K. Expensive — do one at a time.
    Returns the single best pair to correct.
    """
    best_pair = None
    best_tau = -2.0

    for qid, pid in universe:
        h = human_qrels[qid][pid]
        g = current_grades[qid].get(pid, 0)
        if h == g:
            continue

        current_grades[qid][pid] = h
        leaderboard.update_pairs([(qid, pid, g, h)])
        new_ranking = leaderboard.get_ranking()
        new_tau = compute_tau_at_k(gold_ranking, new_ranking, K)

        if new_tau > best_tau:
            best_tau = new_tau
            best_pair = (qid, pid)

        current_grades[qid][pid] = g
        leaderboard.update_pairs([(qid, pid, h, g)])

    return best_pair, best_tau


# ── Main correction loop (batch policies) ──────────────────────────────

def _run_single_loop(policy_name, policy_fn, universe, human_qrels, llm_qrels,
                     current_grades, runs, queries, system_names, sys_top10,
                     pair_weights, confidence, gold_ranking, M, adaptive,
                     batch_size, rng, is_oracle=False):
    """Single run of the correction loop for batch policies.
    Returns list of recorded points with tau_all, tau_at_20, max_drop."""
    n_universe = len(universe)

    grades = {}
    for qid in current_grades:
        grades[qid] = dict(current_grades[qid])

    board = Leaderboard(runs, grades, queries, system_names, sys_top10)
    judged = set()
    unjudged = set(universe)
    curve = [_record_metrics(board, gold_ranking, 0, n_universe)]

    target_systems = board.top_M(M)

    while len(judged) < n_universe:
        if adaptive:
            target_systems = board.top_M(M)

        this_batch = min(batch_size, len(unjudged))
        if this_batch == 0:
            break

        if is_oracle:
            for _ in range(this_batch):
                if not unjudged:
                    break
                best_pair, _ = run_oracle_tau(
                    list(unjudged), human_qrels, grades, board,
                    gold_ranking, 1, K=20)
                if best_pair is None:
                    best_pair = next(iter(unjudged))
                qid, pid = best_pair
                old_g = grades[qid].get(pid, 0)
                new_g = human_qrels[qid][pid]
                grades[qid][pid] = new_g
                board.update_pairs([(qid, pid, old_g, new_g)])
                judged.add(best_pair)
                unjudged.discard(best_pair)
        else:
            unjudged_list = list(unjudged)
            scores = policy_fn(unjudged_list, pair_weights, confidence,
                               target_systems, rng)
            unjudged_list.sort(key=lambda k: -scores.get(k, 0.0))
            batch = unjudged_list[:this_batch]

            corrections = []
            for qid, pid in batch:
                old_g = grades[qid].get(pid, 0)
                new_g = human_qrels[qid][pid]
                grades[qid][pid] = new_g
                corrections.append((qid, pid, old_g, new_g))
                judged.add((qid, pid))
                unjudged.discard((qid, pid))
            board.update_pairs(corrections)

        curve.append(_record_metrics(board, gold_ranking, len(judged), n_universe))

    return curve


# ── Adaptive pooling loops (MTF and MM-NS) ───────────────────────────────
# These policies are adaptive on revealed labels: they run one pair at a time,
# updating system priorities/rewards after each correction. They record metrics
# at checkpoint intervals matching the batch_size used by other policies.

def _run_mtf_loop(universe, human_qrels, llm_qrels, current_grades,
                  runs, queries, system_names, sys_top10,
                  gold_ranking, batch_size):
    """Move-to-Front pooling (Cormack et al. 1998).

    All systems start at equal priority. Each step: pick the highest-priority
    system, correct its next unjudged document in run-rank order, reveal the
    human grade. If relevant (grade >= RELEVANCE_THRESHOLD), keep priority.
    If not, move system to end of priority queue."""
    n_universe = len(universe)
    universe_set = set(universe)

    grades = {}
    for qid in current_grades:
        grades[qid] = dict(current_grades[qid])

    board = Leaderboard(runs, grades, queries, system_names, sys_top10)
    curve = [_record_metrics(board, gold_ranking, 0, n_universe)]

    # Build per-system document queues: for each system, interleave queries
    # in round-robin, documents in run-rank order, restricted to universe
    sys_queues = {}
    for sys_name in system_names:
        sys_run = runs.get(sys_name, {})
        # For each query, list of unjudged pids in rank order
        q_iters = {}
        for qid in queries:
            pids = sys_run.get(qid, [])
            q_iters[qid] = [pid for pid in pids if (qid, pid) in universe_set]
        sys_queues[sys_name] = q_iters

    # Priority: systems in a deque, front = highest priority
    priority = deque(system_names)
    judged = set()
    judged_count = 0

    # Cursor per (system, query): index into sys_queues[sys][qid]
    cursors = {sys_name: {qid: 0 for qid in queries} for sys_name in system_names}

    while judged_count < n_universe:
        # Find next pair from highest-priority system
        found = False
        systems_tried = 0
        while systems_tried < len(priority):
            sys_name = priority[0]
            # Round-robin across queries for this system
            pair = _next_unjudged_for_system(
                sys_name, runs, queries, universe_set, judged, cursors)
            if pair is not None:
                found = True
                break
            # System exhausted, move to back
            priority.rotate(-1)
            systems_tried += 1

        if not found:
            break

        qid, pid = pair
        old_g = grades[qid].get(pid, 0)
        new_g = human_qrels[qid][pid]
        grades[qid][pid] = new_g
        board.update_pairs([(qid, pid, old_g, new_g)])
        judged.add((qid, pid))
        judged_count += 1

        # MTF priority update: if relevant, keep at front; if not, move to back
        is_relevant = (new_g >= RELEVANCE_THRESHOLD)
        if not is_relevant:
            priority.rotate(-1)  # current system moves to back

        # Record at checkpoint intervals
        if judged_count % batch_size == 0 or judged_count == n_universe:
            curve.append(_record_metrics(board, gold_ranking, judged_count, n_universe))

    # Final point if not already recorded
    if curve[-1]["budget"] < judged_count / n_universe:
        curve.append(_record_metrics(board, gold_ranking, judged_count, n_universe))

    return curve


def _next_unjudged_for_system(sys_name, runs, queries, universe_set, judged, cursors):
    """Find the next unjudged (qid, pid) for a system in run-rank order,
    round-robin across queries."""
    sys_run = runs.get(sys_name, {})
    # Try each query in order, find the shallowest unjudged document
    best_pair = None
    best_rank = float('inf')
    for qid in queries:
        pids = sys_run.get(qid, [])
        cursor = cursors[sys_name][qid]
        while cursor < len(pids):
            pid = pids[cursor]
            k = (qid, pid)
            if k in judged or k not in universe_set:
                cursor += 1
                continue
            if cursor < best_rank:
                best_rank = cursor
                best_pair = k
            break
        cursors[sys_name][qid] = cursor
    return best_pair


def _run_mmns_loop(universe, human_qrels, llm_qrels, current_grades,
                   runs, queries, system_names, sys_top10,
                   gold_ranking, batch_size, window=50, epsilon=0.1,
                   rng=None):
    """MaxMean Non-Stationary bandit (Losada et al. 2016, 'Feeling Lucky?').

    Each system is an arm. Pulling an arm corrects that system's next unjudged
    document (in run-rank order) and reveals the human grade. Reward is binarized
    relevance (grade >= RELEVANCE_THRESHOLD). Pick the arm with the highest
    windowed mean reward (non-stationary estimate over recent pulls, since a
    run's precision falls with depth). Epsilon-greedy exploration."""
    if rng is None:
        rng = np.random.RandomState(SEED)

    n_universe = len(universe)
    universe_set = set(universe)

    grades = {}
    for qid in current_grades:
        grades[qid] = dict(current_grades[qid])

    board = Leaderboard(runs, grades, queries, system_names, sys_top10)
    curve = [_record_metrics(board, gold_ranking, 0, n_universe)]

    # Per-system reward windows
    reward_windows = {sys_name: deque(maxlen=window) for sys_name in system_names}
    # Seed each arm with one pull worth of prior (0.5) so we don't divide by zero
    for sys_name in system_names:
        reward_windows[sys_name].append(0.5)

    # Track which systems still have unjudged documents
    cursors = {sys_name: {qid: 0 for qid in queries} for sys_name in system_names}
    judged = set()
    judged_count = 0

    # Systems that still have documents to judge
    active_systems = set(system_names)

    while judged_count < n_universe and active_systems:
        # Epsilon-greedy: explore or exploit
        if rng.random() < epsilon:
            sys_name = rng.choice(list(active_systems))
        else:
            # Pick arm with highest windowed mean reward
            best_sys = None
            best_mean = -1.0
            for s in active_systems:
                w = reward_windows[s]
                m = sum(w) / len(w) if w else 0.0
                if m > best_mean:
                    best_mean = m
                    best_sys = s
            sys_name = best_sys

        # Find next unjudged document for this system
        pair = _next_unjudged_for_system(
            sys_name, runs, queries, universe_set, judged, cursors)

        if pair is None:
            active_systems.discard(sys_name)
            continue

        qid, pid = pair
        old_g = grades[qid].get(pid, 0)
        new_g = human_qrels[qid][pid]
        grades[qid][pid] = new_g
        board.update_pairs([(qid, pid, old_g, new_g)])
        judged.add((qid, pid))
        judged_count += 1

        # Update reward window for this arm
        reward = 1.0 if new_g >= RELEVANCE_THRESHOLD else 0.0
        reward_windows[sys_name].append(reward)

        # Record at checkpoint intervals
        if judged_count % batch_size == 0 or judged_count == n_universe:
            curve.append(_record_metrics(board, gold_ranking, judged_count, n_universe))

    # Final point if not already recorded
    if curve[-1]["budget"] < judged_count / n_universe:
        curve.append(_record_metrics(board, gold_ranking, judged_count, n_universe))

    return curve


# ── Area computation ─────────────────────────────────────────────────────

def area_under_curve(curve, metric_key):
    """Trapezoidal area under a policy curve."""
    budgets = np.array([p["budget"] for p in curve])
    vals = np.array([p[metric_key] for p in curve])
    return float(np.trapezoid(vals, budgets))


# ── Plotting ─────────────────────────────────────────────────────────────

POLICY_CFG = {
    "naive":                    {"color": "tab:cyan",   "ls": "-",  "label": "Naive (uncertainty)"},
    "depth_k":                  {"color": "tab:olive",  "ls": "-",  "label": "Depth-k"},
    "mtf":                      {"color": "tab:green",  "ls": "-",  "label": "MTF (Cormack)"},
    "mm_ns":                    {"color": "tab:purple", "ls": "-",  "label": "MM-NS (Losada)"},
    "popularity":               {"color": "tab:brown",  "ls": "-",  "label": "Popularity"},
    "leverage":                 {"color": "tab:orange", "ls": "-",  "label": "Leverage"},
    "lara":         {"color": "tab:red",   "ls": "-",  "label": "LARA"},
    "oracle_tau":   {"color": "black",      "ls": "--", "label": "Oracle (tau@20)"},
}

def _get_policy_cfg(pol_name):
    return POLICY_CFG.get(pol_name, {"color": "tab:gray", "ls": "-", "label": pol_name})


def plot_year(year, results, output_path):
    """Three-panel plot: tau@20, tau_all, max_drop vs budget."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    for ax, metric, ylabel, lower_better in [
        (ax1, "tau_at_20", r"Kendall $\tau$@20", False),
        (ax2, "tau_all", r"Kendall $\tau$ (all)", False),
        (ax3, "max_drop", "Max Drop (rank positions)", True),
    ]:
        for pol_name, curve in results.items():
            cfg = _get_policy_cfg(pol_name)
            b = [p["budget"] for p in curve]
            v = [p[metric] for p in curve]
            ax.plot(b, v, color=cfg["color"], linestyle=cfg["ls"],
                    label=cfg["label"], linewidth=1.5)

        ax.set_xlabel("Human budget (fraction of universe)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_title(f"DL {year}", fontsize=11)
        ax.legend(fontsize=7, loc="lower right" if not lower_better else "upper right")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Top-M Targeted Human Correction")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="Years to run (default: all)")
    parser.add_argument("--M", type=int, default=20,
                        help="Top-M system set size (default: 20)")
    parser.add_argument("--adaptive", action="store_true", default=True,
                        help="Adaptive top-M (default: True)")
    parser.add_argument("--static", action="store_true",
                        help="Use static top-M instead of adaptive")
    parser.add_argument("--batch-fraction", type=float, default=BATCH_FRACTION)
    parser.add_argument("--oracle", action="store_true",
                        help="Run oracle_tau policy (very slow)")
    parser.add_argument("--oracle-batch", type=int, default=None,
                        help="Oracle batch size (default: same as others)")
    parser.add_argument("--no-lara", action="store_true",
                        help="Skip LARA calibration policy")
    parser.add_argument("--mmns-window", type=int, default=50,
                        help="MM-NS reward window size (default: 50)")
    parser.add_argument("--mmns-epsilon", type=float, default=0.1,
                        help="MM-NS epsilon-greedy exploration rate (default: 0.1)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.static:
        args.adaptive = False

    years_to_run = args.years if args.years else sorted(YEARS.keys())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mode_label = "adaptive" if args.adaptive else "static"
    print(f"=== TOP-M TARGETED CORRECTION (M={args.M}, {mode_label}) ===\n")

    all_summary = {}
    all_curve_rows = []

    for year in years_to_run:
        cfg = YEARS[year]
        print(f"--- Year {year} ---")

        # Load data
        human_qrels = load_qrels(cfg["qrels"])
        year_queries = set(human_qrels.keys())
        llm_qrels = load_llm_qrels(cfg["scores"], year_queries)
        year_queries &= set(llm_qrels.keys())

        runs = load_system_runs(cfg["runs_dir"])
        if year in V2_YEARS:
            canonicalize_runs(runs, load_canonical_map())

        conf_data = load_confidence(year)

        # Load full softmax for LARA calibrator
        softmax_probs = load_softmax_probs(cfg["scores"], year_queries)

        # Build universe: pooled (q,p) pairs with both human and LLM grades
        universe = []
        for qid in sorted(year_queries):
            for pid in human_qrels.get(qid, {}):
                if pid in llm_qrels.get(qid, {}):
                    universe.append((qid, pid))

        queries = sorted(year_queries)
        system_names = sorted(runs.keys())
        n_universe = len(universe)
        batch_size = max(1, int(round(args.batch_fraction * n_universe)))

        print(f"  {len(queries)} queries, {len(system_names)} systems, "
              f"{n_universe} pooled pairs, batch={batch_size}")

        # Build run weights and pool depth
        pair_weights, sys_top10 = build_run_weights(runs, queries, system_names)
        universe_set = set(universe)
        pool_depth, pool_nsys = build_pool_depth(runs, queries, system_names, universe_set)

        # Build confidence dict
        confidence = {}
        n_conf_found = 0
        for qid, pid in universe:
            m = conf_data.get((qid, pid))
            if m is not None:
                confidence[(qid, pid)] = m
                n_conf_found += 1
            else:
                confidence[(qid, pid)] = 1.0  # assume confident if missing
        print(f"  Confidence signals found for {n_conf_found}/{n_universe} pairs")

        # Initial grades: all LLM
        init_grades = {}
        for qid in queries:
            init_grades[qid] = {}
            for pid in human_qrels.get(qid, {}):
                if pid in llm_qrels.get(qid, {}):
                    init_grades[qid][pid] = llm_qrels[qid][pid]

        # Gold ranking
        gold_board = Leaderboard(runs, {qid: dict(human_qrels.get(qid, {}))
                                        for qid in queries},
                                 queries, system_names, sys_top10)
        gold_ranking = gold_board.get_ranking()

        # LLM ranking for reporting
        llm_board = Leaderboard(runs, init_grades, queries, system_names, sys_top10)
        llm_ranking = llm_board.get_ranking()
        llm_tau20 = compute_tau_at_k(gold_ranking, llm_ranking, args.M)
        llm_tau_all = compute_tau(gold_ranking, llm_ranking)
        llm_max_drop = compute_max_drop(gold_ranking, llm_ranking)
        print(f"  LLM baseline: tau@{args.M}={llm_tau20:.4f}, tau_all={llm_tau_all:.4f}, "
              f"max_drop={llm_max_drop}")

        gold_topM = set(gold_ranking[:args.M])
        llm_topM = set(llm_ranking[:args.M])
        overlap = len(gold_topM & llm_topM)
        print(f"  Top-{args.M} overlap (gold vs LLM): {overlap}/{args.M}")

        # ── Define and run policies ──────────────────────────────────────

        year_results = {}

        # 1. Batch policies
        batch_policies = [
            ("naive",       score_naive),
            ("depth_k",     score_depth_k(pool_depth, pool_nsys)),
            ("popularity",  score_popularity),
            ("leverage",    score_leverage),
        ]

        for pol_name, pol_fn in batch_policies:
            print(f"  Running policy: {pol_name}...")
            curve = _run_single_loop(
                pol_name, pol_fn, universe, human_qrels, llm_qrels,
                init_grades, runs, queries, system_names, sys_top10,
                pair_weights, confidence, gold_ranking, args.M,
                args.adaptive, batch_size,
                rng=np.random.RandomState(args.seed))
            year_results[pol_name] = curve

        # 2. Adaptive pooling policies (one pair at a time)
        print(f"  Running policy: mtf...")
        year_results["mtf"] = _run_mtf_loop(
            universe, human_qrels, llm_qrels, init_grades,
            runs, queries, system_names, sys_top10,
            gold_ranking, batch_size)

        print(f"  Running policy: mm_ns...")
        year_results["mm_ns"] = _run_mmns_loop(
            universe, human_qrels, llm_qrels, init_grades,
            runs, queries, system_names, sys_top10,
            gold_ranking, batch_size,
            window=args.mmns_window, epsilon=args.mmns_epsilon,
            rng=np.random.RandomState(args.seed))

        # 3. LARA (calibrated margin selection + calibrated remainder)
        if not args.no_lara:
            print(f"  Running policy: lara...")
            year_results["lara"] = _run_lara(
                universe, human_qrels, llm_qrels, init_grades,
                softmax_probs, runs, queries, system_names, sys_top10,
                pair_weights, confidence, gold_ranking, args.M,
                args.adaptive, batch_size, seed=args.seed)

        # 4. Oracle (optional)
        if args.oracle:
            print(f"  Running policy: oracle_tau (SLOW)...")
            oracle_bs = args.oracle_batch if args.oracle_batch else batch_size
            curve = _run_single_loop(
                "oracle_tau", None, universe, human_qrels, llm_qrels,
                init_grades, runs, queries, system_names, sys_top10,
                pair_weights, confidence, gold_ranking, args.M,
                args.adaptive, oracle_bs,
                rng=np.random.RandomState(args.seed), is_oracle=True)
            year_results["oracle_tau"] = curve

        # ── Report areas ─────────────────────────────────────────────────

        print(f"\n  {'Policy':<20s} {'area_tau@20':>12s} {'area_tau_all':>12s} "
              f"{'area_max_drop':>14s}")
        print(f"  {'-'*62}")

        summary = {"year": year, "M": args.M, "mode": mode_label,
                   "n_queries": len(queries), "n_systems": len(system_names),
                   "n_universe": n_universe, "llm_tau_at_20": llm_tau20,
                   "llm_tau_all": llm_tau_all, "llm_max_drop": llm_max_drop,
                   "top_M_overlap": overlap}

        for pol_name, curve in year_results.items():
            a20 = area_under_curve(curve, "tau_at_20")
            a_all = area_under_curve(curve, "tau_all")
            a_drop = area_under_curve(curve, "max_drop")
            print(f"  {pol_name:<20s} {a20:>12.4f} {a_all:>12.4f} {a_drop:>14.4f}")
            summary[f"{pol_name}_area_tau_at_20"] = a20
            summary[f"{pol_name}_area_tau_all"] = a_all
            summary[f"{pol_name}_area_max_drop"] = a_drop

        all_summary[year] = summary

        # Collect all curve points for this year into the global results list
        for pol_name, curve in year_results.items():
            for p in curve:
                all_curve_rows.append({
                    "year": year, "policy": pol_name,
                    "budget": p["budget"],
                    "tau_at_20": p["tau_at_20"],
                    "tau_all": p["tau_all"],
                    "max_drop": p["max_drop"],
                })

        # Plot
        plot_year(year, year_results,
                  OUTPUT_DIR / f"{year}_topM_correction_M{args.M}_{mode_label}.pdf")

        print()

    # ── Final outputs ────────────────────────────────────────────────────

    # final_simple_results.csv: all curves, all years, all policies
    results_df = pd.DataFrame(all_curve_rows)
    results_path = OUTPUT_DIR / "final_simple_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved {results_path} ({len(results_df)} rows)")

    # Summary table (areas)
    print("\n=== SUMMARY (areas) ===")
    summary_df = pd.DataFrame(all_summary.values())
    print(summary_df.to_string(index=False))
    summary_df.to_csv(OUTPUT_DIR / f"summary_M{args.M}_{mode_label}.csv", index=False)

    print(f"\nDone! Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
