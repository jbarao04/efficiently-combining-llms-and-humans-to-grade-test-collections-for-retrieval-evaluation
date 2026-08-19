# Restricted-pool policy comparison on eligible-only pairs (non-zero leverage)

import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import math
import os
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from v2_id_mapping import V2_YEARS, load_canonical_map, canonicalize_runs

# ── Constants ──────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "results" / "thesis_verification" / "t18b_restricted_pool"

CONFIDENCE_CSV = BASE_DIR / "results" / "spectral" / "confidence_passage_linear.csv"

TAU_THRESHOLD = 0.95
K_TOP         = 20
M_TOP         = 20
BATCH_FRACTION = 0.01
SEED          = 42
RELEVANCE_THRESHOLD = 2

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


# ── Data loading (same as run_topM_correction.py) ─────────────────────────

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
    for fname in sorted(os.listdir(str(runs_dir))):
        if not fname.endswith(".txt"):
            continue
        sname = fname[:-4]
        sr = defaultdict(list)
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


def load_confidence(year):
    df = pd.read_csv(CONFIDENCE_CSV, dtype={"query_id": str, "passage_id": str})
    df = df[df["year"] == year]
    return {(row["query_id"], row["passage_id"]): row["margin"]
            for _, row in df.iterrows()}


def load_softmax_probs(jsonl_path, year_queries):
    probs = {}
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            qid = str(rec["query_id"])
            if qid not in year_queries:
                continue
            pid = str(rec["passage_id"])
            p = rec["probs"]
            vec = np.array([float(p.get(str(k), 0.0)) for k in range(4)],
                           dtype=np.float64)
            probs[(qid, pid)] = vec
    return probs


# ── nDCG and ranking ──────────────────────────────────────────────────────

def ndcg_at_k(ranked_pids, qrels_q, k=10):
    if not qrels_q:
        return 0.0
    dcg = sum(qrels_q.get(p, 0) / math.log2(i + 2)
              for i, p in enumerate(ranked_pids[:k]))
    idcg = sum(g / math.log2(i + 2)
               for i, g in enumerate(sorted(qrels_q.values(), reverse=True)[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def rank_systems(scores, system_names):
    return [n for n, _ in sorted(zip(system_names, scores),
                                  key=lambda x: (-x[1], x[0]))]


def compute_tau(gold, pred):
    rg = {n: i for i, n in enumerate(gold)}
    tau, _ = kendalltau(list(range(len(gold))), [rg[n] for n in pred])
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


def max_drop(gold, pred):
    gr = {n: i for i, n in enumerate(gold)}
    pr = {n: i for i, n in enumerate(pred)}
    return max(pr[n] - gr[n] for n in gold)


# ── Leaderboard (from run_topM_correction.py) ────────────────────────────

class Leaderboard:
    def __init__(self, runs, current_grades, queries, system_names, sys_top10):
        self.runs = runs
        self.grades = current_grades
        self.queries = queries
        self.system_names = system_names
        self.sys_top10 = sys_top10
        self.n_q = len(queries)
        self.n_sys = len(system_names)
        self.sys_idx = {s: i for i, s in enumerate(system_names)}
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
        return set(self.get_ranking()[:M])


# ── Run weights and pool depth ────────────────────────────────────────────

def build_run_weights(runs, queries, system_names):
    pair_weights = defaultdict(dict)
    sys_top10 = {}
    for sys_name in system_names:
        sys_run = runs.get(sys_name, {})
        top10 = {}
        for qid in queries:
            pids = sys_run.get(qid, [])[:10]
            top10[qid] = pids
            for rank_idx, pid in enumerate(pids):
                w = 1.0 / math.log2(rank_idx + 2)
                pair_weights[(qid, pid)][sys_name] = w
        sys_top10[sys_name] = top10
    return dict(pair_weights), sys_top10


def build_pool_depth(runs, queries, system_names, universe_set):
    depth = {}
    n_sys = {}
    for sys_name in system_names:
        sys_run = runs.get(sys_name, {})
        for qid in queries:
            for rank_idx, pid in enumerate(sys_run.get(qid, [])):
                k = (qid, pid)
                if k not in universe_set:
                    continue
                r = rank_idx + 1
                if k not in depth or r < depth[k]:
                    depth[k] = r
                n_sys[k] = n_sys.get(k, 0) + 1
    return depth, n_sys


# ── Eligibility computation ──────────────────────────────────────────────

def compute_eligible_all_runs(universe, pair_weights):
    """Eligible under all-runs C: pair has any non-zero weight in pair_weights."""
    eligible = set()
    for k in universe:
        if k in pair_weights and len(pair_weights[k]) > 0:
            eligible.add(k)
    return eligible


def compute_eligible_adaptive_top20(universe, pair_weights, init_top20):
    """Eligible under adaptive-top-20 C: pair appears in top-10 of at least
    one system in the initial top-20 set."""
    eligible = set()
    for k in universe:
        ws = pair_weights.get(k, {})
        if any(s in init_top20 for s in ws):
            eligible.add(k)
    return eligible


# ── Retrieval count for top-10 ────────────────────────────────────────────

def build_retrieval_count_top10(runs, queries, system_names, universe_set):
    """Count how many systems retrieve each pair in their top-10."""
    ret_count = defaultdict(int)
    for sn in system_names:
        sr = runs.get(sn, {})
        for qid in queries:
            for pid in sr.get(qid, [])[:10]:
                k = (qid, pid)
                if k in universe_set:
                    ret_count[k] += 1
    return dict(ret_count)


# ── Record metrics helper ────────────────────────────────────────────────

def record_metrics(board, gold_ranking, judged_count, n_universe):
    ranking = board.get_ranking()
    return {
        "budget": judged_count / n_universe if n_universe > 0 else 0.0,
        "budget_pairs": judged_count,
        "tau_all": compute_tau(gold_ranking, ranking),
        "tau_at_20": compute_tau_at_k(gold_ranking, ranking, K_TOP),
        "max_drop": max_drop(gold_ranking, ranking),
    }


# ── Policy scoring functions ─────────────────────────────────────────────

def score_naive(pair_keys, pair_weights, confidence, target_systems, rng):
    return {k: 1.0 - confidence.get(k, 1.0) for k in pair_keys}


def score_popularity(pair_keys, pair_weights, confidence, target_systems, rng):
    return {k: sum(pair_weights.get(k, {}).values()) for k in pair_keys}


def score_leverage(pair_keys, pair_weights, confidence, target_systems, rng):
    t_list = list(target_systems)
    scores = {}
    for k in pair_keys:
        ws = pair_weights.get(k, {})
        vals = [ws.get(s, 0.0) for s in t_list]
        scores[k] = float(np.var(vals)) if vals else 0.0
    return scores


def score_depth_k(pool_depth, pool_nsys):
    def _score(pair_keys, pair_weights, confidence, target_systems, rng):
        return {k: -pool_depth.get(k, 9999) + pool_nsys.get(k, 0) * 1e-6
                for k in pair_keys}
    return _score


def score_retrieval_count(ret_count):
    """Descending count of systems retrieving the pair in top-10."""
    def _score(pair_keys, pair_weights, confidence, target_systems, rng):
        return {k: ret_count.get(k, 0) for k in pair_keys}
    return _score


# ── Correction loop (batch policies) ─────────────────────────────────────

def run_single_loop(policy_fn, universe, human_qrels, init_grades,
                    runs, queries, system_names, sys_top10,
                    pair_weights, confidence, gold_ranking, M,
                    batch_size, rng, n_total_universe):
    """Run correction sweep. n_total_universe is the denominator for budget
    (may differ from len(universe) when running restricted pool but reporting
    budget as fraction of full pool)."""
    n_pool = len(universe)
    grades = {}
    for qid in init_grades:
        grades[qid] = dict(init_grades[qid])
    board = Leaderboard(runs, grades, queries, system_names, sys_top10)
    judged = set()
    unjudged = set(universe)
    curve = [record_metrics(board, gold_ranking, 0, n_total_universe)]
    target_systems = board.top_M(M)

    while len(judged) < n_pool:
        target_systems = board.top_M(M)
        this_batch = min(batch_size, len(unjudged))
        if this_batch == 0:
            break
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
        curve.append(record_metrics(board, gold_ranking, len(judged), n_total_universe))
    return curve


# ── MTF loop ──────────────────────────────────────────────────────────────

def _next_unjudged_for_system(sys_name, runs, queries, universe_set, judged, cursors):
    sys_run = runs.get(sys_name, {})
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


def run_mtf_loop(universe, human_qrels, init_grades,
                 runs, queries, system_names, sys_top10,
                 gold_ranking, batch_size, n_total_universe):
    n_pool = len(universe)
    universe_set = set(universe)
    grades = {}
    for qid in init_grades:
        grades[qid] = dict(init_grades[qid])
    board = Leaderboard(runs, grades, queries, system_names, sys_top10)
    curve = [record_metrics(board, gold_ranking, 0, n_total_universe)]
    priority = deque(system_names)
    judged = set()
    judged_count = 0
    cursors = {sn: {qid: 0 for qid in queries} for sn in system_names}

    while judged_count < n_pool:
        found = False
        systems_tried = 0
        while systems_tried < len(priority):
            sys_name = priority[0]
            pair = _next_unjudged_for_system(
                sys_name, runs, queries, universe_set, judged, cursors)
            if pair is not None:
                found = True
                break
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
        if new_g < RELEVANCE_THRESHOLD:
            priority.rotate(-1)
        if judged_count % batch_size == 0 or judged_count == n_pool:
            curve.append(record_metrics(board, gold_ranking, judged_count, n_total_universe))
    if curve[-1]["budget_pairs"] < judged_count:
        curve.append(record_metrics(board, gold_ranking, judged_count, n_total_universe))
    return curve


# ── LARA (calibrated margin + calibrated remainder) ──────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

CALIBRATOR_WARMUP = 50


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
        self._trained = False
        if random_state is not None:
            torch.manual_seed(random_state)

    @property
    def is_active(self):
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
        for _ in range(2):
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

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        if not self.is_active:
            return X.copy()
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            cumulative_probs = torch.sigmoid(logits)
            return self._from_ordinal(cumulative_probs).cpu().numpy()

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
        probs = self.predict_proba(X)
        sorted_p = np.sort(probs, axis=1)[:, ::-1]
        return sorted_p[:, 0] - sorted_p[:, 1]


def build_calibrated_grades(calibrator, softmax_probs, universe, judged,
                            human_grades, llm_grades):
    grades = {}
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


def run_lara(universe, human_qrels, init_grades, softmax_probs,
             runs, queries, system_names, sys_top10,
             pair_weights, confidence, gold_ranking, M,
             batch_size, n_total_universe, seed=42):
    n_pool = len(universe)
    n_classes = 4
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
    curve = [record_metrics(board_raw, gold_ranking, 0, n_total_universe)]

    while len(judged) < n_pool:
        this_batch = min(batch_size, len(unjudged))
        if this_batch == 0:
            break
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

        if calibrator.is_active:
            cal_grades = build_calibrated_grades(
                calibrator, softmax_probs, universe, judged,
                human_qrels, init_grades)
            board_cal = Leaderboard(runs, cal_grades, queries,
                                    system_names, sys_top10)
            curve.append(record_metrics(
                board_cal, gold_ranking, len(judged), n_total_universe))
        else:
            curve.append(record_metrics(
                board_raw, gold_ranking, len(judged), n_total_universe))
    return curve


# ── Random baseline ──────────────────────────────────────────────────────

def run_random(universe, human_qrels, init_grades,
               runs, queries, system_names, sys_top10,
               gold_ranking, batch_size, n_total_universe,
               n_draws=10, seed=42):
    """Random over given universe (restricted or full). 10 draws for speed."""
    rng = np.random.RandomState(seed)
    n_pool = len(universe)
    all_curves = []
    for draw in range(n_draws):
        perm = rng.permutation(n_pool)
        shuffled = [universe[i] for i in perm]
        grades = {}
        for qid in init_grades:
            grades[qid] = dict(init_grades[qid])
        board = Leaderboard(runs, grades, queries, system_names, sys_top10)
        judged_count = 0
        curve = [record_metrics(board, gold_ranking, 0, n_total_universe)]

        while judged_count < n_pool:
            this_batch = min(batch_size, n_pool - judged_count)
            corrections = []
            for i in range(this_batch):
                qid, pid = shuffled[judged_count + i]
                old_g = grades[qid].get(pid, 0)
                new_g = human_qrels[qid][pid]
                grades[qid][pid] = new_g
                corrections.append((qid, pid, old_g, new_g))
            board.update_pairs(corrections)
            judged_count += this_batch
            curve.append(record_metrics(board, gold_ranking, judged_count, n_total_universe))
        all_curves.append(curve)

    # Average across draws
    n_steps = min(len(c) for c in all_curves)
    avg_curve = []
    for step in range(n_steps):
        avg_curve.append({
            "budget": np.mean([c[step]["budget"] for c in all_curves]),
            "budget_pairs": int(np.mean([c[step]["budget_pairs"] for c in all_curves])),
            "tau_all": np.mean([c[step]["tau_all"] for c in all_curves]),
            "tau_at_20": np.mean([c[step]["tau_at_20"] for c in all_curves]),
            "max_drop": np.mean([c[step]["max_drop"] for c in all_curves]),
        })
    return avg_curve


# ── First-touch threshold ────────────────────────────────────────────────

def first_touch_threshold(curve, metric="tau_at_20", thr=TAU_THRESHOLD):
    """Smallest budget at which metric first reaches thr."""
    for pt in curve:
        if pt[metric] >= thr:
            return pt["budget"], pt["budget_pairs"], True
    return None, None, False


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  T18b: RESTRICTED-POOL POLICY COMPARISON")
    print("=" * 70)

    composition_rows = []
    sweep_rows = []
    first_touch_rows = []
    gap_rows = []
    all_report_data = {}

    for year in sorted(YEARS_CFG.keys()):
        cfg = YEARS_CFG[year]
        print(f"\n{'='*65}")
        print(f"  YEAR {year}")
        print(f"{'='*65}")

        # Load data
        human_qrels = load_qrels(cfg["qrels"])
        year_qs = set(human_qrels.keys())
        llm_qrels = load_llm_qrels(cfg["scores"], year_qs)
        year_qs &= set(llm_qrels.keys())

        runs = load_system_runs(cfg["runs_dir"])
        if year in V2_YEARS:
            canonicalize_runs(runs, load_canonical_map())

        softmax_probs = load_softmax_probs(cfg["scores"], year_qs)
        conf_data = load_confidence(year)

        queries = sorted(year_qs)
        system_names = sorted(runs.keys())

        # Universe
        universe = [(qid, pid)
                    for qid in queries
                    for pid in human_qrels.get(qid, {})
                    if pid in llm_qrels.get(qid, {})]
        n_universe = len(universe)
        universe_set = set(universe)

        print(f"  {len(queries)} queries, {len(system_names)} systems, {n_universe:,} pairs")

        # Build structures
        pair_weights, sys_top10 = build_run_weights(runs, queries, system_names)
        pool_depth, pool_nsys = build_pool_depth(runs, queries, system_names, universe_set)
        ret_count = build_retrieval_count_top10(runs, queries, system_names, universe_set)

        confidence = {}
        for qid, pid in universe:
            m = conf_data.get((qid, pid))
            confidence[(qid, pid)] = m if m is not None else 1.0

        # Initial grades: all LLM
        init_grades = {}
        for qid in queries:
            init_grades[qid] = {}
            for pid in human_qrels.get(qid, {}):
                if pid in llm_qrels.get(qid, {}):
                    init_grades[qid][pid] = llm_qrels[qid][pid]

        # Gold ranking (full human, all systems)
        gold_board = Leaderboard(runs, {qid: dict(human_qrels.get(qid, {}))
                                        for qid in queries},
                                 queries, system_names, sys_top10)
        gold_ranking = gold_board.get_ranking()

        # Initial LLM ranking (to get initial top-20)
        llm_board = Leaderboard(runs, init_grades, queries, system_names, sys_top10)
        init_top20 = llm_board.top_M(M_TOP)

        # ── POOL COMPOSITION ─────────────────────────────────────────────
        eligible_all = compute_eligible_all_runs(universe, pair_weights)
        eligible_t20 = compute_eligible_adaptive_top20(universe, pair_weights, init_top20)

        n_elig_all = len(eligible_all)
        n_zero_all = n_universe - n_elig_all
        n_elig_t20 = len(eligible_t20)
        n_zero_t20 = n_universe - n_elig_t20

        composition_rows.append({
            "year": year, "c_construction": "all_runs",
            "n_pairs_total": n_universe,
            "n_zero_leverage": n_zero_all,
            "share_zero_leverage": round(n_zero_all / n_universe, 4),
            "n_eligible": n_elig_all,
            "share_eligible": round(n_elig_all / n_universe, 4),
        })
        composition_rows.append({
            "year": year, "c_construction": "adaptive_top20",
            "n_pairs_total": n_universe,
            "n_zero_leverage": n_zero_t20,
            "share_zero_leverage": round(n_zero_t20 / n_universe, 4),
            "n_eligible": n_elig_t20,
            "share_eligible": round(n_elig_t20 / n_universe, 4),
        })

        print(f"\n  Pool composition:")
        print(f"    all_runs:       {n_elig_all:>6,} eligible ({100*n_elig_all/n_universe:.1f}%), "
              f"{n_zero_all:>6,} zero-leverage ({100*n_zero_all/n_universe:.1f}%)")
        print(f"    adaptive_top20: {n_elig_t20:>6,} eligible ({100*n_elig_t20/n_universe:.1f}%), "
              f"{n_zero_t20:>6,} zero-leverage ({100*n_zero_t20/n_universe:.1f}%)")

        if 100 * n_zero_t20 / n_universe > 85:
            print(f"\n  WARNING: adaptive_top20 zero-leverage share = "
                  f"{100*n_zero_t20/n_universe:.1f}% > 85%")
            print("  The policy operates on a very small slice of the pool.")

        # ── Use all_runs eligibility for the restricted pool ─────────────
        # (all_runs is the right filter: pair_weights covers all systems,
        #  and a pair with zero weight in ALL systems has zero C_pp regardless
        #  of which top-M set is used)
        eligible_list = sorted(eligible_all)
        n_eligible = len(eligible_list)
        batch_size = max(1, int(round(BATCH_FRACTION * n_universe)))
        batch_size_r = max(1, int(round(BATCH_FRACTION * n_eligible)))

        # ── RUN SWEEPS ───────────────────────────────────────────────────
        print(f"\n  Running sweeps (batch={batch_size_r} restricted, {batch_size} full)...",
              flush=True)

        # Define remainder treatments
        remainder_treatment = {
            "random": "llm_grade",
            "naive": "llm_grade",
            "depth_k": "llm_grade",
            "popularity": "llm_grade",
            "leverage": "llm_grade",
            "mtf": "llm_grade",
            "retrieval_count": "llm_grade",
            "lara": "calibrated_remainder",
        }

        year_curves = {}  # {(policy, pool): curve}

        # ── Full-pool sweeps: only naive and leverage (needed for gap 1) ──
        full_pool_policies = [
            ("naive",    score_naive),
            ("leverage", score_leverage),
        ]
        for pol_name, pol_fn in full_pool_policies:
            print(f"    {pol_name} (full)...", end="", flush=True)
            curve_f = run_single_loop(
                pol_fn, universe, human_qrels, init_grades,
                runs, queries, system_names, sys_top10,
                pair_weights, confidence, gold_ranking, M_TOP,
                batch_size, np.random.RandomState(SEED), n_universe)
            year_curves[(pol_name, "full")] = curve_f
            ft_b, ft_p, ft_ok = first_touch_threshold(curve_f)
            print(f" {ft_b*100:.1f}%" if ft_ok else " never", flush=True)

        # ── Restricted-pool sweeps: all policies ─────────────────────────
        restricted_batch_policies = [
            ("naive",            score_naive),
            ("depth_k",          score_depth_k(pool_depth, pool_nsys)),
            ("popularity",       score_popularity),
            ("leverage",         score_leverage),
            ("retrieval_count",  score_retrieval_count(ret_count)),
        ]
        for pol_name, pol_fn in restricted_batch_policies:
            print(f"    {pol_name} (restricted)...", end="", flush=True)
            curve_r = run_single_loop(
                pol_fn, eligible_list, human_qrels, init_grades,
                runs, queries, system_names, sys_top10,
                pair_weights, confidence, gold_ranking, M_TOP,
                batch_size_r, np.random.RandomState(SEED), n_universe)
            year_curves[(pol_name, "restricted")] = curve_r
            ft_b, ft_p, ft_ok = first_touch_threshold(curve_r)
            print(f" {ft_b*100:.1f}%" if ft_ok else " never", flush=True)

        # MTF (restricted only — full pool too slow, not needed for key gaps)
        print(f"    mtf (restricted)...", end="", flush=True)
        curve = run_mtf_loop(
            eligible_list, human_qrels, init_grades,
            runs, queries, system_names, sys_top10,
            gold_ranking, batch_size_r, n_universe)
        year_curves[("mtf", "restricted")] = curve
        ft_b, ft_p, ft_ok = first_touch_threshold(curve)
        print(f" {ft_b*100:.1f}%" if ft_ok else " never", flush=True)

        # LARA (restricted only — full pool too slow)
        print(f"    lara (restricted)...", end="", flush=True)
        curve = run_lara(
            eligible_list, human_qrels, init_grades, softmax_probs,
            runs, queries, system_names, sys_top10,
            pair_weights, confidence, gold_ranking, M_TOP,
            batch_size_r, n_universe, seed=SEED)
        year_curves[("lara", "restricted")] = curve
        ft_b, ft_p, ft_ok = first_touch_threshold(curve)
        print(f" {ft_b*100:.1f}%" if ft_ok else " never", flush=True)

        # Random (restricted only, 10 draws)
        print(f"    random (restricted, 10 draws)...", end="", flush=True)
        curve = run_random(
            eligible_list, human_qrels, init_grades,
            runs, queries, system_names, sys_top10,
            gold_ranking, batch_size_r, n_universe, n_draws=10, seed=SEED)
        year_curves[("random", "restricted")] = curve
        ft_b, ft_p, ft_ok = first_touch_threshold(curve)
        print(f" {ft_b*100:.1f}%" if ft_ok else " never", flush=True)

        # ── COLLECT SWEEP ROWS ───────────────────────────────────────────
        all_policies = ["random", "naive", "lara", "depth_k", "popularity",
                        "leverage", "mtf", "retrieval_count"]

        for pol in all_policies:
            for pool in ["full", "restricted"]:
                curve = year_curves.get((pol, pool))
                if curve is None:
                    continue
                for pt in curve:
                    sweep_rows.append({
                        "year": year,
                        "policy": pol,
                        "pool": pool,
                        "budget_restricted_pct": round(pt["budget_pairs"] / n_eligible * 100, 2) if pool == "restricted" else None,
                        "budget_original_pct": round(pt["budget"] * 100, 2),
                        "budget_pairs": pt["budget_pairs"],
                        "tau_all": round(pt["tau_all"], 4),
                        "tau_at_20": round(pt["tau_at_20"], 4),
                        "max_drop": pt["max_drop"],
                        "draw_id": -1,
                    })

        # ── FIRST-TOUCH THRESHOLDS ───────────────────────────────────────
        print(f"\n  First-touch thresholds (tau@20 >= {TAU_THRESHOLD}):")
        print(f"  {'Policy':<18} {'Pool':<12} {'Orig%':>7} {'Restr%':>8} "
              f"{'Pairs':>7} {'Elig bought':>12} {'Remainder':>20}")
        print(f"  {'-'*88}")

        year_ft = {}  # (pol, pool) -> (orig_pct, restr_pct, pairs, eligible_bought)

        for pol in all_policies:
            for pool in ["full", "restricted"]:
                curve = year_curves.get((pol, pool))
                if curve is None:
                    continue
                ft_b, ft_p, ft_ok = first_touch_threshold(curve)
                if ft_ok:
                    orig_pct = round(ft_b * 100, 2)
                    restr_pct = round(ft_p / n_eligible * 100, 2) if pool == "restricted" else None
                    # Count eligible pairs bought
                    elig_bought = min(ft_p, n_eligible) if pool == "restricted" else None
                    if pool == "full":
                        # Need to figure out how many of the first ft_p pairs are eligible
                        # For batch policies, we know the ordering; approximate via the fraction
                        elig_bought = None  # not easily computed for full pool
                else:
                    orig_pct = None
                    restr_pct = None
                    ft_p = None
                    elig_bought = None

                year_ft[(pol, pool)] = (orig_pct, restr_pct, ft_p, elig_bought)

                rt = remainder_treatment.get(pol, "llm_grade")
                op = f"{orig_pct:.1f}" if orig_pct is not None else "N/A"
                rp = f"{restr_pct:.1f}" if restr_pct is not None else "—"
                pp = f"{ft_p:,}" if ft_p is not None else "N/A"
                eb = f"{elig_bought:,}" if elig_bought is not None else "—"

                print(f"  {pol:<18} {pool:<12} {op:>7} {rp:>8} {pp:>7} {eb:>12} {rt:>20}")

                first_touch_rows.append({
                    "year": year,
                    "policy": pol,
                    "pool": pool,
                    "denominator": "original" if pool == "full" else "both",
                    "threshold_pct": orig_pct,
                    "threshold_restricted_pct": restr_pct,
                    "threshold_pairs": ft_p,
                    "eligible_pairs_bought": elig_bought,
                    "remainder_treatment": rt,
                })

        # ── POLICY GAPS ──────────────────────────────────────────────────
        print(f"\n  Policy gaps (pp of original denominator):")

        def get_orig_pct(pol, pool):
            entry = year_ft.get((pol, pool))
            if entry is None:
                return None
            return entry[0]

        comparisons = [
            ("leverage_vs_naive_unrestricted", "leverage", "naive", "full"),
            ("leverage_vs_naive_restricted", "leverage", "naive", "restricted"),
            ("leverage_vs_retrieval_count_restricted", "leverage", "retrieval_count", "restricted"),
        ]

        for comp_name, pol_a, pol_b, pool in comparisons:
            a_pct = get_orig_pct(pol_a, pool)
            b_pct = get_orig_pct(pol_b, pool)
            if a_pct is not None and b_pct is not None:
                gap = round(b_pct - a_pct, 2)
            else:
                gap = None

            # Also get eligible pairs bought for restricted pool
            a_elig = year_ft.get((pol_a, pool), (None, None, None, None))[3]
            b_elig = year_ft.get((pol_b, pool), (None, None, None, None))[3]
            elig_gap = (b_elig - a_elig) if (a_elig is not None and b_elig is not None) else None

            gap_str = f"{gap:+.1f}" if gap is not None else "N/A"
            print(f"    {comp_name}: {gap_str} pp")

            gap_rows.append({
                "year": year,
                "comparison": comp_name,
                "pool": pool,
                "gap_pct_points": gap,
                "gap_eligible_pairs": elig_gap,
                "pol_a_orig_pct": a_pct,
                "pol_b_orig_pct": b_pct,
            })

        all_report_data[year] = {
            "n_universe": n_universe,
            "n_eligible_all": n_elig_all,
            "n_eligible_t20": n_elig_t20,
            "year_ft": dict(year_ft),
        }

    # ── SAVE OUTPUTS ─────────────────────────────────────────────────────
    print("\n\nSaving outputs...")

    pd.DataFrame(composition_rows).to_csv(
        OUTPUT_DIR / "pool_composition.csv", index=False)
    print(f"  pool_composition.csv")

    sweep_df = pd.DataFrame(sweep_rows)
    try:
        sweep_df.to_parquet(OUTPUT_DIR / "restricted_sweeps.parquet", index=False)
        print(f"  restricted_sweeps.parquet ({len(sweep_df)} rows)")
    except Exception:
        sweep_df.to_csv(OUTPUT_DIR / "restricted_sweeps.csv", index=False)
        print(f"  restricted_sweeps.csv ({len(sweep_df)} rows)")

    pd.DataFrame(first_touch_rows).to_csv(
        OUTPUT_DIR / "first_touch_thresholds.csv", index=False)
    print(f"  first_touch_thresholds.csv")

    pd.DataFrame(gap_rows).to_csv(
        OUTPUT_DIR / "policy_gaps.csv", index=False)
    print(f"  policy_gaps.csv")

    # ── REPORT ───────────────────────────────────────────────────────────
    write_report(composition_rows, first_touch_rows, gap_rows, all_report_data)
    print("\nDone.")


def write_report(composition_rows, first_touch_rows, gap_rows, all_report_data):
    comp_df = pd.DataFrame(composition_rows)
    ft_df = pd.DataFrame(first_touch_rows)
    gap_df = pd.DataFrame(gap_rows)

    lines = [
        "# T18b Restricted-Pool Policy Comparison — Report",
        "",
        "## Question",
        "",
        "How much of the run-aware advantage over judge-only selection is:",
        "1. The eligibility filter (never spending budget on zero-leverage pairs)",
        "2. The coarse run signal (retrieval_count: how many systems retrieve a pair)",
        "3. The fine leverage ordering (variance of run weights over adaptive top-20)",
        "",
        "## Pool Composition",
        "",
        "Two C constructions:",
        "- **all_runs**: pair appears in ANY system's top-10",
        "- **adaptive_top20**: pair appears in top-10 of a system in the initial top-20 set",
        "",
        "| Year | Construction | Total | Zero-lev | Share% | Eligible | Share% |",
        "|------|-------------|-------|----------|--------|----------|--------|",
    ]

    for _, row in comp_df.iterrows():
        lines.append(
            f"| {int(row['year'])} | {row['c_construction']} "
            f"| {int(row['n_pairs_total']):,} "
            f"| {int(row['n_zero_leverage']):,} "
            f"| {row['share_zero_leverage']*100:.1f} "
            f"| {int(row['n_eligible']):,} "
            f"| {row['share_eligible']*100:.1f} |"
        )
    lines.append("")

    # First-touch table
    lines += [
        "## First-Touch Thresholds (tau@20 >= 0.95, original denominator %)",
        "",
        "| Year | Policy | Full% | Restricted% | Remainder |",
        "|------|--------|-------|-------------|-----------|",
    ]

    all_policies = ["random", "naive", "lara", "depth_k", "popularity",
                    "leverage", "mtf", "retrieval_count"]

    for year in sorted(all_report_data.keys()):
        for pol in all_policies:
            ft_full = ft_df[(ft_df["year"] == year) & (ft_df["policy"] == pol) & (ft_df["pool"] == "full")]
            ft_rest = ft_df[(ft_df["year"] == year) & (ft_df["policy"] == pol) & (ft_df["pool"] == "restricted")]
            f_val = ft_full["threshold_pct"].values[0] if len(ft_full) > 0 and pd.notna(ft_full["threshold_pct"].values[0]) else None
            r_val = ft_rest["threshold_pct"].values[0] if len(ft_rest) > 0 and pd.notna(ft_rest["threshold_pct"].values[0]) else None
            if len(ft_full) > 0:
                rt = ft_full["remainder_treatment"].values[0]
            elif len(ft_rest) > 0:
                rt = ft_rest["remainder_treatment"].values[0]
            else:
                rt = "calibrated_remainder" if pol == "lara" else "llm_grade"
            fv = f"{f_val:.1f}" if f_val is not None else "N/A"
            rv = f"{r_val:.1f}" if r_val is not None else "N/A"
            lines.append(f"| {year} | {pol} | {fv} | {rv} | {rt} |")
    lines.append("")

    # Gap analysis
    lines += [
        "## The Three Key Numbers",
        "",
        "Per year, at first touch of tau@20 = 0.95:",
        "",
        "1. **leverage minus naive, unrestricted pool** (thesis's current claim)",
        "2. **leverage minus naive, restricted pool** (ordering effect isolated)",
        "3. **leverage minus retrieval_count, restricted pool** (fine ordering vs crude count)",
        "",
        "| Year | (1) Full gap | (2) Restricted gap | (3) vs ret_count |",
        "|------|-----------|--------------------|------------------|",
    ]

    for year in sorted(all_report_data.keys()):
        g1 = gap_df[(gap_df["year"] == year) & (gap_df["comparison"] == "leverage_vs_naive_unrestricted")]
        g2 = gap_df[(gap_df["year"] == year) & (gap_df["comparison"] == "leverage_vs_naive_restricted")]
        g3 = gap_df[(gap_df["year"] == year) & (gap_df["comparison"] == "leverage_vs_retrieval_count_restricted")]
        v1 = f"{g1['gap_pct_points'].values[0]:+.1f}" if len(g1) > 0 and g1['gap_pct_points'].values[0] is not None else "N/A"
        v2 = f"{g2['gap_pct_points'].values[0]:+.1f}" if len(g2) > 0 and g2['gap_pct_points'].values[0] is not None else "N/A"
        v3 = f"{g3['gap_pct_points'].values[0]:+.1f}" if len(g3) > 0 and g3['gap_pct_points'].values[0] is not None else "N/A"
        lines.append(f"| {year} | {v1} | {v2} | {v3} |")

    lines.append("")

    # Interpretation
    g1_vals = gap_df[gap_df["comparison"] == "leverage_vs_naive_unrestricted"]["gap_pct_points"].dropna()
    g2_vals = gap_df[gap_df["comparison"] == "leverage_vs_naive_restricted"]["gap_pct_points"].dropna()
    g3_vals = gap_df[gap_df["comparison"] == "leverage_vs_retrieval_count_restricted"]["gap_pct_points"].dropna()

    g1_mean = g1_vals.mean() if len(g1_vals) > 0 else float("nan")
    g2_mean = g2_vals.mean() if len(g2_vals) > 0 else float("nan")
    g3_mean = g3_vals.mean() if len(g3_vals) > 0 else float("nan")

    lines += [
        "## Interpretation",
        "",
        f"Mean across years:",
        f"- (1) leverage vs naive, full pool: {g1_mean:+.1f} pp" if not np.isnan(g1_mean) else "- (1) N/A",
        f"- (2) leverage vs naive, restricted: {g2_mean:+.1f} pp" if not np.isnan(g2_mean) else "- (2) N/A",
        f"- (3) leverage vs retrieval_count, restricted: {g3_mean:+.1f} pp" if not np.isnan(g3_mean) else "- (3) N/A",
        "",
    ]

    if not np.isnan(g1_mean) and not np.isnan(g2_mean):
        filter_share = abs(g1_mean - g2_mean)
        if abs(g2_mean) < 0.5 * abs(g1_mean) and abs(g1_mean) > 1:
            lines.append(
                f"**The restricted gap (2) is substantially smaller than the full gap (1). "
                f"A significant part of the advantage ({filter_share:.1f} pp) is the eligibility "
                f"filter, not fine ordering.**"
            )
        elif abs(g2_mean - g1_mean) < 2:
            lines.append(
                f"**The restricted gap (2) is close to the full gap (1). "
                f"The advantage is a real ordering effect within the eligible pool, "
                f"not mainly the eligibility filter.**"
            )
        else:
            lines.append(
                f"**Mixed picture: filter contributes {filter_share:.1f} pp, "
                f"ordering within eligible pool contributes {abs(g2_mean):.1f} pp.**"
            )
        lines.append("")

    if not np.isnan(g3_mean):
        if abs(g3_mean) < 1.5:
            lines.append(
                f"**The leverage vs retrieval_count gap (3) averages {g3_mean:+.1f} pp. "
                f"The crude count captures most of the run-side advantage; fine leverage "
                f"ordering adds little beyond it.**"
            )
        else:
            lines.append(
                f"**The leverage vs retrieval_count gap (3) averages {g3_mean:+.1f} pp. "
                f"Fine leverage ordering provides a meaningful advantage beyond the crude "
                f"retrieval count signal.**"
            )
    lines.append("")

    lines += [
        "---",
        "Budgets reported as percentage of the full universe (original denominator) "
        "for direct comparability with the thesis's Table 4. Positive gap means "
        "pol_b (naive/retrieval_count) reaches threshold LATER than pol_a (leverage).",
    ]

    report_path = OUTPUT_DIR / "REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  REPORT.md")


if __name__ == "__main__":
    main()
