# Passage-length confound control for sentence-permutation stability feature

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from scipy import stats
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

V1_PASSAGES = BASE_DIR / "runpod" / "upload_package" / "data" / "v1" / "judged_passages.jsonl"
V2_PASSAGES = BASE_DIR / "runpod" / "upload_package" / "data" / "v2" / "judged_passages.jsonl"

SCORES_V1 = BASE_DIR / "results" / "scoring" / "normal_scores" / "scores_v1.jsonl"
SCORES_V2 = BASE_DIR / "results" / "scoring" / "normal_scores" / "scores_v2.jsonl"

B1B_DIR = BASE_DIR / "results" / "scoring" / "feature_family_b"
RUN_FILES = {
    0: (B1B_DIR / "b1b_sent_run0_v1.jsonl", B1B_DIR / "b1b_sent_run0_v2.jsonl"),
    1: (B1B_DIR / "b1b_sent_run1_v1.jsonl", B1B_DIR / "b1b_sent_run1_v2.jsonl"),
    2: (B1B_DIR / "b1b_sent_run2_v1.jsonl", B1B_DIR / "b1b_sent_run2_v2.jsonl"),
}

B1B_CSV = BASE_DIR / "results" / "level2" / "b1b_features.csv"

OUTPUT_DIR = BASE_DIR / "results" / "thesis_verification" / "t16_b1b_control"

N_BOOTSTRAP = 5000
SEED = 42
RAW_TAU = 0.338  # headline b1b_stability_tau vs target, from thesis Section 4

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def kendall_tau_ci(x, y, n_boot=N_BOOTSTRAP, seed=SEED):
    """Kendall tau-b with percentile bootstrap CI."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    tau, pval = stats.kendalltau(x, y)
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        t, _ = stats.kendalltau(x[idx], y[idx])
        if not np.isnan(t):
            boots.append(t)
    ci_lo = float(np.percentile(boots, 2.5))
    ci_hi = float(np.percentile(boots, 97.5))
    return float(tau), float(pval), ci_lo, ci_hi, n


def partial_kendall_tau(y, x, covariates):
    """
    Approximate partial Kendall tau of y vs x controlling for covariates,
    via double residualisation (rank-based proxy using linear regression on
    ranked values — standard practice when partial Kendall has no closed form).
    """
    cov_mat = np.column_stack(covariates)
    reg_y = LinearRegression().fit(cov_mat, y)
    reg_x = LinearRegression().fit(cov_mat, x)
    resid_y = y - reg_y.predict(cov_mat)
    resid_x = x - reg_x.predict(cov_mat)
    tau, _ = stats.kendalltau(resid_y, resid_x)
    return float(tau)


def load_passages_jsonl(path):
    """Returns {pid: text}."""
    passages = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pid = str(obj["pid"])
            passages[pid] = obj["passage"]
    return passages


def load_scores_jsonl(paths):
    """Returns {qid: {pid: {"score": int, "probs": {0:f,...}}}}."""
    data = {}
    for path in paths:
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


def expected_score(probs):
    return sum(g * probs[g] for g in range(4))


# ---------------------------------------------------------------------------
# Step 1a: spaCy sentence/token counting
# ---------------------------------------------------------------------------

def build_passage_structure(passages_v1, passages_v2):
    """
    Run spaCy en_core_web_md on all passages (exactly as generate_b1_perturbations.py).
    Returns DataFrame with columns: passage_id, corpus_version, n_sentences, n_tokens, is_permutable
    """
    print("Loading spaCy en_core_web_md ...")
    nlp = spacy.load("en_core_web_md")
    # Ensure sentence detection is active (same check as generate_b1_perturbations.py)
    if "parser" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    all_passages = []  # [(pid, version, text)]
    for pid, text in passages_v1.items():
        all_passages.append((pid, "v1", text))
    for pid, text in passages_v2.items():
        all_passages.append((pid, "v2", text))

    print(f"Analyzing {len(all_passages)} passages with spaCy ...")
    pids   = [x[0] for x in all_passages]
    versions = [x[1] for x in all_passages]
    texts  = [x[2] for x in all_passages]

    rows = []
    for i, doc in enumerate(nlp.pipe(texts, batch_size=256)):
        # Identical to analyze_passage in generate_b1_perturbations.py
        sentence_spans = [(s.start_char, s.end_char) for s in doc.sents]
        n_sent = len(sentence_spans)
        n_tok  = len(doc)
        rows.append({
            "passage_id":     pids[i],
            "corpus_version": versions[i],
            "n_sentences":    n_sent,
            "n_tokens":       n_tok,
            "is_permutable":  int(n_sent >= 2),
        })
        if (i + 1) % 10000 == 0 or (i + 1) == len(all_passages):
            print(f"  {i+1}/{len(all_passages)} done")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 1b: Aggregate to query level
# ---------------------------------------------------------------------------

def build_query_confounds(struct_df, original_scores, b1b_df):
    """
    For each query in b1b_df, aggregate passage-level features over the
    passages that appear in the original LLM scores (same pool b1b used).
    """
    # Build pid → struct lookup
    struct_lookup = struct_df.set_index("passage_id")[
        ["n_sentences", "n_tokens", "is_permutable"]
    ].to_dict("index")

    rows = []
    missing_pids = 0
    for _, row in b1b_df.iterrows():
        qid = str(row["query_id"])
        year = int(row["year"])
        if qid not in original_scores:
            continue
        pids_in_pool = list(original_scores[qid].keys())

        n_sent_vals = []
        n_tok_vals  = []
        is_perm_vals = []
        for pid in pids_in_pool:
            if pid in struct_lookup:
                s = struct_lookup[pid]
                n_sent_vals.append(s["n_sentences"])
                n_tok_vals.append(s["n_tokens"])
                is_perm_vals.append(s["is_permutable"])
            else:
                missing_pids += 1

        if len(n_sent_vals) < 5:
            continue

        n_sent_arr = np.array(n_sent_vals, dtype=float)
        n_tok_arr  = np.array(n_tok_vals, dtype=float)
        is_perm_arr = np.array(is_perm_vals, dtype=float)

        rows.append({
            "query_id":            qid,
            "year":                year,
            "n_passages":          len(n_sent_vals),
            "q_frac_permutable":   float(np.mean(is_perm_arr)),
            "q_mean_sentences":    float(np.mean(n_sent_arr)),
            "q_median_sentences":  float(np.median(n_sent_arr)),
            "q_mean_tokens":       float(np.mean(n_tok_arr)),
            "q_sd_sentences":      float(np.std(n_sent_arr, ddof=1)),
            "b1b_stability_tau":   float(row["b1b_stability_tau"]),
            "target":              float(row["spearman"]),
        })

    if missing_pids:
        print(f"  WARNING: {missing_pids} (passage_id, query) combos missing from struct_df")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 5: Restricted-pool stability tau
# ---------------------------------------------------------------------------

def compute_restricted_tau(original_scores, runs, struct_lookup_perm, b1b_df):
    """
    Recompute b1b_stability_tau using only passages with is_permutable=1.
    Returns DataFrame: query_id, n_permutable_passages, b1b_tau_restricted
    """
    rows = []
    for _, brow in b1b_df.iterrows():
        qid = str(brow["query_id"])
        if qid not in original_scores:
            continue

        # Intersect with all runs
        common_pids = set(original_scores[qid].keys())
        skip = False
        for r_data in runs.values():
            if qid not in r_data:
                skip = True
                break
            common_pids &= set(r_data[qid].keys())
        if skip:
            continue

        # Keep only permutable passages
        perm_pids = [p for p in common_pids if struct_lookup_perm.get(p, 0) == 1]
        n_perm = len(perm_pids)

        if n_perm < 5:
            rows.append({
                "query_id": qid,
                "n_permutable_passages": n_perm,
                "b1b_tau_restricted": np.nan,
            })
            continue

        es_orig = np.array([expected_score(original_scores[qid][p]["probs"])
                            for p in perm_pids])
        taus = []
        for r_idx in sorted(runs.keys()):
            es_r = np.array([expected_score(runs[r_idx][qid][p]["probs"])
                             for p in perm_pids])
            tau, _ = stats.kendalltau(es_orig, es_r)
            if not np.isnan(tau):
                taus.append(float(tau))

        tau_val = float(np.mean(taus)) if taus else np.nan
        rows.append({
            "query_id": qid,
            "n_permutable_passages": n_perm,
            "b1b_tau_restricted": round(tau_val, 6),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    log("=" * 70)
    log("T16: B1b Structural Control for Passage Length")
    log("=" * 70)

    # ------------------------------------------------------------------ #
    # Load data
    # ------------------------------------------------------------------ #
    log("\nLoading passages ...")
    passages_v1 = load_passages_jsonl(V1_PASSAGES)
    passages_v2 = load_passages_jsonl(V2_PASSAGES)
    log(f"  v1: {len(passages_v1)} passages, v2: {len(passages_v2)} passages")
    log(f"  Total: {len(passages_v1) + len(passages_v2)} passages")

    log("\nLoading LLM scores ...")
    original_scores = load_scores_jsonl([SCORES_V1, SCORES_V2])
    log(f"  {len(original_scores)} queries in original scores")

    log("\nLoading b1b run scores ...")
    runs = {}
    for r_idx, (f_v1, f_v2) in RUN_FILES.items():
        runs[r_idx] = load_scores_jsonl([f_v1, f_v2])
    log(f"  {len(runs)} runs loaded")

    log("\nLoading b1b_features.csv ...")
    b1b_df = pd.read_csv(B1B_CSV)
    b1b_df["query_id"] = b1b_df["query_id"].astype(str)
    b1b_df = b1b_df.dropna(subset=["b1b_stability_tau", "spearman"])
    log(f"  {len(b1b_df)} queries with valid b1b_stability_tau")

    # ------------------------------------------------------------------ #
    # Step 1a: Passage-level structure
    # ------------------------------------------------------------------ #
    log("\n" + "=" * 70)
    log("STEP 1a: Passage-level sentence/token counts (spaCy en_core_web_md)")
    log("=" * 70)

    struct_path = OUTPUT_DIR / "passage_structure.csv"
    if struct_path.exists():
        log(f"  Loading cached {struct_path.name} ...")
        struct_df = pd.read_csv(struct_path, dtype={"passage_id": str})
    else:
        struct_df = build_passage_structure(passages_v1, passages_v2)
        struct_df.to_csv(struct_path, index=False)
        log(f"  Saved: {struct_path}")

    # Verify the 28.3% figure
    n_total = len(struct_df)
    n_single = (struct_df["n_sentences"] <= 1).sum()
    pct_single = 100.0 * n_single / n_total
    log(f"\n  Total passages analyzed: {n_total}")
    log(f"  Passages with <= 1 sentence: {n_single} ({pct_single:.1f}%)")
    log(f"  Expected (from generate_b1_perturbations.py): 28.3%")

    if abs(pct_single - 28.3) > 2.0:
        log(f"\n  STOP: pct_single={pct_single:.1f}% deviates from 28.3% by more than 2pp.")
        log("  This suggests a different splitter was used. Aborting.")
        sys.exit(1)
    else:
        log(f"  CHECK PASSED: {pct_single:.1f}% matches 28.3% (within 2pp tolerance)")

    log(f"\n  n_sentences distribution:")
    for pct in [25, 50, 75, 90, 95]:
        log(f"    p{pct}: {np.percentile(struct_df['n_sentences'], pct):.1f}")
    log(f"    mean: {struct_df['n_sentences'].mean():.2f}, sd: {struct_df['n_sentences'].std():.2f}")
    log(f"\n  Tokeniser: spaCy en_core_web_md (len(doc))")

    # ------------------------------------------------------------------ #
    # Step 1b: Query-level confounds
    # ------------------------------------------------------------------ #
    log("\n" + "=" * 70)
    log("STEP 1b: Query-level confound aggregation")
    log("=" * 70)

    qconf_df = build_query_confounds(struct_df, original_scores, b1b_df)
    log(f"  Queries with valid confounds: {len(qconf_df)}")

    # Distribution of q_frac_permutable
    log(f"\n  q_frac_permutable distribution:")
    frac = qconf_df["q_frac_permutable"].values
    log(f"    mean={frac.mean():.4f}  sd={frac.std():.4f}  "
        f"min={frac.min():.4f}  max={frac.max():.4f}")
    for pct in [10, 25, 50, 75, 90]:
        log(f"    p{pct}: {np.percentile(frac, pct):.4f}")

    qconf_df.to_csv(OUTPUT_DIR / "query_confounds.csv", index=False, float_format="%.6f")
    log(f"\n  Saved: query_confounds.csv")

    # ------------------------------------------------------------------ #
    # Step 2: Three framing correlations
    # ------------------------------------------------------------------ #
    log("\n" + "=" * 70)
    log("STEP 2: Three framing correlations")
    log("=" * 70)

    b1b_tau   = qconf_df["b1b_stability_tau"].values
    target    = qconf_df["target"].values
    q_frac    = qconf_df["q_frac_permutable"].values
    q_mean_s  = qconf_df["q_mean_sentences"].values

    tau_frac_vs_b1b, p1, ci1l, ci1h, n1 = kendall_tau_ci(q_frac, b1b_tau)
    tau_frac_vs_tgt, p2, ci2l, ci2h, n2 = kendall_tau_ci(q_frac, target)
    tau_msent_vs_b1b, p3, ci3l, ci3h, n3 = kendall_tau_ci(q_mean_s, b1b_tau)

    log(f"\n  (1) tau(q_frac_permutable, b1b_stability_tau) = {tau_frac_vs_b1b:+.4f}"
        f"  [{ci1l:+.4f}, {ci1h:+.4f}]  p={p1:.3e}")
    log(f"  (2) tau(q_frac_permutable, target_spearman)   = {tau_frac_vs_tgt:+.4f}"
        f"  [{ci2l:+.4f}, {ci2h:+.4f}]  p={p2:.3e}")
    log(f"  (3) tau(q_mean_sentences,  b1b_stability_tau) = {tau_msent_vs_b1b:+.4f}"
        f"  [{ci3l:+.4f}, {ci3h:+.4f}]  p={p3:.3e}")

    # Diagnose which case this is
    log("\n  Diagnosis:")
    frac_b1b_nontrivial = abs(tau_frac_vs_b1b) > 0.05
    frac_tgt_nontrivial = abs(tau_frac_vs_tgt) > 0.05
    if frac_b1b_nontrivial and frac_tgt_nontrivial:
        log("  CASE: Both correlations non-trivial => confound affects BOTH feature and target.")
        log("  => Potential spurious inflation of B1b's signal. Residualisation is needed.")
    elif frac_b1b_nontrivial and not frac_tgt_nontrivial:
        log("  CASE: Confound predicts B1b but NOT target => noise attenuation, not inflation.")
        log("  => B1b's raw tau is if anything attenuated by this confound.")
    else:
        log("  CASE: Confound does not substantially predict B1b. Artefact unlikely.")

    # ------------------------------------------------------------------ #
    # Step 3: Residualisation
    # ------------------------------------------------------------------ #
    log("\n" + "=" * 70)
    log("STEP 3: Residualisation on four nested confound sets")
    log("=" * 70)

    confound_sets = {
        "perm_only": ["q_frac_permutable"],
        "sent":      ["q_frac_permutable", "q_mean_sentences", "q_median_sentences"],
        "sent_tok":  ["q_frac_permutable", "q_mean_sentences", "q_median_sentences", "q_mean_tokens"],
        "full":      ["q_frac_permutable", "q_mean_sentences", "q_median_sentences",
                      "q_mean_tokens", "q_sd_sentences"],
    }

    res_rows = []

    log(f"\n  {'Set':<12} {'R²(conf→B1b)':>14} {'Resid τ':>10} {'Partial τ':>10} {'Drop':>8}")
    log("  " + "-" * 58)

    for cs_name, cols in confound_sets.items():
        X = qconf_df[cols].values
        y_b1b = qconf_df["b1b_stability_tau"].values
        y_tgt = qconf_df["target"].values

        reg = LinearRegression().fit(X, y_b1b)
        r2 = float(reg.score(X, y_b1b))
        resid = y_b1b - reg.predict(X)

        tau_resid, _, _, _, _ = kendall_tau_ci(resid, y_tgt)
        drop = RAW_TAU - tau_resid

        # Partial tau: controlling for all confounds
        cov_list = [qconf_df[c].values for c in cols]
        ptau = partial_kendall_tau(y_b1b, y_tgt, cov_list)

        log(f"  {cs_name:<12} {r2:>14.4f} {tau_resid:>10.4f} {ptau:>10.4f} {drop:>8.4f}")

        res_rows.append({
            "confound_set":          cs_name,
            "r2_confounds_on_b1b":   round(r2, 6),
            "residual_tau_vs_target": round(tau_resid, 6),
            "partial_tau":           round(ptau, 6),
            "drop_from_raw":         round(drop, 6),
        })

    res_df = pd.DataFrame(res_rows)
    res_df.to_csv(OUTPUT_DIR / "residualisation.csv", index=False, float_format="%.6f")
    log(f"\n  Saved: residualisation.csv")

    # Gate check
    full_resid_tau = res_df[res_df["confound_set"] == "full"]["residual_tau_vs_target"].values[0]
    if full_resid_tau < 0.25:
        log(f"\n  GATE: full-confound residual tau = {full_resid_tau:.4f} < 0.25")
        log("  B1b is inside the static plateau after controlling for passage structure.")
        log("  Section 4 headline needs rewriting. STOPPING before Step 4.")
        _write_report(log_lines, qconf_df, res_df, None, None, full_resid_tau, OUTPUT_DIR)
        return
    else:
        log(f"\n  GATE PASSED: full residual tau = {full_resid_tau:.4f} >= 0.25")

    # ------------------------------------------------------------------ #
    # Step 4: Stratified tau
    # ------------------------------------------------------------------ #
    log("\n" + "=" * 70)
    log("STEP 4: Stratified tau (terciles)")
    log("=" * 70)

    strat_rows = []

    for strat_var in ["q_frac_permutable", "q_median_sentences"]:
        vals = qconf_df[strat_var].values
        tercile_edges = np.percentile(vals, [33.3, 66.7])
        labels = []
        for v in vals:
            if v <= tercile_edges[0]:
                labels.append("low")
            elif v <= tercile_edges[1]:
                labels.append("mid")
            else:
                labels.append("high")
        labels = np.array(labels)

        log(f"\n  Strat variable: {strat_var}  (edges: {tercile_edges[0]:.3f}, {tercile_edges[1]:.3f})")
        log(f"  {'Tercile':<8} {'N':>5} {'tau':>8} {'ci_lo':>8} {'ci_hi':>8} {'mean_b1b':>10} {'range_b1b':>12}")
        log("  " + "-" * 65)

        for tercile in ["low", "mid", "high"]:
            mask = labels == tercile
            n_q = mask.sum()
            sub_b1b = qconf_df["b1b_stability_tau"].values[mask]
            sub_tgt = qconf_df["target"].values[mask]

            if n_q >= 5:
                tau_s, _, ci_lo, ci_hi, _ = kendall_tau_ci(sub_b1b, sub_tgt,
                                                            n_boot=N_BOOTSTRAP)
            else:
                tau_s, ci_lo, ci_hi = np.nan, np.nan, np.nan

            mean_b1b = float(np.mean(sub_b1b)) if n_q > 0 else np.nan
            range_b1b = f"[{sub_b1b.min():.3f},{sub_b1b.max():.3f}]" if n_q > 0 else "n/a"

            log(f"  {tercile:<8} {n_q:>5} {tau_s:>8.4f} {ci_lo:>8.4f} {ci_hi:>8.4f} "
                f"{mean_b1b:>10.4f} {range_b1b:>12}")
            strat_rows.append({
                "strata_variable": strat_var,
                "tercile":         tercile,
                "n_queries":       int(n_q),
                "tau":             round(tau_s, 6) if not np.isnan(tau_s) else np.nan,
                "ci_lo":           round(ci_lo, 6) if not np.isnan(ci_lo) else np.nan,
                "ci_hi":           round(ci_hi, 6) if not np.isnan(ci_hi) else np.nan,
                "mean_b1b":        round(mean_b1b, 6) if not np.isnan(mean_b1b) else np.nan,
            })

        log(f"  (N per tercile ~{int(mask.sum())}; bootstrap CIs are wide at this n)")

    strat_df = pd.DataFrame(strat_rows)
    strat_df.to_csv(OUTPUT_DIR / "stratified_tau.csv", index=False, float_format="%.6f")
    log(f"\n  Saved: stratified_tau.csv")

    # ------------------------------------------------------------------ #
    # Step 5: Restricted-pool version
    # ------------------------------------------------------------------ #
    log("\n" + "=" * 70)
    log("STEP 5: Restricted-pool stability tau (permutable passages only)")
    log("=" * 70)

    struct_lookup_perm = struct_df.set_index("passage_id")["is_permutable"].to_dict()

    restr_df = compute_restricted_tau(original_scores, runs, struct_lookup_perm, b1b_df)
    restr_merged = restr_df.merge(
        b1b_df[["query_id", "spearman"]],
        on="query_id", how="inner"
    )

    valid = restr_merged.dropna(subset=["b1b_tau_restricted"])
    n_valid   = len(valid)
    n_too_few = (restr_merged["b1b_tau_restricted"].isna()).sum()

    log(f"\n  Queries with >= 5 permutable passages: {n_valid}")
    log(f"  Queries excluded (< 5 permutable):     {n_too_few}")

    tau_restr, p_r, ci_lo_r, ci_hi_r, _ = kendall_tau_ci(
        valid["b1b_tau_restricted"].values,
        valid["spearman"].values,
    )
    log(f"\n  Restricted-pool tau vs target:  {tau_restr:+.4f}  [{ci_lo_r:+.4f}, {ci_hi_r:+.4f}]")
    log(f"  Raw b1b_stability_tau:          {RAW_TAU:+.4f}")
    log(f"  Attenuation (raw - restricted): {RAW_TAU - tau_restr:+.4f}")

    # Build restricted_pool.csv with summary row
    restr_out = restr_merged[["query_id", "n_permutable_passages", "b1b_tau_restricted"]].copy()
    summary_row = pd.DataFrame([{
        "query_id":               "SUMMARY",
        "n_permutable_passages":  n_valid,
        "b1b_tau_restricted":     round(tau_restr, 6),
    }])
    restr_out = pd.concat([restr_out, summary_row], ignore_index=True)
    restr_out.to_csv(OUTPUT_DIR / "restricted_pool.csv", index=False, float_format="%.6f")
    log(f"\n  Saved: restricted_pool.csv")

    # ------------------------------------------------------------------ #
    # Write REPORT.md
    # ------------------------------------------------------------------ #
    _write_report(log_lines, qconf_df, res_df, strat_df, restr_merged,
                  full_resid_tau, OUTPUT_DIR,
                  tau_restr=tau_restr, ci_lo_r=ci_lo_r, ci_hi_r=ci_hi_r,
                  tau_frac_vs_b1b=tau_frac_vs_b1b, tau_frac_vs_tgt=tau_frac_vs_tgt,
                  tau_msent_vs_b1b=tau_msent_vs_b1b,
                  pct_single=pct_single)

    log("\nAll done. Outputs in: " + str(OUTPUT_DIR))


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_report(log_lines, qconf_df, res_df, strat_df, restr_df,
                  full_resid_tau, out_dir, **kw):
    frac = qconf_df["q_frac_permutable"].values
    tau_frac_b1b = kw.get("tau_frac_vs_b1b", float("nan"))
    tau_frac_tgt = kw.get("tau_frac_vs_tgt", float("nan"))
    tau_msent_b1b = kw.get("tau_msent_vs_b1b", float("nan"))
    tau_restr  = kw.get("tau_restr", float("nan"))
    ci_lo_r    = kw.get("ci_lo_r", float("nan"))
    ci_hi_r    = kw.get("ci_hi_r", float("nan"))
    pct_single = kw.get("pct_single", float("nan"))

    lines = []
    lines.append("# T16: B1b Structural Control for Passage Length\n")
    lines.append("## Objective\n")
    lines.append(
        "Test whether `b1b_stability_tau` (Kendall tau = 0.338 vs per-query Spearman) "
        "is an artefact of passage length: single-sentence passages return unchanged "
        "under permutation, so their stability is zero-by-construction. A query whose "
        "pool is dominated by short passages looks artificially stable.\n"
    )

    lines.append("\n## Splitter verification\n")
    lines.append(
        f"spaCy model used: `en_core_web_md`, sentence boundary via `doc.sents` "
        f"(identical to `generate_b1_perturbations.py :: analyze_passage`).\n\n"
        f"Corpus-wide single-sentence rate: **{pct_single:.1f}%** "
        f"(target: 28.3%, tolerance ±2pp). "
        f"{'PASSED' if abs(pct_single - 28.3) <= 2.0 else 'FAILED — wrong splitter'}.\n"
    )

    lines.append("\n## Step 2: Three framing correlations\n")
    lines.append(
        f"| Pair | tau |\n|------|-----|\n"
        f"| q_frac_permutable vs b1b_stability_tau | {tau_frac_b1b:+.4f} |\n"
        f"| q_frac_permutable vs target_spearman   | {tau_frac_tgt:+.4f} |\n"
        f"| q_mean_sentences  vs b1b_stability_tau | {tau_msent_b1b:+.4f} |\n"
    )

    lines.append("\n## Step 3: Residualisation\n")
    lines.append(
        "Regress `b1b_stability_tau` on each confound set, score residual vs target.\n\n"
    )
    lines.append("| Confound set | R² on B1b | Residual tau | Partial tau | Drop from 0.338 |\n")
    lines.append("|---|---|---|---|---|\n")
    for _, row in res_df.iterrows():
        lines.append(
            f"| {row['confound_set']} | {row['r2_confounds_on_b1b']:.4f} | "
            f"{row['residual_tau_vs_target']:.4f} | {row['partial_tau']:.4f} | "
            f"{row['drop_from_raw']:.4f} |\n"
        )

    gate = "PASSED" if full_resid_tau >= 0.25 else "FAILED"
    lines.append(
        f"\nGate (full confound residual >= 0.25): **{gate}** "
        f"(residual tau = {full_resid_tau:.4f})\n"
    )

    if strat_df is not None:
        lines.append("\n## Step 4: Stratified tau\n")
        lines.append(
            "Note: ~100 queries per tercile — bootstrap CIs are wide and differences "
            "between strata should not be over-interpreted.\n\n"
        )
        lines.append("| Strat variable | Tercile | N | tau | CI |\n")
        lines.append("|---|---|---|---|---|\n")
        for _, row in strat_df.iterrows():
            ci_str = f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]"
            lines.append(
                f"| {row['strata_variable']} | {row['tercile']} | {int(row['n_queries'])} | "
                f"{row['tau']:.4f} | {ci_str} |\n"
            )

    if restr_df is not None:
        lines.append("\n## Step 5: Restricted-pool tau\n")
        n_valid = (~restr_df["b1b_tau_restricted"].isna()).sum()
        n_excl  = restr_df["b1b_tau_restricted"].isna().sum()
        lines.append(
            f"Recomputed `b1b_stability_tau` using only passages with n_sentences >= 2 "
            f"(discarding single-sentence passages from ranking comparison).\n\n"
            f"- Queries with >= 5 permutable passages: **{n_valid}**\n"
            f"- Excluded (< 5 permutable): {n_excl}\n"
            f"- Restricted-pool tau vs target: **{tau_restr:.4f}** "
            f"[{ci_lo_r:.4f}, {ci_hi_r:.4f}]\n"
            f"- Raw tau: 0.338\n"
            f"- Attenuation: {RAW_TAU - tau_restr:.4f}\n"
        )

    lines.append("\n## Conclusion\n")
    if full_resid_tau >= 0.25:
        lines.append(
            f"**B1b survives structural control.** After removing all variance in "
            f"`b1b_stability_tau` attributable to passage-length confounds (q_frac_permutable, "
            f"q_mean_sentences, q_median_sentences, q_mean_tokens, q_sd_sentences), "
            f"the residual Kendall tau against the per-query target is "
            f"**{full_resid_tau:.4f}** (drop from raw 0.338: "
            f"{res_df[res_df['confound_set']=='full']['drop_from_raw'].values[0]:.4f}). "
            f"The restricted-pool replication (permutable passages only) gives tau = "
            f"{tau_restr:.4f}. The Section 4 headline is not a passage-length artefact.\n"
        )
    else:
        lines.append(
            f"**B1b does NOT survive structural control.** Full-confound residual tau = "
            f"{full_resid_tau:.4f} < 0.25 threshold. The feature is inside the static "
            f"plateau once passage structure is removed. Section 4 headline needs rewriting.\n"
        )

    report_path = out_dir / "REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\nSaved: {report_path}")


if __name__ == "__main__":
    main()
