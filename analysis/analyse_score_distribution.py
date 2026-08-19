# Score distribution analysis: tests whether LLM output features predict per-query reliability

import json
import pathlib
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.random.seed(42)

ROOT = pathlib.Path(__file__).resolve().parent.parent

# ── Paths ────────────────────────────────────────────────────────────────
SCORE_FILES = [
    ROOT / "results/scoring/normal_scores/scores_v1.jsonl",
    ROOT / "results/scoring/normal_scores/scores_v2.jsonl",
]

QRELS_FILES = [
    ROOT / "data_prep/data/trec-dl/2019/qrels.txt",
    ROOT / "data_prep/data/trec-dl/2020/qrels.txt",
    ROOT / "data_prep/data/trec-dl-v2/2021/qrels_dedup.txt",
    ROOT / "data_prep/data/trec-dl-v2/2022/qrels_dedup.txt",
    ROOT / "data_prep/data/trec-dl-v2/2023/qrels_dedup.txt",
]

QUERY_FILES = [
    ROOT / "data_prep/data/trec-dl/2019/queries.tsv",
    ROOT / "data_prep/data/trec-dl/2020/queries.tsv",
    ROOT / "data_prep/data/trec-dl-v2/2021/queries.tsv",
    ROOT / "data_prep/data/trec-dl-v2/2022/queries.tsv",
    ROOT / "data_prep/data/trec-dl-v2/2023/queries.tsv",
]

PASSAGE_FILES = [
    ROOT / "data_prep/data/trec-dl/judged_passages/judged_passages.jsonl",
    ROOT / "data_prep/data/trec-dl-v2/judged_passages/judged_passages.jsonl",
]

FEATURES_FILE = ROOT / "results/level2/per_query_features_v4.csv"
B1B_FILE = ROOT / "results/level2/b1b_features.csv"

N_BOOT = 10_000


# ── Loading ──────────────────────────────────────────────────────────────

def load_scores():
    records = defaultdict(lambda: {"probs_sum": np.zeros(4), "count": 0})
    for f in SCORE_FILES:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                qid, pid = str(r["query_id"]), str(r["passage_id"])
                probs = r["probs"]
                p = np.array([probs[str(g)] for g in range(4)])
                key = (qid, pid)
                records[key]["probs_sum"] += p
                records[key]["count"] += 1

    rows = []
    for (qid, pid), v in records.items():
        avg_p = v["probs_sum"] / v["count"]
        es = sum(g * avg_p[g] for g in range(4))
        score = int(np.argmax(avg_p))
        rows.append({"query_id": qid, "passage_id": pid,
                      "score": score, "es": es,
                      "p0": avg_p[0], "p1": avg_p[1], "p2": avg_p[2], "p3": avg_p[3]})
    return pd.DataFrame(rows)


def load_qrels():
    rows = []
    for f in QRELS_FILES:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                qid, _, pid, grade = parts[0], parts[1], parts[2], int(parts[3])
                rows.append({"query_id": str(qid), "passage_id": str(pid), "grade": grade})
    return pd.DataFrame(rows)


def load_queries():
    rows = []
    for f in QUERY_FILES:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    rows.append({"query_id": str(parts[0]), "query_text": parts[1]})
    return pd.DataFrame(rows).drop_duplicates("query_id")


def load_passages():
    rows = []
    for f in PASSAGE_FILES:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                rows.append({"passage_id": str(r["pid"]), "passage": r["passage"]})
    return pd.DataFrame(rows).drop_duplicates("passage_id")


def load_features():
    feat = pd.read_csv(FEATURES_FILE)
    feat["query_id"] = feat["query_id"].astype(str)
    b1b = pd.read_csv(B1B_FILE)
    b1b["query_id"] = b1b["query_id"].astype(str)
    merged = feat[["query_id", "spearman", "fisher_ratio"]].merge(
        b1b[["query_id", "b1b_stability_tau"]], on="query_id", how="inner"
    )
    return merged


# ── Utilities ────────────────────────────────────────────────────────────

def kendall_with_ci(x, y, n_boot=N_BOOT):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = np.array(x)[mask], np.array(y)[mask]
    n = len(x)
    if n < 5:
        return np.nan, (np.nan, np.nan), np.nan, n
    tau, p = stats.kendalltau(x, y)
    rng = np.random.RandomState(42)
    taus = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        t, _ = stats.kendalltau(x[idx], y[idx])
        taus[i] = t
    lo, hi = np.nanpercentile(taus, [2.5, 97.5])
    return tau, (lo, hi), p, n


def partial_corr(x, y, controls):
    mask = np.isfinite(x) & np.isfinite(y)
    for c in controls:
        mask &= np.isfinite(c)
    x, y = np.array(x)[mask], np.array(y)[mask]
    cs = np.column_stack([np.array(c)[mask] for c in controls])
    n = len(x)
    if n < 5:
        return np.nan, np.nan, n

    from numpy.linalg import lstsq
    A = np.column_stack([cs, np.ones(n)])
    rx = x - A @ lstsq(A, x, rcond=None)[0]
    ry = y - A @ lstsq(A, y, rcond=None)[0]
    r, p = stats.pearsonr(rx, ry)
    return r, p, n


def norm01(x):
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    scores_df = load_scores()
    qrels_df = load_qrels()
    features_df = load_features()
    queries_df = load_queries()
    passages_df = load_passages()

    merged = scores_df.merge(qrels_df, on=["query_id", "passage_id"], how="inner")
    print(f"  Scores: {len(scores_df):,} rows")
    print(f"  Qrels: {len(qrels_df):,} rows")
    print(f"  Merged (score+qrel): {len(merged):,} rows")
    print(f"  Queries with features: {len(features_df)}")
    print(f"  Query texts: {len(queries_df)}")
    print(f"  Passage texts: {len(passages_df):,}")

    query_ids = sorted(set(features_df["query_id"]))
    print(f"  Confirmed query count: {len(query_ids)}")

    spearman = features_df.set_index("query_id")["spearman"]
    fisher = features_df.set_index("query_id")["fisher_ratio"]
    b1b = features_df.set_index("query_id")["b1b_stability_tau"]

    scored_by_q = {qid: g for qid, g in scores_df.groupby("query_id")}

    # ═══════════════════════════════════════════════════════════════════
    # HYPOTHESIS 1: Score-2 internal structure
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("HYPOTHESIS 1: Score-2 internal structure")
    print("="*70)

    h1_rows = []
    h1_skip = 0
    for qid in query_ids:
        if qid not in scored_by_q:
            h1_skip += 1
            continue
        qdata = scored_by_q[qid]
        n_total = len(qdata)
        row = {"query_id": qid}

        for target_score, prefix in [(2, "score2"), (0, "score0")]:
            es_vals = qdata.loc[qdata["score"] == target_score, "es"].values
            n_bucket = len(es_vals)
            row[f"{prefix}_count"] = n_bucket
            row[f"{prefix}_frac"] = n_bucket / n_total

            if n_bucket >= 10:
                row[f"{prefix}_es_std"] = np.std(es_vals, ddof=1)
                q25, q75 = np.percentile(es_vals, [25, 75])
                row[f"{prefix}_es_iqr"] = q75 - q25
                row[f"{prefix}_es_range"] = es_vals.max() - es_vals.min()
                row[f"{prefix}_es_skew"] = stats.skew(es_vals)
            else:
                row[f"{prefix}_es_std"] = np.nan
                row[f"{prefix}_es_iqr"] = np.nan
                row[f"{prefix}_es_range"] = np.nan
                row[f"{prefix}_es_skew"] = np.nan

        # Score-2 high fraction
        s2_es = qdata.loc[qdata["score"] == 2, "es"].values
        row["score2_high_frac"] = np.mean(s2_es > 2.0) if len(s2_es) >= 10 else np.nan

        h1_rows.append(row)

    h1 = pd.DataFrame(h1_rows).set_index("query_id")
    h1_cols = [c for c in h1.columns if c.startswith("score")]
    print(f"  H1: {len(h1)} queries computed, {h1_skip} skipped")
    print(f"  Score-2 queries with >=10 passages: {h1['score2_es_std'].notna().sum()}")
    print(f"  Score-0 queries with >=10 passages: {h1['score0_es_std'].notna().sum()}")

    # ═══════════════════════════════════════════════════════════════════
    # HYPOTHESIS 2: Top-passage confidence gap
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("HYPOTHESIS 2: Top-passage confidence gap")
    print("="*70)

    h2_rows = []
    for qid in query_ids:
        if qid not in scored_by_q:
            continue
        qdata = scored_by_q[qid]
        es_sorted = np.sort(qdata["es"].values)[::-1]
        n = len(es_sorted)

        row = {"query_id": qid}
        row["top5_mean_es"] = np.mean(es_sorted[:5]) if n >= 5 else np.nan
        row["top10_mean_es"] = np.mean(es_sorted[:10]) if n >= 10 else np.nan
        row["top5_top20_gap"] = (np.mean(es_sorted[:5]) - np.mean(es_sorted[5:20])) if n >= 20 else np.nan
        row["top10_rest_gap"] = (np.mean(es_sorted[:10]) - np.mean(es_sorted[10:])) if n >= 11 else np.nan
        row["max_es"] = es_sorted[0] if n > 0 else np.nan
        row["top5_es_std"] = np.std(es_sorted[:5], ddof=1) if n >= 5 else np.nan

        s3_es = qdata.loc[qdata["score"] == 3, "es"].values
        s2_es = qdata.loc[qdata["score"] == 2, "es"].values
        row["score3_mean_es"] = np.mean(s3_es) if len(s3_es) > 0 else np.nan
        row["score3_score2_gap"] = (np.mean(s3_es) - np.mean(s2_es)) if (len(s3_es) > 0 and len(s2_es) > 0) else np.nan

        h2_rows.append(row)

    h2 = pd.DataFrame(h2_rows).set_index("query_id")
    print(f"  H2: {len(h2)} queries computed")
    print(f"  Queries with score-3 passages: {h2['score3_mean_es'].notna().sum()}")

    # ═══════════════════════════════════════════════════════════════════
    # HYPOTHESIS 3: Textual cue correlation (vectorized)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("HYPOTHESIS 3: Judge's self-consistency with textual cues")
    print("="*70)

    # Build lookup dicts
    passage_text = passages_df.set_index("passage_id")["passage"].to_dict()
    query_text_map = queries_df.set_index("query_id")["query_text"].to_dict()

    # Pre-compute text features for all passages in scores_df
    print("  Computing text features for all passages...")
    sys.stdout.flush()

    # Only compute for passages we have text for
    pids_in_scores = scores_df["passage_id"].unique()
    text_feat_rows = []
    missing_text = 0
    for pid in pids_in_scores:
        ptxt = passage_text.get(pid)
        if ptxt is None:
            missing_text += 1
            continue
        tokens = ptxt.split()
        has_digit = 1 if any(c.isdigit() for c in ptxt) else 0
        dot_pos = ptxt.find(". ")
        if 0 < dot_pos < 200:
            first_sent = ptxt[:dot_pos]
        else:
            first_sent = ptxt[:100]
        text_feat_rows.append({
            "passage_id": pid,
            "passage_length": len(tokens),
            "has_digit": has_digit,
            "first_sentence_length": len(first_sent.split()),
            "passage_lower_tokens": set(ptxt.lower().split()),
        })

    print(f"  Text features computed for {len(text_feat_rows):,} passages ({missing_text} missing text)")

    # Build a dict for fast lookup
    text_feat_dict = {r["passage_id"]: r for r in text_feat_rows}

    h3_rows = []
    h3_skip = 0
    for qid in query_ids:
        if qid not in scored_by_q:
            continue
        qt = query_text_map.get(qid, "")
        if not qt:
            h3_skip += 1
            continue
        query_terms = set(qt.lower().split())
        n_qt = len(query_terms)

        qdata = scored_by_q[qid]
        lengths = []
        overlaps = []
        digits = []
        fst_lens = []
        es_vals = []

        for pid, es in zip(qdata["passage_id"].values, qdata["es"].values):
            tf = text_feat_dict.get(pid)
            if tf is None:
                continue
            lengths.append(tf["passage_length"])
            overlap = len(query_terms & tf["passage_lower_tokens"]) / n_qt if n_qt > 0 else 0
            overlaps.append(overlap)
            digits.append(tf["has_digit"])
            fst_lens.append(tf["first_sentence_length"])
            es_vals.append(es)

        if len(es_vals) < 10:
            h3_skip += 1
            continue

        lengths = np.array(lengths, dtype=float)
        overlaps = np.array(overlaps, dtype=float)
        digits = np.array(digits, dtype=float)
        fst_lens = np.array(fst_lens, dtype=float)
        es_vals = np.array(es_vals)

        composite = norm01(lengths) + norm01(overlaps) + norm01(digits) + norm01(fst_lens)

        row = {"query_id": qid}
        if np.std(composite) > 1e-12 and np.std(es_vals) > 1e-12:
            row["text_score_r"], _ = stats.pearsonr(composite, es_vals)
        else:
            row["text_score_r"] = np.nan

        if np.std(lengths) > 1e-12:
            row["length_es_r"], _ = stats.pearsonr(lengths, es_vals)
        else:
            row["length_es_r"] = np.nan

        if np.std(overlaps) > 1e-12:
            row["overlap_es_r"], _ = stats.pearsonr(overlaps, es_vals)
        else:
            row["overlap_es_r"] = np.nan

        if np.std(digits) > 1e-12:
            row["digit_es_r"], _ = stats.pearsonr(digits, es_vals)
        else:
            row["digit_es_r"] = np.nan

        h3_rows.append(row)

    h3 = pd.DataFrame(h3_rows).set_index("query_id")
    print(f"  H3: {len(h3)} queries computed, {h3_skip} skipped")

    # ═══════════════════════════════════════════════════════════════════
    # HYPOTHESIS 4 (bonus): Low-E[s] bimodality
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("HYPOTHESIS 4 (bonus): Bottom-of-ranking E[s] bimodality")
    print("="*70)

    h4_rows = []
    h4_skip = 0
    for qid in query_ids:
        if qid not in scored_by_q:
            continue
        qdata = scored_by_q[qid]
        es_low = qdata.loc[qdata["score"].isin([0, 1]), "es"].values

        if len(es_low) < 10:
            h4_skip += 1
            continue

        row = {"query_id": qid}
        row["low_es_range"] = es_low.max() - es_low.min()
        row["low_es_std"] = np.std(es_low, ddof=1)
        q25, q75 = np.percentile(es_low, [25, 75])
        row["low_es_iqr"] = q75 - q25
        med = np.median(es_low)
        above = es_low[es_low > med]
        below = es_low[es_low <= med]
        row["low_top_gap"] = (np.mean(above) - np.mean(below)) if len(above) > 0 and len(below) > 0 else np.nan

        if len(es_low) >= 4:
            km = KMeans(n_clusters=2, n_init=10, random_state=42)
            km.fit(es_low.reshape(-1, 1))
            centers = sorted(km.cluster_centers_.ravel())
            row["low_cluster_sep"] = centers[1] - centers[0]
        else:
            row["low_cluster_sep"] = np.nan

        h4_rows.append(row)

    h4 = pd.DataFrame(h4_rows).set_index("query_id")
    print(f"  H4: {len(h4)} queries computed, {h4_skip} skipped")

    # ═══════════════════════════════════════════════════════════════════
    # Combine all features and compute correlations
    # ═══════════════════════════════════════════════════════════════════
    all_new = h1.join(h2, how="outer").join(h3, how="outer").join(h4, how="outer")
    new_feature_cols = [c for c in all_new.columns if c != "query_id"]

    common_qids = sorted(set(all_new.index) & set(spearman.index))
    print(f"\n  Queries in common for correlation: {len(common_qids)}")

    print("\n  Computing Kendall tau with bootstrap CIs (this takes a minute)...")
    sys.stdout.flush()

    results = []
    for i, col in enumerate(new_feature_cols):
        vals = all_new.loc[common_qids, col].values
        sp = spearman.loc[common_qids].values
        tau, ci, p, n = kendall_with_ci(vals, sp)
        results.append({"feature": col, "tau": tau, "ci_lo": ci[0], "ci_hi": ci[1], "p": p, "n": n})
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(new_feature_cols)} features done...")
            sys.stdout.flush()

    results_df = pd.DataFrame(results).sort_values("tau", key=abs, ascending=False)

    print("\n" + "="*70)
    print("=== ALL NEW FEATURES: Kendall tau with per-query Spearman ===")
    print("="*70)
    print(f"\n{'Feature':<28s} {'Kendall tau':>11s}  {'95% CI':>20s}  {'p-value':>10s}  {'n':>5s}")
    print("-" * 80)
    for _, r in results_df.iterrows():
        ci_str = f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
        print(f"{r['feature']:<28s} {r['tau']:+.4f}       {ci_str:>20s}  {r['p']:.2e}  {int(r['n']):>5d}")

    # ═══════════════════════════════════════════════════════════════════
    # Comparison to existing features
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("=== COMPARISON TO EXISTING FEATURES ===")
    print("="*70)

    sp_arr = spearman.loc[common_qids].values
    fi_arr = fisher.loc[common_qids].values
    b1b_arr = b1b.loc[common_qids].values

    fisher_tau, fisher_ci, fisher_p, fisher_n = kendall_with_ci(fi_arr, sp_arr)
    b1b_tau, b1b_ci, b1b_p, b1b_n = kendall_with_ci(b1b_arr, sp_arr)

    print(f"\n{'Feature':<28s} {'Tau':>7s}  {'Unique > Fisher?':>22s}  {'Unique > B1b?':>22s}")
    print("-" * 85)

    top5 = results_df.head(5)
    for _, r in top5.iterrows():
        col = r["feature"]
        vals = all_new.loc[common_qids, col].values
        pr_f, pp_f, _ = partial_corr(vals, sp_arr, [fi_arr])
        pr_b, pp_b, _ = partial_corr(vals, sp_arr, [b1b_arr])
        f_str = f"r={pr_f:+.3f}, p={pp_f:.3f}" if not np.isnan(pr_f) else "N/A"
        b_str = f"r={pr_b:+.3f}, p={pp_b:.3f}" if not np.isnan(pr_b) else "N/A"
        print(f"{col:<28s} {r['tau']:+.4f}   {f_str:>22s}  {b_str:>22s}")

    print(f"{'fisher_ratio':<28s} {fisher_tau:+.4f}   {'---':>22s}  {'':>22s}")
    print(f"{'b1b_stability_tau':<28s} {b1b_tau:+.4f}   {'':>22s}  {'---':>22s}")

    # ═══════════════════════════════════════════════════════════════════
    # Partial correlations for top features (|tau| > 0.15)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("=== PARTIAL CORRELATIONS FOR TOP NEW FEATURES (|tau| > 0.15) ===")
    print("="*70)

    strong = results_df[results_df["tau"].abs() > 0.15]
    if len(strong) == 0:
        print("  No features with |tau| > 0.15.")
    else:
        print(f"\n{'Feature':<28s} {'pr|Fisher':>12s} {'pr|B1b':>12s} {'pr|Both':>12s} {'r(F)':>8s} {'r(B1b)':>8s}")
        print("-" * 85)
        for _, r in strong.iterrows():
            col = r["feature"]
            vals = all_new.loc[common_qids, col].values

            pr_f, pp_f, _ = partial_corr(vals, sp_arr, [fi_arr])
            pr_b, pp_b, _ = partial_corr(vals, sp_arr, [b1b_arr])
            pr_fb, pp_fb, _ = partial_corr(vals, sp_arr, [fi_arr, b1b_arr])

            mask = np.isfinite(vals)
            r_fisher = stats.pearsonr(vals[mask], fi_arr[mask])[0] if mask.sum() > 5 else np.nan
            r_b1b = stats.pearsonr(vals[mask], b1b_arr[mask])[0] if mask.sum() > 5 else np.nan

            def fmt_pr(r_val, p_val):
                if np.isnan(r_val):
                    return "N/A"
                sig = "*" if p_val < 0.05 else ""
                return f"{r_val:+.3f}{sig}"

            print(f"{col:<28s} {fmt_pr(pr_f, pp_f):>12s} {fmt_pr(pr_b, pp_b):>12s} {fmt_pr(pr_fb, pp_fb):>12s} {r_fisher:+.3f}   {r_b1b:+.3f}")

    # ═══════════════════════════════════════════════════════════════════
    # Key question
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("=== KEY QUESTION ===")
    print("="*70)

    any_significant = False
    for _, r in strong.iterrows():
        col = r["feature"]
        vals = all_new.loc[common_qids, col].values
        pr_fb, pp_fb, _ = partial_corr(vals, sp_arr, [fi_arr, b1b_arr])
        if not np.isnan(pp_fb) and pp_fb < 0.05:
            any_significant = True
            print(f"  YES: {col} has partial r = {pr_fb:+.3f} (p = {pp_fb:.4f}) after controlling for Fisher + B1b")

    if not any_significant:
        print("  No new features have significant partial correlation with Spearman")
        print("  after controlling for BOTH Fisher and B1b.")
        print("  => Fisher + B1b already capture everything in the output distribution.")

    print("\nDo any new features have significant partial correlation with Spearman")
    print("after controlling for BOTH Fisher and B1b?")
    if any_significant:
        print("=> YES: there is a third axis worth pursuing.")
    else:
        print("=> NO: Fisher + B1b already capture everything in the output distribution.")


if __name__ == "__main__":
    main()
