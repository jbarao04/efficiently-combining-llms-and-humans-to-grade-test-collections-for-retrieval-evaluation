# Build LARA confusion tables and calibration weights for audit
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent

YEAR_CFGS = {
    2019: {
        "qrels": BASE / "data_prep" / "data" / "trec-dl" / "2019" / "qrels.txt",
        "scores": BASE / "results" / "scoring" / "normal_scores" / "scores_v1.jsonl",
    },
    2020: {
        "qrels": BASE / "data_prep" / "data" / "trec-dl" / "2020" / "qrels.txt",
        "scores": BASE / "results" / "scoring" / "normal_scores" / "scores_v1.jsonl",
    },
    2021: {
        "qrels": BASE / "data_prep" / "data" / "trec-dl-v2" / "2021" / "qrels_dedup.txt",
        "scores": BASE / "results" / "scoring" / "normal_scores" / "scores_v2.jsonl",
    },
    2022: {
        "qrels": BASE / "data_prep" / "data" / "trec-dl-v2" / "2022" / "qrels_dedup.txt",
        "scores": BASE / "results" / "scoring" / "normal_scores" / "scores_v2.jsonl",
    },
    2023: {
        "qrels": BASE / "data_prep" / "data" / "trec-dl-v2" / "2023" / "qrels_dedup.txt",
        "scores": BASE / "results" / "scoring" / "normal_scores" / "scores_v2.jsonl",
    },
}

OUT_DIR = BASE / "results" / "thesis_verification" / "t13_lara_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRADES = [0, 1, 2, 3]
CONFIDENCE_RATE = 0.0001  # original LARA parameter
WARMUP = 50               # revised LARA warmup threshold


def load_qrels(path):
    qrels = defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qrels[parts[0]][parts[2]] = int(parts[3])
    return dict(qrels)


def load_scores(jsonl_path, year_queries):
    records = []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            qid = str(rec["query_id"])
            if qid not in year_queries:
                continue
            probs = {int(k): float(v) for k, v in rec["probs"].items()}
            records.append({
                "qid": qid,
                "pid": str(rec["passage_id"]),
                "llm_grade": int(rec["score"]),
                "max_prob": max(probs.values()),
                "probs": probs,
            })
    return records


def assign_bin(val, edges):
    if edges is None:
        return 0
    idx = np.searchsorted(edges[1:-1], val, side="right")
    return int(idx)


def make_conf_bins_equal_freq(values, n_bins):
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(values, quantiles)
    edges = np.unique(edges)
    if len(edges) < 2:
        return None
    return edges


all_confusion_rows = []
all_weight_rows = []

POOL_SIZES = {
    2019: 9000,  # approximate, will compute from data
    2020: 9000,
    2021: 9000,
    2022: 9000,
    2023: 9000,
}

for year, cfg in sorted(YEAR_CFGS.items()):
    print(f"\n=== Year {year} ===")
    human_qrels = load_qrels(cfg["qrels"])
    year_queries = set(human_qrels.keys())
    score_recs = load_scores(cfg["scores"], year_queries)

    # Build pair-level data
    rows = []
    for rec in score_recs:
        qid, pid = rec["qid"], rec["pid"]
        if qid in human_qrels and pid in human_qrels[qid]:
            rows.append({
                "qid": qid,
                "pid": pid,
                "llm_grade": rec["llm_grade"],
                "human_grade": human_qrels[qid][pid],
                "max_prob": rec["max_prob"],
            })
    df = pd.DataFrame(rows)
    n_pairs = len(df)
    POOL_SIZES[year] = n_pairs
    print(f"  {n_pairs} judged pairs with both human and LLM grades")

    # Grade distribution check
    print("  LLM grade distribution:")
    for g in GRADES:
        cnt = (df["llm_grade"] == g).sum()
        pct = 100 * cnt / n_pairs
        print(f"    grade {g}: {cnt:6d} ({pct:.1f}%)")
    print("  Human grade distribution:")
    for g in GRADES:
        cnt = (df["human_grade"] == g).sum()
        pct = 100 * cnt / n_pairs
        print(f"    grade {g}: {cnt:6d} ({pct:.1f}%)")

    # Determine binning (replicate compute_error_estimates.py logic)
    chosen_bins = None
    chosen_edges = None
    MIN_CELL = 20
    for n_bins in [3, 2, 0]:
        if n_bins == 0:
            chosen_bins = 0
            chosen_edges = None
            break
        edges = make_conf_bins_equal_freq(df["max_prob"].values, n_bins)
        if edges is None:
            continue
        df_tmp = df.copy()
        df_tmp["_bin"] = df_tmp["max_prob"].apply(lambda v: assign_bin(v, edges))
        cell_counts = df_tmp.groupby(["llm_grade", "_bin"]).size()
        if cell_counts.min() >= MIN_CELL:
            chosen_bins = n_bins
            chosen_edges = edges
            break
    print(f"  -> Using {chosen_bins} confidence bins")

    # Assign bins
    df["conf_bin"] = df["max_prob"].apply(lambda v: assign_bin(v, chosen_edges))

    # Build full confusion table (no bins) - for question 12/13
    print("\n  Full confusion table (llm_grade rows, human_grade cols), counts:")
    print("  LLM\\Human ", end="")
    for h in GRADES:
        print(f"  h={h}  ", end="")
    print()
    for l_grade in GRADES:
        sub = df[df["llm_grade"] == l_grade]
        row_total = len(sub)
        print(f"  l={l_grade}       ", end="")
        for h_grade in GRADES:
            cnt = (sub["human_grade"] == h_grade).sum()
            print(f"  {cnt:5d} ", end="")
            # Save to confusion_rows (bin=-1 means "all bins pooled")
            all_confusion_rows.append({
                "year": year,
                "confidence_bin": "all",
                "llm_grade": l_grade,
                "human_grade": h_grade,
                "count": int(cnt),
            })
        print(f"  | total={row_total}")

    # Build per-bin confusion table
    n_conf_bins = chosen_bins if chosen_bins and chosen_bins > 0 else 1
    for cb in range(n_conf_bins):
        sub_bin = df[df["conf_bin"] == cb]
        for l_grade in GRADES:
            sub = sub_bin[sub_bin["llm_grade"] == l_grade]
            for h_grade in GRADES:
                cnt = (sub["human_grade"] == h_grade).sum()
                all_confusion_rows.append({
                    "year": year,
                    "confidence_bin": str(cb),
                    "llm_grade": l_grade,
                    "human_grade": h_grade,
                    "count": int(cnt),
                })

    # Compute calibration weights for original LARA (confidence_rate=0.0001)
    # blend_weight = min(1.0, n_samples * confidence_rate)
    print(f"\n  Original LARA blend weights (confidence_rate={CONFIDENCE_RATE}):")
    budget_pcts = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
    for pct in budget_pcts:
        n_samples = int(round(pct * n_pairs))
        w = min(1.0, n_samples * CONFIDENCE_RATE)
        print(f"    budget={pct:.0%}, n_samples={n_samples}, blend_weight={w:.4f}")
        all_weight_rows.append({
            "year": year,
            "budget_pct": pct,
            "parameter_name": "confidence_rate_blend_weight",
            "value": round(w, 6),
            "n_samples": n_samples,
        })
        # Revised LARA: is_active = (n_samples >= WARMUP)
        is_active_revised = (n_samples >= WARMUP)
        all_weight_rows.append({
            "year": year,
            "budget_pct": pct,
            "parameter_name": "revised_calibrator_is_active",
            "value": float(is_active_revised),
            "n_samples": n_samples,
        })

# Grade-1 focused analysis (question 13)
print("\n\n=== GRADE-1 CELL FOCUS (Q13) ===")
pooled = defaultdict(int)
for r in all_confusion_rows:
    if r["confidence_bin"] == "all":
        pooled[(r["llm_grade"], r["human_grade"])] += r["count"]

print("Pooled across all years (llm=1 row):")
l1_total = sum(pooled[(1, h)] for h in GRADES)
for h in GRADES:
    cnt = pooled[(1, h)]
    pct = 100 * cnt / l1_total if l1_total > 0 else 0
    print(f"  llm=1, human={h}: {cnt:6d} ({pct:.1f}%)")

print("\nPooled across all years (human=1 col):")
h1_total = sum(pooled[(l, 1)] for l in GRADES)
for l in GRADES:
    cnt = pooled[(l, 1)]
    pct = 100 * cnt / h1_total if h1_total > 0 else 0
    print(f"  llm={l}, human=1: {cnt:6d} ({pct:.1f}%)")

# LLM grade=1 prevalence vs human grade=1 prevalence
all_pairs_total = sum(pooled.values())
llm1_total = sum(pooled[(1, h)] for h in GRADES)
human1_total = sum(pooled[(l, 1)] for l in GRADES)
print(f"\nTotal pairs (all years): {all_pairs_total}")
print(f"LLM assigns grade 1: {llm1_total} ({100*llm1_total/all_pairs_total:.1f}%)")
print(f"Human assigns grade 1: {human1_total} ({100*human1_total/all_pairs_total:.1f}%)")
print(f"Cell (llm=1, human=1): {pooled[(1,1)]} ({100*pooled[(1,1)]/all_pairs_total:.1f}%)")

# Save outputs
conf_df = pd.DataFrame(all_confusion_rows)
conf_df.to_csv(OUT_DIR / "confusion_counts.csv", index=False)
print(f"\nSaved: {OUT_DIR}/confusion_counts.csv")

weights_df = pd.DataFrame(all_weight_rows)
weights_df.to_csv(OUT_DIR / "calibration_weights.csv", index=False)
print(f"Saved: {OUT_DIR}/calibration_weights.csv")

print("\nDone.")
