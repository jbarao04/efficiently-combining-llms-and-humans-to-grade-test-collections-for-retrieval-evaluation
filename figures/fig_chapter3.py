import json
import sys
sys.path.insert(0, '.')
import plot_style                          # applies apply_theme() on import
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import pearsonr

# --- Load data ---
def load_qrels(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, pid, rel = parts[0], parts[2], int(parts[3])
                qrels.setdefault(qid, {})[pid] = rel
    return qrels

def load_scores(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records

qrels = {}
for p in ["data_prep/data/trec-dl/merged/qrels_merged.txt",
          "data_prep/data/trec-dl-v2/merged/qrels_merged.txt"]:
    q = load_qrels(p)
    for qid, pids in q.items():
        qrels.setdefault(qid, {}).update(pids)

all_scores = []
for p in ["results/scoring/normal_scores/scores_v1.jsonl",
          "results/scoring/normal_scores/scores_v2.jsonl"]:
    all_scores.extend(load_scores(p))

# --- Build intersection pairs ---
human_grades = []
llm_grades = []
per_query = defaultdict(lambda: {"human": [], "llm": []})

for rec in all_scores:
    qid = str(rec["query_id"])
    pid = str(rec["passage_id"])
    if qid in qrels and pid in qrels[qid]:
        h = qrels[qid][pid]
        l = int(rec["score"])
        human_grades.append(h)
        llm_grades.append(l)
        per_query[qid]["human"].append(h)
        per_query[qid]["llm"].append(l)

total = len(human_grades)
print(f"Total intersection pairs: {total}")

# ── Figure 1: Grade distribution ────────────────────────────────────────────
human_counts = np.zeros(4)
llm_counts   = np.zeros(4)
for g in human_grades:
    human_counts[g] += 1
for g in llm_grades:
    llm_counts[g] += 1

human_pct = human_counts / total * 100
llm_pct   = llm_counts   / total * 100

print("\nGrade distribution (%):")
for g in range(4):
    print(f"  Grade {g}: Human={human_pct[g]:.1f}%, LLM={llm_pct[g]:.1f}%")

fig, ax = plt.subplots(figsize=(5, 3.5))
x = np.arange(4)
w = 0.35

# Okabe-Ito colours — match the other thesis figures
c_human = '#8FA8C8'                # muted slate blue — matches histogram bars
c_llm   = plot_style.OI_ORANGE    # warm gold, same as Leverage in Ch5/6 plots

bars_h = ax.bar(x - w/2, human_pct, w, color=c_human, label='Human (NIST)')
bars_l = ax.bar(x + w/2, llm_pct,   w, color=c_llm,   label='LLM (Llama-3.1-8B)')

for bar in bars_h:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9,
            color=plot_style._GREY30)
for bar in bars_l:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9,
            color=plot_style._GREY30)

ax.set_xlabel('Relevance grade', fontsize=11)
ax.set_ylabel('Percentage of pairs', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(['0', '1', '2', '3'], fontsize=10)
ax.tick_params(axis='y', labelsize=10)

# Headroom: 35% above tallest bar so labels + legend fit
ymax = max(human_pct.max(), llm_pct.max())
ax.set_ylim(0, ymax * 1.35)

# Legend in upper-right: grade-3 bars are small (7% / 13%), plenty of room
ax.legend(fontsize=10, loc='upper right', frameon=False)

plt.savefig('fig_grade_distribution.pdf', bbox_inches='tight')
plt.close()
print("\nSaved fig_grade_distribution.pdf")

# ── Figure 2: Per-query Pearson histogram ────────────────────────────────────
pearson_vals = []
for qid, data in per_query.items():
    if len(data["human"]) < 5:
        continue
    h = np.array(data["human"])
    l = np.array(data["llm"])
    if np.std(h) == 0 or np.std(l) == 0:
        continue
    r, _ = pearsonr(l, h)
    pearson_vals.append(r)

pearson_vals = np.array(pearson_vals)
mean_r = np.mean(pearson_vals)
std_r  = np.std(pearson_vals)
print(f"\nPer-query Pearson r:")
print(f"  N queries: {len(pearson_vals)}")
print(f"  Mean: {mean_r:.4f}")
print(f"  Std:  {std_r:.4f}")
print(f"  Min:  {np.min(pearson_vals):.4f}")
print(f"  Max:  {np.max(pearson_vals):.4f}")

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.hist(pearson_vals, bins=25, range=(-0.2, 0.9),
        color='#8FA8C8', edgecolor=plot_style._GREY30, linewidth=0.5)
ax.axvline(mean_r, color=plot_style.OI_VERMILLION, linestyle='--', linewidth=1.2)

# Draw first to get true ylim, then annotate to the LEFT of the mean line
fig.canvas.draw()
ymax = ax.get_ylim()[1]

# Place annotation to the LEFT of the dashed line, right-aligned,
# at 88% height — the left tail (-0.2 to ~0.2) is sparsely populated
ax.text(mean_r - 0.03, ymax * 0.88,
        f'mean = {mean_r:.2f}',
        ha='right', va='top',
        fontsize=10, color=plot_style.OI_VERMILLION)

ax.set_xlabel('Per-query Pearson r (LLM vs. human)', fontsize=11)
ax.set_ylabel('Number of queries', fontsize=11)
ax.tick_params(axis='both', labelsize=10)

plt.savefig('fig_reliability_histogram.pdf', bbox_inches='tight')
plt.close()
print("Saved fig_reliability_histogram.pdf")
