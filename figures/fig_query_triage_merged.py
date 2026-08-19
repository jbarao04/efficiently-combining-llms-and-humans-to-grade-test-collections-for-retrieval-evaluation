# Figure: merged query triage curves with random band and oracle overlays

import os
import sys
sys.path.insert(0, ".")
import plot_style  # applies theme on import

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────
T12    = "results/thesis_verification/t12_resampling"
QL_CSV = f"{T12}/ql_curves_full.csv"
AC_CSV = f"{T12}/area_ci.csv"
YEARS  = [2019, 2020, 2021, 2022, 2023]

# ── Colours ────────────────────────────────────────────────────────────
COL_RAND_LINE = "#888888"
COL_RAND_FILL = "#CCCCCC"
COL_REL       = plot_style.OI_PINK
COL_DMG       = plot_style.OI_ORANGE

ALPHA_BAND = 0.50
LW         = 1.6

# ── Load data ───────────────────────────────────────────────────────────
ql = pd.read_csv(QL_CSV)
ac = pd.read_csv(AC_CSV)

print("=" * 66)
print("fig_query_triage_curves_merged.pdf - numbers used")
print("=" * 66)
print(f"\nSource: {QL_CSV}")
print(f"Policies: {sorted(ql.policy.unique().tolist())}")
print(f"Years:    {sorted(ql.year.unique().tolist())}")
print(f"budget_pct range: {ql.budget_pct.min():.0f}–{ql.budget_pct.max():.0f}  "
      f"(x-axis = budget_pct / 100)")

for pol in ("random_mean", "random_lo", "random_hi",
            "reliability_oracle", "damage_oracle"):
    assert pol in ql.policy.values, f"MISSING policy '{pol}' in {QL_CSV}"
print("Policy presence check: OK")

# ── Average curves across years (unweighted) ────────────────────────────
# budget_pct is 0–100 integer grid; all years share the same grid.
def avg(policy, metric):
    sub = ql[ql.policy == policy].groupby("budget_pct")[metric].mean()
    return sub.sort_index().values

budget = ql.budget_pct.drop_duplicates().sort_values().values / 100.0  # 0→1

print("\nAll-LLM endpoint (budget_pct = 0), averaged across years:")
for pol in ("random_mean", "damage_oracle", "reliability_oracle"):
    v20  = avg(pol, "tau_at_20")[0]
    vall = avg(pol, "tau_all")[0]
    print(f"  {pol:22s}  tau@20={v20:.4f}  tau_all={vall:.4f}")

# ── Area numbers from area_ci.csv ───────────────────────────────────────
print("\n" + "=" * 66)
print("Area above random (from area_ci.csv) - for caption / S5.3 table")
print("=" * 66)
for metric, mlabel in [("tau_at_20", "tau@20"), ("tau_all", "tau_all")]:
    sub = ac[ac.metric == metric]
    print(f"\n  {mlabel}")
    print(f"  {'Year':>4}  {'Rel oracle':>10}  {'[lo, hi]':>22}  "
          f"{'Dmg oracle':>10}  {'[lo, hi]':>22}")
    for y in YEARS:
        rr = sub[(sub.year == y) & (sub.policy == "reliability_oracle")].iloc[0]
        dr = sub[(sub.year == y) & (sub.policy == "damage_oracle")].iloc[0]
        print(f"  {y}  {rr.area_vs_random:+.4f}  "
              f"[{rr.lo_2p5:+.4f}, {rr.hi_97p5:+.4f}]  "
              f"{dr.area_vs_random:+.4f}  "
              f"[{dr.lo_2p5:+.4f}, {dr.hi_97p5:+.4f}]")
    all_zero_rel = all(
        ac[(ac.year == y) & (ac.policy == "reliability_oracle") & (ac.metric == metric)
           ].iloc[0].lo_2p5 < 0
        and
        ac[(ac.year == y) & (ac.policy == "reliability_oracle") & (ac.metric == metric)
           ].iloc[0].hi_97p5 > 0
        for y in YEARS)
    all_zero_dmg = all(
        ac[(ac.year == y) & (ac.policy == "damage_oracle") & (ac.metric == metric)
           ].iloc[0].lo_2p5 < 0
        and
        ac[(ac.year == y) & (ac.policy == "damage_oracle") & (ac.metric == metric)
           ].iloc[0].hi_97p5 > 0
        for y in YEARS)
    print(f"  All 95% CIs include zero - rel: {all_zero_rel}, dmg: {all_zero_dmg}")

# ── LaTeX table snippet ─────────────────────────────────────────────────
print("\n" + "=" * 66)
print("CORRECTED S5.3 LaTeX row - paste into Overleaf")
print("(Reliability oracle only; b1b/Fisher are not in t12_resampling output")
print(" and remain as originally stated, flagged with a comment.)")
print("=" * 66)

sub20 = ac[ac.metric == "tau_at_20"]
rel_vals = {
    y: sub20[(sub20.year == y) & (sub20.policy == "reliability_oracle")
             ].iloc[0].area_vs_random
    for y in YEARS
}
rel_row = " & ".join(f"${rel_vals[y]:+.3f}$" for y in YEARS)

print()
print(r"\begin{tabular}{lccccc}")
print(r"Area improvement over random, $\tau@20$"
      r" & 2019 & 2020 & 2021 & 2022 & 2023 \\")
print(r"\midrule")
print(r"Sentence-reordering stability"
      r"  % stale - not re-run in t12_resampling")
print(r" & $+0.003$ & $-0.013$ & $+0.026$ & $-0.011$ & $-0.003$ \\")
print(r"Fisher discriminant ratio"
      r"  % stale - not re-run in t12_resampling")
print(r" & $+0.012$ & $+0.013$ & $+0.044$ & $-0.008$ & $-0.030$ \\")
print(r"Reliability oracle"
      r"  % corrected from t12_resampling/area_ci.csv")
print(f" & {rel_row} \\\\")
print(r"\end{tabular}")
print()
print("NOTE: Reliability oracle values are positive (old code sign-flipped")
print("by descending-axis trapz bug). All 95% CIs include zero on tau@20.")

# ── Caption note — two uncertainty sources ──────────────────────────────
print("\n" + "=" * 66)
print("CAPTION NOTE — two uncertainty sources (paste into Overleaf)")
print("=" * 66)
print("""
Grey band: ordering uncertainty -- 1,000 random permutations of which
  queries are relabelled first, holding the query pool fixed.
  Area CIs (area_ci.csv): query-resampling uncertainty -- bootstrap
  over which queries are in the pool.
  A curve above the band centre means this oracle ordering weakly
  outperforms a random ordering for these specific queries; a CI
  spanning zero means the advantage is not reproduced across different
  query pools.

Suggested caption sentence:
  "The grey band shows query-ordering uncertainty (1,000 random
   permutations of the relabelling order, fixed query pool); 95 %
   confidence intervals on the area above the band mean, obtained by
   query-pool bootstrap (Table X), span zero for both oracles."
""")

# ── Plot ────────────────────────────────────────────────────────────────
# Disable constrained_layout for this figure: we need manual bottom margin
# for the shared legend and constrained_layout + tight_layout conflict.
matplotlib.rcParams["figure.constrained_layout.use"] = False

fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5), sharey=False)

panels = [
    ("tau_at_20", r"$\tau@20$"),
    ("tau_all",   r"$\tau_{\mathrm{all}}$"),
]

legend_handles = []
for ax, (metric, ylabel) in zip(axes, panels):
    rand_mu = avg("random_mean", metric)
    rand_lo = avg("random_lo",   metric)
    rand_hi = avg("random_hi",   metric)
    rel_mu  = avg("reliability_oracle", metric)
    dmg_mu  = avg("damage_oracle",      metric)

    # 1. Random band - visually dominant background
    h_band = ax.fill_between(budget, rand_lo, rand_hi,
                             color=COL_RAND_FILL, alpha=ALPHA_BAND, lw=0,
                             label="Random (2.5-97.5%, 1000 permutations)")

    # 2. Random mean - dashed, explicit legend entry
    h_mean, = ax.plot(budget, rand_mu,
                      color=COL_RAND_LINE, lw=1.2, ls="--", zorder=3,
                      label="Random mean")

    # 3. Oracle curves - solid, same weight
    h_rel, = ax.plot(budget, rel_mu,
                     color=COL_REL, lw=LW, zorder=4,
                     label="Reliability oracle")
    h_dmg, = ax.plot(budget, dmg_mu,
                     color=COL_DMG, lw=LW, zorder=4,
                     label="Damage oracle")

    # Annotate the all-LLM endpoint (budget = 0)
    all_llm = rand_mu[0]
    ax.annotate(
        f"all-LLM\n{all_llm:.3f}",
        xy=(0.0, all_llm),
        xytext=(0.07, all_llm - 0.022),
        fontsize=8, color="#555555",
        arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.6),
    )

    ax.set_xlabel("Fraction of judged pairs bought", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(0, 1)

    ymin = min(rand_lo.min(), rel_mu.min(), dmg_mu.min()) - 0.01
    ax.set_ylim(ymin, 1.002)

    # Collect handles from first panel only (both panels have same legend)
    if not legend_handles:
        legend_handles = [h_band, h_mean, h_rel, h_dmg]

# Shared legend below both panels in one horizontal row
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.0),
    ncols=4,
    fontsize=9,
    frameon=False,
    handlelength=1.4,
    columnspacing=1.0,
)
fig.subplots_adjust(left=0.10, right=0.97, top=0.97, bottom=0.22,
                    wspace=0.35)
fig.savefig("fig_query_triage_curves_merged.pdf")
print("\nSaved fig_query_triage_curves_merged.pdf")

# ── Delete stale files ──────────────────────────────────────────────────
for path in [
    "results/triage/overall_summary.json",
    "results/triage_structural/structural_triage_summary.csv",
]:
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted {path}")
    else:
        print(f"Already absent: {path}")
