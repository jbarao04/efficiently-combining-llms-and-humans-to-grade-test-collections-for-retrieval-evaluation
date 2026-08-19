# Figure: tau@20 vs budget for each year with threshold markers

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

sys.path.insert(0, ".")
import plot_style  # noqa: E402  (applies theme on import)

# ── data ────────────────────────────────────────────────────────────────────

CURVES_CSV = "results/thesis_verification/t12_resampling/curves_full.csv"
THRESH_CSV = "results/thesis_verification/t12_resampling/thresholds_ci.csv"

curves = pd.read_csv(CURVES_CSV)
thresh = pd.read_csv(THRESH_CSV)

# ── policy specification ─────────────────────────────────────────────────────

POLICIES = ["random", "lara", "mtf", "leverage", "product_cal"]

LABELS = {
    "random":      "Random",
    "lara":        "Confidence (LARA)",
    "mtf":         "Move-to-front",
    "leverage":    "Leverage",
    "product_cal": "Product (calibrated)",
}

_LARA_BLUE = "#2594C8"          # deepened sky blue, readable on white

COLORS = {
    "random":      plot_style.OI_BLACK,
    "lara":        _LARA_BLUE,
    "mtf":         plot_style.OI_GREEN,
    "leverage":    plot_style.OI_VERMILLION,
    "product_cal": plot_style.OI_PINK,
}

# product_cal rendered as secondary overlay (dashed, thinner)
LINEWIDTHS = {p: 1.2 for p in POLICIES}
LINEWIDTHS["product_cal"] = 0.8
LINESTYLES = {p: "-" for p in POLICIES}
LINESTYLES["product_cal"] = "--"

YEARS = [2019, 2020, 2021, 2022, 2023]
X_MAX  = 50          # x-axis right edge (%)
Y_MIN  = 0.55        # clipped y lower bound
Y_MAX  = 1.00
TAU_RULE = 0.95

MARKER_FILL_SIZE   = 5   # filled = first-touch
MARKER_HOLLOW_SIZE = 8   # hollow = sustained (rings around filled when coincide)

# ── filter ──────────────────────────────────────────────────────────────────

curves = curves[curves["policy"].isin(POLICIES) & (curves["budget_pct"] <= X_MAX)].copy()
thresh = thresh[thresh["policy"].isin(POLICIES)].copy()

# ── figure ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(
    1, 5,
    figsize=(13.5, 3.6),
    sharey=True,
)

base_fs = plt.rcParams["font.size"]

for ax, year in zip(axes, YEARS):
    yr_c = curves[curves["year"] == year]
    yr_t = thresh[thresh["year"] == year].set_index("policy")

    for policy in POLICIES:
        pol = yr_c[yr_c["policy"] == policy].sort_values("budget_pct")
        if pol.empty:
            continue

        color = COLORS[policy]
        ax.plot(
            pol["budget_pct"],
            pol["tau_at_20"],
            color=color,
            linewidth=LINEWIDTHS[policy],
            linestyle=LINESTYLES[policy],
            label=LABELS[policy],
            zorder=3,
        )

        # ── LARA excursion annotation (drop below Y_MIN) ──────────────
        if policy == "lara":
            min_tau = pol["tau_at_20"].min()
            min_bud = pol.loc[pol["tau_at_20"].idxmin(), "budget_pct"]
            if min_tau < Y_MIN:
                # arrow pointing down from bottom edge at the drop's x position
                ax.annotate(
                    f"\u2193 LARA {min_tau:.2f}",
                    xy=(min_bud, Y_MIN),
                    xytext=(min_bud + 1.5, Y_MIN + 0.025),
                    fontsize=base_fs * 0.68,
                    color=color,
                    va="bottom", ha="left",
                    arrowprops=dict(
                        arrowstyle="-",
                        color=color,
                        lw=0.7,
                    ),
                    clip_on=False,
                    zorder=10,
                )

        # ── threshold markers ─────────────────────────────────────────
        if policy not in yr_t.index:
            continue
        row = yr_t.loc[policy]

        ft = row["first_touch_point"]
        st = row["sustained_point"]

        # Hollow marker (sustained) drawn first so filled sits on top
        if pd.notna(st) and st <= X_MAX:
            tau_st = float(np.interp(st, pol["budget_pct"].values, pol["tau_at_20"].values))
            ax.plot(st, tau_st, "o",
                    color=color, markersize=MARKER_HOLLOW_SIZE,
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=1.3, zorder=5)

        # Filled marker (first touch) on top
        if pd.notna(ft) and ft <= X_MAX:
            tau_ft = float(np.interp(ft, pol["budget_pct"].values, pol["tau_at_20"].values))
            ax.plot(ft, tau_ft, "o",
                    color=color, markersize=MARKER_FILL_SIZE,
                    zorder=6)

        # ── right-pointing arrow when any threshold is off-axis ───────
        ft_off = pd.notna(ft) and ft > X_MAX
        st_off = pd.notna(st) and st > X_MAX
        if ft_off or st_off:
            tau_edge = float(np.interp(X_MAX, pol["budget_pct"].values, pol["tau_at_20"].values))
            ax.annotate(
                "\u2192",
                xy=(X_MAX, tau_edge),
                fontsize=base_fs * 0.80,
                color=color,
                ha="left", va="center",
                clip_on=False,
                zorder=10,
            )

    # ── reference line at 0.95 ────────────────────────────────────────────
    ax.axhline(TAU_RULE, color=plot_style._GREY30, linewidth=0.9,
               linestyle="--", zorder=1)

    ax.set_title(str(year))
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xticks([0, 25, 50])
    ax.set_xlabel("Budget (%)")

axes[0].set_ylabel(r"$\tau$@20")

# ── legend ───────────────────────────────────────────────────────────────────

policy_handles = [
    mlines.Line2D([], [],
                  color=COLORS[p],
                  linewidth=LINEWIDTHS[p],
                  linestyle=LINESTYLES[p],
                  label=LABELS[p])
    for p in POLICIES
]
marker_handles = [
    mlines.Line2D([], [], color=plot_style._GREY30,
                  marker="o", linestyle="none",
                  markersize=MARKER_FILL_SIZE,
                  label="First touch (filled)"),
    mlines.Line2D([], [], color=plot_style._GREY30,
                  marker="o", linestyle="none",
                  markersize=MARKER_HOLLOW_SIZE,
                  markerfacecolor="white",
                  markeredgecolor=plot_style._GREY30,
                  markeredgewidth=1.3,
                  label="Sustained (hollow)"),
]

fig.legend(
    handles=policy_handles + marker_handles,
    loc="lower center",
    ncol=len(policy_handles) + len(marker_handles),
    bbox_to_anchor=(0.5, -0.20),
    frameon=False,
    fontsize=base_fs * 0.83,
)

# ── save ─────────────────────────────────────────────────────────────────────

out = "fig_tau20_by_regime.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"Saved {out}")
