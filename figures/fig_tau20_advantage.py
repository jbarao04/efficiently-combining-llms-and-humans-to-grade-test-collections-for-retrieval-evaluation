# Figure: tau@20 advantage over random for four correction policies with significance strips

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import pandas as pd
import plot_style  # applies theme on import

# ── data ──────────────────────────────────────────────────────────────────────
DATA_PATH = "results/thesis_verification/t12_resampling/difference_by_budget.csv"

POLICIES = ["lara", "mtf", "leverage", "product_cal"]

POLICY_LABELS = {
    "lara":        "Confidence (LARA)",
    "mtf":         "Move-to-front",
    "leverage":    "Leverage",
    "product_cal": "Product (calibrated)",
}

POLICY_COLORS = {
    "lara":        plot_style.OI_SKY,
    "mtf":         plot_style.OI_GREEN,
    "leverage":    plot_style.OI_VERMILLION,
    "product_cal": plot_style.OI_PINK,
}

YEARS = [2019, 2020, 2021, 2022, 2023]
X_MAX  = 30
METRIC = "tau_at_20"
Y_LO, Y_HI = -0.30, 0.45
Y_TICKS    = [-0.2, 0.0, 0.2, 0.4]

# ── load & filter ─────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df = df[
    (df["metric"] == METRIC)
    & (df["policy"].isin(POLICIES))
    & (df["budget_pct"] <= X_MAX)
].copy()

# ── figure layout: main row + significance-strip row ─────────────────────────
fig = plt.figure(figsize=(7.0, 3.1))
fig.set_layout_engine(None)   # override constrained_layout set by plot_style
gs  = gridspec.GridSpec(
    2, 5,
    figure=fig,
    height_ratios=[5, 0.55],
    hspace=0.06,
    wspace=0.08,
)
main_axes  = [fig.add_subplot(gs[0, i]) for i in range(5)]
strip_axes = [fig.add_subplot(gs[1, i]) for i in range(5)]

# share y-axis across main panels
for ax in main_axes[1:]:
    ax.sharey(main_axes[0])

# ── strip y-position constants ────────────────────────────────────────────────
STRIP_H    = 0.55   # bar height in strip units
STRIP_GAP  = 1.15   # centre-to-centre spacing (more room between rows)
STRIP_Y    = {p: i * STRIP_GAP for i, p in enumerate(POLICIES)}

# ── draw panels ───────────────────────────────────────────────────────────────
for col_i, (ax, sax, year) in enumerate(zip(main_axes, strip_axes, YEARS)):
    ydata = df[df["year"] == year]

    for policy in POLICIES:
        pdata = ydata[ydata["policy"] == policy].sort_values("budget_pct")
        if pdata.empty:
            continue

        x   = pdata["budget_pct"].values
        mid = pdata["mean_diff_vs_random"].values
        lo  = pdata["lo_2p5"].values
        hi  = pdata["hi_97p5"].values
        col = POLICY_COLORS[policy]

        ax.plot(x, mid, color=col, linewidth=1.8, alpha=1.0, zorder=3)

        # significance strip: colour where CI excludes zero
        sig = pdata[pdata["excludes_zero"]]["budget_pct"].values
        if sig.size:
            segments = [(bx - 0.5, 1.0) for bx in sig]   # (xstart, width)
            yc = STRIP_Y[policy]
            sax.broken_barh(
                segments,
                (yc, STRIP_H),
                facecolors=col,
                linewidth=0,
            )

    # zero reference — bold solid line
    ax.axhline(0, color="black", linewidth=1.0, linestyle="-", zorder=4)

    ax.set_xlim(0, X_MAX)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_yticks(Y_TICKS)
    ax.set_title(str(year))
    ax.set_xlabel("Budget (%)")

    # strip formatting
    sax.set_xlim(0, X_MAX)
    sax.set_ylim(-0.3, len(POLICIES) * STRIP_GAP - 0.2)
    sax.axis("off")

main_axes[0].set_ylabel(r"$\Delta\,\tau_{@20}$ vs.\ Random")

# remove y-tick labels from non-leftmost panels
for ax in main_axes[1:]:
    ax.tick_params(labelleft=False)

# ── legend: one Line2D per policy ────────────────────────────────────────────
handles = [
    mlines.Line2D([], [], color=POLICY_COLORS[p], linewidth=1.8,
                  label=POLICY_LABELS[p])
    for p in POLICIES
]

fig.legend(
    handles=handles,
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.13),
    frameon=False,
    fontsize=plt.rcParams["font.size"] * 0.88,
)

# ── save ──────────────────────────────────────────────────────────────────────
fig.savefig("fig_tau20_advantage.pdf", bbox_inches="tight")
print("Saved fig_tau20_advantage.pdf")
