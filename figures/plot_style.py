# Matplotlib/seaborn thesis style: Okabe-Ito palette and METHOD_COLORS mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# 1. Okabe-Ito palette (colourblind-safe)
# ═══════════════════════════════════════════════════════════════════

OI_BLACK      = "#000000"
OI_ORANGE     = "#E69F00"
OI_BLUE       = "#0072B2"
OI_GREEN      = "#009E73"
OI_VERMILLION = "#D55E00"
OI_PINK       = "#CC79A7"
OI_YELLOW     = "#F0E442"
OI_SKY        = "#56B4E9"

# ═══════════════════════════════════════════════════════════════════
# 2. Method / condition colour mapping
#    Every label variant maps to one canonical colour.
# ═══════════════════════════════════════════════════════════════════

METHOD_COLORS = {
    # ── Correction / triage policies (Chapter 6) ──────────────────
    "Random":                   OI_BLACK,
    "random":                   OI_BLACK,

    "Confidence":               OI_BLUE,
    "confidence":               OI_BLUE,
    "Judge only (margin)":      OI_BLUE,
    "judge_only_margin":        OI_BLUE,
    "Margin":                   OI_BLUE,
    "margin":                   OI_BLUE,

    "LARA":                     OI_SKY,
    "Judge only (LARA)":        OI_SKY,
    "judge_only_lara":          OI_SKY,

    "Leverage":                 OI_VERMILLION,
    "leverage":                 OI_VERMILLION,
    "leverage_calibrated":      OI_VERMILLION,
    "Leverage (calibrated)":    OI_VERMILLION,

    "MTF":                      OI_GREEN,
    "Move-to-front":            OI_GREEN,
    "move_to_front":            OI_GREEN,

    "Depth-k":                  OI_ORANGE,
    "depth_k":                  OI_ORANGE,
    "Pooling depth":            OI_ORANGE,
    "pooling_depth":            OI_ORANGE,

    "Popularity":               OI_YELLOW,
    "popularity":               OI_YELLOW,
    "retrieval_popularity":     OI_YELLOW,

    "Oracle":                   OI_PINK,
    "oracle":                   OI_PINK,
    "Impact oracle":            OI_PINK,
    "impact_oracle":            OI_PINK,
    "Passage oracle":           OI_PINK,
    "passage_oracle":           OI_PINK,

    # ── Feature families (Chapter 4) ─────────────────────────────
    "Family A":                 OI_ORANGE,
    "Score distribution":       OI_ORANGE,

    "Family B":                 OI_VERMILLION,
    "Perturbation (B1b)":       OI_VERMILLION,
    "B1b":                      OI_VERMILLION,

    "Family C":                 OI_BLUE,
    "Probability distribution": OI_BLUE,

    "Fisher ratio":             OI_GREEN,
    "fisher_ratio":             OI_GREEN,

    # ── Grade sources (Chapter 3 / EDA) ──────────────────────────
    "Human":                    OI_BLACK,
    "human":                    OI_BLACK,
    "LLM":                      OI_BLUE,
    "llm":                      OI_BLUE,
    "Llama-3.1-8B":             OI_BLUE,
}

# ═══════════════════════════════════════════════════════════════════
# 3. theme_thesis — the main styling function
# ═══════════════════════════════════════════════════════════════════

_GREY85 = "#D9D9D9"
_GREY93 = "#EDEDED"
_GREY30 = "#4D4D4D"


def apply_theme(base_size: float = 11) -> None:
    """Set rcParams to match the thesis theme.

    Built on the same spec as theme_thesis() in the R version:
      - serif font family
      - no minor gridlines; major gridlines thin grey85
      - grey30 axis lines and ticks
      - no in-plot title / subtitle / suptitle (captions live in LaTeX)
      - legend at the bottom, no legend title
    """
    mpl.rcParams.update({
        # ── font ──────────────────────────────────────────────────
        "font.family":        "serif",
        "font.size":          base_size,
        "axes.titlesize":     base_size,
        "axes.labelsize":     base_size,
        "xtick.labelsize":    base_size * 0.9,
        "ytick.labelsize":    base_size * 0.9,
        "legend.fontsize":    base_size * 0.9,

        # ── axes and ticks (grey30, 0.3 pt) ───────────────────────
        "axes.linewidth":     0.3,
        "axes.edgecolor":     _GREY30,
        "xtick.color":        _GREY30,
        "ytick.color":        _GREY30,
        "xtick.major.width":  0.3,
        "ytick.major.width":  0.3,
        "xtick.minor.width":  0.0,
        "ytick.minor.width":  0.0,
        "xtick.direction":    "out",
        "ytick.direction":    "out",

        # ── grid: major only, thin grey85 ─────────────────────────
        "axes.grid":          True,
        "grid.color":         _GREY85,
        "grid.linewidth":     0.3,
        "axes.grid.which":    "major",

        # ── spines: left and bottom only ──────────────────────────
        "axes.spines.top":    False,
        "axes.spines.right":  False,

        # ── legend: bottom, no title ──────────────────────────────
        "legend.loc":             "lower center",
        "legend.frameon":         False,
        "legend.title_fontsize":  0,   # suppresses title

        # ── suppress in-plot titles (captions in LaTeX) ───────────
        "figure.titlesize":   0.1,

        # ── white background, tight layout ────────────────────────
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "savefig.facecolor":  "white",
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "figure.constrained_layout.use": True,
    })


def style_facet_strips(fig: plt.Figure) -> None:
    """Style suptitle / subplot titles to mimic ggplot2 facet strips.

    Call after creating faceted subplots (e.g. via fig.subplots or
    seaborn FacetGrid). Sets strip background to grey93 and text to
    bold at 0.85× base size.
    """
    base = mpl.rcParams["font.size"]
    for ax in fig.get_axes():
        title = ax.get_title()
        if title:
            ax.set_title(
                title,
                fontsize=base * 0.85,
                fontweight="bold",
                backgroundcolor=_GREY93,
                pad=4,
            )


# ═══════════════════════════════════════════════════════════════════
# 4. Convenience helpers
# ═══════════════════════════════════════════════════════════════════

def get_color(label: str) -> str:
    """Return the Okabe-Ito colour for a method label, with fallback."""
    return METHOD_COLORS.get(label, OI_BLACK)


def color_list(labels) -> list[str]:
    """Return an ordered list of colours matching a sequence of labels."""
    return [get_color(l) for l in labels]


def apply_method_colors(ax: plt.Axes, labels: list[str]) -> None:
    """Recolour existing lines on *ax* to match METHOD_COLORS order."""
    for line, label in zip(ax.get_lines(), labels):
        line.set_color(get_color(label))


# ═══════════════════════════════════════════════════════════════════
# 5. Apply theme on import
# ═══════════════════════════════════════════════════════════════════

apply_theme()
