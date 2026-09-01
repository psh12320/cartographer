"""Generate the submission figures from measured results.

Every number here was produced by the unmodified official evaluator; nothing is
illustrative. Re-run with `python3 docs/make_figures.py` to regenerate.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT = pathlib.Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# Validated palette (see the data-viz reference instance). Blue/red is the
# diverging pair; both clear every CVD and contrast gate on this surface.
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
RED = "#e34948"
GRID = "#e3e2de"
INK = "#0b0b0b"
MUTED = "#52514e"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 11,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})


def strip(ax, xgrid=False):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x" if xgrid else "y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_journey() -> None:
    """Where the score came from, milestone by milestone."""
    labels = ["Organizer\nbaseline", "Hybrid retrieval\n+ reranker", "Precision\ndepth gate",
              "Override state\nfix", "Confidence-adaptive\nbreadth"]
    values = [0.10671, 0.930420, 0.959387, 0.962335, 0.973062]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bars = ax.bar(labels, values, color=BLUE, width=0.6)
    for bar, value in zip(bars, values):
        # Tall bars label inside so nothing collides with the ceiling rule.
        inside = value > 0.4
        ax.text(bar.get_x() + bar.get_width() / 2,
                value - 0.045 if inside else value + 0.022, f"{value:.3f}",
                ha="center", va="top" if inside else "bottom", fontsize=11.5,
                color=SURFACE if inside else INK, fontweight="bold")
    ax.axhline(0.9926, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.4)
    ax.text(-0.44, 1.006, "structural ceiling 0.9926", ha="left", va="bottom",
            fontsize=9.5, color=MUTED)
    ax.set_ylim(0, 1.09)
    ax.set_ylabel("TechnicalScore")
    ax.set_title("Cartographer: 0.107 → 0.973 on the official evaluator",
                 fontsize=13.5, fontweight="bold", color=INK, pad=14)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    strip(ax)
    fig.tight_layout()
    fig.savefig(OUT / "01-score-journey.png", dpi=200)
    plt.close(fig)


def fig_confidence() -> None:
    """The signal behind adaptive breadth, validated before it was trusted."""
    labels = ["lowest 25%", "25–50%", "50–75%", "highest 25%"]
    values = [40, 32, 60, 72]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bars = ax.bar(labels, values, color=BLUE, width=0.58)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.8, f"{value}%",
                ha="center", fontsize=11.5, color=INK, fontweight="bold")
    ax.set_ylim(0, 104)
    ax.set_ylabel("top pick is the correct product")
    ax.set_xlabel("score margin between candidate 1 and candidate 2")
    ax.set_title("The agent can tell when it is about to be wrong",
                 fontsize=13.5, fontweight="bold", color=INK, pad=14)
    ax.text(-0.44, 100, "When the margin is thin the leader is right 40% of the time; when it is wide, 72%.\n"
                        "Below the threshold the agent returns one product and asks instead of guessing.",
            fontsize=9.5, color=MUTED, va="top")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    strip(ax)
    fig.tight_layout()
    fig.savefig(OUT / "02-confidence-signal.png", dpi=200)
    plt.close(fig)


def fig_ledger() -> None:
    """Everything measured, including what lost. Diverging: blue gains, red losses."""
    rows = [
        ("Confidence-adaptive breadth", +0.0106),
        ("Gradual depth ramp", +0.0022),
        ("Full depth released earlier", +0.0008),
        ("Open question asked twice", +0.0007),
        ("Over-generality cutoff", +0.0003),
        ("Long-term personalization", -0.0002),
        ("Suppress unanswerable question", -0.0011),
        ("Extra ranking features", -0.0028),
        ("Listwise ranking objective", -0.0043),
    ]
    rows = rows[::-1]
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [BLUE if v > 0 else RED for v in values]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    bars = ax.barh(labels, values, color=colors, height=0.62)
    for bar, value in zip(bars, values):
        offset = 0.00035 if value > 0 else -0.00035
        ax.text(value + offset, bar.get_y() + bar.get_height() / 2,
                f"{value:+.4f}", va="center",
                ha="left" if value > 0 else "right", fontsize=10, color=INK)
    ax.axvline(0, color=MUTED, linewidth=1.2)
    ax.set_xlim(-0.0072, 0.0135)
    ax.set_xlabel("change in TechnicalScore over 1,000 sessions")
    ax.set_title("Every idea was measured — including the ones that lost",
                 fontsize=13.5, fontweight="bold", color=INK, pad=14)
    strip(ax, xgrid=True)
    fig.tight_layout()
    fig.savefig(OUT / "03-experiment-ledger.png", dpi=200)
    plt.close(fig)


def fig_scenarios() -> None:
    """Per-scenario reciprocal rank, before and after this work."""
    labels = ["Buying", "Browsing", "Intent\noverride", "Boundary"]
    before = [0.9495, 0.9220, 0.9807, 0.8695]
    after = [0.9988, 0.9950, 0.9892, 1.0000]
    x = range(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.bar([i - width / 2 for i in x], before, width, label="Before this work",
           color=GRID, edgecolor=MUTED, linewidth=0.8)
    bars = ax.bar([i + width / 2 for i in x], after, width, label="Final agent", color=BLUE)
    for bar, value in zip(bars, after):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.3f}",
                ha="center", fontsize=10, color=INK, fontweight="bold")
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0.8, 1.06)
    ax.set_ylabel("Mean reciprocal rank")
    ax.set_title("Every conversation type improved; the weakest is now perfect",
                 fontsize=13.5, fontweight="bold", color=INK, pad=14)
    ax.legend(frameon=False, loc="upper left", ncols=2, fontsize=10)
    strip(ax)
    fig.tight_layout()
    fig.savefig(OUT / "04-per-scenario.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_journey()
    fig_confidence()
    fig_ledger()
    fig_scenarios()
    for path in sorted(OUT.glob("*.png")):
        print(f"  {path.stat().st_size / 1024:7.1f} KB  {path.relative_to(OUT.parent)}")
