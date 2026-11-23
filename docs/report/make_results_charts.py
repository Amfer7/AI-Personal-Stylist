"""Charts for the Review 3 Results and Discussion slides (R2, R4, R7).

Numbers are copied from docs/report/results.md (Tables 8.2.2, 8.5.2 and Section 8.6), which in turn
come from Fashion144k_v1/eval_weak_labels/, Module4/cache/eval/*_report.md and
Module4/cache/human_study/result.json. PNGs go to docs/figures/ (images stay out of git).

    python docs/report/make_results_charts.py
"""
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "figures"

# Reference palette (dataviz skill), light mode on a white slide.
ACCENT = "#2a78d6"   # Module 4 / the one series being read
CONTEXT = "#898781"  # baselines and comparison bars
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 14,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def _clean(ax, grid_axis="x"):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _title(fig, title, subtitle):
    fig.text(0.02, 0.965, title, fontsize=20, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.895, subtitle, fontsize=13, color=INK_2, va="top")


def recall_chart():
    """R2: Module 1 weak-label recall by garment group."""
    rows = [  # group, photos naming it, recall %
        ("Upper body", 68774, 97.6), ("Lower body", 47203, 94.8), ("Bag", 23714, 83.8),
        ("Hat", 10823, 79.5), ("Dress", 24695, 76.6), ("Shoes", 48042, 75.0),
        ("Scarf", 3304, 62.5), ("Belt", 3127, 19.4), ("Sunglasses", 4861, 5.6),
    ]
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=200)
    fig.subplots_adjust(left=0.2, right=0.95, top=0.8, bottom=0.11)
    labels = [f"{g}  ({n:,})" for g, n, _ in rows][::-1]
    vals = [r for _, _, r in rows][::-1]
    bars = ax.barh(labels, vals, height=0.6, color=ACCENT)
    for b, v in zip(bars, vals):
        ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center",
                fontsize=14, color=INK)
    ax.set_xlim(0, 105)
    ax.set_xticks(range(0, 101, 25))
    ax.set_xticklabels([f"{t}%" for t in range(0, 101, 25)])
    ax.set_xlabel("Recall: share of photos naming the item where Module 1 found it")
    _clean(ax)
    _title(fig, "Module 1 finds clothing reliably; small accessories are often missed",
           "Weak-label recall on all 144,169 Fashion144k photos (title words as labels; "
           "a lower bound). In brackets: photos naming the item")
    fig.savefig(OUT / "r2_m1_recall.png")
    plt.close(fig)


def hit_rate_chart():
    """R4: Module 4 vs baselines, hit rate with 95% bootstrap CIs, one panel per dataset."""
    methods = ["Module 4", "CLIP similarity", "Trend only", "Random"]
    data = {  # dataset: ([(rate, lo, hi)] in method order, chance)
        "Fashion144k": ([(72.2, 70.2, 74.1), (42.2, 40.1, 44.4), (32.6, 30.5, 34.5),
                         (30.4, 28.4, 32.4)], 31.4),
        "Pinterest": ([(69.3, 67.3, 71.3), (42.7, 40.5, 44.9), (34.5, 32.5, 36.4),
                       (37.8, 35.7, 40.0)], 36.0),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), dpi=200, sharey=True)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.76, bottom=0.12, wspace=0.12)
    ys = list(range(len(methods)))[::-1]
    for ax, (name, (rows, chance)) in zip(axes, data.items()):
        colors = [ACCENT] + [CONTEXT] * (len(methods) - 1)
        rates = [r for r, _, _ in rows]
        ax.barh(ys, rates, height=0.6, color=colors)
        ax.errorbar(rates, ys, xerr=[[r - lo for r, lo, _ in rows], [hi - r for r, _, hi in rows]],
                    fmt="none", ecolor=INK, elinewidth=1.2, capsize=4)
        for y, (r, _, hi) in zip(ys, rows):
            ax.text(hi + 1.5, y, f"{r:.1f}%", va="center", fontsize=14, color=INK,
                    fontweight="bold" if y == ys[0] else "normal")
        ax.axvline(chance, color=INK_2, linewidth=1.5, linestyle=(0, (4, 3)))
        ax.text(chance + 0.8, -0.62, f"chance {chance:.1f}%", fontsize=12, color=INK_2,
                va="center")
        ax.set_title(f"{name} (2,000 test outfits)", fontsize=15, color=INK, loc="left", pad=10)
        ax.set_xlim(0, 85)
        ax.set_ylim(-0.85, len(methods) - 0.45)
        ax.set_xticks(range(0, 81, 20))
        ax.set_xticklabels([f"{t}%" for t in range(0, 81, 20)])
        _clean(ax)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(methods)
    fig.supxlabel("Hit rate: share of outfits where the method picks the spoiled piece",
                  fontsize=13, color=INK_2, y=0.02)
    _title(fig, "Module 4 finds the spoiled piece about 30 points more often than any baseline",
           "One random piece per test outfit replaced by an allowed intruder. "
           "Whiskers: 95% bootstrap CI. All baselines p < 1e-60 (McNemar)")
    fig.savefig(OUT / "r4_m4_vs_baselines.png")
    plt.close(fig)


def human_study_chart():
    """R7: human pilot, share preferring each swap and the lift with its CIs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6.75), dpi=200,
                                   gridspec_kw={"width_ratios": [1, 1.25]})
    fig.subplots_adjust(left=0.08, right=0.96, top=0.74, bottom=0.14, wspace=0.35)

    # Left: share of decided votes preferring the swapped outfit over the original.
    names = ["Module 4's swap", "Random allowed swap"]
    shares = [37.4, 37.8]
    votes = [489, 492]
    ax1.bar(names, shares, width=0.5, color=[ACCENT, CONTEXT])
    for x, (s, v) in enumerate(zip(shares, votes)):
        ax1.text(x, s + 1.2, f"{s:.1f}%", ha="center", fontsize=16, color=INK, fontweight="bold")
        ax1.text(x, s / 2, f"{v} votes", ha="center", fontsize=12, color="white")
    ax1.set_ylim(0, 50)
    ax1.set_yticks(range(0, 51, 10))
    ax1.set_yticklabels([f"{t}%" for t in range(0, 51, 10)])
    ax1.tick_params(axis="y", colors=MUTED)
    ax1.tick_params(axis="x", colors=INK)
    ax1.set_title("Prefer the swapped outfit to the original", fontsize=14, color=INK,
                  loc="left", pad=10)
    _clean(ax1, grid_axis="y")

    # Right: lift (Module 4 minus random) with two intervals.
    rows = [("Over raters", -5.3, 4.8), ("Over raters\nand outfits", -10.9, 9.6)]
    lift = -0.4
    for y, (label, lo, hi) in zip([1, 0], rows):
        ax2.hlines(y, lo, hi, color=INK_2, linewidth=2)
        ax2.vlines([lo, hi], y - 0.08, y + 0.08, color=INK_2, linewidth=2)
        ax2.plot(lift, y, "o", markersize=11, color=ACCENT, markeredgecolor="white",
                 markeredgewidth=2, zorder=3)
        ax2.text(hi + 0.8, y, f"{lo:+.1f} to {hi:+.1f}", va="center", fontsize=12, color=INK_2)
    ax2.axvline(0, color=INK, linewidth=1)
    ax2.text(0.4, 1.42, "no difference", fontsize=12, color=INK_2)
    ax2.text(lift - 0.8, 1.25, f"lift {lift:+.1f} pts", ha="right", va="center", fontsize=14,
             color=INK, fontweight="bold")
    ax2.set_yticks([1, 0])
    ax2.set_yticklabels([r[0] for r in rows])
    ax2.set_ylim(-0.5, 1.6)
    ax2.set_xlim(-14, 19)
    ax2.set_xlabel("Lift: Module 4 minus random (percentage points), 95% CI")
    ax2.set_title("Lift with its 95% interval", fontsize=14, color=INK, loc="left", pad=10)
    _clean(ax2)

    _title(fig, "Pilot: raters did not prefer Module 4's swap to a random swap",
           "Blind A/B form, 26 raters, 20 street outfits × 2 conditions = 1,040 answers "
           "(59 \"Can't decide\" excluded). Both swaps from the same H&M catalogue")
    fig.savefig(OUT / "r7_human_study.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    recall_chart()
    hit_rate_chart()
    human_study_chart()
    print("wrote", *sorted(p.name for p in OUT.glob("r[247]_*.png")))
