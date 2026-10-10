"""Slide figures for Module 4 (D10).

    python Module4/render_recommendation.py --tradeoff   # demos/recommendation_tradeoff.png
    python Module4/render_recommendation.py --examples   # demos/recommendation_demo.png

tradeoff: the report sets' weight sweep, per catalogue. Left: harmony gain vs trend gain of the top
suggestion as the trend weight goes 0 -> 1 (the calibrated weight circled). Right: weak-link hit rate
by trend weight. Both gains are in σ units, so they share one scale.
examples: one real report-set outfit per verdict (swap / refresh / keep), scored on the demo pool
(its own outfit excluded), each drawn as an outfit board.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from demo_panel import outfit_tiles, piece_crop, swap_board  # noqa: E402
from m4_settings import REPO, M4Settings  # noqa: E402
from outfit_board import outfit_board  # noqa: E402

SERIES = {"fashion144k": ("#2a78d6", "Fashion144k"), "pinterest": ("#eb6834", "Pinterest")}
SUMMARIES = {"fashion144k": "report_summary.json", "pinterest": "pinterest_report_summary.json"}
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
VERDICTS = ("swap", "refresh", "keep")
HEADLINE = {"swap": "Swap: {piece} → {to}", "refresh": "Refresh: already strong, a trendier option",
            "keep": "Keep: already strong"}
# slide legibility: fragmentary SegFormer crops (tiny, split, specked) read as noise on a slide
MIN_AREA, MIN_MAIN, MIN_FILL = 4000, 0.85, 0.35
EVAL_SET_SIZE = 2000


def crop_readable(path, min_area: int = MIN_AREA, min_main: float = MIN_MAIN,
                  min_fill: float = MIN_FILL) -> bool:
    """True if the RGBA cut-out exists and is big, one main piece, and solid enough to read."""
    if not path or not os.path.exists(path):
        return False
    try:
        alpha = np.asarray(Image.open(path).convert("RGBA"))[..., 3] > 0
    except Exception:
        return False
    total = int(alpha.sum())
    if total < min_area:
        return False
    labels, n = ndimage.label(alpha, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())[1:]
    main = int(sizes.argmax()) + 1
    if sizes.max() / total < min_main:
        return False
    ys, xs = np.nonzero(labels == main)
    box = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    return bool(len(ys) / box >= min_fill)


def drawn_crops(r, state, pool, settings, verdict: str) -> list:
    """Crop paths render_examples draws for this outfit (a swap's replaced piece is not drawn)."""
    if verdict == "keep":
        return [piece_crop(state, p, pool, settings) for p in state.pieces]
    s = r.suggestions[0]
    return ([piece_crop(state, p, pool, settings) for p in state.pieces if p.file != s.piece_file]
            + [settings.abs(s.crop_path)])


def load_sweeps(eval_dir: str) -> dict:
    out = {}
    for src, name in SUMMARIES.items():
        path = os.path.join(eval_dir, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                out[src] = json.load(f)["weight_sweep"]
    return out


def _style(ax):
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT2)
    ax.tick_params(colors=TEXT2)


def render_tradeoff(sweeps: dict, chosen_w: float, out_path: str) -> str:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4))
    for src, rows in sweeps.items():
        colour, label = SERIES[src]
        w = np.array([r["w_t"] for r in rows])
        dh = np.array([r["mean_dH"] for r in rows])
        dt = np.array([r["mean_dT"] for r in rows])
        hr = np.array([r["hit_rate"] for r in rows])
        k = int(np.argmin(np.abs(w - chosen_w)))
        a.plot(dt, dh, color=colour, linewidth=2, marker="o", markersize=4, label=label)
        a.plot(dt[k], dh[k], marker="o", markersize=12, markerfacecolor="none",
               markeredgecolor=colour, markeredgewidth=2)
        off, ha = ((-8, 12), "right") if src == "pinterest" else ((10, -16), "left")
        a.annotate(f"w = {w[k]:.1f}", (dt[k], dh[k]), xytext=off, textcoords="offset points",
                   color=TEXT2, fontsize=9, ha=ha)
        b.plot(w, hr, color=colour, linewidth=2, marker="o", markersize=4, label=label)
    b.axvline(chosen_w, color=TEXT2, linewidth=1, linestyle="--")
    b.annotate(f"chosen w = {chosen_w:.1f}", (chosen_w, b.get_ylim()[1]), xytext=(6, -14),
               textcoords="offset points", color=TEXT2, fontsize=9)
    a.set_xlabel("Trend gain of the top suggestion (σ)", color=TEXT)
    a.set_ylabel("Harmony gain of the top suggestion (σ)", color=TEXT)
    a.set_title("Harmony vs trend as the trend weight rises", color=TEXT, loc="left")
    b.set_xlabel("Trend weight w", color=TEXT)
    b.set_ylabel("Finds the corrupted piece", color=TEXT)
    b.set_title("Weak-link hit rate by trend weight", color=TEXT, loc="left")
    for ax in (a, b):
        _style(ax)
        ax.legend(frameon=False, labelcolor=TEXT)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    return out_path


def pick_examples(rec, sources, limit: int = EVAL_SET_SIZE) -> dict:
    """First report-set outfit per verdict whose drawn tiles are all readable, scanning `sources` in order."""
    from eval_recovery import eval_sets
    from outfit_state import state_from_pool
    found = {}
    for src in sources:
        _, report = eval_sets(rec.pool, EVAL_SET_SIZE, source=src)
        for key in report[:limit]:
            state = state_from_pool(rec.pool, src, key)
            table = rec.score(state, mode="demo", exclude=(src, key))
            r = rec.decide(table)
            if r.verdict in found or not (r.verdict == "keep" or r.suggestions):
                continue
            if len(state.pieces) < 2:
                continue
            if all(crop_readable(c) for c in drawn_crops(r, state, rec.pool, rec.settings, r.verdict)):
                found[r.verdict] = (src, key, table, r)
            if len(found) == len(VERDICTS):
                return found
    return found


def render_examples(rec, found: dict, out_path: str) -> str:
    verdicts = [v for v in VERDICTS if v in found]
    if not verdicts:
        raise ValueError("no examples found")
    heads = []
    fig, axes = plt.subplots(len(verdicts), 1, figsize=(10, 3.2 * len(verdicts)), squeeze=False)
    with tempfile.TemporaryDirectory() as tmp:
        for ax, v in zip(axes[:, 0], verdicts):
            src, key, table, r = found[v]
            png = os.path.join(tmp, f"{v}.png")
            if v == "keep":
                outfit_board(outfit_tiles(table.state, rec.pool, rec.settings), png)
                head = HEADLINE[v]
            else:
                swap_board(r, table.state, rec.pool, rec.settings, 0, png)
                s = r.suggestions[0]
                head = HEADLINE[v].format(piece=s.from_type, to=s.to_type)
            s0 = r.suggestions[0] if r.suggestions else None
            nums = (f"harmony {s0.harmony_pct_before:.0f} → {s0.harmony_pct_after:.0f}" if s0
                    else f"harmony {r.baseline['harmony_pct']:.0f}")
            ax.imshow(Image.open(png))
            ax.set_axis_off()
            heads.append(f"{head}   ({nums}; {SERIES[src][1]} outfit {key})")
    fig.tight_layout()
    # titles sit at the figure's left edge: boards differ in width, so axes-relative titles clip
    for ax, text in zip(axes[:, 0], heads):
        fig.text(0.02, ax.get_position().y1, text, color=TEXT, ha="left", va="bottom", fontsize=10)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tradeoff", action="store_true")
    ap.add_argument("--examples", action="store_true")
    ap.add_argument("--device")
    args = ap.parse_args(argv)
    s = M4Settings.load()
    demos = os.path.join(REPO, "demos")
    if args.tradeoff:
        sweeps = load_sweeps(s.abs(os.path.join(s.cache_dir, "eval")))
        print(render_tradeoff(sweeps, s.w_t, os.path.join(demos, "recommendation_tradeoff.png")))
    if args.examples:
        from recommender import Recommender
        rec = Recommender.from_settings(s, args.device)
        found = pick_examples(rec, [st.name for st in s.stores])
        print(f"verdicts found: {sorted(found)}")
        print(render_examples(rec, found, os.path.join(demos, "recommendation_demo.png")))


if __name__ == "__main__":
    main()
