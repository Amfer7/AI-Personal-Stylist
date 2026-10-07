"""
render_trend_demo.py
====================
Per-garment trendiness for a Module 1 perception output folder, plus a slide
figure in the same style as Module 1/2's demo figures:
    original photo  |  each garment crop + its trendiness  |  per-garment bar chart

Reuses Module 1's output as-is (metadata.json + one <stem>.png / <stem>.npy per
garment) — no re-segmentation or re-embedding — and scores each garment with
the fitted PerCategoryTrendModel.

Usage:
    python render_trend_demo.py \
        --outfit_dir ../Fashion144k_v1/demo_out/demo_live \
        --photo      ../samples/image2.jpg \
        --out        ../demos/trend_demo.png
"""

import os
import json
import argparse

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trend_model import PerCategoryTrendModel

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "per_category_trend_model.pkl")

# Same 0-100 bands as gradio_demo_per_garment.py. The score describes RECENT
# Pinterest activity (growth + recency); it is not a forecast — see README §Validation.
BANDS = [
    (0, 10, "Not seen recently", "#9e9e9e"),
    (10, 30, "Rarely appearing", "#bdbdbd"),
    (30, 50, "Occasional", "#90caf9"),
    (50, 70, "Actively present", "#42a5f5"),
    (70, 90, "Rising", "#66bb6a"),
    (90, 101, "Frequently seen now", "#2e7d32"),
]
WEAK_COLOR = "#d32f2f"


def band_for_score(score):
    for lo, hi, label, color in BANDS:
        if lo <= score < hi:
            return label, color
    return BANDS[-1][2], BANDS[-1][3]


def _on_white(png_path):
    im = Image.open(png_path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im).convert("RGB")


def score_outfit_dir(outfit_dir, model):
    """Score every garment listed in metadata.json (stale *.npy files from
    earlier runs in the same folder are ignored). Returns score_outfit()'s dict,
    with 'file' carried through on each garment."""
    meta = json.load(open(os.path.join(outfit_dir, "metadata.json")))
    garments = []
    for g in meta:
        npy = os.path.join(outfit_dir, os.path.splitext(g["file"])[0] + ".npy")
        if not os.path.exists(npy):
            continue
        garments.append({
            "embedding": np.load(npy).astype(np.float32).reshape(-1),
            "category": g["broad_category"],
            "specific_category": g.get("specific_category", g.get("category", "?")),
            "file": g["file"],
        })
    return model.score_outfit(garments)


def _is_weak(g, result):
    return (g["trendiness_score"] is not None
            and g["category"] == result["weak_link_category"]
            and g["trendiness_score"] == result["weak_link_score"])


def render(outfit_dir, photo=None, out="trend_demo.png", result=None, model=None):
    """Compose the trend slide. Returns (out, result)."""
    if result is None:
        model = model or PerCategoryTrendModel.load(DEFAULT_MODEL)
        result = score_outfit_dir(outfit_dir, model)
    garments = result["garments"]
    n = len(garments)
    ncols = n + 1 + (1 if photo else 0)

    fig, axes = plt.subplots(1, ncols, figsize=(3.0 * ncols, 4.6),
                             gridspec_kw={"width_ratios": [1] * (ncols - 1) + [1.6]})
    col = 0
    if photo:
        axes[col].imshow(Image.open(photo).convert("RGB"))
        axes[col].set_title("Input photo", fontsize=11, fontweight="bold")
        axes[col].axis("off")
        col += 1

    for g in garments:
        ax = axes[col]
        ax.imshow(_on_white(os.path.join(outfit_dir, g["file"])))
        s = g["trendiness_score"]
        if s is None:
            title, color = f"{g['category']}\nno confident match", "#616161"
        else:
            label, _ = band_for_score(s)
            title, color = f"{g['category']}\n{s:.0f}/100 · {label}", "black"
        if _is_weak(g, result):
            title += "\n⚠ weak link"
            color = WEAK_COLOR
            for side in ax.spines.values():
                side.set_edgecolor(WEAK_COLOR)
                side.set_linewidth(3)
            ax.set_xticks([]); ax.set_yticks([])
        else:
            ax.axis("off")
        ax.set_title(title, fontsize=9, color=color,
                     fontweight="bold" if _is_weak(g, result) else "normal")
        col += 1

    # Bar panel
    ax = axes[col]
    names = [f"{g['category']}\n({g['specific_category']})" for g in garments]
    vals = [g["trendiness_score"] or 0 for g in garments]
    colors = [WEAK_COLOR if _is_weak(g, result) else
              (band_for_score(g["trendiness_score"])[1] if g["trendiness_score"] is not None else "#e0e0e0")
              for g in garments]
    y = np.arange(n)[::-1]
    ax.barh(y, vals, color=colors)
    for yi, g, v in zip(y, garments, vals):
        ax.text(v + 1.5, yi, "—" if g["trendiness_score"] is None else f"{v:.0f}",
                va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 110)
    ax.set_ylim(-1.1, n - 0.5)
    avg = result["outfit_avg_trendiness"]
    if avg is not None:
        ax.axvline(avg, color="black", ls="--", lw=1)
        ax.text(avg, -0.75, f" outfit avg {avg:.0f}", fontsize=8, va="center")
    ax.set_title("Trendiness /100", fontsize=11, fontweight="bold")
    ax.set_xlabel("vs. recent Pinterest activity, same category")

    fig.suptitle("Module 3 — Per-garment trendiness (Pinterest) + weak link",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[trend] {n} garments -> wrote {out}")
    return out, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outfit_dir", required=True, help="Module 1 folder with metadata.json + *.png/*.npy")
    ap.add_argument("--photo", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="trend_demo.png")
    args = ap.parse_args()
    _, result = render(args.outfit_dir, args.photo, args.out,
                       model=PerCategoryTrendModel.load(args.model))
    for g in result["garments"]:
        print(f"  {g['category']:10s} {g['specific_category']:15s} {g['trendiness_score']}")
    print(f"  weak link: {result['weak_link_category']} ({result['weak_link_score']})  "
          f"avg: {result['outfit_avg_trendiness']}")


if __name__ == "__main__":
    main()
