"""
render_perception_demo.py
=========================
Compose a single slide figure from a Module 1 perception output folder:
    original photo  |  each segmented garment crop + its extracted attributes.

Reads the metadata.json / *.png produced by run_pipeline.py (or 01_segment_batch.py),
so it works on a freshly-run single photo or any existing outputs/<idx>/ folder.

Usage:
    python render_perception_demo.py \
        --outfit_dir ../Fashion144k_v1/demo_out/demo_outfit \
        --photo      ../Fashion144k_v1/photos/<original>.jpg \
        --out        perception_demo.png
"""

import os
import json
import argparse

from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _on_white(png_path):
    """Composite an RGBA garment crop onto white so transparency reads cleanly."""
    im = Image.open(png_path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im).convert("RGB")


def _caption(g):
    a = g.get("attributes", {}) or {}
    col = (a.get("color") or {}).get("name", "?")
    parts = [
        g.get("broad_category", g.get("category", "?")),
        f"{col} · {a.get('pattern', '?')}",
        f"{a.get('formality', '?')} ({a.get('formality_score', '?')})",
        f"{a.get('silhouette', '?')} · {a.get('fit', '?')}",
    ]
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outfit_dir", required=True, help="Folder with metadata.json + garment *.png")
    ap.add_argument("--photo", default=None, help="Original input photo (optional, shown at left)")
    ap.add_argument("--out", default="perception_demo.png")
    args = ap.parse_args()

    garments = json.load(open(os.path.join(args.outfit_dir, "metadata.json")))
    n = len(garments)
    ncols = n + (1 if args.photo else 0)

    fig, axes = plt.subplots(1, ncols, figsize=(3.0 * ncols, 4.4))
    if ncols == 1:
        axes = [axes]

    col = 0
    if args.photo:
        axes[col].imshow(Image.open(args.photo).convert("RGB"))
        axes[col].set_title("Input photo", fontsize=11, fontweight="bold")
        axes[col].axis("off")
        col += 1

    for g in garments:
        crop = _on_white(os.path.join(args.outfit_dir, g["file"]))
        axes[col].imshow(crop)
        axes[col].set_title(_caption(g), fontsize=9)
        axes[col].axis("off")
        col += 1

    fig.suptitle("Module 1 — Garment Perception: segmentation + attributes",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=150)
    print(f"[perception] {n} garments -> wrote {args.out}")


if __name__ == "__main__":
    main()
