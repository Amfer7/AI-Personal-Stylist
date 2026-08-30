"""
demo_score.py
=============
Module 2 compatibility demo: show the trained GNN scores a REAL outfit higher than
a CORRUPTED one (same outfit with a single same-category garment swapped for a
mismatched item from a different outfit).

Uses the chosen model (attr-subset) checkpoint + the prebuilt feature store, so it
needs no re-segmentation. Prints a per-example table + an aggregate "real > corrupted"
rate over many test outfits, and saves a figure for the slide.

Note on the score: BPR trains a *ranking* model, so the raw outfit score is only
meaningful *relatively* (absolute logits are all large-negative; a plain sigmoid
collapses to ~0). We therefore display a **Harmoniousness percentile (0–100)** = where
an outfit's raw score falls within the distribution of real test outfits. Real outfits
spread across 0–100; corrupting one garment pushes it down.

Usage:
    python demo_score.py \
        --feature_store ../Fashion144k_v1/feature_store \
        --split_mat     ../Fashion144k_v1/split.mat \
        --relvotes_mat  ../Fashion144k_v1/feat/relvotes.mat \
        --output_root   ../Fashion144k_v1/outputs \
        --checkpoint    ../Fashion144k_v1/ck_attr_subset_s42/clip_outfit_gnn_best.pt \
        --out           compatibility_demo.png
"""

import os
import sys
import argparse
from importlib.util import spec_from_file_location, module_from_spec

import numpy as np
import torch

# Fashion144K filenames + our table symbols are non-ASCII; force UTF-8 stdout so a
# redirected/piped run on Windows (cp1252 default) does not crash mid-print.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(mod_file, name):
    spec = spec_from_file_location(name, os.path.join(HERE, mod_file))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


od = _load("02_outfit_dataset.py", "outfit_dataset")
tg = _load("03_train_gnn.py", "train_gnn")


@torch.no_grad()
def raw_score(model, dataset, nodes, attr_mask, device):
    """Raw outfit score (logit). Ranking-meaningful; calibrated to a percentile below."""
    g = dataset.build_graph(nodes, 0.0, use_attributes=True, attr_mask=attr_mask)
    raw, _ = model(g["x"].to(device), g["edge_index"].to(device), g["edge_attr"].to(device))
    return float(raw.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_store", default=os.path.join(HERE, "../Fashion144k_v1/feature_store"))
    ap.add_argument("--split_mat", default=os.path.join(HERE, "../Fashion144k_v1/split.mat"))
    ap.add_argument("--relvotes_mat", default=os.path.join(HERE, "../Fashion144k_v1/feat/relvotes.mat"))
    ap.add_argument("--output_root", default=os.path.join(HERE, "../Fashion144k_v1/outputs"))
    ap.add_argument("--checkpoint",
                    default=os.path.join(HERE, "../Fashion144k_v1/ck_attr_subset_s42/clip_outfit_gnn_best.pt"))
    ap.add_argument("--num_examples", type=int, default=8, help="Outfits shown in the table/figure")
    ap.add_argument("--num_eval", type=int, default=500, help="Outfits for the aggregate rate + calibration")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(HERE, "compatibility_demo.png"))
    ap.add_argument("--score_dir", default=None,
                    help="Score a single segmented outfit folder (run_pipeline.py output) end-to-end "
                         "and render a Harmoniousness figure, instead of the aggregate demo.")
    ap.add_argument("--photo", default=None, help="Original photo to show in --score_dir mode")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tg.set_seed(args.seed)
    print(f"[demo] device={device}  checkpoint={os.path.basename(args.checkpoint)}")

    split = od.load_split(args.split_mat)
    relvotes = od.load_relvotes(args.relvotes_mat)
    ds = od.OutfitDataset(split["test"], args.output_root, relvotes, feature_store=args.feature_store)
    pool = ds.build_pool(ds.valid_indices)
    attr_mask = od.attr_subset_indices()  # the chosen attr-subset model (11 reliable dims)

    # Rebuild the exact architecture the checkpoint was trained with (attr-subset: 11 attr dims, 5-D edges).
    model = tg.CLIPOutfitGNN(attr_dim=len(attr_mask), edge_dim=od.EDGE_ATTR_DIM).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    # ---- aggregate: how often does the model rank the real outfit above the corrupted one? ----
    # Also collect real-outfit raw scores to calibrate the 0-100 percentile display.
    wins = total = 0
    raw_margins, ref = [], []
    for idx in ds.valid_indices[:args.num_eval]:
        nodes = ds.load_outfit_nodes(idx)
        neg = tg.make_negative_sample(ds, nodes, pool, idx)
        if neg is None:
            continue
        r = raw_score(model, ds, nodes, attr_mask, device)
        c = raw_score(model, ds, neg, attr_mask, device)
        wins += int(r > c)
        raw_margins.append(r - c)
        ref.append(r)
        total += 1
    rate = wins / max(total, 1)
    ref = np.sort(np.asarray(ref, dtype=np.float64))

    def pct(raw):  # Harmoniousness percentile (0-100) vs real test outfits
        return 100.0 * np.searchsorted(ref, raw, side="right") / len(ref)

    print(f"\n[demo] Real > Corrupted on {wins}/{total} test outfits = {rate:.1%} "
          f"(mean raw margin {np.mean(raw_margins):+.3f})\n")

    # ---- single-photo mode: score one real segmented outfit end-to-end ----
    if args.score_dir:
        import json
        garments = json.load(open(os.path.join(args.score_dir, "metadata.json")))
        nodes = [od.build_node_from_meta(args.score_dir, g) for g in garments]
        raw = raw_score(model, ds, nodes, attr_mask, device)
        score = pct(raw)
        neg = tg.make_negative_sample(ds, nodes, pool, current_idx=-1)
        corr = pct(raw_score(model, ds, neg, attr_mask, device)) if neg is not None else None
        print(f"[demo] {os.path.basename(args.score_dir)}: {len(nodes)} garments "
              f"({', '.join(n['category'] for n in nodes)})")
        print(f"[demo] Harmoniousness = {score:.0f}/100"
              + (f"   (drops to {corr:.0f}/100 if one garment is swapped)" if corr is not None else ""))
        _photo_figure(args.photo, args.score_dir, garments, score, corr, args.out)
        print(f"\n[demo] wrote figure -> {args.out}")
        return

    # ---- per-example table + figure ----
    rows = []
    for idx in ds.valid_indices:
        nodes = ds.load_outfit_nodes(idx)
        neg = tg.make_negative_sample(ds, nodes, pool, idx)
        if neg is None:
            continue
        r = raw_score(model, ds, nodes, attr_mask, device)
        c = raw_score(model, ds, neg, attr_mask, device)
        cats = [n["category"] for n in nodes]
        rows.append((idx, len(nodes), cats, pct(r), pct(c)))
        if len(rows) >= args.num_examples:
            break

    print(f"{'outfit':>7}  {'#g':>2}  {'real':>6}  {'corrupt':>7}  {'Δ':>6}  ok  categories")
    for idx, n, cats, pr, pc in rows:
        print(f"{idx:>7}  {n:>2}  {pr:>5.0f}  {pc:>6.0f}   {pr-pc:>+5.0f}  "
              f"{'✓' if pr > pc else '✗'}   {', '.join(cats)}")

    _figure(rows, rate, total, args.out)
    print(f"\n[demo] wrote figure -> {args.out}")


def _figure(rows, rate, total, out_path):
    """Left: paired Harmoniousness percentile (real vs corrupted) per example outfit.
    Right: overall win-rate vs the 50% chance line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"#{idx}\n({n})" for idx, n, _, _, _ in rows]
    real = [pr for *_, pr, _ in rows]
    corr = [pc for *_, pc in rows]
    x = np.arange(len(rows))
    w = 0.38

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(max(8, 1.25 * len(rows)), 4.3),
                                   gridspec_kw={"width_ratios": [len(rows), 2]})

    axA.bar(x - w / 2, real, w, label="Real outfit", color="#2e7d32")
    axA.bar(x + w / 2, corr, w, label="Corrupted (1 swap)", color="#c62828")
    axA.set_xticks(x)
    axA.set_xticklabels(labels, fontsize=8)
    axA.set_ylabel("Harmoniousness percentile (0–100)")
    axA.set_ylim(0, 100)
    axA.set_title("Per-outfit: corrupting a garment lowers the score")
    axA.legend(loc="upper right", fontsize=8)
    axA.grid(axis="y", alpha=0.3)

    axB.bar([0], [rate * 100], width=0.6, color="#2e7d32")
    axB.axhline(50, color="gray", linestyle="--", linewidth=1)
    axB.text(0, 52, "chance 50%", ha="center", fontsize=8, color="gray")
    axB.text(0, rate * 100 + 1, f"{rate:.1%}", ha="center", fontsize=12, fontweight="bold")
    axB.set_xticks([0])
    axB.set_xticklabels([f"real > corrupt\n(n={total})"], fontsize=8)
    axB.set_ylim(0, 100)
    axB.set_ylabel("% ranked correctly")
    axB.set_title("Overall")

    fig.suptitle("Module 2 GNN — outfit compatibility: real vs. corrupted (one-swap)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)


def _on_white(png_path):
    """Composite an RGBA garment crop onto white so transparency reads cleanly."""
    from PIL import Image
    im = Image.open(png_path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im).convert("RGB")


def _photo_figure(photo_path, outfit_dir, garments, score, corr, out_path):
    """End-to-end single-photo demo: input photo -> detected garments -> Harmoniousness."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    n = len(garments)
    widths = ([2.2] if photo_path else []) + [1.0] * n + [1.8]
    fig, axes = plt.subplots(1, len(widths), figsize=(2.6 * len(widths), 4.4),
                             gridspec_kw={"width_ratios": widths})
    col = 0
    if photo_path:
        axes[col].imshow(Image.open(photo_path).convert("RGB"))
        axes[col].set_title("Input photo", fontsize=11, fontweight="bold")
        axes[col].axis("off")
        col += 1

    for g in garments:
        a = g.get("attributes", {}) or {}
        cap = f"{g.get('broad_category', '?')}\n{(a.get('color') or {}).get('name', '?')}"
        axes[col].imshow(_on_white(os.path.join(outfit_dir, g["file"])))
        axes[col].set_title(cap, fontsize=9)
        axes[col].axis("off")
        col += 1

    ax = axes[col]
    ax.bar([0], [score], width=0.6, color="#2e7d32")
    if corr is not None:
        ax.axhline(corr, color="#c62828", linestyle="--", linewidth=1.2)
        ax.text(0, corr + 1.5, f"{corr:.0f} if 1 item swapped", ha="center",
                fontsize=7.5, color="#c62828")
    ax.text(0, score + 2, f"{score:.0f}", ha="center", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_ylabel("percentile vs real outfits")
    ax.set_title("Harmoniousness /100", fontsize=10, fontweight="bold")

    fig.suptitle("End-to-end demo — photo → garments → GNN Harmoniousness score",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)


if __name__ == "__main__":
    main()

