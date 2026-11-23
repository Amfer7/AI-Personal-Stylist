"""Visual calibration of the near-duplicate threshold t_dup (D8).

For a few garment types, draws pool pairs whose CLIP cosine falls in each band (0.90 ... 0.99) and
that also pass the rest of the near-duplicate rule (matching colour histograms and crop sizes, see
candidate_pool.is_duplicate), i.e. pairs the dedupe would merge at that threshold. It renders one
contact sheet per type: one row per band, each cell = the two crops side by side. A
reviewer marks the lowest band where pairs look like "the same item"; that band becomes t_dup in
m4_settings.json (status "calibrated"), followed by  python Module4/scripts/build_cache.py --force.

    python Module4/scripts/calibrate_dup.py [--types shoe bag pants jacket dress] [--pairs 6]
Sheets go to <cache_dir>/dup_calibration/<type>.png (gitignored).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from candidate_pool import CandidatePool  # noqa: E402
from m2_bridge import apply_vram_cap, pick_device  # noqa: E402
from m4_settings import M4Settings  # noqa: E402
from swap_groups import TYPE_ID  # noqa: E402

BANDS = (0.90, 0.93, 0.95, 0.97, 0.99)
HALF_WIDTH = 0.005


def sample_band_pairs(emb, rows, lo, hi, n_pairs, rng, max_anchors=2000, pool=None):
    """Up to n_pairs (row_a, row_b, cosine) with lo <= cosine < hi, anchors drawn at random. With a
    pool, pairs must also match on colour and size (the rest of the near-duplicate rule)."""
    rows_t = torch.as_tensor(rows, device=emb.device)
    E = emb[rows_t]
    out = []
    for a in rng.permutation(len(rows))[:max_anchors]:
        sims = E @ E[a]
        sims[a] = -1
        ok = (sims >= lo) & (sims < hi)
        if pool is not None:
            r = int(rows[a])
            t_dup, max_l1, max_ratio = pool.dup_rule
            pool.dup_rule = (-2.0, max_l1, max_ratio)               # colour + size only; the band sets cosine
            try:
                ok &= pool.dup_mask(rows_t, pool.emb[r], pool.colour[r], float(pool.area[r]))
            finally:
                pool.dup_rule = (t_dup, max_l1, max_ratio)
        hits = torch.nonzero(ok).squeeze(1)
        if len(hits):
            b = int(hits[int(rng.integers(len(hits)))])
            out.append((int(rows[a]), int(rows[b]), float(sims[b])))
            if len(out) == n_pairs:
                break
    return out


def _on_white(path):
    from PIL import Image
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im).convert("RGB")


def render_sheet(pool, settings, type_name, pairs_by_band, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = max(1, max(len(p) for p in pairs_by_band.values())) * 2
    fig, axes = plt.subplots(len(pairs_by_band), cols, figsize=(1.4 * cols, 1.8 * len(pairs_by_band)),
                             squeeze=False)
    for i, (band, pairs) in enumerate(pairs_by_band.items()):
        for ax in axes[i]:
            ax.axis("off")
        axes[i][0].set_title(f"cos ≈ {band:.2f}", fontsize=8, loc="left")
        for j, (a, b, cos) in enumerate(pairs):
            for k, r in enumerate((a, b)):
                axes[i][2 * j + k].imshow(_on_white(settings.abs(pool.crop_path(r))))
            axes[i][2 * j].set_xlabel(f"{cos:.3f}", fontsize=7)
    fig.suptitle(f"Near-duplicate calibration: {type_name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--types", nargs="+", default=["shoe", "bag", "pants", "jacket", "dress"])
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    s = M4Settings.load()
    dev = pick_device()
    apply_vram_cap(s.vram_gb, dev)
    pool = CandidatePool.load(s, dev)
    out_dir = s.abs(os.path.join(s.cache_dir, "dup_calibration"))
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for t in args.types:
        rows = np.nonzero(pool.type == TYPE_ID[t])[0]
        rows = rows[[os.path.exists(s.abs(pool.crop_path(r))) for r in rows]]   # crops are not all local
        bands = {b: sample_band_pairs(pool.emb, rows, b - HALF_WIDTH, b + HALF_WIDTH, args.pairs, rng, pool=pool)
                 for b in BANDS}
        path = os.path.join(out_dir, f"{t}.png")
        render_sheet(pool, s, t, bands, path)
        print(f"[dup-cal] {t}: {[len(v) for v in bands.values()]} pairs per band -> {path}")


if __name__ == "__main__":
    main()
