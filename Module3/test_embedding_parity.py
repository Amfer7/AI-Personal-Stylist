"""
test_embedding_parity.py

Gate for the LEAN batched segment->embed path (optimization "A") added to
pinterest_garment_segmentation.py.

The lean path skips the two CLIP passes the Pinterest trend run never uses
(subcategory classification + attribute analysis) and embeds all crops of an
image in ONE batched CLIP forward instead of one crop at a time. That is only
safe if the batched embedding is identical (within fp16 tolerance) to the
canonical per-crop clip_extract_embeddings.embed_image — otherwise the Pinterest
cluster centroids and demo-time user-photo embeddings would live in subtly
different vector spaces and cosine matching would silently degrade.

This test asserts that parity on real Pinterest crops. It must pass before the
full lean run is launched.

Run directly:  python test_embedding_parity.py --metadata_csv ../PinterestLatest/pinterest_dataset/metadata.csv
Run on pytest: pytest test_embedding_parity.py
"""

import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

import clip_extract_embeddings as cee
from pinterest_garment_segmentation import extract_crops, embed_crops_batched

DEFAULT_METADATA_CSV = os.path.join(
    os.path.dirname(__file__), "..", "PinterestLatest", "pinterest_dataset", "metadata.csv"
)


def _resolve(base_dir: str, rel: str) -> str:
    p = str(rel).replace("\\", "/")
    return p if os.path.isabs(p) else os.path.join(base_dir, p)


def collect_crops(metadata_csv: str, target: int = 20, device=None):
    """Segment the first handful of Pinterest images to collect ~`target` real
    garment crops to test parity on."""
    df = pd.read_csv(metadata_csv).dropna(subset=["image_path"]).reset_index(drop=True)
    base = os.path.dirname(os.path.abspath(metadata_csv))
    crops = []
    i = 0
    while len(crops) < target and i < len(df):
        path = _resolve(base, df.loc[i, "image_path"])
        try:
            img = Image.open(path).convert("RGB")
            crops.extend(extract_crops(img, device=device))
        except Exception:
            pass
        i += 1
    return crops[:target]


def test_lean_embeddings_match_per_crop(metadata_csv=None, device=None, atol=2e-3):
    """pytest entry point: batched lean embeddings == per-crop embed_image."""
    metadata_csv = metadata_csv or DEFAULT_METADATA_CSV
    crops = collect_crops(metadata_csv, device=device)
    assert len(crops) >= 5, f"need at least 5 crops to test parity, got {len(crops)}"

    batched = embed_crops_batched(crops, device=device)
    per_crop = np.stack([cee.embed_image(c["image_crop"], device=device) for c in crops])

    max_abs_diff = float(np.abs(batched - per_crop).max())
    # both rows are unit-norm, so row-wise dot == cosine similarity
    min_row_cosine = float((batched * per_crop).sum(axis=1).min())
    print(f"crops={len(crops)}  max_abs_diff={max_abs_diff:.2e}  "
          f"min_row_cosine={min_row_cosine:.6f}")

    assert np.allclose(batched, per_crop, atol=atol), (
        f"PARITY FAILED: lean batched embeddings differ from embed_image "
        f"(max_abs_diff={max_abs_diff:.2e} > atol={atol:.0e}). Do NOT run the "
        f"lean path — fall back to --mode full.")
    assert min_row_cosine > 0.9999, (
        f"PARITY FAILED: min row cosine {min_row_cosine:.6f} < 0.9999")
    print("PARITY OK — lean batched path matches embed_image; safe to run.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata_csv", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    test_lean_embeddings_match_per_crop(metadata_csv=args.metadata_csv, device=args.device)
