"""
pinterest_garment_segmentation.py

Extends the trend model from whole-photo scoring to PER-GARMENT scoring.

For every Pinterest photo, this:
  1. Runs Prateek's real fashion_segmenter.py (SegFormer-B2-Clothes + CLIP
     zero-shot subcategory classification) to detect individual garments
     (top, bottom, dress, shoe, accessory, ...) in the photo.
  2. Embeds each garment crop with the shared clip_extract_embeddings.py
     (the exact same embedding code used everywhere else in the project).
  3. Records one row per GARMENT (not per photo): image_id, date_ym,
     broad_category, specific_category, confidence, embedding.

Output: a garment-level manifest (CSV) + an aligned embeddings.npy — the
per-garment equivalent of the earlier pinterest_embeddings.npy.

IMPORTANT — read before running the full dataset:
Segmentation (SegFormer) is a much heavier per-image cost than the plain
CLIP-embedding-only pass used for the whole-photo trend model. Module 2's own
README flagged the same lesson for Fashion144K: pilot on a small --limit
first to check garment-detection rate and timing before committing to the
full ~39K-image run. Do the same here.

Usage (Colab):
    python pinterest_garment_segmentation.py \
        --metadata_csv /path/to/metadata.csv \
        --output_dir /content/drive/MyDrive/Poster/garment_data \
        --limit 1000        # PILOT FIRST — remove --limit only after checking results
"""

import argparse
import json
import os
import pickle
import sys
import time
from typing import List, Optional

import numpy as np
import pandas as pd
from PIL import Image

# fashion_segmenter.py and clip_extract_embeddings.py must be importable
# (same folder as this script, or added to sys.path beforehand).
import fashion_segmenter
import clip_extract_embeddings as cee

CHECKPOINT_FLUSH_EVERY = 200        # save progress this often, not just at the end
MAX_CONSECUTIVE_FAILURES = 25       # circuit breaker — mirrors Module 2's own lesson
MIN_PIXEL_AREA = 500                # matches fashion_segmenter's own default


def load_checkpoint(checkpoint_path: str):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            return pickle.load(f)
    return {"completed_ids": set(), "records": [], "embeddings": []}


def save_checkpoint(checkpoint_path: str, state: dict):
    tmp_path = checkpoint_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp_path, checkpoint_path)  # atomic, avoids a corrupt checkpoint on crash


def process_dataset(
    metadata_csv: str,
    output_dir: str,
    image_col: str = "image_path",
    date_col: str = "date_ym",
    limit: Optional[int] = None,
    device: Optional[str] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "segmentation_checkpoint.pkl")
    fail_log_path = os.path.join(output_dir, "segmentation_failures.log")

    df = pd.read_csv(metadata_csv)
    df = df.dropna(subset=[image_col, date_col]).reset_index(drop=True)
    base_dir = os.path.dirname(os.path.abspath(metadata_csv))
    df[image_col] = df[image_col].astype(str).str.replace("\\", "/", regex=False)
    df[image_col] = df[image_col].apply(
        lambda p: p if os.path.isabs(p) else os.path.join(base_dir, p)
    )

    if limit:
        df = df.head(limit)
        print(f"PILOT MODE: limited to first {limit} images.")

    state = load_checkpoint(checkpoint_path)
    completed = state["completed_ids"]
    records = state["records"]
    embeddings = state["embeddings"]
    print(f"Resuming: {len(completed)} images already processed, "
          f"{len(records)} garment records so far.")

    todo = [(i, row) for i, row in df.iterrows() if str(i) not in completed]
    print(f"{len(todo)} images remaining out of {len(df)} total.")

    consecutive_failures = 0
    start_time = time.time()

    with open(fail_log_path, "a") as fail_log:
        for n, (idx, row) in enumerate(todo):
            image_path = row[image_col]
            try:
                image = Image.open(image_path).convert("RGB")
                garments = fashion_segmenter.extract_garment_segments(
                    image, min_pixel_area=MIN_PIXEL_AREA, device=device
                )

                for g in garments:
                    emb = cee.embed_image(g["image_crop"], device=device)
                    records.append({
                        "source_idx": idx,
                        "image_path": image_path,
                        "date_ym": row[date_col],
                        "query": row.get("query", None),
                        "broad_category": g["broad_category"],
                        "specific_category": g["specific_category"],
                        "confidence": g["confidence"],
                        "pixel_count": g["pixel_count"],
                    })
                    embeddings.append(emb)

                completed.add(str(idx))
                consecutive_failures = 0

            except Exception as e:
                consecutive_failures += 1
                fail_log.write(f"{image_path}\t{repr(e)}\n")
                fail_log.flush()
                completed.add(str(idx))  # don't retry a permanently-broken file forever
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"\nABORTING: {MAX_CONSECUTIVE_FAILURES} consecutive failures — "
                          f"something is systemically broken (bad path? OOM? model not "
                          f"loading?), not just a few bad images. Check {fail_log_path}.")
                    save_checkpoint(checkpoint_path, {
                        "completed_ids": completed, "records": records, "embeddings": embeddings
                    })
                    sys.exit(1)

            if (n + 1) % CHECKPOINT_FLUSH_EVERY == 0:
                save_checkpoint(checkpoint_path, {
                    "completed_ids": completed, "records": records, "embeddings": embeddings
                })
                elapsed = time.time() - start_time
                rate = (n + 1) / elapsed
                remaining = (len(todo) - n - 1) / rate if rate > 0 else float("inf")
                print(f"  [{n+1}/{len(todo)}] {rate:.2f} img/s, "
                      f"~{remaining/60:.1f} min remaining, {len(records)} garments so far")

    save_checkpoint(checkpoint_path, {
        "completed_ids": completed, "records": records, "embeddings": embeddings
    })

    manifest_df = pd.DataFrame(records)
    manifest_path = os.path.join(output_dir, "garment_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)

    embeddings_arr = np.array(embeddings, dtype=np.float32)
    embeddings_path = os.path.join(output_dir, "garment_embeddings.npy")
    np.save(embeddings_path, embeddings_arr)

    print(f"\nDONE. {len(manifest_df)} garments from {len(completed)} images.")
    print(f"Manifest: {manifest_path}")
    print(f"Embeddings: {embeddings_path} shape={embeddings_arr.shape}")
    print("\nGarments per broad_category:")
    print(manifest_df["broad_category"].value_counts())

    return manifest_df, embeddings_arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=None,
                         help="Process only the first N images. STRONGLY recommended "
                              "for a pilot run before committing to the full dataset.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    process_dataset(
        metadata_csv=args.metadata_csv,
        output_dir=args.output_dir,
        limit=args.limit,
        device=args.device,
    )


if __name__ == "__main__":
    main()
