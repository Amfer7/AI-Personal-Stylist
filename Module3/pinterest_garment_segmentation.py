"""
pinterest_garment_segmentation.py

Extends the trend model from whole-photo scoring to PER-GARMENT scoring.

For every Pinterest photo, this:
  1. Runs Prateek's real fashion_segmenter.py (SegFormer-B2-Clothes) to detect
     individual garment regions (top, bottom, dress, shoe, accessory, ...).
  2. Embeds each garment crop with the shared clip_extract_embeddings.py
     (the exact same embedding code used everywhere else in the project).
  3. Records one row per GARMENT (not per photo): source_idx, image_path,
     date_ym, query, broad_category, specific_category, confidence, pixel_count.

Output: a garment-level manifest (CSV) + an aligned garment_embeddings.npy — the
per-garment equivalent of the earlier pinterest_embeddings.npy.

-----------------------------------------------------------------------------
Two modes (default: lean):

  --mode lean  (optimization "A", the default)
      The trend pipeline only ever consumes the EMBEDDING + broad_category.
      So the lean path skips the two CLIP passes it never uses:
        * subcategory classification (classify_crop_clip) and
        * attribute analysis (analyze_garment)
      and embeds ALL crops of an image in ONE batched CLIP forward instead of
      one crop at a time. broad_category is derived directly from the SegFormer
      region label (provably identical to the full path's broad_category, since
      every region label maps to exactly one broad category). This cuts the
      CLIP image-encodes per crop from ~3 to 1 (batched) and roughly halves the
      run. Tradeoff: specific_category is the coarse SegFormer label (not the
      fine t-shirt/sweater/jacket), and the softmax `confidence` is not
      produced (set to 1.0). The trend model uses neither.
      SAFETY: the lean embedding is verified identical to embed_image by
      test_embedding_parity.py — run it before a full lean run.

  --mode full
      The original behaviour: the vendored extract_garment_segments (CLIP
      classify + attributes) + per-crop embed_image. Preserves fine-grained
      specific_category and the CLIP confidence. Slower. Kept as a fallback.
-----------------------------------------------------------------------------

Checkpointing (both modes) is APPEND-ONLY: the manifest CSV and a raw float32
embeddings file are appended as images complete, and the manifest IS the
resume checkpoint (completed images are recomputed from its source_idx column).
This avoids the earlier design's O(n^2) re-pickling of the entire growing
records+embeddings list on every flush (which also caused a OneDrive write
storm). On finish, the raw float32 file is converted to garment_embeddings.npy.

Usage:
    # PILOT FIRST — check garment-detection rate and timing on a small --limit:
    python pinterest_garment_segmentation.py \
        --metadata_csv ../PinterestLatest/pinterest_dataset/metadata.csv \
        --output_dir   ../PinterestLatest/garment_data \
        --limit 1000

    # then remove --limit for the full run (resumable if interrupted).
"""

import argparse
import csv
import os
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

CHECKPOINT_FLUSH_EVERY = 200        # flush buffered rows/embeddings this often
MAX_CONSECUTIVE_FAILURES = 25       # circuit breaker — mirrors Module 2's lesson
MIN_PIXEL_AREA = 500                # matches fashion_segmenter's own default
EMB_DIM = 512                       # CLIP ViT-B/32 embedding dimensionality

MANIFEST_COLUMNS = [
    "source_idx", "image_path", "date_ym", "query",
    "broad_category", "specific_category", "confidence", "pixel_count",
]

# Each SegFormer region label maps to exactly one broad category. This is the
# SAME broad_category the full path (fashion_segmenter.extract_garment_segments)
# produces, because every fine subcategory candidate for a given region stays
# within one broad bucket (all Upper-clothes subcats -> top, all Pants -> bottom,
# etc.), so the fine CLIP classification never changes the broad result.
SEG_LABEL_TO_BROAD = {
    "Upper-clothes": "top",
    "Pants": "bottom",
    "Skirt": "bottom",
    "Dress": "dress",
    "Shoe": "accessory",      # merged left/right shoe
    "Bag": "accessory",
    "Hat": "accessory",
    "Sunglasses": "accessory",
    "Belt": "accessory",
    "Scarf": "accessory",
}


# ---------------------------------------------------------------------------
# Lean path (optimization "A")
# ---------------------------------------------------------------------------

def extract_crops(image: Image.Image, min_pixel_area: int = MIN_PIXEL_AREA,
                  device: Optional[str] = None) -> List[dict]:
    """SegFormer-parse an image and return per-garment transparent crops WITHOUT
    running any CLIP (no subcategory classification, no attributes).

    Reuses the vendored fashion_segmenter.segment_clothing() for the actual
    SegFormer forward, and replicates the exact crop construction used by the
    vendored extract_garment_segments (full-mask alpha, tight bbox crop) so the
    resulting crops are identical to the full path's crops.
    """
    fs = fashion_segmenter
    pred_mask, id2label = fs.segment_clothing(image, device=device)
    img_rgba = np.array(image.convert("RGBA"))

    target_classes = {
        idx: label for idx, label in id2label.items()
        if label in fs.FINE_GRAINED_CANDIDATES
    }
    shoe_indices = [idx for idx, label in target_classes.items() if "shoe" in label.lower()]

    merged_classes = []
    for class_idx, class_label in target_classes.items():
        if class_idx in shoe_indices:
            continue
        merged_classes.append((class_label, [class_idx]))
    if shoe_indices:
        merged_classes.append(("Shoe", shoe_indices))

    crops = []
    for label_name, class_indices in merged_classes:
        binary_mask = np.isin(pred_mask, class_indices).astype(np.uint8)
        pixel_count = int(np.sum(binary_mask))
        if pixel_count < min_pixel_area:
            continue

        y_indices, x_indices = np.where(binary_mask > 0)
        x1, y1 = int(x_indices.min()), int(y_indices.min())
        x2, y2 = int(x_indices.max()) + 1, int(y_indices.max()) + 1

        garment_rgba = img_rgba.copy()
        garment_rgba[:, :, 3] = binary_mask * 255
        crop_image = Image.fromarray(garment_rgba).crop((x1, y1, x2, y2))

        crops.append({
            "broad_category": SEG_LABEL_TO_BROAD.get(label_name, "accessory"),
            "specific_category": label_name.lower().replace("-", "_"),
            "image_crop": crop_image,
            "pixel_count": pixel_count,
            "bbox": [x1, y1, x2, y2],
        })
    return crops


def embed_crops_batched(crops: List[dict], device: Optional[str] = None) -> np.ndarray:
    """Embed a list of crops in ONE batched CLIP forward.

    Mirrors clip_extract_embeddings.embed_image EXACTLY (same _load_rgb
    white-composite, same preprocess instance, same encode_image + float +
    L2-normalize) — only batched. Parity with the per-crop path is asserted by
    test_embedding_parity.py. Returns (N, 512) float32, rows aligned to `crops`.
    """
    import torch
    if not crops:
        return np.empty((0, EMB_DIM), dtype=np.float32)
    model, preprocess, dev = cee.get_clip_model(device=device)
    tensors = torch.stack([preprocess(cee._load_rgb(c["image_crop"])) for c in crops]).to(dev)
    with torch.no_grad():
        embs = model.encode_image(tensors)
    embs = embs.float()
    embs = embs / embs.norm(dim=-1, keepdim=True)
    return embs.cpu().numpy().astype(np.float32)


def process_image_lean(image: Image.Image, device: Optional[str] = None):
    """Returns (rows_without_idx_meta, embeddings) for one image, lean mode."""
    crops = extract_crops(image, device=device)
    embs = embed_crops_batched(crops, device=device)
    rows = [{
        "broad_category": c["broad_category"],
        "specific_category": c["specific_category"],
        "confidence": 1.0,            # seg-only; no CLIP classifier in lean mode
        "pixel_count": c["pixel_count"],
    } for c in crops]
    return rows, embs


def process_image_full(image: Image.Image, device: Optional[str] = None):
    """Returns (rows, embeddings) for one image, original per-crop behaviour."""
    garments = fashion_segmenter.extract_garment_segments(
        image, min_pixel_area=MIN_PIXEL_AREA, device=device
    )
    rows, embs = [], []
    for g in garments:
        embs.append(cee.embed_image(g["image_crop"], device=device))
        rows.append({
            "broad_category": g["broad_category"],
            "specific_category": g["specific_category"],
            "confidence": g["confidence"],
            "pixel_count": g["pixel_count"],
        })
    embs_arr = (np.stack(embs).astype(np.float32) if embs
                else np.empty((0, EMB_DIM), dtype=np.float32))
    return rows, embs_arr


# ---------------------------------------------------------------------------
# Append-only storage (optimization "B")
# ---------------------------------------------------------------------------

def _raw_emb_rows(raw_path: str) -> int:
    if not os.path.exists(raw_path):
        return 0
    return os.path.getsize(raw_path) // (EMB_DIM * 4)


def _reconcile(manifest_path: str, raw_path: str) -> set:
    """Align the manifest CSV and the raw embeddings file to the same length
    (defends against a crash mid-flush), and return the set of already-completed
    source_idx values so processing can resume."""
    if not os.path.exists(manifest_path):
        # start clean: drop any orphan raw embeddings too
        if os.path.exists(raw_path):
            os.remove(raw_path)
        return set()

    man = pd.read_csv(manifest_path)
    n_man = len(man)
    n_emb = _raw_emb_rows(raw_path)
    n = min(n_man, n_emb)

    if n_man != n_emb:
        print(f"  [resume] manifest has {n_man} rows, embeddings {n_emb} — "
              f"trimming both to {n} for alignment.")
        if n_man > n:
            man = man.iloc[:n]
            man.to_csv(manifest_path, index=False)
        if n_emb > n:
            with open(raw_path, "r+b") as f:
                f.truncate(n * EMB_DIM * 4)

    completed = set(int(x) for x in man["source_idx"].iloc[:n].tolist())
    return completed


def process_dataset(
    metadata_csv: str,
    output_dir: str,
    mode: str = "lean",
    image_col: str = "image_path",
    date_col: str = "date_ym",
    limit: Optional[int] = None,
    device: Optional[str] = None,
    flush_every: int = CHECKPOINT_FLUSH_EVERY,
):
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "garment_manifest.csv")
    raw_path = os.path.join(output_dir, "garment_embeddings.f32")
    npy_path = os.path.join(output_dir, "garment_embeddings.npy")
    fail_log_path = os.path.join(output_dir, "segmentation_failures.log")

    process_image = process_image_lean if mode == "lean" else process_image_full
    print(f"MODE: {mode}")

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

    completed = _reconcile(manifest_path, raw_path)
    print(f"Resuming: {len(completed)} images already processed.")

    todo = [(i, row) for i, row in df.iterrows() if int(i) not in completed]
    print(f"{len(todo)} images remaining out of {len(df)} total.")

    manifest_is_new = not os.path.exists(manifest_path)
    manifest_f = open(manifest_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(manifest_f, fieldnames=MANIFEST_COLUMNS)
    if manifest_is_new:
        writer.writeheader()
    raw_f = open(raw_path, "ab")

    row_buffer: List[dict] = []
    emb_buffer: List[np.ndarray] = []

    def flush():
        if row_buffer:
            writer.writerows(row_buffer)
            manifest_f.flush()
            np.concatenate(emb_buffer, axis=0).astype(np.float32).tofile(raw_f)
            raw_f.flush()
            row_buffer.clear()
            emb_buffer.clear()

    consecutive_failures = 0
    n_garments = 0
    start_time = time.time()

    try:
        with open(fail_log_path, "a", encoding="utf-8") as fail_log:
            for n, (idx, row) in enumerate(todo):
                image_path = row[image_col]
                try:
                    image = Image.open(image_path).convert("RGB")
                    rows, embs = process_image(image, device=device)

                    for r in rows:
                        r.update({
                            "source_idx": int(idx),
                            "image_path": image_path,
                            "date_ym": row[date_col],
                            "query": row.get("query", None),
                        })
                    row_buffer.extend(rows)
                    if len(embs):
                        emb_buffer.append(embs)
                    n_garments += len(rows)
                    consecutive_failures = 0

                except Exception as e:
                    consecutive_failures += 1
                    fail_log.write(f"{image_path}\t{repr(e)}\n")
                    fail_log.flush()
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        print(f"\nABORTING: {MAX_CONSECUTIVE_FAILURES} consecutive "
                              f"failures — something is systemically broken (bad "
                              f"path? OOM? model not loading?), not just a few bad "
                              f"images. Check {fail_log_path}.")
                        flush()
                        sys.exit(1)
                    continue

                if (n + 1) % flush_every == 0:
                    flush()
                    elapsed = time.time() - start_time
                    rate = (n + 1) / elapsed
                    remaining = (len(todo) - n - 1) / rate if rate > 0 else float("inf")
                    print(f"  [{n+1}/{len(todo)}] {rate:.2f} img/s, "
                          f"~{remaining/60:.1f} min remaining, {n_garments} new garments")
        flush()
    finally:
        manifest_f.close()
        raw_f.close()

    # Convert the append-only raw float32 file into the aligned .npy the trend
    # model / stability test expect, and verify row alignment with the manifest.
    total_rows = _raw_emb_rows(raw_path)
    embeddings_arr = np.fromfile(raw_path, dtype=np.float32).reshape(total_rows, EMB_DIM)
    np.save(npy_path, embeddings_arr)

    manifest_df = pd.read_csv(manifest_path)
    assert len(manifest_df) == len(embeddings_arr), (
        f"manifest ({len(manifest_df)}) / embeddings ({len(embeddings_arr)}) "
        f"row mismatch after run")

    print(f"\nDONE. {len(manifest_df)} garments.")
    print(f"Manifest:   {manifest_path}")
    print(f"Embeddings: {npy_path} shape={embeddings_arr.shape}")
    print("\nGarments per broad_category:")
    print(manifest_df["broad_category"].value_counts())
    return manifest_df, embeddings_arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=["lean", "full"], default="lean",
                        help="lean (default, optimization A) skips unused CLIP "
                             "classify+attributes and batches embedding; full is "
                             "the original per-crop path.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N images. STRONGLY "
                             "recommended for a pilot before the full dataset.")
    parser.add_argument("--flush_every", type=int, default=CHECKPOINT_FLUSH_EVERY)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    process_dataset(
        metadata_csv=args.metadata_csv,
        output_dir=args.output_dir,
        mode=args.mode,
        limit=args.limit,
        device=args.device,
        flush_every=args.flush_every,
    )


if __name__ == "__main__":
    main()
