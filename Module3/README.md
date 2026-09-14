# Per-Garment Trend Model — Extension

Extends the whole-photo trend model to score **individual garments**
(top, bottom, dress, shoe, accessory) separately, using Prateek's real
segmentation and embedding code — pulled directly from
https://github.com/Amfer7/AI-Personal-Stylist, not reconstructed from a
description.

## Files

| File | What it is |
|---|---|
| `fashion_segmenter.py` | Prateek's real segmentation code (verbatim, from the repo) |
| `attribute_analyzer.py` | Required dependency of the above (verbatim) — we don't use its attribute output, just need it importable |
| `clip_extract_embeddings.py` | Prateek's real embedding code (verbatim) — **byte-identical to `Module1/clip_extract_embeddings.py`** (same 512-D ViT-B/32, L2-normalized convention). `test_module1_sync.py` enforces this stays true |
| `trend_model.py` | Original whole-photo `TrendModel` (unchanged) **+** new `PerCategoryTrendModel`, which fits one `TrendModel` per garment category by reusing the existing, already-tested logic — no duplicated clustering/scoring code |
| `pinterest_garment_segmentation.py` | **New.** Batch-segments the Pinterest dataset into individual garments + embeds each crop. Resumable, checkpointed, circuit-breaker on repeated failures (mirrors Module 2's own documented lessons) |
| `cluster_stability_test.py` | **New.** 80/20 split validation via Adjusted Rand Index — the correct stability test for a clustering method (K-Means has no loss/epochs to validate the way the GNN does) |
| `test_real_photo_per_garment.py` | **New.** Segments a real photo, scores each garment separately, reports the weak link |
| `test_module1_sync.py` | **New.** Guards the three vendored files above against silent drift — asserts each is content-identical to its `Module1/` original (line endings ignored). Run `python test_module1_sync.py` (or under pytest) after touching Module 1's perception code |
| `gradio_demo.py` | Unchanged — still does whole-photo scoring. Per-garment demo not yet wired into a UI (see below) |

## ⚠️ Before running on the full dataset: PILOT FIRST

Segmentation (SegFormer) is much heavier per-image than the CLIP-only pass
the whole-photo model used. This is the same lesson your teammates already
documented for Fashion144K in Module 2 — pilot small, check the results,
then commit to the full run.

```python
!python pinterest_garment_segmentation.py \
    --metadata_csv /path/to/metadata.csv \
    --output_dir /content/drive/MyDrive/Poster/garment_data \
    --limit 1000
```

Check the printed category breakdown and the failure log
(`garment_data/segmentation_failures.log`) before removing `--limit`.

## Full pipeline, in order

```python
# 1. PILOT (see above) — check garment detection rate and timing first

# 2. Full run — remove --limit once the pilot looks right. Resumable if interrupted.
!python pinterest_garment_segmentation.py \
    --metadata_csv /path/to/metadata.csv \
    --output_dir /content/drive/MyDrive/Poster/garment_data

# 3. Fit the per-category trend model
from trend_model import build_per_category_from_manifest
model = build_per_category_from_manifest(
    manifest_csv="/content/drive/MyDrive/Poster/garment_data/garment_manifest.csv",
    embeddings_path="/content/drive/MyDrive/Poster/garment_data/garment_embeddings.npy",
)
model.save("/content/drive/MyDrive/Poster/per_category_trend_model.pkl")
print(model.summary())

# 4. Cluster stability test (the correctly-scoped alternative to "train/test split
#    with epochs" — K-Means has no loss curve, but stability under resampling
#    IS a meaningful, standard clustering validation)
!python cluster_stability_test.py \
    --manifest_csv /content/drive/MyDrive/Poster/garment_data/garment_manifest.csv \
    --embeddings /content/drive/MyDrive/Poster/garment_data/garment_embeddings.npy

# 5. Test against a real photo
!python test_real_photo_per_garment.py \
    --image /path/to/some/outfit_photo.jpg \
    --model /content/drive/MyDrive/Poster/per_category_trend_model.pkl
```

## Why not "train in epochs like the GNN"

The GNN trains a neural network with gradient descent — it has a loss that
improves over epochs, so train/val/test splits and epoch counts are
meaningful. K-Means has no loss to converge and nothing to "train" epoch-
over-epoch; it's a one-shot fit. The correctly-scoped equivalent test is
**stability under resampling** (`cluster_stability_test.py`, via Adjusted
Rand Index) — does the clustering reflect real structure, or would a
different random subset of the same size give different clusters?
Separately, `temporal_holdout_validation.py` (already built earlier) tests
whether the *growth signal* has predictive validity forward in time — a
different, complementary question from cluster stability.

## Dataset caching — avoid re-unzipping every session

Extract the dataset to Drive **once**, permanently:

```python
!unzip -q -n /content/drive/MyDrive/Poster/pinterest_final_dataset.zip \
    "*/pinterest_dataset/*" -d /content/drive/MyDrive/Poster/pinterest_dataset_extracted
```

Every future session, point `DATASET_DIR` straight at that Drive path —
no unzip step needed again, ever.

## Integration point for Module 4 (recommendation engine)

`PerCategoryTrendModel.score_outfit(garments)` returns per-garment
trendiness scores, an outfit-average, and a `weak_link_category` — the
trend-model analogue of the GNN's weak-link/per-item harmony scores. Module
4 combines both signals when evaluating a candidate garment swap.
