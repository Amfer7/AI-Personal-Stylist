# Module 1 — Changes Log

Adds the **fashion-theory features and per-garment attributes** the PDF's Module 1
promises ("unified feature representation = embeddings + attributes + theory metrics"),
which the original code did not implement — it only did SegFormer segmentation + CLIP
embeddings. This was done on the **SegFormer path** (the chosen path), with no training
and no new model downloads (reuses the already-loaded CLIP ViT-B/32).

## Summary

| Item | Before | After |
|---|---|---|
| Per-garment output | embedding (`.npy`) + basic metadata | + `attributes` block (colour, silhouette, pattern, fabric, fit, formality) |
| Outfit-level output | none | `outfit_features.json` (hue contrast, volume balance, formality coherence) |
| New model downloads | — | none (reuses loaded CLIP) |
| Added runtime cost | — | ~34 ms/garment on CPU after a one-time prompt encode (~0.4 s); near-zero on GPU |

## New file: `attribute_analyzer.py`

Self-contained analyzer (~290 lines). Two families of signals, both derived from data
the SegFormer path already produces (the transparent RGBA crop — its **alpha channel is
the pixel-accurate garment mask**):

1. **Pixel-based (deterministic, model-free):**
   - `analyze_color(rgb, mask)` — dominant RGB (k-means) → HSV + human-readable name,
     measured on **fabric pixels only** (ignores background).
   - `analyze_silhouette(mask)` — shape class (`fitted` / `boxy` / `a_line` / `tapered`)
     from mask geometry (aspect ratio, extent, top-vs-bottom width profile).

2. **Semantic (CLIP zero-shot, reuses ViT-B/32):**
   - `analyze_semantic(...)` — `pattern` (solid/striped/checked/floral/graphic/dotted),
     `fabric` (denim/cotton/leather/knit/silk/corduroy), `fit` (slim/regular/oversized),
     `formality` (casual/smart_casual/formal) + a continuous `formality_score` (0–100).
   - Text prompts are encoded **once and cached** (`_TEXT_CACHE`), so per-garment cost is
     a single image encode (which the pipeline already does for subcategory classification).

3. **Outfit-level fashion-theory features:**
   - `compute_outfit_features(garments)` → `hue_contrast`, `top_bottom_hue_distance`
     (circular hue geometry), `volume_balance` + `top_bottom_area_ratio` (relative garment
     areas), `formality_coherence` (spread of formality across the outfit).

Degrades gracefully: if CLIP is unavailable, colour + silhouette still populate.

Public entry point: `analyze_garment(crop_rgba, clip_model=None, clip_preprocess=None, device="cpu")`.

## Edited: `fashion_segmenter.py`

- Import `analyze_garment`, `compute_outfit_features` (with the same local/package
  fallback pattern as the rest of the module).
- `extract_garment_segments()`: fetch the CLIP handles once via `get_models(...)`, call
  `analyze_garment(...)` per garment, attach the result as `attributes` on each garment dict.
- `process_image_segformer()`:
  - each `metadata.json` entry now carries an additive `attributes` field;
  - after the garment loop, `compute_outfit_features(...)` is written to a **separate
    `outfit_features.json`** (see design note below).

## Edited: `__init__.py`

Exported `analyze_garment`, `analyze_color`, `analyze_silhouette`, `compute_outfit_features`.

## Edited: `test_pipeline.py`

Added 5 deterministic (no-download) tests: colour ignores background pixels, A-line and
boxy silhouette classification, circular hue distance, and outfit-level aggregation.

## Output contract

`metadata.json` stays a **JSON list** (attributes added per item) — backward-compatible,
so the downstream GNN loader that does `json.load(f)` → iterate keeps working unchanged.
Outfit-level metrics go to the new `outfit_features.json`.

Per garment, `attributes` now looks like:
```json
"attributes": {
  "color": {"dominant_rgb": [233,196,203], "hue": 348.8, "saturation": 0.158,
             "value": 0.915, "name": "red"},
  "silhouette": "boxy", "aspect_ratio": 1.165, "extent": 0.772, "area_px": 53097,
  "pattern": "dotted", "pattern_conf": 0.431,
  "fabric": "silk", "fabric_conf": 0.5799,
  "fit": "regular", "fit_conf": 0.5182,
  "formality": "formal", "formality_conf": 0.6461, "formality_score": 72.6
}
```

`outfit_features.json`:
```json
{"hue_contrast": 21.7, "top_bottom_hue_distance": 3.3, "volume_balance": 23.8,
 "top_bottom_area_ratio": 0.238, "formality_coherence": 79.9, "num_garments": 3}
```

## Verification

- All deterministic attribute tests pass; all files compile; `fashion_segmenter` imports
  with wiring intact.
- Full CLIP path verified on a synthetic crop and on a **real photo end-to-end**
  (`image.png`, poolside pink co-ord): 3 garments in ~12 s on CPU incl. model loads.
  - Colour + silhouette were accurate (pinks correct; boxy top / A-line wide-leg pants /
    tapered sandal); the three outfit metrics matched the photo (near-monochrome →
    `top_bottom_hue_distance` 3.3; bottom-heavy → `volume_balance` 23.8).
  - **Known weak spot: `fabric`** — CLIP zero-shot mislabelled linen pants as "corduroy"
    at low confidence (0.34). Colour/silhouette/theory metrics are reliable; fabric/fit
    are the noisy signals.

## Known limitations / follow-ups

- **Fabric/fit accuracy** is the candidate for a "Tier B" upgrade: train attribute heads on
  Fashion144K. The metadata schema would not change, so nothing downstream would break.
- **Belts/accessories** are not always split out as separate items (SegFormer belt class
  didn't fire on the test photo); target-class handling could be tightened if needed.
