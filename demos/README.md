# Demos

A **Gradio** web app at the repo root runs every finished module on one photo:

```bash
pip install -r Module1/requirements.txt   # includes gradio
python demo.py                            # opens in your browser
```

Upload or paste an outfit photo (or click an example from `samples/`), choose **Module 1**–**4** or
**All**, and click **Run**. Results open in one tab per module; the line above the tabs says when the
run is done or why it stopped. Models load once, so the first run is slow (~30–60 s) and later runs are fast.

```bash
python demo.py --advanced                         # also show the device and Module 4 options
python demo.py --share --auth team:password       # public link for remote viewers (password required)
```

The app writes these figures into this folder. **They're generated locally and never committed**
(the repo ignores all images):

| File | Module | Shows |
|---|---|---|
| `perception_demo.png` | 1 | photo → segmented garments + attributes |
| `harmoniousness_demo.png` | 2 | photo → harmony score (percentile vs real outfits) |
| `compatibility_demo.png` | 2 | real outfits beat corrupted ones on 91.2% of 500 test outfits (from `Module2/demo_score.py`) |
| `trend_demo.png` | 3 | per-garment trend scores |

Notes:
- `_ref_scores.json` caches the reference distribution behind the harmony percentile. The first run
  builds it; later runs load it instantly.
- Needs the trained checkpoint, feature store and trend model locally (`Fashion144k_v1/`,
  `Module3/per_category_trend_model.pkl`). They're not in git because of their size.
- Sample photos live in `samples/` (gitignored). Put your own test photos there.
