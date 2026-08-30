# Demos — completed modules (live)

One command, from the repo root:

```bash
python demo.py                    # default photo (image2.jpg)
python demo.py my_outfit.jpg      # any full-body outfit photo
python demo.py --open             # also pop the figures open
python demo.py --aggregate        # also rebuild the 91.2% testing figure
```

Produces (into this folder):

| File | Module | Shows |
|---|---|---|
| `perception_demo.png` | 1 | photo → segmented garments + attributes |
| `harmoniousness_demo.png` | 2 | photo → garments → **Harmoniousness score /100** (end-to-end) |
| `compatibility_demo.png` | 2 | **real > corrupted on 91.2%** of 500 test outfits (testing) |

Notes:
- First run builds `_ref_scores.json` (percentile calibration); later runs are instant.
- The score is a **percentile vs real outfits** (BPR is a ranking model, so raw scores
  aren't calibrated probabilities). image2.jpg scores **80/100**, dropping to 75 if a
  garment is swapped.
- Needs the trained checkpoint + feature store under `Fashion144k_v1/` (present on the
  training machine). Run once before presenting to warm up model loading.
