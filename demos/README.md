# Demos — completed modules (live)

A small **Gradio GUI**, from the repo root:

```bash
pip install gradio                # one-time
python demo.py                    # opens the GUI in your browser
```

In the browser: **upload or paste** an outfit photo, choose which module to run
(**Module 1**, **Module 2**, or **Both**), and click **Run**. The figures render inline
and are also written into this folder. Module 2 needs Module 1's segmentation, so
picking Module 2 runs Module 1 silently first.

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
