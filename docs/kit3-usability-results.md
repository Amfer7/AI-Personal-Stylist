# Kit 3 results: usability test of the demo (2026-11-23)

Run by Claude against `python demo.py --advanced` on the demo machine (RTX 3070 Ti 8 GB), driven in
headless Chromium. Note that a second `demo.py` instance (port 7860) was running at the same time
and sharing the GPU. Screenshots and raw run logs: `Module4/cache/usability/` (not in git).
Odd inputs came from Wikimedia Commons (group photo, mirror selfie, flat-lay, landscape) or were made
locally (product-page mock-up from an H&M image, 44 MP upscale, 110 px thumbnail, transparent PNG,
`.txt` file, text file renamed `.jpg`). The test example photo is `samples/image2.jpg`.

## Checklist

| # | Check | Result |
|---|---|---|
| 1 | Cold start, second run | **Pass.** Server up in 4 s; first run (All) 27.1 s, second 3.9 s. UI promises ~30–60 s. |
| 2 | Module 1 alone | **Pass**, notes: 1.8 s; labels correct (sunglasses, shirt, pants, shoe) but figure text is tiny; sunglasses read as "red"; shoes filed under "accessory". |
| 3 | Module 2 alone | **Pass**, notes: 1.5 s, 77/100 shown. "Drops to 70/100 if one garment is swapped" is unexplained; the chart says "percentile vs real outfits" but the text does not. |
| 4 | Module 3 alone | **Pass.** 0.8 s; per-piece scores and the least on-trend piece (shoe, 69) marked in red. |
| 5 | Module 4 alone | **Pass.** 0.7 s; verdict, 3 suggestions, board, per-piece table. ("Also worth improving" does not apply: the photo gets an "already strong" verdict.) |
| 6 | All | **Pass.** 3.5 s warm. |
| 7 | Trend slider | **Pass.** Re-ranks in < 1 s without a run. Weight 0 → 1 suggestion; 1 → trendier picks incl. pants at harmony 77 → 70; back to 0.2 restores the original three. |
| 8 | Try this swap | **Pass**, but all three options read "shoe (same type) (hm)" (bug 8). Board and text update for each. |
| 9 | Advanced options | **Fail.** Pool `eval` changes the suggestions (Pinterest items). Same type, trend floor and rule show no visible change on this photo, with no hint why. Device `cpu` after a GPU run crashes (bug 4). |
| 10 | Photo A then photo B | **Pass.** A with All, then B with Module 1 only: tabs 2–4 empty, nothing of A left. |
| 11 | Odd inputs | **5 of 10 handled well** (table below). |
| 12 | First-time understanding | **Not tested:** needs a person who has not seen the demo. Heuristic notes under bugs 12–13. |

| Input | What happened | OK? |
|---|---|---|
| Group photo (≈20 people) | 2 garments, harmony 90, a verdict; no "several people" warning | no |
| Mirror selfie | 3 garments (t-shirt, "pants" = boxer shorts, sandals on the floor as shoe); sensible run | yes |
| Product-page screenshot | 3 garments (t-shirt, skirt, dress from one product), harmony 100, a verdict | no |
| Flat-lay | 5 garments incl. "hat" (a notebook), harmony 100, a verdict; no "no person" note | no |
| Landscape, no person | 3 "garments" cut from snow and trees, "Already strong", dress swap suggested | no |
| Pet photo | "No person found, only an animal" | yes |
| 14 MB / 44 MP photo | CUDA out of memory; the raw PyTorch message in the status line. 12 MP: 8.2 s, 24 MP: 13.2 s (both fine but slow) | no |
| 110 × 195 px photo | Runs; harmony 38 vs 77 for the same outfit at full size, sunglasses and shoes lost; no warning | yes (degraded) |
| PNG with transparency | Same result as the JPEG (77) | yes |
| Non-image file | `.txt`: rejected on upload, "Invalid file type only image/* allowed" | yes |
| | Text file renamed `.jpg`: broken preview, Run gives a bare red "Error", empty status and log | no |

## Bug log

| # | Step | What I did | Expected | What happened | Severity |
|---|---|---|---|---|---|
| 1 | Odd inputs | Landscape, flat-lay, product page | "No person found" (as for pets) | Garments made up from snow/trees, a notebook as a hat; harmony up to 100 and a verdict. The person check only stops animals. | major |
| 2 | Repeat runs | Same photo, Module 2, six runs | Same score | 77 or 79 (swap drop 70 or 72). Tracks Module 1 time: ~1.4 s runs give 77, ~0.6 s runs give 79. First run after start showed 79 → 85 on the board, later runs 77 → 83. | major |
| 3 | Odd inputs | 14 MB, 44 MP photo | Downscaled and analysed | CUDA out of memory (3 GiB for SegFormer's upsampled logits); raw PyTorch text in the status. 12–24 MP pass but take 8–13 s. Downscaling on upload would fix both. | major |
| 4 | Advanced options | Device `cpu` after a GPU run | Ignored ("applies on the first run") | "Expected all tensors to be on the same device… cuda:0 … cpu" error | major |
| 5 | Odd inputs | Text file renamed `.jpg` | "Could not read this image" | Broken preview; Run gives a bare "Error" badge, empty status and log. iPhone HEIC photos may fail the same way (not tested; check in Kit 2). | minor |
| 6 | Odd inputs | Group photo | Warning or "one person per photo" | Silently analyses 2 garments | minor |
| 7 | Odd inputs | 110 px photo | Low-resolution warning | Runs; harmony 38 vs 77 at full size | minor |
| 8 | Try this swap | Open the dropdown | Distinguishable options | "1./2./3. shoe (same type) (hm)": identical labels | minor |
| 9 | Advanced options | Same type, trend floor, rule | A visible change or an explanation | No visible change on the example photo | minor |
| 10 | Module 4 | Read the per-piece table | Plain words | "selection score", "best ΔH (σ)", "best ΔT (σ)", "clears" | minor |
| 11 | Modules 1–3 | Read the figures at 1400 px | Readable labels | Attribute and axis text ~7 px; unreadable on a projector | cosmetic |
| 12 | Module 4 | Read suggestion captions | Full caption | Cut off: "shoe (same type) · harmony 7…" | cosmetic |
| 13 | Module 1 | Example photo | Correct attributes | Dark-brown sunglasses "red"; shoes grouped as "accessory" | cosmetic |
| 14 | Landing | Examples row | Clear thumbnail | Dog example letterboxed into a thin strip | cosmetic |

**Totals:** 14 issues: 0 blocker, 4 major, 6 minor, 4 cosmetic.

**Heuristic notes for check 12** (not a substitute for a first-time user): what to do first is clear
(headline, upload box, Examples). The word "refresh" appears only in the Log tab; the UI says
"Already strong: optional trendier options that keep the look". Harmony 77/100 reads as a grade, but
it is a percentile against real outfits, which only the chart axis says.
