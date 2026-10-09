# Module 4: Swap Recommendation

**Status: built and evaluated on Fashion144k; all settings calibrated.** Pinterest joins the catalogue
after Phase 0; the demo integration is next.

**In one line:** given an outfit photo, says **which one piece to change and what to change it to**,
using Module 2 (harmony) and Module 3 (trend) to judge every possible replacement. If the outfit is
already strong, it says so.

```
your outfit ──► for each piece: try every allowed replacement ──► score each "what-if" outfit
                                                                   (M2 harmony + M3 trend)
            ──► pick the piece whose replacements help most ──► top 3 suggestions, or "keep it"
```

- Plain-language design (all 11 decisions explained): [`docs/module4-design.md`](../docs/module4-design.md)
- Detailed engineering spec (alternatives, evidence, file references, tests): [`docs/module4-design-detailed.md`](../docs/module4-design-detailed.md)

## How to run

Run everything from the repo root. It needs the Fashion144k data, the M2 checkpoint and the M3 trend
model locally (see the root README, **Data**). Every path and setting lives in
[`m4_settings.json`](m4_settings.json), all repo-relative.

```bash
# 1. Build the cache once (~20 s on the RTX 3070 Ti). Re-run with --force after changing the
#    checkpoint, trend model, stores or duplicate rule; Module 4 refuses to load a stale cache.
python Module4/build_cache.py

# 2. Recommend for a photo (runs Module 1 first) or for an existing Module 1 outfit folder
python Module4/run_recommend.py --photo samples/image2.jpg
python Module4/run_recommend.py --outfit_dir <M1 output folder> --pool eval --json rec.json

# Options: --pool demo|eval   --rule max_gain|topk_mean_gain|slot_percentile   --w-t 0.3
#          --same-type (strict like-for-like)   --no-trend-drop (trend floor)   --device cpu

# 3. Evaluation (spec §6): calibrate on the dev set, then run the report set once
python Module4/eval_recovery.py --set dev --freeze    # ~5 min; writes the calibrated settings
python Module4/eval_recovery.py --set report          # ~5 min; refuses a second run without --force

# 4. Tests
python -m pytest Module4                  # everything (needs the data + a built cache)
python -m pytest Module4 -m "not real"    # unit tests only; run anywhere
```

The cache goes to `Module4/cache/` (gitignored): `pool/` (projection, metadata, co-occurrence,
harmony reference, fingerprint), `dup_calibration/` (contact sheets), `runs/` (Module 1 output for
photos) and `eval/` (per-outfit records, summaries and Markdown reports).

## How it works

1. **Cache build** (`build_cache.py`, once). For each of the 480,079 garments in the feature store:
   the first layer of the GNN (the **input projection**, 555 → 128 numbers) is computed ahead of time,
   along with the M3 trend score and style cluster. **Reposts** (the same photo posted more than once)
   are found and merged, so each garment is offered once.
   The build also counts which garment types appear together in training outfits (the
   **co-occurrence matrix**) and scores every real test outfit to get a **harmony reference**.
2. **Scoring** (`Recommender.score()`). For each piece of your outfit, every allowed replacement is
   swapped in and the whole what-if outfit is re-scored. This uses a **dense, batched copy of the M2
   GNN** (`dense_gnn.py`) that handles hundreds of thousands of what-if outfits per piece in one
   GPU pass. Tests check that it matches M2's own `model(build_graph(...))`.
3. **Deciding** (`Recommender.decide()`). The stored gains are turned into a verdict (`swap`,
   `refresh` or `keep`) and three style-diverse suggestions. This step makes **no GNN calls**, so
   changing the trend weight or the selection rule is free.

## Files

| File | What it does |
|---|---|
| `m4_settings.json` / `m4_settings.py` | Every path and setting; `status` marks which ones are still provisional |
| `m2_bridge.py` | The single doorway into Module 2 and Module 3 code (never copied, never modified) |
| `swap_groups.py` | The 16 garment types, the swap groups (D4) and the co-occurrence filter |
| `dense_gnn.py` | Batched GNN scorer with GPU-memory-aware chunking (halves the chunk on out-of-memory) |
| `candidate_pool.py` | Builds and loads the candidate pool: projection, trend, repost groups, the near-duplicate rule, crop paths |
| `calibrate_dup.py` | Contact sheets of garment pairs that pass the near-duplicate rule, by CLIP cosine band |
| `harmony_reference.py` | Harmony percentiles and the score spreads (σ) used to standardise gains |
| `outfit_state.py` | An outfit as a list of pieces; loads Module 1 output (with area normalisation) |
| `swap_scorer.py` | Outfit score + every swap's score for one piece |
| `selection.py` | Gains, the three selection rules, style-diverse top 3 |
| `explain.py` | The descriptive "what changed" line |
| `recommendation.py` | The `Recommendation` result: JSON in/out and a text summary |
| `recommender.py` | Ties it together: `score()`, `decide()`, `recommend()` |
| `build_cache.py`, `run_recommend.py` | Command-line entry points |
| `eval_recovery.py` | The evaluation: outfit sets, corruption, baselines, calibration, reports |
| `eval_metrics.py` | Pure helpers for it: per-outfit summaries, threshold, Pareto weight, bootstrap CIs, McNemar |
| `conftest.py`, `test_*.py` | Tests: 73 in total, one file per unit. Tests that need the real data are marked `real` |

## Measured performance

`samples/image2.jpg` (4 pieces) against the **full demo pool** on the RTX 3070 Ti:
**0.16 s per recommendation once warm, peak GPU memory 1.92 GiB.** The target was ≤ 5 s and ≤ 6 GB,
so the spec's fallback (a cheap pre-ranker to shrink the candidate list) wasn't needed. Every
candidate is scored; nothing is sampled.

## Evaluation results

**Protocol** (spec §6). Two disjoint, seeded sets of 2,000 Fashion144k test outfits: **dev** for all
tuning, **report** for the final numbers, run once with frozen settings. In every outfit one random piece
is **corrupted**: swapped for a random garment from the same swap group that passes the co-occurrence
filter, which is exactly the space M4 searches. M4 then looks for replacements in the test split only
(the **eval pool**), never the outfit itself or its reposts.

**Report set** (rule `slot_percentile`, w_t = 0.2, τ = 0.333; 95% bootstrap CIs):

| Method | Finds the corrupted piece | Original garment's median rank | Original in top 10 |
|---|---|---|---|
| **M4** | **71.8%** (69.8–73.7) | **top 4.8%** | 3.2% |
| CLIP coherence (no GNN) | 38.6% | top 34.5% | 0.6% |
| Random | 30.4% | top 50.6% | 0.0% |
| Trend only | 32.4% | top 48.3% | 0.0% |
| Chance (1 / pieces) | 31.4% | top 50% | |

All baselines are below M4 at p < 1e-99 (exact McNemar test).

- **Harder when the corruption is subtle.** M4 finds 52% of subtle corruptions (mean harmony drop
  0.21σ), 78% of medium ones (1.22σ) and 86% of obvious ones (2.40σ).
- **Original recovery is the independent evidence** (what the person actually wore), and it's mixed.
  The original lands in the top 5% of 10,000–38,000 candidates, far above every baseline. It's rarely
  the very top pick, though: the GNN rates its own top pick above the original in 95% of outfits. So
  M4 agrees with human choices broadly, not exactly.
- **Verdicts.** At τ, M4 says "swap" for 75% of corrupted outfits and for 21% of real ones. That 21% is an
  upper bound on false alarms, since some real outfits can be improved.
- **Trend costs some accuracy.** Going from w_t = 0 to 0.2 makes the top suggestion much trendier (mean
  ΔT 0.04σ → 1.51σ) while keeping 94% of the harmony gain, but detection drops from 75.4% to 71.8%.
  The demo slider lets users choose.
- **Selection rules barely differ:** `max_gain` 71.5%, `topk_mean_gain` 72.1%, `slot_percentile` 71.8%.
  `slot_percentile` won on dev (73.1% at w_t = 0). Its edge over `topk_mean_gain` isn't significant
  (p = 0.37); its edge over `max_gain` is (p = 0.015).

Full tables (per swap group, per harmony drop, the whole weight sweep) are written to
`Module4/cache/eval/report_report.md` by the evaluation run.

## Settings (all calibrated)

| Setting | Value | How it was chosen |
|---|---|---|
| Selection rule | `slot_percentile` | Highest dev hit rate at w_t = 0 (spec D2) |
| Trend weight | w_t = 0.2, w_h = 0.8 | Largest weight keeping ≥ 90% of the harmony-only gain on dev (D3). Gain = (1 − w)·ΔH + w·ΔT |
| Swap threshold τ | 0.333 for `slot_percentile` (also stored for the other rules) | Best balanced accuracy, real vs corrupted dev outfits (D7) |
| Refresh tolerance ε | 0.2σ | M4's detection limit: on dev its hit rate is no better than chance for harmony drops below ~0.2σ, so a smaller loss "keeps the look" |
| Near-duplicate rule | CLIP cosine ≥ 0.99, colour-histogram L1 ≤ 0.25, size ratio ≤ 1.25 | See "Reposts and near-duplicates" below |

## Reposts and near-duplicates

The first version merged two garments whenever their CLIP embeddings had cosine ≥ 0.99, joining
chains of such pairs into groups. Evaluation exposed the flaw. On tiny, mostly empty crops (for example
a left and a right shoe far apart) CLIP's embedding collapses, and about 7.7 million pairs of
*different* garments score ≥ 0.99. One "duplicate" group held 4,651 distinct garments. Requiring
matching colours and sizes as well wasn't enough on its own: tiny black or white crops still matched.

The fix works on **whole outfits**. Two outfits are **reposts** when they have the same number of
garments (at least 2) and every garment matches under the near-duplicate rule above. Groups don't chain:
each outfit must match its group's first outfit directly. On Fashion144k this finds 72 repost pairs.
Spot checks of the original photos showed the same photo posted under different IDs every time. The
demo pool now merges 238 garments (it used to merge 7,859). The same per-garment rule still keeps
your own item out of your suggestions (D8b).

## Things worth knowing

- **"What changed" is descriptive, not causal.** The line under each suggestion (for example "formality
  now closer to your pants") describes how the new piece differs. The GNN is non-linear, so this line
  doesn't claim to explain why it scored higher.
- **Area normalisation.** Module 1 reports garment size in raw pixels, and the GNN compares the sizes of
  pieces (`volume_contrast`). Catalogue garments come from 256×384 photos, but a user's photo can be any
  size. A phone photo would make every catalogue piece look tiny next to the user's own. So when a user
  outfit is loaded, each piece's area is rescaled to the same 98,304-pixel frame:
  `area × 98,304 / (photo width × photo height)`. Phase 0 must store Pinterest areas in that frame too.
- **Uses the relative `file` field, never `crop_path`.** `crop_path` in the Fashion144k metadata is an
  absolute path from an older machine.
- **GPU memory is capped at 6 GB** (the GPU is shared) through `vram_gb` in the settings.

## Next

1. **Phase 0** code for Pinterest (debugged on the 224×224 copies), then the **demo integration** (D10).
2. When full-resolution Pinterest arrives: rebuild, re-run the evaluation for both sources, human study.
   See [`docs/next-steps.md`](../docs/next-steps.md).
