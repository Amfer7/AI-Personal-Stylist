# Module 4: Swap Recommendation

**Status: built, evaluated on both catalogues (Fashion144k + Pinterest), all settings calibrated, and
in the demo** (`python demo.py`, pick **Module 4** or **All**).

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

# Demo (all four modules, live trend-weight slider, "Try this swap" board)
python demo.py                 # --vram-gb 4 for a smaller GPU
# Slide figures -> demos/recommendation_tradeoff.png, demos/recommendation_demo.png
python Module4/render_recommendation.py --tradeoff --examples

# 3. Evaluation (spec §6): calibrate on the dev set, then run the report set once
python Module4/eval_recovery.py --set dev --freeze    # ~5 min; writes the calibrated settings
python Module4/eval_recovery.py --set report          # ~5 min; refuses a second run without --force (add --source pinterest for Pinterest outfits)

# 4. Tests
python -m pytest Module4                  # everything (needs the data + a built cache)
python -m pytest Module4 -m "not real"    # unit tests only; run anywhere
```

`render_recommendation.py` needs scipy. For the slide examples it picks, per verdict, the first
report-set outfit whose drawn crops are all readable (`crop_readable`: at least 4,000 opaque pixels, the
largest connected piece at least 85% of them, and that piece filling at least 35% of its bounding box),
because first-match examples turned out to be fragmentary crops.

The browser click-through of the demo (slider responsiveness, "Try this swap" redraw) has not yet been
checked by hand; the automated tests cover the headless behaviour.

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

**Protocol** (spec §6). Per catalogue, two disjoint, seeded sets of 2,000 test outfits: **dev** for all
tuning (Fashion144k only), **report** for the final numbers, run with frozen settings. In every outfit
one random piece is **corrupted**: swapped for a random garment from the same swap group that passes the
co-occurrence filter, which is exactly the space M4 searches. M4 then looks for replacements in the
test split of **both** catalogues (the **eval pool**), never the outfit itself or its reposts.

**History.** First calibrated and reported on 2026-10-09 with a Fashion144k-only pool (71.8%). When
Pinterest joined (2026-10-10) the trend model was refit and the harmony reference changed, so the
settings were recalibrated on the same dev set and both report sets run; the Fashion144k report set
was re-run with `--force` because the system changed. The 9 Oct results are kept in
`Module4/cache/eval_2026-10-09_f144k_only/`.

**Report sets** (rule `slot_percentile`, w_t = 0.2, τ = 0.392; 95% bootstrap CIs):

| Method | Fashion144k: finds the corrupted piece | Pinterest: finds the corrupted piece | Original's median rank (F / P) |
|---|---|---|---|
| **M4** | **72.2%** (70.2–74.1) | **69.3%** (67.3–71.3) | **top 4.8% / 5.4%** |
| CLIP coherence (no GNN) | 42.2% | 42.7% | top 31.5% / 31.9% |
| Random | 30.4% | 37.8% | top 50.6% / 49.9% |
| Trend only | 32.6% | 34.5% | top 45.5% / 52.2% |
| Chance (1 / pieces) | 31.4% | 36.0% | top 50% |

All baselines are below M4 at p < 1e-60 on both catalogues (exact McNemar test).

- **Pinterest is only ~3 points lower,** although the GNN's own AUC drops 0.09 on Pinterest (0.727 vs
  0.816; it was trained on 2010s Fashion144k). Hit rate compares pieces *within* one outfit, so an
  error that shifts a whole outfit's score cancels out. Fine-tuning the GNN is deferred (see
  `docs/next-steps.md`).
- **Harder when the corruption is subtle.** Fashion144k 55% / 74% / 88% (subtle / medium / obvious);
  Pinterest 53% / 72% / 82%.
- **Original recovery is the independent evidence** (what the person actually wore). The original lands
  in the top ~5% of candidates on both catalogues, far above every baseline, but is rarely the very top
  pick: the GNN rates its own top pick above the original in 96–97% of outfits.
- **Verdicts.** At τ, M4 says "swap" for 70% of corrupted outfits and for 16% (Fashion144k) / 17%
  (Pinterest) of real ones, an upper bound on false alarms. Before recalibration it was 75% / 21%.
- **Trend costs some accuracy.** From w_t = 0 to 0.2 the top suggestion's trend gain rises from ~0σ to
  1.55σ (Fashion144k) / 1.42σ (Pinterest) while keeping 94% / 93% of the harmony gain; hit rate drops
  75.7% → 72.2% / 72.9% → 69.3%. The demo slider lets users choose
  (`demos/recommendation_tradeoff.png`).
- **Selection rules barely differ:** Fashion144k 70.0 / 70.5 / 72.2%, Pinterest 68.9 / 68.8 / 69.3%
  (`max_gain` / `topk_mean_gain` / `slot_percentile`).
- **Limits.** Corruptions are random same-group garments, so these numbers measure catching clear
  intruders, not taste; mixed outfits (one catalogue's piece in the other's outfit) are suggested but
  not separately tested. The human study is the independent check.

Full tables are written to `Module4/cache/eval/report_report.md` and `pinterest_report_report.md`.

## Settings (all calibrated)

| Setting | Value | How it was chosen |
|---|---|---|
| Selection rule | `slot_percentile` | Highest dev hit rate at w_t = 0 (spec D2) |
| Trend weight | w_t = 0.2, w_h = 0.8 | Largest weight keeping ≥ 90% of the harmony-only gain on dev (D3). Gain = (1 − w)·ΔH + w·ΔT |
| Swap threshold τ | 0.392 for `slot_percentile` (0.333 before Pinterest joined; also stored for the other rules) | Best balanced accuracy, real vs corrupted dev outfits (D7) |
| Refresh tolerance ε | 0.2σ | M4's detection limit: on dev its hit rate is no better than chance for harmony drops below ~0.2σ, so a smaller loss "keeps the look" |
| Near-duplicate rule | CLIP cosine ≥ 0.99, colour-histogram L1 ≤ 0.25, size ratio ≤ 1.25 | See "Reposts and near-duplicates" below |

Recalibrated 2026-10-10 on the both-catalogue pool; rule, w_t and ε came out unchanged.

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
  `area × 98,304 / (photo width × photo height)`. Pinterest photos are resized to that frame before Module 1 (PinterestData/build_resized.py).
- **Uses the relative `file` field, never `crop_path`.** `crop_path` in the Fashion144k metadata is an
  absolute path from an older machine.
- **GPU memory is capped at 6 GB** (the GPU is shared) through `vram_gb` in the settings.
- **Catalogue crop quality.** Some pool garments carry wrong Module 1 type labels (for example a
  Pinterest crop of sneakers labelled "top", a sweatshirt labelled "jacket", denim shorts labelled
  "pants"), and some Pinterest crops contain the pin's overlaid text. These can appear in live
  suggestions. Recorded as open in [`docs/next-steps.md`](../docs/next-steps.md).
- **Two trend-weight conventions.** The demo slider mixes gain = (1 − w)·ΔH + w·ΔT, the same mix as the
  evaluation and the trade-off figure. `run_recommend.py --w-t` keeps `w_h` from the settings (0.8) and
  changes only `w_t`. At the calibrated w = 0.2 the two agree.
- **Demo model sharing.** In **All**, Module 4 loads first, and Modules 2 and 3 reuse its GNN and trend
  model, so everything reports one harmony reference (the full test set). The old 500-outfit reference
  `demos/_ref_scores.json` is no longer read.
- **Slide chart colours** (passed the dataviz palette validator on light): Fashion144k `#2a78d6`,
  Pinterest `#eb6834`, text `#0b0b0b` / `#52514e`, grid `#e6e5e0`, white background.

## Facts the design relies on

Checked against the data on 2026-10-07. If one stops being true, the design needs another look.

- Module 1 emits exactly the 16 specific types in `swap_groups.py`, and only 4 broad categories occur
  (`top`, `bottom`, `dress`, `accessory`), so every swap group stays inside one broad category.
- No outfit holds two pieces of one type (left and right shoes are merged), so a swap is always one piece
  for one piece. A type that is already worn has a co-occurrence count of 0, so it is blocked.
- Train co-occurrence: pants+leggings, pants+shorts, leggings+shorts and any two upper types all **0**;
  skirt+leggings 2,878; shorts+skirt 8. The minimum count of 20 blocks exactly the impossible pairs.
- M3 cluster ids are unique only within a broad category, so the diversity key is (broad, cluster).
- Feasibility check before the build: the dense scorer matched `model(build_graph)` to 1.5e-5 on 1,000
  random swaps. A 5-piece outfit against its whole broad-category pool (823,170 what-if outfits) scored
  in 0.42 s using 0.84 GB.

## GPU memory budget (6 GB cap)

| Item | Both catalogues (634,742 rows) |
|---|---|
| CUDA context (not counted by the allocator) | ~0.3–0.5 GB |
| Pool projection, 128 × fp32 | 0.32 GB |
| Pool CLIP embeddings, 512 × fp32 (own-item exclusion) | 1.28 GB |
| Scoring chunk (sized from the cap) | ≤ ~0.6 GB |
| **Request-time peak (allocator)** | **~2.3 GB** |

If memory gets tight, the embeddings can drop to fp16, which halves them. The cosine-threshold tests
would then need a ±1e-3 tolerance.

## Adding another catalogue

No Module 4 code changes are needed. Pinterest was added this way:

1. Build a feature store in the Fashion144k layout (570-wide rows, `index.json` with `category`,
   `broad_category` and the relative `file`), with crops under `<output_root>/<outfit_key>/`. Store areas
   in the 98,304-px frame, or every new garment looks oversized next to the others.
2. Write a split file: `{outfit_key: "train" | "val" | "test"}`.
3. Add an entry to `stores` in `m4_settings.json`.
4. Run `python Module4/build_cache.py`. The fingerprint changes, so everything is rebuilt: projection,
   trend, cross-catalogue reposts, co-occurrence and the harmony reference.
5. Recalibrate with `eval_recovery.py --set dev --freeze`.

## Next

1. **Human study** (spec §6.4): blind pairwise preferences on outfit boards (`outfit_board.py`).
2. **GNN fine-tune on both catalogues** (deferred; see `docs/next-steps.md`).
3. Re-check the near-duplicate threshold on the mixed pool (`calibrate_dup.py`, needs a visual check).
   See [`docs/next-steps.md`](../docs/next-steps.md).
