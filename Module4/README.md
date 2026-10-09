# Module 4: Swap Recommendation

**Status: core built (Fashion144k); evaluation next.** Pinterest joins the catalogue after Phase 0.

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
#    checkpoint, trend model, stores or t_dup; Module 4 refuses to load a stale cache.
python Module4/build_cache.py

# 2. Recommend for a photo (runs Module 1 first) or for an existing Module 1 outfit folder
python Module4/run_recommend.py --photo samples/image2.jpg
python Module4/run_recommend.py --outfit_dir <M1 output folder> --pool eval --json rec.json

# Options: --pool demo|eval   --rule max_gain|topk_mean_gain|slot_percentile   --w-t 0.3
#          --same-type (strict like-for-like)   --no-trend-drop (trend floor)   --device cpu

# 3. Tests
python -m pytest Module4                  # everything (needs the data + a built cache)
python -m pytest Module4 -m "not real"    # unit tests only; run anywhere
```

The cache goes to `Module4/cache/` (gitignored): `pool/` (projection, metadata, co-occurrence,
harmony reference, fingerprint), `dup_calibration/` (contact sheets) and `runs/` (Module 1 output for
photos).

## How it works

1. **Cache build** (`build_cache.py`, once). For each of the 480,079 garments in the feature store:
   the first layer of the GNN (the **input projection**, 555 → 128 numbers) is computed ahead of time,
   along with the M3 trend score and style cluster, and a near-duplicate representative (**dedupe**).
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
| `candidate_pool.py` | Builds and loads the candidate pool: projection, trend, dedupe, crop paths |
| `calibrate_dup.py` | Contact sheets for choosing the near-duplicate threshold `t_dup` by eye |
| `harmony_reference.py` | Harmony percentiles and the score spreads (σ) used to standardise gains |
| `outfit_state.py` | An outfit as a list of pieces; loads Module 1 output (with area normalisation) |
| `swap_scorer.py` | Outfit score + every swap's score for one piece |
| `selection.py` | Gains, the three selection rules, style-diverse top 3 |
| `explain.py` | The descriptive "what changed" line |
| `recommendation.py` | The `Recommendation` result: JSON in/out and a text summary |
| `recommender.py` | Ties it together: `score()`, `decide()`, `recommend()` |
| `build_cache.py`, `run_recommend.py` | Command-line entry points |
| `conftest.py`, `test_*.py` | Tests: 60 in total, one file per unit. Tests that need the real data are marked `real` |

## Measured performance

`samples/image2.jpg` (4 pieces) against the **full demo pool** on the RTX 3070 Ti:
**0.16 s per recommendation once warm, peak GPU memory 1.92 GiB.** The target was ≤ 5 s and ≤ 6 GB,
so the spec's fallback (a cheap pre-ranker to shrink the candidate list) wasn't needed. Every
candidate is scored; nothing is sampled.

## Settings: calibrated and provisional

- **Near-duplicate threshold `t_dup` = 0.99 (calibrated).** `calibrate_dup.py` draws pairs of
  garments at CLIP cosine similarity 0.90 / 0.93 / 0.95 / 0.97 / 0.99 for shoes, bags, pants,
  jackets and dresses. The rule is to pick the lowest band where the pairs look like the same item,
  and 0.99 was chosen, meaning pairs at 0.97 and below still looked like different items. This
  matches the spec's warning that distinct garments of one type often score 0.85–0.95. With 0.99,
  the demo pool keeps 472,220 of 480,079 garments (7,859 near-duplicates merged).
- **Still provisional:** the trend weight `w_t` (0, so harmony only), the selection rule
  (`topk_mean_gain`, k = 5), the verdict threshold `tau` and the harmony noise band `eps`. These are
  placeholders until **evaluation** (the next build step) calibrates them on held-out outfits. Every
  recommendation lists them under `provisional` so the output never looks more final than it is.

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

1. **Evaluation:** corruption recovery, original recovery, baselines, and calibration of `w_t`, the
   rule, `tau` and `eps` (spec §6).
2. **Phase 0** code for Pinterest, then the demo integration (D10). See [`docs/next-steps.md`](../docs/next-steps.md).
