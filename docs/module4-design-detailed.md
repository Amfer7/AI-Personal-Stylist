# Module 4: Swap Recommendation Engine, detailed design

_Plain-language version: [`module4-design.md`](module4-design.md)._

_Date: 2026-10-05 · Status: **decided, not yet built** · **all decisions D1–D11 made 2026-10-07**_

## 1. Goal

Given an outfit photo, answer: **"Which one piece should I swap, and what for?"**
Output: the piece to swap, the top-3 replacements (with crop images), and for
each one how much it changes **harmony** (Module 2 GNN) and **trendiness**
(Module 3), plus a combined score.

Success criteria:
- On real outfits where one garment was deliberately swapped for a mismatched
  one, M4 identifies the corrupted piece much more often than chance **and** than the
  random / trend-only / CLIP-coherence baselines (§6, report set, 95% CIs).
- M4 ranks the **original** (actually worn) garment near the top of its candidates (original recovery).
- Human raters prefer M4's fix over the corrupted outfit and over a random-swap fix (§6.4).
- Runs inside the root `demo.py` (Gradio) in **≤ 5 s per photo once warm** on the RTX 3070 Ti, peak VRAM ≤ 6 GB.

Out of scope (YAGNI): multi-piece swaps (**deferred**, but the API is designed for them: `recommend()`
takes an outfit state and returns ranked *edits*, and calibration is keyed by step number, so a
greedy sequence (tier 1) or beam search (tier 2) can wrap it without a rewrite; structural
add/remove/dress↔top+bottom edits (tier 3) are a separate project), user preference/personalisation, size/price/
availability, generating new garments. Retraining the GNN is out of scope **unless the
Pinterest AUC gate fails** (§2.1, Phase 0).

### Facts that constrain the design (verified in the repo)

1. **No trained per-item harmony score exists.** The GNN's `per_item` head is not
   in the training loss ([`Module2/README.md`](../Module2/README.md), "Engineering history"), so it can't be read off as a
   "harmony weak link". Per-piece harmony has to be measured by *changing* a piece
   and re-scoring the whole outfit.
2. **The GNN needs more than a CLIP vector per garment:** 512 CLIP + 32 colour
   histogram + attribute features (silhouette, pattern, fabric, fit, colour,
   formality), plus pairwise edge features (hue separation, formality difference,
   volume contrast).
3. **Fashion144k feature store** (`Fashion144k_v1/feature_store/`) holds **480,079
   garments** with all of the above, plus crop PNGs under `Fashion144k_v1/outputs/`.
   The GNN can score any of them directly.
4. **Pinterest garments** (143,612 from 54,876 photos) have only embeddings + a coarse label: lean mode
   saved no crops or attributes. The GNN can't score them without re-segmenting the photos.
   The existing `--mode full` of `Module3/pinterest_garment_segmentation.py` does **not** fix this:
   it adds fine category + confidence but still discards the crop, colour histogram and attributes.
4a. **Pinterest images on disk are degraded** (verified 2026-10-07): 55,795 JPEGs in
   `PinterestData/pinterest_dataset/images/`, all **224×224**, produced by a plain
   `img.resize((224, 224))` (`pinterest_large_scale.py:493`). The resize does not preserve aspect ratio, so portrait pins are
   squashed vertically. Fashion144k photos are 256×384 at their native aspect. The full-resolution raw
   scrape is not in this repo (README_DATASET.md places it on the original scraping machine).
4b. **Feature store breakdown** (verified 2026-10-07, `index.json` × `split.mat`): 144,169 outfits →
   train 86,501 / val 14,416 / test 43,250 (2 unsplit). Garments: train 288,133 / val 47,769 /
   test 144,170. **16** specific types (test-split count): shoe 28,329 · pants 19,169 · bag 17,130 ·
   jacket 15,025 · skirt 11,701 · dress 11,246 · shirt_blouse 8,436 · sweater 8,015 · leggings 7,268 ·
   hat 6,643 · scarf 3,163 · top 3,131 · t_shirt 2,726 · belt 1,405 · sunglasses 548 · shorts 235.
   `index.json` stores `category` per row directly; no need to parse filenames.
5. `broad_category = accessory` mixes shoe / bag / hat / scarf / belt / sunglasses, so swapping within a
   broad category allows nonsense (shoe → scarf). M2's node `category` *is* the broad category
   (`02_outfit_dataset.py:201`). Upper-body types (jacket, t_shirt, …) share one segmentation slot (D4).
6. Module 3's trend score is **descriptive, not predictive** (temporal backtest,
   `Module3/README.md`). Module 2's ranking AUC is ~0.80.

## 2. Decisions

| # | Decision | Recommended | Alternatives | Why |
|---|---|---|---|---|
| D1 | **Candidate pool** | **Unified pool: Fashion144k store + Pinterest re-processed into the same store format**, with a `source` column. **Eval pool = test split of both sources; demo pool = everything.** GNN retrained only if the Pinterest AUC gate fails. Details in §2.1 | Fashion144k only (stale 2010s catalogue); a third dataset such as Polyvore (product shots, different from the on-body crops M1/M2 were built on); retraining the GNN unconditionally (throws away the validated AUC without evidence it's needed) | Pinterest is the only *current* source and the one M3 measures trend on. Same M1 code path, so one schema. Split-by-use keeps eval honest (no train garments) while the demo gets maximum variety |
| D2 | **How to pick the piece to swap** | **Swap-based, with a pluggable selection rule.** Every piece's shortlist is scored as what-if outfits, and each piece is summarised by a rule: `max_gain`, `topk_mean_gain` (k=5, **default**) or `slot_percentile` (share of candidates whose combined score beats the current piece). The piece with the highest summary is swapped. §6 measures all three rules' hit rate and **the winner ships as the default** | Leave-one-out harmony: removing a piece changes outfit size, which confounds the GNN, and "remove your shoes" isn't actionable. Trend-only: ignores the validated signal. Plain `max_gain` as the only rule: winner's curse, since garment types with volatile GNN scores (dress, jacket) win by luck | Same-size what-if outfits keep comparisons fair, and "which piece" and "what replacement" come from one calculation. `topk_mean` resists one-outlier wins while staying in score points (needed by D7). Choosing the rule empirically turns a judgement call into a reportable ablation. Suggestions are the chosen piece's style-diverse top 3 (D8c) |
| D3 | **Combined score** | **Rank on standardised gains; display percentiles.** `gain = w_H·ΔH + w_T·ΔT`, where ΔH = Δ raw GNN logit ÷ std of logits over real test outfits, and ΔT = trend(candidate) − trend(current piece) ÷ std of garment trend scores. **w_T is chosen by a Pareto sweep** (§6): the largest w_T that keeps ≥ 90% of the harmony-only gain. Configurable flag/slider. Harmony reference rebuilt on **all** real test outfits (Fashion144k + Pinterest after Phase 0). No confident trend match → ΔT = 0 | Fixed `0.7·H% + 0.3·T` on 0–100 scales (the original draft); 0.5/0.5; harmony-only; trend-only | Verified 2026-10-07: H% is a percentile vs ~500 outfits (`demo_score.py:129`), so it **ceilings at 100** (strong outfits' swaps tie) and moves in 0.2-pt steps. Outfit T is a **mean over pieces** (`trend_model.py:579`), so one swap's trend effect shrinks as 1/n (the same swap counts 2.5× less in a 5-piece outfit than in a 2-piece one), making "70/30" behave like ~95/5. Standardised units make the weights mean what they say. Trend has no ground truth to tune against, so the trade-off curve is reported instead of a hand-picked weight |
| D4 | **Swap groups** (what may replace what) | **Slot-based groups + co-occurrence filter.** Groups: **upper** = {t_shirt, sweater, shirt_blouse, jacket, top}; **lower** = {pants, leggings, shorts, skirt}; **dress** = {dress}; each accessory alone (shoe, bag, hat, belt, scarf, sunglasses). **Filter:** reject a candidate whose type co-occurs with any remaining piece's type in < **20** train outfits (Fashion144k train + Pinterest train after Phase 0; cached). Output names type changes ("jacket → shirt_blouse"). **`--same-type`** flag = strict like-for-like. Type read from `index.json` `category` | Outerwear vs inner-top split (original draft); broad category only (allows shoe→sunglasses); exact type only (blocks pants→skirt; relies on noisy CLIP subtype) | Verified 2026-10-07: M1 segments **one garment per SegFormer region** (`fashion_segmenter.py:46`), so outfits never hold two upper garments (co-occurrence lift **0.00** between all upper types) and layering doesn't exist. The outer/inner split guarded nothing. Skirt + leggings is common (lift 0.75, tights) but pants + leggings never occurs (0.00), so free lower swaps could create impossible outfits. The data-learned filter blocks those without hand rules. Groups block *structural* nonsense; style is the GNN's call |
| D5 | **Candidate shortlist per piece** | **Exhaustive: score every candidate** in the piece's swap group that passes the D4 filter (no sampling, deterministic). Enabled by (1) a **vectorised swap builder** (fixed nodes once; all candidates' 5-D edge features in one tensor op from `node_scalars.npy`), the primary path, with a parity test vs `build_graph`; (2) the GNN input projection (570→128) **precomputed for the whole pool** (~620k × 128, ~300 MB), with a parity test; (3) chunked batches under a **6 GB VRAM cap** (GPU is shared). **Trend floor off by default**; `--no-trend-drop` restores it. **Performance gate:** `samples/image2.jpg`, full demo pool, **≤ 5 s warm** on the RTX 3070 Ti. Fallback only if it fails: a cheap pre-ranker narrows each group before the GNN | Seeded random sample K = 512 (original draft: ~2% coverage of e.g. 94k shoes, so the true best is usually missed and output depends on the seed); top-K by trend (biased, low variety); hard trend floor (blocks the best fix for a trendy but clashing piece, skews D2's `slot_percentile`, and D3/D7 already price trend) | Verified 2026-10-07: the GNN is small (2 edge-aware layers, hidden 128, `03_train_gnn.py:85`), edge features are 5 closed-form scalars (`02_outfit_dataset.py:208`), and the per-node input projection doesn't depend on the graph. The cost was the Python `build_graph` loop, not the GNN. Group sizes vary ~50× (shoes vs sunglasses), which D2's ablation and D7's calibration absorb |
| D6 | **Output shape** | **One layout per D7 verdict.** `swap`: chosen piece highlighted → top 3 replacements, plus **"also worth improving"** (other pieces that also clear τ, each with its single best option). `refresh`: "already strong", up to 3 trendier options with ΔH ≥ −ε. `keep`: "already strong, no swap". **Suggestion card:** crop image · type change if any ("jacket → shirt_blouse") · harmony % before→after · piece trend before→after · source (Fashion144k / Pinterest) · a **"what changed" line** from the attribute and edge-feature deltas (formality gap, hue relation, volume), labelled descriptive, not causal. **Per-piece panel** on every verdict: selection score, best ΔH, best ΔT, ✓ if ≥ τ. `Recommendation` serialises to JSON, consumed by demo, figure and eval | Top-1 only; top-3 for every piece (clutter); a single layout regardless of verdict | Exhaustive scoring (D5) makes full per-piece information free. Verdict-specific layouts keep the message honest. Type changes (D4) and source (D1) must be visible. The GNN is non-linear in its features, so true attribution is out of scope; the descriptive line gives users a reason without overclaiming (caveat in README) |
| D7 | **When the outfit is already good** | **Data-calibrated threshold + three verdicts.** Threshold τ on the chosen piece's selection score = the value that best separates real test outfits (should be "no swap") from corrupted ones (should be "swap"), by balanced accuracy, through the same best-of-K search. Verdicts: **`swap`** (score ≥ τ: piece + top 3) · **`refresh`** (below τ, but a candidate has ΔT > 0 with ΔH ≥ −ε: "already strong, optional trendier option") · **`keep`** (below τ, nothing qualifies: "already strong, no swap"). ε also calibrated on real outfits (the ΔH noise band) | Fixed "< 2 points" (arbitrary, and in units D3 no longer uses); always recommend something | Searching every candidate per piece *always* finds some positive gain by luck, so the threshold must be calibrated against that same search. Calibration accounts for it automatically and gives reportable false-alarm and detection rates. `refresh` gives an already-good outfit useful advice (update, not fix). D3's raw logits remove the 100-ceiling, so "already strong" is measured |
| D8 | **Near-duplicates & variety** | **Three layers.** (a) **Pool-level dedupe at build time:** group near-identical garments (CLIP cosine ≥ t_dup, chunked GPU matmul over the whole pool) and keep one representative (largest crop). (b) **Exclude the user's own item:** drop candidates ≥ t_dup to the current piece. (c) **Style-diverse top 3:** #1 = best overall; #2/#3 = best from *different* M3 style clusters that still clear the verdict bar (≥ τ for `swap`; ΔH ≥ −ε for `refresh`); fill by gain if too few qualify. Unmatched-cluster garments form one bucket, separated by t_dup. **t_dup calibrated visually:** contact sheet of pairs at cosine bands 0.90/0.93/0.95/0.97/0.99; a reviewer marks where pairs become "the same item" | Fixed cosine > 0.95 among listed suggestions only (original draft); MMR with a hand-tuned λ | Same-type CLIP embeddings are concentrated (distinct items often 0.85–0.95), so a guessed threshold either drops real options or keeps reposts. Dedupe alone doesn't create variety (black boot / ankle boot / bootie). Reposts and repins inflate the winner's curse without adding information. Suggesting the user's own item is the most visible failure. M3 clusters give diversity from existing structure without trading away quality |
| D9 | **Evaluation** | Full protocol in §6: **group-consistent corruption** (same D4 group + filter) with severity strata; **dev/report split** (2,000 + 2,000 per source; all tuning on dev, report run once); metrics incl. **original recovery** (rank of the garment actually worn); baselines **random / trend-only / CLIP-coherence**; **human blind A/B study** (~100 outfits, 8–10 raters, ≥ 3 per pair); bootstrap CIs + paired tests. VLM judge **under consideration** | M2's `make_negative_sample` as-is (original draft); qualitative examples only; GNN-judged ΔH only | Verified: M2's corruption swaps within the *broad* category (shoe → scarf), which is too easy and often impossible to undo under D4. With exhaustive search, GNN-judged ΔH is near-circular, so original recovery (human ground truth) and the human study provide independent evidence. Many tuned knobs require a held-out report set |
| D10 | **Demo** | **Module 4 (recommend)** added to the existing Gradio `demo.py` ("All" runs M1–M4), reusing the already-loaded GNN + trend model and loading the pool cache once. **Interactive panel:** verdict headline; suggestion-card gallery (D6); "also worth improving"; per-piece table; **trend-weight slider that re-ranks from cached ΔH/ΔT without re-running the GNN**; advanced options (pool demo/eval, same-type, no-trend-drop, selection rule). **"Try this swap"**: outfit board (remaining crops + new piece, collage code shared with the human study) with its harmony/trend. **One harmony reference everywhere:** M2's panel switches to the full-test-set reference (D3) so M2 and M4 agree. M3 panel text no longer calls the trend weak link "the swap target for Module 4". Slides: `demos/recommendation_demo.png` (one real example per verdict, from the report set) + `demos/recommendation_tradeoff.png`. **Hardware:** main machine RTX 3070 Ti (gate ≤ 5 s); a laptop RTX 3060 is also supported (timing reported, not gated). VRAM cap is a flag (`--vram-gb`, default 6) | Separate Gradio app; static figure only | Verified 2026-10-07: `demo.py` is a Gradio app with load-once singletons and an M1/M2/M3/All radio, so M4 slots in. Its M2 panel uses the 500-outfit `_ref_scores.json`, which would disagree with M4's numbers. Caching the per-candidate deltas makes the D3 trade-off explorable live at no GPU cost |
| D11 | **M3 trend formula** | **Settle in Phase 0** (step 0.6), after M3 is refit on the full-res Pinterest garments and **before** any M4 calibration. Backtest target = **forward share** (the cluster's share of activity 1/3/6 months later), with **rolling monthly cutoffs Jan–May 2026**. Compare growth weight ∈ {0.6 (current), 0.3, 0 = recency-only}, with growth-only as a reference. **Rule:** best pooled forward-share ρ; statistical tie → the simpler one (recency-only). UI label stays descriptive ("Trend (current popularity)") | Leave 0.6/0.4 and revisit after M4 (original draft); switch to recency-only without testing | Verified (`Module3/README.md:94`): growth has no forward power at 3 mo (ρ −0.10) and is *negatively* predictive at 1/6 mo (−0.45/−0.35, mean reversion). M4 *acts* on the score, so 60% growth weight steers users toward styles about to decline. "Change later" isn't free: D3's sweep, D7's τ/ε and D8's clusters are all calibrated on trend scores. M3 must be refit anyway, since Phase 0 changes every Pinterest embedding. Forward share matches what M4 needs (still widely seen when worn), and rolling cutoffs give more evidence than one cutoff over ~53 clusters |

### 2.1 D1 in detail: Phase 0, Pinterest into the pool (runs before M4)

**Step 0.1: Source images (blocking).** The 224×224 squashed images (fact 4a) are not good enough
for the GNN's silhouette, volume and colour features. **Decided 2026-10-07: the full-resolution
original Pinterest dataset is being supplied (option 1).** Options 2–3 are kept only as fallbacks.
1. Get the **full-resolution raw scrape** from the original scrape (`pinterest_data/raw/`).
2. Otherwise **re-download originals by pin ID**: every filename in `pinterest_dataset/images/`
   is a pin ID, so the same gallery-dl setup can fetch those exact pins at full resolution.
   `metadata.csv` keeps `pin_id`, `orig_width`/`orig_height` (e.g. 735×1102) and the original
   file's MD5 `hash`, so each re-download is checked against the original and mismatches are
   rejected. Pins deleted since the scrape are dropped and the count reported.
3. Last resort: use the 224×224 set, and report it as a known limitation with its own AUC.

**Step 0.2: Store-mode segmentation.** Add `--mode store` to
`Module3/pinterest_garment_segmentation.py`. It runs the same vendored M1 path as Fashion144k and writes, per
photo, the same artefacts the Fashion144k store was built from (crop PNG, `.npy`, `metadata.json`
with attributes + colour histogram). Then build `PinterestData/feature_store/` with the
**same builder** used for `Fashion144k_v1/feature_store/` (identical `row_dim` 570 layout). Guards:
`test_module1_sync.py`, `test_embedding_parity.py`, plus a new check that a Fashion144k photo
re-run through store mode reproduces its existing store row.

**Step 0.3: Pinterest split.** Seeded, outfit-level (= photo-level) train / val / test, matching
Fashion144k's proportions (~60 / 10 / 30).

**Step 0.4: AUC gate.** Run M2's existing real-vs-corrupted test (`make_negative_sample`) on
Pinterest test outfits with ≥ 2 garments.
- AUC within **0.03** of the Fashion144k test AUC → keep M2 frozen.
- Lower → fine-tune M2 on Fashion144k train + Pinterest train. Adopt it only if Pinterest test AUC rises
  **and** Fashion144k test AUC doesn't drop by more than 0.01. Report both.

**Step 0.6: Refit M3 + settle the trend formula (D11).** Refit `PerCategoryTrendModel` on the
full-resolution Pinterest garments, re-run cluster stability (ARI) and run the forward-share
rolling-origin backtest for growth weights {0.6, 0.3, 0}. Apply D11's rule, freeze the formula, and update
`Module3/README.md`. Runs **before** step 0.5's pool build (trend scores and cluster ids are cached there).

**Step 0.5: Unified pool.** `candidate_pool.py` concatenates both stores, tags each row with `source`
∈ {fashion144k, pinterest} and `split`, and serves `--pool eval` (test rows only) or
`--pool demo` (all rows). Every candidate is scored (D5), so small groups (e.g. Fashion144k test has 235
shorts and 548 sunglasses) need no special handling; their per-group counts are reported in eval.

**Not doing:** a third dataset. Revisit only if a swap group is still too small after Phase 0.

## 3. Architecture

New folder `Module4/`. Each unit has one job:

| Unit | Responsibility | Depends on |
|---|---|---|
| `candidate_pool.py` | Build once and cache (`Module4/cache/`, gitignored): per store row → source, swap group, trend score, M3 cluster id, split, duplicate-group representative flag (D8a), precomputed input projection (D5). Vectorised: per category, one `embeddings @ centroids.T` matmul; dedupe by chunked cosine on GPU | both feature stores, M3 `PerCategoryTrendModel`, splits |
| `calibrate_dup.py` | Contact sheet of pool pairs at cosine bands for visual t_dup calibration (D8) | `candidate_pool`, crop PNGs |
| `swap_groups.py` | `swap_group(category)` → group key, and `CooccurrenceFilter.allowed(cand_type, other_types)` (D4). Group mapping is a pure function; the filter loads a cached type×type count matrix built from train outfits | `index.json`, split |
| `recommender.py` | `Recommender.recommend(outfit_dir) -> Recommendation`: baseline scores, per-piece shortlist (D5), batched GNN scoring of swapped outfits, combined score (D3), dedupe (D8), pick piece (D2), threshold (D7) | M2 `PhotoScorer` (reused for GNN + dataset + percentile), M3 model, `candidate_pool` |
| `render_recommendation.py` | Slide figure per verdict (D6): outfit with the swap piece highlighted → suggestion cards (crop, type change, harmony/trend before→after, source, "what changed") + per-piece panel | matplotlib, crop PNGs, `Recommendation` JSON |
| `explain.py` | Descriptive "what changed" line from attribute/edge-feature deltas (D6). Pure function | node scalars |
| `eval_recovery.py` | §6 protocol: dev/report sets, group-consistent corruption, all metrics, baselines, sweeps, calibration, bootstrap CIs | `recommender`, `swap_groups` |
| `human_study/` | Collage generation for the blind A/B study + analysis (win rates, Fleiss' κ) (§6.4) | `render_recommendation` |
| `README.md` | How to run, results | — |

`Recommendation` is a plain dataclass: `baseline {harmony, trend, combined}`,
`swap_piece {file, category, group}`, `suggestions [ {source, outfit_id, file, crop_path,
from_type, to_type, harmony_pct_before/after, trend_before/after, d_harmony_z, d_trend_z, gain,
what_changed: str} ×≤3 ]`, `also_improve [ {piece, best suggestion} ]`, `per_piece {file:
{best_gain, topk_mean_gain, slot_percentile, best_dH, best_dT, clears_tau}}`, `selection_rule: str`,
`verdict: "swap" | "refresh" | "keep"`. Serialises to JSON (`to_json()`).

## 4. Data flow (one photo)

```
photo ─► M1 run_single_image ─► outfit_dir (metadata.json, *.png, *.npy)
            │
            ├─► nodes = build_node_from_meta(...)           (M2, existing)
            ├─► r0 = GNN logit(nodes); T_i = M3 trend per piece   baseline
            └─► for each piece i:
                  cands = pool[swap_group(i)] ∩ cooc_ok(others)   (all of them; trend floor only with --no-trend-drop)
                  graphs = vectorised swap: node i := cand, edges recomputed in one tensor op
                  r = GNN logit(chunks ≤ 6 GB VRAM)              precomputed input projection
                  ΔH = (r − r0) / σ_logit ; ΔT = (trend(cand) − T_i) / σ_trend
                  gain = w_H·ΔH + w_T·ΔT                         (ranking)
                  display: H% = pct(r) vs full test reference, outfit-mean T, deltas
            ─► s_i = select_rule(gains_i)   (topk_mean k=5 | max | slot_percentile)
            ─► pick i* = argmax s_i ; top-3 for i*: best, then best from other M3 clusters (D8c)
               (pool already deduped; candidates ≥ t_dup to the current piece excluded)
            ─► s_i* ≥ τ ? swap : (∃ cand ΔT>0 ∧ ΔH≥−ε ? refresh : keep)
```

Performance: graph construction (`build_graph` is a Python loop over edges) was the
bottleneck, not the GNN, so the vectorised swap builder is the primary path (D5). Gate: ≤ 5 s
warm for `samples/image2.jpg` against the full demo pool, with VRAM capped at 6 GB.

## 5. Error handling

- 0 garments detected → "no garments detected" (same as M2/M3).
- 1-garment outfit → still works (the swap changes harmony relative to an outfit of 1).
- Piece or candidate whose category has no trend model, or no confident cluster match → ΔT = 0
  (neutral), flagged in the output. For display, an unscored current piece counts as the outfit mean.
- Empty shortlist (only possible with `--no-trend-drop` or a tiny group after the D4 filter) → relax the
  trend floor for that piece; if still empty, skip the piece and flag it.
- VRAM: chunk size derived from the 6 GB cap; on CUDA OOM, halve the chunk and retry. No GPU → CPU
  with a warning (slow, but correct).
- Missing crop PNG for a suggestion → show the text row without the image.

## 6. Evaluation (D9)

### 6.1 Protocol
- **Outfit sets:** per source (Fashion144k; Pinterest after Phase 0), seeded draws from the test split:
  **dev = 2,000** outfits (all tuning: D2 rule, D3 weight, D7 τ/ε, D8 diversity) and a disjoint
  **report = 2,000** outfits (final numbers, run **once** with the frozen settings). Pool = `--pool eval`
  (test rows only), excluding the source outfit.
- **Corruption (group-consistent):** replace one random piece with a random garment from the **same
  D4 swap group** that passes the co-occurrence filter, i.e. the same space M4 searches. Verified
  2026-10-07: M2's `make_negative_sample` (`03_train_gnn.py:135`) draws from the same *broad*
  category (`02_outfit_dataset.py:201`), so >50% of shoe corruptions become bag/hat/scarf/belt/sunglasses.
  That makes the task too easy and often impossible to undo under D4. M2's original corruption is used
  **only** in the Phase 0 AUC gate, for comparability with M2's reported AUC.
- **Severity strata:** every result is also reported by thirds of the corruption's own harmony drop
  (subtle / medium / obvious).
- **Real (uncorrupted) outfits** are run alongside, for the D7 verdict calibration.

### 6.2 Metrics
1. **Weak-link hit rate:** does M4 pick the corrupted piece? vs chance (mean 1/#pieces) and baselines.
2. **Original recovery** (headline, ground truth = what the person actually wore): rank of the
   removed original garment among all candidates for that slot. Report recall@1/3/10 and median
   percentile rank. If D8a collapsed the original into a duplicate group, its representative counts.
3. **Harmony recovery (ΔH):** harmony of corrupted → M4-fixed → original. **Labelled GNN-judged
   (circular).**
4. **Weight sweep (D3):** w_T ∈ {0, 0.1, …, 1}. For each, mean ΔH and ΔT of the top suggestion plus hit rate.
   Pareto plot `demos/recommendation_tradeoff.png`. Default w_T = the largest value keeping ≥ 90% of the
   w_T = 0 harmony gain. w_T = 0 and w_T = 1 are the harmony-only and trend-only ablations.
5. **Verdict calibration (D7):** τ by balanced accuracy (real vs corrupted), ε = ΔH noise band on real
   outfits. Report the false-alarm rate on real outfits (an upper bound, since some real outfits are improvable) and the
   detection rate on corrupted ones.
6. **Selection-rule ablation (D2):** hit rate for `max_gain` vs `topk_mean_gain` vs `slot_percentile`
   (no extra GNN calls). The winner becomes the default.
7. **Latency / VRAM (D5):** ≤ 5 s warm, ≤ 6 GB peak.

### 6.3 Baselines (same corruptions)
- **Random:** random piece + random same-group candidate (controls for "any change looks like a fix").
- **Trend-only:** w_T = 1.
- **CLIP coherence (no GNN):** weak link = the piece with the lowest mean cosine to the other pieces;
  replacement = the candidate with the highest cosine to the mean of the other pieces.

### 6.4 Human A/B study (independent judge)
Blind pairwise preference on outfit collages: **M4-fixed vs corrupted** and **M4-fixed vs random-swap
fix**. ~100 report-set outfits, 8–10 volunteer raters, each pair rated by ≥ 3 raters, with randomised
left/right order. Report win rates with 95% CIs and inter-rater agreement (Fleiss' κ). Rating page:
a shared web page that stores votes (built at implementation time).

**Under consideration (not in scope):** a vision-language-model judge (e.g. Claude
rating outfit pairs) as a scalable second opinion, first validated on real vs corrupted pairs; it
costs API credits.

### 6.5 Statistics & caveats
- 95% bootstrap CIs on every metric. Paired tests between methods: McNemar for hit rates, paired
  bootstrap for ranks. Results broken down by source and by swap group.
- Caveat: the GNN both detects and fixes, so hit rate and ΔH partly measure the GNN against itself.
  Original recovery, the CLIP-coherence baseline and the human study are the independent evidence.

## 7. Testing

- `swap_groups`: unit tests for every specific type in the store (16 types, incl. `bag`). Filter
  tests: skirt+leggings outfit → swapping skirt never yields pants; `--same-type` returns only
  the current type.
- `recommender` ranking/fusion: unit test with a stub scorer + stub trend model on a
  3-piece synthetic outfit; asserts standardised gains and weights (D3), each selection rule (D2),
  `--no-trend-drop`, own-item exclusion and cluster-diverse top 3 (D8), and all three verdicts (D7).
- Smoke: `recommend()` on `samples/image2.jpg`'s M1 output returns a valid `Recommendation` (verdict + ≤ 3
  suggestions, JSON round-trips).
- **Demo (D10):** headless `analyze(image2.jpg, All)` returns all four module outputs; moving the trend
  slider re-ranks without a GNN call (asserted via a call counter); M2 and M4 report the same
  harmony % for the unswapped outfit.
- **Parity (D5):** vectorised swap builder == `build_graph` (edge_index + edge_attr, exact) and
  precomputed projection == model's first layer (allclose), on ≥ 1,000 random swaps.
- **Performance (D5):** `samples/image2.jpg`, full demo pool, ≤ 5 s warm, peak VRAM ≤ 6 GB.
- Existing guards (`test_module1_sync.py`, `test_embedding_parity.py`) still pass.
   