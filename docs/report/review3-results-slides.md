# Review 3 – Results and Discussion slides

Slide content for the "Results and Discussion" section of `PPT-Template_Phase3- Review3.pptx`
(the template allows as many slides as needed). Every number is from `docs/report/results.md`.
Each slide has: on-slide text, the visual to place, and speaker notes. The R2, R4 and R7 charts
are made by `python docs/report/make_results_charts.py` into `docs/figures/` (PNGs are gitignored,
so re-run it after a clone).

---

## Slide R1 – Results at a glance: expected vs achieved

**On slide** (table)

| Objective | Expected | Achieved | Status |
|---|---|---|---|
| Find garments in real photos | Reliable garment detection | Upper body 97.6 %, lower body 94.8 % recall; 16 failures in 144,169 photos | ✅ Met |
| Score outfit harmony | AUC ≈ 0.80 | 0.80 (Fashion144k); 0.73 (Pinterest) | ✅ Met / ⚠️ weaker on modern photos |
| Trend score | Forecast rising styles | Stable clusters, but **does not forecast** (ρ < 0) | ⚠️ Changed to "current popularity" |
| Recommend the swap | Beat simple baselines | **72 % vs 42 %** best baseline (chance 31 %) | ✅ Met |
| Suggestions people prefer | Beat a random swap | **Lift −0.4 points** (26 raters) | ❌ Not shown |
| Working demo | ≤ 5 s per recommendation | 0.16 s per recommendation; ~4 s for all modules | ✅ Met |

**Speaker notes.** Four of six objectives were met as planned. Two results changed the project:
the trend backtest (score is descriptive, not a forecast) and the human pilot (no measurable lift
over random). Both are covered in the next slides.

---

## Slide R2 – Module 1: Perception accuracy

**On slide**

- No labelled test set exists, so we used the **garment words in each Fashion144k photo title** as
  weak labels (all 144,169 photos; recall, lower bound)
- **Regions found reliably:** upper body **97.6 %**, lower body **94.8 %**, bag 83.8 %, shoes 75.0 %
- **Weak spots:** subtypes (shorts read as pants: 4.4 %), small accessories at 400 px (belt 19 %,
  sunglasses 6 %)
- **Review 1 suggestion, non-human subjects:** YOLO11 person/animal check
  - Before: a dog's face scored as a "shirt" with harmony 100/100
  - After: **13 / 13** animal-only photos stopped, **0 / 600** real photos stopped by mistake

**Visual.** `docs/figures/r2_m1_recall.png` (recall by garment group), plus
`docs/figures/nonhuman_subject_check.png`.

**Speaker notes.** Subtype errors stay inside one swap group (shorts and pants are both "lower"), so
Module 4 still searches the right replacements. We compared three ways of spotting animals; CLIP's
"is this a person?" score was rejected because it scored people in sarees and lehengas below 0.3.

---

## Slide R3 – Modules 2 and 3: Harmony and Trend

**On slide**

*Harmony GNN (3 variants × 3 seeds, official test split)*

| Variant | Mean AUC |
|---|---|
| **Attribute subset (shipped)** | **0.801** |
| CLIP + colour only | 0.792 |
| All attributes | 0.786 |

- Statistical tie (seed noise ±0.02); noisy attributes never help
- Transfer: Fashion144k **0.816** → Pinterest **0.727** (trained on 2010s photos)

*Trend model*

- Cluster stability (ARI): dress 0.96, accessory 0.89, bottom 0.76, top 0.61
- Backtest: growth does **not** predict future share (ρ −0.10 at 3 months, −0.45 at 1 month)
- → Trend score presented as **current popularity**, not a forecast

**Visual.** Two small tables, or one bar chart of AUC by variant with seed spread as error bars.

**Speaker notes.** An earlier single-seed run claimed the baseline beats attributes; three seeds
showed that was noise, so every result is reported over three seeds. Trend clusters that have just
surged tend to fall back (mean reversion), so we changed the wording rather than claim a forecast.

---

## Slide R4 – Module 4 vs other methods

**On slide**

*Does Module 4 find the spoiled piece? (2,000 test outfits per dataset)*

| Method | Fashion144k | Pinterest |
|---|---|---|
| **Module 4 (GNN + trend)** | **72.2 %** | **69.3 %** |
| CLIP similarity (no GNN) | 42.2 % | 42.7 % |
| Trend only | 32.6 % | 34.5 % |
| Random | 30.4 % | 37.8 % |
| Chance | 31.4 % | 36.0 % |

- The **real garment the person wore** ranks in the **top 4.8 % / 5.4 %** of candidates
  (baselines: top 31–52 %)
- All baselines below Module 4 at p < 1e-60 (McNemar test)
- Only **3 points lower on Pinterest**, despite the GNN's 0.09 AUC drop
- **0.16 s** per recommendation, every candidate scored (target ≤ 5 s)

**Visual.** `docs/figures/r4_m4_vs_baselines.png` (hit rate by method, one panel per dataset,
95 % CIs, chance as a dashed line).

**Speaker notes.** The test spoils one random piece and checks whether Module 4 picks it. The
original-rank measure is the stronger evidence, because it is a real person's choice. Accuracy
depends on how much the spoiled piece hurts: 55 % for subtle cases, 88 % for obvious ones.

---

## Slide R5 – Trade-offs: trend weight and selection rule

**On slide**

- Score = 0.8 × harmony gain + 0.2 × trend gain
- From weight 0 → 0.2: trend gain **0 → 1.55σ**, harmony gain kept **94 %**, hit rate 75.7 % → 72.2 %
- Trend alone (weight 1.0) makes outfits **less** harmonious → trend must not overpower compatibility
- Selection rule ablation: max gain 70.0 %, top-k mean 70.5 %, **share of candidates that beat the
  worn piece 72.2 %**
- Swap threshold calibrated on a separate dev set: **"swap" for 70 % of spoiled outfits, 16 % of real
  ones**

**Visual.** `demos/recommendation_tradeoff.png` (the trend-weight curve).

**Speaker notes.** 0.2 is the largest trend weight that keeps at least 90 % of the harmony gain on
the dev set. The slider in the demo lets the user move along this curve. The 16 % on real outfits is
an upper bound on false alarms, since some real outfits can be improved.

---

## Slide R6 – Demo catalogue: H&M product photos

**On slide**

- Suggestions now come from **54,737 clean H&M product cut-outs** (street crops were 35 % unreadable)
- Problem found: the GNN favoured studio photos (**+0.43σ**, a photo-style bias, not taste)
- Fix: **CORAL alignment** of H&M embeddings to street photos per garment type

| | Raw | Mean shift | **CORAL** |
|---|---|---|---|
| Classifier tells H&M from street (0.5 = can't) | 0.98 | 0.76 | **0.60** |
| Best H&M vs best street clothing | – | −0.3 to −0.4σ | **≈ 0** |

- **Manual QA (Kit 4):** 24 of 480 random items flagged (**5.0 %**), mostly H&M's own wrong labels
- Verdicts unchanged vs street suggestions (swap 57 vs 56 of 400 outfits)

**Visual.** `demos/recommendation_demo.png` (one example per verdict).

**Speaker notes.** H&M is used only for what the demo shows. All evaluation numbers on R4–R5 use
street photos, so they are unaffected. Remaining issue: H&M swaps pick shoes about twice as often
as street swaps.

---

## Slide R7 – Human study pilot (Kit 1)

**On slide**

- **Question:** do people prefer Module 4's swap to a random allowed swap?
- Blind A/B form: 20 outfits × 2 = 40 questions, **26 raters, 1,040 answers**
- Both replacements from the same H&M catalogue, so photo style cancels out

| | Prefer the swapped outfit |
|---|---|
| Module 4's swap | **37.4 %** |
| Random allowed swap | **37.8 %** |
| **Lift** | **−0.4 points** (95 % CI −5.3 to +4.8) |

- Model's harmony gain vs human preference: Spearman **0.10**
- **Result: no measurable lift. Reported as a negative pilot.**

**Visual.** `docs/figures/r7_human_study.png` (share preferring each swap, and the lift with both
95 % intervals).

**Speaker notes.** No position bias (A chosen 53.5 %) and no rater dominates. Pinterest outfits
leaned positive (+9 points) and Fashion144k negative (−10), neither significant. Likely reasons:
the GNN learned to catch clear intruders, not finer taste; studio cut-outs next to street photos
look odd; and 20 outfits is a small sample. This is the main input to our future work: a
style-invariant garment encoder and a GNN retrained on modern outfits with harder negatives.

---

## Slide R8 – System testing results

**On slide**

*Performance (RTX 3070 Ti)*
- Cold start: **27 s** first run (models load), then **~4 s** for all modules; single module 0.7–1.8 s
- Trend slider re-ranks in **< 1 s** without re-running the GNN

*Usability walkthrough (Kit 3)*
- 10 of 12 checks passed; 5 of 10 unusual inputs handled well
- **14 issues:** 0 blocker, 4 major, 6 minor, 4 cosmetic
- Major: photos without a person (landscape, flat-lay) still analysed; harmony 77 vs 79 on reruns;
  44 MP photo runs out of GPU memory; switching to CPU after a GPU run crashes

*Automated tests*
- **160 tests, all passing** (Module 1: 23, Module 3: 7, Module 4: 130)

**Visual.** Severity bar (0 / 4 / 6 / 4) and a screenshot of the demo.

**Speaker notes.** The usability run was a scripted browser walkthrough, not first-time users.
Module 2 is checked by its AUC gate and by a test that Module 4's batched GNN matches Module 2's
original code. The four major bugs are logged with fixes planned.

---

## Slide R9 – Deviations from the plan and why

**On slide**

| Planned | What happened | Reason | Action taken |
|---|---|---|---|
| Trend score forecasts rising styles | Negative correlation with future share | Styles that surge fall back (mean reversion) | Reworded as current popularity |
| GNN within 0.03 AUC on Pinterest | 0.09 drop | Trained on 2010s Fashion144k | Fine-tune deferred: costs Module 4 only ~3 points |
| Suggestions from street photos | 35 % unreadable crops | Small, merged or captioned crops | H&M catalogue + CORAL alignment |
| Human study shows a lift | Lift ≈ 0 | GNN learns intruders, not taste (likely) | Style-invariant encoder + GNN retrain (future work) |
| Phone-photo test set (Kit 2) | Not done | No team photos collected | Listed as a limitation |

**Speaker notes.** Each deviation was measured, explained and either fixed or turned into future
work. The negative results are reported as they are, not hidden.
