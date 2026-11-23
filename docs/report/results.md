# Chapter 8 – Results and Discussion

This chapter reports what was measured for each module and for the system as a whole, and what the
numbers mean. Every figure comes from a run recorded in the repository (module READMEs,
`docs/next-steps.md`, and the result files under `Module4/cache/` and `Fashion144k_v1/`). Where a
result is negative or a test was not done, it is reported as such.

Sections 8.2–8.5 cover the four modules in pipeline order. Section 8.6 reports the human study,
Section 8.7 the system-level tests, and Section 8.8 discusses the results together.

---

## 8.1 Evaluation Setup

### 8.1.1 Data

| Dataset | Size used | Split | Used for |
|---|---|---|---|
| **Fashion144k** | 144,169 photos → 480,079 garments | Official train / val / test split (`split.mat`): 86,501 train outfits, 43,250 test outfits | GNN training and testing; Module 4 evaluation; harmony reference |
| **Pinterest scrape** | 61,118 images (55,781 unique pins) → 154,663 garments; dated 2011-10 to 2026-09-11 | Seeded 60 / 10 / 30 split, identical files kept together (`split.json`) | Trend model; Module 4 evaluation; GNN transfer test |
| **H&M product catalogue** | 55,862 adult articles (54,737 pass the crop gate) | None (never used for evaluation) | Suggestions shown in the demo only |
| `samples/nonhuman/` | 16 Unsplash photos (13 dressed pets, 1 undressed dog, 2 person + pet) | – | Person / animal check |

The Module 4 evaluation draws two disjoint, seeded sets of 2,000 **test-split** outfits per street
catalogue: a **dev set** for all tuning and a **report set** that was run once with frozen settings.
No reported Module 4 number was tuned on the outfits it is reported on.

### 8.1.2 Hardware

All runs used one Windows machine with an NVIDIA RTX 3070 Ti (8 GB). Module 4 is capped at 6 GB of
GPU memory so that the other models fit beside it.

### 8.1.3 Metrics

| Metric | Module | Meaning |
|---|---|---|
| **AUC** | M2 | Share of (real outfit, corrupted copy) pairs where the real outfit scores higher. 0.5 = guessing, 1.0 = perfect |
| **Weak-label recall** | M1 | Share of photos whose title names a garment where Module 1 found that garment |
| **ARI** (Adjusted Rand Index) | M3 | Agreement between clusterings fitted on two 80 % resamples. 1 = identical, 0 = chance |
| **Spearman ρ** | M3 | Rank correlation between the trend score now and a style's later growth or share |
| **Hit rate** | M4 | Share of corrupted outfits where Module 4 picks the corrupted piece as the one to swap |
| **Original's median rank** | M4 | Where the garment the person actually wore ranks among all candidates for that piece (top x %) |
| **False-alarm rate** | M4 | Share of real, uncorrupted outfits that get a "swap" verdict (an upper bound: some real outfits can be improved) |
| **Lift** | Human study | Share preferring Module 4's swap minus share preferring a random allowed swap |

95 % confidence intervals are bootstrap intervals unless stated otherwise.

---

## 8.2 Module 1 – Perception

### 8.2.1 Processing at scale

| Run | Input | Output | Failures |
|---|---|---|---|
| Fashion144k batch | 144,169 photos | 480,079 garment crops with embeddings and attributes | 16 photos (≈ 0.01 %), ~8–9 images/s |
| Pinterest store mode | 55,781 photos (resized to the 98,304-px Fashion144k frame) | 154,663 garments | 560 photos with no garments found |
| H&M catalogue | 55,897 adult articles | 55,862 garment folders | 35 empty cut-outs |

### 8.2.2 Garment detection accuracy (weak labels)

Fashion144k has no garment-level ground truth. As a substitute, `Module1/eval_weak_labels.py` uses
the garment words in each photo's title as weak labels and measures **recall** over all 144,169
photos (outputs reused, nothing re-segmented). This is a **lower bound**: titles sometimes name
items that are hidden or out of frame. Precision cannot be measured this way, because titles omit
items.

| Group | Photos naming it | Recall | Exact subtype recall |
|---|---|---|---|
| Upper body | 68,774 | **97.6 %** | jacket 74 %, sweater 48 %, shirt/blouse 30 %, t-shirt 22 %, top 14 % |
| Lower body | 47,203 | **94.8 %** | skirt 86 %, pants 75 %, leggings 42 %, shorts 4.4 % |
| Bag | 23,714 | 83.8 % | |
| Hat | 10,823 | 79.5 % | |
| Dress | 24,695 | 76.6 % | |
| Shoes | 48,042 | 75.0 % | |
| Scarf | 3,304 | 62.5 % | |
| Belt | 3,127 | 19.4 % | |
| Sunglasses | 4,861 | 5.6 % | |

95 % Wilson intervals are within ±1 point for every group except belt (±1.4) and scarf (±1.7).

**Discussion.**

- **Garment regions are found reliably.** Upper and lower body are found in 95–98 % of the photos
  that name them.
- **Subtypes are the weak point.** SegFormer gives one region per body part and CLIP zero-shot
  names it. When a title names both a jacket and a shirt, only one can match, and CLIP leans
  towards "jacket" (7,751 shirt/blouse misses were read as jacket).
- **Shorts are almost always called pants** (5,969 of 9,149 misses). Module 1 outputs only 844
  shorts among 480,079 garments. Shorts and pants are in the same Module 4 swap group, so this
  does not change which replacements are searched.
- **Small accessories are often missed** at Fashion144k's 400-px resolution: belts 19 %, sunglasses
  6 % (sunglasses pushed up on the head are labelled as hair).
- Tights and stockings have no Module 1 type and come out as pants or leggings in 87 % of 3,855
  photos.

### 8.2.3 Non-human subjects (Review 1 suggestion)

Review 1 asked how the pipeline behaves on animals. Before the change, every one of the 16 test
photos produced an outfit: the undressed golden retriever's face became a "shirt_blouse" with
harmony 100/100, and a cat in a suit produced 6 garments. The median harmony of the 13 dressed
pets was 89.7, and all of them received suggestions.

Three signals were compared before choosing one:

| Signal | Animal-only photos | Real outfit photos | Decision |
|---|---|---|---|
| Whole-image CLIP P(person) | All ≤ 0.23 | 29 % of 300 Pinterest photos < 0.3 (sarees, lehengas, sherwanis) | Rejected |
| SegFormer body-part pixel share | Overlaps with real photos | – | Rejected |
| **YOLO11m-seg: "animal and no person"** | **13 / 13 caught** | **0 / 600 fired** (300 Fashion144k, 300 Pinterest) | **Adopted** |

| Photo group | Before | After |
|---|---|---|
| 13 animal-only photos | 13 outfits scored | **13 stopped** with "No person found, only an animal" |
| 2 photos of a pet with only a person's legs visible | Garments taken from the pet | "No garments found" |
| 2 person + pet photos | Pet included in the outfit | Pet masked out; e.g. man with cat: harmony 31 → 41 |

The check adds about 0.3 s per photo (CPU). Its limits are covered in Section 8.7.2: it stops
animals only, so a landscape or a flat-lay is still analysed.

---

## 8.3 Module 2 – Harmony GNN

### 8.3.1 Compatibility accuracy

Three node descriptions were trained on the full training split, each with three seeds, and tested
on the official test split (best-validation checkpoint):

| Variant | Node size | Seed 42 | Seed 43 | Seed 44 | **Mean** | Std |
|---|---|---|---|---|---|---|
| **Attribute subset** (shipped) | 555 | 0.8155 | 0.7917 | 0.7958 | **0.8010** | 0.0104 |
| Baseline (CLIP + colour only) | 544 | 0.7951 | 0.8140 | 0.7677 | 0.7923 | 0.0190 |
| Full attributes | 570 | 0.7780 | 0.7915 | 0.7885 | 0.7860 | 0.0058 |

**Discussion.**

- The GNN ranks a real outfit above its one-garment-corrupted copy about **80 %** of the time.
- **The three variants are a statistical tie.** The seed-to-seed spread (±0.02) is larger than the
  gaps between variants. The only effect that holds on every seed is that full attributes never
  win: CLIP's noisy zero-shot guesses for pattern, fabric and fit do not help.
- The attribute subset was shipped because it has the highest mean and keeps the interpretable
  attributes Modules 3 and 4 use, at no measured cost. The shipped checkpoint is seed 42
  (`ck_attr_subset_s42`, test AUC 0.8155).
- **No overfitting:** validation and test AUC agree (0.7784 vs 0.7780 on the run checked).
- An earlier single-seed comparison had claimed the baseline beats attributes by 0.017. The
  three-seed run showed this was a seed artefact, which is why all variants are reported over three
  seeds.

### 8.3.2 Training efficiency

| Change | Effect |
|---|---|
| Precomputed feature store (480,079 rows × 570, memory-mapped) | Training never re-opens images; identical results |
| GPU minibatching | 10-epoch pilot ~300 s → 14 s, AUC unchanged |
| Rejection sampling for negatives | ~27 min → ~1 min per epoch (GPU was 3 % busy before) |
| **Result** | A full 10-epoch run takes about **11 minutes** |

### 8.3.3 Transfer to other photo styles

The GNN was trained on 2010s Fashion144k street photos. Two checks measured how well it transfers.

**Pinterest (modern street photos).** `Module2/auc_gate.py` measures the same real-vs-corrupted AUC
on both test sets:

| Test set | Outfits | AUC |
|---|---|---|
| Fashion144k | 41,713 | **0.8155** |
| Pinterest | 14,560 | **0.7272** |

The drop of 0.088 fails the planned gate (≤ 0.03). The same drop appeared on a 224 × 224 pilot
(0.731), so image resolution is not the cause. Fine-tuning on Pinterest was deferred, because the
drop costs Module 4 only about 3 points of hit rate (Section 8.5.2).

**H&M (studio product photos).** `Module4/hm_domain_check.py` tested whether the GNN can rank H&M
candidates, using the candidate nearest to the worn piece as a stand-in for the original
(lower rank = better; chance = 0.5):

| Measure | Raw H&M | Mean shift | **CORAL** (shipped) | Street candidates |
|---|---|---|---|---|
| Linear classifier tells H&M from street (0.5 = chance) | 0.98 (sample) | 0.76 | **0.60** | – |
| Average logit of H&M vs street candidates | +0.43σ | ≈ −0.05σ | ≈ 0 | – |
| Best H&M vs best street candidate, clothing | – | −0.30 to −0.42σ | **−0.00 to +0.12σ** | – |
| Stand-in GNN rank | 0.46 | 0.38 | **0.36** | 0.28 |

On raw H&M embeddings the GNN was close to chance and scored studio photos 0.43σ higher than
street photos purely for their photo style. CORAL alignment (matching each type's mean and
covariance to the street stores) removed the bias and recovered part of the ranking ability, but
not all of it (0.36 vs 0.28). This is why H&M is used only for suggestions shown in the demo and
never for evaluation.

---

## 8.4 Module 3 – Trend Analysis

### 8.4.1 Cluster stability

80/20 resampling, ARI, on the shipped model (`per_category_trend_model_store.pkl`, 149,320 Pinterest
garments up to 2026-08):

| Category | k | ARI |
|---|---|---|
| Dress | 8 | **0.955** |
| Accessory | 15 | 0.888 |
| Bottom | 15 | 0.759 |
| Top | 15 | 0.611 |

On the earlier, smaller manifest the values were accessory 0.885, bottom 0.881, dress 0.864 and top
0.759. Dresses were unstable at k = 15 (ARI 0.58). A k sweep showed the cause was k, not the data
(ARI 0.95 at k = 4, 0.94 at k = 8, 0.89 at k = 12, 0.58 at k = 15), so dresses use k = 8. Tops are
now the least stable category and may need their own k.

### 8.4.2 Does the trend score predict the future?

**Temporal holdout.** The model was fitted on data up to 2026-05 and its growth signal compared with
what actually happened afterwards (Spearman ρ, pooled over 53 clusters):

| Window | Accessory | Bottom | Dress | Top | **Pooled ρ** |
|---|---|---|---|---|---|
| 1 month | −0.45 | −0.59 | −0.36 | −0.39 | **−0.45** (p < 0.001) |
| 3 months (model default) | −0.31 | 0.25 | −0.86 | 0.20 | −0.10 (p = 0.48) |
| 6 months | −0.10 | −0.34 | −0.55 | −0.51 | **−0.35** (p = 0.01) |

**Forward-share backtest** (rolling cutoffs Jan–May 2026, 52 blocks): mean ρ between the score and a
style's share 1, 3 and 6 months later.

| Growth weight in the score | Mean ρ |
|---|---|
| 0 (recency only) | −0.253 |
| 0.3 | −0.243 |
| **0.6 (shipped)** | **−0.155** |
| 1.0 (growth only) | −0.081 |

**Discussion.** Growth has no forward predictive power at the default 3-month window and is
*negatively* predictive at 1 and 6 months: styles that have just surged tend to fall back. Every
weighting has a negative correlation with future share. The 0.6 weight was kept because it beats
recency-only (+0.098, CI +0.036 to +0.162), but the score is presented as **current popularity**
("Rising", "Frequently seen now"), never as a forecast. Pinterest's results also lean heavily
towards recent posts (~500 garments/month in 2024 vs ~27,000 in Aug 2026), so only relative
comparisons between clusters are used.

---

## 8.5 Module 4 – Swap Recommendation

### 8.5.1 Protocol

Real recommendations have no ground truth, so Module 4 is tested by corruption. In each test outfit
one random piece is replaced by a random garment from the same swap group that passes the
co-occurrence filter, which is exactly the space Module 4 searches. Module 4 must then:

1. **find the corrupted piece** (hit rate), and
2. rank the **garment the person actually wore** near the top of its candidates (original's rank).

The second measure is the more independent one, since it rests on a real person's choice rather
than on the model's own score. Candidates come from the test split of both street catalogues (the
eval pool, 190,602 representatives), never from the outfit itself or its reposts. Settings: rule
`slot_percentile`, w_t = 0.2, τ = 0.392, ε = 0.2σ, all calibrated on the dev set.

### 8.5.2 Main results

Report sets, 2,000 outfits each:

| Method | Fashion144k hit rate | Pinterest hit rate | Original's median rank (F / P) |
|---|---|---|---|
| **Module 4** | **72.2 %** (70.2–74.1) | **69.3 %** (67.3–71.3) | **top 4.8 % / 5.4 %** |
| CLIP coherence (no GNN) | 42.2 % (40.1–44.4) | 42.7 % (40.5–44.9) | top 31.5 % / 31.9 % |
| Trend only | 32.6 % (30.5–34.5) | 34.5 % (32.5–36.4) | top 45.5 % / 52.2 % |
| Random | 30.4 % (28.4–32.4) | 37.8 % (35.7–40.0) | top 50.6 % / 49.9 % |
| Chance (1 / pieces) | 31.4 % | 36.0 % | top 50 % |

Every baseline is below Module 4 with p < 1e-60 on both catalogues (exact McNemar test).

**Discussion.**

- Module 4 finds the corrupted piece about **2.3× as often as chance** and 30 points above the best
  baseline on both catalogues.
- **Pinterest is only ~3 points lower,** although the GNN's AUC drops by 0.09 there. Hit rate
  compares pieces within one outfit, so an error that shifts the whole outfit's score cancels out.
- **The original garment is in the top ~5 % of candidates,** far above every baseline, but it is
  rarely the single top pick (Recall@10 is 2.9 % / 1.7 % among thousands of candidates). The GNN
  rates its own top pick above the original in 96–97 % of outfits. Module 4 therefore finds
  replacements that are at least as harmonious *by the GNN's judgement*; whether people agree is
  tested in Section 8.6.

### 8.5.3 Breakdown

**By how strongly the corruption hurts harmony:**

| Corruption | Fashion144k hit rate | Pinterest hit rate |
|---|---|---|
| Subtle (mean drop 0.24σ / 0.12σ) | 54.6 % | 53.2 % |
| Medium (1.19σ / 1.18σ) | 73.8 % | 72.3 % |
| Obvious (2.41σ / 2.45σ) | 88.3 % | 82.3 % |

Below a harmony drop of about 0.2σ, the hit rate is no better than chance. This is Module 4's
detection limit, and it is the basis for the refresh tolerance ε = 0.2σ: a change smaller than that
"keeps the look".

**By garment group:**

| Group | F144k n | F144k hit rate | Pinterest n | Pinterest hit rate |
|---|---|---|---|---|
| Upper body | 531 | 68.5 % | 658 | 65.8 % |
| Lower body | 559 | 71.0 % | 557 | 65.9 % |
| Dress | 150 | 67.3 % | 205 | 72.2 % |
| Shoe | 393 | 78.4 % | 242 | 74.0 % |
| Bag | 227 | 69.6 % | 252 | 77.8 % |
| Hat | 77 | 84.4 % | 31 | 77.4 % |
| Scarf / belt / sunglasses | 63 | 77–100 % | 55 | 65–73 % |

Clothing is the hardest to judge; the small accessory groups have too few outfits for a reliable
rate.

### 8.5.4 Verdicts

At τ = 0.392, Module 4 says "swap" for **69.8 % / 70.3 %** of corrupted outfits and for **15.9 % /
17.1 %** of real ones (Fashion144k / Pinterest). The real-outfit figure is an upper bound on false
alarms, because some real outfits can genuinely be improved.

### 8.5.5 Trend weight

| w_t | Mean ΔH (σ) | Mean ΔT (σ) | Hit rate (F144k) |
|---|---|---|---|
| 0.0 | 2.49 | 0.03 | 75.6 % |
| 0.1 | 2.44 | 0.96 | 75.4 % |
| **0.2 (shipped)** | **2.34** | **1.55** | **72.2 %** |
| 0.3 | 2.21 | 1.91 | 66.2 % |
| 0.5 | 1.87 | 2.40 | 51.8 % |
| 1.0 | −0.64 | 2.59 | 32.6 % |

Moving from w_t = 0 to 0.2 raises the top suggestion's trend gain from ~0σ to 1.55σ (Pinterest
1.42σ) while keeping 94 % (93 %) of the harmony gain, at a cost of 3.4 points of hit rate. w_t = 0.2
is the largest weight that keeps at least 90 % of the harmony gain on the dev set. Trend alone
(w_t = 1) makes harmony *worse* on average, which confirms that trend must not overpower
compatibility. The demo slider lets the user move along this curve.

### 8.5.6 Selection-rule ablation

| Rule | Fashion144k | Pinterest |
|---|---|---|
| `max_gain` (best single candidate) | 70.0 % | 68.9 % |
| `topk_mean_gain` | 70.5 % | 68.8 % |
| **`slot_percentile`** (share of candidates that beat the worn piece) | **72.2 %** | **69.3 %** |

The rules differ by at most about 2 points. `slot_percentile` is slightly ahead and is less sensitive
to a single lucky high score.

### 8.5.7 Recalibration after Pinterest joined the pool

The settings were first calibrated on a Fashion144k-only pool. When Pinterest joined, the trend
model was refitted and the harmony reference changed, so the settings were recalibrated on the same
dev set (rule, w_t and ε came out unchanged; τ 0.333 → 0.392):

| | Fashion144k (F144k pool, 9 Oct) | Fashion144k (both stores) | Pinterest (both stores) |
|---|---|---|---|
| Hit rate | 71.8 % | 72.2 % | 69.3 % |
| Best baseline | 38.6 % | 42.2 % | 42.7 % |
| False alarms on real outfits | 21.1 % | 15.9 % | 17.1 % |
| Detection on corrupted outfits | 75.0 % | 69.8 % | 70.3 % |

The higher τ trades some detection for fewer false alarms: about 1 in 6 real outfits now gets
"swap", down from 1 in 5.

### 8.5.8 Demo catalogue results

The demo shows suggestions from the H&M catalogue only (Section 8.3.3). These results describe that
catalogue; they do not change the evaluation above, which never uses H&M.

**Crop quality.** Before the crop quality gate, 34.8 % of the suggestions shown on 200 random test
outfits were unreadable street-photo crops (31.4 % for Fashion144k inputs, 47.6 % for Pinterest).
After the gate, 0 % failed it, 97 % of outfits kept the same verdict, and the top suggestion's
harmony percentile moved from 87.3 to 87.0.

**Manual catalogue QA (Kit 4).** 480 random suggestable H&M items (30 per type × 16 types) were
checked by eye:

| Result | Value |
|---|---|
| Items flagged | **24 of 480 (5.0 %)** |
| By reason | wrong type 13, unreadable 6, other 4, background left in 1 |
| Worst types | sunglasses 13 % (thin frames mostly erased), scarf 10 % (ponchos, wraps) |
| Clean types | bag, belt, jacket, shoe (0 %) |

The main failure is H&M's own category labels, not the cut-outs. The rate is a lower bound: a chain
belt with studio background inside the loop, noted at build time, was not flagged.

**Verdicts and suggestion mix.** On 400 test outfits (200 per street catalogue), H&M suggestions
versus street suggestions for the same outfits:

| | H&M suggestions | Street suggestions |
|---|---|---|
| Swap / refresh / keep | 57 / 341 / 2 | 56 / 343 / 1 |
| Swaps that change the shoe | 18 of 57 | 9 of 56 |
| Median harmony gain of the top swap | +1.57σ | +2.08σ |

The verdicts are almost unchanged. H&M swaps still pick shoes about twice as often (before CORAL it
was 21 of 55), and the gains are smaller because the H&M pool is a sixth the size and no longer
boosted by photo style. Two fixes were made to refresh suggestions: at most one suggestion per worn
piece (`refresh_per_piece`), and a refresh must have a real trend gain and a positive combined gain.
The second fix removed all 8 suggestions with a non-positive gain (of 1,190) without changing any
verdict.

### 8.5.9 Speed

| Measurement | Result |
|---|---|
| One recommendation, 4-piece photo, Fashion144k pool, warm | **0.16 s**, 1.92 GiB peak GPU memory (target ≤ 5 s, ≤ 6 GB) |
| Same photo, H&M demo pool (command line) | 41,866 what-if outfits scored in 0.69 s |
| Request-time GPU memory, both street catalogues | ~2.3 GB peak |

Every allowed candidate is scored; nothing is sampled, so the planned fallback (a cheap pre-ranker)
was not needed.

---

## 8.6 Human Evaluation (Kit 1)

**Question:** when Module 4 says "swap this piece", do people prefer its replacement to a random
allowed replacement?

**Design.** 20 street test outfits with a "swap" verdict (10 Fashion144k, 10 Pinterest). Each was
shown twice, blind and with sides randomised: original vs Module 4's top swap, and original vs a
random replacement for the same piece from the same H&M catalogue. Both replacements are H&M product
photos, so any liking for (or against) studio photos cancels out in the lift. 26 raters outside the
team answered all 40 questions through a Google Form (22–23 Nov 2026): 1,040 answers, 59 "Can't
decide".

| Condition | Decided votes | Prefer the swapped outfit |
|---|---|---|
| Module 4's swap | 489 | **37.4 %** |
| Random allowed swap | 492 | **37.8 %** |
| **Lift** | | **−0.4 points** (95 % CI over raters −5.3 to +4.8; over raters and outfits −10.9 to +9.6) |

| Breakdown (not significant) | Lift |
|---|---|
| Pinterest outfits (10) | +9.3 points (CI −2 to +20) |
| Fashion144k outfits (10) | −10.4 points (CI −26 to +3) |
| Outfits where Module 4 beat random | 9 of 20 |
| Correlation of the model's harmony gain with human preference (40 questions) | Spearman 0.10 |

Data checks were clean: no position bias (Outfit A chosen 53.5 %) and no single rater dominates.

**Discussion.** **This pilot found no measurable lift.** People did not prefer Module 4's swap to a
random allowed swap, and the GNN's harmony gain barely tracks human preference (ρ = 0.10), even
though Module 4's swaps had much larger model gains (ΔH 1.3–4.5σ vs −1.4 to +1.4σ for random). The
original was preferred 62 % of the time in both conditions. That figure mixes "the outfit was fine"
with "a street photo next to a studio cut-out looks odd", so it does not on its own say the swap
verdicts are wrong, but the lift, which cancels the photo style, is about zero.

Three explanations fit the data and cannot be separated with 20 outfits:

1. the GNN, trained on 2010s Fashion144k with random-swap negatives, learned to spot intruders but
   not finer taste (consistent with the subtle-corruption hit rate of ~55 %);
2. the H&M suggestions are judged in a different photo style from the street outfit, which
   adds noise to both conditions;
3. the pilot is small (CI ±10 points over outfits).

The result is reported as a negative pilot. It directly motivates the planned future work: a
style-invariant garment encoder and a GNN retrained on modern outfit data with hard negatives.

---

## 8.7 System-Level Testing

### 8.7.1 End-to-end response time (demo)

| Step | Time |
|---|---|
| Server start | 4 s |
| First run, all modules (models load) | 27.1 s (the UI promises 30–60 s) |
| Later runs, all modules | 3.5–3.9 s |
| Single module | 0.7–1.8 s |
| Trend slider re-rank (no GNN call) | < 1 s |
| 12 MP / 24 MP photo | 8.2 s / 13.2 s |

### 8.7.2 Usability test (Kit 3)

A scripted walkthrough of the demo (`demo.py --advanced`) in a headless browser covered 12 checks and
10 unusual inputs. It was not done by first-time users, so "is it clear to a newcomer?" was not
tested.

| Result | Value |
|---|---|
| Checks passed | 10 of 12 (1 failed: advanced options; 1 not tested: first-time understanding) |
| Unusual inputs handled well | 5 of 10 |
| Issues found | **14:** 0 blocker, 4 major, 6 minor, 4 cosmetic |

| Input | Outcome | OK? |
|---|---|---|
| Pet photo | Stopped: "No person found, only an animal" | yes |
| Mirror selfie | Sensible garments and verdict | yes |
| PNG with transparency | Same result as the JPEG | yes |
| Non-image file (`.txt`) | Rejected on upload | yes |
| 110 px photo | Runs, but harmony 38 vs 77 at full size, with no warning | degraded |
| Landscape / flat-lay / product page | Garments invented from snow, a notebook read as a hat; a verdict given | no |
| Group photo | Analysed as one outfit, with no warning | no |
| 44 MP photo | GPU out of memory in SegFormer | no |
| Text file renamed `.jpg` | Bare "Error" | no |

The four major issues are: no stop for photos without a person (the check in Section 8.2.3 covers
animals only); the same photo scoring harmony 77 or 79 on reruns (traced to a reused output folder
that keeps garment files from earlier photos; fix pending); out-of-memory on very large photos
(downscale on upload); and a crash when switching the device to CPU after a GPU run.

### 8.7.3 Automated tests

| Suite | Tests | Result |
|---|---|---|
| Module 1 (`Module1/`) | 23 | All pass |
| Module 3 (`Module3/`) | 7 | All pass |
| Module 4 incl. demo UI (`Module4/`) | 130, including the `real` tests that need the data, GPU and cache | All pass |

Run on 2026-11-23: 160 tests, all passing.

Module 2 has no pytest suite. It is checked by `Module2/auc_gate.py` and by the evaluation itself;
`Module4/test_dense_gnn.py` checks that Module 4's batched GNN gives the same scores as Module 2's
original code.

### 8.7.4 Not done

**Kit 2 (phone-photo test set)** was not carried out: no team phone photos were collected, so there
is no measured success rate on real phone photos. The tooling (`Module4/photo_batch.py`) was tested
on `samples/nonhuman/` only.

---

## 8.8 Discussion

### 8.8.1 Results against the objectives

| Objective | Result | Verdict |
|---|---|---|
| Turn a real photo into described garments | Upper / lower body found in 95–98 % of photos (weak labels); 0.01 % failures on 144,169 photos; animals stopped 13/13 | **Met**, with weak subtypes and small accessories |
| Score how well an outfit's pieces go together | AUC 0.80 on Fashion144k, 0.73 on Pinterest | **Met** on its training domain; weaker on modern photos |
| Score how popular a garment's style is now | Stable clusters (ARI 0.61–0.96); score describes current popularity | **Met as a descriptive score**; it does not forecast (ρ < 0) |
| Pick the piece to change and what to change it to | 72 % / 69 % hit rate vs 42 % best baseline; original in top 5 %; 0.16 s | **Met** on the corruption benchmark |
| Suggestions people actually prefer | Lift −0.4 points vs random in a 26-rater pilot | **Not shown** |
| Usable demo | All modules run in ~4 s warm; 14 usability issues, 4 major | **Working**, with known input-handling gaps |

### 8.8.2 What the results show

- **The modular design works as intended.** Each module was measured separately, and the module
  results explain the system's behaviour: Module 1's subtype errors stay inside one swap group,
  Module 2's domain gap barely reaches Module 4's hit rate, and Module 3's trend term can be
  weighed against harmony on a measured curve.
- **Trend does not overpower compatibility.** At the shipped weight, 94 % of the harmony gain is
  kept while trend gain rises to 1.5σ. Trend alone makes outfits less harmonious.
- **Negative findings changed the design.** The trend backtest turned the trend score from a
  forecast into a description of current popularity. The H&M domain check led to CORAL alignment
  and kept H&M out of the evaluation. The human study shows that the GNN's notion of harmony is the
  main limit on recommendation quality.

### 8.8.3 Threats to validity

- **The corruption benchmark is self-referential in part.** Corrupted pieces are random garments,
  so the hit rate measures catching clear intruders, not taste. The original-rank measure is more
  independent but still uses the GNN's ranking.
- **Weak labels give recall only,** and as a lower bound.
- **The human study is a pilot:** 20 outfits, 26 raters, and suggestions in a different photo style
  from the outfit.
- **Mixed outfits** (a piece from one catalogue in an outfit from another) are suggested in the demo
  but not separately evaluated.
- **No phone-photo test set** (Kit 2 not done) and **no first-time-user test** of the interface.
- **Single machine:** all timings are from one RTX 3070 Ti.

---

## 8.9 Summary of Key Results

| Module | Key result |
|---|---|
| M1 Perception | 480,079 garments from 144,169 photos (16 failures); recall upper 97.6 %, lower 94.8 %, shoes 75.0 %; animal-only photos stopped 13 / 13 with 0 / 600 false stops |
| M2 Harmony | Test AUC 0.8010 mean over 3 seeds (shipped 0.8155); Pinterest 0.7272; 10-epoch training in ~11 min |
| M3 Trend | ARI dress 0.955, accessory 0.888, bottom 0.759, top 0.611; growth not forward-predictive (pooled ρ −0.10 at 3 months) |
| M4 Recommendation | Hit rate 72.2 % (F144k) / 69.3 % (Pinterest) vs best baseline 42.2 % / 42.7 % and chance 31.4 % / 36.0 %; original in top 4.8 % / 5.4 %; 0.16 s per photo |
| Demo catalogue | 54,737 H&M items; 5.0 % flagged in manual QA; CORAL removes the +0.43σ studio-photo bias |
| Human study | 26 raters, 1,040 answers: Module 4 37.4 % vs random 37.8 %, lift −0.4 points (no measurable lift) |
| Demo | 27 s cold, ~4 s warm (all modules); 14 usability issues (0 blocker, 4 major) |

---

## Suggested figures for Chapter 8

1. **Figure 8.1 – Module 1 output.** `demos/perception_demo.png`: segmentation overlay with garment
   types and attributes.
2. **Figure 8.2 – Non-human subject check.** `docs/figures/nonhuman_subject_check.png`: before and
   after on the pet photos.
3. **Figure 8.3 – Harmony score.** `demos/harmoniousness_demo.png` or
   `demos/compatibility_demo.png`: a real outfit vs a corrupted copy.
4. **Figure 8.4 – Trend score.** `demos/trend_demo.png`: per-garment trend with the least on-trend
   piece marked.
5. **Figure 8.5 – Hit rate vs baselines.** A bar chart of Table 8.5.2 for both catalogues, with the
   chance level as a line.
6. **Figure 8.6 – Harmony / trend trade-off.** `demos/recommendation_tradeoff.png`: the w_t sweep of
   Section 8.5.5.
7. **Figure 8.7 – Example recommendations.** `demos/recommendation_demo.png` (one example per verdict)
   and `demos/recommendation_board.png` (outfit with the swap applied).
8. **Figure 8.8 – Human study.** Bar chart of the share preferring each condition, with the lift and
   its 95 % interval.
