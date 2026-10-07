# Module 2 — Compatibility GNN · mini-README

*The whole story of Module 2 in one place: what it started as, what it's for, how it works,
every change made, a dedicated optimisations section, and the final Fashion144K results.*

---

## TL;DR

- **What:** turns Module 1's per-garment perception (SegFormer crops + CLIP embeddings +
  attributes) into **per-outfit graphs** and trains a **GNN** to score outfit *compatibility*
  ("does this outfit go together?"), supervised by Fashion144K's crowd *fashionability* votes.
- **Where it ended up:** a fully **local, GPU** pipeline (Colab/Drive/external-repo apparatus
  removed) that segments all **144,169** outfits, caches features once, and trains in **minutes**.
- **Headline result (full 86,501-outfit train split, official test set, seed-averaged 42/43/44):**
  **Test AUC ≈ 0.79–0.80 — a statistical three-way tie.** Attr-subset **0.8010 ±0.010** ≥
  CLIP+colour baseline **0.7923 ±0.019** ≥ full-attribute model **0.7860 ±0.006**; no pairwise
  gap is significant at n=3. The one robust effect: the **full 26-D attribute set is dominated**
  (removing the noisy CLIP-zero-shot pattern/fabric/fit one-hots never hurts). Chosen model:
  **attr-subset**, tied-best on AUC *and* it keeps the interpretable attrs/edges the downstream
  modules need — at zero accuracy cost.
- **Key engineering win:** found and fixed an O(pool) negative-sampling bottleneck that only
  appeared at full scale (GPU stuck at 3%, ~27 min/epoch → **~1 min/epoch, ~27× faster**).

---

## 1. What it initially was (baseline, as received)

Module2 arrived as a **Google Colab pipeline over the Fashion144K dataset** — three scripts
driven by a notebook:

| File | Role |
|---|---|
| `01_segment_batch.py` | Stage 1–2: bulk-run Module 1 (SegFormer + CLIP) over all Fashion144K images |
| `02_outfit_dataset.py` | Stage 3–5: build per-outfit graphs from `outputs/` + Fashion144K `.mat` files |
| `03_train_gnn.py` | Stage 6–7: define + train the CLIP-native Outfit GNN, evaluate by AUC |
| `fashion144k_sample1.ipynb` | Colab driver: mount Drive, unzip data, clone external repo, run 01→02→03 (since removed; replaced by `run_all.py`) |

**Data flow:**
```
Fashion144k images ──01_segment_batch──▶ outputs/<idx>/{metadata.json, *.png, *.npy}
        │                                        │
   photos list (ordering)                        ▼
   split.mat / relvotes.mat ──02_outfit_dataset──▶ per-outfit graphs
                                                  │
                                        03_train_gnn (BPR + AUC)
                                                  ▼
                                      clip_outfit_gnn_epochN.pt
```

The notebook mounted Drive, unzipped `Fashion144k_v1.zip`, **cloned an external repo**
(`github.com/external-org/AI-based-personal-stylist`) for `real_image_pipeline/`, then ran the
pilot and full passes.

**Dataset — Fashion144K:** ~144,169 real full-body outfit photos with user attribute labels and
a crowd-derived *fashionability* score. Ships `split.mat` (train/val/test ids) and
`relvotes.mat` (fashionability, 1–10).

---

## 2. The goal

Produce a single **outfit compatibility / "Harmoniousness" score**. Concretely:

1. Represent an outfit as a **graph**: nodes = garments, edges = pairwise garment relationships.
2. Learn, from real vs. deliberately-corrupted outfits, to **rank real outfits above corrupted
   ones** (a garment swapped for a mismatched one).
3. Evaluate honestly by **AUC on the official Fashion144K test split**, comparable to prior work.
4. Make retraining **cheap enough to sweep** (many parameter tweaks in one sitting).

---

## 3. How it works (architecture)

**Graph construction (`02_outfit_dataset.py`)**
- One graph per outfit; keep outfits with `metadata.json` and **≥2 garments**.
- **Node features (570-D):** `[512-D CLIP embedding | 32-D HSV colour histogram | 26-D attributes]`.
  - Colour histogram (16 hue + 8 sat + 8 val) computed fresh from each crop composited on white
    (same pixels CLIP sees) — replaces the dataset's `col_cco.mat`.
  - 26-D attributes from Module 1: one-hots for silhouette(4)/pattern(6)/fabric(6)/fit(3) + 7
    scalars (formality, hue sin/cos, saturation, value, extent, aspect). Missing → zeros.
- **Edge features (5-D, directed):** fashion-theory cues —
  `[category-pair weight, hue separation, formality difference, volume contrast (area log-ratio),
  same-category flag]`. A full `CATEGORY_EDGE_WEIGHTS` table (top↔bottom 1.0 … default 0.3)
  supplies the prior.

**Model (`03_train_gnn.py`) — `CLIPOutfitGNN`**
- `570-D → Linear projection → 2× EdgeAwareGNNLayer → mean-pool → outfit score`.
- **`EdgeAwareGNNLayer`:** weighted message passing where **both** the aggregation weight
  (`softplus(gate(edge_attr))`) **and** the message content (`x[src] + msg_edge(edge_attr)`) are
  functions of the edge features. Generalises the original fixed-weight layer.
- **No `nn.Embedding` / vocab lookup** — deliberately, to fix the old train/inference
  representation mismatch. Inputs are real CLIP vectors end-to-end.

**Training objective — fashionability-weighted BPR**
- **Positive** = a real outfit. **Negative** = the same outfit with **one garment swapped** for a
  same-category item from a *different* outfit.
- Loss pushes `score(positive) > score(negative)`, each pair weighted by the outfit's
  fashionability (`relvotes/10`). Adam, best-val checkpointing.
- **Metric:** AUC over positive-vs-corrupted score pairs (val each epoch; official test at the end).

---

## 4. What changed (journey)

### 4.1 Dropped Colab → local GPU
Removed the Drive/notebook/external-repo apparatus. `01_segment_batch.py` now imports the
**sibling `../Module1`** (`--module1_dir`, `--repo_dir` kept as legacy). New **`run_all.py`**
replaces the notebook (stages `segment`/`train`/`all`, `--device cuda`, paths derived from
`--data_root`, fails early if data is missing).

### 4.2 Fed Module 1 attributes into the graph
Added `encode_node_attributes` (26-D), `edge_features` (5-D), and upgraded the layer to
`EdgeAwareGNNLayer` (above). Node features widened 544-D → **570-D**.
*Not used:* `outfit_features.json` aggregates (the GNN needs the per-edge form, which is
recomputed instead).

### 4.3 Ablation flags
`--no_attributes` trains the **baseline**: nodes `[512 CLIP | 32 colour]` (544-D), edges = scalar
category weight (1-D) — the pre-integration model. `--attr_subset` trains the **denoised** variant:
keeps only the *reliable* attr dims (silhouette one-hot + the 7 colour/geometry/formality scalars =
11-D, node 555-D), dropping the noisy CLIP-zero-shot pattern/fabric/fit one-hots; edges stay 5-D so
only the *node* attrs are ablated. Together these let us ask *"do the attributes actually help AUC,
and is it the noisy ones dragging them down?"* The final log line is tagged
`with_attributes` / `no_attributes` / `attr_subset`.

### 4.4 Windows / local port (making the full run possible)
- **Unicode stdout fix** (`01_segment_batch.py`): Fashion144K has non-ASCII filenames (e.g.
  Polish `ń`); Windows piped stdout defaults to cp1252 and crashed mid-run. Now stdout/stderr
  reconfigure to UTF-8 (`errors=replace`) and fail-log writes use `encoding="utf-8"`.
- **SegFormer-only fp16 autocast** (`Module1/fashion_segmenter.py`): `torch.no_grad(), _amp(dev)`
  around the SegFormer forward *only* (fp32 model → real speedup), `logits.float()` before
  interpolate. Scoped **away from** the CLIP paths that caused the earlier dtype incident (§5).

### 4.5 Pilot-cap flag
`--limit_train N` (opt-in, defaults off) caps the number of **train** outfits (val/test stay
full) so pilots are reproducible without re-segmenting a subset.

---

## 5. Optimisations (the engineering core)

Goal: retrain the GNN **many times** cheaply. Everything result-preserving except graph
batching (parity-verified). Tiers 0–3 were designed/validated on 5k–10k pilots; **Tier 4** is the
fix that only the full 144k scale revealed.

| Tier | Optimisation | Effect |
|---|---|---|
| **0** | Segmentation: batched per-outfit CLIP embedding; **consecutive-failure circuit breaker** in `01` (`--max_consecutive_failures`, default 25) | avoids a systemic error silently marking the whole dataset "done" |
| **1** | **Feature store** (`build_feature_store.py`, `--feature_store`): one-time memmap of `[512 CLIP | 32 colour | 26 attr]` + scalars; colour histogram computed **once** | training never reopens a PNG; **bit-identical** graphs, exact CPU AUC match; ~1 GB for 144k via `mmap_mode='r'` |
| **2** | **GPU minibatching** (`--batch_size`): `collate` packs B outfits into one disconnected graph; segment-mean readout; vectorised BPR | **~20× faster** — a 10-epoch 5k run went **~300s → 14s**, AUC unchanged (parity test) |
| **3** | **Multi-negative** (`--num_negatives`, K=4 nudged 5k test AUC 0.6568→0.6665); `--val_eval_max` | better signal; DataLoader workers obviated (no I/O left) |
| **4** ⭐ | **Negative sampling O(pool) → O(1)** | **~27× faster at full scale** |

### fp16 caveat (Tier 0)
A blanket fp16 autocast was **reverted** early: it cast image features to float32 while cached
CLIP **text** features stayed float16 → `float != Half` matmul that failed *every* image. CLIP is
already fp16 on CUDA, so autocast added nothing there. The **later** SegFormer-only autocast
(§4.4) is the safe, scoped version (it never touches CLIP).

### Tier 4 — the full-scale bottleneck
- **Symptom:** ~**27 min/epoch**, **GPU at 3%** (CPU-bound). The store + batching had made the
  GPU work trivial, so nothing hid the CPU cost anymore.
- **Cause:** `make_negative_sample` did `[c for c in category_pool[cat] if c[0] != idx]` — a full
  scan/copy of the category pool **per negative**. At full scale pools are ~100k (`accessory`
  114,241), so every `batch × num_negatives` draw scanned ~100k entries. The 5k/10k pilots never
  exercised this (pools ~8.6× smaller → ~75× cheaper total), so Tiers 0–3 missed it.
- **Fix:** **rejection sampling** — pick a random pool entry, retry only on the rare same-outfit
  collision. O(1) amortised, **identical draw distribution**. → **~1 min/epoch**, GPU no longer
  starved, val/test AUC preserved.

---

## 6. Results

### 6.1 Does adding Module 1 attributes help AUC?

**Full-scale, seed-averaged (42/43/44), best-val, official 43,250-outfit test split:**

| Arm | node dim | edges | s42 | s43 | s44 | **mean** | std |
|---|---|---|---|---|---|---|---|
| Full attributes (26-D) | 570 | 5-D | 0.7780 | 0.7915 | 0.7885 | **0.7860** | 0.0058 |
| Baseline (`--no_attributes`) | 544 | 1-D | 0.7951 | 0.8140 | 0.7677 | **0.7923** | 0.0190 |
| **Attr subset** (`--attr_subset`, 11-D) | 555 | 5-D | 0.8155 | 0.7917 | 0.7958 | **0.8010** | 0.0104 |

**Verdict: a statistical three-way tie.** Ranking by mean is subset ≥ baseline ≥ full-attr, but
**no pairwise gap is significant at n=3** — per-seed noise (baseline std alone ±0.019) swamps the
between-arm gaps (~0.006–0.015). Paired by seed, subset ≥ full-attr on *all three* seeds
(monotone but t≈1.3, n.s.), while subset-vs-baseline **flips sign** (seed 43 the baseline won) →
a coin toss. **The compatibility score is essentially insensitive to the attribute treatment.**

**The one robust effect:** the **full 26-D set is dominated** — same-or-lower AUC than both
alternatives on every seed. So the noisy CLIP-zero-shot **pattern/fabric/fit** one-hots never
help; removing them (the subset) recovers reliable attrs back to ~baseline parity.

**⚠️ Correction to earlier single-seed claims.** Prior versions of this doc reported a full-scale
**seed-42-only** result — "baseline beats attributes by −0.0171" — as the verdict. That was a
**seed artifact**: seed 42 was simultaneously full-attr's worst *and* subset's best draw. With
three seeds the effect **evaporates**. Lesson: at ±0.02 per-seed noise, no full-scale single-seed
claim was trustworthy. (The earlier 5k seed-averaged result already agreed: at 5k the full-attr
set lost −0.013 — consistent with "the full noisy set is the worst arm".)

**Why (interpretation):** CLIP already encodes colour/pattern/formality *visually*, so the explicit
attributes are largely redundant — they can't add much, and the noisy dims add variance. Hence the
score barely moves whichever way you treat them.

**Chosen model — attr subset**, decided on the tiebreakers since AUC can't separate them: it is
tied-best on AUC (top mean, beats full-attr every seed) **and** free-carries the reliable
attributes + 5-D fashion-theory edges that Modules 3–4 (recommendation/explainability, trend
analysis) need — at **zero measured accuracy cost**. The full 26-D set is never worth shipping.

### 6.2 Full-run health
- **Segmentation:** 144,169 / 144,169 complete, 16 failures (~0.01%), ~8–9 img/s.
- **Feature store:** 480,079 garment rows × 570-D, ~1.04 GB; consistency verified
  (`index.num_rows == features rows == node_scalars rows == rows-in-index == 480,079`, 144,169
  outfits).
- **Training:** 83,352 usable train / 13,925 usable val. **val ≈ test** in both arms
  (with-attr 0.7784 val ≈ 0.7780 test) → **no overfitting**. Whole 10-epoch run ≈ 11 min after
  the Tier-4 fix.

### 6.3 Open caveats & future work
- All three arms are now **seed-averaged (42/43/44)** at full scale; the single-seed caveat is
  resolved (and it mattered — it overturned the seed-42 verdict, see §6.1).
- With only **3 seeds** the arms are a statistical tie; more seeds could resolve the ~0.009
  subset-vs-baseline gap, but the practical decision (ship subset, drop the full set) is stable.
- The subset's val AUC was **still climbing at epoch 10** on seed 42 (0.799→0.803) — it may be
  mildly *under*-reported vs the other arms, which had plateaued. More epochs is a cheap probe.
- **Learned attributes (Fashionpedia) — future work.** The current attributes are CLIP-zero-shot
  over the *same* encoder that gives the node embedding, so they're redundant by construction.
  Fashionpedia's supervised fine-grained attributes come from a different backbone and could add
  orthogonal signal the CLIP embedding lacks — the one attribute avenue with real upside left.
  Scoped as future work; needs no re-segmentation (run the tagger over saved crops, rebuild store).

### 6.4 Raising AUC above ~0.80 — ranked levers to try

**Framing first (important):** our own seed-averaged results (§6.1) show the between-arm gaps are
smaller than per-seed noise, so **~0.80 is close to the ceiling of the task *as currently
defined*.** That ceiling is set mostly by the **negative sampler**, not the model — so the biggest
levers attack the negatives/eval, not the architecture. Ranked cheapest-and-most-likely first:

**A. The negative definition sets the ceiling (biggest conceptual lever).**
Our AUC only measures "real outfit vs. same outfit with **one random same-category swap**." Two
consequences: (1) many negatives are **false negatives** — a random top-for-top swap is often
*equally compatible*, so those pairs are unrankable by construction and cap AUC well below 1.0;
(2) random swaps are **easy**, so the training signal is weak. Experiments:
  - **Hard-negative mining** (`make_negative_sample`): sample K candidate swaps, score all K with
    the *current* model, keep the **hardest** (highest-scoring) negative instead of using all K.
    Reuses the existing `--num_negatives` machinery. *Most likely to raise real discrimination.*
  - **Fashionability-constrained negatives:** corrupt only high-vote outfits, draw substitutes from
    *low*-vote outfits → fewer tie pairs, more meaningful "real > corrupted" gap.
  - Evaluate against a **fixed, cached** negative set (not re-sampled per seed) so AUC stops moving
    under noise and arms become comparable.

**B. Cheap training-recipe wins (~30 min total, full run is ~11 min).**
  - **L2-normalize the CLIP embeddings** before the input projection — CLIP vectors are meant to be
    unit-norm; if the store holds raw vectors this is a free, stabilising bump. *Check this first.*
  - **AdamW (`weight_decay≈1e-4`) + cosine LR schedule + warmup**; sweep `lr ∈ {1e-4, 3e-4, 5e-4}`.
    Current plain Adam at `1e-4`/10 epochs may be undertrained.
  - **More negatives per step** (`--num_negatives 8–16`) — nearly free given batching; BPR loves it.
  - **More epochs (20–30)** with the schedule; best-val checkpointing already guards overfitting
    (and the subset's val AUC was *still climbing* at epoch 10, §6.3).

**C. Architecture levers (moderate effort).**
  - **Richer readout:** concat **mean ‖ max** pool, or an attention-weighted readout, instead of
    plain mean-pool → outfit score.
  - **Use the unused `per_item` head for an explicit pairwise term:** score the outfit as
    `pooled MLP + Σ_{i<j} f(h_i, h_j)` (classic type-aware / pairwise compatibility). Aligns the
    model with "do these pieces go together" far better than a single pooled MLP.
  - 3 layers / wider hidden: expect little — graphs are tiny (2–8 fully-connected nodes), 2 layers
    already reach every node.

**D. Feature pruning (already partly done).** §6.1 shows the full 26-D attrs *hurt*; we already ship
`--attr_subset`. Worth an ablation: drop the 32-D colour histogram too (CLIP likely encodes colour).

**Recommended order:** (1) L2-normalize CLIP, (2) AdamW+cosine + lr sweep + more epochs,
(3) hard-negative mining, (4) mean‖max readout or pairwise term. Steps 1–2 are low-risk ~30 min;
step 3 is where the real jump should come, because it attacks the ceiling directly.

**⚠️ Re-baseline caveat:** a higher AUC on a *harder* negative set is **not** comparable to 0.80 on
the current easy set. If you change the negatives, re-run all arms/seeds so the comparison stays
honest.

---

## 7. Files & how to run

| File | Role |
|---|---|
| `01_segment_batch.py` | Bulk SegFormer+CLIP over Fashion144K → `outputs/<idx>/` (resumable, circuit breaker) |
| `02_outfit_dataset.py` | Loaders, colour histogram, attribute/edge encoders, `OutfitDataset` (+ store backing) |
| `03_train_gnn.py` | `CLIPOutfitGNN`, BPR training, AUC eval, ablation + pilot flags |
| `build_feature_store.py` | One-time memmap feature store (Tier 1) |
| `run_all.py` | Local runner (replaces the notebook): `segment` / `train` / `all` |

**Expected dataset layout** (override with `--data_root`):
```
Fashion144k_v1/
  photos.txt   photos/   split.mat   feat/relvotes.mat
```

**Fast path (recommended):**
```bash
# 1. Segment everything (resumable)
python run_all.py --stage segment

# 2. Build the feature store once (prefer a LOCAL, non-OneDrive --store_dir)
python build_feature_store.py \
    --output_root ../Fashion144k_v1/outputs \
    --store_dir   ../Fashion144k_v1/feature_store

# 3. Train — with attributes vs baseline
python 03_train_gnn.py --output_root ../Fashion144k_v1/outputs \
    --split_mat ../Fashion144k_v1/split.mat \
    --relvotes_mat ../Fashion144k_v1/feat/relvotes.mat \
    --checkpoint_dir ../Fashion144k_v1/gnn_checkpoints \
    --feature_store ../Fashion144k_v1/feature_store \
    --epochs 10 --batch_size 64 --num_negatives 4 --seed 42        # full attributes
#   ... add --attr_subset    for the chosen model (reliable attrs only, drop pattern/fabric/fit)
#   ... add --no_attributes  for the CLIP+colour baseline
#   ... add --limit_train 10000 for a quick pilot
```

**Key flags:** `--feature_store` (fast path), `--batch_size` (Tier 2), `--num_negatives` (Tier 3),
`--attr_subset` / `--no_attributes` (ablations), `--limit_train` (pilot cap), `--seed` (reproducible).

---

## 8. Code-quality pass
- **Tier-4 fix:** negative sampling O(pool) → O(1) rejection sampling.
- Removed dead `bpr_loss()`; removed legacy `--grad_accum` (arg + passthrough); removed a
  redundant local `defaultdict` import.
- Left intact deliberately: the `item_score` / `per_item` head (unused in the loss, but removing
  it changes the model `state_dict` and would break existing checkpoints).
- **`--attr_subset` ablation** added (`attr_subset_indices` in `02`, threaded through
  `build_graph`/`evaluate`/`train` in `03`, passthrough in `run_all`). Reuses the existing feature
  store — the 26-D attrs are column-masked at graph-build time, so **no re-segmentation or store
  rebuild** is needed to try attribute subsets.
