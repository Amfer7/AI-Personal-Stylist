# Module 2: Harmony (outfit compatibility GNN)

**In one line:** scores how well the pieces of an outfit go together, as a **harmony percentile
(0–100)** relative to real outfits.

```
Module 1 output ──► outfit graph ──► GNN ──► harmony score
(garments)         (nodes + edges)
```

**Result:** on Fashion144k's official test split, the model ranks a real outfit above the same outfit
with one garment swapped **~80% of the time (AUC ≈ 0.80)**.

How it was trained, in plain terms: [`docs/gnn-training.md`](../docs/gnn-training.md). This README
is the full engineering reference.

---

## How it works

**Data: Fashion144k.** 144,169 real full-body outfit photos, each with a crowd **fashionability**
vote (1–10) and official train / validation / test splits (`split.mat`, `relvotes.mat`).

```
Fashion144k photos ──01_segment_batch──► outputs/<idx>/{metadata.json, *.png, *.npy}
                                                 │
                   build_feature_store ──► feature_store/  (one memory-mapped file)
                                                 │
   split.mat / relvotes.mat ──02_outfit_dataset──► one graph per outfit
                                                 │
                                       03_train_gnn (BPR + AUC)
                                                 ▼
                                    clip_outfit_gnn_best.pt
```

1. **Segment everything** (`01_segment_batch.py`): runs Module 1 over all photos. It's resumable,
   and a **circuit breaker** stops the run if many images fail in a row.
2. **Cache features once** (`build_feature_store.py`): all 480,079 garments' numbers go into one
   **memory-mapped** file, so training never re-opens images.
3. **Build a graph per outfit** (`02_outfit_dataset.py`). Outfits with ≥ 2 garments are kept.
   - **node** = one garment = 570 numbers: 512-D CLIP embedding + 32-D HSV colour histogram
     (16 hue + 8 saturation + 8 value bins, computed from the cut-out on white) + 26-D attributes
     (one-hots for silhouette 4 / pattern 6 / fabric 6 / fit 3, plus 7 scalars: formality, hue sin/cos,
     saturation, value, extent, aspect). Missing attributes become zeros.
   - **edge** = one directed pair of garments = 5 numbers: category-pair weight (from
     `CATEGORY_EDGE_WEIGHTS`, top↔bottom 1.0 … default 0.3), **hue separation**, **formality
     difference**, **volume contrast** (log area ratio), same-category flag.
4. **Train the GNN** (`03_train_gnn.py`). `CLIPOutfitGNN` is a linear projection → 2 ×
   `EdgeAwareGNNLayer` → mean-pool → one score. In each layer, edges shape both the message weight
   (`softplus(gate(edge_attr))`) and its content (`x[src] + msg_edge(edge_attr)`), so this is
   **message passing**. It consumes real CLIP vectors end to end, with no vocabulary lookup.
   - **Loss:** fashionability-weighted **BPR**. A real outfit should outscore a copy with one garment
     swapped for a same-category garment from another outfit (**negative sampling**). Each pair is
     weighted by `relvotes / 10`.
   - **Optimiser and selection:** Adam (lr 1e-4); the checkpoint with the best validation **AUC** is kept.
5. **Score a photo** (`demo_score.py`). The raw score is a ranking number (all large negatives, so a
   sigmoid collapses to ~0), so it's shown as a **percentile** against real test outfits. Two modes:
   - **Aggregate:** scores many test outfits real-vs-corrupted, then prints a table, the overall win
     rate, and a figure (the "it works" slide).
   - **Single photo** (`--score_dir <segmented outfit> --photo <photo>`): scores one outfit plus a
     one-swap corrupted copy, and renders photo → garments → harmony /100 → "drops to X if one item
     is swapped". The `PhotoScorer` class loads the model and reference once so `demo.py` can score
     new photos instantly.

The shipped model uses the **attribute subset** (`--attr_subset`): the 11 reliable attribute
dimensions only (silhouette one-hot + the 7 scalars, node = 555-D), dropping CLIP's noisy
pattern/fabric/fit guesses.

## Results

**Full scale:** 86,501-outfit train split, official 43,250-outfit test split, best-validation
checkpoint, three seeds.

| Variant | Node | Edges | Seed 42 | Seed 43 | Seed 44 | **Mean** | Std |
|---|---|---|---|---|---|---|---|
| **Attribute subset** (`--attr_subset`, chosen) | 555 | 5-D | 0.8155 | 0.7917 | 0.7958 | **0.8010** | 0.0104 |
| Baseline (`--no_attributes`: CLIP + colour) | 544 | 1-D | 0.7951 | 0.8140 | 0.7677 | **0.7923** | 0.0190 |
| Full attributes (26-D) | 570 | 5-D | 0.7780 | 0.7915 | 0.7885 | **0.7860** | 0.0058 |

- **A statistical tie.** Seed-to-seed noise (±0.02) is larger than the gaps between variants, so no
  pairwise difference is significant with three seeds. Subset vs baseline even flips on seed 43.
- **The one robust effect:** full attributes are never the best on any seed. The noisy zero-shot
  pattern/fabric/fit one-hots never help.
- **Why it barely moves:** CLIP already encodes colour, pattern and formality visually, so explicit
  attributes are largely redundant.
- **Why the subset ships:** it has the top mean, beats full attributes on every seed, and keeps the
  interpretable attributes and fashion-theory edges that Modules 3–4 use, at no measured accuracy cost.
- **Lesson:** an earlier single-seed (42) result claimed "the baseline beats attributes by 0.017". It
  was a seed artefact: seed 42 was full attributes' worst draw and the subset's best.

**Run health**
- Segmentation: 144,169 / 144,169 photos, 16 failures (~0.01%), ~8–9 images/s.
- Feature store: 480,079 garment rows × 570-D, ~1.04 GB, row counts cross-checked against the index.
- Training: 83,352 usable train / 13,925 usable val outfits. Validation ≈ test AUC (0.7784 vs 0.7780),
  so **no overfitting**. A 10-epoch run takes ≈ 11 min.

## Engineering history

Module 2 started as a **Google Colab notebook**: it mounted Drive, unzipped the dataset, cloned an
external repository for the perception code, and ran the three scripts. It's now a **local GPU
pipeline**. `01_segment_batch.py` imports the sibling `../Module1`, and `run_all.py` replaces the
notebook.

Other changes along the way:
- **Module 1 attributes into the graph:** added the 26-D node attributes and 5-D edges, and upgraded
  the layer to `EdgeAwareGNNLayer` (nodes widened 544 → 570-D). `outfit_features.json` isn't used,
  because the GNN needs per-edge values, which are recomputed instead.
- **Windows fixes:** Fashion144k has non-ASCII filenames, and the default Windows console encoding
  crashed mid-run, so stdout/stderr and failure logs are forced to UTF-8. SegFormer runs under fp16
  autocast; CLIP does not (see below).
- **Clean-up:** removed a dead `bpr_loss()` and the legacy `--grad_accum`. The unused `per_item` head
  is kept, because removing it would break existing checkpoints.

### Speed-ups

The goal was to make retraining cheap enough to run many experiments. Every change keeps results
identical, except graph batching, which was parity-tested.

| Tier | Change | Effect |
|---|---|---|
| 0 | Batched CLIP embedding per outfit; **circuit breaker** (`--max_consecutive_failures`, default 25) | A systemic error can't silently mark the whole dataset "done" |
| 1 | **Feature store** (`--feature_store`): one memory-mapped file; colour histograms computed once | Training never opens a PNG; bit-identical graphs |
| 2 | **GPU minibatching** (`--batch_size`): many outfits packed into one disconnected graph; vectorised BPR | ~20× faster (10-epoch 5k pilot: ~300 s → 14 s), AUC unchanged |
| 3 | **Multiple negatives** per outfit (`--num_negatives 4`) | Stronger signal (5k pilot AUC 0.657 → 0.667) |
| 4 | **Negative sampling O(pool) → O(1)** | ~27× faster at full scale |

**Tier 4 only showed up at full scale.** An epoch took ~27 min with the GPU 3% busy. The cause:
building each negative copied the whole category pool (~100k garments) to exclude the current
outfit. The fix is **rejection sampling**: pick at random and retry on the rare collision. That gives
the same draw distribution at ~1 min per epoch.

**fp16 incident:** a blanket fp16 autocast was reverted. It made image features float32 while cached
CLIP text features stayed float16, so every image failed. Only the SegFormer forward pass uses fp16 now.

## Raising AUC above ~0.80

~0.80 is close to the ceiling of the task *as currently defined*. A random same-category swap is
often just as good as the original (a **false negative**), and random swaps are easy. Cheapest first:

1. **L2-normalise** CLIP embeddings before the input projection.
2. **AdamW** (`weight_decay≈1e-4`) + cosine LR with warmup; sweep `lr ∈ {1e-4, 3e-4, 5e-4}`;
   20–30 epochs. The subset's validation AUC was still rising at epoch 10. Also try `--num_negatives 8–16`.
3. **Hard-negative mining:** score K candidate swaps with the current model and train on the hardest.
   This is where the real jump should come from. A related idea is fashionability-constrained negatives:
   corrupt high-vote outfits using garments from low-vote outfits.
4. **Richer readout:** mean ‖ max pooling, attention pooling, or an explicit pairwise term using
   the unused `per_item` head.
5. Smaller ideas: drop the 32-D colour histogram (CLIP probably covers colour); add supervised
   attributes (e.g. Fashionpedia) from a different backbone for genuinely new signal.

**Caveat:** if the negatives change, re-run every variant and seed against a fixed, cached negative
set. A higher AUC on a harder test isn't comparable to 0.80.

## Run it

Expects `Fashion144k_v1/` at the repo root (`photos/`, `photos.txt`, `split.mat`, `feat/relvotes.mat`).

```bash
python run_all.py --stage segment                       # 1. segment (resumable, hours on GPU)
python build_feature_store.py --output_root ../Fashion144k_v1/outputs \
                              --store_dir   ../Fashion144k_v1/feature_store   # 2. cache
python 03_train_gnn.py --output_root ../Fashion144k_v1/outputs \
    --split_mat ../Fashion144k_v1/split.mat --relvotes_mat ../Fashion144k_v1/feat/relvotes.mat \
    --checkpoint_dir ../Fashion144k_v1/gnn_checkpoints --feature_store ../Fashion144k_v1/feature_store \
    --epochs 10 --batch_size 64 --num_negatives 4 --seed 42 --attr_subset          # 3. train
python demo_score.py                                    # 4. real-vs-corrupted figure + win rate
```
Or use the repo-root web demo: `python demo.py`, then choose **Module 2**.

**Key flags:** `--attr_subset` / `--no_attributes` (variants), `--feature_store`, `--batch_size`,
`--num_negatives`, `--limit_train N` (cap train outfits for a quick pilot; val/test stay full),
`--seed`. Keep the feature store on a local (non-OneDrive) disk.

## Files

| File | Role |
|---|---|
| `01_segment_batch.py` | Bulk Module 1 over Fashion144k (resumable, circuit breaker) |
| `02_outfit_dataset.py` | Graph building: node/edge features, colour histograms, dataset class |
| `03_train_gnn.py` | The model (`CLIPOutfitGNN`), BPR training, AUC evaluation, variant flags |
| `build_feature_store.py` | One-time memory-mapped feature cache |
| `demo_score.py` | Scores a photo as a percentile; real-vs-corrupted demo figure |
| `run_all.py` | One-command runner (`segment` / `train` / `all`) |

## Known limits
- Scores the **whole outfit** only. Its per-garment head isn't trained, so it can't say which piece is
  weak on its own (Module 4 finds that by trying swaps).
- A "spoiled" outfit isn't always worse, which caps the achievable AUC.
- The three attribute variants are a statistical tie at three seeds.
