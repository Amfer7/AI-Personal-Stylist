# Module 2A — Compatibility GNN (documentation)

This is the PDF's **Module 2: Compatibility Analysis** — it turns Module 1's per-garment
outputs into per-outfit graphs and trains a GNN to produce an outfit compatibility /
"Harmoniousness" score. Module2A is written as a **Colab pipeline over the Fashion144K
dataset**.

> **Status of this document:** section 1 describes Module2A **as originally received
> (baseline, before any of our changes)**. Section 2 is a placeholder where we will record
> our modifications later.

---

## 1. Baseline — as received

### 1.1 Files

| File | Role |
|---|---|
| `01_segment_batch.py` | Stage 1–2: bulk-run Module 1 (SegFormer + CLIP) over all Fashion144K images |
| `02_outfit_dataset.py` | Stage 3–5: build per-outfit graphs from `outputs/` + Fashion144K `.mat` files |
| `03_train_gnn.py` | Stage 6–7: define + train the CLIP-native Outfit GNN, evaluate by AUC |
| `fashion144k_sample1.ipynb` | Colab driver: mounts Drive, unzips data, clones repo, runs 01→02→03 |

### 1.2 Data flow

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

### 1.3 `01_segment_batch.py` — bulk perception

- Runs Module 1's `process_image_segformer` + `extract_folder_embeddings` over every
  Fashion144K image, **in the exact order of the `photos` list**, so outfit index `i` →
  `outputs/<i>/`. This alignment is what lets `split.mat` / `relvotes.mat` index directly
  into the segmented outputs.
- **Resumable:** completed indices are appended to a checkpoint file and skipped on restart;
  failures (missing file / no garments / exceptions) are logged separately. Built for Colab
  sessions that disconnect mid-run.
- Pilot via `--limit 2000`, then rerun without limit for the full ~144K.

### 1.4 `02_outfit_dataset.py` — graph construction

- **Loaders:** `load_split` (train/val/test ids from `split.mat`), `load_relvotes`
  (fashionability score, 1–10 scale, from `relvotes.mat`), `load_photos_list`.
- **`compute_colour_histogram`** — a **32-D HSV histogram** (16 hue + 8 sat + 8 val),
  computed fresh from each garment crop composited on white (same pixels CLIP sees).
  Replaces the dataset's precomputed `col_cco.mat`.
- **`CATEGORY_EDGE_WEIGHTS`** — full pairwise domain weight table over
  `top/bottom/outerwear/dress/shoe/bag/accessory` (e.g. top↔bottom = 1.0, dress↔shoe = 0.9,
  default 0.3, same-category 0.2). Fills the gap in Module 1's handoff snippet, which only
  special-cased top↔bottom.
- **`build_category_pool`** — indexes every garment by broad category (cached to pickle) for
  BPR negative sampling.
- **`OutfitDataset`** (`torch.utils.data.Dataset`) — one item = one outfit graph:
  - keeps only outfits with a `metadata.json` and **≥2 garments**;
  - node features = **`[512-D CLIP embedding ‖ 32-D colour histogram] = 544-D`**;
  - fully-connected edges weighted by category pair;
  - target = the outfit's fashionability score.

### 1.5 `03_train_gnn.py` — model + training

- **`WeightedGNNLayer`** — weighted-mean message passing (`index_add_` aggregation
  normalized by incident edge-weight sum) → `Linear` → `LayerNorm` → `ReLU`, with a residual
  (`x + agg`).
- **`CLIPOutfitGNN`** — `544-D → Linear projection → 2× WeightedGNNLayer → ` per-item scores
  + **mean-pooled** outfit compatibility score. Deliberately **has no `nn.Embedding` / vocab
  lookup** — this is the stated fix for the old train/inference representation mismatch.
- **Training:** fashionability-weighted **BPR loss**.
  - Positive = the real outfit.
  - Negative = the same outfit with **one garment swapped** for a same-category item from a
    *different* outfit (`make_negative_sample`).
  - Adam, gradient accumulation (default 16), 5 epochs.
- **Evaluation:** AUC over positive-vs-corrupted score pairs — on the val split each epoch,
  and finally on the official `test` split.

### 1.6 `fashion144k_sample1.ipynb` — Colab driver

Mounts Google Drive → unzips `Fashion144k_v1.zip` → **clones the external repo
`github.com/bugsNburgers/AI-based-personal-stylist`** and imports its `real_image_pipeline/`
→ sets Drive paths → runs `01` (pilot, then full) → `02` (sanity check) → `03` (train).

### 1.7 Dataset used

**Fashion144K** (~144,169 real full-body outfit photos with user attribute labels and a
crowd-derived *fashionability* score; ships `split.mat` and `relvotes.mat`).

---

## 2. Our changes

### 2.1 Run locally on GPU — dropped Colab (done)

Machine has an RTX 3070 Ti (CUDA), so the Colab/Drive/external-repo apparatus was removed.

- **`01_segment_batch.py`** — no longer requires `--repo_dir`/an external repo. New
  `--module1_dir` defaults to the sibling **`../Module1`**, so it imports the *local*
  Module 1 pipeline (and therefore automatically produces the new per-garment `attributes`
  and `outfit_features.json` during segmentation). `--repo_dir` is kept as a legacy override.
  Usage examples updated to local paths.
- **`run_all.py`** (new) — local runner that **replaces `fashion144k_sample1.ipynb`**.
  Stages `segment` / `train` / `all`; defaults `--device cuda`; derives all paths from
  `--data_root` (default `../Fashion144k_v1`); fails early with a clear message if the
  dataset isn't present yet. Calls `01` and `03` as subprocesses.
- **`02_outfit_dataset.py`, `03_train_gnn.py`** — unchanged so far (already path-driven via
  args; `03` loads `02` by file path). No repointing needed.
- **`fashion144k_sample1.ipynb`** — **superseded** by `run_all.py`; kept for reference, safe
  to delete.

Expected local dataset layout (override with `--data_root`):
```
Fashion144k_v1/
  photos.txt            photos/            split.mat            feat/relvotes.mat
```

Run:
```
python run_all.py --stage all --limit 2000      # pilot
python run_all.py --stage segment               # full (resumable)
python run_all.py --stage train --epochs 5      # after segmentation
```

**Verified** (1-image synthetic dataset, GPU): `run_all --stage segment` ran the local
Module 1 pipeline end-to-end, wrote `outputs/0/` with `attributes` + `outfit_features.json`
+ embeddings + checkpoint; `02` then built a valid `(3, 544)` graph (512 CLIP + 32 colour)
with a fully-connected `(2, 6)` edge index.

### 2.2 Feed Module 1 attributes into the graph (done)

The per-garment `attributes` written by Module 1 are now consumed by the GNN, as **node
features** and as **learned edge features**.

**`02_outfit_dataset.py`**
- New `encode_node_attributes(attr)` → a **26-D** vector: one-hots for
  `silhouette`(4) / `pattern`(6) / `fabric`(6) / `fit`(3) + 7 scalars (formality/100,
  hue sin & cos, saturation, value, extent, normalised aspect ratio). Missing/unknown
  fields become zeros, so pre-attribute outputs still load.
- `build_node_from_meta(outfit_dir, g)` builds a per-node dict (embedding + colour hist +
  attr vector + raw hue/formality/area for edges).
- `edge_features(a, b)` → a **5-D** directed edge vector encoding the PDF's fashion-theory
  cues: `[category weight, hue separation, formality difference, volume contrast (area
  log-ratio), same-category flag]`.
- `load_outfit_nodes` now returns a **list of node dicts**; `build_graph(nodes,
  fashion_score)` emits `x` (**570-D** = 512 CLIP + 32 colour + 26 attrs) and `edge_attr`
  (`[E, 5]`) alongside the legacy scalar `edge_weight`.
- Module-level constants `NODE_ATTR_DIM = 26`, `EDGE_ATTR_DIM = 5`.

**`03_train_gnn.py`**
- `WeightedGNNLayer` → **`EdgeAwareGNNLayer`**: both the aggregation weight
  (`softplus(gate(edge_attr))`) and the message (`x[src] + msg_edge(edge_attr)`) are
  functions of the edge features. The old fixed category weight is now just one component
  of the learned edge gate, so the layer strictly generalises the original.
- `CLIPOutfitGNN` input widened to `512 + 32 + NODE_ATTR_DIM`; `forward(x, edge_index,
  edge_attr)`.
- `make_negative_sample` operates on node lists and swaps in a substitute garment's full
  node (embedding + colour + attributes), read from the substitute outfit's `metadata.json`.
- Train/eval loops pass `edge_attr` instead of the scalar weight.

**Verified** (GPU, 8 synthetic outfits): `02` emits `x=(N, 570)`, `edge_attr=(E, 5)`; `03`
trains for 2 epochs, computes val/test AUC, and saves checkpoints without error. (AUC values
on random synthetic data are meaningless — this was a functional/shape check; real numbers
come from the Fashion144K run.)

**Not yet used:** `outfit_features.json` (the outfit-level aggregates) is still not fed to
the model — the pairwise theory cues are recomputed per-edge instead, which is the form the
GNN needs. The aggregate file remains available for analysis/inspection.

### 2.3 Ablation flag (done)

`03_train_gnn.py --no_attributes` (also `run_all.py --no_attributes`) trains the **baseline**:
nodes = `[512 CLIP | 32 colour]` (544-D), edges = scalar category weight only (1-D) — i.e.
the pre-integration model. Without the flag, the full 570-D nodes + 5-D fashion-theory edges
are used. This lets us quantify whether the Module 1 attributes actually raise AUC. The final
line is tagged `with_attributes` / `no_attributes`. Verified both modes run on GPU.

Run the comparison:
```
python run_all.py --stage train --epochs 5                 # with attributes
python run_all.py --stage train --epochs 5 --no_attributes  # baseline
```

### 2.4 Dataset status

Full **Fashion144K** installed at `../Fashion144k_v1/` and verified:
144,169 entries in `photos.txt`; splits 86,501 train / 14,416 val / 43,250 test;
`relvotes` in [1, 10]; **0% missing files** in a random 5,000 sample.

### 2.5 Pilot result (2,000 outfits)

Segmentation: 2,000 outfits in **5.3 min** on the RTX 3070 Ti (6.3 img/s), 1 failure,
**1,874 usable** (≥2 garments). Garments/outfit peak at 3.

Ablation (10 epochs, `grad_accum=8`; intersection of the official splits with the first
2,000 indices → 1,127 train / 187 val / ~560 test):

| Model | Test AUC |
|---|---|
| With attributes (570-D nodes + 5-D fashion-theory edges) | **0.6503** |
| Baseline (`--no_attributes`, 544-D + scalar edges) | 0.6314 |

Attributes give **+0.019 test AUC** — a positive but noise-level signal at this scale
(single run, stochastic negatives, no seed, ~560 test outfits; `[FINAL]` uses the last
epoch, not best-val). **Not conclusive** — superseded by §2.6.

### 2.6 Seed-averaged ablation (5,000 outfits) — authoritative

Setup: 5,000 segmented; 2,848 usable train / 485 val; best-val checkpoint; full test set;
seeds 42/43/44; 10 epochs; `grad_accum=8`.

| Model | seed42 | seed43 | seed44 | mean | std |
|---|---|---|---|---|---|
| With attributes | 0.6678 | 0.6579 | 0.6568 | **0.6608** | 0.0049 |
| Baseline (`--no_attributes`) | 0.6834 | 0.6720 | 0.6655 | **0.6736** | 0.0074 |

**Delta = −0.0128 (attributes HURT).** The baseline wins on **all three seeds**, and the gap
(~0.013) is ~2× the run-to-run std — a small but **consistent, real** effect. The 2k pilot's
+0.019 was noise (last-epoch, single run).

**Interpretation:** at pilot scale the explicit attributes don't add information the CLIP
embedding lacks, and hurt slightly. Likely causes: (a) noisy attributes (esp. fabric/fit —
CLIP zero-shot is weak there), (b) redundancy — CLIP already encodes colour/pattern/formality
visually, (c) the attribute model has more parameters and overfits 2,848 train outfits. This
does **not** waste the Module 1 attribute work — those attributes are still valuable for the
recommendation module (weak-link/explainability) and trend analysis; they just don't improve
the compatibility AUC here.

**Open caveats:** 2,848 train is small — the richer features *may* pay off with more data
(86k full train); and a **subset** (reliable colour/silhouette only, dropping noisy
fabric/fit/pattern) might help even though the full set hurts. Both are testable if desired.

### 2.7 Performance optimisations (see `plan.md`)

Implemented so the GNN can be retrained many times cheaply. All result-preserving except
batching (verified for parity). Attribute path stays fully functional throughout.

- **Tier 1 — feature store** (`build_feature_store.py`, `--feature_store`): one-time memmap of
  `[512 CLIP | 32 colour | 26 attr]` + scalars; training reads it instead of reopening PNGs /
  recomputing colour histograms. Verified **bit-identical** graphs and **exact** CPU AUC match
  (disk == store). ~1 GB for 144k, loaded via `mmap_mode='r'` (evictable, never bricks RAM).
- **Tier 2 — GPU minibatching** (`--batch_size`): `collate` packs B outfits into one
  disconnected graph; segment-mean readout. Batched forward == stacked single forwards
  (parity test). **~20× faster: a 10-epoch 5k run went ~300s → 14s**, AUC unchanged.
- **Tier 3 — extras**: `--num_negatives` (K negatives/positive; K=4 nudged test AUC
  0.6568→0.6665), `--val_eval_max`. DataLoader workers **obviated** (no I/O bottleneck left).
- **Tier 0 — segmentation** (Module 1): batched per-outfit CLIP embedding (kept). **fp16
  autocast was reverted** after it caused a dtype incident (float32 image vs float16 cached
  text features → `float != Half`), which failed every image; CLIP is already fp16 on CUDA so
  autocast added no benefit. Added a **consecutive-failure circuit breaker** to `01`
  (`--max_consecutive_failures`, default 25) so a systemic error can't silently mark the whole
  dataset "done" again.

Typical fast command:
```
python build_feature_store.py --output_root ../Fashion144k_v1/outputs --store_dir ../Fashion144k_v1/feature_store
python run_all.py --stage train --epochs 10 --batch_size 64 --feature_store ../Fashion144k_v1/feature_store
```
