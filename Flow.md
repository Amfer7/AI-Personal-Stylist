# Flow — How Module 1 and Module 2 Work

A plain-language walkthrough of the whole pipeline, with the important technical terms
explained as they come up. Read top to bottom and you'll understand what every file does and
how a photo becomes an outfit-compatibility score.

---

## The big picture

**Goal:** take a real photo of a person, understand each piece of clothing they're wearing,
and judge how well the outfit "goes together."

The system has two halves:

- **Module 1 — Perception.** Looks at a photo and turns each garment into numbers a computer
  can reason about (a "fingerprint" per garment, plus described attributes like colour and
  formality).
- **Module 2A — Compatibility.** Takes those numbers, treats the outfit as a **graph**
  (garments = dots, relationships = lines), and trains a model to score how harmonious the
  outfit is.

```
Photo ──▶ [Module 1: Perception] ──▶ per-garment numbers ──▶ [Module 2A: GNN] ──▶ compatibility score
```

**Two key AI models used throughout:**
- **SegFormer** — a neural network that looks at an image and labels *every pixel* ("this pixel
  is shirt, this is pants, this is skin/background"). This is called **semantic segmentation**.
- **CLIP** — a model from OpenAI trained on 400M image+caption pairs. It can turn any image into
  a list of 512 numbers (an **embedding**) that captures what the image "looks like," and it can
  compare an image to text prompts ("a striped shirt") without any extra training
  (**zero-shot**).

---

## Module 1 — Perception (turning a photo into garment data)

Think of Module 1 as an assembly line with three stations. The main script that runs the line
is `fashion_segmenter.py` (function `process_image_segformer`).

### Stage 1 — Find and cut out each garment
**Files:** `fashion_segmenter.py`, `bg_removal.py`

- **SegFormer** labels every pixel of the photo into regions: upper-clothes, pants, skirt,
  dress, shoes, bag, hat, etc. (skin, hair, and background are recognised and thrown away).
- Each clothing region is cut out into its own image with a transparent background (a
  **transparent RGBA PNG** — the "alpha" channel marks which pixels are fabric). This means a
  garment is isolated with *no room, no skin* — just the clothing.
- `bg_removal.py` does this cut-out using the SegFormer mask, with a classic computer-vision
  fallback (**GrabCut**) for small accessories the model doesn't cover.

*Why it matters:* clean cut-outs mean later steps see the clothing only, not the background.

### Stage 2 — Name each garment precisely
**File:** `fashion_segmenter.py` (function `classify_crop_clip`)

SegFormer only says "upper-clothes." To get the specific type, we show the cut-out to **CLIP**
and ask which text prompt fits best — "a t-shirt", "a sweater", "a jacket" — and take the
highest match. This **zero-shot classification** turns "upper-clothes" into "t_shirt".

`category_mapping.py` is the dictionary that keeps category names tidy (e.g. "jeans" → "pants",
and it groups everything into **broad categories**: top / bottom / outerwear / dress / shoe /
bag / accessory). Broad categories matter later for the graph.

### Stage 3 — Describe and fingerprint each garment
**Files:** `attribute_analyzer.py`, `clip_extract_embeddings.py`

Two kinds of output per garment:

1. **The embedding (the fingerprint).** `clip_extract_embeddings.py` runs the cut-out through
   CLIP to get a **512-dimensional vector**, normalised to length 1 (**L2-normalised**). Two
   garments that look alike get similar vectors. This is the main numeric input to Module 2.

2. **The attributes (the description).** `attribute_analyzer.py` produces human-readable
   properties, in two ways:
   - **Measured from pixels (deterministic, no AI guessing):**
     - *Colour* — the dominant colour of the fabric pixels, as RGB and **HSV** (Hue =
       which colour, Saturation = how vivid, Value = how bright), plus a name like "red".
       Computed with **k-means** clustering over the fabric pixels only.
     - *Silhouette* — the garment's shape (fitted / boxy / A-line / tapered), worked out from
       the mask's geometry (height-to-width ratio, how much it fills its box, whether it widens
       toward the hem).
   - **Guessed by CLIP zero-shot (uses AI, noisier):**
     - *Pattern* (solid / striped / floral / …), *Fabric* (denim / cotton / knit / …),
       *Fit* (slim / regular / oversized), *Formality* (casual → formal), plus a continuous
       **formality score** from 0–100.

`attribute_analyzer.py` also computes **outfit-level "fashion-theory" features** by comparing
garments to each other:
- **Hue distance** — how far apart the colours are (colour coordination).
- **Volume balance** — relative sizes of top vs bottom.
- **Formality coherence** — whether all pieces sit at a similar formality level.

### What Module 1 saves (the "output contract")
For each photo it writes a folder `outputs/<id>/` containing:
```
<garment>_<n>.png      # the transparent cut-out
<garment>_<n>.npy      # the 512-number CLIP embedding
metadata.json          # per-garment: category, bbox, confidence, and the attributes block
outfit_features.json   # the outfit-level hue/volume/formality numbers
<id>_vis.jpg           # an annotated picture for eyeballing the result
```
This exact layout is the "contract" Module 2 relies on.

### Supporting files
- `run_pipeline.py` — command-line entry point to run one image or a folder.
- `yolo_detect_and_crop.py` — an *alternative* detector (YOLO) kept in the repo, **not** the
  default (we standardised on SegFormer).
- `test_pipeline.py` — automated checks (embedding is 512-D and unit-length, colour ignores
  background, silhouette classification, etc.).
- `changes.md` — log of what was added to Module 1 (the attributes work).

---

## Module 2A — Compatibility (scoring how well an outfit works)

Module 2A trains a model on **Fashion144K** — a public dataset of ~144,000 real outfit photos,
each with a crowd-sourced **fashionability score (1–10)** and official train/validation/test
splits. It has three scripts (numbered in run order) plus helpers.

### The dataset it learns from
- `split.mat` — which outfits are for training / validation / testing.
- `relvotes.mat` — the 1–10 fashionability score per outfit (the learning signal).

### Step 1 — Run Module 1 over the whole dataset
**File:** `01_segment_batch.py`

This feeds all 144K photos through Module 1's assembly line, producing an `outputs/<id>/`
folder for each. It is **resumable** (keeps a checkpoint of finished items, so it can be
stopped and restarted) and has a **circuit breaker** that halts the run if many images fail in
a row — so a bug can't silently ruin the whole dataset. This step is a one-time cost (hours on
a GPU).

### Step 2 — Turn each outfit into a graph
**File:** `02_outfit_dataset.py`

A **graph** is dots (**nodes**) connected by lines (**edges**). Here:
- **Each garment is a node.** Its features are the numbers from Module 1 stitched together:
  `[512 CLIP embedding | 32-number colour histogram | 26 attribute numbers] = 570 numbers`.
  (The 26 attributes are the silhouette/pattern/fabric/fit/formality, encoded as numbers.)
- **Every pair of garments is connected by an edge.** Each edge carries 5 numbers describing
  the *relationship* — the "fashion-theory" cues: a category-pairing weight (top↔bottom matters
  more than top↔hat), **hue separation**, **formality difference**, **volume contrast**, and a
  same-category flag.

This file also:
- Builds the graph on demand for each outfit (`build_graph`).
- Loads features from the **feature store** (see Optimisations) instead of re-reading images.
- `collate` — packs many outfit-graphs into one big batch so the GPU can process them together.

### Step 3 — The model and how it learns
**File:** `03_train_gnn.py`

**The model: `CLIPOutfitGNN`** — a **Graph Neural Network (GNN)**. A GNN lets each node "talk"
to its neighbours and update itself based on them; doing this twice lets each garment's
representation account for the rest of the outfit. This is called **message passing**. Our
`EdgeAwareGNNLayer` makes the messages depend on the *edge* features too — so a big hue clash or
formality mismatch actually changes what the model learns. After two rounds, the node vectors
are **pooled** (averaged) into one outfit vector, which becomes a single **compatibility
score**.

**How it's trained — learning by comparison (BPR):**
We rarely know "this outfit scores 0.7." But we can create easy comparisons:
- Take a **real** outfit (positive example).
- Make a **corrupted** version by swapping one garment for a random same-category item from a
  different outfit (**negative sampling**). This is almost certainly worse.
- Train the model so the real outfit scores **higher** than the corrupted one. This is
  **BPR loss** (Bayesian Personalized Ranking — a ranking objective), and it's weighted by the
  outfit's fashionability so nicer outfits count more.

**How it's judged — AUC:**
On held-out test outfits we again make real-vs-corrupted pairs and check how often the model
ranks the real one higher. That fraction is the **AUC** (Area Under the Curve): 0.5 = random
guessing, 1.0 = perfect. We keep the model from the epoch with the best validation AUC
(**best-val checkpoint**) and report the final number on the test split.

**The `--no_attributes` switch (ablation):**
An **ablation** means "remove a part and see if the score drops." This flag trains the model
*without* the Module 1 attributes (just CLIP + colour) so we can measure whether the attributes
actually help. (On the 5k pilot they slightly hurt — CLIP already captures most of that info,
and the guessed fabric/fit are noisy. Decision on keeping them is separate.)

### Helper files
- `build_feature_store.py` — the one-time precompute (see below).
- `run_all.py` — a simple local runner that ties Steps 1–3 together with sensible defaults
  (replaces the old Colab notebook `fashion144k_sample1.ipynb`).
- `MODULE2A.md` — detailed documentation of Module 2A and every change made to it.
- `plan.md` — the performance-optimisation plan and its status.

---

## How the two modules connect

Module 2 never re-opens photos. It reads only Module 1's `outputs/<id>/` files:
- the `.npy` embeddings and the `attributes` in `metadata.json` become **node features**,
- the attributes are also compared pairwise to become **edge features**,
- `broad_category` decides the edge-pairing weights.

So the "contract" from Module 1 (Stage 3) is literally the input format for Module 2 (Step 2).

---

## Optimisations (why training is fast now)

Training the GNN is run many times while tuning it, so speed matters. Three changes made it
~20× faster (a full run went from minutes to ~14 seconds on the pilot), with identical results:

1. **Feature store** (`build_feature_store.py`) — all the garment numbers are precomputed once
   into a single **memory-mapped file** (`memmap`: a file the program reads like an array
   without loading it all into RAM). Training then never re-opens PNGs or recomputes colour.
2. **GPU minibatching** — instead of one outfit at a time, many outfits are packed into one big
   graph and processed together, so the GPU is actually kept busy.
3. **Reproducibility + quality knobs** — a fixed random **seed** (so runs are repeatable),
   best-val checkpointing, and optional **multiple negatives per positive** for a stronger
   learning signal.

(Segmentation itself — Module 1 over 144K images — is a separate one-time cost and runs at
~8 images/second on the GPU.)

---

## What changed from the original modules

This project started from earlier versions of both modules. Here's what we changed and why, in
plain terms.

### Module 1 — added the "describe each garment" stage
- **Originally:** Module 1 only cut out garments and made the 512-number CLIP embedding
  (the fingerprint). It did **not** describe them.
- **Now:** we added `attribute_analyzer.py`, which gives every garment a description block —
  colour and silhouette (measured directly from pixels) and pattern/fabric/fit/formality
  (guessed by CLIP zero-shot) — plus outfit-level "fashion-theory" numbers (hue distance,
  volume balance, formality coherence). These are saved into `metadata.json` and a new
  `outfit_features.json`.
- **Why:** the report/design calls for attribute understanding, not just a raw fingerprint.
- **Kept safe:** the old output format still works (attributes are *added*, nothing removed), so
  Module 2 didn't break. A short-lived attempt to speed up segmentation with **fp16** (half-
  precision maths) was **reverted** after it caused errors; we kept only the safe speedups.

### Module 2A — made it local, attribute-aware, and much faster
- **Originally:** it was a **Colab notebook** that cloned a *different* external repo, ran on
  Google Drive paths, and trained the GNN using only the CLIP embedding + a colour histogram,
  with hand-set edge weights. Training processed one outfit at a time and re-read image files on
  every step.
- **Now, the main changes:**
  1. **Runs locally.** Dropped Colab/Drive/external-repo cloning; it imports the local Module 1
     and runs on the local GPU. Added `run_all.py` as a simple one-command runner (replacing the
     old notebook).
  2. **Uses the new attributes.** Garment nodes now include the attribute numbers
     (`512 CLIP + 32 colour + 26 attributes = 570`), and the edges carry learned
     "fashion-theory" relationships (hue separation, formality difference, volume contrast)
     instead of a single fixed weight (`EdgeAwareGNNLayer`).
  3. **Ablation switch.** Added `--no_attributes` so we can measure whether the attributes
     actually help (they slightly hurt on the pilot — documented, decision pending).
  4. **Reproducible and honest scoring.** Added a fixed random **seed** and **best-val
     checkpoint** selection so the reported AUC is stable and comparable across runs.
  5. **Big speedups (identical results):** a one-time **feature store** (memmap) so training
     stops re-reading images, and **GPU minibatching** so many outfits train together — together
     ~20× faster. Plus optional **multiple negatives** for a stronger signal.
  6. **Safety fix.** The bulk-segmentation script (`01_segment_batch.py`) now has a
     **circuit breaker**: if many images fail in a row it stops, instead of silently marking the
     whole dataset "done" (a flaw that previously let a bug run away through thousands of images).

---

## Mini-glossary

- **Embedding** — a list of numbers that represents an image's content; similar images → similar
  numbers.
- **CLIP** — model that makes image embeddings and matches images to text prompts (zero-shot).
- **SegFormer** — model that labels every pixel (semantic segmentation).
- **Zero-shot** — classifying into categories the model was never specifically trained on, using
  text prompts.
- **RGBA / alpha** — an image with a transparency channel; alpha marks which pixels are the
  garment.
- **HSV** — Hue/Saturation/Value, a colour representation that's easy to reason about.
- **Graph / node / edge** — outfit as dots (garments) and lines (relationships).
- **GNN (Graph Neural Network)** — a model where nodes update themselves from their neighbours
  (message passing).
- **BPR loss** — a training objective that teaches "real outfit should rank above a corrupted
  one."
- **Negative sampling** — creating a deliberately-worse outfit to compare against.
- **AUC** — score for how well the model ranks good vs bad (0.5 random, 1.0 perfect).
- **Ablation** — removing a component to measure its contribution.
- **memmap** — a file accessed like an array without loading it fully into memory.
