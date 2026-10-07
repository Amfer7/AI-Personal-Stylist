# How the harmony GNN was trained (Module 2)

A short account of how Module 2's model learns, what went wrong along the way, and what the
results mean. For the full engineering reference, see [`Module2/README.md`](../Module2/README.md).

---

## In one paragraph

Each outfit becomes a small **graph**: garments are **nodes**, and every pair of garments is joined by
an **edge** that describes how they relate (colour gap, formality gap, size contrast). A **graph neural
network (GNN)** reads the graph and outputs one number: how well the outfit goes together. It learns
by comparison. A real outfit should score **higher** than the same outfit with one garment swapped for
a random one. On Fashion144k's official test split it gets this right about **80% of the time**
(AUC ≈ 0.80).

## How training works

| Step | Plain terms | Technical detail |
|---|---|---|
| **Input** | Each garment is described by its look, colour and attributes | Node = 570 numbers: 512-D CLIP embedding + 32-D HSV colour histogram + 26-D attributes ([`02_outfit_dataset.py`](../Module2/02_outfit_dataset.py)) |
| **Relationships** | Each pair of garments gets a short description of how they relate | Edge = 5 numbers: category-pair weight, hue separation, formality difference, volume contrast, same-category flag |
| **Model** | Garments "talk" to each other twice, then the outfit gets one score | `CLIPOutfitGNN`: linear projection → 2 × `EdgeAwareGNNLayer` (message passing where edges shape both the weight and the content of messages) → mean-pool → score ([`03_train_gnn.py`](../Module2/03_train_gnn.py)) |
| **Learning signal** | "The real outfit should beat a spoiled copy" | **BPR loss** (a ranking loss) on real vs one-garment-swapped pairs (**negative sampling**), weighted by each outfit's crowd fashionability vote |
| **Judging** | How often does the real outfit win? | **AUC** on held-out real-vs-corrupted pairs; 0.5 = guessing, 1.0 = perfect. The checkpoint with the best validation AUC is kept |

### The loss, exactly

```
loss = mean over pairs of  −w · log σ( score(real) − score(corrupted) ),   w = fashionability / 10
```
It only asks the real outfit to outscore its corrupted twin. Weighting by the crowd vote makes
well-liked outfits count more, which is how the one label Fashion144k provides gets used. The optimiser
is Adam (lr 1e-4), and each epoch keeps the checkpoint with the best validation AUC
(`clip_outfit_gnn_best.pt`). Test-time corruptions are re-seeded so every model faces the same pairs.

### One training step

1. Take a batch of outfits. For each, build the real graph and K corrupted graphs (one swap each,
   `--num_negatives`, e.g. 4).
2. Pack everything into one batched graph and score it in a single GPU pass.
3. Compute the weighted BPR loss over each (real, corrupted) pair.
4. Backpropagate, take an Adam step, and repeat. A full 10-epoch run takes about 11 minutes.

## Results

Three versions were trained on the full training split (86,501 outfits), each with three random
seeds (42, 43, 44), and tested on the official 43,250-outfit test split:

| Version | What the garment description includes | Mean test AUC |
|---|---|---|
| **Attribute subset** (chosen) | CLIP + colour + the *reliable* attributes (silhouette, colour, formality, shape) | **0.8010** |
| Baseline | CLIP + colour only | 0.7923 |
| Full attributes | everything, including noisy pattern/fabric/fit guesses | 0.7860 |

The **attribute subset** is 11 numbers per garment: a 4-way silhouette one-hot (fitted / boxy /
A-line / tapered) plus 7 continuous values: formality score (÷100), hue as sin and cos (so red at 0° and
359° stay close), saturation, value (brightness), extent (how much of its bounding box the garment fills),
and aspect ratio (clamped to min(aspect/4, 1)).

**These differences are not statistically significant** with three seeds; it's effectively a tie.
The subset was chosen because it has the highest mean *and* keeps the interpretable attributes that
Modules 3–4 rely on, at no measured accuracy cost. One caveat: the subset beats full attributes on every
seed, but the baseline doesn't. On seed 44, full attributes scored 0.7885 vs the baseline's 0.7677.

## Problems we hit, and how they were fixed

| Problem | Fix | Effect |
|---|---|---|
| Every training step re-opened garment images and recomputed colours | A one-time **feature store**: all garment numbers precomputed into one memory-mapped file ([`build_feature_store.py`](../Module2/build_feature_store.py)) | Training never touches images again; identical results |
| The GPU processed one outfit at a time | **Minibatching**: many outfits packed into one disconnected graph, with vectorised scoring and loss | 10-epoch pilot: ~300 s → 14 s, AUC unchanged |
| At full scale, building each spoiled outfit scanned a ~100k-garment list | **Rejection sampling**: pick a random garment, retry on the rare collision | GPU went from 3% busy and ~27 min/epoch to ~1 min/epoch (~27× faster) |
| Mixed-precision (fp16) speed-up broke CLIP (float/half mismatch) and failed every image | Restricted fp16 to SegFormer only; added a **circuit breaker** that stops a batch run after repeated failures ([`01_segment_batch.py`](../Module2/01_segment_batch.py)) | No more silent mass failures |
| Training and inference used different garment representations (a vocabulary lookup) | The model now consumes real CLIP vectors end to end | Consistent behaviour at inference |
| Noisy attributes and seed-to-seed variation made comparisons unreliable | Compared three variants × three seeds, best-validation checkpoints, official test split | Honest, reproducible numbers |

## Where errors come from

- **Perception, not the GNN.** The documented mistakes happen upstream in Module 1: an earlier detector
  found "pants" on chests in mirror selfies, and CLIP once labelled linen pants as "corduroy" (confidence
  0.34). See [`Module1/README.md`](../Module1/README.md) ("Why SegFormer", "Known limits"). No case of
  the GNN itself producing a garment or an
  explanation is documented.
- **Some "spoiled" outfits aren't actually worse.** A random same-category swap can produce an equally
  good outfit, but training treats every swap as worse. This caps the achievable AUC and hasn't been measured.
- **The score is a ranking, not a probability.** The demo shows it as a **percentile** relative to real
  test outfits ([`demo_score.py`](../Module2/demo_score.py)).

## Ideas for going beyond 0.80

Ranked cheapest first in [`Module2/README.md`](../Module2/README.md) ("Raising AUC above ~0.80"): normalise CLIP vectors,
a better optimiser schedule, **hard-negative mining** (the most promising), and a richer readout.
If the definition of a "spoiled" outfit changes, all variants must be re-run, because a higher AUC on a
harder test isn't comparable to 0.80.
