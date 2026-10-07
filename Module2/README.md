# Module 2: Harmony (outfit compatibility GNN)

**In one line:** scores how well the pieces of an outfit go together, as a **harmony percentile
(0–100)** relative to real outfits.

```
Module 1 output ──► outfit graph ──► GNN ──► harmony score
(garments)         (nodes + edges)
```

**Result:** on Fashion144k's official test split, the model ranks a real outfit above the same outfit
with one garment swapped **~80% of the time (AUC ≈ 0.80)**.

- How it was trained, the problems hit, and what the results mean: [`docs/gnn-training.md`](../docs/gnn-training.md)
- The full engineering log (every change, optimisation and ablation): [`miniREADME.md`](miniREADME.md)

---

## How it works

**Data: Fashion144k.** ~144,000 real outfit photos, each with a crowd **fashionability** vote (1–10)
and official train / validation / test splits (`split.mat`, `relvotes.mat`).

1. **Segment everything** (`01_segment_batch.py`): runs Module 1 over all 144k photos. It's resumable,
   and a **circuit breaker** stops the run if many images fail in a row.
2. **Cache features once** (`build_feature_store.py`): all 480,079 garments' numbers go into one
   **memory-mapped** file, so training never re-opens images.
3. **Build a graph per outfit** (`02_outfit_dataset.py`):
   - **node** = one garment = 570 numbers (512-D CLIP embedding + 32-D colour histogram + 26-D attributes)
   - **edge** = one pair of garments = 5 numbers (category-pair weight, **hue separation**,
     **formality difference**, **volume contrast**, same-category flag)
4. **Train the GNN** (`03_train_gnn.py`): `CLIPOutfitGNN` passes messages between garments twice
   (**message passing**, with edges shaping the messages), averages them, and outputs one score.
   It learns with **BPR loss**: a real outfit should outscore a copy with one garment swapped
   (**negative sampling**). Model selection uses validation **AUC**.
5. **Score a photo** (`demo_score.py`): the raw score is a ranking number (all large negatives, so a
   sigmoid collapses to ~0), so it's shown as a **percentile** against real test outfits. Two modes:
   - **Aggregate:** scores many test outfits real-vs-corrupted, then prints a table, the overall win rate,
     and a figure (the "it works" slide).
   - **Single photo** (`--score_dir <segmented outfit> --photo <photo>`): scores one outfit plus a
     one-swap corrupted copy, and renders photo → garments → harmony /100 → "drops to X if one item is
     swapped". The `PhotoScorer` class loads the model and reference once so `demo.py` can score new
     photos instantly.

The shipped model uses the **attribute subset**: the reliable attributes only, dropping CLIP's noisy
pattern/fabric/fit guesses.

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

## Files

| File | Role |
|---|---|
| `01_segment_batch.py` | Bulk Module 1 over Fashion144k (resumable, circuit breaker) |
| `02_outfit_dataset.py` | Graph building: node/edge features, colour histograms, dataset class |
| `03_train_gnn.py` | The model (`CLIPOutfitGNN`), BPR training, AUC evaluation, ablation flags |
| `build_feature_store.py` | One-time memory-mapped feature cache |
| `demo_score.py` | Scores a photo as a percentile; real-vs-corrupted demo figure |
| `run_all.py` | One-command runner (`segment` / `train` / `all`) |
| `miniREADME.md` | Full engineering log and results |

## Known limits
- Scores the **whole outfit** only. Its per-garment head isn't trained, so it can't say which piece is
  weak on its own (Module 4 finds that by trying swaps).
- A "spoiled" outfit isn't always worse, which caps the achievable AUC.
- The three attribute variants are a statistical tie at three seeds.
