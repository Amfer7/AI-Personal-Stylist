# Style-Invariant Garment Embedding: Roadmap

**Goal:** one garment embedding in which a street photo, a phone photo under ordinary lighting and a
studio product shot of the same garment land together, shared by the harmony GNN, the trend model and
the suggestion catalogue, so that photo style no longer tilts what Module 4 recommends.

**Why:** the GNN is trained on street photos (Fashion144k) and suggests H&M studio shots. CORAL
alignment (2026-11-21) closed most of the gap, but a classifier still tells H&M from street at 60%,
H&M candidates rank worse (stand-in rank 0.36 vs 0.28 on street) and swaps still pick shoes about
twice as often as with street suggestions (`docs/next-steps.md`). A product-only model would just move
the gap to the user's own photo. The fix is a learned bridge between the photo styles.

**Shape of the work:** four phases, each with its own detailed plan, each ending in a measured gate.
Phase 1 is written in full (`2026-11-21-phase1-evaluation-suite.md`). Phases 2–4 get their detailed
plans once Phase 1's baseline numbers exist and their datasets are on disk, because their code reads
data formats that must be checked against the real files first.

---

## Data to acquire (people, not code)

| Dataset | Why | How | Needed by |
|---|---|---|---|
| DeepFashion2 | Same garment photographed by a consumer and by a shop (`pair_id` + `style`), 13 clothing categories | Fill the access form on the DeepFashion2 GitHub page to receive the zip password; unzip to `DeepFashion2/` at the repo root (`train/`, `validation/`, each with `image/` and `annos/`) | Phase 1 Task 5, Phase 2 |
| DeepFashion Consumer-to-Shop benchmark | More consumer/shop pairs for clothing | Agreement form on the DeepFashion project page | Phase 2 (optional, more pairs) |
| Polyvore Outfits (disjoint split) | Outfits made of product images: compatibility training data in the catalogue's own style, and literature-comparable compatibility AUC / FITB | Download from the `fashion-compatibility` project (Vasileva et al., ECCV 2018) to `PolyvoreOutfits/` | Phase 3 |
| Phone-photo set | Real user photos under ordinary lighting, the input the demo actually sees | Capture protocol in Phase 1 Task 6; stored in `samples/phone_set/` (gitignored) | Phase 1 baseline, Phase 4 human study |

Check each download's size against free disk first (about 163 GB free on C: on 2026-11-21). Add every
new data folder to `.gitignore` before unzipping.

**Known coverage gap:** DeepFashion2 and Consumer-to-Shop are clothing only. Shoes, bags, hats, belts,
scarves and sunglasses get no real consumer/shop pairs; Phase 2 covers them with photo-style
augmentation of product images (Street2Shop has accessory pairs, but most of its image URLs are dead;
try it, do not depend on it).

---

## Key design decisions

1. **Two embeddings, one job each.** The fine-tuned encoder drifts away from CLIP's text tower, which
   would break every zero-shot use of CLIP (Module 1 attributes, Module 4 crop gate). So:
   - `features.npy[:, :512]` becomes the style-invariant embedding (GNN input, trend clusters,
     duplicate detection, candidate similarity). The row layout (570 float32) is unchanged.
   - A new `clip_zs.npy` (garments × 512, float32) in each feature store keeps the original CLIP
     embedding for zero-shot consumers (`crop_quality.py`). Module 1's attribute analyzer runs on the
     photo with its own CLIP and is unaffected.
   - `encoder.json` in each feature store records which encoder wrote columns 0–511; Module 4's cache
     fingerprint includes it.
2. **Keep 512 dimensions** so the feature store, the GNN input projection and every downstream shape
   stay as they are.
3. **Measure before changing.** Phase 1 records today's numbers on every metric the later gates use.
4. **Trend from H&M transactions** (31.8M dated purchases, unused) is a separate plan, not part of
   this roadmap; it does not depend on the encoder.

---

## Phase 1: Evaluation suite (detailed plan written)

Builds the measurements every later gate uses and records today's baseline:
- suggestion type-mix metric with a pass/fail threshold (`Module4/type_mix.py`)
- DeepFashion2 pair loader, cut-out renderer, retrieval / pair-cosine / domain-classifier metrics,
  and an invariance evaluation CLI with a pluggable encoder (`Module1/style_encoder/`)
- phone-photo set protocol and a batch runner (`Module4/phone_set.py`)
- baseline run of all of them with today's CLIP + CORAL system

**Gate:** all metrics run end to end and their baseline values are logged in `docs/next-steps.md`.

## Phase 2: Train the style-invariant encoder

Task outline (detailed plan after Phase 1):
1. `Module1/style_encoder/photo_style_aug.py`: phone-photo augmentation — white balance 2700–7500 K,
   exposure and gamma, soft shadow gradients, JPEG quality 40–90, Gaussian and motion blur, sensor
   noise, perspective up to 8%, partial occlusion. Seeded; tested for determinism and output size.
2. `Module1/style_encoder/pair_dataset.py`: DeepFashion2 (and Consumer-to-Shop if obtained) pair
   cut-outs from Phase 1's loader; unpaired product cut-outs (H&M, Polyvore) paired with their own
   augmented copy; a batch sampler that groups each batch by category so in-batch negatives are
   same-category (hard).
3. `Module1/style_encoder/model.py`: CLIP ViT-B/32 visual tower initialised from OpenAI weights,
   first 6 of 12 blocks frozen, 512-d output, L2-normalised.
4. `Module1/style_encoder/train.py`: symmetric InfoNCE on consumer/shop pairs + InfoNCE on
   product/augmented pairs + a distillation term (L2 to frozen CLIP on Fashion144k street crops) that
   keeps the garment semantics the GNN relies on. fp16 autocast, gradient checkpointing, sized for the
   8 GB GPU; three seeds; checkpoints and a metrics JSON in `Module1/style_encoder/checkpoints/`.
5. `StyleEncoder` with the same `encode(images) -> (n, 512)` interface as Phase 1's `ClipEncoder`.

**Gate** (DeepFashion2 validation, held out; H&M check sample):
- Recall@1 and Recall@10 consumer→shop above the CLIP baseline by more than the seed-to-seed spread.
- Consumer-vs-shop domain classifier accuracy lower than the baseline's.
- H&M-vs-street domain classifier ≤ 0.55 **without** CORAL.
- Type linear-probe accuracy on Fashion144k crops no more than 1 point under CLIP's.

## Phase 3: Re-embed the stores, refit the trend model, retrain the GNN

Task outline:
1. `build_feature_store.py --encoder style`: writes the style embedding into columns 0–511,
   `clip_zs.npy` and `encoder.json`; re-embed Fashion144k (480,079), Pinterest (154,663) and H&M
   (55,862) crops directly into the stores. `crop_quality.py` reads `clip_zs.npy`.
2. Module 3: refit the per-category trend model in the new space; `k_sweep.py`, cluster-stability ARI
   and the forward-share backtest compared with the current model.
3. `PolyvoreOutfits` adapter: Polyvore items → Module 1-style folders (cut-out, style embedding,
   attributes from Module 1's analyzer, type mapping from Polyvore's semantic category) → a feature
   store.
4. Module 2 training on Fashion144k + Polyvore with source-balanced batches; hard negatives (swap one
   item for a same-type item ranked 10–100 nearest to it in the style space), mixed 50/50 with
   random negatives; a fixed, cached validation negative set so every variant is scored on the same
   negatives; three seeds.

**Gate:** AUC on the fixed hard-negative validation set at least the current model's on the same set
(re-scored); Polyvore compatibility AUC and FITB reported; in the domain check, H&M and street
stand-in ranks within seed noise of each other.

## Phase 4: Module 4 integration and human study

Task outline:
1. Point `m4_settings.json` at the new checkpoint and stores; H&M alignment method `none` unless the
   domain check still shows a gap; rebuild the cache.
2. Recalibrate every setting marked calibrated (`w_t`, `rule`, `tau`, `eps`, `t_dup`) with the
   existing evaluation (`eval_recovery.py`); the status block records it.
3. Type-mix gate (`type_mix.py`) passes: no type's swap share above 1.5× its street share.
4. Blind A/B human study on the phone-photo set: today's system vs the new one, raters pick the better
   outfit (boards from `outfit_board.py`).

**Gate:** the type-mix gate passes, and raters prefer the new system on the phone-photo set with a
95% confidence interval above 50%.

---

## Risks

| Risk | Effect | Mitigation |
|---|---|---|
| DeepFashion2 access is slow to arrive | Phase 2 cannot start | Request it first; Phase 1 Tasks 1–4 and 6 do not need it |
| No accessory pairs | Shoes and bags rely on augmentation only | Augmentation of product images; report accessory metrics separately |
| Fine-tuning erodes semantics the GNN uses | GNN AUC drops | Distillation term; linear-probe gate in Phase 2; GNN gate in Phase 3 |
| Re-embedding changes every downstream number | Calibrated settings go stale | Phase 4 recalibrates everything; Phase 1 baseline kept for comparison |
| 8 GB VRAM | Small contrastive batches weaken InfoNCE | Freeze half the tower, fp16, gradient checkpointing, category-grouped batches so each negative is informative |
