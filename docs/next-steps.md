# Next steps & backlog

_Updated 2026-10-07._

Current state: **M1 Perception: complete · M2 Harmony GNN: complete (AUC ≈ 0.80) · M3 Trend: complete · M4 Recommendation: designed, not built.**

---

## 1. Module 4: Recommendation engine

Design is complete: see [`module4-design.md`](module4-design.md) (plain language) and
[`module4-design-detailed.md`](module4-design-detailed.md) (engineering spec).

**Phase 0: Pinterest into the catalogue (do first)**
- [ ] Get the **full-resolution Pinterest dataset** (in progress: being supplied). The copies in
      `PinterestData/pinterest_dataset/images/` are 224×224 and squashed.
- [ ] Add a "store" mode to `Module3/pinterest_garment_segmentation.py` that saves everything M2 needs,
      then build `PinterestData/feature_store/` with the same builder as Fashion144k.
- [ ] Seeded train / val / test split for Pinterest.
- [ ] AUC gate: does M2 still work on Pinterest outfits? Fine-tune only if it drops by more than 0.03.
- [ ] Refit M3 on the new garments and settle the trend formula (decision D11).
- [ ] Build the unified candidate pool.

**Then:** candidate pool → swap groups → recommender → evaluation (incl. human study) → demo integration.

**Under consideration:** a vision-language-model judge as an extra evaluation signal.

---

## 2. Module 3 polish

- [x] **Dress clustering soft spot:** caused by k=15, not the data. `k_sweep.py` → dress k=8; ARI 0.56 → **0.86**.
- [x] **Temporal holdout:** growth is **not** forward-predictive at 3 mo (pooled ρ −0.10) and *negatively*
      predictive at 1/6 mo (mean reversion). Score reframed as descriptive.
- [x] **`date_ym` check:** partial Sept 2026 month excluded via `fit_per_category.py --end_month 2026-08`.
- [x] **Trend-model config:** no longer open. Decided as M4 decision D11 (re-test in Phase 0 against
      forward share; ties go to recency-only).
- [ ] **Housekeeping:** `garment_embeddings.f32` (~294 MB) is redundant once the `.npy` exists. Delete
      when no re-run is planned (gitignored either way).

---

## 3. Module 2: pushing AUC past ~0.80

From [`Module2/README.md`](../Module2/README.md) ("Raising AUC above ~0.80"), cheapest first:
- [ ] L2-normalise CLIP embeddings before the input projection.
- [ ] AdamW (`weight_decay≈1e-4`) + cosine LR + warmup; sweep `lr ∈ {1e-4, 3e-4, 5e-4}`; 20–30 epochs.
- [ ] **Hard-negative mining**, where the real jump should come.
- [ ] Richer readout (mean ‖ max) or an explicit pairwise term.
- [ ] **Note:** if the negative set changes, re-run all variants and seeds, because the AUCs aren't comparable otherwise.

---

## 4. Demo / presentation

- [x] Root `demo.py` covers M1 + M2 + M3.
- [ ] Add Module 4 to `demo.py` (designed: decision D10).
- [ ] Click through the UIs in a browser before presenting.

---

## 5. Repo hygiene

- [x] One root README linking to per-module READMEs; docs in `docs/`.
- [x] **No images in git.** `.gitignore` blocks all image formats; figures and sample photos stay local.
- [x] Images removed from git history (2026-10-07).
- [ ] Before any M3 re-run: `Module3/test_module1_sync.py` and `Module3/test_embedding_parity.py` must pass.
