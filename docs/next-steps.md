# Next steps & backlog

_Updated 2026-10-09._

Current state: **M1 Perception: complete · M2 Harmony GNN: complete (AUC ≈ 0.80) · M3 Trend: complete · M4 Recommendation: built and evaluated on Fashion144k; demo integration next.**

---

## 1. Module 4: Recommendation engine

Design is complete: see [`module4-design.md`](module4-design.md) (plain language) and
[`module4-design-detailed.md`](module4-design-detailed.md) (engineering spec).

**Phase 0: Pinterest into the catalogue.** Code done and debugged on a 2,000-photo pilot of the
224×224 copies (2026-10-09). Full run done 2026-10-10 (steps 1–4 below; step 5 wired up, full
evaluation pending a calibration decision). Pilot numbers (224×224, 2,000 photos) were AUC 0.731 vs
0.816 and backtest weight 0.6; the full run confirmed both.
- [x] Store mode in `Module3/pinterest_garment_segmentation.py`, the same Module 1 path as Fashion144k
      (shared `segment_photos()` loop); feature store with areas in the 98,304-px frame; seeded 60/10/30
      split with identical files kept together; parity guard against Fashion144k store rows.
- [x] AUC gate (`Module2/auc_gate.py`) and D11 backtest (`Module3/forward_share_backtest.py`).
- [x] Module 4 builds and recommends from both stores (settings entry in the runbook below).
- [x] **Full-resolution dataset.** The raw scrape in `PinterestData/pinterest_data/raw/` (43 GB) holds
      the original of every pin. `PinterestData/build_fullres_metadata.py` writes
      `pinterest_data/metadata_fullres.csv`: all 61,118 rows / 55,781 pins, each verified by MD5. About
      half the files are named `pinterest_<pin>_<n>.<ext>` (an earlier version of the script missed
      those and matched only 30,290 pins).
- [x] `PinterestData/build_resized.py` scales every original to ~98,304 px with its aspect ratio kept
      (the Fashion144k frame; 2:3 portraits come out exactly 256×384) into `pinterest_data/images_98k/`
      and writes `pinterest_data/metadata_98k.csv`. Same resolution as the GNN's training photos, ~2×
      faster store mode, ~0.1 MB of output per photo. Store mode no longer writes `_vis.jpg` overlays.
- [x] Steps 1–2 on `PinterestData/pinterest_data/metadata_98k.csv` (2026-10-10): 55,781 photos, 560 with
      no garments, 154,663 feature-store rows.
- [x] Step 3, AUC gate: Fashion144k test **0.8155** (41,713 outfits), Pinterest test **0.7272** (14,560
      outfits), drop 0.088 > 0.03: **failed**. Same as the 224×224 pilot (0.731), so resolution was
      not the cause. Fine-tuning **deferred** (owner decision 2026-10-10, see below).
- [x] Step 4 (2026-10-10): refit → `Module3/per_category_trend_model_store.pkl` (current `.pkl`
      untouched). Stability ARI: accessory 0.888, bottom 0.759, dress 0.955, top **0.611** (moderate; was
      0.759 on the old manifest). Backtest: D11 keeps growth weight **0.6** (+0.098 over recency-only, CI
      +0.036 to +0.162), but every weight has **negative** mean ρ with forward share (0.6: −0.155), so the
      score does not forecast; high-scoring clusters tend to lose share (as in the temporal holdout).
      Describe it as "current popularity", never as a forecast.
- [x] Step 5 setup (2026-10-10): `m4_settings.json` lists both stores and points `trend_model` at the
      refit `.pkl`. Cache rebuilt: 634,742 pool rows (480,079 + 154,663), 100% trend-scored, harmony
      reference over 56,273 test outfits (σ_logit 3.119, σ_trend 26.58). Old cache kept in
      `Module4/cache/pool_f144k_only_backup/`.
      - Tests: four `real` tests assumed pool row = Fashion144k store row; they now read features
        through a cross-store `pool_features` fixture (`conftest.py`) and use each outfit's own source.
        73/73 pass.
      - `eval_recovery.py --source pinterest` evaluates Pinterest test outfits (candidates still come
        from both stores); results go to `pinterest_*` files with their own `report_done.json`.
      - Smoke run, dev set, 100 outfits each, **not final**: M4 hit rate Fashion144k 0.66 (best baseline
        0.41, chance 0.32), Pinterest 0.61 (best baseline CLIP coherence 0.45, chance 0.35).
- [x] **Recalibrated** (owner decision 2026-10-10): the frozen settings were tuned with the old trend
      model on a Fashion144k-only pool, and the harmony reference changed (σ_logit 3.046 → 3.119,
      σ_trend 28.37 → 26.58, new trend clusters). Re-ran dev `--freeze`, then the report set with
      `--force` (the system changed, not a re-roll) and Pinterest's report set for the first time.
      9 Oct results kept in `Module4/cache/eval_2026-10-09_f144k_only/`, old settings in
      `Module4/cache/m4_settings_before_recalib.json`.
      - Settings: rule `slot_percentile`, w_t 0.2 and ε 0.2σ **unchanged**; τ 0.333 → **0.392**.
      - Report sets, 2,000 outfits each, candidates from both stores:

        | | Fashion144k (9 Oct, F144k pool) | Fashion144k (both stores) | Pinterest (both stores) |
        |---|---|---|---|
        | M4 hit rate | 0.718 (0.698–0.737) | **0.722** (0.702–0.741) | **0.693** (0.673–0.713) |
        | Best baseline (CLIP coherence) | 0.386 | 0.422 | 0.427 |
        | Random / chance | 0.304 / 0.314 | 0.304 / 0.314 | 0.378 / 0.360 |
        | Original's median % rank | 4.8% | 4.8% | 5.4% |
        | False alarms on real outfits | 21.1% | 15.9% | 17.1% |
        | Detection on corrupted outfits | 75.0% | 69.8% | 70.3% |

      - Pinterest is ~3 points below Fashion144k on hit rate despite the 0.09 AUC gap; both are far
        above every baseline. Weakest Pinterest groups: upper 0.658, lower 0.659; best: bag 0.778.
      - The higher τ trades detection for fewer false alarms (about 1 in 6 real outfits now gets
        "swap", down from 1 in 5).
- [ ] Open: re-check `t_dup` (0.99) on the mixed pool with the visual sheets
      (`python Module4/calibrate_dup.py`, needs an owner look), as the implementation plan's Pinterest
      section asks. The recalibration above re-tuned w_t, τ, ε and the rule only.
- [x] **Catalogue crop quality** (found 2026-10-10 while rendering `demos/recommendation_demo.png`;
      fixed 2026-11-19 with a demo-pool crop gate, see the end of this item and
      [`Module4/README.md`](../Module4/README.md) "Crop quality gate"). Some pool garments are mislabelled by Module 1 (Pinterest outfit 1336's `top_0.png` is a
      pair of Converse sneakers that SegFormer labelled "top"; M4 offered it as a "refresh" for a
      sweater; traced 2026-10-10: the code passes the label through correctly; also a sweatshirt
      labelled "jacket" and denim shorts labelled "pants"), some Fashion144k outfits are
      fragmentary crops, and Pinterest crops can carry the pin's overlaid caption text. These reach the
      live demo's suggestions too. Options: a minimum crop area / fill ratio, a CLIP check that the crop
      matches its label, or excluding text-heavy pins.
      - Audit 2026-11-19 (1,500 random demo-pool crops per catalogue, through the slide filter
        `crop_readable`: area ≥ 4,000 px, main piece ≥ 85%, fill ≥ 35%): pass rate Fashion144k **38%**,
        Pinterest **47%**. The demo pool (`rep_demo`) only requires that the crop file exists.
        - Fashion144k fails mostly on size (55% under 4,000 px; median crop 98×71 px).
        - Pinterest fails more often on "split" crops (13% vs 7%): collages, flat-lays and product shots
          next to a model, where several garments merge into one mask, plus caption text.
        - Pinterest crops that pass can still carry caption text or holes where an arm or hair hid the
          garment.
        - One global threshold would empty whole types (shoes pass 3/294 and 10/196, belts and
          sunglasses 0) because they are small, and shoes come as pairs (two pieces). Any filter has to
          be per type.
      - Fix 2026-11-19: `Module4/crop_quality.py` measures all 634,742 crops (six shape metrics from the
        alpha mask, two CLIP scores from the stored embeddings: own-group probability and "text"
        probability) into `cache/crop_metrics.npz`; `crop_gate` in `m4_settings.json` sets per-type bars,
        applied when the pool loads (no rebuild to retune). Bars set from contact sheets
        (`Module4/calibrate_crops.py`), one row per rejection reason:
        - Holes check off for shoes (strappy sandals, touching boots); label check off for accessories
          (CLIP reads dark boots, side-on bags, knit scarves as other groups; with one prompt per group it
          also rejected cardigans, so groups now have several prompts each).
        - Text check: at > 0.5 it catches caption overlays almost exclusively on clothing; on shoes it
          misfires on 0.4% of plain small shoes (accepted).
        - Keeps 341,355 of 633,810 demo candidates; smallest group Pinterest sunglasses 221.
        - 200 random test outfits, before → after: shown suggestions failing the gate 34.8% → 0%
          (Fashion144k inputs 31.4%, Pinterest inputs 47.6%); same verdict 97%; top suggestion harmony
          percentile 87.3 → 87.0. `rep_eval` unchanged (tested), so the evaluation results stand.
        - Still possible: low-resolution crops that pass (a 1,100-px shoe pair) and small stray pieces.
        - Added 2026-11-19 (owner found an unreadable "sweater" in the demo: Fashion144k 80390, where
          SegFormer cut the crocheted blanket she sits on together with a sweater mostly hidden by hair;
          one solid piece, so every shape test passed, fill 0.39 and cover 0.21 just over the bars):
          CLIP `junk_p` (fabric scraps, messy collage, blurry fragment, patterned background, hair,
          shadow, skin vs the garment prompts), clothing only, ≤ 0.4. That crop scores 0.60 (worst 1%).
          Removes ~1.6% of clothing; on contact sheets ≥ 0.4 is almost all shapeless blobs, 0.3–0.4 is
          mixed. Rejected alternatives: stricter fill/cover (0.45 / 0.3 drops 9%, incl. good hats and
          leggings); a "blanket" prompt (knit sweaters read as blankets). Demo candidates now 320,744.
          Owner look at `Module4/cache/crop_calibration/*.png` still to do.
- [ ] Open: the AUC gate only tests single-source outfits; mixed outfits (a Pinterest piece swapped
      into a Fashion144k outfit, or the reverse) are untested apart from the M4 evaluation.

Runbook (repo root; `<store>` = `PinterestData/store`):
1. `python Module3/pinterest_garment_segmentation.py --mode store --metadata_csv <metadata.csv> --output_dir <store>`
   (~10 photos/s on the 224×224 copies; full resolution will be slower; resumable).
2. `python Module2/build_feature_store.py --output_root <store>/outputs --store_dir <store>/feature_store --photo_sizes <store>/photos.csv --area_frame_px 98304`
3. `python Module2/auc_gate.py --checkpoint Fashion144k_v1/ck_attr_subset_s42/clip_outfit_gnn_best.pt --fashion_store Fashion144k_v1/feature_store --fashion_split Fashion144k_v1/split.mat --pinterest_store <store>/feature_store --pinterest_split <store>/split.json`.
   If it fails, build the fine-tune path (merge both stores for `03_train_gnn.py`, warm start, adopt
   only if Pinterest AUC rises and Fashion144k drops by no more than 0.01).
4. Refit M3 on `<store>/garment_manifest.csv` + `garment_embeddings.npy`, re-run the stability test, then
   `python Module3/forward_share_backtest.py --manifest_csv ... --embeddings ...` and freeze the D11 weight.
5. Add `{"name": "pinterest", "store_dir": "PinterestData/store/feature_store", "output_root": "PinterestData/store/outputs", "split": "PinterestData/store/split.json"}`
   to `Module4/m4_settings.json` → `stores`; point `trend_model` at the refit model;
   `python Module4/build_cache.py`; re-run the evaluation for both sources.

**M4 core on Fashion144k (build step 2): done** (2026-10-09). See [`Module4/README.md`](../Module4/README.md).
- [x] Candidate pool, swap groups + co-occurrence filter, dense GNN scorer, harmony reference,
      selection rules, verdicts, dedupe, CLI.
- [x] Performance gate: `image2.jpg` vs the full demo pool, **0.16 s warm, 1.92 GiB peak** (target ≤ 5 s, ≤ 6 GB).
- [x] Dedupe rebuilt at outfit level (reposts), after the CLIP-only rule turned out to merge distinct tiny crops.

**Evaluation (build step 3): done** (2026-10-09). `Module4/eval_recovery.py`, spec §6.
- [x] Dev / report sets (2,000 + 2,000), group-consistent corruption, original recovery, three baselines.
- [x] Calibrated and frozen: rule `slot_percentile`, w_t = 0.2, τ = 0.333, ε = 0.2σ.
- [x] Report set run once: hit rate **71.8%** (chance 31.4%, best baseline 38.6%); original in the
      top **4.8%** of candidates.

**Next:** human study. Phase 0, the both-source evaluation and the demo integration (D10) are done
(2026-10-10).
- [ ] **GNN fine-tune: deferred** (owner decision 2026-10-10). Known limitation: the GNN is trained on
      2010s Fashion144k; AUC 0.727 on Pinterest vs 0.816, but M4 hit rate only 0.693 vs 0.722, so the
      most fine-tuning could add to M4 is ~3 points. Data is ready; picking it up needs a two-store
      loader for `03_train_gnn.py` (split.mat + split.json), a stand-in weight for Pinterest (no
      `relvotes` ratings), warm start, and validation on both sources; then rebuild the cache,
      recalibrate and re-run both report sets. Adopt only if Pinterest AUC rises and Fashion144k
      drops ≤ 0.01.
- [ ] Open question for later: about 1 in 6 real outfits gets a "swap" verdict (was 1 in 5 before the
      recalibration). The human study will show how many of those are real improvements.

**Under consideration:** a vision-language-model judge as an extra evaluation signal.

**Scope and display catalogue** (2026-11-19):
- [x] **Indian wear out of scope for now** (owner decision 2026-11-19, after the demo offered a saree to
      a man). The 44 Indian queries of the Pinterest scrape are in `demo_exclude_queries`; their photos
      never become demo suggestions (`PinterestData/build_outfit_queries.py` → `store/outfit_queries.csv`).
      Removes 14,762 of 55,781 Pinterest photos, 17,110 demo candidates (324,253 left; Pinterest dresses
      11,220 → 5,003). Demo only: evaluation, trend model and GNN still include Indian wear.
      - Before: 21% of Pinterest demo candidates came from Indian queries, 6% from menswear queries.
      - A CLIP "saree / lehenga / kurta / salwar / anarkali" detector is unusable as a filter: it scores
        > 0.5 on 24% of Fashion144k (Western street style) and 16% of the remaining Pinterest garments,
        and the highest-scoring remaining Pinterest garments are almost all Western.
- [ ] Open: **no gender signal anywhere.** M2 scores whether pieces go together, not whether they suit
      the wearer, and both catalogues are mostly womenswear (6% of Pinterest demo candidates come from
      menswear queries). Plan: a Menswear / Womenswear / Any choice in the demo, with a gender tag per
      catalogue garment (queries for Pinterest, labels for a product catalogue).
- [ ] Open: **a clean, display-only product catalogue.** Studio product photos would read better than
      street-photo crops, and `rep_demo` / `rep_eval` are already separate, so the evaluation is not
      affected. Candidates: H&M Personalized Fashion Recommendations (Kaggle 2022, ~105k articles,
      Ladieswear/Menswear groups, 2018–20 purchases; check the competition terms), Fashion Product Images
      (Myntra, Kaggle, ~44k, gender + usage labels; drop usage "Ethnic" while Indian wear is out of
      scope), Polyvore Outfits (365k items in 68k outfits, no gender, academic access on request).
      Needs background removal instead of SegFormer, a type mapping, and a check that GNN harmony and
      M3 trend matching behave on studio shots.

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
- [x] Module 4 in `demo.py` (D10, 2026-10-10): verdict, suggestion gallery, also-worth-improving,
      per-piece table, live trend-weight slider (no GNN call, tested), advanced options, "Try this
      swap" board (`Module4/outfit_board.py`, shared with the human study), M2 and M4 on one GNN and
      one harmony reference (tested equal), `--vram-gb`. Slide figures:
      `demos/recommendation_tradeoff.png`, `demos/recommendation_demo.png`. Slide examples filtered
      for readable crops (owner decision). Manual browser check still to do.
      Build plans for the core and the demo were deleted 2026-10-13 once done; facts that existed only there are in `Module4/README.md`.
- [ ] Click through the UIs in a browser before presenting.

---

## 5. Repo hygiene

- [x] One root README linking to per-module READMEs; docs in `docs/`.
- [x] **No images in git.** `.gitignore` blocks all image formats; figures and sample photos stay local.
- [x] Images removed from git history (2026-10-07).
- [ ] Before any M3 re-run: `Module3/test_module1_sync.py` and `Module3/test_embedding_parity.py` must pass.
