# Next steps & backlog

_Updated 2026-10-09._

Current state: **M1 Perception: complete · M2 Harmony GNN: complete (AUC ≈ 0.80) · M3 Trend: complete · M4 Recommendation: built and evaluated on Fashion144k; demo integration next.**

---

## 0. Phase 3 Review 3 (2026-11-24)

Template: `PPT-Template_Phase3- Review3.pptx` (blank). Checked against the repo 2026-11-22:
- Have: implementation for all four modules, results with baselines (M2 AUC 0.7469 at Review 2 →
  0.8155; M4 hit rate 72.2% vs best baseline 42.2%, chance 31.4%), tests (M1 9, M3 7, M4 77 quick +
  31 `real`, all pass; M1's only pass when run from `Module1/`; M2 has no pytest suite, it is gated by
  `auc_gate.py` and the evaluation instead), slide figures in `demos/`.
- Missing from the repo: Review 2 panel remarks, project report, IEEE paper draft, target venues.
- Phase 3 Review 2 deck (`PW26_DS_04_Sem7_review2.pdf`, local only) already presented all four
  modules (M4 on the 634,742-garment street pool, pseudocode, 72% / 69% hit rates, ablation),
  two other-domain slides (virtual try-on, interior design) and 12 references. Its demo pool claim
  is now out of date: suggestions come from H&M (55,862), street stores only score and evaluate.
- New since Review 2: crop quality gate, Indian wear out of demo suggestions, H&M catalogue + domain
  check + CORAL alignment, refresh cap, style-invariant roadmap.
- [x] **Report Chapter 7 (System Design) drafted 2026-11-23** from the code:
  `docs/report/chapter7-system-design.md`. The brief (`capstone sem7 report.docx`) lists only
  Fashion144k + Pinterest; the chapter records the third store (H&M, demo-only). Out-of-date docs
  found: root README names `per_category_trend_model.pkl` (the app loads `..._store.pkl`);
  `module4-design.md` §1 says Pinterest ~144k garments and a two-store catalogue (store has 154,663
  rows, demo uses H&M); `Module1/src/run_pipeline.py` docstring describes the YOLO-World/U2Net path and
  `real_image_pipeline/` paths (default is SegFormer). Chapters 5–6 not in the repo, so not cross-checked.
- [x] **Report Chapter 8 (Results and Discussion) drafted 2026-11-23:** `docs/report/results.md`,
  from the module READMEs, this file and the result files in `Module4/cache/` (eval reports, human
  study, catalogue QA). Chapter number assumed to follow Chapter 7. Test run that day: Module 1 23,
  Module 3 7, Module 4 130 (incl. `real`), all pass. Not in the chapter: the "M2 AUC 0.7469 at
  Review 2" figure above (no source found in the repo); Kit 2 reported as not done.
- [x] **Review 1 suggestion: non-human subjects (done 2026-11-22).** 16 Unsplash photos in
  `samples/nonhuman/` (13 dressed pets, 1 undressed dog, 2 person + pet; sources in `manifest.csv`).
  - Before: every photo produced an outfit. The undressed golden retriever's face became a
    "shirt_blouse" with harmony 100/100; a cat in a suit got 6 garments (pants, shoe, sunglasses on
    paws and fur); median harmony of the 13 dressed pets 89.7, all with suggestions.
  - Signals compared: whole-image CLIP P(person) separates the animal photos (≤ 0.23) but puts
    real people in elaborate garments below 0.3 (Pinterest saree / lehenga / sherwani; 29% of 300
    Pinterest photos < 0.3, Fashion144k minimum 0.34): rejected. SegFormer body-part pixel share
    overlaps: rejected. COCO YOLO11m-seg "animal and no person": all 13 animal-only photos, 0 of 600
    real photos (300 Fashion144k, 300 Pinterest): adopted.
  - `Module1/src/subject_check.py` in `run_single_image()` (live path only, stores unchanged): animal
    and no person → stop with "No person found, only an animal"; person + animal → animal-only
    pixels (grown 1% of the longer side) excluded before garment parsing. CPU (~0.3 s), since the
    installed torchvision is a CPU build and its NMS rejects CUDA tensors.
  - After: 13/13 animal-only photos stop. The 2 pet photos showing only a person's legs give "no
    garments" (SegFormer had put every garment on the pet; legs-only people come out as background).
    The 2 person + pet photos are analysed with the pet removed (man with cat: harmony 31 → 41).
  - Also fixed: a photo with 0 garments crashed Module 2 in the demo with a raw traceback; the demo
    now stops with "No garments found". Tests: Module 1 19, Module 3 7, Module 4 112 pass.
  - Limits: COCO species names are unreliable ("bear" for a cat in a suit, "sheep" for a greyhound),
    so the message says "an animal"; two people in one photo are still parsed as one outfit.
- Changes since Phase 2 (sem 6): YOLOS / DeepFashion2 segmentation → SegFormer; trend score is
  descriptive, not a forecast (backtest ρ < 0).
- Demo risk: never clicked through in a browser with H&M suggestions; no gender filter.
- End-to-end check 2026-11-22 (`run_recommend.py --photo samples/image2.jpg`, H&M pool): runs, 30 s
  cold, 41,866 what-if outfits in 0.69 s, verdict refresh. **Bug:** suggestion 2 (pants → H&M
  299668015) has harmony 79 → 73, ΔT_z 2.3e-8 and overall gain −0.16. It passes the refresh test
  ΔT > 0 only through float32 rounding (82.86000061 vs 82.86: same trend cluster); ΔH −0.199σ is
  just inside ε 0.2σ, and the one-per-piece cap pulls it in ahead of better shoe options.
  - [x] Fixed 2026-11-22: ΔT is computed against the worn trend rounded to float32 (same cluster →
    exactly 0), and refresh candidates need combined gain > 0. Two tests, 110 Module 4 tests pass
    (incl. `real`). 400 test outfits (200 per street source, H&M suggestions), before → after:
    verdicts identical (refresh 330, swap 68, keep 2), suggestions with gain ≤ 0 8 → 0 (of 1,190),
    8 outfits' suggestions changed. Harmony-down suggestions 62 → 54: the rest are refresh picks
    within ε whose trend gain outweighs the dip (by design). Evaluation results unaffected (they
    score swap verdicts). On `image2.jpg` all three refresh picks are now shoes, since no other
    piece has a trendier option with a positive gain: the known shoe lean.
- Not for this review (future work): style-invariant encoder, GNN fine-tune, human study.
- [x] **Module 1 accuracy, weak labels (2026-11-22).** `Module1/scripts/eval_weak_labels.py`: the poster's
  garment words in each Fashion144k file name as weak labels, recall only (a lower bound: titles
  name hidden or out-of-frame items and omit others). All 144,169 photos (outputs reused, nothing
  re-segmented); results in `Fashion144k_v1/eval_weak_labels/`. Group recall: upper 97.6% (n 68,774),
  lower 94.8% (47,203), bag 83.8%, hat 79.5%, dress 76.6%, shoe 75.0% (48,042), scarf 62.5%, belt
  19.4%, sunglasses 5.6% (4,861). Exact types: skirt 86%, jacket 74%, pants 75%, sweater 48%,
  leggings 42%, shirt/blouse 30%, t-shirt 22%, top 14%, **shorts 4.4%** (9,571).
  - Shorts are almost always called pants (5,969 of 9,149 misses); Module 1 outputs only 844 shorts
    of 480,079 Fashion144k garments. CLIP zero-shot subtype weakness; same swap group, so Module 4
    still searches the right replacements.
  - Upper-body subtype is weak because one region per body part and CLIP leans to "jacket"
    (shirt/blouse titles: 7,751 misses read as jacket).
  - Small accessories at 400 px: belts 19%, sunglasses 6% (often on the head, read as hair).
  - Tights / stockings (no type) come out as pants or leggings in 87% of 3,855 photos.
  - Tests: 4 in `Module1/tests/test_eval_weak_labels.py`. Full run reads 144k metadata files (~10 min).
- **Team testing kits (2026-11-22), `docs/review3-kits.md`.** Four human tasks, owners to fill in,
  results due 23 Nov 18:00:
  1. Human study pilot: `Module4/scripts/human_study.py build` made 40 blind A/B questions from 20 street
     test outfits with a "swap" verdict (10 Fashion144k, 10 Pinterest): original vs Module 4's top
     swap, and original vs a random replacement from the same candidate list (same H&M catalogue, so
     a liking for studio photos cancels out). First build: tiny street crops next to full-size H&M
     cut-outs gave the replacement away, and some gate-passing crops were unreadable; fixed with
     `outfit_board(upscale=True)` and a 4,000-px minimum garment area (the slide figures' bar). With
     that bar the swaps are mostly clothing (shoe 1 of 20; the first build had 8 of 20 shoes).
  2. Phone-photo set: `Module4/scripts/photo_batch.py --photos <folder>` (results.csv, summary, overlay
     sheet); checked on `samples/nonhuman/` (same results as above).
  3. Usability checklist and bug log for the demo (owner: the user, 2026-11-22).
  4. Catalogue QA: `Module4/scripts/catalogue_qa.py sheets` drew 30 random suggestable H&M items per type
     (480 items, 16 sheets) with a `marks.csv` to fill; `score` gives defect rates. Visible already:
     studio background left inside closed loops (chain belt).
  Tests: human study 3, photo batch 3, catalogue QA 2, outfit board +1; Module 4 unit tests 88 pass.
- [x] **Kit 1 result: human study pilot (2026-11-23).** 26 raters (Google Form, 22–23 Nov), 40
  questions, 1,040 answers, 59 "Can't decide". Swapped outfit preferred over the original:
  **Module 4 37.4%, random 37.8%, lift −0.4 points** (95% CI over raters −5.3 to +4.8; over raters
  and outfits −10.9 to +9.6). **No measurable lift: people did not prefer Module 4's swap to a
  random allowed swap.** Data checks clean: no A/B position bias (A chosen 53.5%), no rater
  dominates. Breakdown (10 outfits each, not significant): Pinterest lift +9.3 (CI −2 to +20),
  Fashion144k −10.4 (CI −26 to +3); Module 4 beats random on 9 of 20 outfits. Worst: pants → shorts
  (9% prefer). Model harmony gain does not track human preference (Spearman 0.10 over the 40
  questions; Module 4's ΔH z 1.3–4.5 vs random −1.4 to +1.4). Originals preferred 62% in both
  conditions; this mixes "the outfit was fine" with "street photo plus studio cut-out looks odd", so
  it does not by itself say the swap verdict is wrong, but the lift (which cancels the photo style)
  is ~0. For the slide: a negative pilot, reported as such; it supports the style-invariant encoder
  and GNN-on-modern-data roadmap. Responses: `Human Study – Outfit Preference Survey.csv` (project
  root, not in git); score in `Module4/cache/human_study/result.json`. The form export has one extra
  unnamed column (3 non-empty cells: 3, 5, 5), ignored by the scorer.
- [x] **Kit 3 result: usability test (2026-11-23), run by Claude** in headless Chromium against
  `demo.py --advanced`; full checklist and bug log in `docs/kit3-usability-results.md`, screenshots
  in `Module4/cache/usability/`. Cold start: server 4 s, first run 27.1 s, then 3.9 s (All); single
  modules 0.7–1.8 s; slider re-rank < 1 s. 14 issues: 0 blocker, 4 major, 6 minor, 4 cosmetic.
  Majors: (1) no "no person" stop for non-animal photos: a landscape gets garments cut from snow and
  a verdict; (2) the same photo scores harmony 77 or 79 on reruns (follows Module 1 time, ~1.4 s →
  77, ~0.6 s → 79): root cause not yet found; (3) a 44 MP photo runs out of GPU memory in
  SegFormer's upsampling (12 and 24 MP pass in 8–13 s): downscale on upload; (4) Device `cpu` after
  a GPU run crashes. Odd inputs handled well 5 of 10. First-time understanding still needs a real
  first-time user.
  Bugs to fix (numbers as in the bug log):
  - [ ] B1 (major) No-person photos are analysed: extend the person check to stop any photo without
        a person (landscape, flat-lay, product page), not only animals.
  - [ ] B2 (major) Harmony 77 vs 79 for the same photo on reruns: find why Module 1 takes two paths
        (~1.4 s vs ~0.6 s) and gives different garments/embeddings; then make it deterministic.
        Lead (2026-11-23, not yet confirmed): the demo reuses one folder
        (`Fashion144k_v1/demo_out/demo_live`) and nothing clears it between photos; it holds 15
        garment PNGs from earlier photos (scarf from 19 Nov, shorts 20 Nov, sweater 22 Nov), and
        `extract_folder_embeddings` embeds every PNG in the folder (hence "15 embeddings" for a
        4-garment photo). Next: check whether the scorers read these stale files, then clear the
        folder at the start of each run. Fixes deferred to after Review 3 (owner decision).
  - [ ] B3 (major) 44 MP photo → CUDA out of memory in SegFormer's upsampling: downscale on upload
        (long side ~1600 px), which also cuts 12–24 MP runs from 8–13 s; show a friendly message
        on any out-of-memory error instead of the PyTorch text.
  - [ ] B4 (major) Device `cpu` after a GPU run crashes (tensors on two devices): ignore the change
        after the first run, or disable the radio once models are loaded.
  - [ ] B5 (minor) Unreadable image (text renamed `.jpg`) gives a bare "Error": catch it and say
        "could not read this image". Check iPhone HEIC with the Kit 2 photos.
  - [ ] B6 (minor) Group photo analysed silently: warn when more than one person is found.
  - [ ] B7 (minor) Tiny photo (110 px) analysed without warning (harmony 38 vs 77): warn below a
        minimum size.
  - [ ] B8 (minor) "Try this swap" options identical ("shoe (same type) (hm)"): add the item name or
        the harmony/trend change to each label.
  - [ ] B9 (minor) Same type, trend floor and rule show no visible change on the example photo:
        say "no change for this outfit" or explain each option.
  - [ ] B10 (minor) Per-piece table jargon (selection score, best ΔH (σ), best ΔT (σ), clears) and
        "drops to 70/100 if one garment is swapped": plain-word headers and a one-line note;
        say that harmony is a percentile against real outfits.
  - [ ] B11 (cosmetic) Figure text in Modules 1–3 ~7 px: larger fonts for the projector.
  - [ ] B12 (cosmetic) Suggestion captions cut off ("harmony 7…").
  - [ ] B13 (cosmetic) Sunglasses colour "red" (dark brown); shoes grouped as "accessory".
  - [ ] B14 (cosmetic) Dog example thumbnail letterboxed into a thin strip.
- [x] **Kit 4 result: catalogue QA (2026-11-23).** 480 items (30 per type × 16), **24 flagged = 5.0%**:
  wrong_type 13, unreadable 6, other 4, background 1. By type: sunglasses 13% (3 frames mostly
  erased, 1 leftover background), scarf 10% (ponchos/wraps, a beanie on a scarf), 6.7% for dress,
  hat (headbands), pants (jumpsuit, corset belt), shirt/blouse (tunic dresses), skirt, t-shirt,
  top; bag, belt, jacket, shoe 0%. Main failure is H&M's own category labels (wrong_type), not the
  cut-outs; thin-framed sunglasses are where the background removal breaks. **Lower bound:**
  belt #2 (chain belt, studio background inside the loop, the defect noted at build time) was not
  flagged. Handed back as `kit4results.xlsx` (Excel turned item ids into floats); marks re-joined to
  the original rows by type + cell (all 480 match) in `Module4/cache/catalogue_qa/marks_filled.csv`,
  score in `result.json` there.
- Demo on review day (owner decision 2026-11-22): runs on this machine, reached remotely through a
  Gradio share link. To do with the frontend overhaul: `--share` and `--auth` flags in `demo.py`
  (a share link is public, so it needs a password); pre-screen the team's demo photos for verdicts;
  warm start; an offline run (`HF_HUB_OFFLINE=1`); a backup screen recording.
- Demo front-end polish for Review 3 (2026-11-23, `demo.py`): rose/slate theme and CSS (launch
  arguments, as Gradio 6 requires); results in tabs (1 · Perception, 2 · Harmony, 3 · Trend,
  4 · Recommendation, Log); a single-module run opens its own tab, All stays put; a status line
  above the tabs shows "Done in Ns", a stop message (e.g. the person check on an animal photo) or
  an error from any tab; Run sits right under the module choice; three examples (image2, image.png
  and a dog in a sweater for the person check); the device and Module 4 options are hidden unless
  `--advanced` (they keep the calibrated defaults). Done from the review-day list: `--share` (refuses
  to start without a password), `--auth user:password` or `DEMO_AUTH`, `--port`. Still to do: the
  photo pre-screen, warm start, offline run, screen recording.
  Checked: the full chain through the running server with `gradio_client` (image2 → "Already strong"
  verdict and "✅ Done"; dog → "⚠️ No person found…"); light and dark screenshots. A click-through in a
  browser was not done here (Chrome automation fails on this machine), so test it by hand once.
  Tests: `Module4/tests/test_demo_ui.py` (8, no models); demo real tests 18 pass; Module 4 unit tests 96 pass.
  Follow-up (same day, owner's call): big centred heading with the project name "AI-Powered Personal
  Stylist" (also the browser tab title); warnings in an orange rounded box (`status_html`, errors and
  "Done" stay plain); examples cut to two (image2 and the dog; `image.png` dropped). UI tests 9 pass.
  The module choice is now a full-width bar under "What each module does"; each option is sized to
  its text, centred in the row, and the row wraps on narrow windows.

---

## 1. Module 4: Recommendation engine

Design is complete: see [`module4-design.md`](module4-design.md) (plain language) and
[`module4-design-detailed.md`](module4-design-detailed.md) (engineering spec).

**Phase 0: Pinterest into the catalogue.** Code done and debugged on a 2,000-photo pilot of the
224×224 copies (2026-10-09). Full run done 2026-10-10 (steps 1–4 below; step 5 wired up, full
evaluation pending a calibration decision). Pilot numbers (224×224, 2,000 photos) were AUC 0.731 vs
0.816 and backtest weight 0.6; the full run confirmed both.
- [x] Store mode in `Module3/scripts/pinterest_garment_segmentation.py`, the same Module 1 path as Fashion144k
      (shared `segment_photos()` loop); feature store with areas in the 98,304-px frame; seeded 60/10/30
      split with identical files kept together; parity guard against Fashion144k store rows.
- [x] AUC gate (`Module2/scripts/auc_gate.py`) and D11 backtest (`Module3/scripts/forward_share_backtest.py`).
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
      (`python Module4/scripts/calibrate_dup.py`, needs an owner look), as the implementation plan's Pinterest
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
      - Fix 2026-11-19: `Module4/src/crop_quality.py` measures all 634,742 crops (six shape metrics from the
        alpha mask, two CLIP scores from the stored embeddings: own-group probability and "text"
        probability) into `cache/crop_metrics.npz`; `crop_gate` in `m4_settings.json` sets per-type bars,
        applied when the pool loads (no rebuild to retune). Bars set from contact sheets
        (`Module4/scripts/calibrate_crops.py`), one row per rejection reason:
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
1. `python Module3/scripts/pinterest_garment_segmentation.py --mode store --metadata_csv <metadata.csv> --output_dir <store>`
   (~10 photos/s on the 224×224 copies; full resolution will be slower; resumable).
2. `python Module2/scripts/build_feature_store.py --output_root <store>/outputs --store_dir <store>/feature_store --photo_sizes <store>/photos.csv --area_frame_px 98304`
3. `python Module2/scripts/auc_gate.py --checkpoint Fashion144k_v1/ck_attr_subset_s42/clip_outfit_gnn_best.pt --fashion_store Fashion144k_v1/feature_store --fashion_split Fashion144k_v1/split.mat --pinterest_store <store>/feature_store --pinterest_split <store>/split.json`.
   If it fails, build the fine-tune path (merge both stores for `03_train_gnn.py`, warm start, adopt
   only if Pinterest AUC rises and Fashion144k drops by no more than 0.01).
4. Refit M3 on `<store>/garment_manifest.csv` + `garment_embeddings.npy`, re-run the stability test, then
   `python Module3/scripts/forward_share_backtest.py --manifest_csv ... --embeddings ...` and freeze the D11 weight.
5. Add `{"name": "pinterest", "store_dir": "PinterestData/store/feature_store", "output_root": "PinterestData/store/outputs", "split": "PinterestData/store/split.json"}`
   to `Module4/config/m4_settings.json` → `stores`; point `trend_model` at the refit model;
   `python Module4/scripts/build_cache.py`; re-run the evaluation for both sources.

**M4 core on Fashion144k (build step 2): done** (2026-10-09). See [`Module4/README.md`](../Module4/README.md).
- [x] Candidate pool, swap groups + co-occurrence filter, dense GNN scorer, harmony reference,
      selection rules, verdicts, dedupe, CLI.
- [x] Performance gate: `image2.jpg` vs the full demo pool, **0.16 s warm, 1.92 GiB peak** (target ≤ 5 s, ≤ 6 GB).
- [x] Dedupe rebuilt at outfit level (reposts), after the CLIP-only rule turned out to merge distinct tiny crops.

**Evaluation (build step 3): done** (2026-10-09). `Module4/src/eval_recovery.py`, spec §6.
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
      - **H&M chosen; work in progress (checkpoint 2026-11-20).** Downloaded to `H&M Dataset/`
        (gitignored): 105,542 articles, 105,100 with a photo, all clean flat product shots on a plain
        light-grey background. `index_group_name` gives the gender signal (Menswear 12,553; Ladieswear
        39,737; Divided 15,149; Sport, split by `section_name`); Baby/Children (34,711) is dropped.
      - Background removal: rembg compared on hard items (white clothes on grey, sunglasses, belts).
        BiRefNet-lite is cleanest (the only one that keeps white sunglasses) but 3.5 s/image on CPU;
        u2net is nearly as good at 0.18 s; isnet-general-use failed on a white blouse and print trousers.
        u2net is used for the domain check; pick the model for the full catalogue after it (BiRefNet if
        onnxruntime-gpu works).
      - `Module4/src/hm_catalogue.py`: H&M type → Module 4 type map, adult filter, gender, cut-out, then a
        Module 1-style folder per article (PNG + CLIP `.npy` + Module 1 attributes), so Module 2's
        `build_feature_store.py` builds the store. A product photo has no body scale, so `pixel_count`
        is the median area of that type in the existing stores. Resumable (skips finished articles).
      - `Module4/scripts/hm_domain_check.py` (written, **not yet run**): type agreement (CLIP label_p), embedding
        gap (nearest same-type store garment, H&M-vs-store linear classifier), trend-cluster coverage,
        and the main test, GNN transfer: for test outfits of both catalogues, the candidate nearest to
        the worn piece is a stand-in for the original; compare its GNN rank among H&M candidates with
        its rank among an equal-sized set of store candidates.
      - Stopped at: the 150-per-type sample was 2,294 of ~2,400 items into `H&M Dataset/check_store`.
        Resume with:
        1. `python Module4/src/hm_catalogue.py --out "H&M Dataset/check_store" --per_type 150` (finishes the rest)
        2. `cd Module2 && python build_feature_store.py --output_root "../H&M Dataset/check_store/outputs" --store_dir "../H&M Dataset/check_store/feature_store"`
        3. `python Module4/scripts/hm_domain_check.py --store "H&M Dataset/check_store" --n 1000`
           (expect first-run fixes; writes `domain_check.json` and `gnn_transfer.csv` there)
      - **Sample finished, store built, check run (2026-11-20).** One fix: H&M article ids start with a
        zero and the feature store keys outfits by integer, so folders are now named
        `str(int(article_id))` (`hm_catalogue.item_key`; `articles.csv` keeps both). 2,400 items, 0 empty
        cut-outs; the check ran first time. Results (`H&M Dataset/check_store/domain_check.json`):
        - Type agreement: fine. H&M label_p is higher than the store's for every type except dress
          (median 0.23, 89% pass 0.02) and scarf (0.44, 84% pass); junk_p is near zero.
        - Embedding gap: large. A linear classifier tells H&M from store embeddings at **98.3%**;
          nearest same-type store garment cos 0.84–0.95 (median ~0.89) vs 0.92–0.97 store-to-store.
        - Trend: every item gets a score, but H&M collapses onto fewer M3 clusters (effective clusters
          1.4–5.3 vs 3.7–9.5; dress 1.45 vs 4.0) and mean trend shifts by type (pants 88 vs 64, hat 18
          vs 36). Trend scores on raw H&M embeddings reflect photo style, not only the garment.
        - GNN transfer: **near chance.** Stand-in rank percentile mean 0.46 (chance 0.5; store 0.35),
          top-10% 17% (store 24%), and it does not improve as the stand-in gets closer to the worn piece
          (0.50 at cos > 0.9). H&M candidates score **+0.43σ** higher on average than store
          candidates and beat the original outfit 20% of the time vs 10%: a domain bias, not taste.
          Pinterest outfits show it more than Fashion144k (+0.59σ vs +0.27σ).
        - Verdict: raw H&M embeddings are not safe for the demo pool. Harmony and trend both read the
          studio-shot style.
      - **Mean alignment tried (`--align mean`, `domain_check_mean.json`).** Each type's H&M CLIP
        centroid is moved onto the store's (demo representatives), then renormalised; no labels used.
        - Domain classifier 98.3% → 24% (no longer separable); nearest store garment cos 0.89 → 0.93.
        - GNN bias gone: logit shift +0.43σ → **−0.06σ**; share beating the original 20% → 7.4%
          (store 10%).
        - GNN ranking only partly recovers: stand-in rank 0.456 → **0.416** (store 0.349); top-10% stays
          at 17%. Pinterest outfits gain more (0.45 → 0.40) than Fashion144k (0.46 → 0.43).
        - Trend: clusters spread further for most types (pants 2.6 → 4.8, shoe 3.4 → 6.8, skirt
          3.4 → 6.5) but stay under the store's; scarf gets worse (5.3 → 2.3).
        - Side effect: the CLIP label gate gets worse on shifted accessories (belt 99% → 83% pass,
          scarf 84% → 64%). H&M types come from product metadata, so the gate should read the raw
          embeddings, or H&M should skip the label part of the gate.
        - Open decision: adopt the mean shift for the full catalogue (it removes the bias that would
          let H&M items crowd out store suggestions) or try stronger alignment first (CORAL, or a
          mapping learnt on paired data).
      - **Decided (owner, 2026-11-20): mean shift for the full catalogue; demo suggestions from H&M
        only.** Street stores stay in the pool for scoring, co-occurrence and evaluation; H&M has no
        split, so it never enters `rep_eval`, the harmony reference or `cooc.npy`.
        - Shared code: `hm_catalogue.street_type_means` (mean unit CLIP embedding per type over all
          typed rows of the street stores, read from their feature files, so it does not depend on the
          pool cache) and `shift_to_means`; `hm_catalogue.py --align <store_dir>` rewrites
          `features.npy` from `features_raw.npy`. `hm_domain_check.py --align mean` uses the same code.
        - Re-run with these means (the first run used gated demo representatives): classifier 47.6%
          (chance), logit shift −0.05σ, beats-original 7.7% (store 10.2%), stand-in rank 0.423, top-10%
          14.7%. Same picture as before.
        - Background removal: onnxruntime-gpu 1.23.2 (the CUDA 12 build; 1.24+ wants CUDA 13, torch
          here is cu126) replaces onnxruntime. On the GPU u2net takes 0.11 s/image; BiRefNet-lite is
          *slower* on the 8 GB GPU (6.5–8 s) than on CPU (3.5 s), so the catalogue uses u2net.
        - Settings: `StoreSpec.product_photos` (the crop gate skips its CLIP label/junk/text checks for
          these rows: type comes from product data, and the shift makes CLIP misread accessories) and
          `demo_sources` (stores whose garments may be demo suggestions; applied at load time).
          `pool_fingerprint` hashes a product store's `features.npy`, since `--align` rewrites it in place.
        - Full catalogue: 55,897 adult articles (`H&M Dataset/store`, seeded with the 2,400 sample items).
      - **H&M is the demo pool (2026-11-20).** 55,862 written (35 empty cut-outs), feature store built
        and aligned, `hm` store in `m4_settings.json` (`product_photos: true`, empty `split.json`),
        `demo_sources: ["hm"]`. Cache rebuilt: 690,604 rows; 54,737 H&M demo representatives pass the
        shape gate (shoe 2,719, bag 802, belt 376, sunglasses 410). Eval reps (190,602) and the harmony
        reference (56,273 test outfits, σ_logit 3.119, σ_trend 26.58) are unchanged, so calibrated
        results still hold.
        - The first cache rebuild was stopped by Claude Code for low system memory during the crop
          shape scan (12 processes over 690k crops); `build_cache.py --workers 4` finished it (scan
          424 s).
        - Tests: two pool tests assumed the old demo pool (all demo rows pass the CLIP gate; demo pool
          larger than eval pool); updated to the product-photo gate and `demo_sources`. 103 pass.
        - Demo verdicts, same 400 test outfits, street vs H&M suggestions: swap 14.0% vs 13.8%,
          refresh 85.8% vs 85.5%, keep 0.2% vs 0.8%; 90.8% get the same verdict. Top suggestion's
          harmony gain is smaller with H&M (median +0.84σ vs +1.34σ): a pool a sixth the size, no
          longer boosted by photo style.
        - Visual check (4 random swap outfits): suggestions are clean product cut-outs and the
          gallery alternatives are coherent (black flats; pale-blue bags). u2net leaves grey studio
          background inside closed loops (inside a belt-bag strap).
        - [ ] Open: **H&M swaps lean to shoes** (21 of 55 swaps vs 9 of 56 with street suggestions;
          pants 10 vs 17). After alignment the average logit shift is near zero for every type, but
          the best H&M candidate scores under the best store one by ~0.17σ for shoe / hat / belt and
          0.28–0.35σ for clothing (`gnn_transfer_mean.csv` by type), so H&M clothing loses a little
          more than accessories. Options: per-type alignment of the top tail rather than the mean,
          or align the colour / attribute features too (only the CLIP part is shifted now).
          - **Trend side too (2026-11-21).** All H&M rows get a trend score (100% matched, every
            type), but H&M clothing piles onto one M3 cluster per type: top-cluster share sweater 0.69,
            shirt 0.65, dress 0.78, top 0.59 (street 0.17–0.37). So clothing trend has almost no top
            tail (H&M p90 vs street p90: shirt 57 vs 96, jacket 71 vs 96, sweater 71 vs 91), while
            H&M shoes sit above street (p90 86 vs 76; bag/belt also high). Refresh (~85% of verdicts)
            needs ΔT > 0 and ranks on 0.8·ΔH + 0.2·ΔT, so shoes win on both terms. `decide()` pools all
            pieces' refresh candidates and `diverse_top` diversifies by cluster only, so one winning
            piece can take all three slots. Likely fix: score trend on the un-shifted (or per-type
            aligned) embedding, and/or cap refresh suggestions to one or two per piece.
          - [x] **Refresh cap (2026-11-21).** `refresh_per_piece` (default 1): refresh suggestions
            come from different worn pieces while other pieces have options; the cap is lifted only to
            fill the three slots (`diverse_top(groups=, max_per_group=)`). Swap verdicts are one piece
            by design and unchanged. 105 tests pass.
          - [x] **CORAL alignment (2026-11-21), adopted.** The per-type mean shift left the studio
            photos' tighter spread. `hm_catalogue.py --align --method coral` (now the default) matches
            each type's mean and covariance (Ledoit-Wolf on the H&M side). `hm_domain_check.py --align
            mean|coral` on the full catalogue, 1,000 test outfits:
            - domain classifier 0.76 → 0.60 (the earlier "chance" figure was the 150-per-type check
              sample); best H&M vs best street candidate: clothing −0.30…−0.42σ → −0.00…+0.12σ, shoe
              −0.16σ → +0.03σ; stand-in GNN rank 0.38 → 0.36 (street 0.28).
            - effective trend clusters shirt 2.8 → 5.9, sweater 3.0 → 5.8 (street 9.1 / 8.9).
            - The check crashed on the full catalogue (H&M has more shorts than the stores); the
              street comparison sample is now capped at what exists.
            - Cache rebuilt (`--workers 4`): demo reps 54,737; eval reps and harmony reference unchanged
              (σ_logit 3.1194), so calibrated results hold. 108 tests pass.
          - **After both fixes, 400 test outfits, H&M vs street suggestions** (same outfits):
            verdicts swap 57 / 56, refresh 341 / 343, keep 2 / 1. Refresh suggestions: shoes 17% of
            H&M suggestions vs 17% street; pants 19% vs 16%. Swaps: shoe 18 of 57 vs 9 of 56 (was 21
            of 55 before CORAL); pants 12 vs 16. Median harmony gain of the top swap +1.57σ vs +2.08σ.
          - [ ] **Remaining:** H&M swaps still pick shoes about twice as often as street (small sample:
            57 swaps). Next to check: per-type `slot_percentile` (share of candidates beating the worn
            piece), which depends on the whole candidate distribution, not the best one.
        - [ ] Click through the demo in a browser with H&M suggestions.
      - **Style-invariant embedding: planned (2026-11-21).** Decision: replace the CORAL patch with a
        learned bridge between photo styles. Product-only training would move the gap to the user's
        phone photo, so one garment embedding is trained for consumer, phone and studio photos alike.
        Roadmap `docs/superpowers/plans/2026-11-21-style-invariant-roadmap.md`: Phase 1 evaluation
        suite and baseline (detailed plan `2026-11-21-phase1-evaluation-suite.md`), Phase 2 encoder
        (contrastive fine-tune of CLIP ViT-B/32 on DeepFashion2 consumer/shop pairs + phone-style
        augmentation + distillation), Phase 3 re-embed stores, refit trend, retrain GNN with Polyvore
        Outfits and hard negatives, Phase 4 Module 4 recalibration and blind human study on phone
        photos. Zero-shot CLIP stays in a separate `clip_zs.npy`, since fine-tuning breaks text alignment.
        - [ ] Request DeepFashion2 access (form on its GitHub page; password by email).
        - [ ] Collect the phone-photo set (`docs/phone-photo-set.md`, written in Phase 1 Task 6).

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
      swap" board (`Module4/src/outfit_board.py`, shared with the human study), M2 and M4 on one GNN and
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
- [x] **Module folders restructured (2026-11-23, branch `restructure-modules`, owner request).** Each
  module now has `src/` (code other modules or the demo import; still runnable), `scripts/`
  (command-line tools nothing imports), `tests/` and `config/` (Module 4's `m4_settings.json`,
  requirements). Rule used: a file imported by any non-test code goes in `src/`. Trained models,
  weights and caches stay at the module root, so no data path changed. Paths in code, docs and
  this file were rewritten to the new layout. Notable moves: `run_recommend.py`, `eval_recovery.py`,
  `hm_catalogue.py` in `Module4/src/` (imported elsewhere); Module 2's `01_`–`03_` and
  `demo_score.py` in `Module2/src/`. Verified before and after: M1 23, M3 7, M4 97 quick + 33 `real`,
  all pass (M1's now also from the repo root: `python -m pytest Module1`); 58 of 59 src/scripts files load (the exception, `Module3/scripts/gradio_demo.py`, needs
  a `pinterest_trend_model.pkl` that does not exist, as before); a full demo run on `image2.jpg` gives
  the same harmony 77, verdict and 41,866 what-if outfits.
- [ ] Before any M3 re-run: `Module3/tests/test_module1_sync.py` and `Module3/tests/test_embedding_parity.py` must pass.
