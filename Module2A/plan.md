# Performance Plan — Segmentation + GNN Training

Goal: make the pipeline fast enough to **retrain the GNN many times** with small parameter
tweaks. Today training is I/O-bound (~90 outfits/s); the target is GPU-bound with a one-time
precompute so every future run is minutes, not hours.

**Scope note:** attributes vs no-attributes decision is orthogonal — everything here works
for either arm (the `--no_attributes` flag stays functional throughout).

## Guiding principles
- **Precompute once, ever** — not once per run. Persist to disk; every future run loads it.
- **Each training run is its own process** → RAM is reclaimed by the OS at process exit.
  Use `np.memmap` for the feature store so in-run RAM footprint is near-zero and OS-managed.
- **No silent result changes.** Every tier must preserve AUC (within seed noise). The one
  behavioural change (graph minibatching) gets an explicit parity test before we trust it.
- Keep the old on-the-fly path behind a flag until the fast path is verified.

## Parity baseline (regression guard)
Reference number to compare against after each tier (single run, seed 42, 5k pilot,
with attributes): **test AUC ≈ 0.65** (update with the 5k seed-averaged result once known).
After each tier: rerun one 5k pilot config and confirm AUC within ±0.02 of reference
(batching verified more strictly, see Tier 2).

---

## Tier 0 — Cheap segmentation speedup  (Module 1 inference; one-time cost)  — batched embeds kept, fp16 REVERTED
Target: ~1.5–2× the current 7 img/s.

- [x] **Batch** the per-outfit CLIP embedding extraction (all garment crops in one forward). Safe, kept.
- [~] **fp16/autocast** — **REVERTED**. It caused an incident: casting image features to
      float32 while the cached CLIP text features stayed float16 → `float != Half` matmul in
      `_zero_shot`, which failed *every* image. CLIP is already fp16 on CUDA so autocast gave
      no benefit anyway. Segmentation is a one-time cost, so fp16 is not worth the risk.
- [x] **Circuit breaker in `01`** (`--max_consecutive_failures`, default 25): a systemic error
      now aborts the run instead of silently marking the whole dataset "done". (Root cause of
      the incident: on exception `01` marked the index done but did NOT count it toward
      `--limit`, so a systemic failure marched through ~22k images.)
- **Files:** `Module1/clip_extract_embeddings.py` (batched), `Module2A/01_segment_batch.py`
  (circuit breaker). fp16 edits reverted in `fashion_segmenter.py` / `attribute_analyzer.py`.
- **Acceptance:** ✅ segmentation + attributes + embeddings verified working on GPU after
  revert; ✅ `--limit` honored; ✅ no runaway.
- **Lesson:** don't fp16 an already-fp16 model; and any "mark done on failure" loop needs a
  consecutive-failure guard.

## Tier 1 — Persistent precomputed feature store  (kills per-run disk I/O)  ✅ DONE
Target: remove PNG reopen + colour-histogram recompute from **every** training run.

- [x] `Module2A/build_feature_store.py`: memmaps `features.npy` (N×570) + `node_scalars.npy`
      (N×3 = hue/formality/area) + `index.json` (`(idx,file)->row`, `idx->[rows]`).
- [x] Colour histogram computed **once** here.
- [x] `build_pool` (store-backed category pool) — instant, no disk scan.
- [x] `02_outfit_dataset.py`: store-backed `load_outfit_nodes` / `build_node`; on-the-fly path
      kept behind `--feature_store`.
- [x] `--feature_store` wired through `03_train_gnn.py` + `run_all.py`.
- **Acceptance:** ✅ store graphs **bit-identical** to disk graphs (100/100 allclose);
  ✅ CPU run disk==store **exactly** (0.6357 = 0.6357). Store = 36 MB for 5k → ~1 GB for 144k.
- **RAM:** loaded via `mmap_mode='r'` → OS page-cache, evictable, never bricks.

## Tier 2 — Minibatch graphs on GPU  (the compute win)  ✅ DONE
Target: turn the idle GPU into the workhorse; 10–50× over Tier 1 alone.

- [x] `OutfitDataset.collate`: pack B outfits into one disconnected graph (offset edge_index,
      `batch` vector).
- [x] Readout → **segment mean-pool by `batch`** (index_add), one score per outfit.
- [x] Vectorized BPR over the batch.
- [x] `--batch_size` on `03` / `run_all`.
- [ ] Features-in-VRAM — **not needed**: with the memmap store + batching, a full run is ~14s;
      no per-step transfer bottleneck remains. Left as a future option.
- **Acceptance:** ✅ batched forward == stacked single forwards (allclose, 6 graphs);
  ✅ 5k pilot batched test AUC 0.6568 (in with-attr range); **~20× faster** (~300s → 14s).
- **Risk:** edge offsetting + pooling — covered by the parity test (passed).

## Tier 3 — Cheap extras  ✅ DONE (workers obviated)
- [x] Multi-negative sampling (`--num_negatives`) — helped: test AUC 0.6568→0.6665 at K=4.
- [x] Configurable val-eval size (`--val_eval_max`).
- [x] Store-backed category pool (from Tier 1) reused across runs.
- [ ] `DataLoader(num_workers=…)` — **obviated**: at ~14s/run there is no I/O bottleneck to
      prefetch. Not implemented.
- **Files:** `03_train_gnn.py`, `02_outfit_dataset.py`, `run_all.py`.
- **Acceptance:** ✅ runs, no regression; K=4 improved AUC.

---

## Sequencing
0 → 1 → 2 → 3. Tiers 0/1/3 are low-risk and result-preserving; Tier 2 is the one code
change that needs the parity test. Do Tier 1 before the full 139k so the store is built from
the same segmentation pass.

## Expected payoff (estimates)
| Stage | Full 144k × 10-epoch training run |
|---|---|
| Today (I/O-bound) | ~4 h |
| + Tier 1 (cached store) | ~40 min |
| + Tier 2 (batched, VRAM) | a few minutes |

Across dozens of tweak-runs this is the difference between days and an afternoon.

## Verification protocol (run after each tier)
1. Rebuild/reuse the 5k pilot `outputs/`.
2. Run one config (seed 42, with attributes) and compare final test AUC to the parity baseline.
3. For Tier 2 also run the batched-vs-single unit parity test.
4. Record the tier's completion + measured speedup in `MODULE2A.md` §2.

## Done-when
All four tiers merged, parity verified, `MODULE2A.md` updated, and a full-data training run
completes in minutes so parameter sweeps are practical.

---

# Execution log — full-scale run on Fashion144K (local Windows / RTX, 2026-08)

Everything above (Tiers 0–3) was designed and parity-checked on a 5k/10k pilot. This section
records what happened when the pipeline was actually run end-to-end on the **full 144,169-outfit
dataset**, plus the fixes that only the full scale surfaced.

## Windows / local port (prior work, pre-full-run)
- **Unicode stdout fix** (`01_segment_batch.py`): Fashion144K has non-ASCII filenames (e.g.
  Polish `ń`). On Windows a piped/redirected stdout defaults to cp1252 and crashes mid-run;
  now `stdout`/`stderr` are reconfigured to UTF-8 (`errors=replace`) and all fail-log writes
  use `encoding="utf-8"`. This is what the "Unicode-fixed" segmentation restart refers to.
- **SegFormer-only fp16 autocast** (`Module1/fashion_segmenter.py`): re-introduced `torch.no_grad(),
  _amp(dev)` around the SegFormer forward *only* (fp32 model → real speedup), casting
  `outputs.logits.float()` before interpolate. Scoped to SegFormer so it does **not** touch the
  CLIP paths that caused the earlier `float != Half` incident (see Tier 0).

## Stage 1 — Full segmentation  ✅ DONE
- Ran `01_segment_batch.py` over all 144,169 photos (resumable checkpoint). **144,169/144,169
  completed**, 16 failures (~0.01%: missing files / no-garment images). Throughput ~8–9 img/s.
- Outputs at `Fashion144k_v1/outputs/<idx>/` (SegFormer crops + CLIP `.npy` + `metadata.json`).

## Stage 2 — Full feature store  ✅ DONE
- `build_feature_store.py` over the full outputs → **480,079 garment rows × 570** (512 CLIP +
  32 colour + 26 attr), ~1.04 GB `features.npy` + `node_scalars.npy` + 42.6 MB `index.json`.
- **Consistency verified**: `index.num_rows == features rows == node_scalars rows == rows listed
  in index == 480,079`, over **144,169 outfits**, row_dim 570. Safe to train on.
- **Gotcha:** a first build died silently (0-byte log, no traceback) leaving a *fresh 480k
  `features.npy` paired with a stale 5k `index.json`* — an unusable mismatch. Root cause: the
  1 GB memmap sits inside a **OneDrive-synced** folder; sync can lock/kill the file mid-write.
  Re-running with unbuffered output (`python -u`) completed cleanly. **Recommendation:** build
  the store to a local, non-OneDrive path for heavy/repeated runs.

## Stage 3 — GNN training at full scale  ✅ DONE (with attributes)
### Perf bug found only at full scale: negative sampling was O(pool) per negative
- Symptom: **~27 min/epoch, GPU at 3% util** (CPU-bound). The store+batching made the GPU work
  trivial, so nothing hid the CPU cost anymore.
- Cause: `make_negative_sample` did `[c for c in category_pool[cat] if c[0] != idx]` — a full
  scan/copy of the category pool **per negative**. At full scale pools are ~100k (`accessory`
  114,241), so each of `batch×num_negatives` draws scanned ~100k entries. The 5k/10k pilots
  never exercised this (pools ~8.6× smaller, ~75× cheaper total), so Tiers 0–3 missed it.
- Fix: **rejection sampling** — pick a random pool entry, retry only on the rare same-outfit
  collision. O(1) amortised, identical draw distribution. → **~1 min/epoch, ~27× faster**,
  GPU util up, and val/test AUC preserved.

### Results (10 epochs, batch 64, num_negatives 4, lr 1e-4, seed 42, best-val checkpointing)
| Run | usable train | best val AUC | **test AUC** (official 43,250 testids) |
|---|---|---|---|
| 10k pilot (`--limit_train 10000`), with attrs | 9,624 | 0.6874 (ep4) | **0.6673** |
| **Full, with attrs** | 83,352 | 0.7784 (ep6) | **0.7780** |
| **Full, no attrs (ablation)** | 83,352 | 0.7846 (ep9) | **0.7951** |

- Full-data run: val ≈ test (0.7784 ≈ 0.7780) → **no overfitting**; +0.11 test AUC over the
  10k pilot purely from more training data. Whole 10-epoch run ≈ 11 min after the fix.
- **Ablation verdict (seed 42): baseline BEATS attributes by +0.0171** (0.7951 vs 0.7780).
  This resolves the §2.6 open question ("attributes may pay off with more data"): they do **not** —
  the −0.017 gap at 86.5k is consistent with, and slightly wider than, the 5k seed-averaged
  −0.0128. CLIP already encodes colour/pattern/formality visually; the explicit attributes are
  redundant and the noisiest (fabric/fit) add variance.

## Code changes made this session
- `03_train_gnn.py`: **neg-sampling O(pool)→O(1)** (the fix); removed dead `bpr_loss()`; removed
  legacy `--grad_accum`; added opt-in `--limit_train N` (pilot cap on train ids; val/test full).
- `run_all.py`: removed `--grad_accum` passthrough.
- `02_outfit_dataset.py`: removed a redundant local `defaultdict` import.
- Left intact deliberately: `item_score`/`per_item` head (unused in loss, but removing it changes
  the model `state_dict` and would break existing checkpoints).

## Next
- **Done:** no-attributes ablation → baseline wins by +0.0171 (see table above).
- Optional: seed-average (42/43/44) both arms at full scale to tighten the single-seed −0.017.
- Optional: test an **attribute subset** (reliable colour/silhouette only, drop noisy
  fabric/fit/pattern) — may help even though the full attribute set hurts.
- See `miniREADME.md` for the consolidated end-to-end story.
