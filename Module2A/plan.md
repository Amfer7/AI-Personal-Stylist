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
