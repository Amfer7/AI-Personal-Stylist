# Module 4 Core Implementation Plan (Fashion144k only)

**Status (2026-10-09): Tasks 1–11 complete.** 60 tests pass; `t_dup` = 0.99 (calibrated); performance
gate met (0.16 s warm, 1.92 GiB peak). The checkboxes below were not ticked as work went along.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Module 4 swap recommender core on the Fashion144k feature store: candidate pool,
swap groups, vectorised exhaustive GNN scoring under a 6 GB VRAM cap, selection rules, verdicts and
dedupe. The result is a `Recommendation` for any M1 outfit folder.

**Architecture:** A one-off cache build (`build_cache.py`) turns the feature store into a candidate pool:
- the GNN input projection precomputed for every garment;
- M3 trend scores and cluster ids;
- near-duplicate representatives;
- a co-occurrence matrix;
- a harmony reference.

At request time `Recommender.score()` runs every allowed what-if swap through a dense, broadcast
re-implementation of the M2 GNN (`dense_gnn.py`), which is parity-tested against `model(build_graph(...))`.
`Recommender.decide()` then turns the cached per-candidate deltas into a verdict with **no further GNN
calls**, so the trend-weight slider and the evaluation sweeps are free.

**Tech stack:** Python 3.11, PyTorch 2.9 (CUDA), NumPy, SciPy (`connected_components`), Pillow,
matplotlib (contact sheets only) and pytest. Existing M1/M2/M3 code is reused, never copied.

**Spec:** [`module4-design-detailed.md`](module4-design-detailed.md) (decisions D1–D11). This plan covers
build-order **step 2** only. Steps 3–5 are listed at the end.

## Global Constraints

- **VRAM cap: 6 GB** (`--vram-gb`, default 6). The GPU is shared. Enforce it with
  `torch.cuda.set_per_process_memory_fraction`, and on CUDA OOM halve the chunk and retry.
- **Performance gate (D5):** `samples/image2.jpg`, full demo pool, `recommend()` **≤ 5 s warm** on the
  RTX 3070 Ti, peak VRAM ≤ 6 GB.
- **Use the relative `file` field, never `crop_path`**, from `metadata.json` / `index.json`. Crop paths are
  rebuilt as `<output_root>/<outfit_key>/<file>`.
- **No images in git.** Contact sheets, M1 runs and caches go under `Module4/cache/` (already gitignored).
  Tests write synthetic PNGs to pytest's `tmp_path` only.
- **No personal details or absolute paths** in code, docs, settings or the JSON output. All paths are
  repo-relative and resolved through `M4Settings.abs()`.
- **Don't edit vendored M1 files** (`test_module1_sync.py` guards them) and **don't modify Module 2/3 code**.
  Load it through `Module4/m2_bridge.py`.
- **The shipped GNN is the attr-subset checkpoint** (`ck_attr_subset_s42`): node input =
  512 CLIP + 32 colour + 11 attribute dims = **555**, not 570. Always slice store rows with
  `m2_bridge.FEATURE_COLS`.
- **Commits:** ask the project owner before every commit or push (repo rule). Each task ends with a
  proposed commit.
- Run every command from the repo root. Real-data tests are marked `real` and need
  the Fashion144k data plus a built cache. Unit tests run anywhere: `python -m pytest Module4 -m "not real"`.

---

## Verified facts this plan relies on (checked 2026-10-07)

| Fact | Evidence | Consequence |
|---|---|---|
| Store: 480,079 rows × 570, `node_scalars` 480,079 × 3, hue never NaN, CLIP rows L2-normalised | `feature_store/` | Pool = 480,079 rows. Projection cache = 480,079 × 128 fp32 = **246 MB** (~320 MB once Pinterest's ~144k rows join, which is where the spec's "~620k" figure comes from) |
| Checkpoint is attr-subset: `input_proj` takes 555-D input | `demo_score.py`, `attr_subset_indices()` | Spec D5's "570→128" is really **555→128** (docs fixed in Task 11) |
| Only 4 broad categories occur in the store (`top`, `bottom`, `dress`, `accessory`); M1's SegFormer path emits the same 4 | `index.json`, `fashion_segmenter.py:247` | Every D4 swap group stays inside one broad category. The edge-weight table still covers all 7 broad names M2 knows |
| M1's specific-type vocabulary is exactly the 16 store types | `FINE_GRAINED_CANDIDATES` | Any other type is flagged "no swap group" |
| No outfit contains two pieces of the same type (left/right shoes are merged) | `index.json` scan | A swap is always 1 piece → 1 piece. Same-type co-occurrence counts are 0 by construction, so a candidate whose type is already worn is blocked |
| Train co-occurrence: pants+leggings **0**, pants+shorts **0**, leggings+shorts **0**, any two upper types **0**, skirt+leggings **2,878**, shorts+skirt **8** | train split scan | The D4 filter with min 20 works as the spec intends |
| Fashion144k test split per type: shoe 28,329 · pants 19,169 · bag 17,130 · jacket 15,025 · skirt 11,701 · dress 11,246 · shirt_blouse 8,436 · sweater 8,015 · leggings 7,268 · hat 6,643 · scarf 3,163 · top 3,131 · t_shirt 2,726 · belt 1,405 · sunglasses 548 · shorts 235 | spec fact 4b | Asserted in `test_candidate_pool.py` |
| Trend model: broad categories `top`/`bottom`/`dress`/`accessory`, k = 15/15/8/15, match threshold cosine 0.20; cluster ids are unique **per broad category** | `per_category_trend_model.pkl` | Cluster key for D8c diversity = (broad, cluster_id) |
| **M1 does not resize photos; `pixel_count` is in raw pixels.** Fashion144k photos are all 256×384 (98,304 px); `image2.jpg` is 415×737 (3.1×); a phone photo can be ~100× | `fashion_segmenter.py:222`, `outputs/*/*_vis.jpg` | **New requirement (see "Area normalisation" below)** |
| Feasibility prototype (scratchpad, not committed): dense scorer vs `model(build_graph)` on 1,000 random swaps gives **edge features identical (max diff 0.0)** and **logits within 1.5e-5**. A 5-piece outfit against its *whole* broad-category pool (823,170 what-if outfits) scores in **0.42 s warm, 0.84 GB peak VRAM** | prototype run | The D5 gate has ~10× headroom, so the spec's fallback pre-ranker isn't needed |

### Area normalisation (fix to a latent bias; not in the spec)

The GNN edge feature `volume_contrast = tanh(log((area_a+1)/(area_b+1)))` compares raw pixel counts.
Inside one photo the ratio doesn't depend on resolution, so M2's demo is fine. But M4 puts a
**catalogue** crop (256×384 frame) next to the **user's** pieces (any resolution). For `image2.jpg` every
candidate would look ~3× smaller than the pieces around it (edge value shift ≈ tanh(log 3.1) ≈ 0.81),
and for phone photos it would saturate. **Fix:** when loading a user outfit, rescale each piece's area to
the Fashion144k frame: `area × 98,304 / (photo width × photo height)`. The photo size is read from M1's
`<id>_vis.jpg` (same size as the input). Pool rows stay as stored. **Phase 0 must store Pinterest areas in
the same 98,304-px frame** (listed under "How Pinterest plugs in").

### Clarifications of the spec made here (owner please confirm)

1. **Trend floor** (`--no-trend-drop`, off by default): drop candidates whose trend is *known* and
   *below* the current piece's known trend. Candidates without a trend score are kept. If the floor leaves
   nothing, it is relaxed for that piece and flagged (§5).
2. **Bar for suggestions #2/#3 (D8c):** gain ≥ τ for the `max_gain` and `topk_mean_gain` rules. For
   `slot_percentile`, where τ is a fraction rather than a gain, the bar is gain > 0.
3. **`refresh`:** options are pooled across all pieces, ranked by combined gain, and made style-diverse
   with the same D8c routine. Each suggestion records which piece it replaces.
4. **`baseline.combined` dropped.** Gains are relative to the current outfit, so the baseline's combined
   score is 0 by definition. The baseline reports harmony %, harmony logit and mean piece trend.
5. **Dedupe representatives are computed twice:** once over the demo pool, once over test rows only. Otherwise an
   eval-pool duplicate group whose representative is a train row would vanish.
6. **Provisional knobs.** `w_t`, `τ`, `ε`, the selection rule and `t_dup` are calibrated in step 3
   (`t_dup` visually in Task 5). Until then they come from `Module4/m4_settings.json` with
   `status: provisional`, and every `Recommendation` lists them under `provisional`. Provisional values:
   w_t = 0 (rank on the validated harmony signal only), τ = 0 / 0 / 0.5, ε = 0, t_dup = 0.97.
7. **No padding masks.** What-if outfits keep the original size and the reference outfits are batched by
   size, so every dense batch is a full complete graph.

---

## File structure

All new code goes in `Module4/`, flat, following the Module 3 convention of tests next to the code.

| File | Responsibility |
|---|---|
| `m4_settings.json` | Every path and knob, repo-relative; `status` marks provisional values. Committed (no personal data) |
| `m4_settings.py` | `StoreSpec`, `M4Settings` (load/save/`abs()`/`with_()`), `REPO` |
| `m2_bridge.py` | The single doorway to M2 (`od`, `tg`) and M3 (`load_trend_model`); `FEATURE_COLS`, `BROAD_VOCAB`, edge-weight table, device + VRAM cap helpers |
| `swap_groups.py` | 16 types, D4 groups, `replacement_types()`, `CooccurrenceFilter` |
| `dense_gnn.py` | `edge_tensor()` + `DenseGNN` (projection, dense batched logits, VRAM-aware chunking with OOM halving) |
| `candidate_pool.py` | Pool build (projection, trend, dedupe, metadata) + `CandidatePool` (GPU tensors, candidate queries, crop paths) |
| `calibrate_dup.py` | Contact sheets of pool pairs per cosine band, for picking `t_dup` visually (D8) |
| `harmony_reference.py` | `HarmonyReference` (sorted real-test logits, σ_logit, σ_trend, `pct()`), `build_reference()` |
| `outfit_state.py` | `Piece`, `OutfitState` (immutable, `with_swap()` for future multi-piece), `load_outfit_dir()` (area normalisation), `state_from_pool()` |
| `swap_scorer.py` | `SwapScorer`: outfit logit + all swap logits for one piece via `DenseGNN` |
| `selection.py` | Pure functions: gains, the 3 selection rules, cluster keys, style-diverse top-3 |
| `explain.py` | Pure function: descriptive "what changed" line (D6) |
| `recommendation.py` | `Suggestion`, `Recommendation` dataclasses, JSON round-trip, `format_text()` |
| `recommender.py` | `Recommender`: `score()` (all GNN work, cached in a `SwapTable`) / `decide()` (no GNN) / `recommend()` / `from_settings()` |
| `build_cache.py` | CLI: builds the pool, co-occurrence matrix and harmony reference; checks freshness via a fingerprint |
| `run_recommend.py` | CLI: photo (runs M1) or outfit folder → printed recommendation + optional JSON |
| `conftest.py` | sys.path + `real` marker + shared session fixtures |
| `test_*.py` | One test file per unit (listed per task) |

Cache layout (`Module4/cache/`, gitignored): `pool/{meta.npz, proj.npy, files.json, outfit_keys.json,
cooc.npy, reference.json, fingerprint.json}`, `dup_calibration/*.png`, `runs/<photo id>/` (M1 output).

**Interfaces at a glance** (each task restates the ones it uses):

```python
# m4_settings
M4Settings.load(path=SETTINGS_PATH) -> M4Settings; .abs(rel) -> str; .with_(**kw) -> M4Settings; .provisional() -> list[str]
# dense_gnn
DenseGNN(model, device, vram_gb).project(x[N,555]) -> Tensor[N,128]
DenseGNN.logits(h0[B,n,128], hue[B,n] f64, formality[B,n] f64, area[B,n] f64, broad[B,n] long) -> Tensor[B]
DenseGNN.logits_chunked(make_batch(start, stop) -> (h0, hue, formality, area, broad), total, n, max_chunk=None) -> Tensor[total] (cpu)
# candidate_pool
CandidatePool.load(settings, device); .candidate_rows(types, mode="demo"|"eval", exclude=(source, outfit_key)|None) -> LongTensor
.cosine_to(rows, emb[512]) -> Tensor; .outfits() -> dict[(source_id, outfit_ord)] -> np.ndarray rows; .outfit_rows(source, key)
.crop_path(row) -> repo-relative str; .describe(row) -> dict; .source_name(row); .outfit_key(row); .type_name(row)
# recommender
Recommender.from_settings(settings=None, device=None, pool=None, gnn=None)
.score(state, mode="demo", exclude=None, same_type=None, trend_floor=None) -> SwapTable   # all GNN calls
.decide(table, w_t=None, rule=None) -> Recommendation                                 # zero GNN calls
.recommend(outfit_dir, mode="demo", photo_px=None, **score_kw) -> Recommendation
```

---

### Task 1: Scaffolding (settings, M2/M3 bridge, test harness)

**Files:**
- Create: `Module4/m4_settings.json`, `Module4/m4_settings.py`, `Module4/m2_bridge.py`, `Module4/conftest.py`
- Test: `Module4/test_settings.py`
- Modify: `.gitignore` (add `.pytest_cache/`)

**Interfaces:**
- Produces: `REPO`, `SETTINGS_PATH`, `StoreSpec(name, store_dir, output_root, split)`, `M4Settings` (fields below), `m2_bridge.{od, tg, ATTR_MASK, FEATURE_COLS, NODE_DIM, BROAD_VOCAB, BROAD_ID, TREND_MATCH_THRESHOLD, edge_weight_table(device), node_features(node), load_gnn(path, device), load_trend_model(path), pick_device(device=None), apply_vram_cap(vram_gb, device)}`; pytest fixtures `settings`, `device`.

- [ ] **Step 1: Install pytest** (not installed in this environment)

Run: `python -m pip install pytest`
Expected: `Successfully installed pytest-...`

- [ ] **Step 2: Write the settings file** `Module4/m4_settings.json`

```json
{
  "stores": [
    {
      "name": "fashion144k",
      "store_dir": "Fashion144k_v1/feature_store",
      "output_root": "Fashion144k_v1/outputs",
      "split": "Fashion144k_v1/split.mat"
    }
  ],
  "checkpoint": "Fashion144k_v1/ck_attr_subset_s42/clip_outfit_gnn_best.pt",
  "trend_model": "Module3/per_category_trend_model.pkl",
  "cache_dir": "Module4/cache",
  "w_h": 1.0,
  "w_t": 0.0,
  "rule": "topk_mean_gain",
  "topk": 5,
  "tau": {"max_gain": 0.0, "topk_mean_gain": 0.0, "slot_percentile": 0.5},
  "eps": 0.0,
  "t_dup": 0.97,
  "cooc_min": 20,
  "same_type": false,
  "trend_floor": false,
  "vram_gb": 6.0,
  "status": {"w_t": "provisional", "rule": "provisional", "tau": "provisional", "eps": "provisional", "t_dup": "provisional"}
}
```

- [ ] **Step 3: Write the failing test** `Module4/test_settings.py`

```python
import os

import pytest

from m4_settings import M4Settings


def test_default_settings_load_and_round_trip(tmp_path):
    s = M4Settings.load()
    assert s.rule in s.tau
    assert s.vram_gb == 6.0
    assert s.stores[0].name == "fashion144k"
    p = tmp_path / "s.json"
    s.save(str(p))
    assert M4Settings.load(str(p)) == s


def test_paths_are_repo_relative():
    s = M4Settings.load()
    rels = [s.checkpoint, s.trend_model, s.cache_dir]
    for st in s.stores:
        rels += [st.store_dir, st.output_root, st.split]
    for rel in rels:
        assert not os.path.isabs(rel), rel


def test_provisional_lists_uncalibrated_knobs():
    s = M4Settings.load()
    assert "tau" in s.provisional()
    s2 = s.with_(status={**s.status, "tau": "calibrated"})
    assert "tau" not in s2.provisional()
    assert s.status["tau"] == "provisional"  # with_ does not mutate


def test_bridge_constants():
    from m2_bridge import BROAD_VOCAB, FEATURE_COLS, NODE_DIM, od
    assert NODE_DIM == 555
    assert FEATURE_COLS[0] == 0 and FEATURE_COLS[543] == 543 and FEATURE_COLS[-1] == 569
    for pair in od.CATEGORY_EDGE_WEIGHTS:
        assert pair <= set(BROAD_VOCAB)


@pytest.mark.real
def test_checkpoint_loads_as_attr_subset(settings, device):
    from m2_bridge import load_gnn
    model = load_gnn(settings.abs(settings.checkpoint), device)
    assert model.input_proj[0].in_features == 555
```

- [ ] **Step 4: Run it to verify it fails**

Run: `python -m pytest Module4/test_settings.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'm4_settings'`

- [ ] **Step 5: Write `Module4/m4_settings.py`**

```python
"""Module 4 settings: every path and tunable knob, read from m4_settings.json.

Paths are repo-relative and resolved with M4Settings.abs(), so no machine-specific path is ever
stored. Knobs that the evaluation step calibrates (w_t, rule, tau, eps, t_dup) carry a status;
until it reads "calibrated" the value is provisional and every Recommendation says so.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SETTINGS_PATH = os.path.join(HERE, "m4_settings.json")


@dataclass(frozen=True)
class StoreSpec:
    name: str          # "fashion144k" | "pinterest"
    store_dir: str     # feature store (features.npy, node_scalars.npy, index.json)
    output_root: str   # holds <outfit_key>/<file>.png crops
    split: str         # split.mat (Fashion144k) or {outfit_key: "train"|"val"|"test"} JSON


@dataclass(frozen=True)
class M4Settings:
    stores: list
    checkpoint: str
    trend_model: str
    cache_dir: str
    w_h: float
    w_t: float
    rule: str
    topk: int
    tau: dict
    eps: float
    t_dup: float
    cooc_min: int
    same_type: bool
    trend_floor: bool
    vram_gb: float
    status: dict

    @classmethod
    def load(cls, path: str = SETTINGS_PATH) -> "M4Settings":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        d["stores"] = [StoreSpec(**s) for s in d["stores"]]
        return cls(**d)

    def save(self, path: str = SETTINGS_PATH) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
            f.write("\n")

    def abs(self, rel: str) -> str:
        return os.path.join(REPO, rel)

    def with_(self, **kw) -> "M4Settings":
        return replace(self, **kw)

    def provisional(self) -> list:
        return sorted(k for k, v in self.status.items() if v != "calibrated")
```

- [ ] **Step 6: Write `Module4/m2_bridge.py`**

```python
"""Module 4's single doorway into Module 2 and Module 3.

Module 2's files start with digits (02_outfit_dataset.py), so they can't be imported by name;
they are loaded here once, the same way Module2/demo_score.py does. Module 3's trend pickle needs
`trend_model` importable by bare name, so Module3 goes on sys.path.
"""
from __future__ import annotations

import os
import sys
import warnings
from importlib.util import module_from_spec, spec_from_file_location

import numpy as np
import torch

from m4_settings import REPO

_M2 = os.path.join(REPO, "Module2")
_M3 = os.path.join(REPO, "Module3")


def _load(fname: str, name: str):
    spec = spec_from_file_location(name, os.path.join(_M2, fname))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


od = _load("02_outfit_dataset.py", "outfit_dataset")
tg = _load("03_train_gnn.py", "train_gnn")

if _M3 not in sys.path:
    sys.path.insert(0, _M3)
import trend_model as _tm  # noqa: E402

# The shipped checkpoint is the attr-subset arm: [512 CLIP | 32 colour | 11 reliable attrs].
ATTR_MASK = od.attr_subset_indices()
FEATURE_COLS = np.r_[0:544, 544 + np.asarray(ATTR_MASK)]   # columns of a 570-D store row it reads
NODE_DIM = len(FEATURE_COLS)                               # 555

# Every broad category M2's edge table knows. The store only uses the first four.
BROAD_VOCAB = ("top", "bottom", "dress", "accessory", "outerwear", "shoe", "bag")
BROAD_ID = {b: i for i, b in enumerate(BROAD_VOCAB)}

TREND_MATCH_THRESHOLD = _tm.MATCH_CONFIDENCE_THRESHOLD


def edge_weight_table(device) -> torch.Tensor:
    """[V, V] float64 category edge weights, built from od.get_edge_weight (diagonal = 0.2)."""
    return torch.tensor([[od.get_edge_weight(a, b) for b in BROAD_VOCAB] for a in BROAD_VOCAB],
                        dtype=torch.float64, device=device)


def node_features(node: dict) -> np.ndarray:
    """The 555-D input the checkpoint reads, from an od node dict."""
    return np.concatenate([node["embedding"], node["colour"], node["attr_feat"][ATTR_MASK]]).astype(np.float32)


def load_gnn(checkpoint: str, device) -> torch.nn.Module:
    model = tg.CLIPOutfitGNN(attr_dim=len(ATTR_MASK), edge_dim=od.EDGE_ATTR_DIM).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model.eval()


def load_trend_model(path: str):
    return _tm.PerCategoryTrendModel.load(path)


def pick_device(device=None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    warnings.warn("CUDA not available: Module 4 runs on CPU (correct, but slow)")
    return torch.device("cpu")


def apply_vram_cap(vram_gb: float, device: torch.device) -> None:
    """Hard cap on this process's CUDA allocations (the GPU is shared)."""
    if device.type != "cuda":
        return
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, vram_gb * 2**30 / total), device)
```

- [ ] **Step 7: Write `Module4/conftest.py`**

```python
"""pytest setup for Module 4: import path, the `real` marker, shared session fixtures.

Tests marked `real` need the Fashion144k data and (from Task 4 on) a built cache:
    python Module4/build_cache.py
Quick unit run: python -m pytest Module4 -m "not real"
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def pytest_configure(config):
    config.addinivalue_line("markers", "real: needs the real data, a CUDA device and the built cache")


@pytest.fixture(scope="session")
def settings():
    from m4_settings import M4Settings
    return M4Settings.load()


@pytest.fixture(scope="session")
def device(settings):
    from m2_bridge import apply_vram_cap, pick_device
    d = pick_device()
    apply_vram_cap(settings.vram_gb, d)
    return d
```

- [ ] **Step 8: Add `.pytest_cache/` to `.gitignore`** under the `# ── Python ──` block (after `__pycache__/`):

```
.pytest_cache/
```

- [ ] **Step 9: Run the tests**

Run: `python -m pytest Module4/test_settings.py -v`
Expected: 5 passed

- [ ] **Step 10: Commit (ask the owner first)**

```bash
git add .gitignore Module4/m4_settings.json Module4/m4_settings.py Module4/m2_bridge.py Module4/conftest.py Module4/test_settings.py
git commit -m "feat(m4): settings, M2/M3 bridge and test harness"
```

---

### Task 2: Swap groups and co-occurrence filter (D4)

**Files:**
- Create: `Module4/swap_groups.py`
- Test: `Module4/test_swap_groups.py`

**Interfaces:**
- Produces: `TYPES` (16-tuple), `TYPE_ID`, `SWAP_GROUPS`, `BROAD_OF_TYPE`, `swap_group(type_) -> str | None`, `replacement_types(type_, same_type=False) -> tuple`, `CooccurrenceFilter(counts, min_count=20)` with `.from_outfits(iterable[list[str]], min_count)`, `.allowed(cand_type, other_types) -> bool`, `.save(path)`, `.load(path, min_count)`, `.counts` (int64 [16,16]).

- [ ] **Step 1: Write the failing test** `Module4/test_swap_groups.py`

```python
import numpy as np
import pytest

from swap_groups import (BROAD_OF_TYPE, SWAP_GROUPS, TYPES, CooccurrenceFilter, replacement_types,
                         swap_group)

STORE_TYPES = ["shoe", "pants", "bag", "jacket", "skirt", "dress", "shirt_blouse", "sweater", "leggings",
               "hat", "scarf", "top", "t_shirt", "belt", "sunglasses", "shorts"]


def test_every_store_type_has_exactly_one_group():
    assert sorted(STORE_TYPES) == sorted(TYPES)
    for t in STORE_TYPES:
        g = swap_group(t)
        assert g is not None and t in SWAP_GROUPS[g]
    assert sum(len(v) for v in SWAP_GROUPS.values()) == 16


def test_groups_match_spec():
    assert set(SWAP_GROUPS["upper"]) == {"t_shirt", "sweater", "shirt_blouse", "jacket", "top"}
    assert set(SWAP_GROUPS["lower"]) == {"pants", "leggings", "shorts", "skirt"}
    assert SWAP_GROUPS["dress"] == ("dress",)
    for acc in ["shoe", "bag", "hat", "belt", "scarf", "sunglasses"]:
        assert SWAP_GROUPS[acc] == (acc,)


def test_broad_of_type_matches_m1():
    assert {BROAD_OF_TYPE[t] for t in SWAP_GROUPS["upper"]} == {"top"}
    assert {BROAD_OF_TYPE[t] for t in SWAP_GROUPS["lower"]} == {"bottom"}
    assert BROAD_OF_TYPE["dress"] == "dress" and BROAD_OF_TYPE["shoe"] == "accessory"


def test_replacement_types_and_same_type():
    assert set(replacement_types("jacket")) == set(SWAP_GROUPS["upper"])
    assert replacement_types("jacket", same_type=True) == ("jacket",)
    assert replacement_types("hoodie") == ()


def _filter():
    outfits = ([["shirt_blouse", "skirt", "leggings", "shoe"]] * 30
               + [["t_shirt", "pants", "shoe"]] * 30
               + [["jacket", "skirt", "shoe"]] * 30
               + [["sweater", "shorts", "shoe"]] * 5)
    return CooccurrenceFilter.from_outfits(outfits, min_count=20)


def test_counts_are_outfits_containing_both():
    f = _filter()
    from swap_groups import TYPE_ID
    assert f.counts[TYPE_ID["skirt"], TYPE_ID["leggings"]] == 30
    assert f.counts[TYPE_ID["skirt"], TYPE_ID["shoe"]] == 60
    assert f.counts[TYPE_ID["pants"], TYPE_ID["leggings"]] == 0
    assert np.all(np.diag(f.counts) == 0)          # no outfit wears two of one type
    assert np.array_equal(f.counts, f.counts.T)


def test_skirt_plus_leggings_never_swaps_skirt_to_pants():
    f = _filter()
    others = ["shirt_blouse", "leggings", "shoe"]   # swapping the skirt
    allowed = [t for t in replacement_types("skirt") if f.allowed(t, others)]
    assert "pants" not in allowed
    assert "leggings" not in allowed                # already worn: would duplicate the type
    assert allowed == ["skirt"]


def test_rare_pairs_are_blocked_below_min_count():
    f = _filter()
    assert not f.allowed("shorts", ["sweater"])    # 5 < 20
    assert f.allowed("pants", ["t_shirt", "shoe"])


def test_unknown_other_types_are_ignored():
    assert _filter().allowed("skirt", ["shirt_blouse", "hoodie"])


def test_save_load_round_trip(tmp_path):
    f = _filter()
    p = str(tmp_path / "cooc.npy")
    f.save(p)
    g = CooccurrenceFilter.load(p, min_count=20)
    assert np.array_equal(f.counts, g.counts) and g.min_count == 20


@pytest.mark.real
def test_real_counts_from_cache(settings):
    import os
    from candidate_pool import pool_dir
    from swap_groups import TYPE_ID
    c = CooccurrenceFilter.load(os.path.join(pool_dir(settings), "cooc.npy")).counts
    assert c[TYPE_ID["pants"], TYPE_ID["leggings"]] == 0
    assert c[TYPE_ID["skirt"], TYPE_ID["leggings"]] >= 20
    upper = [TYPE_ID[t] for t in SWAP_GROUPS["upper"]]
    assert c[np.ix_(upper, upper)].sum() == 0
```

The last test needs the cache from Task 6. Until then it fails with `FileNotFoundError` (or `ModuleNotFoundError: candidate_pool`), which is expected.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest Module4/test_swap_groups.py -v -m "not real"`
Expected: `ModuleNotFoundError: No module named 'swap_groups'`

- [ ] **Step 3: Write `Module4/swap_groups.py`**

```python
"""Swap groups and the co-occurrence filter (D4): what may replace what.

Groups follow M1's segmentation slots: M1 keeps one garment per SegFormer region, so an outfit has
at most one upper-body piece, and each accessory type appears at most once. The filter is learned
from train outfits: a candidate type is allowed only if it co-occurs with every remaining piece's
type in at least `min_count` outfits. A type already worn by another piece has count 0 (no outfit
wears two of one type), so it is blocked too.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

SWAP_GROUPS = {
    "upper": ("t_shirt", "sweater", "shirt_blouse", "jacket", "top"),
    "lower": ("pants", "leggings", "shorts", "skirt"),
    "dress": ("dress",),
    "shoe": ("shoe",),
    "bag": ("bag",),
    "hat": ("hat",),
    "belt": ("belt",),
    "scarf": ("scarf",),
    "sunglasses": ("sunglasses",),
}
TYPES = tuple(t for ts in SWAP_GROUPS.values() for t in ts)
TYPE_ID = {t: i for i, t in enumerate(TYPES)}
_GROUP_OF = {t: g for g, ts in SWAP_GROUPS.items() for t in ts}

# M1's specific -> broad mapping (fashion_segmenter.py, SegFormer path).
BROAD_OF_TYPE = {
    **{t: "top" for t in SWAP_GROUPS["upper"]},
    **{t: "bottom" for t in SWAP_GROUPS["lower"]},
    "dress": "dress",
    **{t: "accessory" for t in ("shoe", "bag", "hat", "belt", "scarf", "sunglasses")},
}


def swap_group(type_: str):
    return _GROUP_OF.get(type_)


def replacement_types(type_: str, same_type: bool = False) -> tuple:
    g = swap_group(type_)
    if g is None:
        return ()
    return (type_,) if same_type else SWAP_GROUPS[g]


class CooccurrenceFilter:
    def __init__(self, counts, min_count: int = 20):
        self.counts = np.asarray(counts, dtype=np.int64)
        self.min_count = int(min_count)

    @classmethod
    def from_outfits(cls, outfits: Iterable[list], min_count: int = 20) -> "CooccurrenceFilter":
        """outfits: one list of specific types per (train) outfit. counts[a, b] counts pairs of
        distinct pieces, i.e. outfits containing both a and b (no outfit repeats a type)."""
        c = np.zeros((len(TYPES), len(TYPES)), dtype=np.int64)
        for types in outfits:
            ids = [TYPE_ID[t] for t in types if t in TYPE_ID]
            for x in range(len(ids)):
                for y in range(x + 1, len(ids)):
                    c[ids[x], ids[y]] += 1
                    c[ids[y], ids[x]] += 1
        return cls(c, min_count)

    def allowed(self, cand_type: str, other_types) -> bool:
        a = TYPE_ID[cand_type]
        return all(self.counts[a, TYPE_ID[t]] >= self.min_count for t in other_types if t in TYPE_ID)

    def save(self, path: str) -> None:
        np.save(path, self.counts)

    @classmethod
    def load(cls, path: str, min_count: int = 20) -> "CooccurrenceFilter":
        return cls(np.load(path), min_count)
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest Module4/test_swap_groups.py -v -m "not real"`
Expected: 9 passed, 1 deselected

- [ ] **Step 5: Commit (ask the owner first)**

```bash
git add Module4/swap_groups.py Module4/test_swap_groups.py
git commit -m "feat(m4): D4 swap groups and co-occurrence filter"
```

---

### Task 3: Dense GNN scorer with parity tests (D5 core)

**Files:**
- Create: `Module4/dense_gnn.py`
- Test: `Module4/test_dense_gnn.py`
- Modify: `Module4/conftest.py` (add `gnn` fixture)

**Interfaces:**
- Consumes: `m2_bridge.{edge_weight_table, od, ATTR_MASK, FEATURE_COLS, BROAD_ID, load_gnn}`.
- Produces: `edge_tensor(hue, formality, area, broad, wtable) -> float32 [B,n,n,5]` (indexed [batch, src, dst]); `DenseGNN(model, device, vram_gb=6.0)` with `.model`, `.device`, `.wtable`, `.project(x) -> [N,128]`, `.logits(h0, hue, formality, area, broad) -> [B]`, `.chunk_size(n) -> int`, `.logits_chunked(make_batch, total, n, max_chunk=None) -> cpu float32 [total]`.

**Why the dense form is exact:** `build_graph` makes complete directed graphs (every i≠j). In
`EdgeAwareGNNLayer`, `agg_d = Σ_s w_sd (h_s + W e_sd + b) / Σ_s w_sd`. The message is linear in `e`, so
`Σ_s w_sd W e_sd = W (Σ_s w_sd e_sd)`. That avoids ever building a `[B, n, n, 128]` tensor. Edge features
are computed in float64 and then cast, exactly as NumPy does in `od.edge_features`, so they come out
bit-identical. Mean pooling matches the model's batched path.

- [ ] **Step 1: Add the `gnn` fixture** to `Module4/conftest.py` (append):

```python
@pytest.fixture(scope="session")
def gnn(settings, device):
    from dense_gnn import DenseGNN
    from m2_bridge import load_gnn
    return DenseGNN(load_gnn(settings.abs(settings.checkpoint), device), device, settings.vram_gb)
```

- [ ] **Step 2: Write the failing test** `Module4/test_dense_gnn.py`

```python
import json
import os
import random

import numpy as np
import pytest
import torch

from dense_gnn import edge_tensor
from m2_bridge import ATTR_MASK, BROAD_ID, FEATURE_COLS, edge_weight_table, od


def test_edge_tensor_matches_edge_features_including_unknown_hue():
    nodes = [
        {"category": "top", "hue": 350.0, "formality": 70.0, "area": 9000.0},
        {"category": "bottom", "hue": None, "formality": 20.0, "area": 7000.0},
        {"category": "accessory", "hue": 15.0, "formality": 50.0, "area": 500.0},
        {"category": "accessory", "hue": 200.0, "formality": 55.0, "area": 0.0},
    ]
    hue = torch.tensor([[np.nan if n["hue"] is None else n["hue"] for n in nodes]], dtype=torch.float64)
    form = torch.tensor([[n["formality"] for n in nodes]], dtype=torch.float64)
    area = torch.tensor([[n["area"] for n in nodes]], dtype=torch.float64)
    broad = torch.tensor([[BROAD_ID[n["category"]] for n in nodes]])
    E = edge_tensor(hue, form, area, broad, edge_weight_table("cpu"))
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if i != j:
                assert torch.equal(E[0, i, j], torch.from_numpy(od.edge_features(a, b))), (i, j)


# ---------------- real data ----------------

@pytest.fixture(scope="module")
def store(settings):
    d = settings.abs(settings.stores[0].store_dir)
    feat = np.load(os.path.join(d, "features.npy"), mmap_mode="r")
    scal = np.load(os.path.join(d, "node_scalars.npy"))
    with open(os.path.join(d, "index.json"), encoding="utf-8") as f:
        index = json.load(f)
    broad = np.empty(feat.shape[0], dtype=object)
    for es in index["outfits"].values():
        for e in es:
            broad[e["row"]] = e["broad_category"]
    outfits = [[e["row"] for e in es] for es in index["outfits"].values() if es]
    return feat, scal, broad, outfits


def store_node(store, r):
    feat, scal, broad, _ = store
    hue = float(scal[r, 0])
    return {"embedding": np.asarray(feat[r, :512]), "colour": np.asarray(feat[r, 512:544]),
            "attr_feat": np.asarray(feat[r, 544:]), "category": broad[r],
            "hue": None if np.isnan(hue) else hue, "formality": float(scal[r, 1]), "area": float(scal[r, 2])}


def dense_inputs(gnn, store, rows_2d):
    feat, scal, broad, _ = store
    R = np.asarray(rows_2d)
    flat = R.reshape(-1)
    x = torch.from_numpy(np.asarray(feat[flat])[:, FEATURE_COLS].astype(np.float32))
    h0 = gnn.project(x).reshape(*R.shape, -1)
    s = torch.from_numpy(scal[flat].astype(np.float64)).to(gnn.device).reshape(*R.shape, 3)
    b = torch.tensor([BROAD_ID[broad[r]] for r in flat], device=gnn.device).reshape(R.shape)
    return h0, s[..., 0], s[..., 1], s[..., 2], b


def model_logit(gnn, nodes):
    g = od.OutfitDataset.build_graph(nodes, 0.0, attr_mask=ATTR_MASK)
    dev = gnn.device
    with torch.no_grad():
        out, _ = gnn.model(g["x"].to(dev), g["edge_index"].to(dev), g["edge_attr"].to(dev))
    return float(out), g


@pytest.mark.real
def test_feature_cols_reproduce_build_graph_node_input(store):
    rows = random.Random(0).sample(range(store[0].shape[0]), 50)
    g = od.OutfitDataset.build_graph([store_node(store, r) for r in rows], 0.0, attr_mask=ATTR_MASK)
    assert torch.equal(g["x"], torch.from_numpy(np.asarray(store[0][rows])[:, FEATURE_COLS]))


@pytest.mark.real
def test_edges_and_logits_match_build_graph_on_1000_random_swaps(gnn, store):
    rng = random.Random(0)
    outfits = [o for o in store[3] if len(o) >= 2]
    N = store[0].shape[0]
    worst = 0.0
    for _ in range(1000):
        rows = list(rng.choice(outfits))
        rows[rng.randrange(len(rows))] = rng.randrange(N)
        ref, g = model_logit(gnn, [store_node(store, r) for r in rows])
        h0, hue, form, area, broad = dense_inputs(gnn, store, [rows])
        E = edge_tensor(hue, form, area, broad, gnn.wtable)
        src, dst = g["edge_index"]
        assert torch.equal(E[0, src, dst].cpu(), g["edge_attr"]), "edge features differ from build_graph"
        worst = max(worst, abs(float(gnn.logits(h0, hue, form, area, broad)[0]) - ref))
    assert worst < 1e-4, f"max |logit diff| = {worst:.2e}"


@pytest.mark.real
def test_single_garment_outfits_match(gnn, store):
    singles = [o for o in store[3] if len(o) == 1]
    for rows in random.Random(1).sample(singles, 200):
        ref, _ = model_logit(gnn, [store_node(store, rows[0])])
        assert abs(float(gnn.logits(*dense_inputs(gnn, store, [rows]))[0]) - ref) < 1e-4


@pytest.mark.real
def test_chunked_equals_unchunked_even_with_tiny_chunks(gnn, store):
    fives = [o for o in store[3] if len(o) == 5][:300]
    batch = dense_inputs(gnn, store, fives)
    full = gnn.logits(*batch).cpu()
    chunked = gnn.logits_chunked(lambda a, b: tuple(t[a:b] for t in batch), len(fives), 5, max_chunk=7)
    assert torch.allclose(full, chunked, atol=1e-6)
```

- [ ] **Step 3: Run to verify failure**

Run: `python -m pytest Module4/test_dense_gnn.py -v`
Expected: `ModuleNotFoundError: No module named 'dense_gnn'`

- [ ] **Step 4: Write `Module4/dense_gnn.py`**

```python
"""Dense, broadcast re-implementation of CLIPOutfitGNN scoring for many same-size outfits.

Module 2 scores one outfit at a time through build_graph (a Python loop over edges). Module 4 needs
hundreds of thousands of what-if outfits per photo, so this file computes the same logits with
batched tensor ops:

- every outfit in a batch has the same number of nodes n, and every graph is complete (all i != j
  edges, as build_graph makes them), so edges live in a dense [B, n, n, 5] tensor indexed
  [batch, src, dst];
- the message is linear in the edge features, so sum_s w_sd * msg_edge(e_sd) is computed as
  (sum_s w_sd * e_sd) @ W.T + b * sum_s w_sd, which never builds a [B, n, n, 128] tensor;
- the input projection does not depend on the graph, so callers pass projected node states h0
  (CandidatePool.proj holds them for the whole pool).

Parity with model(build_graph(...)) is enforced by test_dense_gnn.py.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from m2_bridge import edge_weight_table


def edge_tensor(hue, formality, area, broad, wtable) -> torch.Tensor:
    """Dense 5-D edge features matching od.edge_features exactly.

    hue / formality / area: float64 [B, n] (hue NaN = unknown); broad: long [B, n] (BROAD_ID);
    wtable: float64 [V, V] from edge_weight_table(). Returns float32 [B, n(src), n(dst), 5] =
    [category weight, hue separation, formality difference, volume contrast, same category]."""
    d = (hue[:, :, None] - hue[:, None, :]).abs() % 360.0
    d = torch.where(d <= 180.0, d, 360.0 - d)
    hue_sep = torch.nan_to_num(d / 180.0, nan=0.0)
    form_diff = (formality[:, :, None] - formality[:, None, :]).abs() / 100.0
    vol = torch.tanh(torch.log((area[:, :, None] + 1.0) / (area[:, None, :] + 1.0)))
    w = wtable[broad[:, :, None], broad[:, None, :]]
    same = (broad[:, :, None] == broad[:, None, :]).to(torch.float64)
    return torch.stack([w, hue_sep, form_diff, vol, same], dim=-1).to(torch.float32)


def _layer(layer, h, E, offdiag):
    w = F.softplus(layer.gate(E)).squeeze(-1) * offdiag            # [B, src, dst], 0 on the diagonal
    wsum = w.sum(dim=1)                                             # [B, dst]
    agg = torch.einsum("bsd,bsk->bdk", w, h)
    agg = agg + torch.einsum("bsd,bsdc->bdc", w, E) @ layer.msg_edge.weight.T
    agg = agg + wsum[..., None] * layer.msg_edge.bias
    agg = agg / wsum.clamp(min=1e-6)[..., None]
    return F.relu(layer.norm(layer.linear(h + agg)))


class DenseGNN:
    def __init__(self, model, device, vram_gb: float = 6.0):
        self.model = model.eval()
        self.device = torch.device(device)
        self.wtable = edge_weight_table(self.device)
        self.cap_bytes = vram_gb * 2**30

    @torch.no_grad()
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """[N, 555] node inputs -> [N, 128] (the model's input_proj)."""
        return self.model.input_proj(x.to(self.device, torch.float32))

    @torch.no_grad()
    def logits(self, h0, hue, formality, area, broad) -> torch.Tensor:
        n = h0.shape[1]
        E = edge_tensor(hue, formality, area, broad, self.wtable)
        offdiag = ~torch.eye(n, dtype=torch.bool, device=h0.device)
        h = _layer(self.model.gnn1, h0, E, offdiag)
        h = _layer(self.model.gnn2, h, E, offdiag)
        return self.model.outfit_score(h.mean(dim=1)).squeeze(-1)

    def chunk_size(self, n: int) -> int:
        """Outfits per batch: half the VRAM left under the cap, at a conservative bytes/outfit."""
        per_outfit = n * 128 * 4 * 16 + n * n * 5 * 8 * 4
        if self.device.type == "cuda":
            budget = self.cap_bytes - torch.cuda.memory_allocated(self.device)
        else:
            budget = 2 * 2**30
        return int(max(1024, min(262144, 0.5 * budget // per_outfit)))

    def logits_chunked(self, make_batch, total: int, n: int, max_chunk=None) -> torch.Tensor:
        """make_batch(start, stop) -> (h0, hue, formality, area, broad) for outfits [start, stop).
        On CUDA OOM the chunk is halved and retried."""
        size = self.chunk_size(n) if max_chunk is None else max_chunk
        out, start = [], 0
        while start < total:
            stop = min(total, start + size)
            try:
                out.append(self.logits(*make_batch(start, stop)).float().cpu())
                start = stop
            except torch.cuda.OutOfMemoryError:
                if size == 1:
                    raise
                torch.cuda.empty_cache()
                size //= 2
        return torch.cat(out) if out else torch.zeros(0)
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest Module4/test_dense_gnn.py -v`
Expected: 5 passed (the 1,000-swap test takes roughly 10–30 s)

- [ ] **Step 6: Commit (ask the owner first)**

```bash
git add Module4/dense_gnn.py Module4/test_dense_gnn.py Module4/conftest.py
git commit -m "feat(m4): dense batched GNN scorer with build_graph parity tests"
```

---

### Task 4: Candidate pool: build, cache, load (D1 Fashion144k, D5 projection, D3/D8c trend)

**Files:**
- Create: `Module4/candidate_pool.py`, `Module4/build_cache.py`
- Test: `Module4/test_candidate_pool.py`
- Modify: `Module4/conftest.py` (add `pool` fixture)

**Interfaces:**
- Consumes: `M4Settings`, `DenseGNN.project`, `m2_bridge.{od, FEATURE_COLS, BROAD_ID, BROAD_VOCAB, TREND_MATCH_THRESHOLD, load_trend_model}`, `swap_groups.{TYPES, TYPE_ID}`.
- Produces: `SPLIT_CODES`, `pool_dir(settings)`, `pool_fingerprint(settings)`, `is_fresh(settings)`, `load_store_split(settings, spec) -> dict[str, str]`, `load_store_arrays(settings) -> (emb [N,512] f32, scal [N,3] f32)`, `trend_scores(emb, broad_names, trend_model) -> (score f32 [N] NaN=unscored, cluster int16 [N] -1=unmatched)`, `build_pool(settings, gnn, trend_model) -> None`, and `CandidatePool` with:
  - numpy: `source, split, type, broad, store_row, outfit_ord, trend, cluster, rep_demo, rep_eval`, `files`, `outfit_keys`
  - device tensors: `proj [N,128]`, `emb [N,512]` (L2-normalised), `hue, formality, area` (float64 [N]), `broad_t, type_t` (long), `trend_t` (float32)
  - methods: `n`, `.load(settings, device)`, `.candidate_rows(types, mode, exclude)`, `.cosine_to(rows, v)`, `.outfits()`, `.outfit_rows(source, key)`, `.source_id(name)`, `.source_name(row)`, `.outfit_key(row)`, `.type_name(row)`, `.crop_path(row)`, `.describe(row)`, `.empty_rows()`

Global row = store offset + store row, so CLIP embeddings and node scalars are read straight from the
store memmaps at load time and aren't duplicated in the cache. This task writes **identity** dedupe
arrays; Task 5 replaces them with real dedupe.

- [ ] **Step 1: Add the `pool` fixture** to `Module4/conftest.py` (append):

```python
@pytest.fixture(scope="session")
def pool(settings, device):
    from candidate_pool import CandidatePool
    return CandidatePool.load(settings, device)
```

- [ ] **Step 2: Write the failing test** `Module4/test_candidate_pool.py`

```python
import os
import random

import numpy as np
import pytest
import torch

from candidate_pool import SPLIT_CODES, load_store_arrays
from m2_bridge import BROAD_VOCAB, FEATURE_COLS, TREND_MATCH_THRESHOLD, load_trend_model
from swap_groups import TYPE_ID, TYPES

pytestmark = pytest.mark.real

SPEC_TEST_COUNTS = {"shoe": 28329, "pants": 19169, "bag": 17130, "jacket": 15025, "skirt": 11701,
                    "dress": 11246, "shirt_blouse": 8436, "sweater": 8015, "leggings": 7268, "hat": 6643,
                    "scarf": 3163, "top": 3131, "t_shirt": 2726, "belt": 1405, "sunglasses": 548, "shorts": 235}


def test_every_store_row_is_in_the_pool_once(pool):
    assert pool.n == 480079
    assert (pool.outfit_ord >= 0).all() and all(f for f in pool.files)
    assert (pool.type >= 0).all()


def test_test_split_counts_match_spec(pool):
    test = pool.split == SPLIT_CODES["test"]
    got = {t: int(((pool.type == TYPE_ID[t]) & test).sum()) for t in TYPES}
    assert got == SPEC_TEST_COUNTS


def test_projection_cache_matches_model(pool, gnn, settings):
    feat = np.load(settings.abs(os.path.join(settings.stores[0].store_dir, "features.npy")), mmap_mode="r")
    rows = np.sort(np.random.default_rng(0).choice(pool.n, 2000, replace=False))
    want = gnn.project(torch.from_numpy(np.asarray(feat[rows])[:, FEATURE_COLS]))
    assert torch.allclose(pool.proj[torch.as_tensor(rows, device=pool.device)], want, atol=1e-5)


def test_trend_scores_match_score_embedding(pool, settings):
    tm = load_trend_model(settings.abs(settings.trend_model))
    emb, _ = load_store_arrays(settings)
    rows = np.random.default_rng(1).choice(pool.n, 2000, replace=False)
    for r in rows:
        cat = BROAD_VOCAB[pool.broad[r]]
        res = tm.score_embedding(emb[r], cat)
        want = np.nan if res["trendiness_score"] is None else res["trendiness_score"]
        if (np.isnan(want) and np.isnan(pool.trend[r])) or np.isclose(want, pool.trend[r], rtol=0, atol=1e-4):
            continue                      # (pool stores float32; the model returns float64)
        # Only float near-ties may disagree: best-vs-second cluster or best-vs-threshold within 1e-5.
        m = tm.models[cat]
        C = np.stack([c.centroid for c in m.clusters.values()])
        e = emb[r] / (np.linalg.norm(emb[r]) + 1e-8)
        s = np.sort(C @ e)
        assert s[-1] - s[-2] < 1e-5 or abs(s[-1] - TREND_MATCH_THRESHOLD) < 1e-5, (r, want, pool.trend[r])


def test_crop_paths_exist(pool, settings):
    for r in random.Random(0).sample(range(pool.n), 100):
        assert os.path.exists(settings.abs(pool.crop_path(r))), pool.crop_path(r)


def test_candidate_rows_respect_type_and_mode(pool):
    rows = pool.candidate_rows(["sunglasses"], mode="demo").cpu().numpy()
    assert (pool.type[rows] == TYPE_ID["sunglasses"]).all()
    ev = pool.candidate_rows(["sunglasses"], mode="eval").cpu().numpy()
    assert (pool.split[ev] == SPLIT_CODES["test"]).all() and len(ev) < len(rows)


def test_exclude_drops_the_source_outfit(pool):
    key = pool.outfit_keys["fashion144k"][int(pool.outfit_ord[0])]
    own = set(pool.outfit_rows("fashion144k", key).tolist())
    t = TYPES[pool.type[0]]
    rows = set(pool.candidate_rows([t], mode="demo", exclude=("fashion144k", key)).cpu().numpy().tolist())
    assert not (own & rows)
```

- [ ] **Step 3: Run to verify failure**

Run: `python -m pytest Module4/test_candidate_pool.py -v`
Expected: `ModuleNotFoundError: No module named 'candidate_pool'`

- [ ] **Step 4: Write `Module4/candidate_pool.py`**

```python
"""Candidate pool (D1, D5, D8a): every garment in the configured feature stores, with what Module 4
needs per row, built once and cached under <cache_dir>/pool/.

Per row: source, split, specific type, broad category, outfit, file, the GNN input projection
(precomputed, D5), M3 trend score + cluster (vectorised, D3/D8c), and near-duplicate
representatives for the demo pool and for the eval pool (D8a). Global row = store offset + store
row, so CLIP embeddings and node scalars are read straight from the stores at load time.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from m2_bridge import BROAD_ID, BROAD_VOCAB, FEATURE_COLS, TREND_MATCH_THRESHOLD, od
from swap_groups import TYPE_ID, TYPES

POOL_FORMAT = 1
SPLIT_CODES = {"train": 0, "val": 1, "test": 2, "none": 3}
_CHUNK = 65536


def pool_dir(settings) -> str:
    return settings.abs(os.path.join(settings.cache_dir, "pool"))


def _sha16(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def pool_fingerprint(settings) -> dict:
    """Everything the cached pool depends on. A mismatch means: rebuild."""
    stores = []
    for s in settings.stores:
        feat = np.load(settings.abs(os.path.join(s.store_dir, "features.npy")), mmap_mode="r")
        stores.append({"name": s.name, "rows": int(feat.shape[0]), "split": _sha16(settings.abs(s.split))})
    return {"format": POOL_FORMAT, "checkpoint": _sha16(settings.abs(settings.checkpoint)),
            "trend_model": _sha16(settings.abs(settings.trend_model)), "t_dup": settings.t_dup,
            "stores": stores}


def is_fresh(settings) -> bool:
    p = os.path.join(pool_dir(settings), "fingerprint.json")
    if not os.path.exists(p):
        return False
    with open(p, encoding="utf-8") as f:
        return json.load(f) == pool_fingerprint(settings)


def load_store_split(settings, spec) -> dict:
    """outfit_key -> "train" | "val" | "test". Keys missing from the split count as "none"."""
    path = settings.abs(spec.split)
    if path.endswith(".mat"):
        s = od.load_split(path)
        return {str(int(i)): name for name, ids in s.items() for i in ids}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_store_arrays(settings):
    """CLIP embeddings [N, 512] and node scalars [N, 3] (hue|NaN, formality, area), pool row order."""
    embs, scals = [], []
    for s in settings.stores:
        d = settings.abs(s.store_dir)
        feat = np.load(os.path.join(d, "features.npy"), mmap_mode="r")
        embs.append(np.ascontiguousarray(feat[:, :512]))
        scals.append(np.load(os.path.join(d, "node_scalars.npy")))
    return np.concatenate(embs), np.concatenate(scals)


def _outfit_sort_key(k: str):
    return (0, int(k), "") if k.isdigit() else (1, 0, k)


def trend_scores(emb: np.ndarray, broad_names: np.ndarray, trend_model):
    """Vectorised PerCategoryTrendModel.score_embedding: one matmul per broad category.
    Returns (score float32 [N], NaN = no model / no confident match; cluster int16 [N], -1 = none)."""
    n = emb.shape[0]
    score = np.full(n, np.nan, np.float32)
    cluster = np.full(n, -1, np.int16)
    for cat, tm in trend_model.models.items():
        rows = np.nonzero(broad_names == cat)[0]
        if rows.size == 0:
            continue
        ids = np.asarray(list(tm.clusters.keys()))
        C = np.stack([tm.clusters[c].centroid for c in ids]).astype(np.float32)
        ts = np.array([tm.clusters[c].trendiness_score for c in ids], np.float32)
        for s in range(0, rows.size, _CHUNK):
            r = rows[s:s + _CHUNK]
            e = emb[r]
            e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
            sims = e @ C.T
            best = sims.argmax(axis=1)
            ok = sims[np.arange(r.size), best] >= TREND_MATCH_THRESHOLD
            score[r[ok]] = ts[best[ok]]
            cluster[r[ok]] = ids[best[ok]]
    return score, cluster


def build_pool(settings, gnn, trend_model) -> None:
    """Build and write <cache_dir>/pool/ (everything except cooc.npy, reference.json, fingerprint)."""
    out = pool_dir(settings)
    os.makedirs(out, exist_ok=True)
    parts, files, outfit_keys, proj = [], [], {}, []
    for sid, spec in enumerate(settings.stores):
        d = settings.abs(spec.store_dir)
        with open(os.path.join(d, "index.json"), encoding="utf-8") as f:
            index = json.load(f)
        n = int(index["num_rows"])
        splits = load_store_split(settings, spec)
        keys = sorted(index["outfits"], key=_outfit_sort_key)
        outfit_keys[spec.name] = keys
        type_ = np.full(n, -1, np.int8)
        broad = np.full(n, -1, np.int8)
        split = np.full(n, SPLIT_CODES["none"], np.uint8)
        ordv = np.full(n, -1, np.int32)
        f_ = [None] * n
        for o, key in enumerate(keys):
            sp = SPLIT_CODES[splits.get(key, "none")]
            for e in index["outfits"][key]:
                r = e["row"]
                type_[r] = TYPE_ID.get(e["category"], -1)
                broad[r] = BROAD_ID[e["broad_category"]]
                split[r] = sp
                ordv[r] = o
                f_[r] = e["file"]
        assert (ordv >= 0).all(), f"{spec.name}: some store rows are not listed in index.json"
        files.extend(f_)
        parts.append({"source": np.full(n, sid, np.uint8), "split": split, "type": type_, "broad": broad,
                      "store_row": np.arange(n, dtype=np.int32), "outfit_ord": ordv})
        feat = np.load(os.path.join(d, "features.npy"), mmap_mode="r")
        for s in range(0, n, _CHUNK):
            x = torch.from_numpy(np.ascontiguousarray(feat[s:s + _CHUNK])[:, FEATURE_COLS])
            proj.append(gnn.project(x).cpu().numpy())
    meta = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    emb, scal = load_store_arrays(settings)
    meta["trend"], meta["cluster"] = trend_scores(emb, np.asarray(BROAD_VOCAB)[meta["broad"]], trend_model)
    n_all = len(files)
    meta["rep_demo"] = np.arange(n_all, dtype=np.int32)                                   # Task 5: dedupe
    meta["rep_eval"] = np.where(meta["split"] == SPLIT_CODES["test"], np.arange(n_all), -1).astype(np.int32)
    np.savez(os.path.join(out, "meta.npz"), **meta)
    np.save(os.path.join(out, "proj.npy"), np.concatenate(proj))
    with open(os.path.join(out, "files.json"), "w", encoding="utf-8") as f:
        json.dump(files, f)
    with open(os.path.join(out, "outfit_keys.json"), "w", encoding="utf-8") as f:
        json.dump(outfit_keys, f)


class CandidatePool:
    def __init__(self, *, stores, proj, emb, scalars, meta, files, outfit_keys, device):
        self.stores = list(stores)
        self.device = torch.device(device)
        self.n = len(files)
        self.files = files
        self.outfit_keys = outfit_keys
        for k in ("source", "split", "type", "broad", "store_row", "outfit_ord", "trend", "cluster",
                  "rep_demo", "rep_eval"):
            setattr(self, k, np.asarray(meta[k]))
        dev = self.device
        self.proj = torch.as_tensor(np.asarray(proj, np.float32), device=dev)
        e = torch.as_tensor(np.asarray(emb, np.float32), device=dev)
        self.emb = e / e.norm(dim=1, keepdim=True).clamp(min=1e-8)
        s = torch.as_tensor(np.asarray(scalars), dtype=torch.float64, device=dev)
        self.hue, self.formality, self.area = s[:, 0].contiguous(), s[:, 1].contiguous(), s[:, 2].contiguous()
        self.broad_t = torch.as_tensor(self.broad.astype(np.int64), device=dev)
        self.type_t = torch.as_tensor(self.type.astype(np.int64), device=dev)
        self.trend_t = torch.as_tensor(self.trend.astype(np.float32), device=dev)
        idx = np.arange(self.n)
        self._is_rep = {"demo": torch.as_tensor(self.rep_demo == idx, device=dev),
                        "eval": torch.as_tensor(self.rep_eval == idx, device=dev)}
        self._key_ord = {name: {k: i for i, k in enumerate(keys)} for name, keys in outfit_keys.items()}
        self._outfits = None

    @classmethod
    def load(cls, settings, device) -> "CandidatePool":
        d = pool_dir(settings)
        if not os.path.exists(os.path.join(d, "fingerprint.json")):
            raise FileNotFoundError(f"No Module 4 cache at {d}: run  python Module4/build_cache.py")
        if not is_fresh(settings):
            raise RuntimeError("Module 4 cache is stale (checkpoint, trend model, stores or t_dup changed): "
                               "run  python Module4/build_cache.py --force")
        emb, scal = load_store_arrays(settings)
        with open(os.path.join(d, "files.json"), encoding="utf-8") as f:
            files = json.load(f)
        with open(os.path.join(d, "outfit_keys.json"), encoding="utf-8") as f:
            keys = json.load(f)
        return cls(stores=settings.stores, proj=np.load(os.path.join(d, "proj.npy")), emb=emb, scalars=scal,
                   meta=dict(np.load(os.path.join(d, "meta.npz"))), files=files, outfit_keys=keys,
                   device=device)

    # ---- queries ----
    def empty_rows(self) -> torch.Tensor:
        return torch.zeros(0, dtype=torch.long, device=self.device)

    def candidate_rows(self, types, mode: str = "demo", exclude=None) -> torch.Tensor:
        """Representative rows (D8a) of the given specific types. mode "eval" = test rows only.
        exclude = (source name, outfit key): drop that outfit's own garments."""
        if not types:
            return self.empty_rows()
        tid = torch.tensor([TYPE_ID[t] for t in types], device=self.device)
        mask = torch.isin(self.type_t, tid) & self._is_rep[mode]
        if exclude is not None:
            own = self.outfit_rows(*exclude)
            mask[torch.as_tensor(own, device=self.device)] = False
        return torch.nonzero(mask).squeeze(1)

    def cosine_to(self, rows: torch.Tensor, v) -> torch.Tensor:
        v = torch.as_tensor(np.asarray(v, np.float32), device=self.device)
        return self.emb[rows] @ (v / v.norm().clamp(min=1e-8))

    def outfits(self) -> dict:
        """{(source_id, outfit_ord): np.ndarray of pool rows}."""
        if self._outfits is None:
            key = (self.source.astype(np.int64) << 32) | self.outfit_ord.astype(np.int64)
            order = np.argsort(key, kind="stable")
            cuts = np.flatnonzero(np.diff(key[order])) + 1
            self._outfits = {(int(self.source[g[0]]), int(self.outfit_ord[g[0]])): g
                             for g in np.split(order, cuts)}
        return self._outfits

    def source_id(self, name: str) -> int:
        return [s.name for s in self.stores].index(name)

    def outfit_rows(self, source: str, key: str) -> np.ndarray:
        return self.outfits()[(self.source_id(source), self._key_ord[source][key])]

    # ---- per-row metadata ----
    def source_name(self, row: int) -> str:
        return self.stores[int(self.source[row])].name

    def outfit_key(self, row: int) -> str:
        return self.outfit_keys[self.source_name(row)][int(self.outfit_ord[row])]

    def type_name(self, row: int) -> str:
        t = int(self.type[row])
        return TYPES[t] if t >= 0 else "unknown"

    def crop_path(self, row: int) -> str:
        """Repo-relative crop PNG path, rebuilt from the relative `file` field (never crop_path)."""
        spec = self.stores[int(self.source[row])]
        return "/".join([spec.output_root, self.outfit_key(row), self.files[row]])

    def describe(self, row: int) -> dict:
        return {"type": self.type_name(row), "hue": float(self.hue[row]),
                "formality": float(self.formality[row]), "area": float(self.area[row])}
```

- [ ] **Step 5: Write `Module4/build_cache.py`** (pool only for now; Task 6 adds the co-occurrence matrix and reference)

```python
"""Build Module 4's cache: candidate pool (projection, trend, dedupe), co-occurrence matrix and harmony
reference. Rebuilds only when the fingerprint (checkpoint, trend model, stores, t_dup) has changed.

    python Module4/build_cache.py            # build if missing or stale
    python Module4/build_cache.py --force    # rebuild anyway
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from candidate_pool import CandidatePool, build_pool, is_fresh, pool_dir, pool_fingerprint  # noqa: E402
from dense_gnn import DenseGNN  # noqa: E402
from m2_bridge import apply_vram_cap, load_gnn, load_trend_model, pick_device  # noqa: E402
from m4_settings import M4Settings  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    s = M4Settings.load()
    dev = pick_device(args.device)
    apply_vram_cap(s.vram_gb, dev)
    if is_fresh(s) and not args.force:
        print(f"[m4-cache] up to date: {pool_dir(s)}")
        return
    fp = os.path.join(pool_dir(s), "fingerprint.json")
    if os.path.exists(fp):
        os.remove(fp)   # an interrupted rebuild must never look fresh
    gnn = DenseGNN(load_gnn(s.abs(s.checkpoint), dev), dev, s.vram_gb)
    t = time.time()
    build_pool(s, gnn, load_trend_model(s.abs(s.trend_model)))
    print(f"[m4-cache] pool built in {time.time() - t:.0f}s")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(pool_fingerprint(s), f, indent=2)
    pool = CandidatePool.load(s, dev)
    scored = (pool.trend == pool.trend).mean()
    print(f"[m4-cache] {pool.n} rows; trend-scored {scored:.1%}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Build the cache**

Run: `python Module4/build_cache.py`
Expected: `[m4-cache] pool built in …s` (about a minute), then `480079 rows; trend-scored …%`

- [ ] **Step 7: Run the tests**

Run: `python -m pytest Module4/test_candidate_pool.py -v`
Expected: 7 passed

- [ ] **Step 8: Commit (ask the owner first)**

```bash
git add Module4/candidate_pool.py Module4/build_cache.py Module4/test_candidate_pool.py Module4/conftest.py
git commit -m "feat(m4): candidate pool cache with precomputed projection and vectorised trend"
```

---

### Task 5: Near-duplicate dedupe + visual t_dup calibration (D8a/b)

**Files:**
- Create: `Module4/calibrate_dup.py`
- Modify: `Module4/candidate_pool.py` (add `dedupe()`; replace the two identity `rep_*` lines in `build_pool`)
- Test: `Module4/test_dedupe.py`

**Interfaces:**
- Consumes: `CandidatePool`, `load_store_arrays`, `M4Settings`.
- Produces: `dedupe(emb, type_ids, area, t_dup, eligible, chunk=2048) -> rep_of int32 [N]` (-1 = not eligible). Rows that are near-identical (cosine ≥ t_dup, transitively, within one specific type) share one representative: the largest crop (max `area`), ties broken by the lowest row. `calibrate_dup.sample_band_pairs(emb, rows, lo, hi, n_pairs, rng)`; CLI `python Module4/calibrate_dup.py`.

- [ ] **Step 1: Write the failing test** `Module4/test_dedupe.py`

```python
import numpy as np
import pytest
import torch

from candidate_pool import SPLIT_CODES, dedupe


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_dedupe_groups_transitively_and_picks_largest_crop():
    rng = np.random.default_rng(0)
    base = _unit(rng.standard_normal((3, 64)))
    noise = lambda: 0.02 * rng.standard_normal(64)
    emb = _unit(np.stack([base[0], base[0] + noise(), base[0] + noise(),   # rows 0-2: one item
                          base[1], base[1] + noise(),                      # rows 3-4: another
                          base[2],                                         # row 5: unique
                          base[0]]))                                       # row 6: same pixels, other type
    types = np.array([3, 3, 3, 3, 3, 3, 7], np.int8)
    area = np.array([10, 50, 50, 5, 1, 9, 99], np.float64)
    rep = dedupe(torch.from_numpy(emb), types, area, t_dup=0.95, eligible=np.ones(7, bool))
    assert rep.tolist() == [1, 1, 1, 3, 3, 5, 6]


def test_dedupe_skips_ineligible_rows():
    emb = _unit(np.eye(4, 8) + 0.0)
    rep = dedupe(torch.from_numpy(emb), np.zeros(4, np.int8), np.ones(4), 0.95,
                 eligible=np.array([True, False, True, True]))
    assert rep.tolist() == [0, -1, 2, 3]


@pytest.mark.real
def test_real_dedupe_invariants(pool, settings):
    idx = np.arange(pool.n)
    for rep in (pool.rep_demo, pool.rep_eval):
        ok = rep >= 0
        assert (rep[rep[ok]] == rep[ok]).all()                 # representatives represent themselves
        assert (pool.type[rep[ok]] == pool.type[ok]).all()    # never across types
    assert (pool.split[pool.rep_eval >= 0] == SPLIT_CODES["test"]).all()
    # distinct representatives of one type are never near-duplicates (exhaustive on sunglasses)
    from swap_groups import TYPE_ID
    rows = np.nonzero((pool.type == TYPE_ID["sunglasses"]) & (pool.rep_demo == idx))[0]
    E = pool.emb[torch.as_tensor(rows, device=pool.device)]
    sims = E @ E.T
    sims.fill_diagonal_(-1)
    assert float(sims.max()) < settings.t_dup
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest Module4/test_dedupe.py -v -m "not real"`
Expected: `ImportError: cannot import name 'dedupe' from 'candidate_pool'`

- [ ] **Step 3: Add `dedupe` to `Module4/candidate_pool.py`** (after `trend_scores`):

```python
def dedupe(emb: torch.Tensor, type_ids: np.ndarray, area: np.ndarray, t_dup: float,
           eligible: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Near-duplicate groups (D8a): within each specific type, rows linked by cosine >= t_dup
    (transitively) share one representative, the largest crop (ties: lowest row).
    emb: [N, 512] L2-normalised, on the GPU when available. Returns rep_of int32 [N], -1 = not eligible."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    rep = np.full(len(type_ids), -1, np.int32)
    for t in np.unique(type_ids[eligible]):
        rows = np.nonzero(eligible & (type_ids == t))[0]
        E = emb[torch.as_tensor(rows, device=emb.device)]
        src, dst = [], []
        for s in range(0, len(rows), chunk):
            ii, jj = torch.nonzero(E[s:s + chunk] @ E.T >= t_dup, as_tuple=True)
            ii = ii + s
            keep = jj > ii
            src.append(ii[keep].cpu().numpy())
            dst.append(jj[keep].cpu().numpy())
        src, dst, m = np.concatenate(src), np.concatenate(dst), len(rows)
        _, comp = connected_components(coo_matrix((np.ones(len(src), np.int8), (src, dst)), shape=(m, m)),
                                       directed=False)
        order = np.lexsort((rows, -area[rows], comp))       # by component, then area desc, then row
        c = comp[order]
        first = np.ones(m, bool)
        first[1:] = c[1:] != c[:-1]
        rep_of_comp = np.empty(comp.max() + 1, np.int64)
        rep_of_comp[c[first]] = rows[order][first]
        rep[rows] = rep_of_comp[comp]
    return rep
```

- [ ] **Step 4: Use it in `build_pool`.** Replace the two `meta["rep_demo"]` / `meta["rep_eval"]` lines with:

```python
    emb_t = torch.as_tensor(emb, device=gnn.device)
    emb_t = emb_t / emb_t.norm(dim=1, keepdim=True).clamp(min=1e-8)
    known = meta["type"] >= 0
    test = meta["split"] == SPLIT_CODES["test"]
    meta["rep_demo"] = dedupe(emb_t, meta["type"], scal[:, 2], settings.t_dup, eligible=known)
    meta["rep_eval"] = dedupe(emb_t, meta["type"], scal[:, 2], settings.t_dup, eligible=known & test)
    del emb_t
```

- [ ] **Step 5: Run the unit tests**

Run: `python -m pytest Module4/test_dedupe.py -v -m "not real"`
Expected: 2 passed, 1 deselected

- [ ] **Step 6: Write `Module4/calibrate_dup.py`**

```python
"""Visual calibration of the near-duplicate threshold t_dup (D8).

For a few garment types, draws pool pairs whose CLIP cosine falls in each band (0.90 ... 0.99) and
renders one contact sheet per type: one row per band, each cell = the two crops side by side. A
reviewer marks the lowest band where pairs look like "the same item"; that band becomes t_dup in
m4_settings.json (status "calibrated"), followed by  python Module4/build_cache.py --force.

    python Module4/calibrate_dup.py [--types shoe bag pants jacket dress] [--pairs 6]
Sheets go to <cache_dir>/dup_calibration/<type>.png (gitignored).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from candidate_pool import CandidatePool  # noqa: E402
from m2_bridge import apply_vram_cap, pick_device  # noqa: E402
from m4_settings import M4Settings  # noqa: E402
from swap_groups import TYPE_ID  # noqa: E402

BANDS = (0.90, 0.93, 0.95, 0.97, 0.99)
HALF_WIDTH = 0.005


def sample_band_pairs(emb, rows, lo, hi, n_pairs, rng, max_anchors=2000):
    """Up to n_pairs (row_a, row_b, cosine) with lo <= cosine < hi, anchors drawn at random."""
    E = emb[torch.as_tensor(rows, device=emb.device)]
    out = []
    for a in rng.permutation(len(rows))[:max_anchors]:
        sims = E @ E[a]
        sims[a] = -1
        hits = torch.nonzero((sims >= lo) & (sims < hi)).squeeze(1)
        if len(hits):
            b = int(hits[int(rng.integers(len(hits)))])
            out.append((int(rows[a]), int(rows[b]), float(sims[b])))
            if len(out) == n_pairs:
                break
    return out


def _on_white(path):
    from PIL import Image
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im).convert("RGB")


def render_sheet(pool, settings, type_name, pairs_by_band, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = max(1, max(len(p) for p in pairs_by_band.values())) * 2
    fig, axes = plt.subplots(len(pairs_by_band), cols, figsize=(1.4 * cols, 1.8 * len(pairs_by_band)),
                             squeeze=False)
    for i, (band, pairs) in enumerate(pairs_by_band.items()):
        for ax in axes[i]:
            ax.axis("off")
        axes[i][0].set_title(f"cos ≈ {band:.2f}", fontsize=8, loc="left")
        for j, (a, b, cos) in enumerate(pairs):
            for k, r in enumerate((a, b)):
                axes[i][2 * j + k].imshow(_on_white(settings.abs(pool.crop_path(r))))
            axes[i][2 * j].set_xlabel(f"{cos:.3f}", fontsize=7)
    fig.suptitle(f"Near-duplicate calibration: {type_name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--types", nargs="+", default=["shoe", "bag", "pants", "jacket", "dress"])
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    s = M4Settings.load()
    dev = pick_device()
    apply_vram_cap(s.vram_gb, dev)
    pool = CandidatePool.load(s, dev)
    out_dir = s.abs(os.path.join(s.cache_dir, "dup_calibration"))
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for t in args.types:
        rows = np.nonzero(pool.type == TYPE_ID[t])[0]
        bands = {b: sample_band_pairs(pool.emb, rows, b - HALF_WIDTH, b + HALF_WIDTH, args.pairs, rng)
                 for b in BANDS}
        path = os.path.join(out_dir, f"{t}.png")
        render_sheet(pool, s, t, bands, path)
        print(f"[dup-cal] {t}: {[len(v) for v in bands.values()]} pairs per band -> {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Rebuild with dedupe, run the real test, render the sheets**

Run: `python Module4/build_cache.py --force` then `python -m pytest Module4/test_dedupe.py Module4/test_candidate_pool.py -v` then `python Module4/calibrate_dup.py`
Expected: tests pass (10 passed); five PNG sheets in `Module4/cache/dup_calibration/`

- [ ] **Step 8: HUMAN CHECKPOINT: the owner picks t_dup.** Review the sheets and pick the lowest band
where pairs read as "the same item". Set `"t_dup"` in `Module4/m4_settings.json` to that value and
`"status": {"t_dup": "calibrated", ...}`. Then run `python Module4/build_cache.py --force` and re-run the
step-7 tests. Record the chosen value and the reasoning in `Module4/README.md` (Task 11).

- [ ] **Step 9: Commit (ask the owner first)**

```bash
git add Module4/candidate_pool.py Module4/calibrate_dup.py Module4/test_dedupe.py Module4/m4_settings.json
git commit -m "feat(m4): near-duplicate dedupe (demo + eval pools) and t_dup contact sheets"
```

---

### Task 6: Harmony reference + co-occurrence cache (D3, D4)

**Files:**
- Create: `Module4/harmony_reference.py`
- Modify: `Module4/build_cache.py` (build the co-occurrence matrix and reference after the pool, before writing the fingerprint)
- Test: `Module4/test_harmony_reference.py`

**Interfaces:**
- Consumes: `CandidatePool.{outfits, split, type, proj, hue, formality, area, broad_t, trend, rep_eval}`, `DenseGNN.logits_chunked`, `CooccurrenceFilter.from_outfits`.
- Produces: `HarmonyReference(logits_sorted: np.ndarray, sigma_logit: float, sigma_trend: float)` with `.pct(raw) -> float in [0, 100]`, `.save(path)`, `.load(path)`; `test_outfits(pool, min_garments=2) -> list[np.ndarray]`; `train_outfit_types(pool) -> iterator[list[str]]`; `build_reference(pool, gnn) -> HarmonyReference`. Files: `pool/cooc.npy`, `pool/reference.json`.

D3: σ_logit = std of raw GNN logits over **all** real test outfits (≥ 2 garments, matching M2's
`OutfitDataset`); σ_trend = std of garment trend scores over eval-pool representatives that have a
score. `pct()` uses the same formula as `PhotoScorer.pct`, so M2's demo can switch to this reference in D10.

- [ ] **Step 1: Write the failing test** `Module4/test_harmony_reference.py`

```python
import numpy as np
import pytest
import torch

from harmony_reference import HarmonyReference, test_outfits


def test_pct_is_monotone_and_bounded():
    ref = HarmonyReference(np.linspace(-3, 3, 7), 1.0, 1.0)
    vals = [ref.pct(x) for x in (-10, -3, -0.5, 0, 2.9, 3, 10)]
    assert vals == sorted(vals)
    assert vals[0] == 0.0 and vals[-1] == 100.0
    assert ref.pct(0.0) == pytest.approx(100 * 4 / 7)   # side="right": ties count as beaten


def test_save_load_round_trip(tmp_path):
    ref = HarmonyReference(np.array([-1.0, 0.5, 2.0]), 0.7, 12.5)
    p = str(tmp_path / "r.json")
    ref.save(p)
    got = HarmonyReference.load(p)
    assert np.array_equal(got.logits_sorted, ref.logits_sorted)
    assert (got.sigma_logit, got.sigma_trend) == (0.7, 12.5)


@pytest.mark.real
def test_reference_covers_all_test_outfits_and_matches_model(pool, gnn, settings):
    import os
    from candidate_pool import pool_dir
    from m2_bridge import ATTR_MASK, BROAD_VOCAB, od
    ref = HarmonyReference.load(os.path.join(pool_dir(settings), "reference.json"))
    outs = test_outfits(pool)
    assert len(ref.logits_sorted) == len(outs) > 40000
    assert ref.sigma_logit > 0 and ref.sigma_trend > 0
    # every size-7 test outfit, re-scored the slow M2 way, must appear in the reference
    feat = np.load(settings.abs(os.path.join(settings.stores[0].store_dir, "features.npy")), mmap_mode="r")
    for rows in [o for o in outs if len(o) == 7]:
        nodes = [{"embedding": np.asarray(feat[r, :512]), "colour": np.asarray(feat[r, 512:544]),
                  "attr_feat": np.asarray(feat[r, 544:]), "category": BROAD_VOCAB[pool.broad[r]],
                  "hue": float(pool.hue[r]), "formality": float(pool.formality[r]), "area": float(pool.area[r])}
                 for r in rows]
        g = od.OutfitDataset.build_graph(nodes, 0.0, attr_mask=ATTR_MASK)
        with torch.no_grad():
            want = float(gnn.model(g["x"].to(gnn.device), g["edge_index"].to(gnn.device),
                                   g["edge_attr"].to(gnn.device))[0])
        assert np.min(np.abs(ref.logits_sorted - want)) < 1e-4
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest Module4/test_harmony_reference.py -v -m "not real"`
Expected: `ModuleNotFoundError: No module named 'harmony_reference'`

- [ ] **Step 3: Write `Module4/harmony_reference.py`**

```python
"""Harmony reference (D3): raw GNN logits of every real test outfit, plus the two standardisers.

- pct(raw): harmony percentile 0-100 vs real test outfits (same formula as M2's PhotoScorer.pct,
  but over the full test set instead of 500 outfits, so it moves in fine steps).
- sigma_logit: std of those logits, the unit of ΔH.
- sigma_trend: std of garment trend scores over the eval pool, the unit of ΔT.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from candidate_pool import SPLIT_CODES
from swap_groups import TYPES


@dataclass
class HarmonyReference:
    logits_sorted: np.ndarray
    sigma_logit: float
    sigma_trend: float

    def pct(self, raw: float) -> float:
        return float(100.0 * np.searchsorted(self.logits_sorted, raw, side="right") / len(self.logits_sorted))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"logits_sorted": self.logits_sorted.tolist(), "sigma_logit": self.sigma_logit,
                       "sigma_trend": self.sigma_trend, "n_outfits": int(len(self.logits_sorted))}, f)

    @classmethod
    def load(cls, path: str) -> "HarmonyReference":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(np.asarray(d["logits_sorted"], np.float64), float(d["sigma_logit"]), float(d["sigma_trend"]))


def test_outfits(pool, min_garments: int = 2) -> list:
    test = SPLIT_CODES["test"]
    return [rows for rows in pool.outfits().values()
            if len(rows) >= min_garments and pool.split[rows[0]] == test]


test_outfits.__test__ = False   # not a pytest test despite the name


def train_outfit_types(pool):
    train = SPLIT_CODES["train"]
    for rows in pool.outfits().values():
        if pool.split[rows[0]] == train:
            yield [TYPES[t] for t in pool.type[rows] if t >= 0]


def build_reference(pool, gnn) -> HarmonyReference:
    by_size = defaultdict(list)
    for rows in test_outfits(pool):
        by_size[len(rows)].append(rows)
    parts = []
    for n, lst in sorted(by_size.items()):
        R = torch.as_tensor(np.stack(lst), device=pool.device)

        def make(a, b, R=R):
            r = R[a:b]
            return pool.proj[r], pool.hue[r], pool.formality[r], pool.area[r], pool.broad_t[r]

        parts.append(gnn.logits_chunked(make, R.shape[0], n).numpy())
    logits = np.sort(np.concatenate(parts)).astype(np.float64)
    reps = pool.rep_eval == np.arange(pool.n)
    return HarmonyReference(logits, float(logits.std()), float(np.nanstd(pool.trend[reps])))
```

- [ ] **Step 4: Extend `Module4/build_cache.py`.** Add the imports:

```python
from harmony_reference import build_reference, train_outfit_types  # noqa: E402
from swap_groups import SWAP_GROUPS, CooccurrenceFilter  # noqa: E402
```

Then replace everything from `print(f"[m4-cache] pool built ...")` to the end of `main()` with code that builds the
co-occurrence matrix and reference **before** writing the fingerprint:

```python
    print(f"[m4-cache] pool built in {time.time() - t:.0f}s")
    pool = CandidatePool(stores=s.stores, **_raw_pool_arrays(s), device=dev)
    CooccurrenceFilter.from_outfits(train_outfit_types(pool), s.cooc_min).save(
        os.path.join(pool_dir(s), "cooc.npy"))
    ref = build_reference(pool, gnn)
    ref.save(os.path.join(pool_dir(s), "reference.json"))
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(pool_fingerprint(s), f, indent=2)
    idx = np.arange(pool.n)
    print(f"[m4-cache] {pool.n} rows; demo reps {(pool.rep_demo == idx).sum()}, "
          f"eval reps {(pool.rep_eval == idx).sum()}; trend-scored {(pool.trend == pool.trend).mean():.1%}")
    print(f"[m4-cache] reference: {len(ref.logits_sorted)} test outfits, sigma_logit {ref.sigma_logit:.4f}, "
          f"sigma_trend {ref.sigma_trend:.3f}")
    for g, types in SWAP_GROUPS.items():
        demo = len(pool.candidate_rows(list(types), "demo"))
        ev = len(pool.candidate_rows(list(types), "eval"))
        print(f"[m4-cache]   group {g:10s} demo {demo:>7}  eval {ev:>7}")
```

and add the helper (above `main`) that loads the freshly written files without the freshness check:

```python
def _raw_pool_arrays(s):
    from candidate_pool import load_store_arrays
    d = pool_dir(s)
    emb, scal = load_store_arrays(s)
    with open(os.path.join(d, "files.json"), encoding="utf-8") as f:
        files = json.load(f)
    with open(os.path.join(d, "outfit_keys.json"), encoding="utf-8") as f:
        keys = json.load(f)
    return dict(proj=np.load(os.path.join(d, "proj.npy")), emb=emb, scalars=scal,
                meta=dict(np.load(os.path.join(d, "meta.npz"))), files=files, outfit_keys=keys)
```

(add `import numpy as np` at the top of `build_cache.py`).

- [ ] **Step 5: Rebuild and run the tests**

Run: `python Module4/build_cache.py --force` then `python -m pytest Module4/test_harmony_reference.py Module4/test_swap_groups.py -v`
Expected: the cache summary prints group sizes; 13 passed

- [ ] **Step 6: Commit (ask the owner first)**

```bash
git add Module4/harmony_reference.py Module4/build_cache.py Module4/test_harmony_reference.py
git commit -m "feat(m4): full-test-set harmony reference and train co-occurrence cache"
```

---

### Task 7: Outfit state (with area normalisation) and the swap scorer

**Files:**
- Create: `Module4/outfit_state.py`, `Module4/swap_scorer.py`
- Test: `Module4/test_outfit_state.py`, `Module4/test_swap_scorer.py`

**Interfaces:**
- Consumes: `od.build_node_from_meta`, `node_features`, `DenseGNN`, `CandidatePool`, trend model (`.score_embedding(emb, broad) -> dict`), `swap_group`, `BROAD_ID`.
- Produces:
  - `REFERENCE_FRAME_PX = 256 * 384`
  - `NoGarmentsDetected(ValueError)`
  - `Piece(file, type, broad, proj: Tensor[128], emb: np[512] unit, hue: float (NaN ok), formality, area, trend: float (NaN = unscored), cluster: int (-1), pool_row: int | None = None)` with `.group` and `.describe() -> dict`
  - `OutfitState(pieces: tuple[Piece, ...], outfit_dir=None, source_key=None, flags=())` with `.with_swap(i, piece)`
  - `photo_pixels(outfit_dir) -> int | None`
  - `load_outfit_dir(outfit_dir, gnn, trend_model, photo_px=None) -> OutfitState`
  - `piece_from_pool(pool, row)`, `state_from_pool(pool, source, key)`
  - `SwapScorer(pool, gnn)` with `.outfit_logit(state) -> float` and `.swap_logits(state, i, rows: LongTensor) -> np.ndarray float32`

- [ ] **Step 1: Write the failing unit test** `Module4/test_outfit_state.py`

```python
import json

import numpy as np
import pytest
import torch
from PIL import Image

from outfit_state import REFERENCE_FRAME_PX, NoGarmentsDetected, load_outfit_dir


class StubGNN:
    device = torch.device("cpu")

    def project(self, x):
        return torch.zeros(x.shape[0], 128)


class StubTrend:
    def score_embedding(self, emb, category):
        if category == "bottom":
            return {"trendiness_score": None, "cluster_id": None}
        return {"trendiness_score": 42.0, "cluster_id": 3}


def _garment(d, name, category, broad, pixels, hue):
    Image.new("RGBA", (20, 30), (200, 30, 30, 255)).save(d / f"{name}.png")
    v = np.random.default_rng(len(name)).standard_normal(512).astype(np.float32)
    np.save(d / f"{name}.npy", v / np.linalg.norm(v))
    return {"file": f"{name}.png", "category": category, "broad_category": broad, "pixel_count": pixels,
            "attributes": {"color": {"hue": hue, "saturation": 0.5, "value": 0.5}, "formality_score": 60}}


def test_loads_pieces_normalises_area_and_scores_trend(tmp_path):
    meta = [_garment(tmp_path, "jacket_0", "jacket", "top", 20000, 350.0),
            _garment(tmp_path, "pants_0", "pants", "bottom", 10000, 210.0)]
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    Image.new("RGB", (512, 384)).save(tmp_path / "photo_vis.jpg")      # 2x the 256x384 frame
    st = load_outfit_dir(str(tmp_path), StubGNN(), StubTrend())
    assert [p.type for p in st.pieces] == ["jacket", "pants"]
    assert [p.group for p in st.pieces] == ["upper", "lower"]
    scale = REFERENCE_FRAME_PX / (512 * 384)
    assert st.pieces[0].area == pytest.approx(20000 * scale)
    assert st.pieces[0].trend == 42.0 and st.pieces[0].cluster == 3
    assert np.isnan(st.pieces[1].trend) and st.pieces[1].cluster == -1
    assert st.pieces[0].formality == 60.0 and st.pieces[0].hue == 350.0
    assert abs(np.linalg.norm(st.pieces[0].emb) - 1) < 1e-5


def test_explicit_photo_size_wins_and_missing_size_is_flagged(tmp_path):
    meta = [_garment(tmp_path, "shoe_0", "shoe", "accessory", 1000, 20.0)]
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    st = load_outfit_dir(str(tmp_path), StubGNN(), StubTrend(), photo_px=REFERENCE_FRAME_PX * 4)
    assert st.pieces[0].area == pytest.approx(250.0)
    st2 = load_outfit_dir(str(tmp_path), StubGNN(), StubTrend())
    assert st2.pieces[0].area == 1000.0 and any("photo size unknown" in f for f in st2.flags)


def test_garment_without_embedding_is_skipped_and_flagged(tmp_path):
    meta = [_garment(tmp_path, "hat_0", "hat", "accessory", 900, 10.0),
            _garment(tmp_path, "bag_0", "bag", "accessory", 900, 10.0)]
    (tmp_path / "bag_0.npy").unlink()
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    st = load_outfit_dir(str(tmp_path), StubGNN(), StubTrend(), photo_px=REFERENCE_FRAME_PX)
    assert [p.file for p in st.pieces] == ["hat_0.png"]
    assert any("bag_0.png" in f for f in st.flags)


def test_no_garments_raises(tmp_path):
    (tmp_path / "metadata.json").write_text("[]")
    with pytest.raises(NoGarmentsDetected):
        load_outfit_dir(str(tmp_path), StubGNN(), StubTrend())


def test_with_swap_is_immutable(tmp_path):
    meta = [_garment(tmp_path, "skirt_0", "skirt", "bottom", 5000, 0.0),
            _garment(tmp_path, "shoe_0", "shoe", "accessory", 900, 0.0)]
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    st = load_outfit_dir(str(tmp_path), StubGNN(), StubTrend(), photo_px=REFERENCE_FRAME_PX)
    st2 = st.with_swap(0, st.pieces[1])
    assert st.pieces[0].type == "skirt" and st2.pieces[0].type == "shoe"
```

- [ ] **Step 2: Write the failing real test** `Module4/test_swap_scorer.py`

```python
import random

import numpy as np
import pytest
import torch

from harmony_reference import test_outfits
from m2_bridge import ATTR_MASK, BROAD_VOCAB, od
from outfit_state import state_from_pool
from swap_scorer import SwapScorer

pytestmark = pytest.mark.real


def _nodes(pool, feat, rows):
    return [{"embedding": np.asarray(feat[r, :512]), "colour": np.asarray(feat[r, 512:544]),
             "attr_feat": np.asarray(feat[r, 544:]), "category": BROAD_VOCAB[pool.broad[r]],
             "hue": float(pool.hue[r]), "formality": float(pool.formality[r]), "area": float(pool.area[r])}
            for r in rows]


def _model_logit(gnn, nodes):
    g = od.OutfitDataset.build_graph(nodes, 0.0, attr_mask=ATTR_MASK)
    with torch.no_grad():
        return float(gnn.model(g["x"].to(gnn.device), g["edge_index"].to(gnn.device),
                               g["edge_attr"].to(gnn.device))[0])


@pytest.fixture(scope="module")
def feat(settings):
    import os
    return np.load(settings.abs(os.path.join(settings.stores[0].store_dir, "features.npy")), mmap_mode="r")


def test_outfit_and_swap_logits_match_model(pool, gnn, feat):
    sc = SwapScorer(pool, gnn)
    rng = random.Random(0)
    outs = test_outfits(pool)
    for rows in rng.sample(outs, 20):
        key = pool.outfit_key(int(rows[0]))
        st = state_from_pool(pool, "fashion144k", key)
        rows = [p.pool_row for p in st.pieces]
        assert abs(sc.outfit_logit(st) - _model_logit(gnn, _nodes(pool, feat, rows))) < 1e-4
        i = rng.randrange(len(rows))
        cands = torch.as_tensor(rng.sample(range(pool.n), 32), device=pool.device)
        got = sc.swap_logits(st, i, cands)
        for c, g in zip(cands.tolist(), got):
            sw = list(rows)
            sw[i] = c
            assert abs(g - _model_logit(gnn, _nodes(pool, feat, sw))) < 1e-4
```

- [ ] **Step 3: Run to verify failure**

Run: `python -m pytest Module4/test_outfit_state.py Module4/test_swap_scorer.py -v`
Expected: `ModuleNotFoundError: No module named 'outfit_state'`

- [ ] **Step 4: Write `Module4/outfit_state.py`**

```python
"""The outfit being recommended for: an immutable tuple of pieces.

A Piece carries everything scoring needs (projected GNN input, CLIP embedding, edge scalars, trend).
Pieces come from an M1 outfit folder (a user photo) or from pool rows (evaluation). OutfitState is
immutable and has with_swap(), so a later greedy multi-piece search can chain swaps without
changing the recommender.

Area normalisation: M1 keeps raw pixel counts, and the GNN's volume-contrast edge compares them
across pieces. Catalogue crops come from 256x384 photos, so a user photo's areas are rescaled to
that 98,304-px frame before they meet catalogue candidates.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from PIL import Image

from m2_bridge import BROAD_ID, BROAD_VOCAB, node_features, od
from swap_groups import TYPES, swap_group

REFERENCE_FRAME_PX = 256 * 384


class NoGarmentsDetected(ValueError):
    pass


@dataclass(frozen=True, eq=False)
class Piece:
    file: str
    type: str
    broad: str
    proj: torch.Tensor
    emb: np.ndarray
    hue: float
    formality: float
    area: float
    trend: float
    cluster: int
    pool_row: int | None = None

    @property
    def group(self):
        return swap_group(self.type)

    def describe(self) -> dict:
        return {"type": self.type, "hue": self.hue, "formality": self.formality, "area": self.area}


@dataclass(frozen=True, eq=False)
class OutfitState:
    pieces: tuple
    outfit_dir: str | None = None
    source_key: tuple | None = None
    flags: tuple = field(default_factory=tuple)

    def with_swap(self, i: int, piece: Piece) -> "OutfitState":
        ps = list(self.pieces)
        ps[i] = piece
        return replace(self, pieces=tuple(ps))


def photo_pixels(outfit_dir: str):
    """Input photo size from M1's <id>_vis.jpg overlay (drawn on the full input image)."""
    for f in sorted(os.listdir(outfit_dir)):
        if f.endswith("_vis.jpg"):
            with Image.open(os.path.join(outfit_dir, f)) as im:
                return im.size[0] * im.size[1]
    return None


def load_outfit_dir(outfit_dir: str, gnn, trend_model, photo_px=None) -> OutfitState:
    meta_path = os.path.join(outfit_dir, "metadata.json")
    garments = []
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            garments = json.load(f)
    flags = []
    photo_px = photo_px or photo_pixels(outfit_dir)
    if photo_px:
        scale = REFERENCE_FRAME_PX / photo_px
    else:
        scale = 1.0
        flags.append("photo size unknown: garment areas not normalised")
    pieces = []
    for g in garments:
        npy = os.path.splitext(os.path.join(outfit_dir, g["file"]))[0] + ".npy"
        if not os.path.exists(npy):
            flags.append(f"{g['file']}: no embedding, skipped")
            continue
        node = od.build_node_from_meta(outfit_dir, g)
        if node["category"] not in BROAD_ID:
            raise ValueError(f"{g['file']}: unknown broad category {node['category']!r}")
        emb = node["embedding"] / (np.linalg.norm(node["embedding"]) + 1e-8)
        t = trend_model.score_embedding(node["embedding"], node["category"])
        pieces.append(Piece(
            file=g["file"], type=g.get("category", "unknown"), broad=node["category"],
            proj=gnn.project(torch.from_numpy(node_features(node))[None])[0],
            emb=emb.astype(np.float32),
            hue=np.nan if node["hue"] is None else float(node["hue"]),
            formality=float(node["formality"]), area=float(node["area"]) * scale,
            trend=np.nan if t["trendiness_score"] is None else float(t["trendiness_score"]),
            cluster=-1 if t.get("cluster_id") is None else int(t["cluster_id"])))
    if not pieces:
        raise NoGarmentsDetected("no garments detected")
    return OutfitState(tuple(pieces), outfit_dir=outfit_dir, flags=tuple(flags))


def piece_from_pool(pool, row: int) -> Piece:
    t = int(pool.type[row])
    return Piece(file=pool.files[row], type=TYPES[t] if t >= 0 else "unknown",
                 broad=BROAD_VOCAB[int(pool.broad[row])], proj=pool.proj[row],
                 emb=pool.emb[row].cpu().numpy(), hue=float(pool.hue[row]),
                 formality=float(pool.formality[row]), area=float(pool.area[row]),
                 trend=float(pool.trend[row]), cluster=int(pool.cluster[row]), pool_row=int(row))


def state_from_pool(pool, source: str, key: str) -> OutfitState:
    rows = pool.outfit_rows(source, key)
    return OutfitState(tuple(piece_from_pool(pool, int(r)) for r in rows), source_key=(source, key))
```

- [ ] **Step 5: Write `Module4/swap_scorer.py`**

```python
"""GNN scoring of an outfit and of every what-if swap of one piece, through DenseGNN.

The fixed pieces are broadcast across the batch and only slot i changes, so building B what-if
outfits is one gather from the pool's precomputed tensors (no Python graph loop).
"""
from __future__ import annotations

import numpy as np
import torch

from m2_bridge import BROAD_ID


class SwapScorer:
    def __init__(self, pool, gnn):
        self.pool, self.gnn = pool, gnn

    def _fixed(self, state):
        dev = self.gnn.device
        ps = state.pieces
        h = torch.stack([p.proj.to(dev) for p in ps])
        f64 = lambda xs: torch.tensor(xs, dtype=torch.float64, device=dev)
        return (h, f64([p.hue for p in ps]), f64([p.formality for p in ps]), f64([p.area for p in ps]),
                torch.tensor([BROAD_ID[p.broad] for p in ps], device=dev))

    def outfit_logit(self, state) -> float:
        h, hue, form, area, broad = self._fixed(state)
        return float(self.gnn.logits(h[None], hue[None], form[None], area[None], broad[None])[0])

    def swap_logits(self, state, i: int, rows: torch.Tensor) -> np.ndarray:
        h, hue, form, area, broad = self._fixed(state)
        n, pool = h.shape[0], self.pool

        def make(a, b):
            r = rows[a:b]
            B = r.shape[0]
            H = h.expand(B, n, -1).clone()
            H[:, i] = pool.proj[r]
            out = [H]
            for fixed, col in ((hue, pool.hue), (form, pool.formality), (area, pool.area), (broad, pool.broad_t)):
                x = fixed.expand(B, n).clone()
                x[:, i] = col[r]
                out.append(x)
            return tuple(out)

        return self.gnn.logits_chunked(make, int(rows.shape[0]), n).numpy().astype(np.float32)
```

- [ ] **Step 6: Run the tests**

Run: `python -m pytest Module4/test_outfit_state.py Module4/test_swap_scorer.py -v`
Expected: 6 passed

- [ ] **Step 7: Commit (ask the owner first)**

```bash
git add Module4/outfit_state.py Module4/swap_scorer.py Module4/test_outfit_state.py Module4/test_swap_scorer.py
git commit -m "feat(m4): outfit state with photo-area normalisation and broadcast swap scorer"
```

---

### Task 8: Selection rules, diversity and the "what changed" line (D2, D8c, D6)

**Files:**
- Create: `Module4/selection.py`, `Module4/explain.py`
- Test: `Module4/test_selection_explain.py`

**Interfaces:**
- Produces: `RULES`; `combine(d_h, d_t, w_h, w_t) -> np.ndarray`; `selection_score(g, rule, k=5) -> float` (-inf when empty); `cluster_keys(broad, cluster) -> int64 array` (unmatched = one bucket per broad category); `diverse_top(g, keys, qualifies, n=3) -> list[int]`; `explain.what_changed(current, candidate, others) -> str` (dicts with `type, hue, formality, area`).

- [ ] **Step 1: Write the failing test** `Module4/test_selection_explain.py`

```python
import math

import numpy as np
import pytest

from explain import what_changed
from selection import cluster_keys, combine, diverse_top, selection_score


def test_combine():
    assert combine(np.array([1.0, -2.0]), np.array([0.5, 1.0]), 1.0, 0.5).tolist() == [1.25, -1.5]


def test_selection_rules():
    g = np.array([1.0, 5.0, 3.0, -2.0, 4.0, 0.0])
    assert selection_score(g, "max_gain") == 5.0
    assert selection_score(g, "topk_mean_gain", k=3) == pytest.approx(4.0)
    assert selection_score(g, "topk_mean_gain", k=10) == pytest.approx(11 / 6)
    assert selection_score(g, "slot_percentile") == pytest.approx(4 / 6)
    assert selection_score(np.array([]), "max_gain") == -math.inf
    with pytest.raises(ValueError):
        selection_score(g, "nope")


def test_cluster_keys_bucket_unmatched_per_broad():
    k = cluster_keys(np.array([0, 0, 1, 0]), np.array([5, -1, 5, -1]))
    assert k[0] != k[2] and k[1] == k[3] and k[1] != k[0]


def test_diverse_top_skips_same_cluster_then_fills():
    g = np.array([3.0, 2.9, 2.0, 1.0, 0.5])
    keys = np.array([5, 5, 6, 7, 0])
    assert diverse_top(g, keys, np.ones(5, bool)) == [0, 2, 3]
    assert diverse_top(g, keys, np.array([1, 1, 0, 0, 0], bool)) == [0, 1, 2]   # too few qualify: fill by gain
    assert diverse_top(np.array([1.0]), np.array([1]), np.ones(1, bool)) == [0]
    assert diverse_top(np.array([]), np.array([]), np.array([], bool)) == []


def test_what_changed_describes_formality_and_type():
    cur = {"type": "skirt", "hue": 0.0, "formality": 20.0, "area": 1000.0}
    cand = {"type": "pants", "hue": 10.0, "formality": 60.0, "area": 1000.0}
    others = [{"type": "shirt_blouse", "hue": 180.0, "formality": 70.0, "area": 1000.0}]
    s = what_changed(cur, cand, others)
    assert s.startswith("Skirt → pants")
    assert "formality closer to the rest of the outfit (gap 50 → 10)" in s
    assert "colour" not in s and "piece relative" not in s


def test_what_changed_colour_volume_and_fallback():
    cur = {"type": "shoe", "hue": 0.0, "formality": 50.0, "area": 500.0}
    cand = {"type": "shoe", "hue": 170.0, "formality": 50.0, "area": 2000.0}
    others = [{"type": "pants", "hue": 10.0, "formality": 50.0, "area": 8000.0}]
    s = what_changed(cur, cand, others)
    assert "more contrast with the rest" in s and "larger piece" in s
    same = what_changed(cur, dict(cur), others)
    assert same.startswith("Similar formality, colour and volume")
    assert what_changed(cur, {**cur, "hue": float("nan")}, others).startswith("Similar")
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest Module4/test_selection_explain.py -v`
Expected: `ModuleNotFoundError: No module named 'explain'`

- [ ] **Step 3: Write `Module4/selection.py`**

```python
"""Pure ranking helpers: combined gain (D3), piece selection rules (D2), style-diverse top-n (D8c)."""
from __future__ import annotations

import numpy as np

RULES = ("max_gain", "topk_mean_gain", "slot_percentile")


def combine(d_h, d_t, w_h: float, w_t: float) -> np.ndarray:
    return w_h * np.asarray(d_h, np.float64) + w_t * np.asarray(d_t, np.float64)


def selection_score(g: np.ndarray, rule: str, k: int = 5) -> float:
    """Summarise one piece's candidate gains. slot_percentile = share of candidates beating the
    current piece (gain > 0)."""
    if rule not in RULES:
        raise ValueError(f"unknown selection rule {rule!r}; expected one of {RULES}")
    if g.size == 0:
        return -np.inf
    if rule == "max_gain":
        return float(g.max())
    if rule == "topk_mean_gain":
        kk = min(k, g.size)
        return float(np.partition(g, g.size - kk)[g.size - kk:].mean())
    return float((g > 0).mean())


def cluster_keys(broad, cluster) -> np.ndarray:
    """One integer per (broad category, M3 cluster); unmatched garments share one bucket per broad."""
    return np.asarray(broad, np.int64) * 1000 + np.asarray(cluster, np.int64) + 1


def diverse_top(g: np.ndarray, keys: np.ndarray, qualifies: np.ndarray, n: int = 3) -> list:
    """#1 = best gain; then the best qualifying candidates from not-yet-used clusters; then fill by gain."""
    order = np.argsort(-np.asarray(g), kind="stable")
    if order.size == 0:
        return []
    picks = [int(order[0])]
    seen = {int(keys[order[0]])}
    for j in order[1:]:
        if len(picks) == n:
            break
        if qualifies[j] and int(keys[j]) not in seen:
            picks.append(int(j))
            seen.add(int(keys[j]))
    for j in order[1:]:
        if len(picks) == n:
            break
        if int(j) not in picks:
            picks.append(int(j))
    return picks
```

- [ ] **Step 4: Write `Module4/explain.py`**

```python
"""Descriptive "what changed" line for a suggestion (D6).

Compares the replacement with the piece it replaces, relative to the rest of the outfit, on the
three cues the GNN's edges carry: formality gap, hue relation and volume. It describes the
difference; it does not claim these are the GNN's reasons (the GNN is non-linear in its inputs).
"""
from __future__ import annotations

import math

FORMALITY_STEP = 5.0   # points on M1's 0-100 formality scale
HUE_STEP = 20.0        # degrees of mean hue gap
VOLUME_RATIO = 1.5     # area ratio that counts as a visible size change


def _hue_gap(h: float, others: list):
    gaps = []
    for o in others:
        if not (math.isnan(h) or math.isnan(o)):
            d = abs(h - o) % 360.0
            gaps.append(d if d <= 180.0 else 360.0 - d)
    return sum(gaps) / len(gaps) if gaps else None


def what_changed(current: dict, candidate: dict, others: list) -> str:
    parts = []
    if candidate["type"] != current["type"]:
        parts.append(f"{current['type']} → {candidate['type']}")
    if others:
        f_rest = [o["formality"] for o in others]
        g0 = sum(abs(current["formality"] - f) for f in f_rest) / len(f_rest)
        g1 = sum(abs(candidate["formality"] - f) for f in f_rest) / len(f_rest)
        if abs(g1 - g0) >= FORMALITY_STEP:
            parts.append(f"formality {'closer to' if g1 < g0 else 'further from'} the rest of the outfit "
                         f"(gap {g0:.0f} → {g1:.0f})")
        h_rest = [o["hue"] for o in others]
        h0, h1 = _hue_gap(current["hue"], h_rest), _hue_gap(candidate["hue"], h_rest)
        if h0 is not None and h1 is not None and abs(h1 - h0) >= HUE_STEP:
            parts.append(f"colour {'closer to' if h1 < h0 else 'more contrast with'} the rest "
                         f"(hue gap {h0:.0f}° → {h1:.0f}°)")
        ratio = (candidate["area"] + 1.0) / (current["area"] + 1.0)
        if ratio >= VOLUME_RATIO:
            parts.append("larger piece relative to the rest")
        elif ratio <= 1.0 / VOLUME_RATIO:
            parts.append("smaller piece relative to the rest")
    if not parts:
        return "Similar formality, colour and volume; the difference is in style details"
    s = "; ".join(parts)
    return s[0].upper() + s[1:]
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest Module4/test_selection_explain.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit (ask the owner first)**

```bash
git add Module4/selection.py Module4/explain.py Module4/test_selection_explain.py
git commit -m "feat(m4): selection rules, style-diverse top-3 and descriptive what-changed line"
```

---

### Task 9: Recommender + Recommendation (D2, D3, D5 search, D6 shape, D7 verdicts, D8b)

**Files:**
- Create: `Module4/recommendation.py`, `Module4/recommender.py`
- Test: `Module4/test_recommender_logic.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `Suggestion` (fields below), `Recommendation` (fields below) with `.to_json()`, `Recommendation.from_json(s)` and `format_text(r) -> str`
  - `PieceSwaps(rows, logits, d_h, d_t, trend_known, flags)` and `SwapTable(state, r0, pieces: list[PieceSwaps], mode)`, where `pieces[i]` belongs to `state.pieces[i]`
  - `Recommender(settings, pool, scorer, reference, cooc, gnn=None, trend_model=None)` with `.score(...)`, `.decide(...)`, `.recommend(...)` and `.from_settings(...)` as in "Interfaces at a glance"

`score()` makes every GNN call and caches per-candidate ΔH/ΔT. `decide()` is pure NumPy, so re-ranking for a new
`w_t` or rule (the D10 slider, the step-3 sweeps and the rule ablation) costs no GNN calls. A call counter
in the test asserts this.

- [ ] **Step 1: Write the failing test** `Module4/test_recommender_logic.py`

```python
import numpy as np
import pytest
import torch

from candidate_pool import SPLIT_CODES, CandidatePool
from harmony_reference import HarmonyReference
from m2_bridge import BROAD_ID
from m4_settings import M4Settings, StoreSpec
from outfit_state import OutfitState, Piece
from recommendation import Recommendation, format_text
from recommender import Recommender
from swap_groups import BROAD_OF_TYPE, TYPE_ID, TYPES, CooccurrenceFilter

# pool row: (type, stub GNN logit, M3 cluster, trend)
ROWS = [
    ("t_shirt", 3.0, 5, 60.0),        # 0
    ("sweater", 2.9, 5, 40.0),        # 1  same cluster as row 0
    ("jacket", 2.0, 6, 70.0),         # 2
    ("top", 1.0, 7, np.nan),          # 3  no trend score
    ("shirt_blouse", 0.5, -1, 55.0),  # 4
    ("pants", 0.2, 1, 90.0),          # 5  blocked for the skirt slot (pants + leggings never co-occur)
    ("skirt", 0.3, 2, 45.0),          # 6
    ("skirt", 9.0, 2, 50.0),          # 7  the worn skirt itself: own-item exclusion
    ("leggings", -0.5, 3, 30.0),      # 8
    ("shoe", 0.1, 4, 80.0),           # 9
    ("shoe", -1.0, 4, 20.0),          # 10
    ("bag", 7.0, 8, 99.0),            # 11 no bag is worn: never a candidate
    ("skirt", -0.2, 2, 40.0),         # 12
]
WORN = [("shirt_blouse_0.png", "shirt_blouse"), ("skirt_0.png", "skirt"),
        ("leggings_0.png", "leggings"), ("shoe_0.png", "shoe")]


class StubScorer:
    def __init__(self, logits):
        self.logits, self.calls = logits, 0

    def outfit_logit(self, state):
        self.calls += 1
        return 0.0

    def swap_logits(self, state, i, rows):
        self.calls += 1
        return np.array([self.logits[int(r)] for r in rows.tolist()], np.float32)


def _unit(rng):
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def make(trend_worn=50.0, sigma_logit=1.0, train_rows=(), **overrides):
    rng = np.random.default_rng(0)
    worn_emb = [_unit(rng) for _ in WORN]
    n = len(ROWS)
    emb = np.stack([_unit(rng) for _ in ROWS])
    emb[7] = worn_emb[1]
    split = np.full(n, SPLIT_CODES["test"], np.uint8)
    split[list(train_rows)] = SPLIT_CODES["train"]
    meta = {"source": np.zeros(n, np.uint8), "split": split,
            "type": np.array([TYPE_ID[t] for t, *_ in ROWS], np.int8),
            "broad": np.array([BROAD_ID[BROAD_OF_TYPE[t]] for t, *_ in ROWS], np.int8),
            "store_row": np.arange(n, dtype=np.int32), "outfit_ord": np.arange(n, dtype=np.int32),
            "trend": np.array([r[3] for r in ROWS], np.float32),
            "cluster": np.array([r[2] for r in ROWS], np.int16),
            "rep_demo": np.arange(n, dtype=np.int32),
            "rep_eval": np.where(split == SPLIT_CODES["test"], np.arange(n), -1).astype(np.int32)}
    scal = np.column_stack([rng.uniform(0, 360, n), rng.uniform(0, 100, n), rng.uniform(500, 5000, n)])
    pool = CandidatePool(stores=[StoreSpec("fashion144k", "s", "o", "x.mat")], proj=np.zeros((n, 128)),
                         emb=emb, scalars=scal, meta=meta, files=[f"{t}_0.png" for t, *_ in ROWS],
                         outfit_keys={"fashion144k": [str(i) for i in range(n)]}, device="cpu")
    pieces = tuple(Piece(file=f, type=t, broad=BROAD_OF_TYPE[t], proj=torch.zeros(128), emb=e,
                         hue=10.0 * i, formality=50.0, area=2000.0, trend=trend_worn, cluster=-1)
                   for i, ((f, t), e) in enumerate(zip(WORN, worn_emb)))
    counts = np.full((len(TYPES), len(TYPES)), 100, np.int64)
    np.fill_diagonal(counts, 0)
    for a, b in [("pants", "leggings"), ("shorts", "leggings"), ("pants", "shorts")]:
        counts[TYPE_ID[a], TYPE_ID[b]] = counts[TYPE_ID[b], TYPE_ID[a]] = 0
    s = M4Settings.load().with_(w_h=1.0, w_t=0.0, rule="topk_mean_gain", topk=5, eps=0.25, t_dup=0.97,
                                tau={"max_gain": 0.0, "topk_mean_gain": 0.0, "slot_percentile": 0.5},
                                trend_floor=False, same_type=False)
    s = s.with_(**overrides)
    scorer = StubScorer([r[1] for r in ROWS])
    ref = HarmonyReference(np.linspace(-5, 5, 101), sigma_logit=sigma_logit, sigma_trend=10.0)
    return Recommender(s, pool, scorer, ref, CooccurrenceFilter(counts, 20)), OutfitState(pieces), scorer


def rows_of(table):
    return [sorted(ps.rows.tolist()) for ps in table.pieces]


def test_candidates_respect_groups_cooccurrence_and_own_item():
    rec, st, _ = make()
    assert rows_of(rec.score(st)) == [[0, 1, 2, 3, 4], [6, 12], [5, 8], [9, 10]]


def test_topk_mean_swaps_upper_with_style_diverse_top3_and_also_improve():
    rec, st, _ = make()
    r = rec.decide(rec.score(st))
    assert r.verdict == "swap" and r.swap_piece["file"] == "shirt_blouse_0.png"
    assert [s.pool_row for s in r.suggestions] == [0, 2, 3]          # row 1 shares row 0's cluster
    assert [a["piece"] for a in r.also_improve] == ["skirt_0.png"]   # 0.05 >= tau; others below
    assert r.suggestions[0].from_type == "shirt_blouse" and r.suggestions[0].to_type == "t_shirt"
    assert r.per_piece["leggings_0.png"]["topk_mean_gain"] == pytest.approx(-0.15)


def test_selection_rules_are_switchable_without_rescoring():
    rec, st, scorer = make()
    table = rec.score(st)
    calls = scorer.calls
    r_max = rec.decide(table, rule="max_gain")
    r_pct = rec.decide(table, rule="slot_percentile")
    assert [p["selection_score"] for p in r_max.per_piece.values()] == pytest.approx([3.0, 0.3, 0.2, 0.1])
    assert [p["selection_score"] for p in r_pct.per_piece.values()] == pytest.approx([1.0, 0.5, 0.5, 0.5])
    assert r_pct.verdict == "swap" and len(r_pct.also_improve) == 3
    for w in np.linspace(0, 1, 11):
        rec.decide(table, w_t=w)
    assert scorer.calls == calls == 1 + len(st.pieces)               # decide() never calls the GNN


def test_gains_are_standardised():
    rec, st, _ = make(sigma_logit=2.0, w_t=0.5)
    table = rec.score(st)
    up = table.pieces[0]
    assert up.d_h.tolist() == pytest.approx([1.5, 1.45, 1.0, 0.5, 0.25])
    assert up.d_t.tolist() == pytest.approx([1.0, -1.0, 2.0, 0.0, 0.5])
    assert up.trend_known.tolist() == [True, True, True, False, True]
    assert rec.decide(table).per_piece["shirt_blouse_0.png"]["best_gain"] == pytest.approx(2.0)


def test_refresh_when_below_tau_but_trendier_options_keep_harmony():
    rec, st, _ = make(tau={"max_gain": 10.0, "topk_mean_gain": 10.0, "slot_percentile": 1.1})
    r = rec.decide(rec.score(st))
    assert r.verdict == "refresh" and r.swap_piece is None
    assert [s.pool_row for s in r.suggestions] == [0, 2, 4]
    assert all(s.d_trend_z > 0 and s.d_harmony_z >= -0.25 for s in r.suggestions)


def test_keep_when_nothing_qualifies():
    rec, st, _ = make(trend_worn=100.0, tau={"max_gain": 10.0, "topk_mean_gain": 10.0, "slot_percentile": 1.1})
    r = rec.decide(rec.score(st))
    assert r.verdict == "keep" and r.suggestions == [] and r.also_improve == []


def test_trend_floor_drops_less_trendy_known_and_relaxes_when_empty():
    rec, st, _ = make(trend_floor=True)
    table = rec.score(st)
    assert rows_of(table) == [[0, 2, 3, 4], [6, 12], [5], [9]]
    assert any("relaxed" in f for f in table.pieces[1].flags)


def test_same_type():
    rec, st, _ = make(same_type=True)
    assert rows_of(rec.score(st)) == [[4], [6, 12], [8], [9, 10]]


def test_eval_mode_and_source_exclusion():
    rec, st, _ = make(train_rows=(0,))
    table = rec.score(st, mode="eval", exclude=("fashion144k", "2"))
    assert rows_of(table)[0] == [1, 3, 4]


def test_json_round_trip_and_text():
    rec, st, _ = make(w_t=0.5)
    r = rec.decide(rec.score(st))
    again = Recommendation.from_json(r.to_json())
    assert again == r
    assert "SWAP your shirt_blouse" in format_text(r)
    assert r.provisional == rec.settings.provisional()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest Module4/test_recommender_logic.py -v`
Expected: `ModuleNotFoundError: No module named 'recommendation'`

- [ ] **Step 3: Write `Module4/recommendation.py`**

```python
"""The Recommendation returned by Module 4 (D6): a plain, JSON-serialisable dataclass consumed by
the demo, the slide figure and the evaluation. All numbers are finite floats or None."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass
class Suggestion:
    piece_file: str            # the worn piece this replaces
    pool_row: int
    source: str                # fashion144k | pinterest
    outfit_id: str
    file: str
    crop_path: str             # repo-relative
    from_type: str
    to_type: str
    harmony_pct_before: float
    harmony_pct_after: float
    trend_before: float | None
    trend_after: float | None
    trend_known: bool          # False: ΔT was set to 0 (no confident trend match)
    d_harmony_z: float
    d_trend_z: float
    gain: float
    what_changed: str          # descriptive, not causal


@dataclass
class Recommendation:
    verdict: str               # swap | refresh | keep
    selection_rule: str
    pool_mode: str             # demo | eval
    weights: dict              # {w_h, w_t}
    thresholds: dict           # {tau, eps}
    baseline: dict             # {harmony_pct, harmony_logit, trend_mean}
    swap_piece: dict | None    # {file, category, group} for verdict "swap"
    suggestions: list
    also_improve: list         # [{piece, selection_score, suggestion}]
    per_piece: dict            # file -> {type, group, n_candidates, selection_score, best_gain, ...}
    flags: list
    provisional: list          # settings not yet calibrated

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_json(cls, s: str) -> "Recommendation":
        d = json.loads(s)
        d["suggestions"] = [Suggestion(**x) for x in d["suggestions"]]
        d["also_improve"] = [{**a, "suggestion": Suggestion(**a["suggestion"])} for a in d["also_improve"]]
        return cls(**d)


def format_text(r: Recommendation) -> str:
    b = r.baseline
    lines = [f"Outfit harmony {b['harmony_pct']:.0f}/100"
             + (f", mean piece trend {b['trend_mean']:.0f}/100" if b["trend_mean"] is not None else "")]
    if r.verdict == "swap":
        lines.append(f"SWAP your {r.swap_piece['category']} ({r.swap_piece['file']})")
    elif r.verdict == "refresh":
        lines.append("ALREADY STRONG. Optional: trendier options that keep the look")
    else:
        lines.append("ALREADY STRONG. No swap recommended")
    for i, s in enumerate(r.suggestions, 1):
        trend = ("" if s.trend_before is None or s.trend_after is None
                 else f"   trend {s.trend_before:.0f} → {s.trend_after:.0f}")
        lines.append(f"  {i}. {s.from_type} → {s.to_type} ({s.piece_file})   harmony "
                     f"{s.harmony_pct_before:.0f} → {s.harmony_pct_after:.0f}{trend}   [{s.source} {s.outfit_id}/{s.file}]")
        lines.append(f"     {s.what_changed}")
    if r.also_improve:
        lines.append("Also worth improving: " + ", ".join(a["piece"] for a in r.also_improve))
    lines.append("Per piece:")
    for f, p in r.per_piece.items():
        sc = "n/a" if p["selection_score"] is None else f"{p['selection_score']:+.3f}"
        lines.append(f"  {f:22s} candidates {p['n_candidates']:>6}  score {sc:>7}  "
                     f"{'✓' if p['clears_tau'] else ' '}  {'; '.join(p['flags'])}")
    if r.flags:
        lines.append("Notes: " + "; ".join(r.flags))
    if r.provisional:
        lines.append("Provisional (not yet calibrated): " + ", ".join(r.provisional))
    return "\n".join(lines)
```

- [ ] **Step 4: Write `Module4/recommender.py`**

```python
"""Module 4 recommender: which one piece to swap, and what for.

score(state)  -> SwapTable : every allowed candidate for every piece through the GNN (D4, D5, D8b),
                             cached as standardised ΔH and ΔT (D3). The only GNN calls.
decide(table) -> Recommendation : selection rule (D2), threshold + verdicts (D7), style-diverse
                             top 3 (D8c), output shape (D6). Pure NumPy, so the trend-weight slider and
                             the evaluation sweeps never re-run the GNN.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from candidate_pool import CandidatePool, pool_dir
from dense_gnn import DenseGNN
from explain import what_changed
from harmony_reference import HarmonyReference
from m2_bridge import apply_vram_cap, load_gnn, load_trend_model, pick_device
from m4_settings import M4Settings
from outfit_state import load_outfit_dir
from recommendation import Recommendation, Suggestion
from selection import cluster_keys, combine, diverse_top, selection_score
from swap_groups import CooccurrenceFilter, replacement_types
from swap_scorer import SwapScorer


def _num(x):
    return None if x is None or not np.isfinite(x) else float(x)


@dataclass
class PieceSwaps:
    rows: np.ndarray           # candidate pool rows
    logits: np.ndarray         # raw GNN logit of each what-if outfit
    d_h: np.ndarray            # (logit - r0) / sigma_logit
    d_t: np.ndarray            # (trend(cand) - trend(piece)) / sigma_trend, 0 where unknown
    trend_known: np.ndarray
    flags: list


@dataclass
class SwapTable:
    state: object
    r0: float
    pieces: list               # pieces[i] belongs to state.pieces[i]
    mode: str


class Recommender:
    def __init__(self, settings, pool, scorer, reference, cooc, gnn=None, trend_model=None):
        self.settings, self.pool, self.scorer = settings, pool, scorer
        self.reference, self.cooc = reference, cooc
        self.gnn, self.trend_model = gnn, trend_model

    @classmethod
    def from_settings(cls, settings=None, device=None, pool=None, gnn=None) -> "Recommender":
        s = settings or M4Settings.load()
        dev = pick_device(device)
        apply_vram_cap(s.vram_gb, dev)
        gnn = gnn or DenseGNN(load_gnn(s.abs(s.checkpoint), dev), dev, s.vram_gb)
        pool = pool or CandidatePool.load(s, dev)
        d = pool_dir(s)
        return cls(s, pool, SwapScorer(pool, gnn), HarmonyReference.load(os.path.join(d, "reference.json")),
                   CooccurrenceFilter.load(os.path.join(d, "cooc.npy"), s.cooc_min), gnn=gnn,
                   trend_model=load_trend_model(s.abs(s.trend_model)))

    # ---------------- GNN stage ----------------
    def _candidates(self, state, i, mode, exclude, same_type, trend_floor):
        p, flags = state.pieces[i], []
        types = replacement_types(p.type, same_type)
        if not types:
            return self.pool.empty_rows(), [f"type '{p.type}' has no swap group: not swapped"]
        others = [q.type for j, q in enumerate(state.pieces) if j != i]
        types = [t for t in types if self.cooc.allowed(t, others)]
        rows = self.pool.candidate_rows(types, mode=mode, exclude=exclude)
        if len(rows):
            rows = rows[self.pool.cosine_to(rows, p.emb) < self.settings.t_dup]       # D8b: not your own item
        if trend_floor and not np.isnan(p.trend) and len(rows):
            t = self.pool.trend_t[rows]
            keep = torch.isnan(t) | (t >= p.trend)
            if bool(keep.any()):
                rows = rows[keep]
            else:
                flags.append("trend floor left no candidates: floor relaxed")
        if not len(rows):
            flags.append("no allowed candidates: piece skipped")
        return rows, flags

    def score(self, state, mode="demo", exclude=None, same_type=None, trend_floor=None) -> SwapTable:
        s = self.settings
        same_type = s.same_type if same_type is None else same_type
        trend_floor = s.trend_floor if trend_floor is None else trend_floor
        r0 = self.scorer.outfit_logit(state)
        out = []
        for i, p in enumerate(state.pieces):
            rows, flags = self._candidates(state, i, mode, exclude, same_type, trend_floor)
            logits = self.scorer.swap_logits(state, i, rows) if len(rows) else np.zeros(0, np.float32)
            rows_np = rows.cpu().numpy()
            ct = self.pool.trend[rows_np].astype(np.float64)
            known = ~np.isnan(ct) & (not np.isnan(p.trend))
            d_t = np.where(known, (ct - p.trend) / self.reference.sigma_trend, 0.0)
            if np.isnan(p.trend):
                flags.append("worn piece has no trend score: ΔT = 0")
            out.append(PieceSwaps(rows_np, logits, (logits - r0) / self.reference.sigma_logit, d_t, known, flags))
        return SwapTable(state, r0, out, mode)

    # ---------------- decision stage (no GNN) ----------------
    def _suggestion(self, table, i, k, gain) -> Suggestion:
        ps, p, pool = table.pieces[i], table.state.pieces[i], self.pool
        row = int(ps.rows[k])
        others = [q.describe() for j, q in enumerate(table.state.pieces) if j != i]
        return Suggestion(
            piece_file=p.file, pool_row=row, source=pool.source_name(row), outfit_id=pool.outfit_key(row),
            file=pool.files[row], crop_path=pool.crop_path(row), from_type=p.type, to_type=pool.type_name(row),
            harmony_pct_before=self.reference.pct(table.r0), harmony_pct_after=self.reference.pct(float(ps.logits[k])),
            trend_before=_num(p.trend), trend_after=_num(pool.trend[row]), trend_known=bool(ps.trend_known[k]),
            d_harmony_z=float(ps.d_h[k]), d_trend_z=float(ps.d_t[k]), gain=float(gain),
            what_changed=what_changed(p.describe(), pool.describe(row), others))

    def decide(self, table, w_t=None, rule=None) -> Recommendation:
        s = self.settings
        w_t = s.w_t if w_t is None else float(w_t)
        rule = rule or s.rule
        tau, eps = float(s.tau[rule]), float(s.eps)
        pieces = table.state.pieces
        G = [combine(ps.d_h, ps.d_t, s.w_h, w_t) for ps in table.pieces]
        S = [selection_score(g, rule, s.topk) for g in G]
        per_piece = {}
        for p, ps, g, sc in zip(pieces, table.pieces, G, S):
            per_piece[p.file] = {
                "type": p.type, "group": p.group, "n_candidates": int(g.size), "selection_score": _num(sc),
                "best_gain": _num(g.max()) if g.size else None,
                "topk_mean_gain": _num(selection_score(g, "topk_mean_gain", s.topk)),
                "slot_percentile": _num(selection_score(g, "slot_percentile")),
                "best_dH": _num(ps.d_h.max()) if g.size else None,
                "best_dT": _num(ps.d_t.max()) if g.size else None,
                "clears_tau": bool(np.isfinite(sc) and sc >= tau), "trend": _num(p.trend), "flags": list(ps.flags)}
        order = sorted(range(len(S)), key=lambda j: S[j], reverse=True)
        verdict, swap_piece, suggestions, also = "keep", None, [], []
        if order and np.isfinite(S[order[0]]) and S[order[0]] >= tau:
            verdict, j = "swap", order[0]
            g, ps, p = G[j], table.pieces[j], pieces[j]
            swap_piece = {"file": p.file, "category": p.type, "group": p.group}
            qualifies = g > 0 if rule == "slot_percentile" else g >= tau
            keys = cluster_keys(self.pool.broad[ps.rows], self.pool.cluster[ps.rows])
            suggestions = [self._suggestion(table, j, k, g[k]) for k in diverse_top(g, keys, qualifies)]
            for o in order[1:]:
                if np.isfinite(S[o]) and S[o] >= tau:
                    k = int(np.argmax(G[o]))
                    also.append({"piece": pieces[o].file, "selection_score": float(S[o]),
                                 "suggestion": self._suggestion(table, o, k, G[o][k])})
        else:
            js, ks, gs = [], [], []
            for j, ps in enumerate(table.pieces):
                ok = np.nonzero(ps.trend_known & (ps.d_t > 0) & (ps.d_h >= -eps))[0]
                js.append(np.full(ok.size, j))
                ks.append(ok)
                gs.append(G[j][ok])
            if js and sum(k.size for k in ks):
                verdict = "refresh"
                js, ks, gs = np.concatenate(js), np.concatenate(ks), np.concatenate(gs)
                rows = np.array([table.pieces[j].rows[k] for j, k in zip(js, ks)])
                keys = cluster_keys(self.pool.broad[rows], self.pool.cluster[rows])
                suggestions = [self._suggestion(table, int(js[x]), int(ks[x]), gs[x])
                               for x in diverse_top(gs, keys, np.ones(gs.size, bool))]
        known = [p.trend for p in pieces if not np.isnan(p.trend)]
        return Recommendation(
            verdict=verdict, selection_rule=rule, pool_mode=table.mode, weights={"w_h": s.w_h, "w_t": w_t},
            thresholds={"tau": tau, "eps": eps},
            baseline={"harmony_pct": self.reference.pct(table.r0), "harmony_logit": float(table.r0),
                      "trend_mean": float(np.mean(known)) if known else None},
            swap_piece=swap_piece, suggestions=suggestions, also_improve=also, per_piece=per_piece,
            flags=list(table.state.flags), provisional=s.provisional())

    def recommend(self, outfit_dir, mode="demo", photo_px=None, **score_kw) -> Recommendation:
        state = load_outfit_dir(outfit_dir, self.gnn, self.trend_model, photo_px=photo_px)
        return self.decide(self.score(state, mode=mode, **score_kw))
```

Note: the refresh branch builds `rows` with a Python comprehension over the qualifying candidates (up to ~10⁵).
If the Task 10 timing shows it matters, replace it with
`np.concatenate([table.pieces[j].rows[ok] for j, ok in ...])`, collected in the same loop.

- [ ] **Step 5: Run the tests**

Run: `python -m pytest Module4/test_recommender_logic.py -v`
Expected: 10 passed

- [ ] **Step 6: Run the whole unit suite**

Run: `python -m pytest Module4 -m "not real" -q`
Expected: all pass

- [ ] **Step 7: Commit (ask the owner first)**

```bash
git add Module4/recommendation.py Module4/recommender.py Module4/test_recommender_logic.py
git commit -m "feat(m4): recommender with cached swap table, selection rules, verdicts and JSON output"
```

---

### Task 10: CLI, end-to-end smoke test and the D5 performance gate

**Files:**
- Create: `Module4/run_recommend.py`, `Module4/test_smoke_image2.py`, `Module4/test_performance.py`
- Modify: `Module4/conftest.py` (add `recommender` and `image2_dir` fixtures)

**Interfaces:**
- Consumes: `Recommender.from_settings`, `load_outfit_dir`, `photo_pixels`, `format_text`, M1's `run_single_image`.
- Produces: `segment_photo(photo, out_root, device=None) -> outfit_dir`; CLI `python Module4/run_recommend.py --photo samples/image2.jpg [--pool eval] [--rule ...] [--w-t ...] [--same-type] [--no-trend-drop] [--json out.json]`.

- [ ] **Step 1: Write `Module4/run_recommend.py`**

```python
"""Module 4 from the command line: photo (runs Module 1) or an M1 outfit folder -> recommendation.

    python Module4/run_recommend.py --photo samples/image2.jpg
    python Module4/run_recommend.py --outfit_dir <M1 output folder> --pool eval --json rec.json
"""
from __future__ import annotations

import argparse
import os
import sys
import time

for _s in (sys.stdout, sys.stderr):           # arrows/ticks on a Windows console
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from m4_settings import REPO, M4Settings  # noqa: E402
from outfit_state import load_outfit_dir  # noqa: E402
from recommendation import format_text  # noqa: E402
from recommender import Recommender  # noqa: E402
from selection import RULES  # noqa: E402


def segment_photo(photo: str, out_root: str, device=None) -> str:
    """Run Module 1 on one photo; returns its outfit folder (<out_root>/<photo stem>)."""
    m1 = os.path.join(REPO, "Module1")
    if m1 not in sys.path:
        sys.path.insert(0, m1)
    from run_pipeline import run_single_image
    image_id = os.path.splitext(os.path.basename(photo))[0]
    run_single_image(image_path=photo, image_id=image_id, output_root=out_root, device=device)
    return os.path.join(out_root, image_id)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--photo")
    src.add_argument("--outfit_dir")
    ap.add_argument("--pool", choices=["demo", "eval"], default="demo")
    ap.add_argument("--rule", choices=RULES)
    ap.add_argument("--w-t", type=float, dest="w_t")
    ap.add_argument("--same-type", action="store_true", help="strict like-for-like swaps")
    ap.add_argument("--no-trend-drop", action="store_true",
                    help="restore the hard trend floor (off by default, D5)")
    ap.add_argument("--json", help="write the Recommendation JSON here")
    ap.add_argument("--device")
    args = ap.parse_args(argv)

    s = M4Settings.load()
    rec = Recommender.from_settings(s, args.device)
    outfit_dir = args.outfit_dir or segment_photo(args.photo, s.abs(os.path.join(s.cache_dir, "runs")), args.device)
    t0 = time.time()
    state = load_outfit_dir(outfit_dir, rec.gnn, rec.trend_model)
    table = rec.score(state, mode=args.pool, same_type=args.same_type or None,
                      trend_floor=args.no_trend_drop or None)
    r = rec.decide(table, w_t=args.w_t, rule=args.rule)
    n = sum(len(p.rows) for p in table.pieces)
    print(format_text(r))
    print(f"[m4] {len(state.pieces)} pieces, {n} what-if outfits scored in {time.time() - t0:.2f}s")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(r.to_json())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add fixtures** to `Module4/conftest.py` (append):

```python
@pytest.fixture(scope="session")
def recommender(settings, device, pool, gnn):
    from recommender import Recommender
    return Recommender.from_settings(settings, device, pool=pool, gnn=gnn)


@pytest.fixture(scope="session")
def image2_dir(settings):
    """M1 output for samples/image2.jpg, computed once into the gitignored cache."""
    out_root = settings.abs(os.path.join(settings.cache_dir, "runs"))
    d = os.path.join(out_root, "image2")
    if not os.path.exists(os.path.join(d, "metadata.json")):
        from run_recommend import segment_photo
        segment_photo(settings.abs("samples/image2.jpg"), out_root)
    return d
```

- [ ] **Step 3: Write `Module4/test_smoke_image2.py`**

```python
import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from m2_bridge import ATTR_MASK, od
from outfit_state import REFERENCE_FRAME_PX, load_outfit_dir, photo_pixels
from recommendation import Recommendation

pytestmark = pytest.mark.real


def test_vis_overlay_has_the_input_photo_size(settings, image2_dir):
    with Image.open(settings.abs("samples/image2.jpg")) as im:
        assert photo_pixels(image2_dir) == im.size[0] * im.size[1]


def test_recommend_on_image2_returns_valid_recommendation(recommender, image2_dir, settings):
    r = recommender.recommend(image2_dir)
    assert r.verdict in {"swap", "refresh", "keep"}
    assert len(r.suggestions) <= 3
    assert len({s.pool_row for s in r.suggestions}) == len(r.suggestions)
    for s in r.suggestions:
        assert os.path.exists(settings.abs(s.crop_path))
        assert not os.path.isabs(s.crop_path)
    assert Recommendation.from_json(r.to_json()) == r
    assert 0.0 <= r.baseline["harmony_pct"] <= 100.0


def test_unswapped_logit_matches_m2_path(recommender, image2_dir):
    """M2 and M4 must agree on the user's own outfit (D10). M2 uses raw areas; the logit is
    resolution-invariant within one photo, so it equals M4's area-normalised logit closely."""
    with open(os.path.join(image2_dir, "metadata.json"), encoding="utf-8") as f:
        garments = json.load(f)
    nodes = [od.build_node_from_meta(image2_dir, g) for g in garments]
    g = od.OutfitDataset.build_graph(nodes, 0.0, attr_mask=ATTR_MASK)
    dev = recommender.gnn.device
    with torch.no_grad():
        m2 = float(recommender.gnn.model(g["x"].to(dev), g["edge_index"].to(dev), g["edge_attr"].to(dev))[0])
    st = load_outfit_dir(image2_dir, recommender.gnn, recommender.trend_model)
    raw = load_outfit_dir(image2_dir, recommender.gnn, recommender.trend_model, photo_px=REFERENCE_FRAME_PX)
    assert abs(recommender.scorer.outfit_logit(raw) - m2) < 1e-4       # unscaled areas: exact parity
    assert abs(recommender.scorer.outfit_logit(st) - m2) < 0.05         # +1 smoothing only
```

- [ ] **Step 4: Write `Module4/test_performance.py`**

```python
"""D5 gate: samples/image2.jpg, full demo pool, <= 5 s warm, peak VRAM <= 6 GB.
Run on its own for a clean VRAM reading:  python -m pytest Module4/test_performance.py -s"""
import statistics
import time

import pytest
import torch

pytestmark = pytest.mark.real


def test_image2_full_demo_pool_within_gate(recommender, image2_dir, settings):
    recommender.recommend(image2_dir)                      # warm-up
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(3):
        t = time.perf_counter()
        recommender.recommend(image2_dir)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    med = statistics.median(times)
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    print(f"\n[perf] warm recommend(): median {med:.2f}s (runs {', '.join(f'{x:.2f}' for x in times)}), "
          f"peak VRAM {peak:.2f} GiB")
    assert med <= 5.0
    assert peak <= settings.vram_gb
```

- [ ] **Step 5: Run the smoke and performance tests**

Run: `python -m pytest Module4/test_smoke_image2.py -v` then `python -m pytest Module4/test_performance.py -s`
Expected: 3 passed; then 1 passed with a `[perf]` line. The prototype predicts well under 2 s and about 2 GiB peak
(the pool tensors are ~1.3 GiB of that).

- [ ] **Step 6: Try the CLI**

Run: `python Module4/run_recommend.py --photo samples/image2.jpg --json Module4/cache/runs/image2_rec.json`
Expected: a verdict headline, up to 3 suggestion lines with a "what changed" line each, a per-piece table, the
provisional-settings note, and `[m4] … what-if outfits scored in …s`.

- [ ] **Step 7: Run every guard**

Run: `python -m pytest Module4 -q` then `python Module3/test_module1_sync.py` then `python Module3/test_embedding_parity.py`
Expected: all Module 4 tests pass; `IN SYNC`; `PARITY OK`

- [ ] **Step 8: Commit (ask the owner first)**

```bash
git add Module4/run_recommend.py Module4/test_smoke_image2.py Module4/test_performance.py Module4/conftest.py
git commit -m "feat(m4): CLI, image2 smoke test and D5 performance gate"
```

---

### Task 11: Documentation

**Files:**
- Modify: `Module4/README.md`, `docs/module4-design-detailed.md`, `docs/module4-design.md`, `docs/next-steps.md`, root `README.md` (Module 4 status line only)

- [ ] **Step 1: `Module4/README.md`.** Change the status to "core built (Fashion144k); evaluation next". Add, in
plain language with the technical terms present:
  - How to run (`build_cache.py`, `run_recommend.py`, tests, `-m "not real"`).
  - The file map from this plan.
  - The measured `[perf]` numbers from Task 10.
  - The `t_dup` chosen in Task 5 and why.
  - The provisional knobs and that step 3 calibrates them.
  - The "what changed" caveat (descriptive, not causal).
  - The area-normalisation note.

  Use `<data_root>`-style placeholders, never machine paths.
- [ ] **Step 2: `docs/module4-design-detailed.md`.** Correct the D5 text and §3:
  - "570→128" becomes "555→128 (attr-subset checkpoint)".
  - The projection is "480,079 × 128 = 246 MB (≈ 320 MB with Pinterest)".
  - Add the measured performance and note that the fallback pre-ranker was not needed.
  - Add the area normalisation to §5 and §2.1 step 0.2.
  - Add the seven clarifications from this plan to the relevant decisions.
  - Update the unit table to the actual file names.
- [ ] **Step 3: `docs/module4-design.md`.** Update the status line. One sentence on photo-size normalisation under "5. How many candidates are checked".
- [ ] **Step 4: `docs/next-steps.md` and root `README.md`.** Mark build step 2 done and point at step 3.
- [ ] **Step 5: Commit (ask the owner first)**

```bash
git add Module4/README.md docs/module4-design-detailed.md docs/module4-design.md docs/next-steps.md README.md
git commit -m "docs(m4): core build status, measured performance, spec corrections"
```

---

## VRAM budget (6 GB cap)

| Item | Fashion144k | + Pinterest (~144k rows) |
|---|---|---|
| CUDA context (not counted by the allocator) | ~0.3–0.5 GB | same |
| Pool `proj` 128 × fp32 | 0.25 GB | 0.32 GB |
| Pool `emb` 512 × fp32 (own-item exclusion; dedupe at build) | 0.98 GB | 1.28 GB |
| Pool scalars + ids + trend (float64/long) | ~0.02 GB | ~0.03 GB |
| Scoring chunk (sized from the cap; prototype peak for 823k outfits) | ≤ ~0.6 GB | same |
| Build only: dedupe sims chunk 2,048 × 94k fp32 + mask | ~0.97 GB | ~1.2 GB |
| **Request-time peak (allocator)** | **~1.9 GB** | **~2.3 GB** |

If headroom ever becomes tight, `emb` can drop to fp16 (halving it). The cosine-threshold tests would then need
a ±1e-3 tolerance band. The cap is still enforced by `set_per_process_memory_fraction`, and OOM halves the chunk.

## How Pinterest plugs in later (step 5, after Phase 0)

No Module 4 code changes. Phase 0 delivers, and M4 consumes, the following:

1. **`PinterestData/feature_store/`** in the exact Fashion144k layout (same builder, `row_dim` 570,
   `index.json` with `category`, `broad_category` and the relative `file`), plus `PinterestData/outputs/<photo_key>/` crops.
   **New Phase 0 requirement:** `node_scalars` area must be in the 98,304-px frame
   (`pixel_count × 98,304 / (width × height)` of the full-resolution photo), as M4 does for user photos.
   Otherwise every Pinterest candidate looks oversized next to Fashion144k and user pieces.
2. **`PinterestData/split.json`** = `{photo_key: "train" | "val" | "test"}` (Phase 0 step 0.3).
3. Add one entry to `Module4/m4_settings.json` → `stores`:
   `{"name": "pinterest", "store_dir": "PinterestData/feature_store", "output_root": "PinterestData/outputs", "split": "PinterestData/split.json"}`.
4. If Phase 0 refits M3 (D11) or fine-tunes M2 (AUC gate), point `trend_model` / `checkpoint` at the new files.
5. `python Module4/build_cache.py`. The fingerprint changes, so everything is rebuilt:
   - the projection for both sources;
   - trend + clusters from the refit M3;
   - **cross-source** dedupe (reposts across datasets merge);
   - co-occurrence from **both** train splits;
   - the harmony reference and σ's from **both** test splits.

   `source` flows into every suggestion card.
6. Re-run step 3's calibration (w_t sweep, τ/ε, rule ablation, t_dup sheets on the mixed pool) and freeze the
   new values. The tests in this plan stay valid. `test_candidate_pool.py`'s `pool.n == 480079` and spec-count
   assertions become per-source (filter on `pool.source == 0`).

## Not in this plan (later build steps)

- **Step 3, evaluation** (`eval_recovery.py`): dev/report sets, group-consistent corruption with
  severity strata, original recovery, baselines (random / trend-only / CLIP coherence), w_t Pareto sweep,
  τ/ε calibration, rule ablation, bootstrap CIs. It builds directly on `state_from_pool()`,
  `score(mode="eval", exclude=…)` and `decide()` (sweeps re-use one `SwapTable`).
- **Step 4, demo (D10):** `demo.py` integration, slider, "Try this swap" board, `render_recommendation.py`,
  M2 panel switched to `HarmonyReference`.
- **Step 5:** Pinterest (above), retune, human study.
- Multi-piece swaps (deferred; `OutfitState.with_swap` + `decide()` are the hooks). VLM judge (not approved).
