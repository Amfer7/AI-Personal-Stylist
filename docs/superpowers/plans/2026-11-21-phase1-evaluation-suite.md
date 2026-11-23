# Phase 1: Evaluation Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** build the measurements that decide whether a style-invariant garment embedding is better than today's CLIP + CORAL system, and record today's baseline on each of them.

**Architecture:** two independent pieces. In Module 4, a suggestion type-mix metric with a pass/fail threshold and a batch runner for real phone photos. In a new `Module1/style_encoder/` package, a DeepFashion2 consumer/shop pair loader, a cut-out renderer, retrieval and domain metrics, and an invariance evaluation CLI that takes any encoder with an `encode(images) -> (n, 512)` method (CLIP now, the style encoder in Phase 2).

**Tech Stack:** Python 3.11, PyTorch 2.9 (CUDA 12.6), OpenAI `clip` (ViT-B/32), NumPy, scikit-learn 1.7, Pillow, pytest.

Roadmap and later phases: `docs/superpowers/plans/2026-11-21-style-invariant-roadmap.md`.

## Global Constraints

- Run tests per module from the repo root: `python -m pytest Module4 -q`, `python -m pytest Module1/style_encoder -q`.
- Tests that need real data, model weights or a GPU carry `@pytest.mark.real`; the quick run is `-m "not real"`.
- Data never enters git: add every new data or cache folder to `.gitignore` before writing to it.
- Embeddings are 512-d float32 and L2-normalised; transparent crops are composited onto white before encoding (Module 1's contract, `_load_rgb` in `Module1/src/clip_extract_embeddings.py`).
- Module 4 must not touch `CandidatePool._is_rep` from outside the class; use the method added in Task 1.
- Log every result and decision in `docs/next-steps.md` as it happens.
- GPU memory: 8 GB; Module 4 respects `vram_gb` from `Module4/config/m4_settings.json`.

---

## File structure

| File | Responsibility |
|---|---|
| `Module4/src/candidate_pool.py` (modify) | `CandidatePool.set_demo_rep()`: swap which rows are demo representatives |
| `Module4/type_mix.py` (create) | Count suggested types per verdict; compare a demo pool with the street pool; CLI with a pass/fail threshold |
| `Module4/test_type_mix.py` (create) | Unit tests for the counting and comparison; one `real` smoke test |
| `Module4/phone_set.py` (create) | Run Module 1 + Module 4 on every photo in a folder; write one Recommendation JSON per photo and a summary |
| `Module4/test_phone_set.py` (create) | Unit tests for the photo manifest and summary |
| `docs/phone-photo-set.md` (create) | Capture protocol for the phone-photo set |
| `Module1/style_encoder/__init__.py` (create) | Package marker |
| `Module1/style_encoder/conftest.py` (create) | Import paths and the `real` marker |
| `Module1/style_encoder/deepfashion2.py` (create) | Load DeepFashion2 annotations into garment references; match consumer/shop pairs |
| `Module1/style_encoder/cutout.py` (create) | Polygon mask → transparent garment crop (Module 1's crop contract) |
| `Module1/style_encoder/metrics.py` (create) | Recall@k (category-restricted), pair cosine, domain classifier accuracy |
| `Module1/style_encoder/encoders.py` (create) | `ClipEncoder`: the baseline encoder behind the shared interface |
| `Module1/style_encoder/eval_invariance.py` (create) | Build crops, encode, compute the metrics, write a JSON report |
| `Module1/style_encoder/test_*.py` (create) | Tests for each of the above |
| `.gitignore` (modify) | `DeepFashion2/`, `PolyvoreOutfits/`, `Module1/style_encoder/cache/` |

---

### Task 1: Suggestion type-mix metric

**Files:**
- Modify: `Module4/src/candidate_pool.py` (class `CandidatePool`, after `empty_rows`)
- Create: `Module4/type_mix.py`
- Test: `Module4/test_type_mix.py`, `Module4/tests/test_candidate_pool.py`

**Interfaces:**
- Consumes: `Recommender.from_settings(settings)`, `Recommender.score(state, mode="demo", exclude=(source, key))`, `Recommender.decide(table)`, `state_from_pool(pool, source, key)`, `test_outfits(pool)`, `gate_demo_pool(settings, meta, scalars)`, `load_store_arrays(settings)`, `pool_dir(settings)`; `Recommendation.verdict`, `Recommendation.suggestions[i].to_type`.
- Produces:
  - `CandidatePool.set_demo_rep(rep_demo: np.ndarray) -> None`
  - `count_types(recs) -> dict` with keys `"verdicts"`, `"swap"`, `"refresh"`, each `{name: int}`
  - `share_ratios(test: dict, ref: dict, min_count: int = 5) -> dict[str, float]`
  - `worst_lean(ratios: dict) -> tuple[str | None, float]`
  - `run(settings, n: int, seed: int = 0) -> dict` with keys `"demo"`, `"street"`, each a `count_types` result
  - CLI `python Module4/type_mix.py --n 400` → JSON at `Module4/cache/eval/type_mix.json`; exit code 1 when the worst swap lean is above `--max_ratio` (default 1.5)

- [ ] **Step 1: Write the failing tests**

`Module4/test_type_mix.py`:
```python
from types import SimpleNamespace as NS

import numpy as np
import pytest

from type_mix import count_types, share_ratios, worst_lean


def _rec(verdict, *types):
    return NS(verdict=verdict, suggestions=[NS(to_type=t) for t in types])


def test_count_types_takes_the_top_swap_and_every_refresh_suggestion():
    recs = [_rec("swap", "shoe", "bag", "shoe"), _rec("swap", "pants"), _rec("refresh", "shoe", "t_shirt", "bag"),
            _rec("refresh", "pants"), _rec("keep")]
    c = count_types(recs)
    assert c["verdicts"] == {"swap": 2, "refresh": 2, "keep": 1}
    assert c["swap"] == {"shoe": 1, "pants": 1}
    assert c["refresh"] == {"shoe": 1, "t_shirt": 1, "bag": 1, "pants": 1}


def test_share_ratios_compare_shares_and_skip_rare_types():
    test = {"shoe": 18, "pants": 12, "bag": 7, "hat": 1}
    ref = {"shoe": 9, "pants": 16, "bag": 4, "hat": 2}
    r = share_ratios(test, ref, min_count=5)
    assert set(r) == {"shoe", "pants", "bag"}                 # hat: fewer than 5 on both sides
    assert r["shoe"] == pytest.approx((18 / 38) / (9 / 31))
    assert r["pants"] == pytest.approx((12 / 38) / (16 / 31))


def test_share_ratio_is_infinite_when_the_reference_never_suggests_the_type():
    assert share_ratios({"belt": 6, "shoe": 4}, {"shoe": 10}, min_count=5)["belt"] == float("inf")


def test_worst_lean_names_the_most_over_represented_type():
    assert worst_lean({"shoe": 1.57, "pants": 0.59, "bag": 1.37}) == ("shoe", 1.57)
    assert worst_lean({}) == (None, 1.0)


@pytest.mark.real
def test_run_returns_both_pools(settings):
    from type_mix import run
    res = run(settings, n=5)
    assert set(res) == {"demo", "street"}
    assert sum(res["demo"]["verdicts"].values()) == sum(res["street"]["verdicts"].values()) == 5
```

Add to `Module4/tests/test_candidate_pool.py`:
```python
def test_set_demo_rep_changes_which_rows_are_demo_candidates():
    import numpy as np
    from test_recommender_logic import make
    rec, _, _ = make()
    pool = rec.pool
    before = pool.candidate_rows(["shoe"], mode="demo").tolist()
    rep = pool.rep_demo.copy()
    rep[9] = -1                                     # row 9 (a shoe) is no longer a demo representative
    pool.set_demo_rep(rep)
    assert before == [9, 10]
    assert pool.candidate_rows(["shoe"], mode="demo").tolist() == [10]
    assert np.array_equal(pool.rep_demo, rep)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest Module4/test_type_mix.py Module4/tests/test_candidate_pool.py -q -m "not real"`
Expected: FAIL with `ModuleNotFoundError: No module named 'type_mix'` and `AttributeError: 'CandidatePool' object has no attribute 'set_demo_rep'`.

- [ ] **Step 3: Implement**

In `Module4/src/candidate_pool.py`, inside `class CandidatePool`, right after `empty_rows`:
```python
    def set_demo_rep(self, rep_demo: np.ndarray) -> None:
        """Replace the demo representatives (rep_demo[i] == i marks a demo row), e.g. to compare the
        H&M demo pool with the street-photo pool on the same outfits."""
        self.rep_demo = np.asarray(rep_demo)
        self._is_rep["demo"] = torch.as_tensor(self.rep_demo == np.arange(self.n), device=self.device)
```

`Module4/type_mix.py`:
```python
"""Suggestion type mix: which garment types Module 4 suggests, per verdict, on a set of test outfits.

A demo pool whose photo style differs from the GNN's training photos can tilt suggestions towards one
garment type (the H&M shoe lean, docs/next-steps.md 2026-11-21). This scores the same test outfits with
the demo pool and with the street-photo pool and compares, per type, its share of the suggestions.

    python Module4/type_mix.py --n 400                   # exit 1 if a type's swap share > 1.5x street
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def count_types(recs) -> dict:
    """Verdict counts; the top suggestion's type for each swap; every suggestion's type for refresh."""
    verdicts, swap, refresh = Counter(), Counter(), Counter()
    for r in recs:
        verdicts[r.verdict] += 1
        if r.verdict == "swap" and r.suggestions:
            swap[r.suggestions[0].to_type] += 1
        elif r.verdict == "refresh":
            refresh.update(s.to_type for s in r.suggestions)
    return {"verdicts": dict(verdicts), "swap": dict(swap), "refresh": dict(refresh)}


def share_ratios(test: dict, ref: dict, min_count: int = 5) -> dict:
    """{type: its share of test / its share of ref}, for types with at least min_count on either side."""
    nt, nr = sum(test.values()), sum(ref.values())
    out = {}
    if nt == 0 or nr == 0:
        return out
    for t in sorted(set(test) | set(ref)):
        a, b = test.get(t, 0), ref.get(t, 0)
        if max(a, b) < min_count:
            continue
        out[t] = float("inf") if b == 0 else (a / nt) / (b / nr)
    return out


def worst_lean(ratios: dict) -> tuple:
    if not ratios:
        return None, 1.0
    t = max(ratios, key=ratios.get)
    return t, ratios[t]


def run(settings, n: int, seed: int = 0) -> dict:
    """Score n random test outfits twice: with the configured demo pool and with the street pool."""
    from candidate_pool import gate_demo_pool, load_store_arrays, pool_dir
    from harmony_reference import test_outfits
    from outfit_state import state_from_pool
    from recommender import Recommender

    rec = Recommender.from_settings(settings)
    pool = rec.pool
    meta = dict(np.load(os.path.join(pool_dir(settings), "meta.npz")))
    _, scal = load_store_arrays(settings)
    street = [s.name for s in settings.stores if not s.product_photos]
    reps = {"demo": pool.rep_demo.copy(),
            "street": gate_demo_pool(settings.with_(demo_sources=street), meta, scal)["rep_demo"]}
    outs = test_outfits(pool)
    outs = [outs[i] for i in np.random.default_rng(seed).permutation(len(outs))[:n]]
    result = {}
    try:
        for name, rep in reps.items():
            pool.set_demo_rep(rep)
            recs = []
            for rows in outs:
                src = pool.stores[int(pool.source[rows[0]])].name
                key = pool.outfit_key(int(rows[0]))
                recs.append(rec.decide(rec.score(state_from_pool(pool, src, key), mode="demo", exclude=(src, key))))
            result[name] = count_types(recs)
    finally:
        pool.set_demo_rep(reps["demo"])
    return result


def main(argv=None) -> int:
    from m4_settings import M4Settings
    s = M4Settings.load()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=400, help="test outfits")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_count", type=int, default=5)
    ap.add_argument("--max_ratio", type=float, default=1.5)
    ap.add_argument("--out", default=s.abs(os.path.join(s.cache_dir, "eval", "type_mix.json")))
    args = ap.parse_args(argv)
    res = run(s, args.n, args.seed)
    for verdict in ("swap", "refresh"):
        res[f"{verdict}_ratios"] = share_ratios(res["demo"][verdict], res["street"][verdict], args.min_count)
    t, r = worst_lean(res["swap_ratios"])
    res["worst_swap_lean"] = {"type": t, "ratio": r, "max_ratio": args.max_ratio, "pass": r <= args.max_ratio}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("swap_ratios", "refresh_ratios", "worst_swap_lean")}, indent=1))
    return 0 if r <= args.max_ratio else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest Module4/test_type_mix.py Module4/tests/test_candidate_pool.py -q -m "not real"`
Expected: PASS.
Then the real smoke test: `python -m pytest Module4/test_type_mix.py -q -m real` → PASS (needs the built cache).

- [ ] **Step 5: Commit**

```bash
git add Module4/src/candidate_pool.py Module4/type_mix.py Module4/test_type_mix.py Module4/tests/test_candidate_pool.py
git commit -m "Module 4: suggestion type-mix metric with a street-pool comparison"
```

---

### Task 2: DeepFashion2 pair loader

**Files:**
- Modify: `.gitignore`
- Create: `Module1/style_encoder/__init__.py`, `Module1/style_encoder/conftest.py`, `Module1/style_encoder/deepfashion2.py`
- Test: `Module1/style_encoder/test_deepfashion2.py`

**Interfaces:**
- Consumes: DeepFashion2 on disk as `<root>/<split>/annos/<id>.json` and `<root>/<split>/image/<id>.jpg`. Each annotation JSON has `"source"` (`"user"` or `"shop"`), `"pair_id"` (int) and `"item1"`, `"item2"`, … each with `"category_name"`, `"style"` (0 = no pair match), `"bounding_box"` `[x1, y1, x2, y2]` and `"segmentation"` (list of flat polygons `[x1, y1, x2, y2, …]`). A consumer item and a shop item with the same `pair_id`, the same `style > 0` and the same category are the same garment.
- Produces:
  - `GarmentRef` (frozen dataclass): `image: str, item: str, source: str, pair_id: int, style: int, category: str, box: tuple, polygons: tuple`
  - `load_refs(root: str, split: str = "validation") -> list[GarmentRef]` (sorted by annotation file, then item key)
  - `match_pairs(refs: list[GarmentRef]) -> list[tuple[int, int]]` — (consumer index, shop index)

- [ ] **Step 1: Add the ignore rules and the package files**

Append to `.gitignore`, under the datasets block:
```
DeepFashion2/
PolyvoreOutfits/
Module1/style_encoder/cache/
Module1/style_encoder/checkpoints/
```

`Module1/style_encoder/__init__.py`:
```python
"""Style-invariant garment embedding: data, metrics and evaluation (roadmap 2026-11-21)."""
```

`Module1/style_encoder/conftest.py`:
```python
"""pytest setup for the style encoder: import paths and the `real` marker.

Tests marked `real` need DeepFashion2 under <repo>/DeepFashion2 and the CLIP weights.
Quick run: python -m pytest Module1/style_encoder -m "not real"
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE1 = os.path.dirname(HERE)
for p in (HERE, MODULE1):
    if p not in sys.path:
        sys.path.insert(0, p)


def pytest_configure(config):
    config.addinivalue_line("markers", "real: needs DeepFashion2, model weights or a GPU")
```

- [ ] **Step 2: Write the failing test**

`Module1/style_encoder/test_deepfashion2.py`:
```python
import json
import os

from deepfashion2 import GarmentRef, load_refs, match_pairs


def _write(root, split, name, anno):
    d = os.path.join(root, split, "annos")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(anno, f)


def _item(cat, style, box=(10, 10, 50, 60)):
    x1, y1, x2, y2 = box
    return {"category_name": cat, "style": style, "bounding_box": list(box),
            "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2]]}


def test_load_refs_reads_every_item_with_its_image_path(tmp_path):
    root = str(tmp_path)
    _write(root, "validation", "000001", {"source": "user", "pair_id": 7,
                                          "item1": _item("trousers", 1), "item2": _item("vest", 0)})
    refs = load_refs(root, "validation")
    assert [(r.item, r.category, r.style) for r in refs] == [("item1", "trousers", 1), ("item2", "vest", 0)]
    assert refs[0].image == os.path.join(root, "validation", "image", "000001.jpg")
    assert refs[0].source == "user" and refs[0].pair_id == 7
    assert refs[0].box == (10, 10, 50, 60)
    assert refs[0].polygons == ((10, 10, 50, 10, 50, 60, 10, 60),)


def test_match_pairs_needs_same_pair_style_and_category(tmp_path):
    root = str(tmp_path)
    _write(root, "validation", "000001", {"source": "user", "pair_id": 1,
                                          "item1": _item("trousers", 1), "item2": _item("vest", 0)})
    _write(root, "validation", "000002", {"source": "shop", "pair_id": 1, "item1": _item("trousers", 1)})
    _write(root, "validation", "000003", {"source": "shop", "pair_id": 1, "item1": _item("skirt", 1)})
    _write(root, "validation", "000004", {"source": "shop", "pair_id": 2, "item1": _item("trousers", 1)})
    refs = load_refs(root, "validation")
    assert match_pairs(refs) == [(0, 2)]       # style 0 never pairs; other category / pair_id do not match
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `python -m pytest Module1/style_encoder/test_deepfashion2.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'deepfashion2'`.

- [ ] **Step 4: Implement**

`Module1/style_encoder/deepfashion2.py`:
```python
"""DeepFashion2 consumer/shop garments: one GarmentRef per annotated item, and the matching pairs.

A consumer ("user") item and a shop item are the same garment when they share pair_id, a style above
0 and the category. Layout: <root>/<split>/annos/<id>.json and <root>/<split>/image/<id>.jpg.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class GarmentRef:
    image: str
    item: str
    source: str        # "user" | "shop"
    pair_id: int
    style: int         # 0 = no consumer/shop match
    category: str
    box: tuple         # (x1, y1, x2, y2)
    polygons: tuple    # flat polygons (x1, y1, x2, y2, ...)


def load_refs(root: str, split: str = "validation") -> list:
    anno_dir, image_dir = os.path.join(root, split, "annos"), os.path.join(root, split, "image")
    refs = []
    for name in sorted(os.listdir(anno_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(anno_dir, name), encoding="utf-8") as f:
            anno = json.load(f)
        image = os.path.join(image_dir, os.path.splitext(name)[0] + ".jpg")
        for key in sorted(k for k in anno if k.startswith("item")):
            it = anno[key]
            refs.append(GarmentRef(image=image, item=key, source=anno["source"], pair_id=int(anno["pair_id"]),
                                   style=int(it["style"]), category=it["category_name"],
                                   box=tuple(it["bounding_box"]),
                                   polygons=tuple(tuple(p) for p in it["segmentation"])))
    return refs


def match_pairs(refs: list) -> list:
    shop = defaultdict(list)
    for j, r in enumerate(refs):
        if r.source == "shop" and r.style > 0:
            shop[(r.pair_id, r.style, r.category)].append(j)
    return [(i, j) for i, r in enumerate(refs) if r.source == "user" and r.style > 0
            for j in shop.get((r.pair_id, r.style, r.category), [])]
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest Module1/style_encoder/test_deepfashion2.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add .gitignore Module1/style_encoder/__init__.py Module1/style_encoder/conftest.py Module1/style_encoder/deepfashion2.py Module1/style_encoder/test_deepfashion2.py
git commit -m "Style encoder: DeepFashion2 consumer/shop pair loader"
```

---

### Task 3: Garment cut-out renderer

**Files:**
- Create: `Module1/style_encoder/cutout.py`
- Test: `Module1/style_encoder/test_cutout.py`

**Interfaces:**
- Consumes: `GarmentRef.polygons`, `GarmentRef.box` (Task 2).
- Produces: `polygon_cutout(image: PIL.Image.Image, polygons, box, pad: float = 0.05) -> PIL.Image.Image` — RGBA, alpha 255 inside the polygons and 0 outside, cropped to the box grown by `pad` of its width/height on each side and clipped to the image. Same contract as Module 1's transparent garment crops.

- [ ] **Step 1: Write the failing test**

`Module1/style_encoder/test_cutout.py`:
```python
import numpy as np
from PIL import Image

from cutout import polygon_cutout


def test_cutout_is_transparent_outside_the_polygon_and_padded_to_the_box():
    img = Image.new("RGB", (100, 100), (200, 30, 30))
    out = polygon_cutout(img, [(20, 20, 60, 20, 60, 60, 20, 60)], (20, 20, 60, 60), pad=0.05)
    assert out.mode == "RGBA"
    assert out.size == (44, 44)                         # 40 px box + 2 px each side
    a = np.asarray(out)[:, :, 3]
    assert a[22, 22] == 255 and a[0, 0] == 0            # inside the polygon / in the padding
    assert tuple(np.asarray(out)[22, 22, :3]) == (200, 30, 30)


def test_cutout_clips_the_padding_at_the_image_edge():
    img = Image.new("RGB", (50, 50), (0, 0, 255))
    out = polygon_cutout(img, [(0, 0, 50, 0, 50, 50, 0, 50)], (0, 0, 50, 50), pad=0.2)
    assert out.size == (50, 50)


def test_degenerate_polygons_are_skipped():
    img = Image.new("RGB", (40, 40), (0, 255, 0))
    out = polygon_cutout(img, [(1, 1, 2, 2), (5, 5, 30, 5, 30, 30, 5, 30)], (5, 5, 30, 30), pad=0.0)
    assert np.asarray(out)[10, 10, 3] == 255
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest Module1/style_encoder/test_cutout.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'cutout'`.

- [ ] **Step 3: Implement**

`Module1/style_encoder/cutout.py`:
```python
"""Polygon mask -> transparent garment crop, the same RGBA contract as Module 1's crops."""
from __future__ import annotations

from PIL import Image, ImageDraw


def polygon_cutout(image: Image.Image, polygons, box, pad: float = 0.05) -> Image.Image:
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    for p in polygons:
        if len(p) >= 6:                                  # at least three points
            draw.polygon(list(zip(p[0::2], p[1::2])), fill=255)
    rgba = image.convert("RGBA")
    rgba.putalpha(mask)
    x1, y1, x2, y2 = box
    px, py = pad * (x2 - x1), pad * (y2 - y1)
    crop = (max(0, int(round(x1 - px))), max(0, int(round(y1 - py))),
            min(image.width, int(round(x2 + px))), min(image.height, int(round(y2 + py))))
    return rgba.crop(crop)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest Module1/style_encoder/test_cutout.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Module1/style_encoder/cutout.py Module1/style_encoder/test_cutout.py
git commit -m "Style encoder: polygon cut-out renderer"
```

---

### Task 4: Invariance metrics

**Files:**
- Create: `Module1/style_encoder/metrics.py`
- Test: `Module1/style_encoder/test_metrics.py`

**Interfaces:**
- Produces:
  - `recall_at_k(q: np.ndarray, g: np.ndarray, gt: np.ndarray, ks=(1, 5, 10), q_groups=None, g_groups=None) -> dict[int, float]` — rows unit-normalised; with groups, only same-group gallery items compete; a tie with the target counts in the target's favour.
  - `pair_cosine(a: np.ndarray, b: np.ndarray) -> float` — mean row-wise cosine.
  - `domain_classifier_acc(a: np.ndarray, b: np.ndarray, folds: int = 5, seed: int = 0) -> float` — cross-validated logistic-regression accuracy at telling `a` rows from `b` rows, on a class-balanced subsample (0.5 = indistinguishable).

- [ ] **Step 1: Write the failing test**

`Module1/style_encoder/test_metrics.py`:
```python
import numpy as np
import pytest

from metrics import domain_classifier_acc, pair_cosine, recall_at_k


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_recall_counts_the_rank_of_the_true_match():
    g = np.eye(4)
    q = _unit(np.array([[1, 0.1, 0, 0], [0.9, 1, 0, 0], [0, 0, 0.2, 1], [0, 0, 1, 0.9]], float))
    gt = np.array([0, 0, 2, 2])                        # rows 1 and 2: the match ranks second
    r = recall_at_k(q, g, gt, ks=(1, 2))
    assert r == {1: 0.5, 2: 1.0}


def test_recall_within_groups_ignores_other_categories():
    g = np.eye(3)
    q = _unit(np.array([[1, 0.2, 0], [1, 0.9, 0]], float))
    gt = np.array([1, 1])                              # query 0 prefers item 0, which is another category
    r = recall_at_k(q, g, gt, ks=(1,), q_groups=["a", "a"], g_groups=["b", "a", "a"])
    assert r == {1: 1.0}


def test_pair_cosine_is_the_mean_rowwise_cosine():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert pair_cosine(a, b) == pytest.approx(0.5)


def test_domain_classifier_separates_shifted_clouds_and_not_identical_ones():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((1000, 16))
    assert domain_classifier_acc(a, rng.standard_normal((1000, 16)) + 3.0) > 0.95
    acc = domain_classifier_acc(a, rng.standard_normal((1000, 16)))
    assert 0.4 < acc < 0.6
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest Module1/style_encoder/test_metrics.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'metrics'`.

- [ ] **Step 3: Implement**

`Module1/style_encoder/metrics.py`:
```python
"""Metrics for photo-style invariance: consumer->shop retrieval, pair cosine, domain separability."""
from __future__ import annotations

import numpy as np


def recall_at_k(q, g, gt, ks=(1, 5, 10), q_groups=None, g_groups=None) -> dict:
    sims = np.asarray(q, np.float32) @ np.asarray(g, np.float32).T
    if q_groups is not None:
        same = np.asarray(q_groups)[:, None] == np.asarray(g_groups)[None, :]
        sims = np.where(same, sims, -np.inf)
    gt = np.asarray(gt)
    target = sims[np.arange(len(gt)), gt]
    rank = (sims > target[:, None]).sum(1)
    return {int(k): float((rank < k).mean()) for k in ks}


def pair_cosine(a, b) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return float((a * b).sum(1).mean())


def domain_classifier_acc(a, b, folds: int = 5, seed: int = 0) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    rng = np.random.default_rng(seed)
    n = min(len(a), len(b))
    x = np.concatenate([np.asarray(a)[rng.choice(len(a), n, replace=False)],
                        np.asarray(b)[rng.choice(len(b), n, replace=False)]])
    y = np.repeat([0, 1], n)
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    return float(cross_val_score(LogisticRegression(max_iter=2000), x, y, cv=cv).mean())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest Module1/style_encoder/test_metrics.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Module1/style_encoder/metrics.py Module1/style_encoder/test_metrics.py
git commit -m "Style encoder: retrieval, pair cosine and domain classifier metrics"
```

---

### Task 5: Encoder interface and the invariance evaluation

**Files:**
- Create: `Module1/style_encoder/encoders.py`, `Module1/style_encoder/eval_invariance.py`
- Test: `Module1/style_encoder/test_eval_invariance.py`

**Interfaces:**
- Consumes: `load_refs`, `match_pairs`, `GarmentRef` (Task 2); `polygon_cutout` (Task 3); `recall_at_k`, `pair_cosine`, `domain_classifier_acc` (Task 4); `get_clip_model(device)` and `_load_rgb(image)` from `Module1/src/clip_extract_embeddings.py`.
- Produces:
  - Encoder protocol: an object with `name: str` and `encode(images: list[PIL.Image.Image]) -> np.ndarray` of shape (n, 512), float32, unit rows. Phase 2's `StyleEncoder` implements the same protocol.
  - `ClipEncoder(device=None, batch=64)`
  - `crop_paths(refs, idx, split: str, cache_dir: str) -> list[str]` — renders and caches each cut-out as PNG once
  - `evaluate(encoder, refs, pairs, split: str, cache_dir: str, max_pairs: int = 5000, seed: int = 0) -> dict` with keys `encoder`, `pairs`, `users`, `shops`, `recall` (`{1, 5, 10}`), `recall_by_category`, `pair_cos`, `nonpair_cos`, `domain_acc`
  - CLI: `python Module1/style_encoder/eval_invariance.py --encoder clip` → `Module1/style_encoder/cache/runs/<encoder>_<split>.json`

- [ ] **Step 1: Write the failing test**

`Module1/style_encoder/test_eval_invariance.py`:
```python
import json
import os

import numpy as np
import pytest
from PIL import Image

from deepfashion2 import load_refs, match_pairs
from eval_invariance import evaluate


class MeanColourEncoder:
    """Test double: a garment's embedding is its mean colour on white."""
    name = "mean-colour"

    def encode(self, images):
        out = []
        for im in images:
            rgba = im.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            rgb = np.asarray(Image.alpha_composite(white, rgba).convert("RGB"), np.float64).reshape(-1, 3)
            v = 255.0 - rgb.mean(0) + 1e-3                 # distance from white, so colours separate
            out.append(v / np.linalg.norm(v))
        return np.asarray(out, np.float32)


def _make_df2(root):
    colours = [(220, 20, 20), (20, 200, 20), (20, 20, 220)]
    for split in ("validation",):
        os.makedirs(os.path.join(root, split, "annos"), exist_ok=True)
        os.makedirs(os.path.join(root, split, "image"), exist_ok=True)
    n = 0
    for pid, c in enumerate(colours, start=1):
        for source in ("user", "shop"):
            n += 1
            name = f"{n:06d}"
            img = Image.new("RGB", (64, 64), (240, 240, 240))
            img.paste(c, (16, 16, 48, 48))
            img.save(os.path.join(root, "validation", "image", name + ".jpg"), quality=95)
            item = {"category_name": "trousers", "style": 1, "bounding_box": [16, 16, 48, 48],
                    "segmentation": [[16, 16, 48, 16, 48, 48, 16, 48]]}
            with open(os.path.join(root, "validation", "annos", name + ".json"), "w", encoding="utf-8") as f:
                json.dump({"source": source, "pair_id": pid, "item1": item}, f)


def test_evaluate_finds_every_true_pair_with_a_separating_encoder(tmp_path):
    root = str(tmp_path / "df2")
    _make_df2(root)
    refs = load_refs(root, "validation")
    pairs = match_pairs(refs)
    assert len(pairs) == 3
    rep = evaluate(MeanColourEncoder(), refs, pairs, "validation", str(tmp_path / "cache"), max_pairs=10)
    assert rep["pairs"] == 3 and rep["users"] == 3 and rep["shops"] == 3
    assert rep["recall"][1] == 1.0
    assert rep["pair_cos"] > rep["nonpair_cos"]
    assert set(rep["recall_by_category"]) == {"trousers"}
    assert os.path.exists(os.path.join(str(tmp_path / "cache"), "crops", "validation", "000001_item1.png"))


@pytest.mark.real
def test_clip_encoder_returns_unit_512_vectors():
    from encoders import ClipEncoder
    e = ClipEncoder()
    x = e.encode([Image.new("RGB", (64, 64), (200, 30, 30)), Image.new("RGBA", (64, 64), (0, 0, 200, 255))])
    assert x.shape == (2, 512) and x.dtype == np.float32
    assert np.allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-3)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest Module1/style_encoder/test_eval_invariance.py -q -m "not real"`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval_invariance'`.

- [ ] **Step 3: Implement**

`Module1/style_encoder/encoders.py`:
```python
"""Garment encoders behind one interface: name + encode(images) -> (n, 512) float32 unit rows."""
from __future__ import annotations

import numpy as np
import torch


class ClipEncoder:
    """OpenAI CLIP ViT-B/32, exactly as Module 1 embeds garments (transparent crops on white)."""
    name = "clip-vit-b32"

    def __init__(self, device=None, batch: int = 64):
        from clip_extract_embeddings import get_clip_model
        self.model, self.preprocess, self.device = get_clip_model(device=device)
        self.batch = batch

    def encode(self, images) -> np.ndarray:
        from clip_extract_embeddings import _load_rgb
        out = []
        for a in range(0, len(images), self.batch):
            x = torch.stack([self.preprocess(_load_rgb(im)) for im in images[a:a + self.batch]]).to(self.device)
            with torch.no_grad():
                e = self.model.encode_image(x).float()
            out.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 512), np.float32)


ENCODERS = {"clip": ClipEncoder}
```

`Module1/style_encoder/eval_invariance.py`:
```python
"""Photo-style invariance on DeepFashion2: does an encoder put a consumer photo of a garment next to
the shop photo of the same garment?

Reports consumer->shop Recall@1/5/10 (gallery = the shop items of the sampled pairs, same category
only), mean cosine of true pairs vs same-category non-pairs, and how well a linear classifier tells
consumer from shop embeddings (0.5 = indistinguishable).

    python Module1/style_encoder/eval_invariance.py --encoder clip
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE1 = os.path.dirname(HERE)
REPO = os.path.dirname(MODULE1)
for p in (HERE, MODULE1):
    if p not in sys.path:
        sys.path.insert(0, p)

from cutout import polygon_cutout  # noqa: E402
from deepfashion2 import load_refs, match_pairs  # noqa: E402
from metrics import domain_classifier_acc, pair_cosine, recall_at_k  # noqa: E402


def crop_paths(refs, idx, split: str, cache_dir: str) -> list:
    paths = []
    for i in idx:
        r = refs[i]
        stem = os.path.splitext(os.path.basename(r.image))[0]
        p = os.path.join(cache_dir, "crops", split, f"{stem}_{r.item}.png")
        if not os.path.exists(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with Image.open(r.image) as im:
                polygon_cutout(im.convert("RGB"), r.polygons, r.box).save(p)
        paths.append(p)
    return paths


def _encode(encoder, paths) -> np.ndarray:
    images = []
    for p in paths:
        with Image.open(p) as im:
            images.append(im.copy())
    return encoder.encode(images)


def evaluate(encoder, refs, pairs, split: str, cache_dir: str, max_pairs: int = 5000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    pairs = [pairs[i] for i in rng.permutation(len(pairs))[:max_pairs]]
    users = sorted({u for u, _ in pairs})
    shops = sorted({s for _, s in pairs})
    u_at = {r: k for k, r in enumerate(users)}
    s_at = {r: k for k, r in enumerate(shops)}
    U = _encode(encoder, crop_paths(refs, users, split, cache_dir))
    S = _encode(encoder, crop_paths(refs, shops, split, cache_dir))
    q = U[[u_at[u] for u, _ in pairs]]
    gt = np.array([s_at[s] for _, s in pairs])
    q_cat = [refs[u].category for u, _ in pairs]
    g_cat = [refs[s].category for s in shops]
    by_cat = defaultdict(list)
    for j, c in enumerate(g_cat):
        by_cat[c].append(j)
    neg = []
    for k in range(len(q)):
        others = [j for j in by_cat[q_cat[k]] if j != gt[k]]
        if others:
            neg.append(float(q[k] @ S[rng.choice(others)]))
    recall_by_category = {}
    for c in sorted(set(q_cat)):
        m = np.array([x == c for x in q_cat])
        recall_by_category[c] = {"queries": int(m.sum()),
                                 **recall_at_k(q[m], S, gt[m], q_groups=[c] * int(m.sum()), g_groups=g_cat)}
    return {"encoder": encoder.name, "pairs": len(pairs), "users": len(users), "shops": len(shops),
            "recall": recall_at_k(q, S, gt, q_groups=q_cat, g_groups=g_cat),
            "recall_by_category": recall_by_category,
            "pair_cos": pair_cosine(q, S[gt]), "nonpair_cos": float(np.mean(neg)) if neg else float("nan"),
            "domain_acc": domain_classifier_acc(U, S) if min(len(U), len(S)) >= 10 else float("nan")}


def main(argv=None) -> None:
    from encoders import ENCODERS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--df2_root", default=os.path.join(REPO, "DeepFashion2"))
    ap.add_argument("--split", default="validation")
    ap.add_argument("--encoder", choices=sorted(ENCODERS), default="clip")
    ap.add_argument("--max_pairs", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_dir", default=os.path.join(HERE, "cache"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    refs = load_refs(args.df2_root, args.split)
    pairs = match_pairs(refs)
    print(f"[invariance] {len(refs)} garments, {len(pairs)} consumer/shop pairs in {args.split}")
    rep = evaluate(ENCODERS[args.encoder](), refs, pairs, args.split, args.cache_dir, args.max_pairs, args.seed)
    out = args.out or os.path.join(args.cache_dir, "runs", f"{args.encoder}_{args.split}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: rep[k] for k in ("pairs", "recall", "pair_cos", "nonpair_cos", "domain_acc")}, indent=1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest Module1/style_encoder -q -m "not real"`
Expected: PASS.
Run: `python -m pytest Module1/style_encoder -q -m real -k clip_encoder`
Expected: PASS (loads the cached CLIP weights).

- [ ] **Step 5: Commit**

```bash
git add Module1/style_encoder/encoders.py Module1/style_encoder/eval_invariance.py Module1/style_encoder/test_eval_invariance.py
git commit -m "Style encoder: encoder interface and DeepFashion2 invariance evaluation"
```

---

### Task 6: Phone-photo set protocol and batch runner

**Files:**
- Create: `docs/phone-photo-set.md`, `Module4/phone_set.py`
- Test: `Module4/test_phone_set.py`

**Interfaces:**
- Consumes: `segment_photo(photo, out_root, device=None) -> str` from `Module4/src/run_recommend.py`; `Recommender.from_settings(settings)`; `Recommender.recommend(outfit_dir) -> Recommendation`; `Recommendation.to_json()`, `.verdict`, `.suggestions[i].to_type`, `.baseline["harmony_pct"]`.
- Produces:
  - `photo_manifest(folder: str) -> list[str]` — sorted `.jpg` / `.jpeg` / `.png` paths
  - `summary_row(photo: str, rec) -> dict` with keys `photo`, `verdict`, `harmony_pct`, `types`
  - CLI `python Module4/phone_set.py --tag baseline` → one JSON per photo in `Module4/cache/eval/phone_set/<tag>/` plus `summary.json`

- [ ] **Step 1: Write the capture protocol**

`docs/phone-photo-set.md`:
```markdown
# Phone-photo set

Real outfit photos taken the way a user would take them, for the style-invariance evaluation and the
Phase 4 human study. Stored in `samples/phone_set/` (gitignored); never committed or shared outside
the team.

## Consent
Everyone photographed agrees in writing that the photo is used only for this project's evaluation
and deleted afterwards. Faces may be cropped out before the photo is added.

## What to capture
- At least 60 photos from at least 10 people; each person in 2-3 different outfits.
- Full outfit visible, head to shoes; one person per photo.
- Spread over these conditions (roughly equal counts):
  | Light | Where |
  |---|---|
  | warm indoor lamp | home |
  | window daylight | home or office |
  | overhead fluorescent | office, shop |
  | outdoor shade | street |
  | outdoor direct sun | street |
- Both mirror selfies and photos taken by someone else.
- Plain and cluttered backgrounds.
- Phone camera defaults: no filters, no portrait mode, no editing.

## Naming
`<person id>_<outfit number>_<light>.jpg`, for example `p03_2_window.jpg`. Keep a sheet
(`samples/phone_set/conditions.csv`: file, person, light, where, selfie yes/no) so results can be
split by condition.
```

- [ ] **Step 2: Write the failing test**

`Module4/test_phone_set.py`:
```python
from types import SimpleNamespace as NS

from phone_set import photo_manifest, summary_row


def test_manifest_lists_photos_only_in_name_order(tmp_path):
    for name in ("p02_1_sun.jpg", "p01_1_lamp.JPG", "notes.txt", "p01_2_window.png", "conditions.csv"):
        (tmp_path / name).write_bytes(b"x")
    names = [p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in photo_manifest(str(tmp_path))]
    assert names == ["p01_1_lamp.JPG", "p01_2_window.png", "p02_1_sun.jpg"]


def test_summary_row_records_verdict_harmony_and_suggested_types():
    rec = NS(verdict="refresh", baseline={"harmony_pct": 71.4},
             suggestions=[NS(to_type="pants"), NS(to_type="shoe")])
    assert summary_row("p01_1_lamp.jpg", rec) == {"photo": "p01_1_lamp.jpg", "verdict": "refresh",
                                                  "harmony_pct": 71.4, "types": ["pants", "shoe"]}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `python -m pytest Module4/test_phone_set.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'phone_set'`.

- [ ] **Step 4: Implement**

`Module4/phone_set.py`:
```python
"""Run Module 1 + Module 4 on every photo of the phone-photo set (docs/phone-photo-set.md).

    python Module4/phone_set.py --tag baseline        # samples/phone_set -> cache/eval/phone_set/baseline/
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

PHOTO_EXT = (".jpg", ".jpeg", ".png")


def photo_manifest(folder: str) -> list:
    return sorted((os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(PHOTO_EXT)),
                  key=lambda p: os.path.basename(p).lower())


def summary_row(photo: str, rec) -> dict:
    return {"photo": os.path.basename(photo), "verdict": rec.verdict,
            "harmony_pct": rec.baseline["harmony_pct"], "types": [s.to_type for s in rec.suggestions]}


def main(argv=None) -> None:
    from m4_settings import M4Settings
    from recommender import Recommender
    from run_recommend import segment_photo
    s = M4Settings.load()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", default=s.abs(os.path.join("samples", "phone_set")))
    ap.add_argument("--tag", required=True, help="run name, e.g. baseline")
    args = ap.parse_args(argv)
    out_dir = s.abs(os.path.join(s.cache_dir, "eval", "phone_set", args.tag))
    seg_root = s.abs(os.path.join(s.cache_dir, "runs", "phone_set"))
    os.makedirs(out_dir, exist_ok=True)
    rec = Recommender.from_settings(s)
    rows = []
    for photo in photo_manifest(args.photos):
        r = rec.recommend(segment_photo(photo, seg_root))
        stem = os.path.splitext(os.path.basename(photo))[0]
        with open(os.path.join(out_dir, stem + ".json"), "w", encoding="utf-8") as f:
            f.write(r.to_json())
        rows.append(summary_row(photo, r))
        print(f"[phone_set] {rows[-1]}")
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest Module4/test_phone_set.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add docs/phone-photo-set.md Module4/phone_set.py Module4/test_phone_set.py
git commit -m "Module 4: phone-photo set protocol and batch runner"
```

---

### Task 7: Record the baseline

**Files:**
- Modify: `docs/next-steps.md` (new subsection under section 1, "Style-invariant embedding: Phase 1 baseline")

- [ ] **Step 1: Run the full test suites**

Run: `python -m pytest Module4 -q` and `python -m pytest Module1/style_encoder -q`
Expected: all pass (the `real` tests need the cache, CLIP weights and DeepFashion2).

- [ ] **Step 2: Type-mix baseline**

Run: `python Module4/type_mix.py --n 400`
Expected: exit code 1 today. On 2026-11-21 the H&M pool's shoe swap share was 18/57 vs 9/56 for street, a ratio of about 2.0 under this metric's denominators, above the 1.5 threshold. Record the JSON's `swap_ratios`, `refresh_ratios` and `worst_swap_lean`.

- [ ] **Step 3: Invariance baseline (needs DeepFashion2)**

Run: `python Module1/style_encoder/eval_invariance.py --encoder clip --max_pairs 5000`
Then with seeds 1 and 2 (`--seed 1 --out Module1/style_encoder/cache/runs/clip_validation_s1.json`, same for 2) to measure the spread the Phase 2 gate compares against.
Record: pairs, Recall@1/5/10 (overall and per category), pair vs non-pair cosine, domain accuracy, each as mean ± spread over the three seeds.

- [ ] **Step 4: Domain-check baseline**

Already measured on 2026-11-21 (`H&M Dataset/store/domain_check_coral.json`): classifier 0.60, stand-in rank 0.36 vs 0.28. Copy the numbers into the new subsection so every baseline sits in one place.

- [ ] **Step 5: Phone-set baseline (needs the photos)**

Run: `python Module4/phone_set.py --tag baseline`
Record: verdict counts, suggested type counts, median harmony %, split by light condition using `samples/phone_set/conditions.csv`.

- [ ] **Step 6: Log and commit**

Write all of the above into `docs/next-steps.md` with the date, then:
```bash
git add docs/next-steps.md
git commit -m "Docs: style-invariant embedding Phase 1 baseline"
```

---

## Self-review

- Spec coverage: bidirectional domain check → the existing `hm_domain_check.py` covers catalogue → street (baseline in Task 7 Step 4); consumer ↔ shop separability and retrieval → Tasks 4–5; same-garment invariance → `pair_cos` vs `nonpair_cos` and Recall@k in Task 5; phone-photo human set → Task 6 (capture and runs; the A/B rating is Phase 4); type-mix metric → Task 1.
- Placeholders: none; every code step has its code and every run step its command and expected outcome.
- Names checked across tasks: `GarmentRef`, `load_refs`, `match_pairs`, `polygon_cutout`, `recall_at_k`, `pair_cosine`, `domain_classifier_acc`, `ClipEncoder`, `ENCODERS`, `crop_paths`, `evaluate`, `set_demo_rep`, `count_types`, `share_ratios`, `worst_lean`, `photo_manifest`, `summary_row`.
