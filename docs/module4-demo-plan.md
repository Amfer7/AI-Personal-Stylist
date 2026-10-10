# Module 4 Demo Integration (D10) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Module 4 (swap recommendation) to the Gradio `demo.py`, with a live trend-weight slider,
a "Try this swap" outfit board, one harmony reference shared by Modules 2 and 4, and the two slide
figures.

**Architecture:** The Gradio wiring stays in `demo.py`; everything it displays is built by plain,
unit-tested Python in `Module4/` (`demo_panel.py` turns a `Recommendation` into widget values,
`outfit_board.py` draws crops side by side, `render_recommendation.py` makes the slide figures). The
GNN runs once per photo (`Recommender.score`); the slider and the rule dropdown only call
`Recommender.decide`, which is pure NumPy.

**Tech Stack:** Python 3.11, Gradio 6.26, PyTorch, NumPy, Pillow 11.1, Matplotlib 3.10, pytest.

## Global Constraints

Copied from the spec (`docs/module4-design-detailed.md`, D10 and §7):

- "All" runs M1–M4, reusing the already-loaded GNN + trend model and loading the pool cache once.
- Interactive panel: verdict headline; suggestion-card gallery (D6); "also worth improving"; per-piece
  table; trend-weight slider that re-ranks from cached ΔH/ΔT **without re-running the GNN**; advanced
  options (pool demo/eval, same-type, no-trend-drop, selection rule).
- "Try this swap": outfit board (remaining crops + new piece, collage code shared with the human study)
  with its harmony/trend.
- One harmony reference everywhere: M2's panel switches to the full-test-set reference (D3) so M2 and
  M4 agree.
- M3 panel text no longer calls the trend weak link "the swap target for Module 4".
- Slides: `demos/recommendation_demo.png` (one real example per verdict, from the report set) +
  `demos/recommendation_tradeoff.png`.
- Hardware: RTX 3070 Ti (gate ≤ 5 s, already met: 0.16 s); a laptop RTX 3060 is also supported
  (timing reported, not gated). VRAM cap is a flag (`--vram-gb`, default 6).
- Tests (§7): headless `analyze(image2.jpg, All)` returns all four module outputs; moving the trend
  slider re-ranks without a GNN call (asserted via a call counter); M2 and M4 report the same harmony %
  for the unswapped outfit.
- Trend weight mixes as gain = (1 − w)·ΔH + w·ΔT, the same mix the evaluation and its trade-off curve
  use (`eval_recovery.py` refuses settings where w_h ≠ 1 − w_t).
- Chart colours (validated with the dataviz palette validator, both PASS on light): Fashion144k
  `#2a78d6`, Pinterest `#eb6834`; text `#0b0b0b` / `#52514e`; grid `#e6e5e0`; white background.

## Before you start

- Work from the repo root `C:\dev\Capstone2`. Tests: `python -m pytest Module4/<file> -v`.
  `Module4/conftest.py` puts `Module4/` on `sys.path` and provides the `settings`, `pool`, `gnn`,
  `recommender` and `image2_dir` fixtures. Tests marked `real` need the data, a GPU and the built cache.
- The working tree holds uncommitted Phase 0 / step 5 work (`Module4/m4_settings.json`,
  `Module4/eval_recovery.py`, `Module4/conftest.py`, several tests, docs). **Commit only the files each
  task names** (`git add <paths>`), never `git add -A`.
- `Module4/cache/` is gitignored and already built for both stores (634,742 rows).
- The calibrated settings are rule `slot_percentile`, w_t 0.2, w_h 0.8, τ 0.3924, ε 0.2.

## File structure

| File | Responsibility |
|---|---|
| `Module4/recommender.py` (modify) | `decide()` gains an optional `w_h`, so the slider can mix (1 − w, w) |
| `Module4/outfit_board.py` (create) | Draw labelled garment crops side by side on white. Shared with the human study |
| `Module4/demo_panel.py` (create) | `Recommendation` → markdown, gallery items, table rows, swap choices, swap board. No Gradio import |
| `Module2/demo_score.py` (modify) | `PhotoScorer` accepts an injected model + reference, and an `area_scale` for the user's photo |
| `demo.py` (modify) | Module 4 radio entry, singletons, panel widgets, slider/rule re-rank, "Try this swap", `--vram-gb` |
| `Module4/render_recommendation.py` (create) | The two slide figures |
| `Module4/test_outfit_board.py`, `Module4/test_demo_panel.py`, `Module4/test_demo_app.py`, `Module4/test_render_recommendation.py` (create) | Tests |
| `Module4/test_recommender_logic.py` (modify) | Test for `decide(w_h=…)` |
| `Module4/README.md`, `Module3/README.md`, `docs/next-steps.md` (modify) | Docs |

---

### Task 1: `decide()` takes the harmony weight

**Files:**
- Modify: `Module4/recommender.py` (`decide`, around lines 117–180)
- Test: `Module4/test_recommender_logic.py`

**Interfaces:**
- Produces: `Recommender.decide(table, w_t=None, rule=None, w_h=None) -> Recommendation`. `w_h=None`
  keeps today's behaviour (`settings.w_h`); `Recommendation.weights` reports the weights actually used.

- [ ] **Step 1: Write the failing test** (append to `Module4/test_recommender_logic.py`)

```python
def test_decide_takes_the_harmony_weight():
    rec, st, _ = make()
    table = rec.score(st)
    # shirt_blouse candidates: ΔH = 3, 2.9, 2, 1, 0.5; ΔT = 1, -1, 2, 0 (unknown), 0.5 (worn trend 50, σ_trend 10)
    r = rec.decide(table, w_t=0.5, w_h=0.5)
    assert r.weights == {"w_h": 0.5, "w_t": 0.5}
    assert r.per_piece["shirt_blouse_0.png"]["best_gain"] == pytest.approx(2.0)
    r_default = rec.decide(table, w_t=0.5)                 # w_h falls back to settings (1.0 here)
    assert r_default.weights == {"w_h": 1.0, "w_t": 0.5}
    assert r_default.per_piece["shirt_blouse_0.png"]["best_gain"] == pytest.approx(3.5)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest Module4/test_recommender_logic.py::test_decide_takes_the_harmony_weight -v`
Expected: FAIL with `TypeError: Recommender.decide() got an unexpected keyword argument 'w_h'`

- [ ] **Step 3: Implement**

In `Module4/recommender.py`, change the head of `decide`:

```python
    def decide(self, table, w_t=None, rule=None, w_h=None) -> Recommendation:
        s = self.settings
        w_t = s.w_t if w_t is None else float(w_t)
        w_h = s.w_h if w_h is None else float(w_h)
        rule = rule or s.rule
        tau, eps = float(s.tau[rule]), float(s.eps)
        pieces = table.state.pieces
        G = [combine(ps.d_h, ps.d_t, w_h, w_t) for ps in table.pieces]
```

and in the `Recommendation(...)` call at the end of `decide`, change `weights={"w_h": s.w_h, "w_t": w_t}`
to `weights={"w_h": w_h, "w_t": w_t}`.

- [ ] **Step 4: Run the module's tests**

Run: `python -m pytest Module4/test_recommender_logic.py -v`
Expected: all PASS (the existing tests still use the settings' w_h).

- [ ] **Step 5: Commit**

```bash
git add Module4/recommender.py Module4/test_recommender_logic.py
git commit -m "Module 4: decide() takes an optional harmony weight"
```

---

### Task 2: Outfit board (shared collage code)

**Files:**
- Create: `Module4/outfit_board.py`
- Test: `Module4/test_outfit_board.py`

**Interfaces:**
- Produces: `outfit_board(tiles: list[tuple[str | None, str, bool]], out_path: str, title: str | None = None) -> str`.
  Each tile is `(crop_path or None, caption, highlight)`. Writes a PNG and returns `out_path`.
  Constants `TILE = 200`, `PAD = 12`, `CAPTION_H = 28`, `BORDER = 4`, `HIGHLIGHT = (235, 104, 52)`,
  `PLACEHOLDER = (230, 229, 224)`.

- [ ] **Step 1: Write the failing test**

```python
"""Outfit board: layout, highlight and the missing-crop placeholder."""
from PIL import Image

from outfit_board import BORDER, CAPTION_H, HIGHLIGHT, PAD, PLACEHOLDER, TILE, outfit_board


def _crop(path, size, colour):
    Image.new("RGBA", size, colour + (255,)).save(path)
    return str(path)


def test_board_layout_highlight_and_placeholder(tmp_path):
    red = _crop(tmp_path / "a.png", (100, 300), (200, 30, 30))      # tall crop: scaled to fit
    blue = _crop(tmp_path / "b.png", (120, 80), (30, 30, 200))
    out = outfit_board([(red, "pants", False), (blue, "NEW bag", True), (None, "shoe", False)],
                       str(tmp_path / "board.png"))
    im = Image.open(out).convert("RGB")
    assert im.size == (PAD + 3 * (TILE + PAD), PAD + TILE + CAPTION_H + PAD)
    x0, y = PAD, PAD
    assert im.getpixel((x0 + TILE // 2, y + TILE // 2)) == (200, 30, 30)        # crop pasted, centred
    x1 = PAD + (TILE + PAD)
    assert im.getpixel((x1 + 1, y + 1)) == HIGHLIGHT                           # highlighted tile
    assert im.getpixel((x1 + BORDER + 2, y + BORDER + 2)) == (255, 255, 255)   # inside the border: white
    x2 = PAD + 2 * (TILE + PAD)
    assert im.getpixel((x2 + 5, y + 5)) == PLACEHOLDER                         # missing crop


def test_title_adds_a_band(tmp_path):
    out = outfit_board([(None, "hat", False)], str(tmp_path / "t.png"), title="Swap your hat")
    assert Image.open(out).size[1] == 36 + PAD + TILE + CAPTION_H + PAD
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest Module4/test_outfit_board.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'outfit_board'`

- [ ] **Step 3: Implement `Module4/outfit_board.py`**

```python
"""Outfit board: garment crops side by side on white, one labelled tile per piece.

Used by the demo's "Try this swap", the slide figure and (later) the human study's collages, so all
three show outfits the same way. Crops are Module 1's RGBA cut-outs; each is scaled to fit its tile
without distortion. A missing crop draws a grey placeholder instead of failing.
"""
from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

TILE, PAD, CAPTION_H, BORDER = 200, 12, 28, 4
TITLE_H = 36
HIGHLIGHT = (235, 104, 52)        # #eb6834: the piece being swapped in
PLACEHOLDER = (230, 229, 224)
TEXT = (11, 11, 11)
MAX_CAPTION = 26                  # characters that fit under a 200-px tile


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def _fit(path):
    if not path or not os.path.exists(path):
        return None
    with Image.open(path) as im:
        im = im.convert("RGBA")
        im.thumbnail((TILE - 2 * BORDER, TILE - 2 * BORDER))
        return im.copy()


def outfit_board(tiles, out_path: str, title: str | None = None) -> str:
    """tiles: [(crop_path or None, caption, highlight)]. Writes a PNG and returns out_path."""
    n = max(1, len(tiles))
    top = TITLE_H if title else 0
    board = Image.new("RGB", (PAD + n * (TILE + PAD), top + PAD + TILE + CAPTION_H + PAD), "white")
    draw = ImageDraw.Draw(board)
    font = _font(15)
    if title:
        draw.text((PAD, PAD // 2 + 4), title, fill=TEXT, font=_font(18))
    for i, (path, caption, highlight) in enumerate(tiles):
        x, y = PAD + i * (TILE + PAD), top + PAD
        im = _fit(path)
        if im is None:
            draw.rectangle([x, y, x + TILE - 1, y + TILE - 1], fill=PLACEHOLDER)
            draw.text((x + 10, y + TILE // 2 - 8), "no image", fill=TEXT, font=font)
        else:
            board.paste(im, (x + (TILE - im.width) // 2, y + (TILE - im.height) // 2), im)
        if highlight:
            draw.rectangle([x, y, x + TILE - 1, y + TILE - 1], outline=HIGHLIGHT, width=BORDER)
        draw.text((x, y + TILE + 6), caption[:MAX_CAPTION], fill=TEXT, font=font)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    board.save(out_path)
    return out_path
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest Module4/test_outfit_board.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add Module4/outfit_board.py Module4/test_outfit_board.py
git commit -m "Module 4: outfit board, the shared collage for demo, slides and human study"
```

---

### Task 3: Demo panel (Recommendation → widget values)

**Files:**
- Create: `Module4/demo_panel.py`
- Test: `Module4/test_demo_panel.py`

**Interfaces:**
- Consumes: `outfit_board(tiles, out_path, title=None)` (Task 2); `Recommendation`, `Suggestion`
  (`Module4/recommendation.py`); `pool.crop_path(row)`; `settings.abs(rel)`.
- Produces (all pure functions):
  - `verdict_markdown(r) -> str` (starts with `"## "`)
  - `gallery_items(r, settings) -> list[tuple[str, str]]` (absolute crop path, caption)
  - `also_markdown(r) -> str` (`""` when nothing else clears τ)
  - `PER_PIECE_HEADERS: list[str]` and `per_piece_rows(r) -> list[list]`
  - `choices(r) -> list[Suggestion]` and `choice_labels(r) -> list[str]` (same order: the top
    suggestions, then one per "also worth improving" piece)
  - `piece_crop(state, piece, pool, settings) -> str | None`
  - `swap_board(r, state, pool, settings, index, out_path) -> tuple[str, str]` (board PNG, text)
  - `outfit_tiles(state, pool, settings) -> list[tuple]` (the outfit as it is, for the "keep" board)

- [ ] **Step 1: Write the failing test**

```python
"""Demo panel: what each widget shows, built from real Recommendations of the stub recommender."""
import os

from PIL import Image

from demo_panel import (PER_PIECE_HEADERS, also_markdown, choice_labels, choices, gallery_items,
                        outfit_tiles, per_piece_rows, swap_board, verdict_markdown)
from test_recommender_logic import make


def _swap():
    rec, st, _ = make()
    table = rec.score(st)
    return rec, table, rec.decide(table)


def test_swap_panel():
    rec, table, r = _swap()
    md = verdict_markdown(r)
    assert md.startswith("## Swap your shirt_blouse")
    assert "Outfit harmony" in md and "rule `topk_mean_gain`" in md
    items = gallery_items(r, rec.settings)
    assert len(items) == 3
    assert items[0][0] == rec.settings.abs(r.suggestions[0].crop_path)
    assert items[0][1].startswith("shirt_blouse → t_shirt")
    assert "skirt_0.png" in also_markdown(r)
    rows = per_piece_rows(r)
    assert len(rows) == 4 and all(len(x) == len(PER_PIECE_HEADERS) for x in rows)
    assert rows[0][0] == "shirt_blouse_0.png" and rows[0][6] == "✓"
    labels = choice_labels(r)
    assert len(labels) == len(choices(r)) == 4
    assert labels[0].startswith("1. shirt_blouse → t_shirt") and labels[3].startswith("Also: skirt_0.png")


def test_keep_panel_has_no_choices():
    rec, st, _ = make(trend_worn=100.0, tau={"max_gain": 99.0, "topk_mean_gain": 99.0, "slot_percentile": 99.0})
    r = rec.decide(rec.score(st))
    assert r.verdict == "keep"
    assert verdict_markdown(r).startswith("## Already strong, no swap recommended")
    assert choice_labels(r) == [] and gallery_items(r, rec.settings) == [] and also_markdown(r) == ""


def test_swap_board_replaces_the_right_piece(tmp_path):
    rec, table, r = _swap()
    out, text = swap_board(r, table.state, rec.pool, rec.settings, 0, str(tmp_path / "b.png"))
    assert os.path.exists(out)
    assert Image.open(out).size[0] == 12 + 4 * (200 + 12)        # 4 pieces, one replaced
    assert text.startswith("shirt_blouse → t_shirt") and "Harmony" in text
    tiles = outfit_tiles(table.state, rec.pool, rec.settings)
    assert [t[1] for t in tiles] == ["shirt_blouse", "skirt", "leggings", "shoe"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest Module4/test_demo_panel.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'demo_panel'`

- [ ] **Step 3: Implement `Module4/demo_panel.py`**

```python
"""Module 4 panel for demo.py: turns a Recommendation into what the Gradio widgets show.

Pure Python (no Gradio import), so it is unit-tested with the stub recommender. demo.py only wires
these values into widgets.
"""
from __future__ import annotations

import os

from outfit_board import outfit_board

PER_PIECE_HEADERS = ["piece", "type", "candidates", "selection score", "best ΔH (σ)", "best ΔT (σ)",
                     "clears τ", "notes"]


def _f(x, fmt="{:+.2f}"):
    return "n/a" if x is None else fmt.format(x)


def verdict_markdown(r) -> str:
    b = r.baseline
    if r.verdict == "swap":
        head = f"## Swap your {r.swap_piece['category']}"
    elif r.verdict == "refresh":
        head = "## Already strong: optional trendier options that keep the look"
    else:
        head = "## Already strong, no swap recommended"
    lines = [head, f"Outfit harmony **{b['harmony_pct']:.0f}/100**"
             + (f" · mean piece trend **{b['trend_mean']:.0f}/100**" if b["trend_mean"] is not None else ""),
             f"Trend weight {r.weights['w_t']:.2f} · rule `{r.selection_rule}` · pool `{r.pool_mode}`"]
    if r.flags:
        lines.append("Notes: " + "; ".join(r.flags))
    if r.provisional:
        lines.append("Provisional (not yet calibrated): " + ", ".join(r.provisional))
    return "\n\n".join(lines)


def _caption(s) -> str:
    trend = ("" if s.trend_before is None or s.trend_after is None
             else f" · trend {s.trend_before:.0f}→{s.trend_after:.0f}")
    return (f"{s.from_type} → {s.to_type} · harmony {s.harmony_pct_before:.0f}→{s.harmony_pct_after:.0f}"
            f"{trend} · {s.source}\n{s.what_changed}")


def gallery_items(r, settings) -> list:
    return [(settings.abs(s.crop_path), _caption(s)) for s in r.suggestions]


def also_markdown(r) -> str:
    if not r.also_improve:
        return ""
    parts = [f"{a['piece']} → {a['suggestion'].to_type} (harmony {a['suggestion'].harmony_pct_before:.0f}"
             f"→{a['suggestion'].harmony_pct_after:.0f})" for a in r.also_improve]
    return "**Also worth improving:** " + "; ".join(parts)


def per_piece_rows(r) -> list:
    return [[f, p["type"], p["n_candidates"], _f(p["selection_score"], "{:+.3f}"), _f(p["best_dH"]),
             _f(p["best_dT"]), "✓" if p["clears_tau"] else "", "; ".join(p["flags"])]
            for f, p in r.per_piece.items()]


def choices(r) -> list:
    return list(r.suggestions) + [a["suggestion"] for a in r.also_improve]


def choice_labels(r) -> list:
    top = [f"{i}. {s.from_type} → {s.to_type} ({s.source})" for i, s in enumerate(r.suggestions, 1)]
    also = [f"Also: {a['piece']}: {a['suggestion'].from_type} → {a['suggestion'].to_type}"
            for a in r.also_improve]
    return top + also


def piece_crop(state, piece, pool, settings):
    if state.outfit_dir:
        return os.path.join(state.outfit_dir, piece.file)
    if piece.pool_row is not None:
        return settings.abs(pool.crop_path(piece.pool_row))
    return None


def outfit_tiles(state, pool, settings) -> list:
    return [(piece_crop(state, p, pool, settings), p.type, False) for p in state.pieces]


def swap_board(r, state, pool, settings, index: int, out_path: str):
    """The outfit with choices(r)[index] swapped in: (board PNG path, one-line summary)."""
    s = choices(r)[index]
    tiles = [(settings.abs(s.crop_path), f"NEW {s.to_type}", True) if p.file == s.piece_file
             else (piece_crop(state, p, pool, settings), p.type, False) for p in state.pieces]
    trend = ("" if s.trend_before is None or s.trend_after is None
             else f" · piece trend {s.trend_before:.0f} → {s.trend_after:.0f}")
    text = (f"{s.from_type} → {s.to_type} · Harmony {s.harmony_pct_before:.0f} → "
            f"{s.harmony_pct_after:.0f}/100{trend} · from {s.source}")
    return outfit_board(tiles, out_path), text
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest Module4/test_demo_panel.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add Module4/demo_panel.py Module4/test_demo_panel.py
git commit -m "Module 4: demo panel view functions"
```

---

### Task 4: Module 2 shares the GNN and the harmony reference

**Files:**
- Modify: `Module2/demo_score.py` (`PhotoScorer.__init__`, `PhotoScorer.score_photo`, lines 89–143)

**Interfaces:**
- Produces: `PhotoScorer(..., model=None, reference=None)`: when `model` is given it is used instead
  of loading `checkpoint`; when `reference` (array of raw logits) is given it replaces the 500-outfit
  `ref_cache`. `PhotoScorer.score_photo(score_dir, photo=None, out=..., area_scale=1.0)` multiplies each
  piece's area by `area_scale` before scoring (Module 4's 98,304-px frame), so its logit equals
  Module 4's for the same photo.
- Tested in Task 5 (`test_m2_and_m4_agree_on_harmony`), which needs the demo wiring.

- [ ] **Step 1: Change `PhotoScorer.__init__`**

Replace the signature and the model/reference loading (keep the dataset loading lines unchanged):

```python
    def __init__(self,
                 checkpoint=os.path.join(HERE, "../Fashion144k_v1/ck_attr_subset_s42/clip_outfit_gnn_best.pt"),
                 split_mat=os.path.join(HERE, "../Fashion144k_v1/split.mat"),
                 relvotes_mat=os.path.join(HERE, "../Fashion144k_v1/feat/relvotes.mat"),
                 output_root=os.path.join(HERE, "../Fashion144k_v1/outputs"),
                 feature_store=os.path.join(HERE, "../Fashion144k_v1/feature_store"),
                 ref_cache=None, num_eval=500, seed=42, device=None, model=None, reference=None):
        """model: an already-loaded CLIPOutfitGNN to reuse (demo.py shares Module 4's). reference: raw
        logits of real test outfits to rank against (demo.py passes Module 4's full-test-set
        HarmonyReference, so M2 and M4 report the same harmony %); None keeps the 500-outfit cache."""
        if model is not None:
            device = next(model.parameters()).device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        tg.set_seed(seed)

        split = od.load_split(split_mat)
        relvotes = od.load_relvotes(relvotes_mat)
        self.ds = od.OutfitDataset(split["test"], output_root, relvotes, feature_store=feature_store)
        self.pool = self.ds.build_pool(self.ds.valid_indices)
        self.attr_mask = od.attr_subset_indices()

        if model is not None:
            self.model = model
        else:
            self.model = tg.CLIPOutfitGNN(attr_dim=len(self.attr_mask), edge_dim=od.EDGE_ATTR_DIM).to(self.device)
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
        self.model.eval()

        if reference is not None:
            self.ref = np.sort(np.asarray(reference, dtype=np.float64))
            print(f"[scorer] using the shared harmony reference ({len(self.ref)} test outfits)")
        elif ref_cache and os.path.exists(ref_cache):
```

(the `elif ref_cache …` / `else:` branches that follow stay exactly as they are).

- [ ] **Step 2: Change `score_photo`**

```python
    def score_photo(self, score_dir, photo=None, out=os.path.join(HERE, "harmoniousness_demo.png"),
                    area_scale=1.0):
        """Score one segmented outfit folder end-to-end and render the figure.
        Returns (score, corr) as 0-100 percentiles (corr is None if no swap was possible).
        area_scale rescales the photo's garment areas (Module 4 passes 98,304 / photo pixels)."""
        garments = json.load(open(os.path.join(score_dir, "metadata.json")))
        nodes = [od.build_node_from_meta(score_dir, g) for g in garments]
        for n in nodes:
            n["area"] *= area_scale
        raw = raw_score(self.model, self.ds, nodes, self.attr_mask, self.device)
```

(the rest of `score_photo` is unchanged).

- [ ] **Step 3: Check the CLI path still runs**

Run: `python -c "import importlib.util as u; s=u.spec_from_file_location('m','Module2/demo_score.py'); m=u.module_from_spec(s); s.loader.exec_module(m); import inspect; print(inspect.signature(m.PhotoScorer.__init__)); print(inspect.signature(m.PhotoScorer.score_photo))"`
Expected: both signatures print, with `model=None, reference=None` and `area_scale=1.0`.

- [ ] **Step 4: Commit**

```bash
git add Module2/demo_score.py
git commit -m "Module 2: PhotoScorer can reuse a loaded GNN, a shared reference and area scaling"
```

---

### Task 5: Module 4 in `demo.py`

**Files:**
- Modify: `demo.py` (whole file; full new version below)
- Test: `Module4/test_demo_app.py`

**Interfaces:**
- Consumes: Task 1 `decide(…, w_h=…)`; Task 3 panel functions; Task 4 `PhotoScorer(model=, reference=)`
  and `score_photo(area_scale=)`; `Recommender.from_settings(settings, device)`, `Recommender.score`,
  `load_outfit_dir`, `photo_pixels`, `REFERENCE_FRAME_PX` (`Module4/outfit_state.py`),
  `HarmonyReference.load`, `pool_dir` (`Module4/candidate_pool.py`).
- Produces (module-level in `demo.py`): `analyze(image_path, module, device, pool_mode, same_type,
  no_trend_drop, rule, w_t) -> tuple` in the order of `OUTPUT_NAMES`; `rerank(st, w_t, rule) -> tuple`
  in the order of `M4_OUTPUT_NAMES`; `try_swap(st, label) -> (board_png, text)`;
  `run_module2(outfit_dir, image_path, out_png) -> (score, corr)`;
  `run_module4(outfit_dir, pool_mode, same_type, no_trend_drop, rule, w_t) -> dict` with keys
  `table`, `rec`; globals `_recommender`, `SETTINGS`, `DEFAULT_OPTIONS`, `MODULE_4`, `ALL`.
  The M4 state passed between events is that dict.

- [ ] **Step 1: Write the failing test `Module4/test_demo_app.py`**

```python
"""D10 demo tests (spec §7): headless All run, slider without GNN calls, M2 == M4 harmony %."""
import importlib.util
import os

import pytest

from m4_settings import REPO

pytestmark = pytest.mark.real
PHOTO = os.path.join(REPO, "samples", "image2.jpg")


@pytest.fixture(scope="module")
def demo(recommender):
    spec = importlib.util.spec_from_file_location("demo_app", os.path.join(REPO, "demo.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m._recommender = recommender          # share the session's pool + GNN: one copy on the GPU
    return m


@pytest.fixture(scope="module")
def all_run(demo):
    return dict(zip(demo.OUTPUT_NAMES, demo.analyze(PHOTO, demo.ALL, "auto", *demo.DEFAULT_OPTIONS)))


def test_all_returns_every_module(all_run):
    for k in ("perception", "harmony", "trend"):
        assert all_run[k] and os.path.exists(all_run[k]), k
    assert "[error]" not in all_run["log"]
    assert all_run["m4_verdict"].startswith("## ")
    assert all_run["m4_state"]["rec"].verdict in {"swap", "refresh", "keep"}


def test_slider_reranks_without_a_gnn_call(demo, all_run, recommender, monkeypatch):
    calls = []
    monkeypatch.setattr(recommender.scorer, "outfit_logit", lambda *a, **k: calls.append("outfit"))
    monkeypatch.setattr(recommender.scorer, "swap_logits", lambda *a, **k: calls.append("swap"))
    out = dict(zip(demo.M4_OUTPUT_NAMES, demo.rerank(all_run["m4_state"], 0.8, "max_gain")))
    assert calls == []
    r = out["m4_state"]["rec"]
    assert r.weights["w_t"] == pytest.approx(0.8) and r.weights["w_h"] == pytest.approx(0.2)
    assert r.selection_rule == "max_gain"
    assert out["m4_state"]["table"] is all_run["m4_state"]["table"]


def test_m2_and_m4_agree_on_harmony(demo, all_run, tmp_path):
    outfit_dir = os.path.join(demo.DEMO_OUT, demo.IMAGE_ID)
    score, _ = demo.run_module2(outfit_dir, PHOTO, str(tmp_path / "h.png"))
    assert score == pytest.approx(all_run["m4_state"]["rec"].baseline["harmony_pct"], abs=0.01)


def test_try_swap_draws_a_board(demo, all_run):
    from demo_panel import choice_labels
    labels = choice_labels(all_run["m4_state"]["rec"])
    if not labels:
        pytest.skip("image2 got a 'keep' verdict: nothing to try")
    png, text = demo.try_swap(all_run["m4_state"], labels[0])
    assert os.path.exists(png) and "Harmony" in text
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest Module4/test_demo_app.py -v`
Expected: FAIL with `AttributeError: module 'demo_app' has no attribute 'OUTPUT_NAMES'`

- [ ] **Step 3: Replace `demo.py` with this version**

```python
"""
demo.py  —  Gradio GUI for all four modules (in-process, models stay warm)
==========================================================================
Upload or paste an outfit photo, choose which module to run, and see the results in the browser:

    Module 1  (perception)     -> segmentation + attributes figure
    Module 2  (harmoniousness) -> end-to-end GNN Harmoniousness score figure
    Module 3  (trend)          -> per-garment Pinterest trendiness figure
    Module 4  (recommend)      -> which piece to swap and what for, with a live trend-weight slider
    All                        -> everything

The heavy models (SegFormer + CLIP, the GNN, the trend model, Module 4's candidate pool) load ONCE
and are reused for every request, so the *first* run is slow and every run after that is fast.
Modules 2 and 4 share one GNN and one harmony reference (all real test outfits of both catalogues),
so they report the same harmony % for your outfit.

Run (from the repo root):
    pip install gradio      # one-time
    python demo.py          # opens the GUI in your browser
    python demo.py --vram-gb 4    # lower GPU-memory cap (default 6, from Module4/m4_settings.json)

Notes:
    - Modules 2-4 need Module 1's segmentation output, so every run starts with Module 1.
    - Module 4 needs its cache: python Module4/build_cache.py
    - Assumes the checkpoint, feature stores and trend model named in Module4/m4_settings.json.
"""

import argparse
import os
import sys
import time
import importlib.util

import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
MOD1 = os.path.join(HERE, "Module1")
MOD2 = os.path.join(HERE, "Module2")
MOD3 = os.path.join(HERE, "Module3")
MOD4 = os.path.join(HERE, "Module4")
# Module1/run_pipeline.py imports its siblings by bare name; make sure they resolve.
if MOD1 not in sys.path:
    sys.path.insert(0, MOD1)
# Module3 and Module4 are appended (not prepended) so Module1's originals win for the vendored
# names they share; the trend pickle needs `trend_model` importable by bare name.
for _p in (MOD3, MOD4):
    if _p not in sys.path:
        sys.path.append(_p)

from m4_settings import M4Settings  # noqa: E402  (light: reads JSON only)

SETTINGS = M4Settings.load()
DATA = os.path.join(HERE, "Fashion144k_v1")
DEMO_OUT = os.path.join(DATA, "demo_out")
DEMOS = os.path.join(HERE, "demos")
CKPT = SETTINGS.abs(SETTINGS.checkpoint)
TREND_MODEL = SETTINGS.abs(SETTINGS.trend_model)      # the same model Module 4 ranks with
BOARD_PNG = os.path.join(DEMOS, "recommendation_board.png")

IMAGE_ID = "demo_live"
EXAMPLE_PHOTO = os.path.join(HERE, "samples", "image2.jpg")

MODULE_1 = "Module 1 (perception)"
MODULE_2 = "Module 2 (harmoniousness)"
MODULE_3 = "Module 3 (trend)"
MODULE_4 = "Module 4 (recommend)"
ALL = "All"
POOLS = ["demo", "eval"]
RULES = ["max_gain", "topk_mean_gain", "slot_percentile"]
# pool, same_type, no_trend_drop, rule, w_t (the calibrated settings)
DEFAULT_OPTIONS = ("demo", False, False, SETTINGS.rule, SETTINGS.w_t)

M4_OUTPUT_NAMES = ["m4_verdict", "m4_gallery", "m4_also", "m4_table", "m4_choice", "m4_board",
                   "m4_board_text", "m4_state"]
OUTPUT_NAMES = ["perception", "harmony", "trend", "log"] + M4_OUTPUT_NAMES


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- lazy, load-once singletons (heavy models) ----
_pipeline = None      # (run_single_image, render)
_scorer = None        # demo_score.PhotoScorer
_trend = None         # (trend model, render_trend_demo.render)
_recommender = None   # Module 4 Recommender (GNN + trend model + candidate pool)


def _get_pipeline():
    """Module 1: run_single_image() + the perception-figure render(). The underlying
    SegFormer/CLIP models cache themselves in their own module globals, so importing
    once means they load once."""
    global _pipeline
    if _pipeline is None:
        rp = _load_module(os.path.join(MOD1, "run_pipeline.py"), "m1_run_pipeline")
        rd = _load_module(os.path.join(MOD1, "render_perception_demo.py"), "m1_render")
        _pipeline = (rp.run_single_image, rd.render)
    return _pipeline


def _get_recommender(device=None):
    """Module 4: GNN, trend model and the candidate-pool cache, loaded once."""
    global _recommender
    if _recommender is None:
        _get_pipeline()   # import Module1's modules first: m2_bridge puts Module3 at the front of sys.path
        from recommender import Recommender
        _recommender = Recommender.from_settings(SETTINGS, device)
    return _recommender


def _get_scorer():
    """Module 2: PhotoScorer, ranking against Module 4's harmony reference and reusing its GNN
    when Module 4 is already loaded."""
    global _scorer
    if _scorer is None:
        from candidate_pool import pool_dir
        from harmony_reference import HarmonyReference
        ref = HarmonyReference.load(os.path.join(pool_dir(SETTINGS), "reference.json"))
        ds_mod = _load_module(os.path.join(MOD2, "demo_score.py"), "m2_demo_score")
        model = _recommender.gnn.model if _recommender is not None else None
        _scorer = ds_mod.PhotoScorer(checkpoint=CKPT, model=model, reference=ref.logits_sorted)
    return _scorer


def _get_trend():
    """Module 3: the trend model Module 4 uses (shared when Module 4 is loaded) + its renderer."""
    global _trend
    if _trend is None:
        rt = _load_module(os.path.join(MOD3, "render_trend_demo.py"), "m3_render")
        if _recommender is not None:
            model = _recommender.trend_model
        else:
            from trend_model import PerCategoryTrendModel
            model = PerCategoryTrendModel.load(TREND_MODEL)
        _trend = (model, rt.render)
    return _trend


# ---- Module 2 and Module 4 steps (also called by the tests) ----
def run_module2(outfit_dir, image_path, out_png):
    """Harmoniousness on Module 4's 98,304-px area frame, so it matches Module 4's baseline."""
    from outfit_state import REFERENCE_FRAME_PX, photo_pixels
    px = photo_pixels(outfit_dir)
    return _get_scorer().score_photo(outfit_dir, image_path, out_png,
                                     area_scale=REFERENCE_FRAME_PX / px if px else 1.0)


def run_module4(outfit_dir, pool_mode, same_type, no_trend_drop, rule, w_t, device=None):
    """The only GNN stage: score every allowed replacement once, then decide."""
    from outfit_state import load_outfit_dir
    rec = _get_recommender(device)
    state = load_outfit_dir(outfit_dir, rec.gnn, rec.trend_model)
    table = rec.score(state, mode=pool_mode, same_type=bool(same_type), trend_floor=bool(no_trend_drop))
    return {"table": table, "rec": rec.decide(table, w_t=w_t, w_h=1.0 - w_t, rule=rule)}


def _m4_empty(message=""):
    return ("", None, "", None, gr.Dropdown(choices=[], value=None), None, message, None)


def _m4_outputs(st):
    from demo_panel import (also_markdown, choice_labels, gallery_items, per_piece_rows,
                            swap_board, verdict_markdown)
    rec, r = _get_recommender(), st["rec"]
    labels = choice_labels(r)
    if labels:
        board, text = swap_board(r, st["table"].state, rec.pool, rec.settings, 0, BOARD_PNG)
    else:
        board, text = None, "Nothing to try: the outfit is already strong."
    return (verdict_markdown(r), gallery_items(r, rec.settings), also_markdown(r), per_piece_rows(r),
            gr.Dropdown(choices=labels, value=labels[0] if labels else None), board, text, st)


def rerank(st, w_t, rule):
    """Slider / rule change: re-decide from the cached ΔH and ΔT. No GNN call."""
    if not st:
        return _m4_empty("Run Module 4 first.")
    rec = _get_recommender()
    return _m4_outputs({"table": st["table"],
                        "rec": rec.decide(st["table"], w_t=w_t, w_h=1.0 - w_t, rule=rule)})


def try_swap(st, label):
    """"Try this swap": the outfit board with the chosen suggestion swapped in."""
    from demo_panel import choice_labels, swap_board
    if not st or not label:
        return None, ""
    rec, r = _get_recommender(), st["rec"]
    return swap_board(r, st["table"].state, rec.pool, rec.settings, choice_labels(r).index(label), BOARD_PNG)


def analyze(image_path, module, device, pool_mode, same_type, no_trend_drop, rule, w_t):
    """Run the requested module(s) on one photo; returns values in the order of OUTPUT_NAMES."""
    if not image_path:
        return (None, None, None, "Please upload or paste an outfit photo first.") + _m4_empty()

    want_perception = module in (MODULE_1, ALL)
    want_harmony = module in (MODULE_2, ALL)
    want_trend = module in (MODULE_3, ALL)
    want_m4 = module in (MODULE_4, ALL)
    if (want_harmony or want_m4) and not os.path.exists(CKPT):
        return (None, None, None, f"Trained checkpoint missing: {CKPT}") + _m4_empty()
    if (want_trend or want_m4) and not os.path.exists(TREND_MODEL):
        return (None, None, None, f"Trend model missing: {TREND_MODEL}\n"
                                  "Fit it with: python Module3/fit_per_category.py") + _m4_empty()

    os.makedirs(DEMOS, exist_ok=True)
    outfit_dir = os.path.join(DEMO_OUT, IMAGE_ID)
    perception_png = os.path.join(DEMOS, "perception_demo.png")
    harmony_png = os.path.join(DEMOS, "harmoniousness_demo.png")
    trend_png = os.path.join(DEMOS, "trend_demo.png")
    dev = None if (not device or device == "auto") else device

    log = []
    t0 = time.time()
    try:
        run_single_image, render = _get_pipeline()
        if want_m4:
            _get_recommender(dev)          # load first, so Module 2 and 3 reuse its GNN + trend model

        # Always run Module 1: the perception figure and every scorer read its output.
        log.append(f">> Module 1 — perception on {os.path.basename(image_path)}")
        res = run_single_image(image_path=image_path, image_id=IMAGE_ID,
                               output_root=DEMO_OUT, device=dev)
        log.append(f"   {res['num_garments']} garments, "
                   f"{res['num_embeddings']} embeddings in {res['elapsed_seconds']}s")

        perception_out = None
        if want_perception:
            render(outfit_dir, image_path, perception_png)
            perception_out = perception_png

        harmony_out = None
        if want_harmony:
            log.append(">> Module 2 — GNN Harmoniousness score")
            score, corr = run_module2(outfit_dir, image_path, harmony_png)
            log.append(f"   Harmoniousness = {score:.0f}/100"
                       + (f"   (drops to {corr:.0f}/100 if one garment is swapped)"
                          if corr is not None else ""))
            harmony_out = harmony_png

        trend_out = None
        if want_trend:
            log.append(">> Module 3 — per-garment Pinterest trendiness")
            trend_model, render_trend = _get_trend()
            _, result = render_trend(outfit_dir, image_path, trend_png, model=trend_model)
            for g in result["garments"]:
                s = g["trendiness_score"]
                log.append(f"   {g['category']:10s} {g['specific_category']:14s} "
                           + ("no confident match" if s is None else f"{s:5.1f}/100"))
            weak = next((g for g in result["garments"]
                         if g["trendiness_score"] is not None
                         and g["category"] == result["weak_link_category"]
                         and g["trendiness_score"] == result["weak_link_score"]), None)
            if weak:
                log.append(f"   outfit avg {result['outfit_avg_trendiness']:.0f}/100 — least on-trend: "
                           f"{weak['specific_category']} ({weak['category']}, "
                           f"{weak['trendiness_score']:.0f}/100)")
            trend_out = trend_png

        m4 = _m4_empty()
        if want_m4:
            log.append(">> Module 4 — swap recommendation")
            t4 = time.time()
            st = run_module4(outfit_dir, pool_mode, same_type, no_trend_drop, rule, w_t, dev)
            r = st["rec"]
            n = sum(len(p.rows) for p in st["table"].pieces)
            log.append(f"   verdict: {r.verdict}" + (f" ({r.swap_piece['category']})" if r.swap_piece else "")
                       + f" · {n} what-if outfits scored in {time.time() - t4:.2f}s")
            m4 = _m4_outputs(st)

        log.append(f">> done in {time.time() - t0:.1f}s")
        return (perception_out, harmony_out, trend_out, "\n".join(log)) + tuple(m4)
    except Exception as e:
        import traceback
        log.append("\n[error] " + str(e))
        log.append(traceback.format_exc())
        return (None, None, None, "\n".join(log)) + _m4_empty()


def build_ui():
    from demo_panel import PER_PIECE_HEADERS
    with gr.Blocks(title="Outfit Compatibility Demo") as ui:
        gr.Markdown(
            "# Outfit Compatibility Demo\n"
            "Upload or paste a full-body outfit photo, choose which module to run, "
            "then click **Run**.\n\n"
            "- **Module 1 (perception)** — segments the photo into garments and reads "
            "their attributes (colour, pattern, formality, …).\n"
            "- **Module 2 (harmoniousness)** — scores the outfit 0–100 with the trained GNN "
            "(also shows the drop if one garment is swapped).\n"
            "- **Module 3 (trend)** — scores *each garment* 0–100 for current popularity on "
            "Pinterest in its category, and marks the least on-trend piece.\n"
            "- **Module 4 (recommend)** — tries every allowed replacement for every piece and says "
            "which one piece to swap and what for, or that the outfit is already strong.\n"
            "- **All** — everything.\n\n"
            "> The **first** run loads the models (~30–60s); every run after that is fast."
        )
        m4_state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(sources=["upload", "clipboard"], type="filepath",
                                 label="Outfit photo (upload or paste)")
                module = gr.Radio([MODULE_1, MODULE_2, MODULE_3, MODULE_4, ALL], value=ALL,
                                  label="Which module to run")
                device = gr.Radio(["auto", "cuda", "cpu"], value="auto", label="Device")
                w_t = gr.Slider(0.0, 1.0, value=SETTINGS.w_t, step=0.05,
                                label="Trend weight (Module 4): 0 = harmony only, 1 = trend only")
                with gr.Accordion("Module 4 advanced options", open=False):
                    pool = gr.Radio(POOLS, value="demo",
                                    label="Candidate pool (demo = all garments, eval = test split only)")
                    same_type = gr.Checkbox(False, label="Same type only (strict like-for-like)")
                    no_trend_drop = gr.Checkbox(False, label="Never suggest a less trendy piece (trend floor)")
                    rule = gr.Dropdown(RULES, value=SETTINGS.rule, label="Selection rule")
                run_btn = gr.Button("Run", variant="primary")
                if os.path.exists(EXAMPLE_PHOTO):
                    gr.Examples(examples=[[EXAMPLE_PHOTO]], inputs=[image],
                                label="Example (samples/image2.jpg)")
            with gr.Column(scale=2):
                perception_img = gr.Image(label="Module 1 — perception", type="filepath")
                harmony_img = gr.Image(label="Module 2 — harmoniousness", type="filepath")
                trend_img = gr.Image(label="Module 3 — per-garment trendiness", type="filepath")
                gr.Markdown("### Module 4 — recommendation")
                verdict = gr.Markdown()
                gallery = gr.Gallery(label="Suggestions", columns=3, height=320)
                also = gr.Markdown()
                with gr.Row():
                    choice = gr.Dropdown(choices=[], label="Try this swap")
                board = gr.Image(label="Your outfit with the swap", type="filepath")
                board_text = gr.Markdown()
                table = gr.Dataframe(headers=PER_PIECE_HEADERS, label="Per piece", interactive=False)
                log = gr.Textbox(label="Log", lines=10)

        m4_outputs = [verdict, gallery, also, table, choice, board, board_text, m4_state]
        run_btn.click(analyze,
                      inputs=[image, module, device, pool, same_type, no_trend_drop, rule, w_t],
                      outputs=[perception_img, harmony_img, trend_img, log] + m4_outputs)
        w_t.release(rerank, inputs=[m4_state, w_t, rule], outputs=m4_outputs)
        rule.change(rerank, inputs=[m4_state, w_t, rule], outputs=m4_outputs)
        choice.change(try_swap, inputs=[m4_state, choice], outputs=[board, board_text])
    return ui


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Outfit compatibility demo (Modules 1-4)")
    ap.add_argument("--vram-gb", type=float, default=None, dest="vram_gb",
                    help=f"GPU memory cap for Module 4 (default {SETTINGS.vram_gb} from m4_settings.json)")
    args = ap.parse_args()
    if args.vram_gb is not None:
        SETTINGS = SETTINGS.with_(vram_gb=args.vram_gb)
    build_ui().launch(inbrowser=True)
```

- [ ] **Step 4: Run the demo tests**

Run: `python -m pytest Module4/test_demo_app.py -v`
Expected: 4 PASS (or 3 PASS + 1 SKIP if `image2.jpg` gets a "keep" verdict). The first test takes
~1 minute (Module 1 and Module 2 models load).

- [ ] **Step 5: Run the whole Module 4 suite**

Run: `python -m pytest Module4 -q`
Expected: all PASS.

- [ ] **Step 6: Launch the app and check it by hand**

Run: `python demo.py`
Check in the browser with `samples/image2.jpg` and **All**: four sections fill in; moving the slider
to 1.0 and releasing changes the suggestions within a second and the log shows no new run; picking
another "Try this swap" entry redraws the board; the Module 2 figure's score equals Module 4's
"Outfit harmony". Stop the app with Ctrl+C.

- [ ] **Step 7: Commit**

```bash
git add demo.py Module4/test_demo_app.py
git commit -m "Demo: Module 4 panel, live trend slider, try-this-swap board, shared reference"
```

---

### Task 6: Slide figures

**Files:**
- Create: `Module4/render_recommendation.py`
- Test: `Module4/test_render_recommendation.py`

**Interfaces:**
- Consumes: `outfit_board`, `outfit_tiles`, `swap_board` (Tasks 2–3); `eval_sets(pool, n, source=)`
  (`Module4/eval_recovery.py`); `state_from_pool(pool, source, key)`; `Recommender.score/decide`; the
  report summaries `Module4/cache/eval/report_summary.json` and `pinterest_report_summary.json`
  (`"weight_sweep"`: list of `{w_t, mean_dH, mean_dT, hit_rate, recall@10}`).
- Produces: `load_sweeps(eval_dir) -> dict[str, list[dict]]`;
  `render_tradeoff(sweeps, chosen_w, out_path) -> str`;
  `pick_examples(rec, sources, limit) -> dict[str, tuple]` (verdict → (source, key, table, r));
  `render_examples(rec, found, out_path) -> str`; CLI flags `--tradeoff`, `--examples`.

- [ ] **Step 1: Write the failing test**

```python
"""Slide figures: the trade-off plot from sweep data, and the per-verdict example figure."""
import os

import pytest
from PIL import Image

from render_recommendation import load_sweeps, pick_examples, render_examples, render_tradeoff

SWEEP = [{"w_t": w, "mean_dH": 2.5 - 1.5 * w, "mean_dT": 2.6 * w, "hit_rate": 0.75 - 0.4 * w, "recall@10": 0.0}
         for w in (0.0, 0.1, 0.2, 0.5, 1.0)]


def test_tradeoff_figure(tmp_path):
    out = render_tradeoff({"fashion144k": SWEEP, "pinterest": SWEEP}, 0.2, str(tmp_path / "t.png"))
    w, h = Image.open(out).size
    assert w > 1500 and h > 600


@pytest.mark.real
def test_load_sweeps_reads_both_report_summaries(settings):
    sweeps = load_sweeps(settings.abs(os.path.join(settings.cache_dir, "eval")))
    assert set(sweeps) == {"fashion144k", "pinterest"}
    assert [r["w_t"] for r in sweeps["pinterest"]][:3] == pytest.approx([0.0, 0.1, 0.2])


@pytest.mark.real
def test_examples_figure(recommender, tmp_path):
    found = pick_examples(recommender, ["pinterest"], limit=40)
    assert found and set(found) <= {"swap", "refresh", "keep"}
    out = render_examples(recommender, found, str(tmp_path / "e.png"))
    assert os.path.exists(out)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest Module4/test_render_recommendation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'render_recommendation'`

- [ ] **Step 3: Implement `Module4/render_recommendation.py`**

```python
"""Slide figures for Module 4 (D10).

    python Module4/render_recommendation.py --tradeoff   # demos/recommendation_tradeoff.png
    python Module4/render_recommendation.py --examples   # demos/recommendation_demo.png

tradeoff: the report sets' weight sweep, per catalogue. Left: harmony gain vs trend gain of the top
suggestion as the trend weight goes 0 -> 1 (the calibrated weight circled). Right: weak-link hit rate
by trend weight. Both gains are in σ units, so they share one scale.
examples: one real report-set outfit per verdict (swap / refresh / keep), scored on the demo pool
(its own outfit excluded), each drawn as an outfit board.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from demo_panel import outfit_tiles, swap_board  # noqa: E402
from m4_settings import REPO, M4Settings  # noqa: E402
from outfit_board import outfit_board  # noqa: E402

SERIES = {"fashion144k": ("#2a78d6", "Fashion144k"), "pinterest": ("#eb6834", "Pinterest")}
SUMMARIES = {"fashion144k": "report_summary.json", "pinterest": "pinterest_report_summary.json"}
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
VERDICTS = ("swap", "refresh", "keep")
HEADLINE = {"swap": "Swap: {piece} → {to}", "refresh": "Refresh: already strong, a trendier option",
            "keep": "Keep: already strong"}


def load_sweeps(eval_dir: str) -> dict:
    out = {}
    for src, name in SUMMARIES.items():
        path = os.path.join(eval_dir, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                out[src] = json.load(f)["weight_sweep"]
    return out


def _style(ax):
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT2)
    ax.tick_params(colors=TEXT2)


def render_tradeoff(sweeps: dict, chosen_w: float, out_path: str) -> str:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4))
    for src, rows in sweeps.items():
        colour, label = SERIES[src]
        w = np.array([r["w_t"] for r in rows])
        dh = np.array([r["mean_dH"] for r in rows])
        dt = np.array([r["mean_dT"] for r in rows])
        hr = np.array([r["hit_rate"] for r in rows])
        k = int(np.argmin(np.abs(w - chosen_w)))
        a.plot(dt, dh, color=colour, linewidth=2, marker="o", markersize=4, label=label)
        a.plot(dt[k], dh[k], marker="o", markersize=12, markerfacecolor="none",
               markeredgecolor=colour, markeredgewidth=2)
        a.annotate(f"w = {w[k]:.1f}", (dt[k], dh[k]), xytext=(10, -14), textcoords="offset points",
                   color=TEXT2, fontsize=9)
        b.plot(w, hr, color=colour, linewidth=2, marker="o", markersize=4, label=label)
    b.axvline(chosen_w, color=TEXT2, linewidth=1, linestyle="--")
    b.annotate(f"chosen w = {chosen_w:.1f}", (chosen_w, b.get_ylim()[1]), xytext=(6, -14),
               textcoords="offset points", color=TEXT2, fontsize=9)
    a.set_xlabel("Trend gain of the top suggestion (σ)", color=TEXT)
    a.set_ylabel("Harmony gain of the top suggestion (σ)", color=TEXT)
    a.set_title("Harmony vs trend as the trend weight rises", color=TEXT, loc="left")
    b.set_xlabel("Trend weight w", color=TEXT)
    b.set_ylabel("Finds the corrupted piece", color=TEXT)
    b.set_title("Weak-link hit rate by trend weight", color=TEXT, loc="left")
    for ax in (a, b):
        _style(ax)
        ax.legend(frameon=False, labelcolor=TEXT)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    return out_path


def pick_examples(rec, sources, limit: int = 400) -> dict:
    """First report-set outfit per verdict, scanning `sources` in order."""
    from eval_recovery import eval_sets
    from outfit_state import state_from_pool
    found = {}
    for src in sources:
        _, report = eval_sets(rec.pool, 2000, source=src)
        for key in report[:limit]:
            state = state_from_pool(rec.pool, src, key)
            table = rec.score(state, mode="demo", exclude=(src, key))
            r = rec.decide(table)
            if r.verdict not in found and (r.verdict == "keep" or r.suggestions):
                found[r.verdict] = (src, key, table, r)
            if len(found) == len(VERDICTS):
                return found
    return found


def render_examples(rec, found: dict, out_path: str) -> str:
    verdicts = [v for v in VERDICTS if v in found]
    fig, axes = plt.subplots(len(verdicts), 1, figsize=(10, 3.2 * len(verdicts)), squeeze=False)
    with tempfile.TemporaryDirectory() as tmp:
        for ax, v in zip(axes[:, 0], verdicts):
            src, key, table, r = found[v]
            png = os.path.join(tmp, f"{v}.png")
            if v == "keep":
                outfit_board(outfit_tiles(table.state, rec.pool, rec.settings), png)
                head = HEADLINE[v]
            else:
                swap_board(r, table.state, rec.pool, rec.settings, 0, png)
                s = r.suggestions[0]
                head = HEADLINE[v].format(piece=s.from_type, to=s.to_type)
            s0 = r.suggestions[0] if r.suggestions else None
            nums = (f"harmony {s0.harmony_pct_before:.0f} → {s0.harmony_pct_after:.0f}" if s0
                    else f"harmony {r.baseline['harmony_pct']:.0f}")
            ax.imshow(Image.open(png))
            ax.set_axis_off()
            ax.set_title(f"{head}   ({nums}; {SERIES[src][1]} outfit {key})", color=TEXT, loc="left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tradeoff", action="store_true")
    ap.add_argument("--examples", action="store_true")
    ap.add_argument("--device")
    args = ap.parse_args(argv)
    s = M4Settings.load()
    demos = os.path.join(REPO, "demos")
    if args.tradeoff:
        sweeps = load_sweeps(s.abs(os.path.join(s.cache_dir, "eval")))
        print(render_tradeoff(sweeps, s.w_t, os.path.join(demos, "recommendation_tradeoff.png")))
    if args.examples:
        from recommender import Recommender
        rec = Recommender.from_settings(s, args.device)
        found = pick_examples(rec, [st.name for st in s.stores])
        print(f"verdicts found: {sorted(found)}")
        print(render_examples(rec, found, os.path.join(demos, "recommendation_demo.png")))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest Module4/test_render_recommendation.py -v`
Expected: 3 PASS

- [ ] **Step 5: Make the two figures and look at them**

Run: `python Module4/render_recommendation.py --tradeoff --examples`
Expected: prints both paths and `verdicts found: ['keep', 'refresh', 'swap']` (a missing verdict means
none occurred in the first 400 report outfits of either source: rerun after raising `limit`).
Open `demos/recommendation_tradeoff.png` and `demos/recommendation_demo.png` and check: no label
collisions, the circled point sits at w = 0.2 on both curves, every board shows crops (no "no image"
tiles), titles readable.

- [ ] **Step 6: Commit**

```bash
git add Module4/render_recommendation.py Module4/test_render_recommendation.py
git commit -m "Module 4: slide figures (trade-off curve, one example per verdict)"
```

(`demos/` is the existing figure folder; commit the two PNGs only if the other demo figures in
`demos/` are tracked: check with `git ls-files demos`.)

---

### Task 7: Documentation

**Files:**
- Modify: `Module4/README.md`, `Module3/README.md`, `docs/next-steps.md`

- [ ] **Step 1: `Module4/README.md` status line** — replace the two lines under the title with:

```markdown
**Status: built, evaluated on both catalogues (Fashion144k + Pinterest), all settings calibrated, and
in the demo** (`python demo.py`, pick **Module 4** or **All**).
```

- [ ] **Step 2: `Module4/README.md` "How to run"** — after the `run_recommend.py` block, add:

```bash
# Demo (all four modules, live trend-weight slider, "Try this swap" board)
python demo.py                 # --vram-gb 4 for a smaller GPU
# Slide figures -> demos/recommendation_tradeoff.png, demos/recommendation_demo.png
python Module4/render_recommendation.py --tradeoff --examples
```

and change the report-set line's comment to
`# ~5 min; refuses a second run without --force (add --source pinterest for Pinterest outfits)`.

- [ ] **Step 3: `Module4/README.md` "Evaluation results"** — replace the section (from
`## Evaluation results` up to `## Settings (all calibrated)`) with:

```markdown
## Evaluation results

**Protocol** (spec §6). Per catalogue, two disjoint, seeded sets of 2,000 test outfits: **dev** for all
tuning (Fashion144k only), **report** for the final numbers, run with frozen settings. In every outfit
one random piece is **corrupted**: swapped for a random garment from the same swap group that passes the
co-occurrence filter, which is exactly the space M4 searches. M4 then looks for replacements in the
test split of **both** catalogues (the **eval pool**), never the outfit itself or its reposts.

**History.** First calibrated and reported on 2026-10-09 with a Fashion144k-only pool (71.8%). When
Pinterest joined (2026-10-10) the trend model was refit and the harmony reference changed, so the
settings were recalibrated on the same dev set and both report sets run; the Fashion144k report set
was re-run with `--force` because the system changed. The 9 Oct results are kept in
`Module4/cache/eval_2026-10-09_f144k_only/`.

**Report sets** (rule `slot_percentile`, w_t = 0.2, τ = 0.392; 95% bootstrap CIs):

| Method | Fashion144k: finds the corrupted piece | Pinterest: finds the corrupted piece | Original's median rank (F / P) |
|---|---|---|---|
| **M4** | **72.2%** (70.2–74.1) | **69.3%** (67.3–71.3) | **top 4.8% / 5.4%** |
| CLIP coherence (no GNN) | 42.2% | 42.7% | top 31.5% / 31.9% |
| Random | 30.4% | 37.8% | top 50.6% / 49.9% |
| Trend only | 32.6% | 34.5% | top 45.5% / 52.2% |
| Chance (1 / pieces) | 31.4% | 36.0% | top 50% |

All baselines are below M4 at p < 1e-60 on both catalogues (exact McNemar test).

- **Pinterest is only ~3 points lower,** although the GNN's own AUC drops 0.09 on Pinterest (0.727 vs
  0.816; it was trained on 2010s Fashion144k). Hit rate compares pieces *within* one outfit, so an
  error that shifts a whole outfit's score cancels out. Fine-tuning the GNN is deferred (see
  `docs/next-steps.md`).
- **Harder when the corruption is subtle.** Fashion144k 55% / 74% / 88% (subtle / medium / obvious);
  Pinterest 53% / 72% / 82%.
- **Original recovery is the independent evidence** (what the person actually wore). The original lands
  in the top ~5% of candidates on both catalogues, far above every baseline, but is rarely the very top
  pick: the GNN rates its own top pick above the original in 96–97% of outfits.
- **Verdicts.** At τ, M4 says "swap" for 70% of corrupted outfits and for 16% (Fashion144k) / 17%
  (Pinterest) of real ones, an upper bound on false alarms. Before recalibration it was 75% / 21%.
- **Trend costs some accuracy.** From w_t = 0 to 0.2 the top suggestion's trend gain rises from ~0σ to
  1.55σ (Fashion144k) / 1.42σ (Pinterest) while keeping 94% / 93% of the harmony gain; hit rate drops
  75.7% → 72.2% / 72.9% → 69.3%. The demo slider lets users choose
  (`demos/recommendation_tradeoff.png`).
- **Selection rules barely differ:** Fashion144k 70.0 / 70.5 / 72.2%, Pinterest 68.9 / 68.8 / 69.3%
  (`max_gain` / `topk_mean_gain` / `slot_percentile`).
- **Limits.** Corruptions are random same-group garments, so these numbers measure catching clear
  intruders, not taste; mixed outfits (one catalogue's piece in the other's outfit) are suggested but
  not separately tested. The human study is the independent check.

Full tables are written to `Module4/cache/eval/report_report.md` and `pinterest_report_report.md`.
```

- [ ] **Step 4: `Module4/README.md` settings table** — change the τ row's value to
`0.392 for \`slot_percentile\` (0.333 before Pinterest joined)` and add after the table:
`Recalibrated 2026-10-10 on the both-catalogue pool; rule, w_t and ε came out unchanged.`

- [ ] **Step 5: `Module4/README.md` "Next"** — replace the section body with:

```markdown
1. **Human study** (spec §6.4): blind pairwise preferences on outfit boards (`outfit_board.py`).
2. **GNN fine-tune on both catalogues** (deferred; see `docs/next-steps.md`).
3. Re-check the near-duplicate threshold on the mixed pool (`calibrate_dup.py`, needs a visual check).
   See [`docs/next-steps.md`](../docs/next-steps.md).
```

and change the line in "Things worth knowing" ending `Phase 0 must store Pinterest areas in that
frame too.` to end `Pinterest photos are resized to that frame before Module 1 (PinterestData/build_resized.py).`

- [ ] **Step 6: `Module3/README.md` results** — under the cluster stability table, add:

```markdown
**Refit on the full Pinterest store (2026-10-10,** `per_category_trend_model_store.pkl`, 149,320
garments up to 2026-08, photos at the Fashion144k frame**):** accessory 0.888, bottom 0.759, dress
0.955, top **0.611** (moderate; tops are the least stable and may need their own k, like dresses).

**Forward-share backtest (D11, same refit):** mean Spearman ρ between the score and a cluster's share
1/3/6 months later, pooled over 52 blocks: growth weight 0 → −0.253, 0.3 → −0.243, **0.6 → −0.155**,
growth-only → −0.081. 0.6 beats recency-only (+0.098, CI +0.036 to +0.162), so the D11 weight stays
0.6, but every weight is negative: high-scoring clusters tend to lose share. The score describes
**current popularity**; it is not a forecast.
```

- [ ] **Step 7: `docs/next-steps.md`** — in the Module 4 section, replace the line
`- [ ] Add Module 4 to \`demo.py\` (designed: decision D10).` with:

```markdown
- [x] Module 4 in `demo.py` (D10, 2026-10-xx): verdict, suggestion gallery, also-worth-improving,
      per-piece table, live trend-weight slider (no GNN call, tested), advanced options, "Try this
      swap" board (`Module4/outfit_board.py`, shared with the human study), M2 and M4 on one GNN and
      one harmony reference (tested equal), `--vram-gb`. Slide figures:
      `demos/recommendation_tradeoff.png`, `demos/recommendation_demo.png`.
```

(write the actual date), and change the `**Next:**` line to start with `human study`.

- [ ] **Step 8: Commit**

```bash
git add Module4/README.md Module3/README.md docs/next-steps.md
git commit -m "Docs: Module 4 demo, both-catalogue results, refit trend model results"
```

---

## Self-review notes

- Spec coverage (D10 + §7): All runs M1–M4 (Task 5) · shared GNN/trend/pool (Task 5 singletons,
  Task 4 injection) · verdict, gallery, also, per-piece (Task 3, 5) · slider without GNN (Task 1, 5,
  tested) · advanced options (Task 5) · try-this-swap board shared with the human study (Task 2, 3, 5)
  · one reference, M2 == M4 tested (Task 4, 5) · M3 text (Task 5 `build_ui` and log wording) · slide
  figures (Task 6) · `--vram-gb` and timing in the log (Task 5).
- The slider mixes (1 − w, w), not (0.8, w): at the calibrated w = 0.2 both give 0.8 / 0.2, and it
  matches the trade-off figure. `run_recommend.py --w-t` keeps the old (settings w_h, w) behaviour.
- The demo's old 500-outfit M2 reference (`demos/_ref_scores.json`) is no longer read; leave the file.
- Module 2 alone does not load Module 4's pool (it loads its own GNN copy, ~1 MB of weights); in **All**
  Module 4 loads first and Module 2/3 reuse its GNN and trend model.
