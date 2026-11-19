"""Visual calibration of the demo pool's crop quality gate (crop_quality.py, settings.crop_gate).

For each garment type, renders one contact sheet of demo-pool crops: one row per rejection reason
(the first test a crop fails), then ones that only just pass (area under 1.5 x the type's bar) and
ones that pass. A reviewer raises a type's bar if the "just passing" row still looks unreadable, or
lowers it if a "rejected" row looks fine; the change applies on the next demo start (no rebuild).
Also prints, per swap group and catalogue, how many demo candidates the gate keeps.

    python Module4/calibrate_crops.py [--types shoe bag dress] [--per_row 10] [--bands label text]
Needs crop metrics: python Module4/build_cache.py. Sheets go to <cache_dir>/crop_calibration/<type>.png.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from candidate_pool import gate_demo_pool, pool_dir  # noqa: E402
from crop_quality import crop_ok, load_metrics  # noqa: E402
from m4_settings import M4Settings  # noqa: E402
from swap_groups import SWAP_GROUPS, TYPE_ID, TYPES  # noqa: E402

CELL, LABEL_H, ROW_TITLE_H = 150, 14, 18
REASONS = ("small", "split", "sparse", "spread", "holes", "label", "text")


def reason(m: dict, r: int, t: str, gate: dict) -> str:
    """Why row r fails the gate (first failing test), for the sheet caption."""
    if m["area"][r] < gate["min_area"].get(t, gate["min_area"]["default"]):
        return f"small {m['area'][r]:.0f}"
    share = m["top2" if t in gate["pair_types"] else "main"][r]
    if share < gate["min_main"]:
        return f"split {share:.2f}"
    if m["fill"][r] < gate["min_fill"]:
        return f"sparse {m['fill'][r]:.2f}"
    if m["cover"][r] < gate["min_cover"]:
        return f"spread {m['cover'][r]:.2f}"
    if m["holes"][r] > gate["max_holes"]:
        return f"holes {m['holes'][r]:.2f}"
    if m["label_p"][r] < gate["min_label_p"].get(t, gate["min_label_p"]["default"]):
        return f"label {m['label_p'][r]:.2f}"
    return f"text {m['text_p'][r]:.2f}"


def _on_white(path: str) -> Image.Image:
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, "white")
    im = Image.alpha_composite(bg, im).convert("RGB")
    s = (CELL - 6) / max(im.size)                       # scale up or down to the cell, like the demo gallery
    return im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))))


def render_sheet(rows_by_band: dict, captions: dict, path_of, out_path: str, per_row: int) -> None:
    bands = list(rows_by_band)
    h = len(bands) * (ROW_TITLE_H + CELL + LABEL_H)
    sheet = Image.new("RGB", (per_row * CELL, h), "white")
    d = ImageDraw.Draw(sheet)
    y = 0
    for b in bands:
        d.text((4, y + 2), f"{b} ({len(rows_by_band[b])} shown)", fill="black")
        y += ROW_TITLE_H
        for i, r in enumerate(rows_by_band[b]):
            sheet.paste(_on_white(path_of(r)), (i * CELL + 3, y + 3))
            d.text((i * CELL + 3, y + CELL), captions[r], fill="black")
        y += CELL + LABEL_H
    sheet.save(out_path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--types", nargs="+", default=list(TYPES))
    ap.add_argument("--per_row", type=int, default=10)
    ap.add_argument("--bands", nargs="+", default=None,
                    help="only these rows, by first word: small split sparse spread holes label text just passing")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    s = M4Settings.load()
    d = pool_dir(s)
    meta = dict(np.load(os.path.join(d, "meta.npz")))
    with open(os.path.join(d, "files.json"), encoding="utf-8") as f:
        files = json.load(f)
    with open(os.path.join(d, "outfit_keys.json"), encoding="utf-8") as f:
        keys = json.load(f)
    m = load_metrics(s, d)
    gate = s.crop_gate
    idx = np.arange(len(files))
    before = meta["rep_demo"] == idx
    scal_area = np.zeros((len(files), 3))
    scal_area[:, 2] = m["area"]                          # stand-in for the store areas; only counts are printed
    after = gate_demo_pool(s, meta, scal_area)["rep_demo"] == idx
    print(f"[crop-cal] demo candidates kept by the gate (of {before.sum()}): {after.sum()}")
    names = [st.name for st in s.stores]
    for g, types in SWAP_GROUPS.items():
        in_g = np.isin(meta["type"], [TYPE_ID[t] for t in types])
        cells = [f"{n} {(after & in_g & (meta['source'] == i)).sum():>6}/{(before & in_g & (meta['source'] == i)).sum():<6}"
                 for i, n in enumerate(names)]
        print(f"[crop-cal]   {g:10s} " + "  ".join(cells))

    def path_of(r):
        st = s.stores[int(meta["source"][r])]
        return s.abs(os.path.join(st.output_root, keys[st.name][int(meta["outfit_ord"][r])], files[r]))

    ok = crop_ok(m, meta["type"], gate)
    out_dir = s.abs(os.path.join(s.cache_dir, "crop_calibration"))
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for t in args.types:
        rows = np.nonzero(before & (meta["type"] == TYPE_ID[t]))[0]
        bar = gate["min_area"].get(t, gate["min_area"]["default"])
        bad = rows[~ok[rows]]
        why = np.array([reason(m, r, t, gate).split()[0] for r in bad])
        bands = {f"rejected: {w} ({(why == w).sum()})": bad[why == w] for w in REASONS if (why == w).any()}
        bands.update({f"just passing (area {bar}-{1.5 * bar:.0f})": rows[ok[rows] & (m["area"][rows] < 1.5 * bar)],
                      "passing": rows[ok[rows]]})
        if args.bands:
            bands = {b: v for b, v in bands.items() if b.removeprefix("rejected: ").split()[0] in args.bands}
        bands = {b: rng.choice(v, min(args.per_row, len(v)), replace=False) for b, v in bands.items()}
        captions = {int(r): (reason(m, r, t, gate) if not ok[r] else f"{m['area'][r]:.0f} px")
                    + f" {names[int(meta['source'][r])][:3]}" for v in bands.values() for r in v}
        out = os.path.join(out_dir, f"{t}.png")
        render_sheet(bands, captions, path_of, out, args.per_row)
        print(f"[crop-cal] {t}: bar {bar} px, keeps {ok[rows].sum()}/{len(rows)} -> {out}")


if __name__ == "__main__":
    main()
