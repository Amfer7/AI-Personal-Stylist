"""Visual QA of the suggestion catalogue: what users actually see as suggestions (H&M cut-outs).

    python Module4/scripts/catalogue_qa.py sheets [--per_type 30] [--seed 0]   # numbered sheets + marks.csv
    python Module4/scripts/catalogue_qa.py score [--marks <marks.csv>]          # defect rate per type

A seeded random sample of demo-pool representatives (the rows that can be suggested, after the
crop gate) per garment type, drawn as numbered contact sheets on white. The reviewer opens marks.csv
and fills `mark` only for bad items (blank = fine) with one of MARKS. Outputs go to
<cache_dir>/catalogue_qa/ (gitignored): <type>.jpg, marks.csv, result.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

MARKS = {"background": "studio background or other non-garment pixels left in the cut-out",
         "wrong_type": "not the garment type of the sheet",
         "unreadable": "cut off, broken, blurry or too small to tell what it is",
         "not_adult": "children's or baby item",
         "other": "anything else (say what in `comment`)"}
COLS = 6


def sample_by_type(rows_by_type: dict, per_type: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for t, rows in rows_by_type.items():
        rows = np.asarray(rows)
        pick = rows if len(rows) <= per_type else rng.choice(rows, per_type, replace=False)
        out[t] = [int(r) for r in pick]
    return out


def defect_rates(marks: pd.DataFrame) -> dict:
    m = marks.copy()
    m["mark"] = m["mark"].fillna("").astype(str).str.strip()
    bad = set(m["mark"]) - set(MARKS) - {""}
    if bad:
        raise ValueError(f"unknown marks {sorted(bad)}; use one of {sorted(MARKS)} or leave blank")

    def stats(g):
        flagged = g[g["mark"] != ""]
        return {"items": len(g), "flagged": len(flagged), "rate": len(flagged) / len(g),
                **{k: int(v) for k, v in flagged["mark"].value_counts().items()}}
    out = {t: stats(g) for t, g in m.groupby("type")}
    out["all"] = stats(m)
    return out


def _out_dir(s) -> str:
    return s.abs(os.path.join(s.cache_dir, "catalogue_qa"))


def sheets(per_type: int, seed: int) -> str:
    from PIL import Image, ImageDraw
    from m4_settings import M4Settings
    from recommender import Recommender
    s = M4Settings.load()
    pool = Recommender.from_settings(s).pool
    reps = np.nonzero(pool.rep_demo == np.arange(pool.n))[0]
    by_type = {}
    for r in reps:
        by_type.setdefault(pool.type_name(int(r)), []).append(int(r))
    picks = sample_by_type(dict(sorted(by_type.items())), per_type, seed)
    out = _out_dir(s)
    os.makedirs(out, exist_ok=True)
    tile, rows_out = 180, []
    for t, rows in picks.items():
        n_rows = (len(rows) + COLS - 1) // COLS
        sheet = Image.new("RGB", (COLS * tile, 40 + n_rows * (tile + 22)), "white")
        d = ImageDraw.Draw(sheet)
        d.text((8, 12), f"{t}: {len(rows)} random suggestion items (of {len(by_type[t])})", fill="black")
        for i, r in enumerate(rows):
            x, y = (i % COLS) * tile, 40 + (i // COLS) * (tile + 22)
            try:
                with Image.open(s.abs(pool.crop_path(r))) as im:
                    im = im.convert("RGBA")
                    im.thumbnail((tile - 12, tile - 12))
                    sheet.paste(im, (x + (tile - im.width) // 2, y + (tile - im.height) // 2), im)
            except OSError:
                d.text((x + 10, y + tile // 2), "missing", fill="red")
            d.rectangle([x + 2, y + 2, x + tile - 3, y + tile - 3], outline=(220, 220, 220))
            d.text((x + 6, y + tile + 4), str(i + 1), fill="black")
            rows_out.append({"type": t, "cell": i + 1, "item": pool.outfit_key(r), "pool_row": r,
                             "mark": "", "comment": ""})
        sheet.save(os.path.join(out, f"{t}.jpg"), quality=90)
    pd.DataFrame(rows_out).to_csv(os.path.join(out, "marks.csv"), index=False)
    print(f"{len(rows_out)} items on {len(picks)} sheets in {out}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sh = sub.add_parser("sheets")
    sh.add_argument("--per_type", type=int, default=30)
    sh.add_argument("--seed", type=int, default=0)
    sc = sub.add_parser("score")
    sc.add_argument("--marks")
    args = ap.parse_args(argv)
    if args.cmd == "sheets":
        sheets(args.per_type, args.seed)
        return
    from m4_settings import M4Settings
    out = _out_dir(M4Settings.load())
    res = defect_rates(pd.read_csv(args.marks or os.path.join(out, "marks.csv"), dtype=str))
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
