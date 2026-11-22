"""Run the whole pipeline (Module 1 -> 4) on every photo in a folder and report what happened.

    python Module4/photo_batch.py --photos samples/phone_set [--out <dir>]

Used for the phone-photo test set and for picking demo photos (docs/review3-kits.md, kit 2). Writes
results.csv (one row per photo: status, garments, harmony, verdict, suggestions, seconds),
summary.json and sheet.jpg (each photo's Module 1 overlay with its verdict) to --out, by default
<cache_dir>/photo_batch/<folder name>/ (gitignored). Models load once for the whole folder.
Photo status: ok | animal_no_person (subject check stopped it) | no_garments | error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

PHOTO_EXT = (".jpg", ".jpeg", ".png", ".webp")


def _stem(f: str) -> str:
    return os.path.splitext(f)[0].rsplit("_", 1)[0]


def result_row(photo: str, status: str, garments: list, rec, seconds: float, note: str = "") -> dict:
    return {"photo": photo, "status": status, "garments": ", ".join(_stem(g) for g in garments),
            "n_garments": len(garments),
            "harmony": None if rec is None else round(rec.baseline["harmony_pct"], 1),
            "verdict": "" if rec is None else rec.verdict,
            "swap_piece": "" if rec is None or not rec.swap_piece else _stem(rec.swap_piece["file"]),
            "suggestions": "" if rec is None else "; ".join(f"{s.from_type}→{s.to_type}" for s in rec.suggestions),
            "seconds": round(seconds, 1), "note": note}


def summary(rows: pd.DataFrame) -> dict:
    v = Counter(x for x in rows["verdict"] if x)
    return {"photos": len(rows), "status": dict(Counter(rows["status"])), "verdict": dict(v.most_common())}


def _sheet(rows: list, run_dir: str, out_path: str, tile: int = 260):
    from PIL import Image, ImageDraw
    cols = 4
    sheet = Image.new("RGB", (cols * tile, ((len(rows) + cols - 1) // cols) * (tile + 34)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, r in enumerate(rows):
        stem = os.path.splitext(r["photo"])[0]
        vis = os.path.join(run_dir, stem, f"{stem}_vis.jpg")
        x, y = (i % cols) * tile, (i // cols) * (tile + 34)
        if os.path.exists(vis):
            with Image.open(vis) as im:
                im = im.convert("RGB")
                im.thumbnail((tile - 6, tile - 6))
                sheet.paste(im, (x + 3, y + 3))
        label = r["verdict"] or r["status"]
        draw.text((x + 4, y + tile + 2), f"{r['photo'][:28]}", fill="black")
        draw.text((x + 4, y + tile + 16), f"{label}  {'' if r['harmony'] is None else r['harmony']}", fill="black")
    sheet.save(out_path, quality=90)


def run(photos_dir: str, out_dir: str | None = None) -> pd.DataFrame:
    import run_recommend as rr
    from m4_settings import M4Settings
    from outfit_state import NoGarmentsDetected, load_outfit_dir
    from recommender import Recommender
    s = M4Settings.load()
    rec = Recommender.from_settings(s)
    name = os.path.basename(os.path.normpath(photos_dir))
    out_dir = out_dir or s.abs(os.path.join(s.cache_dir, "photo_batch", name))
    run_dir = os.path.join(out_dir, "runs")
    os.makedirs(run_dir, exist_ok=True)
    rows = []
    for f in sorted(os.listdir(photos_dir)):
        if not f.lower().endswith(PHOTO_EXT):
            continue
        t0, garments, r, note = time.time(), [], None, ""
        try:
            d = rr.segment_photo(os.path.join(photos_dir, f), run_dir)
            with open(os.path.join(d, "metadata.json"), encoding="utf-8") as fh:
                garments = [g["file"] for g in json.load(fh)]
            state = load_outfit_dir(d, rec.gnn, rec.trend_model)
            r = rec.decide(rec.score(state, mode="demo"))
            status = "ok"
        except SystemExit as e:           # subject check: an animal and no person
            status, note = "animal_no_person", str(e)
        except NoGarmentsDetected:
            status, note = "no_garments", "No garments found"
        except Exception as e:            # keep going: one bad photo must not stop the batch
            status, note = "error", f"{type(e).__name__}: {e}"
        rows.append(result_row(f, status, garments, r, time.time() - t0, note))
        print(f"  {f}: {rows[-1]['verdict'] or status}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "results.csv"), index=False, encoding="utf-8")
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary(df), fh, indent=1)
    _sheet(rows, run_dir, os.path.join(out_dir, "sheet.jpg"))
    print(json.dumps(summary(df), indent=1))
    print(f"results in {out_dir}")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", required=True)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    run(args.photos, args.out)


if __name__ == "__main__":
    main()
