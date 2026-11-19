"""
build_outfit_queries.py

Module 4 can leave whole search topics out of the demo's suggestions (settings: demo_exclude_queries,
e.g. the Indian ethnic-wear queries while they are out of scope). For that it needs the search
queries behind each store photo. A pin can come from several queries (61,118 rows for 55,781 pins),
so this writes one row per (store key, query): store/photos.csv (key, pin_id) joined with
pinterest_data/metadata_98k.csv (pin_id, query) -> store/outfit_queries.csv.

Usage (from the repo root):
    python PinterestData/build_outfit_queries.py
"""

import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def build(photos_csv: str, metadata_csv: str, out_csv: str) -> pd.DataFrame:
    photos = pd.read_csv(photos_csv, usecols=["key", "pin_id"], dtype=str)
    meta = pd.read_csv(metadata_csv, usecols=["pin_id", "query"], dtype=str)
    rows = photos.merge(meta, on="pin_id", how="left")
    missing = rows["query"].isna().sum()
    if missing:
        raise SystemExit(f"{missing} store photos have no query in {metadata_csv}")
    rows = rows[["key", "query"]].drop_duplicates().sort_values(["key", "query"], key=lambda c: (
        c.astype(int) if c.name == "key" else c))
    rows.to_csv(out_csv, index=False)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos_csv", default=os.path.join(HERE, "store", "photos.csv"))
    ap.add_argument("--metadata_csv", default=os.path.join(HERE, "pinterest_data", "metadata_98k.csv"))
    ap.add_argument("--out_csv", default=os.path.join(HERE, "store", "outfit_queries.csv"))
    args = ap.parse_args()
    rows = build(args.photos_csv, args.metadata_csv, args.out_csv)
    print(f"{len(rows)} (key, query) rows for {rows['key'].nunique()} photos, "
          f"{rows['query'].nunique()} queries -> {args.out_csv}")


if __name__ == "__main__":
    main()
