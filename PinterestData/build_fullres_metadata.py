"""
build_fullres_metadata.py

pinterest_dataset/images/ holds 224x224 copies made by a plain resize (aspect ratio not kept). The
full-resolution originals are still in the raw scrape, pinterest_data/raw/pinterest/Search/<query>/,
named pinterest_<pin_id>.<ext>, or pinterest_<pin_id>_<n>.<ext> when the pin was saved with a numeric
suffix (sometimes several files per pin). The `hash` column of metadata.csv is the MD5 of exactly one
of those raw files, so the candidate whose MD5 matches is the one used.

This writes pinterest_data/metadata_fullres.csv: metadata.csv with image_path pointing at the raw
originals (relative to pinterest_data/). Each row is matched by query folder + pin_id and verified by
MD5. A pin downloaded under several queries is the same picture saved several times with slightly
different bytes, so every row of a pin points at one canonical file (the first verified one, in
metadata order); the rows keep their own date and query for the trend model.

Usage (from the repo root):
    python PinterestData/build_fullres_metadata.py
Then run store mode on it:
    python Module3/pinterest_garment_segmentation.py --mode store \
        --metadata_csv PinterestData/pinterest_data/metadata_fullres.csv --output_dir PinterestData/store
"""

import argparse
import glob
import hashlib
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SKIP_EXT = (".json", ".txt", ".part", ".mp3")  # sidecars, captions, unfinished downloads, audio


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata_csv", default=os.path.join(HERE, "pinterest_dataset", "metadata.csv"))
    ap.add_argument("--raw_root", default=os.path.join(HERE, "pinterest_data"))
    ap.add_argument("--out_csv", default=os.path.join(HERE, "pinterest_data", "metadata_fullres.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.metadata_csv)
    search = os.path.join(args.raw_root, "raw", "pinterest", "Search")
    found, verified = [], []
    for _, r in df.iterrows():
        folder = os.path.join(search, str(r["query"]).replace(" ", "+"))
        hits = []
        for pattern in (f"pinterest_{r['pin_id']}.*", f"pinterest_{r['pin_id']}_*"):
            hits += [p for p in glob.glob(os.path.join(folder, pattern)) if not p.endswith(SKIP_EXT)]
        path = next((p for p in hits if md5(p) == r["hash"]), None)
        found.append(os.path.relpath(path, args.raw_root).replace("\\", "/") if path else None)
        verified.append(path is not None)
    df["raw_path"] = found
    print(f"{sum(verified)} of {len(df)} rows matched a raw file with the same MD5")

    canonical = df.dropna(subset=["raw_path"]).drop_duplicates("pin_id").set_index("pin_id")["raw_path"]
    df["image_path"] = df["pin_id"].map(canonical)
    missing = int(df["image_path"].isna().sum())
    df = df.dropna(subset=["image_path"]).drop(columns=["raw_path"])
    print(f"{df['pin_id'].nunique()} pins, {len(df)} rows written ({missing} rows dropped: no verified file for the pin)")
    df.to_csv(args.out_csv, index=False)
    print(f"-> {args.out_csv}")


if __name__ == "__main__":
    main()
