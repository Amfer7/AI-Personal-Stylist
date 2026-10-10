"""
build_resized.py

Fashion144k photos are all 256x384 (98,304 px, the frame garment areas are normalised to). The
Pinterest originals mapped by build_fullres_metadata.py are ~15x larger (median 1000x1448), which
slows Module 1, bloats the garment crops it saves, and puts resolution between the two stores.

This writes a copy of every image in pinterest_data/metadata_fullres.csv scaled to ~98,304 px with the
aspect ratio kept (a 2:3 portrait becomes exactly 256x384; nothing is upscaled), opened the way
Module 1 opens photos (PIL .convert("RGB")) and saved as JPEG quality 95 under
pinterest_data/images_98k/, mirroring the raw/ layout. Then pinterest_data/metadata_98k.csv is the
same table with image_path pointing at the copies (relative to pinterest_data/). Existing copies are
skipped, so it can be re-run.

Usage (from the repo root):
    python PinterestData/build_resized.py
Then run store mode on it:
    python Module3/pinterest_garment_segmentation.py --mode store \
        --metadata_csv PinterestData/pinterest_data/metadata_98k.csv --output_dir PinterestData/store
"""

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET_PX = 256 * 384


def resize_one(job):
    src, dst = job
    if os.path.exists(dst):
        return None
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            s = math.sqrt(TARGET_PX / (w * h))
            if s < 1:
                im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            im.save(dst + ".tmp", "JPEG", quality=95)
        os.replace(dst + ".tmp", dst)
        return None
    except Exception as e:  # reported, row dropped
        return f"{src}\t{e!r}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata_csv", default=os.path.join(HERE, "pinterest_data", "metadata_fullres.csv"))
    ap.add_argument("--out_dir", default="images_98k", help="relative to the metadata csv's folder")
    ap.add_argument("--out_csv", default=os.path.join(HERE, "pinterest_data", "metadata_98k.csv"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(args.metadata_csv))
    df = pd.read_csv(args.metadata_csv)
    new_path = {p: f"{args.out_dir}/{os.path.splitext(p)[0]}.jpg" for p in df["image_path"].unique()}
    assert len(set(new_path.values())) == len(new_path), "two source images map to one output name"

    jobs = [(os.path.join(base, src), os.path.join(base, dst)) for src, dst in new_path.items()]
    print(f"{len(jobs)} images -> {os.path.join(base, args.out_dir)} ({args.workers} workers)")
    failed = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, err in enumerate(ex.map(resize_one, jobs, chunksize=64), 1):
            if err:
                failed.append(err)
            if i % 5000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    bad = {os.path.relpath(e.split("\t")[0], base).replace("\\", "/") for e in failed}
    for e in failed:
        print("FAILED", e)
    df = df[~df["image_path"].isin(bad)].copy()
    df["image_path"] = df["image_path"].map(new_path)
    df.to_csv(args.out_csv, index=False)
    print(f"{len(df)} rows, {df['image_path'].nunique()} images written ({len(failed)} images failed)")
    print(f"-> {args.out_csv}")


if __name__ == "__main__":
    main()
