"""
build_feature_store.py  (Tier 1)
================================
One-time precompute of a memmap feature store from outputs/<idx>/.

For every garment it writes a row [512 CLIP | 32 colour hist | NODE_ATTR_DIM attrs] into a
single memmap, plus a small scalar row [hue, formality, area] used for edge features. The
colour histogram (the expensive per-step recompute in the old training loop) is computed
here ONCE. After this runs, training never opens a PNG again.

Layout written to <store_dir>:
    features.npy       memmap float32 (num_garments, 512+32+ATTR)
    node_scalars.npy   memmap float32 (num_garments, 3) = [hue|nan, formality, area]
    index.json         { row dims, and outfits: {idx: [{row,file,category,broad_category}...]} }

Usage:
    python build_feature_store.py --output_root ../Fashion144k_v1/outputs \
        --store_dir ../Fashion144k_v1/feature_store
"""

import os
import json
import argparse
import numpy as np
from importlib.util import spec_from_file_location, module_from_spec

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = spec_from_file_location("outfit_dataset", os.path.join(HERE, "02_outfit_dataset.py"))
od = module_from_spec(_spec)
_spec.loader.exec_module(od)

EMB_DIM, COLOUR_DIM, ATTR_DIM = 512, 32, od.NODE_ATTR_DIM
ROW_DIM = EMB_DIM + COLOUR_DIM + ATTR_DIM


def scan_outfits(output_root):
    return sorted(
        int(d) for d in os.listdir(output_root)
        if d.isdigit() and os.path.exists(os.path.join(output_root, str(d), "metadata.json"))
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--store_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.store_dir, exist_ok=True)
    idxs = scan_outfits(args.output_root)
    print(f"[store] scanning {len(idxs)} segmented outfits under {args.output_root}")

    # Pass 1: assign a row to every garment that has a usable .npy embedding, build the index.
    outfits_index = {}
    total = 0
    for idx in idxs:
        d = os.path.join(args.output_root, str(idx))
        garments = json.load(open(os.path.join(d, "metadata.json")))
        entries = []
        for g in garments:
            npy = os.path.splitext(os.path.join(d, g["file"]))[0] + ".npy"
            if not os.path.exists(npy):
                continue
            entries.append({"row": total, "file": g["file"],
                            "category": g["category"], "broad_category": g["broad_category"]})
            total += 1
        outfits_index[str(idx)] = entries

    print(f"[store] {total} garment rows; allocating memmaps ({total}x{ROW_DIM} floats "
          f"= {total * ROW_DIM * 4 / 1e6:.0f} MB)")
    feat = np.lib.format.open_memmap(os.path.join(args.store_dir, "features.npy"),
                                     mode="w+", dtype=np.float32, shape=(total, ROW_DIM))
    scal = np.lib.format.open_memmap(os.path.join(args.store_dir, "node_scalars.npy"),
                                     mode="w+", dtype=np.float32, shape=(total, 3))

    # Pass 2: fill rows (this is where the colour histogram is computed, once).
    done = 0
    for idx in idxs:
        d = os.path.join(args.output_root, str(idx))
        garments = {g["file"]: g for g in json.load(open(os.path.join(d, "metadata.json")))}
        for ent in outfits_index[str(idx)]:
            node = od.build_node_from_meta(d, garments[ent["file"]])
            r = ent["row"]
            feat[r, :EMB_DIM] = node["embedding"]
            feat[r, EMB_DIM:EMB_DIM + COLOUR_DIM] = node["colour"]
            feat[r, EMB_DIM + COLOUR_DIM:] = node["attr_feat"]
            scal[r, 0] = node["hue"] if node["hue"] is not None else np.nan
            scal[r, 1] = node["formality"]
            scal[r, 2] = node["area"]
            done += 1
        if done and done % 2000 == 0:
            print(f"[store] filled {done}/{total} rows")

    feat.flush()
    scal.flush()

    index = {
        "row_dim": ROW_DIM, "emb_dim": EMB_DIM, "colour_dim": COLOUR_DIM, "attr_dim": ATTR_DIM,
        "num_rows": total, "outfits": outfits_index,
    }
    with open(os.path.join(args.store_dir, "index.json"), "w") as f:
        json.dump(index, f)
    print(f"[store] done. wrote features.npy, node_scalars.npy, index.json to {args.store_dir}")


if __name__ == "__main__":
    main()
