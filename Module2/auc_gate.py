"""
auc_gate.py  (Module 4, Phase 0 step 0.4)
=========================================
Does the trained harmony GNN still work on Pinterest outfits?

Runs Module 2's own real-vs-corrupted test (03_train_gnn.evaluate with make_negative_sample: one
garment swapped for a same-broad-category garment from another test outfit) on two test sets with the
same checkpoint and seed:

  - Fashion144k test outfits (split.mat), the reference;
  - Pinterest test outfits (split.json from the Pinterest store mode).

Gate (spec D1, step 0.4): Pinterest AUC within 0.03 of Fashion144k -> keep M2 frozen. Lower ->
fine-tune M2 on Fashion144k train + Pinterest train, and adopt it only if the Pinterest test AUC rises
and the Fashion144k test AUC doesn't drop by more than 0.01.

Usage:
    python auc_gate.py \
        --checkpoint ../Fashion144k_v1/ck_attr_subset_s42/clip_outfit_gnn_best.pt \
        --fashion_store ../Fashion144k_v1/feature_store --fashion_split ../Fashion144k_v1/split.mat \
        --pinterest_store ../PinterestData/store/feature_store --pinterest_split ../PinterestData/store/split.json
"""

import argparse
import json
import os
from importlib.util import module_from_spec, spec_from_file_location

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GATE_MARGIN = 0.03


def _load(name, filename):
    spec = spec_from_file_location(name, os.path.join(HERE, filename))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


od = _load("outfit_dataset", "02_outfit_dataset.py")
tg = _load("train_gnn", "03_train_gnn.py")


def test_ids(split_path):
    """Test outfit ids from a split.mat (Fashion144k) or a {key: split} JSON (Pinterest)."""
    if split_path.endswith(".mat"):
        return od.load_split(split_path)["test"]
    with open(split_path, encoding="utf-8") as f:
        split = json.load(f)
    return np.array(sorted(int(k) for k, v in split.items() if v == "test"), dtype=int)


test_ids.__test__ = False


def store_auc(model, store_dir, split_path, device, attr_mask, seed, max_outfits=None):
    ids = test_ids(split_path)
    ds = tg.OutfitDataset(ids, output_root=None, relvotes={}, feature_store=store_dir)
    pool = ds.build_pool(ds.valid_indices)
    tg.set_seed(seed)
    auc = tg.evaluate(model, ds, pool, device, max_outfits=max_outfits, attr_mask=attr_mask)
    return float(auc), len(ds.valid_indices)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fashion_store", required=True)
    ap.add_argument("--fashion_split", required=True)
    ap.add_argument("--pinterest_store", required=True)
    ap.add_argument("--pinterest_split", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_outfits", type=int, default=None, help="cap per test set (quick checks)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    attr_mask = od.attr_subset_indices()     # the shipped checkpoint is the attr-subset arm
    model = tg.CLIPOutfitGNN(attr_dim=len(attr_mask), edge_dim=od.EDGE_ATTR_DIM).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    f_auc, f_n = store_auc(model, args.fashion_store, args.fashion_split, device, attr_mask, args.seed,
                           args.max_outfits)
    p_auc, p_n = store_auc(model, args.pinterest_store, args.pinterest_split, device, attr_mask, args.seed,
                           args.max_outfits)
    drop = f_auc - p_auc
    print(f"Fashion144k test AUC: {f_auc:.4f}  ({f_n} outfits with >= 2 garments)")
    print(f"Pinterest   test AUC: {p_auc:.4f}  ({p_n} outfits with >= 2 garments)")
    print(f"Drop: {drop:+.4f} (gate: <= {GATE_MARGIN})")
    if drop <= GATE_MARGIN:
        print("GATE PASSED: keep Module 2 frozen.")
    else:
        print("GATE FAILED: fine-tune Module 2 on Fashion144k train + Pinterest train (spec step 0.4).")
    return {"fashion144k_auc": f_auc, "pinterest_auc": p_auc, "drop": drop, "passed": drop <= GATE_MARGIN,
            "fashion144k_n": f_n, "pinterest_n": p_n}


if __name__ == "__main__":
    main()
