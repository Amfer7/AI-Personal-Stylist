"""
run_all.py
==========
Local runner for the Module 2 pipeline (replaces fashion144k_sample1.ipynb).

No Colab, no Google Drive, no external repo clone -- it imports Module 1 from the
sibling ../Module1 folder and runs everything on the local GPU.

Stages:
  segment : 01_segment_batch.py  -- SegFormer+CLIP over Fashion144K -> outputs/<idx>/
  train   : 03_train_gnn.py      -- build graphs (02) + train the Outfit GNN, report AUC
  all     : segment then train

Expected local dataset layout (Fashion144k_v1), override with --data_root:
    <data_root>/
      photos.txt           # ordered filename list
      photos/              # the .jpg images
      split.mat            # train/val/test ids
      feat/relvotes.mat    # fashionability scores (1-10)

Examples:
    # Pilot: segment the first 2000 outfits, then train on whatever is segmented
    python run_all.py --stage all --limit 2000

    # Full segmentation run (resumable; safe to stop/restart)
    python run_all.py --stage segment

    # Train only (after a full segmentation run)
    python run_all.py --stage train --epochs 5
"""

import os
import sys
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
CAPSTONE = os.path.abspath(os.path.join(HERE, ".."))

SEG_SCRIPT = os.path.join(HERE, "01_segment_batch.py")
TRAIN_SCRIPT = os.path.join(HERE, "03_train_gnn.py")


def run(cmd):
    """Runs a subprocess, streaming output; aborts run_all on failure."""
    printable = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
    print("\n" + "=" * 70 + f"\n[run_all] $ {printable}\n" + "=" * 70)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[run_all] stage failed (exit {result.returncode}); stopping.")
        sys.exit(result.returncode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["segment", "train", "all"], default="all")
    p.add_argument("--data_root", default=os.path.join(CAPSTONE, "Fashion144k_v1"),
                   help="Fashion144K root (default: ../Fashion144k_v1)")
    p.add_argument("--output_root", default=None,
                   help="Where outputs/<idx>/ go (default: <data_root>/outputs)")
    p.add_argument("--checkpoint_dir", default=None,
                   help="Where GNN checkpoints go (default: <data_root>/gnn_checkpoints)")
    p.add_argument("--device", default="cuda", help="'cuda' or 'cpu' (default: cuda)")
    p.add_argument("--limit", type=int, default=None,
                   help="Segment at most this many new images this run (pilot runs)")
    p.add_argument("--start", type=int, default=0)
    # training passthrough
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_negatives", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--no_attributes", action="store_true",
                   help="Ablation: train without Module 1 attribute/fashion-theory features")
    p.add_argument("--attr_subset", action="store_true",
                   help="Ablation: keep only reliable attr dims (silhouette+scalars), drop pattern/fabric/fit")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible training")
    p.add_argument("--feature_store", default=None,
                   help="Path to a prebuilt feature store dir (Tier 1). Speeds up training.")
    args = p.parse_args()

    photos_list = os.path.join(args.data_root, "photos.txt")
    images_dir = os.path.join(args.data_root, "photos")
    split_mat = os.path.join(args.data_root, "split.mat")
    relvotes_mat = os.path.join(args.data_root, "feat", "relvotes.mat")
    output_root = args.output_root or os.path.join(args.data_root, "outputs")
    checkpoint = os.path.join(args.data_root, "segment_checkpoint.txt")
    checkpoint_dir = args.checkpoint_dir or os.path.join(args.data_root, "gnn_checkpoints")

    # Fail early with a clear message if the dataset isn't in place yet.
    missing = [pth for pth in (photos_list, images_dir, split_mat, relvotes_mat)
               if not os.path.exists(pth)]
    if missing and args.stage in ("segment", "all", "train"):
        print("[run_all] Missing expected dataset files/folders:")
        for m in missing:
            print("   -", m)
        print(f"[run_all] Put Fashion144K under: {args.data_root} "
              "(or pass --data_root), then re-run.")
        sys.exit(1)

    py = sys.executable

    if args.stage in ("segment", "all"):
        cmd = [py, SEG_SCRIPT,
               "--photos_list", photos_list,
               "--images_dir", images_dir,
               "--output_root", output_root,
               "--checkpoint", checkpoint,
               "--device", args.device,
               "--start", str(args.start)]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        run(cmd)

    if args.stage in ("train", "all"):
        cmd = [py, TRAIN_SCRIPT,
               "--output_root", output_root,
               "--split_mat", split_mat,
               "--relvotes_mat", relvotes_mat,
               "--checkpoint_dir", checkpoint_dir,
               "--device", args.device,
               "--epochs", str(args.epochs),
               "--batch_size", str(args.batch_size),
               "--num_negatives", str(args.num_negatives),
               "--lr", str(args.lr),
               "--seed", str(args.seed)]
        if args.no_attributes:
            cmd += ["--no_attributes"]
        if args.attr_subset:
            cmd += ["--attr_subset"]
        if args.feature_store:
            cmd += ["--feature_store", args.feature_store]
        run(cmd)

    print("\n[run_all] done.")


if __name__ == "__main__":
    main()
