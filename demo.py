"""
demo.py  —  one-command live demo for the two completed modules
===============================================================
Runs a single photo end-to-end and writes two slide figures into demos/:

    <photo>  ->  Module 1 perception (segmentation + attributes)     -> demos/perception_demo.png
             ->  Module 2 GNN Harmoniousness score (end-to-end)      -> demos/harmoniousness_demo.png

Usage (from the repo root):
    python demo.py                       # uses image2.jpg
    python demo.py path/to/outfit.jpg    # any full-body outfit photo
    python demo.py --open                # also pop the figures open (Windows)
    python demo.py --aggregate           # also (re)build the 91.2% real-vs-corrupted testing figure

Notes:
    - First run builds a small reference cache (demos/_ref_scores.json); later runs are instant.
    - Assumes the trained checkpoint + feature store are present under Fashion144k_v1/
      (they are on the training machine). Everything else is derived automatically.
"""

import os
import sys
import time
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

DATA = os.path.join(HERE, "Fashion144k_v1")
DEMO_OUT = os.path.join(DATA, "demo_out")
DEMOS = os.path.join(HERE, "demos")
CKPT = os.path.join(DATA, "ck_attr_subset_s42", "clip_outfit_gnn_best.pt")
REF_CACHE = os.path.join(DEMOS, "_ref_scores.json")


def stage(msg):
    print("\n" + "=" * 70 + f"\n>> {msg}\n" + "=" * 70)


def run(cmd):
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"[demo] step failed ({r.returncode}): {' '.join(map(str, cmd))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo", nargs="?", default=os.path.join(HERE, "image2.jpg"),
                    help="Outfit photo to demo (default: image2.jpg)")
    ap.add_argument("--device", default=None, help="'cuda' or 'cpu' (default: auto)")
    ap.add_argument("--open", action="store_true", help="Open the figures when done (Windows)")
    ap.add_argument("--aggregate", action="store_true",
                    help="Also rebuild the aggregate real-vs-corrupted testing figure")
    args = ap.parse_args()

    if not os.path.exists(args.photo):
        sys.exit(f"[demo] photo not found: {args.photo}")
    if not os.path.exists(CKPT):
        sys.exit(f"[demo] trained checkpoint missing: {CKPT}")
    os.makedirs(DEMOS, exist_ok=True)

    image_id = "demo_live"
    outfit_dir = os.path.join(DEMO_OUT, image_id)
    perception_png = os.path.join(DEMOS, "perception_demo.png")
    harmony_png = os.path.join(DEMOS, "harmoniousness_demo.png")
    t0 = time.time()

    stage(f"1/3  Module 1 — perception on {os.path.basename(args.photo)}")
    cmd = [PY, os.path.join(HERE, "Module1", "run_pipeline.py"),
           "--image", args.photo, "--image_id", image_id, "--output_dir", DEMO_OUT]
    if args.device:
        cmd += ["--device", args.device]
    run(cmd)

    stage("2/3  render perception figure")
    run([PY, os.path.join(HERE, "Module1", "render_perception_demo.py"),
         "--outfit_dir", outfit_dir, "--photo", args.photo, "--out", perception_png])

    stage("3/3  Module 2 — GNN Harmoniousness score")
    run([PY, os.path.join(HERE, "Module2", "demo_score.py"),
         "--score_dir", outfit_dir, "--photo", args.photo,
         "--ref_cache", REF_CACHE, "--out", harmony_png])

    if args.aggregate:
        stage("extra  aggregate real-vs-corrupted testing figure")
        run([PY, os.path.join(HERE, "Module2", "demo_score.py"),
             "--out", os.path.join(DEMOS, "compatibility_demo.png")])

    outputs = [perception_png, harmony_png]
    if args.aggregate:
        outputs.append(os.path.join(DEMOS, "compatibility_demo.png"))

    stage(f"done in {time.time() - t0:.1f}s")
    for p in outputs:
        print("   ->", p)
    if args.open:
        for p in outputs:
            try:
                os.startfile(p)  # Windows
            except AttributeError:
                subprocess.run(["xdg-open", p])


if __name__ == "__main__":
    main()
