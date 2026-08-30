"""
01_segment_batch.py
====================
Stage 1-2 of the plan doc: runs Prateek's real_image_pipeline (SegFormer parsing +
CLIP embedding) over the Fashion144K images, in the exact index order given by the
`photos` text file, so outfit index i (as used by split.mat / relvotes.mat /
garflat_cco.mat / col_cco.mat) maps directly to outputs/<i>/.

Resumable: writes completed indices to a checkpoint file, one per line, and skips
them on restart -- important so a long local run can be stopped/resumed freely.

Usage (local, GPU):
    python 01_segment_batch.py \
        --photos_list  ../Fashion144k_v1/photos.txt \
        --images_dir   ../Fashion144k_v1/photos \
        --output_root  ../Fashion144k_v1/outputs \
        --checkpoint   ../Fashion144k_v1/segment_checkpoint.txt \
        --device cuda --start 0 --limit 2000     # pilot run: first 2000 outfits
    # once satisfied with the pilot, rerun without --limit (or a huge --limit)
    # to cover the rest -- already-completed indices are skipped automatically.

--module1_dir defaults to the sibling Module1/ folder, so no external repo clone is
needed. Pass --module1_dir explicitly only if Module1 lives elsewhere.
"""

import os
import sys
import time
import argparse
import traceback


def load_photos_list(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return [line.strip() for line in f if line.strip()]


def load_checkpoint(path):
    if not os.path.exists(path):
        return set()
    with open(path, "r") as f:
        return set(int(line.strip()) for line in f if line.strip().isdigit())


def append_checkpoint(path, idx):
    with open(path, "a") as f:
        f.write(f"{idx}\n")


DEFAULT_MODULE1_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Module1")
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module1_dir", default=DEFAULT_MODULE1_DIR,
                         help="Path to the local Module1/ folder (default: sibling ../Module1)")
    parser.add_argument("--repo_dir", default=None,
                         help="[legacy] external repo with real_image_pipeline/; overrides "
                              "--module1_dir when set")
    parser.add_argument("--photos_list", required=True,
                         help="Path to the `photos` text file (ordered filenames)")
    parser.add_argument("--images_dir", required=True,
                         help="Folder containing the actual Fashion144K .jpg files")
    parser.add_argument("--output_root", required=True,
                         help="Where to write outputs/<idx>/ (put this on Drive so it persists)")
    parser.add_argument("--checkpoint", required=True,
                         help="Checkpoint file tracking completed outfit indices")
    parser.add_argument("--fail_log", default=None,
                         help="Where to log failures (default: <checkpoint>.failed.txt)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None,
                         help="Process at most this many *new* images (pilot runs: e.g. 2000)")
    parser.add_argument("--max_consecutive_failures", type=int, default=25,
                         help="Abort the run after this many consecutive failures (guards "
                              "against a systemic error poisoning the checkpoint)")
    parser.add_argument("--device", default=None, help="'cuda' or 'cpu'; default auto-detect")
    args = parser.parse_args()

    if args.fail_log is None:
        args.fail_log = args.checkpoint + ".failed.txt"

    if args.repo_dir:  # legacy external-repo layout
        module1_path = os.path.join(args.repo_dir, "real_image_pipeline")
    else:
        module1_path = args.module1_dir
    if not os.path.isdir(module1_path):
        raise FileNotFoundError(f"Module1 code not found at: {module1_path}")
    sys.path.insert(0, module1_path)
    print(f"[INFO] Importing Module 1 pipeline from: {module1_path}")
    from fashion_segmenter import process_image_segformer
    from clip_extract_embeddings import extract_folder_embeddings

    photos = load_photos_list(args.photos_list)
    print(f"[INFO] Loaded {len(photos)} filenames from photos list")

    done = load_checkpoint(args.checkpoint)
    print(f"[INFO] {len(done)} outfits already completed per checkpoint")

    os.makedirs(args.output_root, exist_ok=True)

    processed_this_run = 0
    consecutive_failures = 0
    t_start = time.time()

    for idx in range(args.start, len(photos)):
        if args.limit is not None and processed_this_run >= args.limit:
            print(f"[STOP] Reached --limit of {args.limit} new images this run")
            break

        if idx in done:
            continue

        filename = photos[idx]
        image_path = os.path.join(args.images_dir, filename)

        if not os.path.exists(image_path):
            with open(args.fail_log, "a") as f:
                f.write(f"{idx}\tMISSING_FILE\t{filename}\n")
            append_checkpoint(args.checkpoint, idx)  # don't retry forever
            continue

        try:
            result = process_image_segformer(
                image_path=image_path,
                output_root=args.output_root,
                image_id=str(idx),
                device=args.device,
            )
            if result["num_garments"] == 0:
                with open(args.fail_log, "a") as f:
                    f.write(f"{idx}\tNO_GARMENTS\t{filename}\n")
            else:
                extract_folder_embeddings(
                    image_folder=result["output_dir"],
                    force_recompute=False,
                    device=args.device,
                )
        except Exception as e:
            with open(args.fail_log, "a") as f:
                f.write(f"{idx}\tEXCEPTION\t{filename}\t{repr(e)}\n")
                f.write(traceback.format_exc() + "\n")
            append_checkpoint(args.checkpoint, idx)  # mark seen so we don't loop forever
            consecutive_failures += 1
            # Circuit breaker: a systemic bug (e.g. a bad dependency/dtype error) would
            # otherwise fail every image and silently mark the whole dataset "done".
            if consecutive_failures >= args.max_consecutive_failures:
                print(f"[ABORT] {consecutive_failures} consecutive failures -- likely a "
                      f"systemic error, not bad images. Stopping so the checkpoint isn't "
                      f"poisoned. See {args.fail_log}.")
                break
            continue

        append_checkpoint(args.checkpoint, idx)
        processed_this_run += 1
        consecutive_failures = 0

        if processed_this_run % 50 == 0:
            elapsed = time.time() - t_start
            rate = processed_this_run / elapsed
            print(f"[PROGRESS] {processed_this_run} done this run "
                  f"({rate:.2f} img/s, {elapsed/60:.1f} min elapsed)")

    print(f"[DONE] Processed {processed_this_run} new outfits this run. "
          f"Total completed overall: {len(load_checkpoint(args.checkpoint))}")


if __name__ == "__main__":
    main()
