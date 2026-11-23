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

--module1_dir defaults to the sibling Module1/src/ folder, so no external repo clone is
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
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Module1", "src")
)


def main():
    # Fashion144K has non-ASCII filenames (e.g. Polish 'ń'). On Windows, printing them to a
    # piped/redirected stdout defaults to cp1252 and crashes. Force UTF-8 with errors=replace.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--module1_dir", default=DEFAULT_MODULE1_DIR,
                         help="Path to Module 1's code (default: sibling ../../Module1/src)")
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
    photos = load_photos_list(args.photos_list)
    print(f"[INFO] Loaded {len(photos)} filenames from photos list")
    segment_photos(photos, args.images_dir, args.output_root, args.checkpoint, fail_log=args.fail_log,
                   start=args.start, limit=args.limit,
                   max_consecutive_failures=args.max_consecutive_failures, device=args.device,
                   module1_dir=module1_path)


def segment_photos(photos, images_dir, output_root, checkpoint, fail_log=None, start=0, limit=None,
                   max_consecutive_failures=25, device=None, module1_dir=DEFAULT_MODULE1_DIR,
                   after_each=None, save_vis=True):
    """Run Module 1 (SegFormer parsing + attributes + CLIP embeddings) on photos[i] for every index i
    not yet in the checkpoint, writing <output_root>/<i>/. Shared by Fashion144k (this script) and the
    Pinterest store mode (Module3/scripts/pinterest_garment_segmentation.py --mode store), so both stores come
    from the identical code path. after_each(idx, output_dir) runs after a photo with garments is done.
    save_vis=False skips the <i>_vis.jpg overlays (display only; no feature depends on them)."""
    if fail_log is None:
        fail_log = checkpoint + ".failed.txt"
    if not os.path.isdir(module1_dir):
        raise FileNotFoundError(f"Module1 code not found at: {module1_dir}")
    if module1_dir not in sys.path:
        sys.path.insert(0, module1_dir)
    print(f"[INFO] Importing Module 1 pipeline from: {module1_dir}")
    from fashion_segmenter import process_image_segformer
    from clip_extract_embeddings import extract_folder_embeddings

    done = load_checkpoint(checkpoint)
    print(f"[INFO] {len(done)} outfits already completed per checkpoint")

    os.makedirs(output_root, exist_ok=True)

    processed_this_run = 0
    consecutive_failures = 0
    t_start = time.time()

    for idx in range(start, len(photos)):
        if limit is not None and processed_this_run >= limit:
            print(f"[STOP] Reached --limit of {limit} new images this run")
            break

        if idx in done:
            continue

        filename = photos[idx]
        image_path = os.path.join(images_dir, filename)

        if not os.path.exists(image_path):
            with open(fail_log, "a", encoding="utf-8") as f:
                f.write(f"{idx}\tMISSING_FILE\t{filename}\n")
            append_checkpoint(checkpoint, idx)  # don't retry forever
            continue

        try:
            result = process_image_segformer(
                image_path=image_path,
                output_root=output_root,
                image_id=str(idx),
                save_vis=save_vis,
                device=device,
            )
            if result["num_garments"] == 0:
                with open(fail_log, "a", encoding="utf-8") as f:
                    f.write(f"{idx}\tNO_GARMENTS\t{filename}\n")
            else:
                extract_folder_embeddings(
                    image_folder=result["output_dir"],
                    force_recompute=False,
                    device=device,
                )
                if after_each is not None:
                    after_each(idx, result["output_dir"])
        except Exception as e:
            with open(fail_log, "a", encoding="utf-8") as f:
                f.write(f"{idx}\tEXCEPTION\t{filename}\t{repr(e)}\n")
                f.write(traceback.format_exc() + "\n")
            append_checkpoint(checkpoint, idx)  # mark seen so we don't loop forever
            consecutive_failures += 1
            # Circuit breaker: a systemic bug (e.g. a bad dependency/dtype error) would
            # otherwise fail every image and silently mark the whole dataset "done".
            if consecutive_failures >= max_consecutive_failures:
                print(f"[ABORT] {consecutive_failures} consecutive failures -- likely a "
                      f"systemic error, not bad images. Stopping so the checkpoint isn't "
                      f"poisoned. See {fail_log}.")
                break
            continue

        append_checkpoint(checkpoint, idx)
        processed_this_run += 1
        consecutive_failures = 0

        if processed_this_run % 50 == 0:
            elapsed = time.time() - t_start
            rate = processed_this_run / elapsed
            print(f"[PROGRESS] {processed_this_run} done this run "
                  f"({rate:.2f} img/s, {elapsed/60:.1f} min elapsed)")

    print(f"[DONE] Processed {processed_this_run} new outfits this run. "
          f"Total completed overall: {len(load_checkpoint(checkpoint))}")
    return processed_this_run


if __name__ == "__main__":
    main()
