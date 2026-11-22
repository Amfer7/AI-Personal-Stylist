"""
run_pipeline.py
===============
End-to-End Real-Image Garment Understanding Pipeline:
Input Photo -> YOLO-World / YOLOS Detection -> Transparent Isolation (U2Net) -> CLIP Embedding (512-D)

Usage:
  # Single image (Default: YOLO-World):
  python real_image_pipeline/run_pipeline.py --image data/input_images/010931.jpg

  # Batch of real photos:
  python real_image_pipeline/run_pipeline.py --input_dir data/input_images/ --batch_limit 5

  # Custom output root, threshold & model:
  python real_image_pipeline/run_pipeline.py --image my_outfit.jpg --output_dir outputs --threshold 0.25 --model yolo-world
"""

import os
import sys
import argparse
import time
from typing import Optional, List
import numpy as np

# Ensure local package path is reachable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from yolo_detect_and_crop import process_image as process_image_yolo, DEFAULT_MODEL_NAME
from fashion_segmenter import process_image_segformer
from subject_check import check_subjects, decide
from clip_extract_embeddings import extract_folder_embeddings


def run_single_image(
    image_path: str,
    image_id: Optional[str] = None,
    output_root: str = "outputs",
    conf_threshold: float = 0.20,
    model_name: str = "segformer",
    specific: bool = True,
    remove_bg: bool = True,
    device: Optional[str] = None,
    subject_check: bool = True
) -> dict:
    """
    Runs the complete pipeline for a single real image.

    Steps:
      1. Fashion parsing with SegFormer-B2-Clothes + CLIP zero-shot subcategorization.
      2. Pixel-accurate transparent RGBA garment PNGs.
      3. Save metadata.json and visual overlay <image_id>_vis.jpg.
      4. Compute 512-D normalized CLIP vector embeddings (.npy).

    subject_check (SegFormer path): find people and animals first (subject_check.py). An animal with
    no person stops here with status "animal_no_person" and no garments; a pet next to a person is
    masked out. The result's "status" is "ok", "person_with_animal" or "animal_no_person".
    """
    start_time = time.time()
    if image_id is None:
        image_id = os.path.splitext(os.path.basename(image_path))[0]

    print("\n" + "=" * 65)
    print(f"[PIPELINE] AI PERSONAL STYLIST PIPELINE: {image_id}")
    print("=" * 65)
    print(f"  Input Image       : {image_path}")
    print(f"  Detector Model    : {model_name}")
    print(f"  Confidence Thresh : {conf_threshold}")
    print(f"  Background Removal: {'Enabled (Transparent RGBA)' if remove_bg else 'Disabled (Raw BBox)'}")

    # Dispatch to SegFormer (default) or YOLO
    status = "ok"
    if model_name.lower() in ["segformer", "clothes", "default"]:
        exclude_mask = None
        if subject_check:
            subjects = check_subjects(image_path, device=device)
            status = decide(subjects.persons, subjects.animals)
            print(f"  Subjects          : {subjects.persons} person(s), animals {subjects.animals or 'none'} -> {status}")
            if status == "animal_no_person":
                # generic wording: COCO names a cat in a suit "bear" and a greyhound "sheep"
                return {
                    "image_id": image_id, "status": status,
                    "message": "No person found, only an animal. "
                               "The stylist analyses outfits worn by people.",
                    "output_dir": os.path.join(output_root, image_id), "garments": [], "num_garments": 0,
                    "num_embeddings": 0, "elapsed_seconds": round(time.time() - start_time, 2)
                }
            exclude_mask = subjects.exclude_mask
        result = process_image_segformer(
            image_path=image_path,
            output_root=output_root,
            image_id=image_id,
            device=device,
            exclude_mask=exclude_mask
        )
    else:
        result = process_image_yolo(
            image_path=image_path,
            image_id=image_id,
            output_root=output_root,
            conf_threshold=conf_threshold,
            model_name=model_name,
            specific=specific,
            remove_bg=remove_bg,
            device=device
        )

    out_folder = result["output_dir"]

    # Stage 4: CLIP Embeddings
    print(f"\n[EMBED] Extracting CLIP embeddings in '{out_folder}'...")
    embs = extract_folder_embeddings(
        image_folder=out_folder,
        force_recompute=True,
        device=device
    )

    elapsed = time.time() - start_time
    print(f"\n[COMPLETED] '{image_id}' processed in {elapsed:.2f}s.")
    print(f"  - Extracted Garments: {result['num_garments']}")
    print(f"  - Generated Embeddings: {len(embs)}")
    print(f"  - Output Folder: {out_folder}")
    print("=" * 65 + "\n")

    return {
        "image_id": image_id,
        "status": status,
        "message": "",
        "output_dir": out_folder,
        "garments": result["garments"],
        "num_garments": result["num_garments"],
        "num_embeddings": len(embs),
        "elapsed_seconds": round(elapsed, 2)
    }


def run_batch_images(
    input_dir: str,
    output_root: str = "outputs",
    conf_threshold: float = 0.20,
    model_name: str = DEFAULT_MODEL_NAME,
    specific: bool = True,
    remove_bg: bool = True,
    batch_limit: Optional[int] = None,
    device: Optional[str] = None
):
    """Processes all images in a directory."""
    if not os.path.isdir(input_dir):
        print(f"[ERROR] Input directory '{input_dir}' not found.")
        return

    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    all_files = sorted([
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ])

    if batch_limit:
        all_files = all_files[:batch_limit]

    print(f"[BATCH] Found {len(all_files)} images to process in '{input_dir}'...")

    results = []
    for idx, img_path in enumerate(all_files, start=1):
        print(f"\n[{idx}/{len(all_files)}] Starting {os.path.basename(img_path)}...")
        res = run_single_image(
            image_path=img_path,
            output_root=output_root,
            conf_threshold=conf_threshold,
            model_name=model_name,
            specific=specific,
            remove_bg=remove_bg,
            device=device
        )
        results.append(res)

    print("\n" + "#" * 65)
    print(f"[DONE] BATCH PIPELINE FINISHED: {len(results)} images processed.")
    print("#" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="AI-based Personal Stylist: Real-Image Garment Detection -> Crop -> Embeddings"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image file")
    group.add_argument("--input_dir", type=str, help="Path to directory containing input photos")

    parser.add_argument("--image_id", type=str, default=None, help="Optional image ID override")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory root (default: outputs)")
    parser.add_argument("--threshold", type=float, default=0.20, help="Detection confidence threshold (default: 0.20)")
    parser.add_argument("--model", type=str, default="segformer", help="Detector model: 'segformer' (default, SOTA), 'yolo-world', or 'yolos'")
    parser.add_argument("--broad", action="store_true", help="Use broad categories instead of specific garment names")
    parser.add_argument("--no_bg_removal", action="store_true", help="Disable transparent background removal")
    parser.add_argument("--batch_limit", type=int, default=None, help="Max images to process in batch mode")
    parser.add_argument("--device", type=str, default=None, help="Execution device: 'cpu' or 'cuda'")

    args = parser.parse_args()

    model_arg = "valentinafeve/yolos-fashionpedia" if args.model.lower() == "yolos" else args.model

    if args.image:
        run_single_image(
            image_path=args.image,
            image_id=args.image_id,
            output_root=args.output_dir,
            conf_threshold=args.threshold,
            model_name=model_arg,
            specific=not args.broad,
            remove_bg=not args.no_bg_removal,
            device=args.device
        )
    elif args.input_dir:
        run_batch_images(
            input_dir=args.input_dir,
            output_root=args.output_dir,
            conf_threshold=args.threshold,
            model_name=model_arg,
            specific=not args.broad,
            remove_bg=not args.no_bg_removal,
            batch_limit=args.batch_limit,
            device=args.device
        )


if __name__ == "__main__":
    main()
