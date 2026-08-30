"""
yolo_detect_and_crop.py
=======================
Autonomous real-photo garment detection and extraction stage.
Supports:
  1. YOLO-World (Default: `yolov8s-worldv2.pt`): Fast, high-accuracy fashion detector
     optimized for real user photos (selfies, mirror shots, street photos).
  2. YOLOS-Fashionpedia (`valentinafeve/yolos-fashionpedia`): Vision Transformer detector.

Features:
  - Real-image input (no ground-truth annotations needed).
  - Mutually distinct clothing categories with cross-category NMS deduplication.
  - Saliency-protected transparent garment isolation (U2Net) to preserve all fabric/logo details.
  - Standard repository contract outputs/<image_id>/ for downstream CLIP & GNN compatibility.
  - Annotated visual overlay image (<image_id>_vis.jpg) for explainability.
"""

import os
import json
from typing import List, Dict, Tuple, Optional, Set
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch

# Local imports
try:
    from category_mapping import map_category
    from bg_removal import isolate_garment_with_segformer, get_fashion_semantic_mask, isolate_garment_background
except ImportError:
    from real_image_pipeline.category_mapping import map_category
    from real_image_pipeline.bg_removal import isolate_garment_with_segformer, get_fashion_semantic_mask, isolate_garment_background

DEFAULT_MODEL_NAME = "yolo-world"
DEFAULT_YOLO_WORLD_CHECKPOINT = "yolov8s-worldv2.pt"

# Clean, mutually distinct fashion classes for YOLO-World
FASHION_VOCABULARY = [
    "t-shirt", "shirt", "blouse", "sweater", "jacket", "coat",
    "pants", "shorts", "skirt", "dress",
    "shoes", "bag", "sunglasses", "hat", "watch"
]

# Global cached model handles
_CACHED_DETECTOR = None
_CACHED_PROCESSOR = None
_CACHED_MODEL_NAME = None
_CACHED_DEVICE = None


def get_default_device() -> str:
    """Returns 'cuda' if GPU is available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_detection_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None
):
    """
    Loads and caches the fashion object detection model.
    """
    global _CACHED_DETECTOR, _CACHED_PROCESSOR, _CACHED_MODEL_NAME, _CACHED_DEVICE

    if device is None:
        device = get_default_device()

    if _CACHED_DETECTOR is not None and _CACHED_MODEL_NAME == model_name and _CACHED_DEVICE == device:
        return _CACHED_DETECTOR, _CACHED_PROCESSOR

    is_yolo_world = ("world" in model_name.lower() or model_name in ["yolo-world", "yoloworld", "yolo_world"])

    if is_yolo_world:
        ckpt = DEFAULT_YOLO_WORLD_CHECKPOINT if model_name in ["yolo-world", "yoloworld", "yolo_world"] else model_name
        print(f"[INFO] Loading YOLO-World detector ({ckpt}) on {device}...")
        try:
            from ultralytics import YOLOWorld
            model = YOLOWorld(ckpt)
            model.set_classes(FASHION_VOCABULARY)
            _CACHED_DETECTOR = model
            _CACHED_PROCESSOR = None
            _CACHED_MODEL_NAME = model_name
            _CACHED_DEVICE = device
            print(f"[INFO] YOLO-World ready with {len(FASHION_VOCABULARY)} fashion categories.")
            return _CACHED_DETECTOR, None
        except Exception as e:
            print(f"[WARN] Failed to load YOLO-World ({e}), falling back to YOLOS-Fashionpedia...")
            model_name = "valentinafeve/yolos-fashionpedia"

    # Fallback / YOLOS model
    print(f"[INFO] Loading YOLOS-Fashionpedia ({model_name}) on {device}...")
    from transformers import YolosImageProcessor, YolosForObjectDetection
    processor = YolosImageProcessor.from_pretrained(model_name)
    model = YolosForObjectDetection.from_pretrained(model_name).to(device)
    model.eval()

    _CACHED_DETECTOR = model
    _CACHED_PROCESSOR = processor
    _CACHED_MODEL_NAME = model_name
    _CACHED_DEVICE = device
    print("[INFO] YOLOS-Fashionpedia ready.")
    return _CACHED_DETECTOR, _CACHED_PROCESSOR


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Calculates Intersection over Union (IoU) between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def apply_nms(detections: List[Dict], iou_threshold: float = 0.50) -> List[Dict]:
    """
    Applies Non-Maximum Suppression (NMS) within each category
    and Cross-Category Suppression for competing garment definitions on the same body region.
    """
    if not detections:
        return []

    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []

    for cand in sorted_dets:
        discard = False
        for k in kept:
            iou = calculate_iou(cand["bbox"], k["bbox"])
            # Same category overlap
            if cand.get("category") == k.get("category") and iou > 0.40:
                discard = True
                break
            # Competing upper garment classes (e.g. sweater vs t-shirt vs blouse vs dress on same torso)
            cand_broad = cand.get("broad_category")
            k_broad = k.get("broad_category")
            if cand_broad is not None and k_broad is not None and cand_broad == k_broad and iou > 0.45:
                discard = True
                break
            # Extreme box overlap
            if iou > 0.65:
                discard = True
                break
        if not discard:
            kept.append(cand)

    return kept


def detect_garments(
    image_path: str,
    conf_threshold: float = 0.20,
    model_name: str = DEFAULT_MODEL_NAME,
    specific: bool = True,
    allowed_categories: Optional[Set[str]] = None,
    device: Optional[str] = None
) -> Tuple[Image.Image, List[Dict]]:
    """
    Runs garment object detection on a real user photo.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    dev = device or get_default_device()

    detector, processor = load_detection_model(model_name=model_name, device=dev)
    detections = []

    is_yolo_world = processor is None

    if is_yolo_world:
        # YOLO-World inference
        results = detector.predict(image, conf=conf_threshold, verbose=False, device=dev)[0]
        for box in results.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            raw_label = detector.names[cls_id]

            clean_category = map_category(raw_label, specific=specific, allowed_categories=allowed_categories)
            if clean_category is None:
                continue

            xyxy = box.xyxy[0].tolist()
            x1 = max(0, int(round(xyxy[0])))
            y1 = max(0, int(round(xyxy[1])))
            x2 = min(orig_w, int(round(xyxy[2])))
            y2 = min(orig_h, int(round(xyxy[3])))

            if (x2 - x1) < 8 or (y2 - y1) < 8:
                continue

            specific_cat = map_category(raw_label, specific=True) or clean_category
            broad_cat = map_category(raw_label, specific=False) or "accessory"

            detections.append({
                "category": clean_category,
                "specific_category": specific_cat,
                "broad_category": broad_cat,
                "raw_category": raw_label,
                "confidence": round(conf, 4),
                "bbox": [x1, y1, x2, y2]
            })

    else:
        # YOLOS transformer inference
        inputs = processor(images=image, return_tensors="pt").to(dev)
        with torch.no_grad():
            outputs = detector(**inputs)

        target_sizes = torch.tensor([[orig_h, orig_w]], device=dev)
        results = processor.post_process_object_detection(
            outputs, threshold=conf_threshold, target_sizes=target_sizes
        )[0]

        for score, label_idx, box in zip(results["scores"], results["labels"], results["boxes"]):
            score_val = score.item()
            label_name = detector.config.id2label[label_idx.item()]

            clean_category = map_category(label_name, specific=specific, allowed_categories=allowed_categories)
            if clean_category is None:
                continue

            x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            if (x2 - x1) < 8 or (y2 - y1) < 8:
                continue

            specific_cat = map_category(label_name, specific=True) or clean_category
            broad_cat = map_category(label_name, specific=False) or "accessory"

            detections.append({
                "category": clean_category,
                "specific_category": specific_cat,
                "broad_category": broad_cat,
                "raw_category": label_name,
                "confidence": round(score_val, 4),
                "bbox": [x1, y1, x2, y2]
            })

    # Apply deduplication
    cleaned_detections = apply_nms(detections, iou_threshold=0.45)
    return image, cleaned_detections


def draw_visual_overlay(
    image: Image.Image,
    detections: List[Dict]
) -> Image.Image:
    """Draws bounding boxes and garment category labels on a copy of the input image."""
    vis_img = image.copy()
    draw = ImageDraw.Draw(vis_img)

    COLOR_PALETTE = {
        "top": (59, 130, 246),
        "t_shirt": (59, 130, 246),
        "shirt_blouse": (37, 99, 235),
        "sweater": (29, 78, 216),
        "cardigan": (30, 64, 175),
        "bottom": (34, 197, 94),
        "pants": (34, 197, 94),
        "shorts": (16, 185, 129),
        "skirt": (5, 150, 105),
        "outerwear": (249, 115, 22),
        "jacket": (249, 115, 22),
        "coat": (234, 88, 12),
        "vest": (194, 65, 12),
        "dress": (236, 72, 153),
        "jumpsuit": (219, 39, 119),
        "shoe": (168, 85, 247),
        "bag": (234, 179, 8),
        "sunglasses": (245, 158, 11),
        "glasses": (245, 158, 11),
        "watch": (217, 119, 6),
        "hat": (14, 165, 233),
        "accessory": (234, 179, 8),
    }

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cat = det["category"]
        conf = det["confidence"]
        color = COLOR_PALETTE.get(cat, (239, 68, 68))

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label_text = f"{cat} ({conf:.2f})"
        text_bbox = draw.textbbox((x1, max(0, y1 - 18)), label_text, font=font) if font else (x1, y1 - 18, x1 + 80, y1)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1 + 2, max(0, y1 - 18) + 1), label_text, fill=(255, 255, 255), font=font)

    return vis_img


def process_image(
    image_path: str,
    output_root: str = "outputs",
    image_id: Optional[str] = None,
    conf_threshold: float = 0.20,
    model_name: str = DEFAULT_MODEL_NAME,
    specific: bool = True,
    remove_bg: bool = True,
    save_vis: bool = True,
    allowed_categories: Optional[Set[str]] = None,
    device: Optional[str] = None
) -> Dict:
    """
    Complete Stage 1 pipeline:
    Detect specific garments -> Crop & Background Removal -> Save & Metadata.
    """
    if image_id is None:
        image_id = os.path.splitext(os.path.basename(image_path))[0]

    out_dir = os.path.join(output_root, image_id)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[PROCESS] Processing '{image_path}' (ID: {image_id})...")
    image, detections = detect_garments(
        image_path,
        conf_threshold=conf_threshold,
        model_name=model_name,
        specific=specific,
        allowed_categories=allowed_categories,
        device=device
    )

    garments_meta = []
    category_counters = {}

    segformer_pred_mask = None
    if remove_bg and detections:
        print("  [BG] Isolating pixel-perfect garment contours with SegFormer...")
        try:
            segformer_pred_mask, _ = get_fashion_semantic_mask(image, device=device)
        except Exception as e:
            print(f"  [WARN] SegFormer failed ({e}), using bounding box crops...")

    for det in detections:
        cat = det["category"]
        category_counters[cat] = category_counters.get(cat, 0) + 1
        idx = category_counters[cat] - 1
        filename = f"{cat}_{idx}.png"
        out_file_path = os.path.join(out_dir, filename)

        x1, y1, x2, y2 = det["bbox"]
        if remove_bg and segformer_pred_mask is not None:
            crop_rgba = isolate_garment_with_segformer(
                image,
                bbox=[x1, y1, x2, y2],
                category=cat,
                segformer_pred_mask=segformer_pred_mask,
                device=device
            )
        else:
            crop_rgba = image.crop((x1, y1, x2, y2)).convert("RGBA")

        crop_rgba.save(out_file_path, format="PNG")

        meta_entry = {
            "file": filename,
            "category": cat,
            "specific_category": det["specific_category"],
            "broad_category": det["broad_category"],
            "raw_category": det["raw_category"],
            "confidence": det["confidence"],
            "bbox": det["bbox"],
            "crop_path": out_file_path
        }
        garments_meta.append(meta_entry)
        print(f"  [SAVED] {filename} (Garment: '{cat}', conf={det['confidence']:.2f})")

    # Save metadata.json matching repository contract
    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(garments_meta, f, indent=2)

    # Save visual overlay
    if save_vis:
        vis_image = draw_visual_overlay(image, detections)
        vis_path = os.path.join(out_dir, f"{image_id}_vis.jpg")
        vis_image.save(vis_path, quality=95)
        root_vis_path = os.path.join(output_root, f"{image_id}_vis.jpg")
        try:
            vis_image.save(root_vis_path, quality=95)
        except Exception:
            pass
        print(f"  [SAVED] Visual overlay saved to {vis_path}")

    print(f"[DONE] Extracted {len(garments_meta)} specific garments for {image_id}.\n")
    return {
        "image_id": image_id,
        "output_dir": out_dir,
        "garments": garments_meta,
        "metadata_path": meta_path,
        "num_garments": len(garments_meta)
    }
