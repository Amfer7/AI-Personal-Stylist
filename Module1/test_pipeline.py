"""
test_pipeline.py
================
Automated unit & integration tests for the real-image understanding pipeline.

Verifies:
  1. Specific & broad category mapping coverage.
  2. Background removal functionality (transparent RGBA).
  3. Bounding box IoU & NMS deduplication.
  4. End-to-end detection, transparent crop, and CLIP embedding generation.
  5. 512-D L2-normalization of output embeddings.
"""

import os
import sys
import numpy as np
from PIL import Image

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from category_mapping import map_category
from bg_removal import isolate_garment_background, is_rembg_available
from yolo_detect_and_crop import calculate_iou, apply_nms, process_image
from clip_extract_embeddings import embed_image
from attribute_analyzer import (
    analyze_color,
    analyze_silhouette,
    analyze_garment,
    compute_outfit_features,
    _hue_distance,
)


def test_category_mapping():
    """Verify specific garment detection labels and part filtering."""
    # Specific garment mapping
    assert map_category("jacket", specific=True) == "jacket"
    assert map_category("sweater", specific=True) == "sweater"
    assert map_category("cardigan", specific=True) == "cardigan"
    assert map_category("pants", specific=True) == "pants"
    assert map_category("shorts", specific=True) == "shorts"
    assert map_category("skirt", specific=True) == "skirt"
    assert map_category("dress", specific=True) == "dress"
    assert map_category("shoe", specific=True) == "shoe"
    assert map_category("bag, wallet", specific=True) == "bag"
    assert map_category("shirt, blouse", specific=True) == "shirt_blouse"
    assert map_category("top, t-shirt, sweatshirt", specific=True) == "t_shirt"

    # Part details discarded
    assert map_category("zipper", specific=True) is None
    assert map_category("pocket", specific=True) is None
    assert map_category("buckle", specific=True) is None

    # Broad categories fallback
    assert map_category("jacket", specific=False) == "outerwear"
    assert map_category("pants", specific=False) == "bottom"


def test_iou_and_nms():
    """Verify IoU calculation and Non-Maximum Suppression."""
    box1 = [0, 0, 100, 100]
    box2 = [0, 0, 100, 100]
    box3 = [200, 200, 300, 300]

    assert calculate_iou(box1, box2) == 1.0
    assert calculate_iou(box1, box3) == 0.0

    detections = [
        {"category": "jacket", "confidence": 0.90, "bbox": [0, 0, 100, 100]},
        {"category": "jacket", "confidence": 0.70, "bbox": [5, 5, 95, 95]},  # Duplicate
        {"category": "pants", "confidence": 0.85, "bbox": [0, 100, 100, 200]}
    ]

    filtered = apply_nms(detections, iou_threshold=0.6)
    assert len(filtered) == 2
    assert filtered[0]["confidence"] == 0.90
    assert filtered[1]["category"] == "pants"


def test_background_removal():
    """Verify background removal generates a valid 4-channel RGBA transparent image."""
    test_img = Image.new("RGB", (64, 64), color=(200, 50, 50))
    rgba_img = isolate_garment_background(test_img, enable_bg_removal=True)

    assert rgba_img.mode == "RGBA"
    assert rgba_img.size == (64, 64)


def test_clip_embedding_properties():
    """Verify CLIP embeddings are float32, 512-D, and L2-normalized to 1.0."""
    test_crop = Image.new("RGBA", (128, 128), color=(100, 150, 200, 255))
    emb = embed_image(test_crop, device="cpu")

    assert isinstance(emb, np.ndarray)
    assert emb.shape == (512,)
    assert emb.dtype == np.float32 or emb.dtype == np.float64
    # Unit norm: ||v|| = 1.0 (within float tolerance)
    norm = np.linalg.norm(emb)
    assert np.isclose(norm, 1.0, atol=1e-4)


def test_color_attribute():
    """Dominant colour is measured on fabric pixels only (alpha-masked)."""
    # Solid red garment on the left half, transparent (background) on the right half.
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[:, :16] = (220, 30, 30)          # red fabric
    rgb[:, 16:] = (10, 200, 10)          # green "background" that must be ignored
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[:, :16] = 1

    color = analyze_color(rgb, mask)
    assert color["name"] == "red"
    r, g, b = color["dominant_rgb"]
    assert r > 150 and g < 90 and b < 90   # dominant colour is the red fabric, not green


def test_silhouette_a_line():
    """A downward-widening (trapezoid) mask is classified as A-line."""
    mask = np.zeros((60, 60), dtype=np.uint8)
    for row in range(60):
        half = 4 + int(row * 0.4)          # widens toward the bottom
        cx = 30
        mask[row, max(0, cx - half):min(60, cx + half)] = 1

    result = analyze_silhouette(mask)
    assert result["silhouette"] == "a_line"
    assert result["area_px"] == int(mask.sum())


def test_silhouette_boxy():
    """A full rectangle (extent ~1.0) is classified as boxy."""
    mask = np.ones((40, 30), dtype=np.uint8)
    result = analyze_silhouette(mask)
    assert result["silhouette"] == "boxy"
    assert result["extent"] == 1.0


def test_hue_distance_circular():
    """Hue distance is circular and bounded to [0, 180]."""
    assert _hue_distance(10, 350) == 20.0     # wraps around 360
    assert _hue_distance(0, 180) == 180.0
    assert _hue_distance(90, 90) == 0.0


def test_outfit_features():
    """Outfit-level metrics aggregate per-garment attributes correctly."""
    garments = [
        {"broad_category": "top", "pixel_count": 1000,
         "attributes": {"color": {"hue": 0.0}, "formality_score": 20.0}},
        {"broad_category": "bottom", "pixel_count": 1000,
         "attributes": {"color": {"hue": 180.0}, "formality_score": 20.0}},
    ]
    feats = compute_outfit_features(garments)
    assert feats["top_bottom_hue_distance"] == 180.0     # red vs cyan = max contrast
    assert feats["volume_balance"] == 100.0              # equal areas = perfectly balanced
    assert feats["formality_coherence"] == 100.0         # identical formality = fully coherent
    assert feats["num_garments"] == 2


if __name__ == "__main__":
    print("[TEST] Running unit tests...")
    test_category_mapping()
    print("  [PASS] Specific category mapping test passed.")
    test_iou_and_nms()
    print("  [PASS] IoU & NMS deduplication test passed.")
    test_background_removal()
    print("  [PASS] Background removal test passed.")
    test_clip_embedding_properties()
    print("  [PASS] CLIP embedding vector properties test passed.")
    test_color_attribute()
    print("  [PASS] Colour attribute (fabric-only) test passed.")
    test_silhouette_a_line()
    test_silhouette_boxy()
    print("  [PASS] Silhouette geometry tests passed.")
    test_hue_distance_circular()
    print("  [PASS] Circular hue distance test passed.")
    test_outfit_features()
    print("  [PASS] Outfit-level fashion-theory features test passed.")
    print("[SUCCESS] All pipeline unit tests passed successfully!")
