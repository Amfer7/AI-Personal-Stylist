"""
attribute_analyzer.py
=====================
Per-garment attribute extraction and outfit-level fashion-theory features for the
SegFormer garment pipeline (Module 1).

Two families of signals, both computed from data the SegFormer path already produces
(the transparent RGBA crop, whose alpha channel *is* the pixel-accurate garment mask):

  1. Pixel-based attributes (deterministic, model-free, ~free):
       - color: dominant RGB + HSV + human-readable name, measured on fabric pixels only
       - silhouette: shape class (fitted / boxy / a_line / tapered) from mask geometry

  2. Semantic attributes (CLIP zero-shot, reuses the already-loaded ViT-B/32 model):
       - pattern  (solid / striped / checked / floral / graphic / dotted)
       - fabric   (denim / cotton / leather / knit / silk / corduroy)
       - fit       (slim / regular / oversized)
       - formality (casual / smart_casual / formal) + continuous formality_score 0-100

Outfit-level fashion-theory features (arithmetic on the per-garment attributes):
       - hue_contrast, top_bottom_hue_distance   (circular hue geometry)
       - volume_balance                          (relative garment areas)
       - formality_coherence                     (spread of formality across the outfit)

The CLIP text features for the semantic prompt sets are encoded once and cached, so the
marginal cost per garment is a single image encode (which the pipeline already performs
for subcategory classification).
"""

import colorsys
import contextlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

try:
    import torch
    import clip
except ImportError:  # graceful degradation: pixel attributes still work without CLIP
    torch = None
    clip = None


def _amp(dev):
    """fp16 autocast on CUDA (Tier 0 speedup), no-op on CPU."""
    if torch is not None and str(dev).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


# ─────────────────────────────────────────────────────────────────────────────
# CLIP zero-shot prompt groups. Each entry: (label, prompt[, formality_value]).
# ─────────────────────────────────────────────────────────────────────────────
PATTERN_GROUP = [
    ("solid", "a solid color plain garment"),
    ("striped", "a striped garment"),
    ("checked", "a checked or plaid garment"),
    ("floral", "a floral patterned garment"),
    ("graphic", "a garment with a graphic print or logo"),
    ("dotted", "a polka dot garment"),
]
FABRIC_GROUP = [
    ("denim", "a denim garment"),
    ("cotton", "a cotton garment"),
    ("leather", "a leather garment"),
    ("knit", "a wool or knit garment"),
    ("silk", "a silk or satin garment"),
    ("corduroy", "a corduroy garment"),
]
FIT_GROUP = [
    ("slim", "a slim fit tight garment"),
    ("regular", "a regular fit garment"),
    ("oversized", "a loose oversized baggy garment"),
]
FORMALITY_GROUP = [
    ("casual", "casual everyday streetwear clothing", 0.0),
    ("smart_casual", "smart casual clothing", 50.0),
    ("formal", "formal elegant business or evening wear", 100.0),
]

# Cache of encoded text features keyed by (id(clip_model), group_name).
_TEXT_CACHE: Dict[Tuple[int, str], "torch.Tensor"] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Pixel-based attributes
# ─────────────────────────────────────────────────────────────────────────────
def _dominant_rgb(pixels: np.ndarray, max_samples: int = 5000) -> np.ndarray:
    """Returns the dominant RGB colour of a set of fabric pixels via k-means."""
    if len(pixels) == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    data = pixels.astype(np.float32)
    if len(data) > max_samples:
        idx = np.random.choice(len(data), max_samples, replace=False)
        data = data[idx]

    n_unique = len(np.unique(data, axis=0))
    if n_unique < 2:
        return data.mean(axis=0)

    k = min(3, n_unique)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    return centers[int(np.argmax(counts))]


def _color_name(h_deg: float, s: float, v: float) -> str:
    """Maps HSV (h in degrees, s/v in [0,1]) to a coarse human-readable colour name."""
    if v < 0.15:
        return "black"
    if s < 0.12:
        return "white" if v > 0.85 else "gray"
    buckets = [
        (15, "red"), (45, "orange"), (65, "yellow"), (160, "green"),
        (200, "cyan"), (255, "blue"), (290, "purple"), (330, "pink"), (360, "red"),
    ]
    for hi, name in buckets:
        if h_deg < hi:
            return name
    return "red"


def analyze_color(rgb: np.ndarray, mask: np.ndarray) -> Dict:
    """
    Computes the dominant colour of a garment from its fabric pixels only.

    Args:
        rgb:  (H, W, 3) uint8 array.
        mask: (H, W) boolean/uint8 array; True where fabric is present.
    """
    fabric = rgb[mask.astype(bool)]
    if len(fabric) == 0:
        return {"dominant_rgb": [0, 0, 0], "hue": None, "saturation": None,
                "value": None, "name": "unknown"}

    dom = _dominant_rgb(fabric)
    r, g, b = (float(np.clip(c, 0, 255)) / 255.0 for c in dom)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h_deg = round(h * 360.0, 1)

    return {
        "dominant_rgb": [int(round(c)) for c in dom],
        "hue": h_deg,
        "saturation": round(s, 3),
        "value": round(v, 3),
        "name": _color_name(h_deg, s, v),
    }


def analyze_silhouette(mask: np.ndarray) -> Dict:
    """Classifies garment shape from the geometry of its binary mask."""
    mask = mask.astype(bool)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return {"silhouette": "unknown", "aspect_ratio": None,
                "extent": None, "area_px": 0}

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    h = int(y2 - y1 + 1)
    w = int(x2 - x1 + 1)
    area = int(mask.sum())
    extent = area / float(h * w)          # fill ratio within the bounding box
    aspect = h / float(w)

    # Width profile: mean per-row garment width across the top / middle / bottom thirds.
    row_widths = mask[y1:y2 + 1, :].sum(axis=1).astype(float)
    thirds = np.array_split(row_widths, 3)
    top_w = float(thirds[0].mean()) if len(thirds[0]) else 0.0
    bot_w = float(thirds[2].mean()) if len(thirds[2]) else 0.0

    if bot_w > top_w * 1.25:
        silhouette = "a_line"        # flares outward toward the hem (skirts, dresses)
    elif top_w > bot_w * 1.25:
        silhouette = "tapered"       # narrows toward the hem
    elif extent > 0.75:
        silhouette = "boxy"          # fills its box: straight / relaxed
    else:
        silhouette = "fitted"        # contoured

    return {
        "silhouette": silhouette,
        "aspect_ratio": round(aspect, 3),
        "extent": round(extent, 3),
        "area_px": area,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Semantic attributes (CLIP zero-shot)
# ─────────────────────────────────────────────────────────────────────────────
def _encode_text_group(clip_model, device, group_name: str, group: List) -> "torch.Tensor":
    """Encodes (and caches) the text features for one prompt group."""
    key = (id(clip_model), group_name)
    if key not in _TEXT_CACHE:
        prompts = [entry[1] for entry in group]
        tokens = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            tf = clip_model.encode_text(tokens)
            tf = tf / tf.norm(dim=-1, keepdim=True)
        _TEXT_CACHE[key] = tf
    return _TEXT_CACHE[key]


def _zero_shot(image_feat, clip_model, device, group_name: str, group: List) -> Tuple[str, float, np.ndarray]:
    """Returns (best_label, best_confidence, probability_vector) for a prompt group."""
    tf = _encode_text_group(clip_model, device, group_name, group)
    with torch.no_grad():
        logits = 100.0 * image_feat @ tf.T
        probs = logits.softmax(dim=-1).cpu().numpy()[0]
    best = int(np.argmax(probs))
    return group[best][0], round(float(probs[best]), 4), probs


def analyze_semantic(crop_rgba, clip_model, clip_preprocess, device: str) -> Dict:
    """Runs CLIP zero-shot pattern / fabric / fit / formality on one garment crop."""
    from PIL import Image

    # Composite onto white so transparency does not bias the encoder.
    if crop_rgba.mode == "RGBA":
        white = Image.new("RGBA", crop_rgba.size, (255, 255, 255, 255))
        rgb = Image.alpha_composite(white, crop_rgba).convert("RGB")
    else:
        rgb = crop_rgba.convert("RGB")

    img_tensor = clip_preprocess(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        image_feat = clip_model.encode_image(img_tensor)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

    pattern, pattern_c, _ = _zero_shot(image_feat, clip_model, device, "pattern", PATTERN_GROUP)
    fabric, fabric_c, _ = _zero_shot(image_feat, clip_model, device, "fabric", FABRIC_GROUP)
    fit, fit_c, _ = _zero_shot(image_feat, clip_model, device, "fit", FIT_GROUP)
    formality, formality_c, form_probs = _zero_shot(
        image_feat, clip_model, device, "formality", FORMALITY_GROUP
    )
    values = np.array([entry[2] for entry in FORMALITY_GROUP])
    formality_score = round(float((form_probs * values).sum()), 1)

    return {
        "pattern": pattern, "pattern_conf": pattern_c,
        "fabric": fabric, "fabric_conf": fabric_c,
        "fit": fit, "fit_conf": fit_c,
        "formality": formality, "formality_conf": formality_c,
        "formality_score": formality_score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point (per garment)
# ─────────────────────────────────────────────────────────────────────────────
def analyze_garment(
    crop_rgba,
    clip_model=None,
    clip_preprocess=None,
    device: str = "cpu",
) -> Dict:
    """
    Computes the full attribute block for a single garment crop.

    Pixel attributes (colour, silhouette) are always computed. Semantic attributes are
    added only when a CLIP model/preprocess pair is supplied.

    Args:
        crop_rgba: PIL RGBA image whose alpha channel encodes the garment mask.
    """
    arr = np.array(crop_rgba.convert("RGBA"))
    rgb = arr[:, :, :3]
    mask = arr[:, :, 3] > 0

    attrs: Dict = {"color": analyze_color(rgb, mask)}
    attrs.update(analyze_silhouette(mask))

    if clip_model is not None and clip_preprocess is not None and clip is not None:
        attrs.update(analyze_semantic(crop_rgba, clip_model, clip_preprocess, device))

    return attrs


# ─────────────────────────────────────────────────────────────────────────────
# Outfit-level fashion-theory features
# ─────────────────────────────────────────────────────────────────────────────
def _hue_distance(h1: float, h2: float) -> float:
    """Circular distance between two hues in degrees, range [0, 180]."""
    d = abs(h1 - h2) % 360.0
    return d if d <= 180.0 else 360.0 - d


def compute_outfit_features(garments: List[Dict]) -> Dict:
    """
    Derives outfit-level fashion-theory metrics from per-garment attributes.

    Args:
        garments: list of metadata dicts, each with an "attributes" block and a
                  "broad_category" / "pixel_count" field.
    """
    hues, formality_scores = [], []
    top_hues, bottom_hues = [], []
    top_area, bottom_area = 0, 0

    for g in garments:
        attrs = g.get("attributes", {}) or {}
        broad = g.get("broad_category", "accessory")
        area = int(g.get("pixel_count", attrs.get("area_px", 0) or 0))

        hue = (attrs.get("color") or {}).get("hue")
        if hue is not None:
            hues.append(hue)
            if broad in ("top", "outerwear"):
                top_hues.append(hue)
            elif broad == "bottom":
                bottom_hues.append(hue)

        fscore = attrs.get("formality_score")
        if fscore is not None:
            formality_scores.append(fscore)

        if broad in ("top", "outerwear"):
            top_area += area
        elif broad == "bottom":
            bottom_area += area

    # Overall hue contrast: mean pairwise circular distance.
    hue_contrast = None
    if len(hues) >= 2:
        dists = [_hue_distance(hues[i], hues[j])
                 for i in range(len(hues)) for j in range(i + 1, len(hues))]
        hue_contrast = round(float(np.mean(dists)), 1)

    top_bottom_hue_distance = None
    if top_hues and bottom_hues:
        top_bottom_hue_distance = round(_hue_distance(top_hues[0], bottom_hues[0]), 1)

    # Volume balance: 100 = perfectly balanced top/bottom areas, →0 = very lopsided.
    volume_balance = None
    top_bottom_area_ratio = None
    if top_area > 0 and bottom_area > 0:
        lo, hi = sorted((top_area, bottom_area))
        volume_balance = round(100.0 * lo / hi, 1)
        top_bottom_area_ratio = round(top_area / float(bottom_area), 3)

    # Formality coherence: 100 = everything at the same formality level.
    formality_coherence = None
    if len(formality_scores) >= 2:
        std = float(np.std(formality_scores))
        formality_coherence = round(max(0.0, 100.0 - std), 1)

    return {
        "hue_contrast": hue_contrast,
        "top_bottom_hue_distance": top_bottom_hue_distance,
        "volume_balance": volume_balance,
        "top_bottom_area_ratio": top_bottom_area_ratio,
        "formality_coherence": formality_coherence,
        "num_garments": len(garments),
    }
