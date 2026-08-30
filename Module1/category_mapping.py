"""
category_mapping.py
===================
Maps fine-grained detector labels (from YOLO-World and YOLOS-Fashionpedia) to 
specific, clean clothing categories (e.g. 'jacket', 'blouse', 'sweater', 'pants', 'shorts', 'skirt', 'dress'):
  - Specific garment types are preserved (no over-generalization).
  - Part-level labels (e.g. collar, sleeve, zipper, pocket) and fine embellishments
    are mapped to None (discarded) so that downstream stages only receive whole garments.
"""

from typing import Optional, Dict, Set

# Standard specific clothing taxonomy for YOLO-World & YOLOS
FASHION_LABEL_TO_SPECIFIC_CATEGORY: Dict[str, Optional[str]] = {
    # ── Tops & Shirts ──
    "t-shirt": "t_shirt",
    "t_shirt": "t_shirt",
    "shirt": "shirt_blouse",
    "blouse": "shirt_blouse",
    "shirt, blouse": "shirt_blouse",
    "top, t-shirt, sweatshirt": "t_shirt",
    "top": "top",
    "sweater": "sweater",
    "cardigan": "cardigan",
    "sweatshirt": "t_shirt",
    "hoodie": "t_shirt",

    # ── Outerwear ──
    "jacket": "jacket",
    "vest": "vest",
    "coat": "coat",
    "cape": "cape",
    "blazer": "jacket",

    # ── Bottoms ──
    "pants": "pants",
    "jeans": "pants",
    "trousers": "pants",
    "shorts": "shorts",
    "skirt": "skirt",
    "tights, stockings": "stockings",
    "stockings": "stockings",
    "leggings": "pants",

    # ── Full-body / Dresses ──
    "dress": "dress",
    "jumpsuit": "jumpsuit",
    "romper": "jumpsuit",

    # ── Footwear ──
    "shoe": "shoe",
    "shoes": "shoe",
    "sneakers": "shoe",
    "boots": "shoe",
    "sandals": "shoe",
    "heels": "shoe",

    # ── Bags ──
    "bag": "bag",
    "handbag": "bag",
    "backpack": "bag",
    "tote bag": "bag",
    "bag, wallet": "bag",

    # ── Accessories ──
    "sunglasses": "sunglasses",
    "glasses": "glasses",
    "hat": "hat",
    "cap": "hat",
    "beanie": "hat",
    "headband": "headband",
    "headband, head covering, hair accessory": "headband",
    "tie": "tie",
    "glove": "glove",
    "gloves": "glove",
    "watch": "watch",
    "belt": "belt",
    "leg warmer": "leg_warmer",
    "sock": "sock",
    "socks": "sock",
    "scarf": "scarf",
    "umbrella": "umbrella",
    "hood": "hood",

    # ── Part-level details & hardware (Discarded) ──
    "collar": None,
    "lapel": None,
    "epaulette": None,
    "sleeve": None,
    "pocket": None,
    "neckline": None,
    "buckle": None,
    "zipper": None,
    "applique": None,
    "bead": None,
    "bow": None,
    "flower": None,
    "fringe": None,
    "ribbon": None,
    "rivet": None,
    "ruffle": None,
    "sequin": None,
    "tassel": None,
}

# Broad grouped mapping (Top / Bottom / Outerwear / Dress / Shoe / Bag / Accessory)
FASHION_LABEL_TO_BROAD_CATEGORY: Dict[str, Optional[str]] = {
    "t-shirt": "top",
    "t_shirt": "top",
    "shirt": "top",
    "blouse": "top",
    "shirt, blouse": "top",
    "top, t-shirt, sweatshirt": "top",
    "top": "top",
    "sweater": "top",
    "cardigan": "top",
    "sweatshirt": "top",
    "hoodie": "top",

    "jacket": "outerwear",
    "vest": "outerwear",
    "coat": "outerwear",
    "cape": "outerwear",
    "blazer": "outerwear",

    "pants": "bottom",
    "jeans": "bottom",
    "trousers": "bottom",
    "shorts": "bottom",
    "skirt": "bottom",
    "tights, stockings": "bottom",
    "stockings": "bottom",
    "leggings": "bottom",

    "dress": "dress",
    "jumpsuit": "dress",
    "romper": "dress",

    "shoe": "shoe",
    "shoes": "shoe",
    "sneakers": "shoe",
    "boots": "shoe",
    "sandals": "shoe",
    "heels": "shoe",

    "bag": "bag",
    "handbag": "bag",
    "backpack": "bag",
    "tote bag": "bag",
    "bag, wallet": "bag",

    "sunglasses": "accessory",
    "glasses": "accessory",
    "hat": "accessory",
    "cap": "accessory",
    "beanie": "accessory",
    "headband": "accessory",
    "headband, head covering, hair accessory": "accessory",
    "tie": "accessory",
    "glove": "accessory",
    "gloves": "accessory",
    "watch": "accessory",
    "belt": "accessory",
    "leg warmer": "accessory",
    "sock": "accessory",
    "socks": "accessory",
    "scarf": "accessory",
    "umbrella": "accessory",
    "hood": "accessory",
}

# Backward compatibility alias
FASHIONPEDIA_TO_SPECIFIC_CATEGORY = FASHION_LABEL_TO_SPECIFIC_CATEGORY
FASHIONPEDIA_TO_BROAD_CATEGORY = FASHION_LABEL_TO_BROAD_CATEGORY


def map_category(
    label: str,
    specific: bool = True,
    allowed_categories: Optional[Set[str]] = None
) -> Optional[str]:
    """
    Maps a raw detector label to a clean clothing category.

    Args:
        label: Raw string label from detector model (e.g. 't-shirt', 'pants', 'sunglasses').
        specific: If True, preserves specific identity (e.g. 'jacket', 'skirt').
                  If False, maps to broad groups (e.g. 'outerwear', 'bottom').
        allowed_categories: Optional whitelist filter.

    Returns:
        Clean category string (e.g. 'jacket', 'skirt') or None if discarded/unsupported.
    """
    label_lower = label.lower().strip()
    mapping = FASHION_LABEL_TO_SPECIFIC_CATEGORY if specific else FASHION_LABEL_TO_BROAD_CATEGORY
    mapped = mapping.get(label_lower, None)

    if mapped is None and specific:
        # If not explicitly mapped, sanitize raw string directly if not a known sub-part
        if label_lower not in FASHION_LABEL_TO_SPECIFIC_CATEGORY:
            mapped = label_lower.replace(" ", "_").replace("-", "_")

    if mapped is not None and allowed_categories is not None:
        if mapped not in allowed_categories:
            return None

    return mapped
