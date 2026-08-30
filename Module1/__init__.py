"""
real_image_pipeline
===================
Real-Image Garment Understanding & Representation Pipeline:
Detection (YOLOS-Fashionpedia) -> Transparent Extraction (Alpha) -> CLIP Embeddings (512-D).
"""

from .category_mapping import (
    map_category,
    FASHIONPEDIA_TO_SPECIFIC_CATEGORY,
    FASHIONPEDIA_TO_BROAD_CATEGORY
)
from .bg_removal import isolate_garment_background, is_rembg_available
from .yolo_detect_and_crop import detect_garments, process_image
from .clip_extract_embeddings import embed_image, extract_folder_embeddings, extract_all_outputs
from .attribute_analyzer import (
    analyze_garment,
    analyze_color,
    analyze_silhouette,
    compute_outfit_features,
)

__all__ = [
    "map_category",
    "FASHIONPEDIA_TO_SPECIFIC_CATEGORY",
    "FASHIONPEDIA_TO_BROAD_CATEGORY",
    "isolate_garment_background",
    "is_rembg_available",
    "detect_garments",
    "process_image",
    "embed_image",
    "extract_folder_embeddings",
    "extract_all_outputs",
    "analyze_garment",
    "analyze_color",
    "analyze_silhouette",
    "compute_outfit_features"
]
