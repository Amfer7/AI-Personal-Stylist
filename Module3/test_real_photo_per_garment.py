"""
test_real_photo_per_garment.py

The per-garment equivalent of the earlier gradio_demo.py test. Segments a
real photo into individual garments (top, bottom, shoes, ...) using
fashion_segmenter.py, then scores EACH garment separately against the
per-category trend model — rather than treating the whole photo as one
style, this tells you e.g. "top: 82/100, bottom: 34/100" and flags the
weak link, the same shape of output Module 4 needs to combine with the
GNN's harmony score.

Usage:
    python test_real_photo_per_garment.py \
        --image /path/to/outfit_photo.jpg \
        --model /content/drive/MyDrive/Poster/per_category_trend_model.pkl
"""

import argparse

from PIL import Image

import fashion_segmenter
import clip_extract_embeddings as cee
from trend_model import PerCategoryTrendModel


def score_photo(image_path: str, model: PerCategoryTrendModel, device=None):
    image = Image.open(image_path).convert("RGB")
    garments = fashion_segmenter.extract_garment_segments(image, device=device)

    if not garments:
        print("No garments detected in this photo.")
        return None

    garment_inputs = []
    for g in garments:
        emb = cee.embed_image(g["image_crop"], device=device)
        garment_inputs.append({
            "embedding": emb,
            "category": g["broad_category"],
            "specific_category": g["specific_category"],
            "detection_confidence": g["confidence"],
        })

    result = model.score_outfit(garment_inputs)
    return result


def print_result(result: dict):
    print(f"Detected {len(result['garments'])} garment(s):\n")
    for g in result["garments"]:
        score = g["trendiness_score"]
        score_str = f"{score:.1f}/100" if score is not None else "no confident match"
        print(f"  {g['specific_category']:15s} ({g['category']:10s})  "
              f"detection conf={g['detection_confidence']:.2f}  "
              f"trendiness={score_str}")

    print()
    if result["weak_link_category"]:
        print(f"Weak link: '{result['weak_link_category']}' "
              f"(trendiness {result['weak_link_score']:.1f}/100) — "
              f"the item dragging this outfit's overall trend score down.")
    if result["outfit_avg_trendiness"] is not None:
        print(f"Outfit average trendiness: {result['outfit_avg_trendiness']:.1f}/100")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True, help="Path to a saved PerCategoryTrendModel .pkl")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    model = PerCategoryTrendModel.load(args.model)
    print(f"Loaded per-category trend model with categories: {sorted(model.models.keys())}\n")

    result = score_photo(args.image, model, device=args.device)
    if result:
        print_result(result)


if __name__ == "__main__":
    main()
