"""H&M product catalogue for demo suggestions: product photos -> Module 1-style output folders.

Each adult H&M article with a photo and a type Module 4 swaps becomes one single-garment "outfit"
folder, written the way Module 1 writes its own: <out>/outputs/<article_id>/<type>_0.png (the
cut-out, RGBA, cropped to the garment), <type>_0.npy (CLIP embedding, Module 1's embed_image) and
metadata.json (Module 1's attribute block). Module 2's build_feature_store.py then turns the folders
into a feature store, so the catalogue goes through the same feature code as the other stores.

The background is removed with rembg (product shots on a plain light-grey studio background).
A product photo has no body scale, so pixel_count (the GNN's volume-contrast input) is set to the
median garment area of that type in the existing stores; the cut-out's own size is kept in
attributes.area_px.

    python Module4/hm_catalogue.py --out "H&M Dataset/check_store" --per_type 150     # domain check sample
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Module1"))

from m4_settings import M4Settings  # noqa: E402
from swap_groups import BROAD_OF_TYPE, TYPE_ID, TYPES  # noqa: E402

HM_ROOT = "H&M Dataset"
ADULT_GROUPS = ("Ladieswear", "Divided", "Menswear", "Sport")
TYPE_MAP = {
    "T-shirt": "t_shirt", "Polo shirt": "t_shirt",
    "Sweater": "sweater", "Hoodie": "sweater", "Cardigan": "sweater",
    "Shirt": "shirt_blouse", "Blouse": "shirt_blouse",
    "Jacket": "jacket", "Blazer": "jacket", "Coat": "jacket", "Outdoor Waistcoat": "jacket",
    "Top": "top", "Vest top": "top", "Bodysuit": "top",
    "Trousers": "pants", "Outdoor trousers": "pants",
    "Leggings/Tights": "leggings", "Shorts": "shorts", "Skirt": "skirt", "Dress": "dress",
    **{t: "shoe" for t in ("Sneakers", "Boots", "Bootie", "Sandals", "Heeled sandals", "Heels", "Pumps",
                           "Ballerinas", "Flat shoe", "Flat shoes", "Wedge", "Other shoe")},
    **{t: "bag" for t in ("Bag", "Backpack", "Cross-body bag", "Weekend/Gym bag")},
    **{t: "hat" for t in ("Hat/beanie", "Beanie", "Cap/peaked", "Cap", "Hat/brim", "Bucket hat", "Felt hat",
                          "Straw hat")},
    "Belt": "belt", "Scarf": "scarf", "Sunglasses": "sunglasses",
}
MAX_SIDE = 384          # stored cut-out size (CLIP sees 224 px; the demo board shows ~200 px)


def image_path(root: str, article_id: str) -> str:
    return os.path.join(root, "images", article_id[:3], f"{article_id}.jpg")


def load_articles(root: str = HM_ROOT) -> pd.DataFrame:
    """Adult articles with a photo and a Module 4 type. Adds: type, gender ("men" / "women")."""
    a = pd.read_csv(os.path.join(root, "articles.csv"), dtype={"article_id": str})
    a = a[a.index_group_name.isin(ADULT_GROUPS) & (a.section_name != "Kids Sports")].copy()
    a["type"] = a.product_type_name.map(TYPE_MAP)
    a = a[a["type"].notna()]
    men = (a.index_group_name == "Menswear") | a.section_name.str.startswith("Men")
    a["gender"] = np.where(men, "men", "women")
    a = a[[os.path.exists(image_path(root, x)) for x in a.article_id]]
    return a.reset_index(drop=True)


def sample(a: pd.DataFrame, per_type: int, seed: int = 0) -> pd.DataFrame:
    """Up to per_type articles of each type, half men / half women where there are enough."""
    parts = []
    for t in TYPES:
        sub = a[a["type"] == t]
        men, women = sub[sub.gender == "men"], sub[sub.gender == "women"]
        k_m = min(len(men), per_type // 2)
        k_w = min(len(women), per_type - k_m)
        k_m = min(len(men), per_type - k_w)
        parts += [men.sample(k_m, random_state=seed), women.sample(k_w, random_state=seed)]
    return pd.concat(parts).reset_index(drop=True)


def type_areas(settings) -> dict:
    """{type: median garment area (98,304-px frame)} over the existing stores' typed rows."""
    from candidate_pool import load_store_arrays, pool_dir
    meta = np.load(os.path.join(pool_dir(settings), "meta.npz"))
    _, scal = load_store_arrays(settings)
    return {t: float(np.median(scal[meta["type"] == TYPE_ID[t], 2])) for t in TYPES}


def cut_out(img: Image.Image, session) -> Image.Image | None:
    """RGBA cut-out (binary alpha) cropped to the garment and shrunk to MAX_SIDE; None if empty."""
    from rembg import remove
    rgba = np.array(remove(img.convert("RGB"), session=session))
    alpha = rgba[:, :, 3] >= 128
    if alpha.sum() < 100:
        return None
    ys, xs = np.nonzero(alpha)
    rgba[:, :, 3] = np.where(alpha, 255, 0).astype(np.uint8)
    out = Image.fromarray(rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    out.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    a = np.array(out)
    a[:, :, 3] = np.where(a[:, :, 3] >= 128, 255, 0)
    return Image.fromarray(a)


def write_item(out_root: str, row, crop: Image.Image, area: float, clip_bundle) -> None:
    from attribute_analyzer import analyze_garment
    from clip_extract_embeddings import embed_image
    model, preprocess, dev = clip_bundle
    d = os.path.join(out_root, row.article_id)
    os.makedirs(d, exist_ok=True)
    name = f"{row.type}_0"
    crop.save(os.path.join(d, name + ".png"))
    np.save(os.path.join(d, name + ".npy"), embed_image(crop, device=dev))
    meta = [{
        "file": name + ".png", "category": row.type, "specific_category": row.type,
        "broad_category": BROAD_OF_TYPE[row.type], "raw_category": row.product_type_name,
        "confidence": 1.0, "pixel_count": area, "bbox": [0, 0, crop.width, crop.height],
        "attributes": analyze_garment(crop, model, preprocess, dev),
    }]
    with open(os.path.join(d, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def build(out: str, articles: pd.DataFrame, settings, rembg_model: str = "u2net", device=None) -> None:
    import torch
    from clip_extract_embeddings import get_clip_model
    from rembg import new_session
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    clip_bundle = get_clip_model(device=dev)
    session = new_session(rembg_model)
    areas = type_areas(settings)
    out_root = os.path.join(out, "outputs")
    os.makedirs(out_root, exist_ok=True)
    cols = ["article_id", "type", "gender", "product_type_name", "prod_name", "colour_group_name",
            "index_group_name", "section_name"]
    articles[cols].rename(columns={"article_id": "key"}).to_csv(os.path.join(out, "articles.csv"), index=False)
    done = skipped = 0
    for row in articles.itertuples():
        if os.path.exists(os.path.join(out_root, row.article_id, "metadata.json")):
            done += 1
            continue
        crop = cut_out(Image.open(image_path(HM_ROOT, row.article_id)), session)
        if crop is None:
            skipped += 1
            continue
        write_item(out_root, row, crop, areas[row.type], clip_bundle)
        done += 1
        if done % 200 == 0:
            print(f"[hm] {done}/{len(articles)} written, {skipped} empty cut-outs")
    print(f"[hm] done: {done} written, {skipped} empty cut-outs -> {out_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_type", type=int, default=None, help="sample this many per type (default: all)")
    ap.add_argument("--rembg_model", default="u2net")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    s = M4Settings.load()
    os.chdir(s.abs("."))
    a = load_articles()
    print(f"[hm] {len(a)} adult articles with a photo and a Module 4 type")
    print(a.groupby(["type", "gender"]).size().unstack(fill_value=0).to_string())
    if args.per_type:
        a = sample(a, args.per_type)
    build(args.out, a, s, args.rembg_model, args.device)


if __name__ == "__main__":
    main()
