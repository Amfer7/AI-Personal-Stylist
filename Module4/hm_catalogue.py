"""H&M product catalogue for demo suggestions: product photos -> Module 1-style output folders.

Each adult H&M article with a photo and a type Module 4 swaps becomes one single-garment "outfit"
folder, written the way Module 1 writes its own: <out>/outputs/<key>/<type>_0.png (the
cut-out, RGBA, cropped to the garment), <type>_0.npy (CLIP embedding, Module 1's embed_image) and
metadata.json (Module 1's attribute block). <key> is the article id without its leading zero, as
the feature store keys outfits by integer (article_id = key.zfill(10)). Module 2's
build_feature_store.py then turns the folders into a feature store, so the catalogue goes through the
same feature code as the other stores.

The background is removed with rembg (product shots on a plain light-grey studio background).
A product photo has no body scale, so pixel_count (the GNN's volume-contrast input) is set to the
median garment area of that type in the existing stores; the cut-out's own size is kept in
attributes.area_px.

    python Module4/hm_catalogue.py --out "H&M Dataset/check_store" --per_type 150     # domain check sample
    python Module4/hm_catalogue.py --out "H&M Dataset/store"                          # full catalogue
    python Module4/hm_catalogue.py --align "H&M Dataset/store/feature_store"           # after build_feature_store.py

Alignment (--align): CLIP sees the studio-photo style in every H&M embedding (a linear classifier tells
H&M from street crops at 98%), and the GNN scores H&M candidates +0.43 sigma higher for it
(hm_domain_check.py). --method coral (default) maps each type's H&M embeddings onto the mean and
covariance of that type in the street-photo stores (CORAL, Ledoit-Wolf on the H&M side), then
renormalises; --method mean only shifts the mean. A mean shift leaves the studio photos' tighter
spread: H&M clothing piled onto one M3 trend cluster and its best candidates scored ~0.3-0.4 sigma under
the store's, so swaps leaned to shoes. CORAL brings the domain classifier from 76% to 60% and that gap
to about 0 for every type (domain_check_{mean,coral}.json in the store). The raw features are kept in
features_raw.npy and every --align run starts from them; alignment.json records the method.

u2net is the default: BiRefNet-lite cuts out a little cleaner but takes 3.5 s/image on CPU and 6.5 s on
the 8 GB GPU, against 0.11 s for u2net on the GPU.
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


def item_key(article_id: str) -> str:
    return str(int(article_id))


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
    """{type: median garment area (98,304-px frame)} over the street-photo stores' typed rows."""
    from candidate_pool import load_store_arrays, pool_dir
    meta = np.load(os.path.join(pool_dir(settings), "meta.npz"))
    _, scal = load_store_arrays(settings)
    street = ~np.isin(meta["source"], [i for i, sp in enumerate(settings.stores) if sp.product_photos])
    return {t: float(np.median(scal[street & (meta["type"] == TYPE_ID[t]), 2])) for t in TYPES}


def store_types(store_dir: str) -> np.ndarray:
    """Type id per feature-store row (-1 = not a Module 4 type)."""
    with open(os.path.join(store_dir, "index.json"), encoding="utf-8") as f:
        index = json.load(f)
    t = np.full(int(index["num_rows"]), -1, np.int64)
    for es in index["outfits"].values():
        for e in es:
            t[e["row"]] = TYPE_ID.get(e["category"], -1)
    return t


def street_type_stats(settings, chunk: int = 65536) -> tuple:
    """Mean [len(TYPES), 512] and covariance [len(TYPES), 512, 512] of each type's unit CLIP
    embeddings over the street-photo stores' typed rows."""
    total, outer, count = np.zeros((len(TYPES), 512)), np.zeros((len(TYPES), 512, 512)), np.zeros(len(TYPES))
    for spec in settings.stores:
        if spec.product_photos:
            continue
        d = settings.abs(spec.store_dir)
        types = store_types(d)
        feat = np.load(os.path.join(d, "features.npy"), mmap_mode="r")
        for a in range(0, len(types), chunk):
            t = types[a:a + chunk]
            e = np.asarray(feat[a:a + chunk, :512], np.float64)[t >= 0]
            e /= np.linalg.norm(e, axis=1, keepdims=True)
            t = t[t >= 0]
            for k in np.unique(t):
                x = e[t == k]
                count[k] += len(x)
                total[k] += x.sum(0)
                outer[k] += x.T @ x
    n = np.maximum(count, 2)
    means = total / n[:, None]
    covs = (outer - n[:, None, None] * means[:, :, None] * means[:, None, :]) / (n - 1)[:, None, None]
    return means, covs


def street_type_means(settings) -> np.ndarray:
    return street_type_stats(settings)[0]


def shift_to_means(emb: np.ndarray, types: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Shift each type's unit embeddings so their mean is means[type], then renormalise."""
    out = np.asarray(emb, np.float64) / np.linalg.norm(emb, axis=1, keepdims=True)
    for t in np.unique(types[types >= 0]):
        m = types == t
        out[m] += means[t] - out[m].mean(0)
    return (out / np.linalg.norm(out, axis=1, keepdims=True)).astype(np.float32)


def _sqrtm(c: np.ndarray, inverse: bool = False) -> np.ndarray:
    w, v = np.linalg.eigh((c + c.T) / 2)
    w = np.maximum(w, 1e-10 * max(w.max(), 1e-12))
    return (v * w ** (-0.5 if inverse else 0.5)) @ v.T


def coral_transform(emb: np.ndarray, types: np.ndarray, means: np.ndarray, covs: np.ndarray,
                    renorm: bool = True) -> np.ndarray:
    """CORAL per type: whiten each type's unit embeddings with their own (Ledoit-Wolf) covariance,
    recolour with covs[type] and centre on means[type]; then renormalise. A mean shift alone leaves
    the studio photos' tighter spread, which piles H&M clothing onto one M3 trend cluster."""
    from sklearn.covariance import LedoitWolf
    out = np.asarray(emb, np.float64) / np.linalg.norm(emb, axis=1, keepdims=True)
    for t in np.unique(types[types >= 0]):
        m = types == t
        r = out[m] - out[m].mean(0)
        a = _sqrtm(LedoitWolf(assume_centered=True).fit(r).covariance_, inverse=True) @ _sqrtm(covs[t])
        out[m] = means[t] + r @ a
    if renorm:
        out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out.astype(np.float32) if renorm else out


ALIGN_METHODS = ("mean", "coral")


def align_embeddings(emb: np.ndarray, types: np.ndarray, settings, method: str) -> np.ndarray:
    if method == "mean":
        return shift_to_means(emb, types, street_type_means(settings))
    if method == "coral":
        return coral_transform(emb, types, *street_type_stats(settings))
    raise ValueError(f"unknown alignment {method!r}; expected one of {ALIGN_METHODS}")


def align(store_dir: str, settings, method: str = "coral") -> None:
    """Rewrite features.npy with type-aligned embeddings (raw copy kept in features_raw.npy)."""
    import shutil
    path, raw_path = os.path.join(store_dir, "features.npy"), os.path.join(store_dir, "features_raw.npy")
    if not os.path.exists(raw_path):
        shutil.copyfile(path, raw_path)
    feat = np.load(raw_path)
    types = store_types(store_dir)
    feat[:, :512] = align_embeddings(feat[:, :512], types, settings, method)
    np.save(path, feat)
    with open(os.path.join(store_dir, "alignment.json"), "w", encoding="utf-8") as f:
        json.dump({"method": method}, f)
    print(f"[hm] {method}-aligned {len(feat)} rows ({(types >= 0).sum()} typed) -> {path}")


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
    d = os.path.join(out_root, item_key(row.article_id))
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
    # onnxruntime-gpu 1.23 (CUDA 12, matches torch cu126) finds its CUDA libraries through torch
    providers = (["CUDAExecutionProvider"] if dev.startswith("cuda") else []) + ["CPUExecutionProvider"]
    session = new_session(rembg_model, providers=providers)
    areas = type_areas(settings)
    out_root = os.path.join(out, "outputs")
    os.makedirs(out_root, exist_ok=True)
    cols = ["article_id", "type", "gender", "product_type_name", "prod_name", "colour_group_name",
            "index_group_name", "section_name"]
    articles[cols].assign(key=articles.article_id.map(item_key))[["key"] + cols].to_csv(os.path.join(out, "articles.csv"), index=False)
    done = skipped = 0
    for row in articles.itertuples():
        if os.path.exists(os.path.join(out_root, item_key(row.article_id), "metadata.json")):
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
    ap.add_argument("--out", help="catalogue folder to build")
    ap.add_argument("--align", metavar="STORE_DIR", help="align a built feature store's embeddings instead")
    ap.add_argument("--method", choices=ALIGN_METHODS, default="coral", help="alignment (with --align)")
    ap.add_argument("--per_type", type=int, default=None, help="sample this many per type (default: all)")
    ap.add_argument("--rembg_model", default="u2net")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    s = M4Settings.load()
    os.chdir(s.abs("."))
    if args.align:
        return align(args.align, s, args.method)
    if not args.out:
        ap.error("--out or --align is required")
    a = load_articles()
    print(f"[hm] {len(a)} adult articles with a photo and a Module 4 type")
    print(a.groupby(["type", "gender"]).size().unstack(fill_value=0).to_string())
    if args.per_type:
        a = sample(a, args.per_type)
    build(args.out, a, s, args.rembg_model, args.device)


if __name__ == "__main__":
    main()
