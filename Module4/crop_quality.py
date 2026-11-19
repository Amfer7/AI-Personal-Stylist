"""Crop quality gate for the demo pool: suggestions are shown as images, so they must be readable.

Module 1's cut-outs of catalogue photos are not always clean: Fashion144k crops can be tiny and
fragmentary, and Pinterest collages, flat-lays and product shots next to a model merge several
garments into one mask, sometimes with the pin's caption text. Each crop gets shape metrics from its
alpha mask and two CLIP scores from its stored embedding (computed once, cached in
<cache_dir>/crop_metrics.npz, pool row order):

    area   opaque pixels (catalogue photos share the 98,304-px frame)
    main   share of the opaque pixels in the largest connected piece
    top2   share in the two largest pieces (pairs: two shoes, two trouser legs)
    fill   largest piece / its bounding box (low = specks, strands, holes)
    cover  opaque pixels / the crop canvas (low = pieces far apart, e.g. two shoes in opposite corners)
    holes  enclosed gaps / the filled outline (high = garment hollowed out by an arm, hair or a bag)
    label_p  CLIP probability of the crop's own swap group among the group prompts (low = mislabelled,
             e.g. a sneaker labelled "top")
    text_p   CLIP probability of the "text" prompts among all prompts (high = caption overlay)

The gate (settings.crop_gate) keeps a crop if area >= min_area for its type, main (top2 for
pair_types) >= min_main, fill >= min_fill, cover >= min_cover, holes <= max_holes (not for
pair_types: strappy sandals and two touching boots have real gaps), label_p >= min_label_p for its
type and text_p <= max_text_p. The label check is for clothing: CLIP reads dark boots, side-on bags
and knit scarves as other groups, so accessories set it to 0.

It only limits the demo pool (rep_demo); the evaluation pool (rep_eval) is untouched, so calibrated
results stay valid.
"""
from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from scipy import ndimage

from swap_groups import SWAP_GROUPS, TYPES

SHAPE_METRICS = ("area", "main", "top2", "fill", "cover", "holes")
CLIP_METRICS = ("label_p", "text_p")
METRICS = SHAPE_METRICS + CLIP_METRICS
GROUP_PROMPTS = {                  # several prompts per group; a group's probability is their sum
    "upper": ("a t-shirt", "a sweater", "a cardigan", "a shirt", "a blouse", "a jacket", "a coat", "a crop top",
              "a tank top"),
    "lower": ("pants", "jeans", "trousers", "a skirt", "shorts", "leggings"),
    "dress": ("a dress", "a gown"),
    "shoe": ("shoes", "boots", "sneakers", "sandals", "high heels"),
    "bag": ("a bag", "a handbag", "a backpack"),
    "hat": ("a hat", "a cap", "a beanie"),
    "belt": ("a belt",),
    "scarf": ("a scarf",),
    "sunglasses": ("sunglasses",),
}
TEXT_PROMPTS = ("text", "a screenshot of words and letters", "a caption written on an image")
PROMPT_TEMPLATE = "a photo of {}"
CLIP_LOGIT_SCALE = 100.0           # ViT-B/32's learned logit scale
_EIGHT = np.ones((3, 3), bool)


def mask_metrics(alpha: np.ndarray) -> tuple:
    """(area, main, top2, fill, cover, holes) of a boolean mask. An empty mask gives area 0, NaN rest."""
    total = int(alpha.sum())
    if total == 0:
        return (0.0,) + (np.nan,) * (len(SHAPE_METRICS) - 1)
    labels, _ = ndimage.label(alpha, structure=_EIGHT)
    sizes = np.bincount(labels.ravel())[1:]
    order = np.argsort(sizes)[::-1]
    ys, xs = np.nonzero(labels == order[0] + 1)
    box = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    filled = int(ndimage.binary_fill_holes(alpha).sum())
    return (float(total), sizes[order[0]] / total, sizes[order[:2]].sum() / total, len(ys) / box,
            total / alpha.size, (filled - total) / filled)


def crop_metrics(path: str) -> tuple:
    """mask_metrics of an RGBA crop's alpha channel; all NaN if the file is missing or unreadable."""
    try:
        with Image.open(path) as im:
            alpha = np.asarray(im.convert("RGBA").getchannel("A")) > 0
    except Exception:
        return (np.nan,) * len(SHAPE_METRICS)
    return mask_metrics(alpha)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def prompts() -> tuple:
    """(prompt texts, group index per prompt: -1 = text prompt). Order: GROUP_PROMPTS, then TEXT_PROMPTS."""
    texts, group = [], []
    for gi, ps in enumerate(GROUP_PROMPTS.values()):
        texts += [PROMPT_TEMPLATE.format(p) for p in ps]
        group += [gi] * len(ps)
    texts += [PROMPT_TEMPLATE.format(p) for p in TEXT_PROMPTS]
    return texts, np.array(group + [-1] * len(TEXT_PROMPTS))


def prompts_key() -> str:
    """Changes whenever a prompt changes, so cached CLIP scores are recomputed."""
    return hashlib.sha256("\n".join(prompts()[0]).encode()).hexdigest()[:16]


def clip_scores(emb: np.ndarray, type_ids: np.ndarray, text_emb: np.ndarray, prompt_group: np.ndarray,
                chunk: int = 65536) -> dict:
    """label_p and text_p per row. emb: [N, 512] image embeddings; text_emb: [P, 512] unit prompt
    embeddings; prompt_group: [P] group index (order of GROUP_PROMPTS), -1 for text prompts.
    label_p: own group's share among the group prompts; text_p: text prompts' share among all.
    Unknown types get label_p NaN."""
    groups = list(GROUP_PROMPTS)
    group_of = np.array([groups.index(next(g for g, ts in SWAP_GROUPS.items() if t in ts)) for t in TYPES])
    prompt_group = np.asarray(prompt_group)
    is_text = prompt_group < 0
    onehot = (prompt_group[~is_text, None] == np.arange(len(groups))[None, :]).astype(np.float32)   # [P_g, G]
    type_ids = np.asarray(type_ids)
    label_p = np.full(len(emb), np.nan, np.float32)
    text_p = np.empty(len(emb), np.float32)
    for s in range(0, len(emb), chunk):
        e = np.asarray(emb[s:s + chunk], np.float32)
        e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)
        logits = CLIP_LOGIT_SCALE * e @ text_emb.T
        text_p[s:s + chunk] = _softmax(logits)[:, is_text].sum(1)
        pg = _softmax(logits[:, ~is_text]) @ onehot
        t = type_ids[s:s + chunk]
        known = t >= 0
        label_p[s:s + chunk][known] = pg[np.nonzero(known)[0], group_of[t[known]]]
    return {"label_p": label_p, "text_p": text_p}


def encode_prompts(device: str = "cpu") -> np.ndarray:
    """Unit CLIP ViT-B/32 text embeddings of prompts() (Module 1's model)."""
    import clip
    import torch
    model, _ = clip.load("ViT-B/32", device=device)
    with torch.no_grad():
        t = model.encode_text(clip.tokenize(prompts()[0]).to(device)).float()
    return (t / t.norm(dim=1, keepdim=True)).cpu().numpy()


def per_type(bars: dict) -> np.ndarray:
    """{"default": x, type: y, ...} -> one bar per TYPES entry."""
    return np.array([bars.get(t, bars["default"]) for t in TYPES], np.float64)


def crop_ok(metrics: dict, type_ids: np.ndarray, gate: dict) -> np.ndarray:
    """bool [N]: which crops pass the gate. Unknown types and NaN metrics fail."""
    type_ids = np.asarray(type_ids)
    known = type_ids >= 0
    min_area, min_label = per_type(gate["min_area"]), per_type(gate["min_label_p"])
    pair = np.isin(np.arange(len(TYPES)), [TYPES.index(t) for t in gate["pair_types"]])
    t = np.where(known, type_ids, 0)
    share = np.where(pair[t], metrics["top2"], metrics["main"])
    with np.errstate(invalid="ignore"):
        ok = ((metrics["area"] >= min_area[t]) & (share >= gate["min_main"]) & (metrics["fill"] >= gate["min_fill"])
              & (metrics["cover"] >= gate["min_cover"]) & ((metrics["holes"] <= gate["max_holes"]) | pair[t])
              & (metrics["label_p"] >= min_label[t]) & (metrics["text_p"] <= gate["max_text_p"]))
    return ok & known


def metrics_path(settings) -> str:
    return settings.abs(os.path.join(settings.cache_dir, "crop_metrics.npz"))


def _pool_key(pool_dir: str) -> str:
    """Ties the metrics to one pool build: hash of its files.json (row order and file names)."""
    with open(os.path.join(pool_dir, "files.json"), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def metrics_fresh(settings, pool_dir: str) -> bool:
    p = metrics_path(settings)
    if not os.path.exists(p):
        return False
    with np.load(p) as z:
        return (set(METRICS) <= set(z.files) and str(z["pool_key"]) == _pool_key(pool_dir)
                and "prompts_key" in z.files and str(z["prompts_key"]) == prompts_key())


def load_metrics(settings, pool_dir: str) -> dict:
    if not metrics_fresh(settings, pool_dir):
        raise RuntimeError("Module 4 crop metrics are missing or stale: run  python Module4/build_cache.py")
    with np.load(metrics_path(settings)) as z:
        return {k: z[k] for k in METRICS}


def build_metrics(settings, pool_dir: str, workers: int = 12, chunk: int = 2048, device: str = "cpu") -> dict:
    """Compute whatever crop metrics are missing for this pool and write crop_metrics.npz. The shape
    scan reads every crop (~634k: about 8 minutes); the CLIP scores take seconds."""
    from candidate_pool import load_store_arrays
    p, key = metrics_path(settings), _pool_key(pool_dir)
    out = {}
    if os.path.exists(p):
        with np.load(p) as z:
            if str(z["pool_key"]) == key:
                fresh_clip = "prompts_key" in z.files and str(z["prompts_key"]) == prompts_key()
                out = {k: z[k] for k in SHAPE_METRICS + (CLIP_METRICS if fresh_clip else ()) if k in z.files}
    meta = np.load(os.path.join(pool_dir, "meta.npz"))
    if not set(SHAPE_METRICS) <= set(out):
        with open(os.path.join(pool_dir, "files.json"), encoding="utf-8") as f:
            files = json.load(f)
        with open(os.path.join(pool_dir, "outfit_keys.json"), encoding="utf-8") as f:
            keys = json.load(f)
        roots = [settings.abs(s.output_root) for s in settings.stores]
        names = [s.name for s in settings.stores]
        paths = [os.path.join(roots[s], keys[names[s]][o], fn)
                 for s, o, fn in zip(meta["source"], meta["outfit_ord"], files)]
        with ProcessPoolExecutor(workers) as ex:
            arr = np.asarray(list(ex.map(crop_metrics, paths, chunksize=chunk)), np.float32)
        out.update({k: arr[:, i] for i, k in enumerate(SHAPE_METRICS)})
    if not set(CLIP_METRICS) <= set(out):
        emb, _ = load_store_arrays(settings)
        out.update(clip_scores(emb, meta["type"], encode_prompts(device), prompts()[1]))
    np.savez(p, pool_key=key, prompts_key=prompts_key(), **out)
    return out
