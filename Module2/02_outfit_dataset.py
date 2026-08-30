"""
02_outfit_dataset.py
=====================
Stage 3-5 of the plan doc: turns outputs/<idx>/{metadata.json,*.npy,*.png} +
the Fashion144K .mat files into per-outfit graphs ready for the GNN.

- load_split / load_relvotes / load_photos_list: read the .mat / text files directly.
- compute_colour_histogram: 32-D HSV histogram computed fresh from each garment
  crop's pixels (replaces col_cco.mat, per the plan doc).
- CATEGORY_EDGE_WEIGHTS: the full top/bottom/outerwear/dress/shoe/bag/accessory
  pairwise weight table (fills in the gap in Prateek's handoff-guide snippet,
  which only special-cased top/bottom).
- build_category_pool: one-time scan of all outputs/ to index (outfit_idx, filename)
  by category, needed for BPR negative sampling later.
- OutfitDataset: a torch.utils.data.Dataset that lazily builds one graph per outfit.

Usage as a library (imported by 03_train_gnn.py), or standalone to sanity-check:
    python 02_outfit_dataset.py --output_root .../outputs --split_mat split.mat \
        --relvotes_mat relvotes.mat --photos_list photos --check_index 0
"""

import os
import json
import argparse
import pickle
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image
import scipy.io as sio
import torch
from torch.utils.data import Dataset


# ---------------- .mat / text loaders ----------------

def load_split(split_mat_path):
    s = sio.loadmat(split_mat_path)
    return {
        "train": s["trainids"][0].astype(int),
        "val": s["validids"][0].astype(int),
        "test": s["testids"][0].astype(int),
    }


def load_relvotes(relvotes_mat_path):
    r = sio.loadmat(relvotes_mat_path)
    return r["X"][0].astype(np.float32)  # shape (144169,), 1-10 scale


def load_photos_list(photos_txt_path):
    with open(photos_txt_path, "r", encoding="utf-8", errors="replace") as f:
        return [line.strip() for line in f if line.strip()]


# ---------------- colour histogram (replaces col_cco.mat) ----------------

def compute_colour_histogram(png_path, hue_bins=16, sat_bins=8, val_bins=8):
    """32-D histogram (16 hue + 8 sat + 8 val bins), normalized to sum to 1
    per channel-block. Composited onto white background first, matching the
    CLIP embedding step, so colour features and CLIP features see the same
    pixels."""
    img = Image.open(png_path)
    if img.mode == "RGBA":
        white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white_bg, img).convert("RGB")
    else:
        img = img.convert("RGB")

    arr = np.array(img)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    h_hist, _ = np.histogram(hsv[:, :, 0], bins=hue_bins, range=(0, 180))
    s_hist, _ = np.histogram(hsv[:, :, 1], bins=sat_bins, range=(0, 256))
    v_hist, _ = np.histogram(hsv[:, :, 2], bins=val_bins, range=(0, 256))

    def norm(h):
        h = h.astype(np.float32)
        total = h.sum()
        return h / total if total > 0 else h

    return np.concatenate([norm(h_hist), norm(s_hist), norm(v_hist)]).astype(np.float32)


# ---------------- category edge-weight table ----------------
# broad_category values expected from metadata.json: top, bottom, outerwear,
# dress, shoe, bag, accessory (see category_mapping.py's FASHION_LABEL_TO_BROAD_CATEGORY)

CATEGORY_EDGE_WEIGHTS = {
    frozenset({"top", "bottom"}): 1.0,
    frozenset({"outerwear", "bottom"}): 0.8,
    frozenset({"bottom", "shoe"}): 0.8,
    frozenset({"dress", "shoe"}): 0.9,
    frozenset({"top", "outerwear"}): 0.6,
    frozenset({"top", "shoe"}): 0.5,
    frozenset({"dress", "bag"}): 0.5,
    frozenset({"top", "accessory"}): 0.4,
    frozenset({"bottom", "accessory"}): 0.4,
    frozenset({"dress", "accessory"}): 0.4,
    frozenset({"outerwear", "shoe"}): 0.4,
    frozenset({"top", "bag"}): 0.35,
    frozenset({"bottom", "bag"}): 0.35,
    frozenset({"shoe", "accessory"}): 0.3,
    frozenset({"shoe", "bag"}): 0.3,
    frozenset({"bag", "accessory"}): 0.25,
}
DEFAULT_EDGE_WEIGHT = 0.3


def get_edge_weight(cat_a, cat_b):
    if cat_a == cat_b:
        return 0.2  # same-category pairs (rare: e.g. two accessories) get a low weight
    key = frozenset({cat_a, cat_b})
    return CATEGORY_EDGE_WEIGHTS.get(key, DEFAULT_EDGE_WEIGHT)


# ---------------- Module 1 attribute encoding ----------------
# Encodes the `attributes` block written by Module 1 (fashion_segmenter) into fixed-length
# node features, and the PDF's fashion-theory pairwise cues into edge features.

SILHOUETTES = ["fitted", "boxy", "a_line", "tapered"]
PATTERNS = ["solid", "striped", "checked", "floral", "graphic", "dotted"]
FABRICS = ["denim", "cotton", "leather", "knit", "silk", "corduroy"]
FITS = ["slim", "regular", "oversized"]

# 19 one-hot + 7 scalar (formality, hue sin/cos, sat, val, extent, aspect) = 26
NODE_ATTR_DIM = len(SILHOUETTES) + len(PATTERNS) + len(FABRICS) + len(FITS) + 7
# [category weight, hue separation, formality difference, volume contrast, same-category]
EDGE_ATTR_DIM = 5


def _one_hot(value, vocab):
    v = np.zeros(len(vocab), dtype=np.float32)
    if value in vocab:
        v[vocab.index(value)] = 1.0
    return v


def _hue_dist(h1, h2):
    """Circular hue distance in degrees, range [0, 180]."""
    d = abs(h1 - h2) % 360.0
    return d if d <= 180.0 else 360.0 - d


def encode_node_attributes(attr):
    """Module-1 `attributes` dict -> fixed NODE_ATTR_DIM vector. Missing/unknown fields
    become zeros, so older outputs (without the attributes block) degrade gracefully."""
    attr = attr or {}
    color = attr.get("color") or {}
    formality = attr.get("formality_score")
    hue = color.get("hue")
    sat = color.get("saturation")
    val = color.get("value")
    extent = attr.get("extent")
    aspect = attr.get("aspect_ratio")

    scalars = np.array([
        (formality / 100.0) if formality is not None else 0.0,
        np.sin(np.deg2rad(hue)) if hue is not None else 0.0,
        np.cos(np.deg2rad(hue)) if hue is not None else 0.0,
        sat if sat is not None else 0.0,
        val if val is not None else 0.0,
        extent if extent is not None else 0.0,
        min(aspect / 4.0, 1.0) if aspect is not None else 0.0,
    ], dtype=np.float32)

    return np.concatenate([
        _one_hot(attr.get("silhouette"), SILHOUETTES),
        _one_hot(attr.get("pattern"), PATTERNS),
        _one_hot(attr.get("fabric"), FABRICS),
        _one_hot(attr.get("fit"), FITS),
        scalars,
    ]).astype(np.float32)


def attr_subset_indices():
    """Indices into the NODE_ATTR_DIM vector to KEEP for the `--attr_subset` ablation.

    Keeps the *reliable* signals -- the pixel-derived silhouette one-hot and the 7 scalars
    (formality + colour hue/sat/val + silhouette extent/aspect) -- and drops the noisiest,
    CLIP-zero-shot categorical one-hots (pattern / fabric / fit). Tests whether the clean
    attributes help the compatibility score even though the full 26-D set hurts it."""
    keep = list(range(len(SILHOUETTES)))                       # silhouette one-hot: 0..3
    scalar_start = len(SILHOUETTES) + len(PATTERNS) + len(FABRICS) + len(FITS)
    keep += list(range(scalar_start, scalar_start + 7))        # the 7 scalars: 19..25
    return keep


def build_node_from_meta(outfit_dir, g):
    """Builds a single node dict (embedding + colour hist + attribute features + the raw
    scalars needed for edge features) from one garment's metadata entry."""
    png_path = os.path.join(outfit_dir, g["file"])
    npy_path = os.path.splitext(png_path)[0] + ".npy"
    attr = g.get("attributes", {}) or {}
    color = attr.get("color") or {}
    return {
        "embedding": np.load(npy_path).astype(np.float32),
        "colour": compute_colour_histogram(png_path),
        "attr_feat": encode_node_attributes(attr),
        "category": g["broad_category"],
        "hue": color.get("hue"),
        "formality": (attr.get("formality_score") if attr.get("formality_score") is not None else 50.0),
        "area": float(g.get("pixel_count", attr.get("area_px", 0) or 0)),
    }


def edge_features(a, b):
    """Directed edge features from node a (src) to node b (dst): the fashion-theory cues
    named in the PDF (hue separation, formality difference, volume contrast) plus the
    category weight prior and a same-category flag."""
    w = get_edge_weight(a["category"], b["category"])
    if a["hue"] is not None and b["hue"] is not None:
        hue_sep = _hue_dist(a["hue"], b["hue"]) / 180.0
    else:
        hue_sep = 0.0
    form_diff = abs(a["formality"] - b["formality"]) / 100.0
    volume_contrast = float(np.tanh(np.log((a["area"] + 1.0) / (b["area"] + 1.0))))
    same_cat = 1.0 if a["category"] == b["category"] else 0.0
    return np.array([w, hue_sep, form_diff, volume_contrast, same_cat], dtype=np.float32)


# ---------------- category pool for negative sampling ----------------

def build_category_pool(outfit_indices, output_root, cache_path=None):
    """Scans outputs/<idx>/metadata.json for every idx in outfit_indices and
    returns {broad_category: [(outfit_idx, filename), ...]}. Cache to disk
    since this is a full scan over up to 144K folders."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    pool = defaultdict(list)
    for idx in outfit_indices:
        meta_path = os.path.join(output_root, str(idx), "metadata.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, "r") as f:
            garments = json.load(f)
        for g in garments:
            pool[g["broad_category"]].append((idx, g["file"]))

    pool = dict(pool)
    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump(pool, f)
    return pool


# ---------------- the dataset ----------------

class OutfitDataset(Dataset):
    """One item = one outfit's graph. Only includes outfits that actually have
    metadata.json + >=2 garments (need at least 2 nodes for a meaningful graph)."""

    def __init__(self, outfit_indices, output_root, relvotes, min_garments=2, feature_store=None):
        self.output_root = output_root
        self.relvotes = relvotes
        self.min_garments = min_garments

        # Tier 1: optional memmap feature store (built by build_feature_store.py).
        self.store = None
        if feature_store:
            self._load_store(feature_store)

        self.valid_indices = self._filter_valid(outfit_indices)

    def _load_store(self, store_dir):
        with open(os.path.join(store_dir, "index.json"), "r") as f:
            index = json.load(f)
        self.store = {
            "feat": np.load(os.path.join(store_dir, "features.npy"), mmap_mode="r"),
            "scal": np.load(os.path.join(store_dir, "node_scalars.npy"), mmap_mode="r"),
            "outfits": index["outfits"],
            "emb": index["emb_dim"], "col": index["colour_dim"],
        }
        # (outfit_idx, filename) -> row, for O(1) negative-sample lookups
        self.store["row_of"] = {
            (int(idx), e["file"]): e["row"]
            for idx, entries in index["outfits"].items() for e in entries
        }

    def _filter_valid(self, outfit_indices):
        valid = []
        for idx in outfit_indices:
            if self.store is not None:
                entries = self.store["outfits"].get(str(idx))
                if entries is not None and len(entries) >= self.min_garments:
                    valid.append(idx)
                continue
            meta_path = os.path.join(self.output_root, str(idx), "metadata.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, "r") as f:
                garments = json.load(f)
            if len(garments) >= self.min_garments:
                valid.append(idx)
        return valid

    def _node_from_store_row(self, row, broad_category):
        """Builds a node dict from a memmap row (no disk reads of PNG/npy)."""
        feat, scal, e, c = self.store["feat"], self.store["scal"], self.store["emb"], self.store["col"]
        hue = float(scal[row, 0])
        return {
            "embedding": np.asarray(feat[row, :e], dtype=np.float32),
            "colour": np.asarray(feat[row, e:e + c], dtype=np.float32),
            "attr_feat": np.asarray(feat[row, e + c:], dtype=np.float32),
            "category": broad_category,
            "hue": None if np.isnan(hue) else hue,
            "formality": float(scal[row, 1]),
            "area": float(scal[row, 2]),
        }

    def build_node(self, outfit_idx, filename, broad_category=None):
        """Builds a single node from the store (fast) or disk (fallback). Used by
        negative sampling to pull a substitute garment."""
        if self.store is not None:
            row = self.store["row_of"][(int(outfit_idx), filename)]
            return self._node_from_store_row(row, broad_category)
        # disk fallback: read the substitute outfit's metadata for that file
        d = os.path.join(self.output_root, str(outfit_idx))
        g = next(gg for gg in json.load(open(os.path.join(d, "metadata.json")))
                 if gg["file"] == filename)
        return build_node_from_meta(d, g)

    def build_pool(self, indices):
        """broad_category -> [(outfit_idx, filename)] over the given indices (store-backed)."""
        pool = defaultdict(list)
        for idx in indices:
            for e in self.store["outfits"][str(idx)]:
                pool[e["broad_category"]].append((idx, e["file"]))
        return dict(pool)

    def __len__(self):
        return len(self.valid_indices)

    def load_outfit_nodes(self, idx):
        """Returns a list of node dicts (embedding + colour hist + attribute features +
        raw scalars for edge features) for outfit idx -- reused by negative sampling.
        Reads from the memmap store when available, else from disk."""
        if self.store is not None:
            return [self._node_from_store_row(e["row"], e["broad_category"])
                    for e in self.store["outfits"][str(idx)]]
        outfit_dir = os.path.join(self.output_root, str(idx))
        with open(os.path.join(outfit_dir, "metadata.json"), "r") as f:
            garments = json.load(f)
        return [build_node_from_meta(outfit_dir, g) for g in garments]

    @staticmethod
    def build_graph(nodes, fashion_score, use_attributes=True, attr_mask=None):
        """Builds a graph from a list of node dicts.

        use_attributes=True  (default): node x_i = [512 CLIP | 32 colour | 26 attrs] (570),
                                        edge e_ij = 5-D fashion-theory cues.
        use_attributes=False (ablation): node x_i = [512 CLIP | 32 colour] (544),
                                        edge e_ij = [category weight] (1-D) -- the original
                                        baseline before the attribute integration.
        attr_mask (list of int, optional): when use_attributes, keep only these columns of
                                        the 26-D attr vector (the `--attr_subset` ablation).
                                        Edges stay 5-D so only the node attrs are ablated."""
        N = len(nodes)
        if use_attributes:
            if attr_mask is not None:
                feats = [np.concatenate([n["embedding"], n["colour"], n["attr_feat"][attr_mask]])
                         for n in nodes]
            else:
                feats = [np.concatenate([n["embedding"], n["colour"], n["attr_feat"]]) for n in nodes]
        else:
            feats = [np.concatenate([n["embedding"], n["colour"]]) for n in nodes]
        x = torch.tensor(np.stack(feats), dtype=torch.float32)

        edge_dim = EDGE_ATTR_DIM if use_attributes else 1
        src, dst, e_attr = [], [], []
        for i in range(N):
            for j in range(N):
                if i != j:
                    if use_attributes:
                        ef = edge_features(nodes[i], nodes[j])
                    else:
                        ef = np.array([get_edge_weight(nodes[i]["category"], nodes[j]["category"])],
                                      dtype=np.float32)
                    src.append(i)
                    dst.append(j)
                    e_attr.append(ef)

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = (torch.tensor(np.array(e_attr), dtype=torch.float32)
                     if e_attr else torch.zeros((0, edge_dim), dtype=torch.float32))

        return {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "num_nodes": N,
            "fashion_score": torch.tensor(fashion_score, dtype=torch.float32),
        }

    @staticmethod
    def collate(graphs):
        """Packs a list of single-outfit graphs into one disconnected batched graph
        (Tier 2). Node features are concatenated, edge indices offset per outfit, and a
        `batch` vector maps each node to its outfit id for segment-wise readout."""
        xs, eis, eas, batch, fs = [], [], [], [], []
        node_offset = 0
        for bi, g in enumerate(graphs):
            n = g["num_nodes"]
            xs.append(g["x"])
            eis.append(g["edge_index"] + node_offset)
            eas.append(g["edge_attr"])
            batch.append(torch.full((n,), bi, dtype=torch.long))
            fs.append(g["fashion_score"])
            node_offset += n
        return {
            "x": torch.cat(xs, dim=0),
            "edge_index": torch.cat(eis, dim=1),
            "edge_attr": torch.cat(eas, dim=0),
            "batch": torch.cat(batch, dim=0),
            "num_nodes": node_offset,
            "fashion_score": torch.stack(fs),
        }

    def __getitem__(self, i):
        idx = self.valid_indices[i]
        nodes = self.load_outfit_nodes(idx)
        graph = self.build_graph(nodes, float(self.relvotes[idx]))
        graph["outfit_idx"] = idx
        graph["categories"] = [n["category"] for n in nodes]
        return graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--split_mat", required=True)
    parser.add_argument("--relvotes_mat", required=True)
    parser.add_argument("--check_index", type=int, default=0)
    args = parser.parse_args()

    split = load_split(args.split_mat)
    relvotes = load_relvotes(args.relvotes_mat)
    ds = OutfitDataset(split["train"][:50], args.output_root, relvotes)
    print(f"[CHECK] {len(ds)} valid outfits out of first 50 train indices scanned")
    print(f"[CHECK] node dim = {512 + 32 + NODE_ATTR_DIM} "
          f"(512 CLIP + 32 colour + {NODE_ATTR_DIM} attrs), edge dim = {EDGE_ATTR_DIM}")
    if len(ds) > 0:
        item = ds[args.check_index]
        print(f"[CHECK] outfit_idx={item['outfit_idx']} x.shape={item['x'].shape} "
              f"edge_index.shape={item['edge_index'].shape} "
              f"edge_attr.shape={item['edge_attr'].shape} "
              f"fashion_score={item['fashion_score'].item():.2f} "
              f"categories={item['categories']}")
