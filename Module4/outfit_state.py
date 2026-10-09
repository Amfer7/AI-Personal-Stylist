"""The outfit being recommended for: an immutable tuple of pieces.

A Piece carries everything scoring needs (projected GNN input, CLIP embedding, edge scalars, trend).
Pieces come from an M1 outfit folder (a user photo) or from pool rows (evaluation). OutfitState is
immutable and has with_swap(), so a later greedy multi-piece search can chain swaps without
changing the recommender.

Area normalisation: M1 keeps raw pixel counts, and the GNN's volume-contrast edge compares them
across pieces. Catalogue crops come from 256x384 photos, so a user photo's areas are rescaled to
that 98,304-px frame before they meet catalogue candidates.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from PIL import Image

from m2_bridge import BROAD_ID, BROAD_VOCAB, node_features, od
from swap_groups import TYPES, swap_group

REFERENCE_FRAME_PX = 256 * 384


class NoGarmentsDetected(ValueError):
    pass


@dataclass(frozen=True, eq=False)
class Piece:
    file: str
    type: str
    broad: str
    proj: torch.Tensor
    emb: np.ndarray
    hue: float
    formality: float
    area: float
    trend: float
    cluster: int
    pool_row: int | None = None

    @property
    def group(self):
        return swap_group(self.type)

    def describe(self) -> dict:
        return {"type": self.type, "hue": self.hue, "formality": self.formality, "area": self.area}


@dataclass(frozen=True, eq=False)
class OutfitState:
    pieces: tuple
    outfit_dir: str | None = None
    source_key: tuple | None = None
    flags: tuple = field(default_factory=tuple)

    def with_swap(self, i: int, piece: Piece) -> "OutfitState":
        ps = list(self.pieces)
        ps[i] = piece
        return replace(self, pieces=tuple(ps))


def photo_pixels(outfit_dir: str):
    """Input photo size from M1's <id>_vis.jpg overlay (drawn on the full input image)."""
    for f in sorted(os.listdir(outfit_dir)):
        if f.endswith("_vis.jpg"):
            with Image.open(os.path.join(outfit_dir, f)) as im:
                return im.size[0] * im.size[1]
    return None


def load_outfit_dir(outfit_dir: str, gnn, trend_model, photo_px=None) -> OutfitState:
    meta_path = os.path.join(outfit_dir, "metadata.json")
    garments = []
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            garments = json.load(f)
    flags = []
    photo_px = photo_px or photo_pixels(outfit_dir)
    if photo_px:
        scale = REFERENCE_FRAME_PX / photo_px
    else:
        scale = 1.0
        flags.append("photo size unknown: garment areas not normalised")
    pieces = []
    for g in garments:
        npy = os.path.splitext(os.path.join(outfit_dir, g["file"]))[0] + ".npy"
        if not os.path.exists(npy):
            flags.append(f"{g['file']}: no embedding, skipped")
            continue
        node = od.build_node_from_meta(outfit_dir, g)
        if node["category"] not in BROAD_ID:
            raise ValueError(f"{g['file']}: unknown broad category {node['category']!r}")
        emb = node["embedding"] / (np.linalg.norm(node["embedding"]) + 1e-8)
        t = trend_model.score_embedding(node["embedding"], node["category"])
        pieces.append(Piece(
            file=g["file"], type=g.get("category", "unknown"), broad=node["category"],
            proj=gnn.project(torch.from_numpy(node_features(node))[None])[0],
            emb=emb.astype(np.float32),
            hue=np.nan if node["hue"] is None else float(node["hue"]),
            formality=float(node["formality"]), area=float(node["area"]) * scale,
            trend=np.nan if t["trendiness_score"] is None else float(t["trendiness_score"]),
            cluster=-1 if t.get("cluster_id") is None else int(t["cluster_id"])))
    if not pieces:
        raise NoGarmentsDetected("no garments detected")
    return OutfitState(tuple(pieces), outfit_dir=outfit_dir, flags=tuple(flags))


def piece_from_pool(pool, row: int) -> Piece:
    t = int(pool.type[row])
    return Piece(file=pool.files[row], type=TYPES[t] if t >= 0 else "unknown",
                 broad=BROAD_VOCAB[int(pool.broad[row])], proj=pool.proj[row],
                 emb=pool.emb[row].cpu().numpy(), hue=float(pool.hue[row]),
                 formality=float(pool.formality[row]), area=float(pool.area[row]),
                 trend=float(pool.trend[row]), cluster=int(pool.cluster[row]), pool_row=int(row))


def state_from_pool(pool, source: str, key: str) -> OutfitState:
    rows = pool.outfit_rows(source, key)
    return OutfitState(tuple(piece_from_pool(pool, int(r)) for r in rows), source_key=(source, key))
