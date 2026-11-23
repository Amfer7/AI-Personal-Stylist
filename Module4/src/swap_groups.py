"""Swap groups and the co-occurrence filter (D4): what may replace what.

Groups follow M1's segmentation slots: M1 keeps one garment per SegFormer region, so an outfit has
at most one upper-body piece, and each accessory type appears at most once. The filter is learned
from train outfits: a candidate type is allowed only if it co-occurs with every remaining piece's
type in at least `min_count` outfits. A type already worn by another piece has count 0 (no outfit
wears two of one type), so it is blocked too.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

SWAP_GROUPS = {
    "upper": ("t_shirt", "sweater", "shirt_blouse", "jacket", "top"),
    "lower": ("pants", "leggings", "shorts", "skirt"),
    "dress": ("dress",),
    "shoe": ("shoe",),
    "bag": ("bag",),
    "hat": ("hat",),
    "belt": ("belt",),
    "scarf": ("scarf",),
    "sunglasses": ("sunglasses",),
}
TYPES = tuple(t for ts in SWAP_GROUPS.values() for t in ts)
TYPE_ID = {t: i for i, t in enumerate(TYPES)}
_GROUP_OF = {t: g for g, ts in SWAP_GROUPS.items() for t in ts}

# M1's specific -> broad mapping (fashion_segmenter.py, SegFormer path).
BROAD_OF_TYPE = {
    **{t: "top" for t in SWAP_GROUPS["upper"]},
    **{t: "bottom" for t in SWAP_GROUPS["lower"]},
    "dress": "dress",
    **{t: "accessory" for t in ("shoe", "bag", "hat", "belt", "scarf", "sunglasses")},
}


def swap_group(type_: str):
    return _GROUP_OF.get(type_)


def replacement_types(type_: str, same_type: bool = False) -> tuple:
    g = swap_group(type_)
    if g is None:
        return ()
    return (type_,) if same_type else SWAP_GROUPS[g]


class CooccurrenceFilter:
    def __init__(self, counts, min_count: int = 20):
        self.counts = np.asarray(counts, dtype=np.int64)
        self.min_count = int(min_count)

    @classmethod
    def from_outfits(cls, outfits: Iterable[list], min_count: int = 20) -> "CooccurrenceFilter":
        """outfits: one list of specific types per (train) outfit. counts[a, b] counts pairs of
        distinct pieces, i.e. outfits containing both a and b (no outfit repeats a type)."""
        c = np.zeros((len(TYPES), len(TYPES)), dtype=np.int64)
        for types in outfits:
            ids = [TYPE_ID[t] for t in types if t in TYPE_ID]
            for x in range(len(ids)):
                for y in range(x + 1, len(ids)):
                    c[ids[x], ids[y]] += 1
                    c[ids[y], ids[x]] += 1
        return cls(c, min_count)

    def allowed(self, cand_type: str, other_types) -> bool:
        a = TYPE_ID[cand_type]
        return all(self.counts[a, TYPE_ID[t]] >= self.min_count for t in other_types if t in TYPE_ID)

    def save(self, path: str) -> None:
        np.save(path, self.counts)

    @classmethod
    def load(cls, path: str, min_count: int = 20) -> "CooccurrenceFilter":
        return cls(np.load(path), min_count)
