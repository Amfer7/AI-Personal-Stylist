"""Harmony reference (D3): raw GNN logits of every real test outfit, plus the two standardisers.

- pct(raw): harmony percentile 0-100 vs real test outfits (same formula as M2's PhotoScorer.pct,
  but over the full test set instead of 500 outfits, so it moves in fine steps).
- sigma_logit: std of those logits, the unit of ΔH.
- sigma_trend: std of garment trend scores over the eval pool, the unit of ΔT.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from candidate_pool import SPLIT_CODES
from swap_groups import TYPES


@dataclass
class HarmonyReference:
    logits_sorted: np.ndarray
    sigma_logit: float
    sigma_trend: float

    def pct(self, raw: float) -> float:
        return float(100.0 * np.searchsorted(self.logits_sorted, raw, side="right") / len(self.logits_sorted))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"logits_sorted": self.logits_sorted.tolist(), "sigma_logit": self.sigma_logit,
                       "sigma_trend": self.sigma_trend, "n_outfits": int(len(self.logits_sorted))}, f)

    @classmethod
    def load(cls, path: str) -> "HarmonyReference":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(np.asarray(d["logits_sorted"], np.float64), float(d["sigma_logit"]), float(d["sigma_trend"]))


def test_outfits(pool, min_garments: int = 2) -> list:
    test = SPLIT_CODES["test"]
    return [rows for rows in pool.outfits().values()
            if len(rows) >= min_garments and pool.split[rows[0]] == test]


test_outfits.__test__ = False   # not a pytest test despite the name


def train_outfit_types(pool):
    train = SPLIT_CODES["train"]
    for rows in pool.outfits().values():
        if pool.split[rows[0]] == train:
            yield [TYPES[t] for t in pool.type[rows] if t >= 0]


def build_reference(pool, gnn) -> HarmonyReference:
    by_size = defaultdict(list)
    for rows in test_outfits(pool):
        by_size[len(rows)].append(rows)
    parts = []
    for n, lst in sorted(by_size.items()):
        R = torch.as_tensor(np.stack(lst), device=pool.device)

        def make(a, b, R=R):
            r = R[a:b]
            return pool.proj[r], pool.hue[r], pool.formality[r], pool.area[r], pool.broad_t[r]

        parts.append(gnn.logits_chunked(make, R.shape[0], n).numpy())
    logits = np.sort(np.concatenate(parts)).astype(np.float64)
    reps = pool.rep_eval == np.arange(pool.n)
    return HarmonyReference(logits, float(logits.std()), float(np.nanstd(pool.trend[reps])))
