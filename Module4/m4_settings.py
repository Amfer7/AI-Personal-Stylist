"""Module 4 settings: every path and tunable knob, read from m4_settings.json.

Paths are repo-relative and resolved with M4Settings.abs(), so no machine-specific path is ever
stored. Knobs that the evaluation step calibrates (w_t, rule, tau, eps, t_dup) carry a status;
until it reads "calibrated" the value is provisional and every Recommendation says so.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SETTINGS_PATH = os.path.join(HERE, "m4_settings.json")


@dataclass(frozen=True)
class StoreSpec:
    name: str          # "fashion144k" | "pinterest" | "hm"
    store_dir: str     # feature store (features.npy, node_scalars.npy, index.json)
    output_root: str   # holds <outfit_key>/<file>.png crops
    split: str         # split.mat (Fashion144k) or {outfit_key: "train"|"val"|"test"} JSON
    queries: str | None = None   # optional CSV (key, query): search queries behind each outfit photo
    product_photos: bool = False  # studio product cut-outs, one garment per "outfit" (hm_catalogue.py)


@dataclass(frozen=True)
class M4Settings:
    stores: list
    checkpoint: str
    trend_model: str
    cache_dir: str
    w_h: float
    w_t: float
    rule: str
    topk: int
    tau: dict
    eps: float
    t_dup: float
    dup_max_colour_l1: float   # near-duplicate also needs matching colour histograms (L1 of 3 x hist, max 6)
    dup_max_area_ratio: float  # ... and crop areas within this ratio
    crop_gate: dict            # demo-pool crop quality gate (crop_quality.py); not used by the evaluation
    demo_exclude_queries: list  # outfits found by these search queries never become demo suggestions
    demo_sources: list         # stores whose garments may be demo suggestions (names)
    cooc_min: int
    same_type: bool
    trend_floor: bool
    vram_gb: float
    status: dict
    refresh_per_piece: int = 1  # refresh suggestions per worn piece while other pieces have options

    @classmethod
    def load(cls, path: str = SETTINGS_PATH) -> "M4Settings":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        d["stores"] = [StoreSpec(**s) for s in d["stores"]]
        return cls(**d)

    def save(self, path: str = SETTINGS_PATH) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
            f.write("\n")

    def abs(self, rel: str) -> str:
        return os.path.join(REPO, rel)

    def with_(self, **kw) -> "M4Settings":
        return replace(self, **kw)

    def provisional(self) -> list:
        return sorted(k for k, v in self.status.items() if v != "calibrated")
