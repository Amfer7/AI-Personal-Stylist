"""Module 4's single doorway into Module 2 and Module 3.

Module 2's files start with digits (02_outfit_dataset.py), so they can't be imported by name;
they are loaded here once, the same way Module2/src/demo_score.py does. Module 3's trend pickle needs
`trend_model` importable by bare name, so Module3/src goes on sys.path.
"""
from __future__ import annotations

import os
import sys
import warnings
from importlib.util import module_from_spec, spec_from_file_location

import numpy as np
import torch

from m4_settings import REPO

_M2 = os.path.join(REPO, "Module2", "src")
_M3 = os.path.join(REPO, "Module3", "src")


def _load(fname: str, name: str):
    spec = spec_from_file_location(name, os.path.join(_M2, fname))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


od = _load("02_outfit_dataset.py", "outfit_dataset")
tg = _load("03_train_gnn.py", "train_gnn")

if _M3 not in sys.path:
    sys.path.insert(0, _M3)
import trend_model as _tm  # noqa: E402

# The shipped checkpoint is the attr-subset arm: [512 CLIP | 32 colour | 11 reliable attrs].
ATTR_MASK = od.attr_subset_indices()
FEATURE_COLS = np.r_[0:544, 544 + np.asarray(ATTR_MASK)]   # columns of a 570-D store row it reads
NODE_DIM = len(FEATURE_COLS)                               # 555

# Every broad category M2's edge table knows. The store only uses the first four.
BROAD_VOCAB = ("top", "bottom", "dress", "accessory", "outerwear", "shoe", "bag")
BROAD_ID = {b: i for i, b in enumerate(BROAD_VOCAB)}

TREND_MATCH_THRESHOLD = _tm.MATCH_CONFIDENCE_THRESHOLD


def edge_weight_table(device) -> torch.Tensor:
    """[V, V] float64 category edge weights, built from od.get_edge_weight (diagonal = 0.2)."""
    return torch.tensor([[od.get_edge_weight(a, b) for b in BROAD_VOCAB] for a in BROAD_VOCAB],
                        dtype=torch.float64, device=device)


def node_features(node: dict) -> np.ndarray:
    """The 555-D input the checkpoint reads, from an od node dict."""
    return np.concatenate([node["embedding"], node["colour"], node["attr_feat"][ATTR_MASK]]).astype(np.float32)


def load_gnn(checkpoint: str, device) -> torch.nn.Module:
    model = tg.CLIPOutfitGNN(attr_dim=len(ATTR_MASK), edge_dim=od.EDGE_ATTR_DIM).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model.eval()


def load_trend_model(path: str):
    return _tm.PerCategoryTrendModel.load(path)


def pick_device(device=None) -> torch.device:
    """An indexed device ("cuda" becomes "cuda:<current>"), as the VRAM cap API requires."""
    if device is None and torch.cuda.is_available():
        device = "cuda"
    if device:
        d = torch.device(device)
        if d.type == "cuda" and d.index is None:
            d = torch.device("cuda", torch.cuda.current_device())
        return d
    warnings.warn("CUDA not available: Module 4 runs on CPU (correct, but slow)")
    return torch.device("cpu")


def apply_vram_cap(vram_gb: float, device: torch.device) -> None:
    """Hard cap on this process's CUDA allocations (the GPU is shared)."""
    if device.type != "cuda":
        return
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, vram_gb * 2**30 / total), device)
