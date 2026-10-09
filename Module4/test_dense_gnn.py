import json
import os
import random

import numpy as np
import pytest
import torch

from dense_gnn import edge_tensor
from m2_bridge import ATTR_MASK, BROAD_ID, FEATURE_COLS, edge_weight_table, od


def test_edge_tensor_matches_edge_features_including_unknown_hue():
    nodes = [
        {"category": "top", "hue": 350.0, "formality": 70.0, "area": 9000.0},
        {"category": "bottom", "hue": None, "formality": 20.0, "area": 7000.0},
        {"category": "accessory", "hue": 15.0, "formality": 50.0, "area": 500.0},
        {"category": "accessory", "hue": 200.0, "formality": 55.0, "area": 0.0},
    ]
    hue = torch.tensor([[np.nan if n["hue"] is None else n["hue"] for n in nodes]], dtype=torch.float64)
    form = torch.tensor([[n["formality"] for n in nodes]], dtype=torch.float64)
    area = torch.tensor([[n["area"] for n in nodes]], dtype=torch.float64)
    broad = torch.tensor([[BROAD_ID[n["category"]] for n in nodes]])
    E = edge_tensor(hue, form, area, broad, edge_weight_table("cpu"))
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if i != j:
                assert torch.equal(E[0, i, j], torch.from_numpy(od.edge_features(a, b))), (i, j)


# ---------------- real data ----------------

@pytest.fixture(scope="module")
def store(settings):
    d = settings.abs(settings.stores[0].store_dir)
    feat = np.load(os.path.join(d, "features.npy"), mmap_mode="r")
    scal = np.load(os.path.join(d, "node_scalars.npy"))
    with open(os.path.join(d, "index.json"), encoding="utf-8") as f:
        index = json.load(f)
    broad = np.empty(feat.shape[0], dtype=object)
    for es in index["outfits"].values():
        for e in es:
            broad[e["row"]] = e["broad_category"]
    outfits = [[e["row"] for e in es] for es in index["outfits"].values() if es]
    return feat, scal, broad, outfits


def store_node(store, r):
    feat, scal, broad, _ = store
    hue = float(scal[r, 0])
    return {"embedding": np.asarray(feat[r, :512]), "colour": np.asarray(feat[r, 512:544]),
            "attr_feat": np.asarray(feat[r, 544:]), "category": broad[r],
            "hue": None if np.isnan(hue) else hue, "formality": float(scal[r, 1]), "area": float(scal[r, 2])}


def dense_inputs(gnn, store, rows_2d):
    feat, scal, broad, _ = store
    R = np.asarray(rows_2d)
    flat = R.reshape(-1)
    x = torch.from_numpy(np.asarray(feat[flat])[:, FEATURE_COLS].astype(np.float32))
    h0 = gnn.project(x).reshape(*R.shape, -1)
    s = torch.from_numpy(scal[flat].astype(np.float64)).to(gnn.device).reshape(*R.shape, 3)
    b = torch.tensor([BROAD_ID[broad[r]] for r in flat], device=gnn.device).reshape(R.shape)
    return h0, s[..., 0], s[..., 1], s[..., 2], b


def model_logit(gnn, nodes):
    g = od.OutfitDataset.build_graph(nodes, 0.0, attr_mask=ATTR_MASK)
    dev = gnn.device
    with torch.no_grad():
        out, _ = gnn.model(g["x"].to(dev), g["edge_index"].to(dev), g["edge_attr"].to(dev))
    return float(out), g


@pytest.mark.real
def test_feature_cols_reproduce_build_graph_node_input(store):
    rows = random.Random(0).sample(range(store[0].shape[0]), 50)
    g = od.OutfitDataset.build_graph([store_node(store, r) for r in rows], 0.0, attr_mask=ATTR_MASK)
    assert torch.equal(g["x"], torch.from_numpy(np.asarray(store[0][rows])[:, FEATURE_COLS]))


@pytest.mark.real
def test_edges_and_logits_match_build_graph_on_1000_random_swaps(gnn, store):
    rng = random.Random(0)
    outfits = [o for o in store[3] if len(o) >= 2]
    N = store[0].shape[0]
    worst = 0.0
    for _ in range(1000):
        rows = list(rng.choice(outfits))
        rows[rng.randrange(len(rows))] = rng.randrange(N)
        ref, g = model_logit(gnn, [store_node(store, r) for r in rows])
        h0, hue, form, area, broad = dense_inputs(gnn, store, [rows])
        E = edge_tensor(hue, form, area, broad, gnn.wtable)
        src, dst = g["edge_index"]
        assert torch.equal(E[0, src, dst].cpu(), g["edge_attr"]), "edge features differ from build_graph"
        worst = max(worst, abs(float(gnn.logits(h0, hue, form, area, broad)[0]) - ref))
    assert worst < 1e-4, f"max |logit diff| = {worst:.2e}"


@pytest.mark.real
def test_single_garment_outfits_match(gnn, store):
    singles = [o for o in store[3] if len(o) == 1]
    for rows in random.Random(1).sample(singles, 200):
        ref, _ = model_logit(gnn, [store_node(store, rows[0])])
        assert abs(float(gnn.logits(*dense_inputs(gnn, store, [rows]))[0]) - ref) < 1e-4


@pytest.mark.real
def test_chunked_equals_unchunked_even_with_tiny_chunks(gnn, store):
    fives = [o for o in store[3] if len(o) == 5][:300]
    batch = dense_inputs(gnn, store, fives)
    full = gnn.logits(*batch).cpu()
    chunked = gnn.logits_chunked(lambda a, b: tuple(t[a:b] for t in batch), len(fives), 5, max_chunk=7)
    assert torch.allclose(full, chunked, atol=1e-6)
