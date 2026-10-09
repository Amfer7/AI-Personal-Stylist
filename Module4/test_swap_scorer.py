import random

import numpy as np
import pytest
import torch

from harmony_reference import test_outfits
from m2_bridge import ATTR_MASK, BROAD_VOCAB, od
from outfit_state import state_from_pool
from swap_scorer import SwapScorer

pytestmark = pytest.mark.real


def _nodes(pool, feat, rows):
    return [{"embedding": np.asarray(feat[r, :512]), "colour": np.asarray(feat[r, 512:544]),
             "attr_feat": np.asarray(feat[r, 544:]), "category": BROAD_VOCAB[pool.broad[r]],
             "hue": float(pool.hue[r]), "formality": float(pool.formality[r]), "area": float(pool.area[r])}
            for r in rows]


def _model_logit(gnn, nodes):
    g = od.OutfitDataset.build_graph(nodes, 0.0, attr_mask=ATTR_MASK)
    with torch.no_grad():
        return float(gnn.model(g["x"].to(gnn.device), g["edge_index"].to(gnn.device),
                               g["edge_attr"].to(gnn.device))[0])


@pytest.fixture(scope="module")
def feat(settings):
    import os
    return np.load(settings.abs(os.path.join(settings.stores[0].store_dir, "features.npy")), mmap_mode="r")


def test_outfit_and_swap_logits_match_model(pool, gnn, feat):
    sc = SwapScorer(pool, gnn)
    rng = random.Random(0)
    outs = test_outfits(pool)
    for rows in rng.sample(outs, 20):
        key = pool.outfit_key(int(rows[0]))
        st = state_from_pool(pool, "fashion144k", key)
        rows = [p.pool_row for p in st.pieces]
        assert abs(sc.outfit_logit(st) - _model_logit(gnn, _nodes(pool, feat, rows))) < 1e-4
        i = rng.randrange(len(rows))
        cands = torch.as_tensor(rng.sample(range(pool.n), 32), device=pool.device)
        got = sc.swap_logits(st, i, cands)
        for c, g in zip(cands.tolist(), got):
            sw = list(rows)
            sw[i] = c
            assert abs(g - _model_logit(gnn, _nodes(pool, feat, sw))) < 1e-4
