import os
import random

import numpy as np
import pytest
import torch

from candidate_pool import SPLIT_CODES, load_store_arrays
from m2_bridge import BROAD_VOCAB, FEATURE_COLS, TREND_MATCH_THRESHOLD, load_trend_model
from swap_groups import TYPE_ID, TYPES

pytestmark = pytest.mark.real

SPEC_TEST_COUNTS = {"shoe": 28329, "pants": 19169, "bag": 17130, "jacket": 15025, "skirt": 11701,
                    "dress": 11246, "shirt_blouse": 8436, "sweater": 8015, "leggings": 7268, "hat": 6643,
                    "scarf": 3163, "top": 3131, "t_shirt": 2726, "belt": 1405, "sunglasses": 548, "shorts": 235}


def test_every_store_row_is_in_the_pool_once(pool, settings):
    rows = {s.name: np.load(settings.abs(os.path.join(s.store_dir, "node_scalars.npy")), mmap_mode="r").shape[0]
            for s in settings.stores}
    assert rows["fashion144k"] == 480079
    assert pool.n == sum(rows.values())
    for s in settings.stores:
        assert int((pool.source == pool.source_id(s.name)).sum()) == rows[s.name]
    assert (pool.outfit_ord >= 0).all() and all(f for f in pool.files)
    assert (pool.type >= 0).all()


def test_test_split_counts_match_spec(pool):
    test = (pool.split == SPLIT_CODES["test"]) & (pool.source == pool.source_id("fashion144k"))
    got = {t: int(((pool.type == TYPE_ID[t]) & test).sum()) for t in TYPES}
    assert got == SPEC_TEST_COUNTS


def test_projection_cache_matches_model(pool, gnn, pool_features):
    rows = np.sort(np.random.default_rng(0).choice(pool.n, 2000, replace=False))
    want = gnn.project(torch.from_numpy(pool_features[rows][:, FEATURE_COLS]))
    assert torch.allclose(pool.proj[torch.as_tensor(rows, device=pool.device)], want, atol=1e-5)


def test_trend_scores_match_score_embedding(pool, settings):
    tm = load_trend_model(settings.abs(settings.trend_model))
    emb, _ = load_store_arrays(settings)
    rows = np.random.default_rng(1).choice(pool.n, 2000, replace=False)
    for r in rows:
        cat = BROAD_VOCAB[pool.broad[r]]
        res = tm.score_embedding(emb[r], cat)
        want = np.nan if res["trendiness_score"] is None else res["trendiness_score"]
        if (np.isnan(want) and np.isnan(pool.trend[r])) or np.isclose(want, pool.trend[r], rtol=0, atol=1e-4):
            continue                      # (pool stores float32; the model returns float64)
        # Only float near-ties may disagree: best-vs-second cluster or best-vs-threshold within 1e-5.
        m = tm.models[cat]
        C = np.stack([c.centroid for c in m.clusters.values()])
        e = emb[r] / (np.linalg.norm(emb[r]) + 1e-8)
        s = np.sort(C @ e)
        assert s[-1] - s[-2] < 1e-5 or abs(s[-1] - TREND_MATCH_THRESHOLD) < 1e-5, (r, want, pool.trend[r])


def test_demo_pool_only_holds_garments_with_crops(pool, settings):
    demo = np.nonzero(pool.rep_demo >= 0)[0]
    assert pool.has_crop[demo].all()
    for r in random.Random(0).sample(demo.tolist(), 100):
        assert os.path.exists(settings.abs(pool.crop_path(r))), pool.crop_path(r)


def test_candidate_rows_respect_type_and_mode(pool):
    rows = pool.candidate_rows(["sunglasses"], mode="demo").cpu().numpy()
    assert (pool.type[rows] == TYPE_ID["sunglasses"]).all()
    ev = pool.candidate_rows(["sunglasses"], mode="eval").cpu().numpy()
    assert (pool.split[ev] == SPLIT_CODES["test"]).all() and len(ev) < len(rows)


def test_exclude_drops_the_source_outfit(pool):
    key = pool.outfit_keys["fashion144k"][int(pool.outfit_ord[0])]
    own = set(pool.outfit_rows("fashion144k", key).tolist())
    t = TYPES[pool.type[0]]
    rows = set(pool.candidate_rows([t], mode="demo", exclude=("fashion144k", key)).cpu().numpy().tolist())
    assert not (own & rows)
