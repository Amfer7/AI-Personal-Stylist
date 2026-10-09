import numpy as np
import pytest
import torch

from harmony_reference import HarmonyReference, test_outfits


def test_pct_is_monotone_and_bounded():
    ref = HarmonyReference(np.linspace(-3, 3, 7), 1.0, 1.0)
    vals = [ref.pct(x) for x in (-10, -3, -0.5, 0, 2.9, 3, 10)]
    assert vals == sorted(vals)
    assert vals[0] == 0.0 and vals[-1] == 100.0
    assert ref.pct(0.0) == pytest.approx(100 * 4 / 7)   # side="right": ties count as beaten


def test_save_load_round_trip(tmp_path):
    ref = HarmonyReference(np.array([-1.0, 0.5, 2.0]), 0.7, 12.5)
    p = str(tmp_path / "r.json")
    ref.save(p)
    got = HarmonyReference.load(p)
    assert np.array_equal(got.logits_sorted, ref.logits_sorted)
    assert (got.sigma_logit, got.sigma_trend) == (0.7, 12.5)


@pytest.mark.real
def test_reference_covers_all_test_outfits_and_matches_model(pool, gnn, settings):
    import os
    from candidate_pool import pool_dir
    from m2_bridge import ATTR_MASK, BROAD_VOCAB, od
    ref = HarmonyReference.load(os.path.join(pool_dir(settings), "reference.json"))
    outs = test_outfits(pool)
    assert len(ref.logits_sorted) == len(outs) > 40000
    assert ref.sigma_logit > 0 and ref.sigma_trend > 0
    # every size-7 test outfit, re-scored the slow M2 way, must appear in the reference
    feat = np.load(settings.abs(os.path.join(settings.stores[0].store_dir, "features.npy")), mmap_mode="r")
    for rows in [o for o in outs if len(o) == 7]:
        nodes = [{"embedding": np.asarray(feat[r, :512]), "colour": np.asarray(feat[r, 512:544]),
                  "attr_feat": np.asarray(feat[r, 544:]), "category": BROAD_VOCAB[pool.broad[r]],
                  "hue": float(pool.hue[r]), "formality": float(pool.formality[r]), "area": float(pool.area[r])}
                 for r in rows]
        g = od.OutfitDataset.build_graph(nodes, 0.0, attr_mask=ATTR_MASK)
        with torch.no_grad():
            want = float(gnn.model(g["x"].to(gnn.device), g["edge_index"].to(gnn.device),
                                   g["edge_attr"].to(gnn.device))[0])
        assert np.min(np.abs(ref.logits_sorted - want)) < 1e-4
