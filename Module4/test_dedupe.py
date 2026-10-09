import numpy as np
import pytest
import torch

from candidate_pool import SPLIT_CODES, crops_present, dedupe


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_dedupe_groups_transitively_and_picks_largest_crop():
    rng = np.random.default_rng(0)
    base = _unit(rng.standard_normal((3, 64)))
    noise = lambda: 0.02 * rng.standard_normal(64)
    emb = _unit(np.stack([base[0], base[0] + noise(), base[0] + noise(),   # rows 0-2: one item
                          base[1], base[1] + noise(),                      # rows 3-4: another
                          base[2],                                         # row 5: unique
                          base[0]]))                                       # row 6: same pixels, other type
    types = np.array([3, 3, 3, 3, 3, 3, 7], np.int8)
    area = np.array([10, 50, 50, 5, 1, 9, 99], np.float64)
    rep = dedupe(torch.from_numpy(emb), types, area, t_dup=0.95, eligible=np.ones(7, bool))
    assert rep.tolist() == [1, 1, 1, 3, 3, 5, 6]


def test_dedupe_skips_ineligible_rows():
    emb = _unit(np.eye(4, 8) + 0.0)
    rep = dedupe(torch.from_numpy(emb), np.zeros(4, np.int8), np.ones(4), 0.95,
                 eligible=np.array([True, False, True, True]))
    assert rep.tolist() == [0, -1, 2, 3]


def test_crops_present_lists_only_files_on_disk(tmp_path):
    (tmp_path / "1").mkdir()
    (tmp_path / "1" / "bag_0.png").write_bytes(b"")
    (tmp_path / "2_vis.jpg").write_bytes(b"")
    got = crops_present(str(tmp_path), {"1": ["bag_0.png", "shoe_0.png"], "2": ["hat_0.png"]})
    assert got == {"1": {"bag_0.png"}}
    assert crops_present(str(tmp_path / "missing"), {"1": ["bag_0.png"]}) == {}


@pytest.mark.real
def test_real_dedupe_invariants(pool, settings):
    idx = np.arange(pool.n)
    for rep in (pool.rep_demo, pool.rep_eval):
        ok = rep >= 0
        assert (rep[rep[ok]] == rep[ok]).all()                 # representatives represent themselves
        assert (pool.type[rep[ok]] == pool.type[ok]).all()    # never across types
    assert (pool.split[pool.rep_eval >= 0] == SPLIT_CODES["test"]).all()
    # distinct representatives of one type are never near-duplicates (exhaustive on sunglasses)
    from swap_groups import TYPE_ID
    rows = np.nonzero((pool.type == TYPE_ID["sunglasses"]) & (pool.rep_demo == idx))[0]
    E = pool.emb[torch.as_tensor(rows, device=pool.device)]
    sims = E @ E.T
    sims.fill_diagonal_(-1)
    assert float(sims.max()) < settings.t_dup
