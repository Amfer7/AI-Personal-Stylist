import numpy as np
import pytest
import torch

from candidate_pool import SPLIT_CODES, crops_present, repost_groups, representatives


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _groups(emb, types, outfit, area, colour=None):
    n = len(types)
    colour = np.zeros((n, 32), np.float32) if colour is None else np.asarray(colour, np.float32)
    return repost_groups(torch.from_numpy(_unit(emb)), torch.from_numpy(colour), np.asarray(types, np.int8),
                         np.asarray(outfit, np.int64), np.asarray(area, np.float64), 0.95, 0.25, 1.25).tolist()


def test_reposts_need_every_garment_to_match():
    rng = np.random.default_rng(0)
    top, shoe, other = _unit(rng.standard_normal((3, 64)))
    #            outfit 0     outfit 1 (repost)  outfit 2 (same shoe only)  outfits 3, 4 (one garment each)
    emb = np.stack([top, shoe, top, shoe, other, shoe, shoe, shoe])
    types = [1, 2, 1, 2, 1, 2, 2, 2]
    outfit = [0, 0, 1, 1, 2, 2, 3, 4]
    area = [1000] * 8
    assert _groups(emb, types, outfit, area) == [0, 0, 0, 0, 2, 2, 3, 4]


def test_reposts_need_matching_colour_and_size():
    rng = np.random.default_rng(1)
    top, shoe = _unit(rng.standard_normal((2, 64)))
    emb = np.stack([top, shoe, top, shoe, top, shoe])
    colour = np.zeros((6, 32))
    colour[3, 0] = 1.0                                   # outfit 1: shoe colours differ
    area = [1000, 1000, 1000, 1000, 1000, 5000]          # outfit 2: shoe 5x larger
    assert _groups(emb, [1, 2] * 3, [0, 0, 1, 1, 2, 2], area, colour) == [0, 0, 1, 1, 2, 2]


def test_reposts_do_not_chain():
    """0~1 and 1~2 but not 0~2 (areas drift): outfit 2 must not join outfit 0 through outfit 1."""
    rng = np.random.default_rng(2)
    top, shoe = _unit(rng.standard_normal((2, 64)))
    emb = np.stack([top, shoe] * 3)
    area = [1000, 1000, 1200, 1200, 1440, 1440]
    assert _groups(emb, [1, 2] * 3, [0, 0, 1, 1, 2, 2], area) == [0, 0, 0, 0, 2, 2]


def test_representatives_pick_largest_eligible_crop_per_group_and_type():
    repost = np.array([0, 0, 0, 0, 4, 4])
    types = np.array([1, 2, 1, 2, 1, 2], np.int8)
    area = np.array([10, 50, 30, 50, 1, 1], np.float64)
    eligible = np.array([True, True, True, True, True, False])
    assert representatives(repost, types, area, eligible).tolist() == [2, 1, 2, 1, 4, -1]


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
    # every member of a repost group shares its representative's type and group
    mem = np.nonzero((pool.rep_demo >= 0) & (pool.rep_demo != idx))[0]
    assert (pool.repost[pool.rep_demo[mem]] == pool.repost[mem]).all()
    # repost groups are small: same photo posted a few times, never thousands of distinct items
    sizes = np.bincount(pool.repost[pool.repost >= 0])
    assert sizes.max() <= 10 * 8
