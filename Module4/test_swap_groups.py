import numpy as np
import pytest

from swap_groups import (BROAD_OF_TYPE, SWAP_GROUPS, TYPES, CooccurrenceFilter, replacement_types,
                         swap_group)

STORE_TYPES = ["shoe", "pants", "bag", "jacket", "skirt", "dress", "shirt_blouse", "sweater", "leggings",
               "hat", "scarf", "top", "t_shirt", "belt", "sunglasses", "shorts"]


def test_every_store_type_has_exactly_one_group():
    assert sorted(STORE_TYPES) == sorted(TYPES)
    for t in STORE_TYPES:
        g = swap_group(t)
        assert g is not None and t in SWAP_GROUPS[g]
    assert sum(len(v) for v in SWAP_GROUPS.values()) == 16


def test_groups_match_spec():
    assert set(SWAP_GROUPS["upper"]) == {"t_shirt", "sweater", "shirt_blouse", "jacket", "top"}
    assert set(SWAP_GROUPS["lower"]) == {"pants", "leggings", "shorts", "skirt"}
    assert SWAP_GROUPS["dress"] == ("dress",)
    for acc in ["shoe", "bag", "hat", "belt", "scarf", "sunglasses"]:
        assert SWAP_GROUPS[acc] == (acc,)


def test_broad_of_type_matches_m1():
    assert {BROAD_OF_TYPE[t] for t in SWAP_GROUPS["upper"]} == {"top"}
    assert {BROAD_OF_TYPE[t] for t in SWAP_GROUPS["lower"]} == {"bottom"}
    assert BROAD_OF_TYPE["dress"] == "dress" and BROAD_OF_TYPE["shoe"] == "accessory"


def test_replacement_types_and_same_type():
    assert set(replacement_types("jacket")) == set(SWAP_GROUPS["upper"])
    assert replacement_types("jacket", same_type=True) == ("jacket",)
    assert replacement_types("hoodie") == ()


def _filter():
    outfits = ([["shirt_blouse", "skirt", "leggings", "shoe"]] * 30
               + [["t_shirt", "pants", "shoe"]] * 30
               + [["jacket", "skirt", "shoe"]] * 30
               + [["sweater", "shorts", "shoe"]] * 5)
    return CooccurrenceFilter.from_outfits(outfits, min_count=20)


def test_counts_are_outfits_containing_both():
    f = _filter()
    from swap_groups import TYPE_ID
    assert f.counts[TYPE_ID["skirt"], TYPE_ID["leggings"]] == 30
    assert f.counts[TYPE_ID["skirt"], TYPE_ID["shoe"]] == 60
    assert f.counts[TYPE_ID["pants"], TYPE_ID["leggings"]] == 0
    assert np.all(np.diag(f.counts) == 0)          # no outfit wears two of one type
    assert np.array_equal(f.counts, f.counts.T)


def test_skirt_plus_leggings_never_swaps_skirt_to_pants():
    f = _filter()
    others = ["shirt_blouse", "leggings", "shoe"]   # swapping the skirt
    allowed = [t for t in replacement_types("skirt") if f.allowed(t, others)]
    assert "pants" not in allowed
    assert "leggings" not in allowed                # already worn: would duplicate the type
    assert allowed == ["skirt"]


def test_rare_pairs_are_blocked_below_min_count():
    f = _filter()
    assert not f.allowed("shorts", ["sweater"])    # 5 < 20
    assert f.allowed("pants", ["t_shirt", "shoe"])


def test_unknown_other_types_are_ignored():
    assert _filter().allowed("skirt", ["shirt_blouse", "hoodie"])


def test_save_load_round_trip(tmp_path):
    f = _filter()
    p = str(tmp_path / "cooc.npy")
    f.save(p)
    g = CooccurrenceFilter.load(p, min_count=20)
    assert np.array_equal(f.counts, g.counts) and g.min_count == 20


@pytest.mark.real
def test_real_counts_from_cache(settings):
    import os
    from candidate_pool import pool_dir
    from swap_groups import TYPE_ID
    c = CooccurrenceFilter.load(os.path.join(pool_dir(settings), "cooc.npy")).counts
    assert c[TYPE_ID["pants"], TYPE_ID["leggings"]] == 0
    assert c[TYPE_ID["skirt"], TYPE_ID["leggings"]] >= 20
    upper = [TYPE_ID[t] for t in SWAP_GROUPS["upper"]]
    assert c[np.ix_(upper, upper)].sum() == 0
