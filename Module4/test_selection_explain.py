import math

import numpy as np
import pytest

from explain import what_changed
from selection import cluster_keys, combine, diverse_top, selection_score


def test_combine():
    assert combine(np.array([1.0, -2.0]), np.array([0.5, 1.0]), 1.0, 0.5).tolist() == [1.25, -1.5]


def test_selection_rules():
    g = np.array([1.0, 5.0, 3.0, -2.0, 4.0, 0.0])
    assert selection_score(g, "max_gain") == 5.0
    assert selection_score(g, "topk_mean_gain", k=3) == pytest.approx(4.0)
    assert selection_score(g, "topk_mean_gain", k=10) == pytest.approx(11 / 6)
    assert selection_score(g, "slot_percentile") == pytest.approx(4 / 6)
    assert selection_score(np.array([]), "max_gain") == -math.inf
    with pytest.raises(ValueError):
        selection_score(g, "nope")


def test_cluster_keys_bucket_unmatched_per_broad():
    k = cluster_keys(np.array([0, 0, 1, 0]), np.array([5, -1, 5, -1]))
    assert k[0] != k[2] and k[1] == k[3] and k[1] != k[0]


def test_diverse_top_skips_same_cluster_then_fills():
    g = np.array([3.0, 2.9, 2.0, 1.0, 0.5])
    keys = np.array([5, 5, 6, 7, 0])
    assert diverse_top(g, keys, np.ones(5, bool)) == [0, 2, 3]
    assert diverse_top(g, keys, np.array([1, 1, 0, 0, 0], bool)) == [0, 1, 2]   # too few qualify: fill by gain
    assert diverse_top(np.array([1.0]), np.array([1]), np.ones(1, bool)) == [0]
    assert diverse_top(np.array([]), np.array([]), np.array([], bool)) == []


def test_what_changed_describes_formality_and_type():
    cur = {"type": "skirt", "hue": 0.0, "formality": 20.0, "area": 1000.0}
    cand = {"type": "pants", "hue": 10.0, "formality": 60.0, "area": 1000.0}
    others = [{"type": "shirt_blouse", "hue": 180.0, "formality": 70.0, "area": 1000.0}]
    s = what_changed(cur, cand, others)
    assert s.startswith("Skirt → pants")
    assert "formality closer to the rest of the outfit (gap 50 → 10)" in s
    assert "colour" not in s and "piece relative" not in s


def test_what_changed_colour_volume_and_fallback():
    cur = {"type": "shoe", "hue": 0.0, "formality": 50.0, "area": 500.0}
    cand = {"type": "shoe", "hue": 170.0, "formality": 50.0, "area": 2000.0}
    others = [{"type": "pants", "hue": 10.0, "formality": 50.0, "area": 8000.0}]
    s = what_changed(cur, cand, others)
    assert "more contrast with the rest" in s and "larger piece" in s
    same = what_changed(cur, dict(cur), others)
    assert same.startswith("Similar formality, colour and volume")
    assert what_changed(cur, {**cur, "hue": float("nan")}, others).startswith("Similar")
