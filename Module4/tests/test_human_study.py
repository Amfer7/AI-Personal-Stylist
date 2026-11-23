"""Human study pilot: blind A/B pair layout and scoring of the form responses."""
import numpy as np
import pandas as pd
import pytest

from human_study import make_pairs, score_responses


def _items(n):
    return [{"outfit": f"o{i}", "condition": c} for i in range(n) for c in ("m4", "random")]


def test_make_pairs_numbers_questions_and_hides_the_variant_side():
    pairs = make_pairs(_items(10), seed=0)
    assert [p["q"] for p in pairs] == [f"Q{i:02d}" for i in range(1, 21)]
    assert {p["variant_side"] for p in pairs} == {"A", "B"}
    assert sum(p["variant_side"] == "A" for p in pairs) == 10          # balanced: half each side
    assert make_pairs(_items(10), seed=0) == pairs


def test_make_pairs_never_shows_both_versions_of_an_outfit_back_to_back():
    pairs = make_pairs(_items(20), seed=3)
    assert all(pairs[i]["outfit"] != pairs[i + 1]["outfit"] for i in range(len(pairs) - 1))


def test_score_responses_counts_variant_preference_per_condition():
    key = pd.DataFrame([{"q": "Q01", "condition": "m4", "variant_side": "A"},
                        {"q": "Q02", "condition": "random", "variant_side": "B"}])
    responses = pd.DataFrame({
        "Timestamp": ["t1", "t2", "t3", "t4"],
        "Q01. Which outfit goes together better?": ["Outfit A", "Outfit A", "Outfit B", "Can't decide"],
        "Q02. Which outfit goes together better?": ["Outfit A", "Outfit A", "Outfit B", "Outfit A"],
    })
    res = score_responses(responses, key, n_boot=200, seed=0)
    m4, rnd = res["m4"], res["random"]
    assert m4["votes"] == 3 and m4["undecided"] == 1 and m4["prefer_variant"] == pytest.approx(2 / 3)
    assert rnd["votes"] == 4 and rnd["prefer_variant"] == pytest.approx(1 / 4)
    assert res["raters"] == 4
    assert res["lift"] == pytest.approx(2 / 3 - 1 / 4)
    lo, hi = res["lift_ci95"]
    assert lo <= res["lift"] <= hi
