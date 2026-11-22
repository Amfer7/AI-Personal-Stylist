"""Catalogue visual QA: the per-type sample and the defect rates from the reviewer's marks."""
import numpy as np
import pandas as pd
import pytest

from catalogue_qa import MARKS, defect_rates, sample_by_type


def test_sample_by_type_is_seeded_and_capped():
    rows = {"shoe": np.arange(100), "belt": np.arange(100, 110)}
    s = sample_by_type(rows, per_type=30, seed=0)
    assert len(s["shoe"]) == 30 and len(set(s["shoe"])) == 30
    assert sorted(s["belt"]) == list(range(100, 110))            # fewer than 30: all of them
    assert sample_by_type(rows, per_type=30, seed=0) == s


def test_defect_rates_treat_blank_as_ok_and_reject_unknown_marks():
    marks = pd.DataFrame({"type": ["shoe"] * 4 + ["belt"] * 2,
                          "mark": ["", "background", None, "wrong_type", "", "unreadable"]})
    r = defect_rates(marks)
    assert r["shoe"] == {"items": 4, "flagged": 2, "rate": 0.5, "background": 1, "wrong_type": 1}
    assert r["belt"]["rate"] == 0.5
    assert r["all"]["items"] == 6 and r["all"]["flagged"] == 3
    with pytest.raises(ValueError):
        defect_rates(pd.DataFrame({"type": ["shoe"], "mark": ["meh"]}))
    assert "background" in MARKS
