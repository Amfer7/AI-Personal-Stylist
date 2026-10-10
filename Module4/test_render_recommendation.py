"""Slide figures: the trade-off plot from sweep data, and the per-verdict example figure."""
import os

import pytest
from PIL import Image

from render_recommendation import load_sweeps, pick_examples, render_examples, render_tradeoff

SWEEP = [{"w_t": w, "mean_dH": 2.5 - 1.5 * w, "mean_dT": 2.6 * w, "hit_rate": 0.75 - 0.4 * w, "recall@10": 0.0}
         for w in (0.0, 0.1, 0.2, 0.5, 1.0)]


def test_tradeoff_figure(tmp_path):
    out = render_tradeoff({"fashion144k": SWEEP, "pinterest": SWEEP}, 0.2, str(tmp_path / "t.png"))
    w, h = Image.open(out).size
    assert w > 1500 and h > 600


@pytest.mark.real
def test_load_sweeps_reads_both_report_summaries(settings):
    sweeps = load_sweeps(settings.abs(os.path.join(settings.cache_dir, "eval")))
    assert set(sweeps) == {"fashion144k", "pinterest"}
    assert [r["w_t"] for r in sweeps["pinterest"]][:3] == pytest.approx([0.0, 0.1, 0.2])


@pytest.mark.real
def test_examples_figure(recommender, tmp_path):
    found = pick_examples(recommender, ["pinterest"], limit=40)
    assert found and set(found) <= {"swap", "refresh", "keep"}
    out = render_examples(recommender, found, str(tmp_path / "e.png"))
    assert os.path.exists(out)
