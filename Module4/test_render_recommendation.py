"""Slide figures: the trade-off plot from sweep data, and the per-verdict example figure."""
import os

import pytest
import numpy as np
from PIL import Image

from render_recommendation import (crop_readable, drawn_crops, load_sweeps, pick_examples, render_examples,
                                   render_tradeoff)

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
    for v, (src, key, table, r) in found.items():
        crops = drawn_crops(r, table.state, recommender.pool, recommender.settings, v)
        assert len(crops) >= 2 and all(crop_readable(c) for c in crops), (v, src, key)


def test_render_examples_empty_raises(tmp_path):
    with pytest.raises(ValueError, match="no examples found"):
        render_examples(None, {}, str(tmp_path / "e.png"))


def _png(path, alpha):
    h, w = alpha.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = 120
    rgba[..., 3] = alpha
    Image.fromarray(rgba, "RGBA").save(path)
    return str(path)


def test_crop_readable_solid(tmp_path):
    assert crop_readable(_png(tmp_path / "a.png", np.full((150, 100), 255, np.uint8)))


def test_crop_readable_missing(tmp_path):
    assert not crop_readable(str(tmp_path / "nope.png"))


def test_crop_readable_small_region(tmp_path):
    a = np.zeros((120, 120), np.uint8)
    a[10:30, 10:30] = 255
    assert not crop_readable(_png(tmp_path / "s.png", a))


def test_crop_readable_specks(tmp_path):
    a = np.zeros((120, 120), np.uint8)
    a[::10, ::10] = 255
    a[119, 119] = 255
    assert not crop_readable(_png(tmp_path / "k.png", a))
