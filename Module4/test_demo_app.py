"""D10 demo tests (spec §7): headless All run, slider without GNN calls, M2 == M4 harmony %."""
import importlib.util
import os

import pytest

from m4_settings import REPO

pytestmark = pytest.mark.real
PHOTO = os.path.join(REPO, "samples", "image2.jpg")


@pytest.fixture(scope="module")
def demo(recommender):
    spec = importlib.util.spec_from_file_location("demo_app", os.path.join(REPO, "demo.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m._recommender = recommender          # share the session's pool + GNN: one copy on the GPU
    return m


@pytest.fixture(scope="module")
def all_run(demo):
    return dict(zip(demo.OUTPUT_NAMES, demo.analyze(PHOTO, demo.ALL, "auto", *demo.DEFAULT_OPTIONS)))


def test_all_returns_every_module(all_run):
    for k in ("perception", "harmony", "trend"):
        assert all_run[k] and os.path.exists(all_run[k]), k
    assert "[error]" not in all_run["log"]
    assert all_run["m4_verdict"].startswith("## ")
    assert all_run["m4_state"]["rec"].verdict in {"swap", "refresh", "keep"}


def test_slider_reranks_without_a_gnn_call(demo, all_run, recommender, monkeypatch):
    calls = []
    monkeypatch.setattr(recommender.scorer, "outfit_logit", lambda *a, **k: calls.append("outfit"))
    monkeypatch.setattr(recommender.scorer, "swap_logits", lambda *a, **k: calls.append("swap"))
    out = dict(zip(demo.M4_OUTPUT_NAMES, demo.rerank(all_run["m4_state"], 0.8, "max_gain")))
    assert calls == []
    r = out["m4_state"]["rec"]
    assert r.weights["w_t"] == pytest.approx(0.8) and r.weights["w_h"] == pytest.approx(0.2)
    assert r.selection_rule == "max_gain"
    assert out["m4_state"]["table"] is all_run["m4_state"]["table"]


def test_m2_and_m4_agree_on_harmony(demo, all_run, tmp_path):
    outfit_dir = os.path.join(demo.DEMO_OUT, demo.IMAGE_ID)
    score, _ = demo.run_module2(outfit_dir, PHOTO, str(tmp_path / "h.png"))
    assert score == pytest.approx(all_run["m4_state"]["rec"].baseline["harmony_pct"], abs=0.01)


def test_try_swap_draws_a_board(demo, all_run):
    from demo_panel import choice_labels
    labels = choice_labels(all_run["m4_state"]["rec"])
    if not labels:
        pytest.skip("image2 got a 'keep' verdict: nothing to try")
    png, text = demo.try_swap(all_run["m4_state"], labels[0])
    assert os.path.exists(png) and "Harmony" in text
