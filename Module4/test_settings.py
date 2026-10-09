import os

import pytest

from m4_settings import M4Settings


def test_default_settings_load_and_round_trip(tmp_path):
    s = M4Settings.load()
    assert s.rule in s.tau
    assert s.vram_gb == 6.0
    assert s.stores[0].name == "fashion144k"
    p = tmp_path / "s.json"
    s.save(str(p))
    assert M4Settings.load(str(p)) == s


def test_paths_are_repo_relative():
    s = M4Settings.load()
    rels = [s.checkpoint, s.trend_model, s.cache_dir]
    for st in s.stores:
        rels += [st.store_dir, st.output_root, st.split]
    for rel in rels:
        assert not os.path.isabs(rel), rel


def test_provisional_lists_uncalibrated_knobs():
    s = M4Settings.load()
    assert "tau" in s.provisional()
    s2 = s.with_(status={**s.status, "tau": "calibrated"})
    assert "tau" not in s2.provisional()
    assert s.status["tau"] == "provisional"  # with_ does not mutate


def test_bridge_constants():
    from m2_bridge import BROAD_VOCAB, FEATURE_COLS, NODE_DIM, od
    assert NODE_DIM == 555
    assert FEATURE_COLS[0] == 0 and FEATURE_COLS[543] == 543 and FEATURE_COLS[-1] == 569
    for pair in od.CATEGORY_EDGE_WEIGHTS:
        assert pair <= set(BROAD_VOCAB)


@pytest.mark.real
def test_checkpoint_loads_as_attr_subset(settings, device):
    from m2_bridge import load_gnn
    model = load_gnn(settings.abs(settings.checkpoint), device)
    assert model.input_proj[0].in_features == 555
