import json

import numpy as np
import pytest
import torch
from PIL import Image

from outfit_state import REFERENCE_FRAME_PX, NoGarmentsDetected, load_outfit_dir


class StubGNN:
    device = torch.device("cpu")

    def project(self, x):
        return torch.zeros(x.shape[0], 128)


class StubTrend:
    def score_embedding(self, emb, category):
        if category == "bottom":
            return {"trendiness_score": None, "cluster_id": None}
        return {"trendiness_score": 42.0, "cluster_id": 3}


def _garment(d, name, category, broad, pixels, hue):
    Image.new("RGBA", (20, 30), (200, 30, 30, 255)).save(d / f"{name}.png")
    v = np.random.default_rng(len(name)).standard_normal(512).astype(np.float32)
    np.save(d / f"{name}.npy", v / np.linalg.norm(v))
    return {"file": f"{name}.png", "category": category, "broad_category": broad, "pixel_count": pixels,
            "attributes": {"color": {"hue": hue, "saturation": 0.5, "value": 0.5}, "formality_score": 60}}


def test_loads_pieces_normalises_area_and_scores_trend(tmp_path):
    meta = [_garment(tmp_path, "jacket_0", "jacket", "top", 20000, 350.0),
            _garment(tmp_path, "pants_0", "pants", "bottom", 10000, 210.0)]
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    Image.new("RGB", (512, 384)).save(tmp_path / "photo_vis.jpg")      # 2x the 256x384 frame
    st = load_outfit_dir(str(tmp_path), StubGNN(), StubTrend())
    assert [p.type for p in st.pieces] == ["jacket", "pants"]
    assert [p.group for p in st.pieces] == ["upper", "lower"]
    scale = REFERENCE_FRAME_PX / (512 * 384)
    assert st.pieces[0].area == pytest.approx(20000 * scale)
    assert st.pieces[0].trend == 42.0 and st.pieces[0].cluster == 3
    assert np.isnan(st.pieces[1].trend) and st.pieces[1].cluster == -1
    assert st.pieces[0].formality == 60.0 and st.pieces[0].hue == 350.0
    assert abs(np.linalg.norm(st.pieces[0].emb) - 1) < 1e-5


def test_explicit_photo_size_wins_and_missing_size_is_flagged(tmp_path):
    meta = [_garment(tmp_path, "shoe_0", "shoe", "accessory", 1000, 20.0)]
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    st = load_outfit_dir(str(tmp_path), StubGNN(), StubTrend(), photo_px=REFERENCE_FRAME_PX * 4)
    assert st.pieces[0].area == pytest.approx(250.0)
    st2 = load_outfit_dir(str(tmp_path), StubGNN(), StubTrend())
    assert st2.pieces[0].area == 1000.0 and any("photo size unknown" in f for f in st2.flags)


def test_garment_without_embedding_is_skipped_and_flagged(tmp_path):
    meta = [_garment(tmp_path, "hat_0", "hat", "accessory", 900, 10.0),
            _garment(tmp_path, "bag_0", "bag", "accessory", 900, 10.0)]
    (tmp_path / "bag_0.npy").unlink()
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    st = load_outfit_dir(str(tmp_path), StubGNN(), StubTrend(), photo_px=REFERENCE_FRAME_PX)
    assert [p.file for p in st.pieces] == ["hat_0.png"]
    assert any("bag_0.png" in f for f in st.flags)


def test_no_garments_raises(tmp_path):
    (tmp_path / "metadata.json").write_text("[]")
    with pytest.raises(NoGarmentsDetected):
        load_outfit_dir(str(tmp_path), StubGNN(), StubTrend())


def test_with_swap_is_immutable(tmp_path):
    meta = [_garment(tmp_path, "skirt_0", "skirt", "bottom", 5000, 0.0),
            _garment(tmp_path, "shoe_0", "shoe", "accessory", 900, 0.0)]
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    st = load_outfit_dir(str(tmp_path), StubGNN(), StubTrend(), photo_px=REFERENCE_FRAME_PX)
    st2 = st.with_swap(0, st.pieces[1])
    assert st.pieces[0].type == "skirt" and st2.pieces[0].type == "shoe"
