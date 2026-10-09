import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from m2_bridge import ATTR_MASK, od
from outfit_state import REFERENCE_FRAME_PX, load_outfit_dir, photo_pixels
from recommendation import Recommendation

pytestmark = pytest.mark.real


def test_vis_overlay_has_the_input_photo_size(settings, image2_dir):
    with Image.open(settings.abs("samples/image2.jpg")) as im:
        assert photo_pixels(image2_dir) == im.size[0] * im.size[1]


def test_recommend_on_image2_returns_valid_recommendation(recommender, image2_dir, settings):
    r = recommender.recommend(image2_dir)
    assert r.verdict in {"swap", "refresh", "keep"}
    assert len(r.suggestions) <= 3
    assert len({s.pool_row for s in r.suggestions}) == len(r.suggestions)
    for s in r.suggestions:
        assert os.path.exists(settings.abs(s.crop_path))
        assert not os.path.isabs(s.crop_path)
    assert Recommendation.from_json(r.to_json()) == r
    assert 0.0 <= r.baseline["harmony_pct"] <= 100.0


def test_unswapped_logit_matches_m2_path(recommender, image2_dir):
    """M2 and M4 must agree on the user's own outfit (D10). M2 uses raw areas; the logit is
    resolution-invariant within one photo, so it equals M4's area-normalised logit closely."""
    with open(os.path.join(image2_dir, "metadata.json"), encoding="utf-8") as f:
        garments = json.load(f)
    nodes = [od.build_node_from_meta(image2_dir, g) for g in garments]
    g = od.OutfitDataset.build_graph(nodes, 0.0, attr_mask=ATTR_MASK)
    dev = recommender.gnn.device
    with torch.no_grad():
        m2 = float(recommender.gnn.model(g["x"].to(dev), g["edge_index"].to(dev), g["edge_attr"].to(dev))[0])
    st = load_outfit_dir(image2_dir, recommender.gnn, recommender.trend_model)
    raw = load_outfit_dir(image2_dir, recommender.gnn, recommender.trend_model, photo_px=REFERENCE_FRAME_PX)
    assert abs(recommender.scorer.outfit_logit(raw) - m2) < 1e-4       # unscaled areas: exact parity
    assert abs(recommender.scorer.outfit_logit(st) - m2) < 0.05         # +1 smoothing only
