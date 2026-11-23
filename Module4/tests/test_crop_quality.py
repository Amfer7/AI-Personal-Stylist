"""Crop quality gate: shape metrics, the per-type rule, unreadable files."""
import numpy as np
from PIL import Image

from crop_quality import GROUP_PROMPTS, METRICS, TEXT, clip_scores, crop_metrics, crop_ok, mask_metrics, prompts
from swap_groups import TYPE_ID

GATE = {"min_main": 0.85, "min_fill": 0.35, "min_cover": 0.2, "max_holes": 0.1, "min_label_p": {"default": 0.05, "shoe": 0.0},
        "max_text_p": 0.5, "max_junk_p": {"default": 0.4, "belt": 1.0}, "min_area": {"default": 100, "belt": 10},
        "pair_types": ["shoe"]}
PASSING = (0.5, 0.0, 0.9, 0.1, 0.1)  # cover, holes, label_p, text_p, junk_p that clear every bar


def _metrics(rows):
    a = np.array([tuple(r) + PASSING[len(r) - 4:] for r in rows], np.float64)
    return dict(zip(METRICS, a.T))


def test_mask_metrics_solid_pair_and_specks():
    m = np.zeros((20, 20), bool)
    m[2:12, 2:12] = True                                     # one solid square
    assert mask_metrics(m) == (100.0, 1.0, 1.0, 1.0, 100 / 400, 0.0)
    m[2:12, 15:19] = True                                    # second piece: a pair
    area, main, top2, fill, cover, holes = mask_metrics(m)
    assert area == 140 and np.isclose(main, 100 / 140) and np.isclose(top2, 1.0) and fill == 1.0
    assert np.isclose(cover, 140 / 400) and holes == 0.0
    r = np.zeros((20, 20), bool)
    r[0:10, 0:10] = True
    r[3:7, 3:7] = False                                      # a ring: 16 of 100 enclosed pixels are a hole
    assert np.isclose(mask_metrics(r)[5], 16 / 100)
    d = np.zeros((20, 20), bool)
    d[np.arange(20), np.arange(20)] = True                   # a diagonal strand: one piece, mostly empty box
    assert np.isclose(mask_metrics(d)[3], 20 / 400)
    assert mask_metrics(np.zeros((5, 5), bool))[0] == 0.0
    assert len(mask_metrics(np.zeros((5, 5), bool))) == 6


def test_crop_metrics_reads_alpha_and_survives_bad_files(tmp_path):
    im = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    im.paste((255, 0, 0, 255), (0, 0, 5, 10))
    im.save(tmp_path / "c.png")
    assert crop_metrics(str(tmp_path / "c.png"))[:2] == (50.0, 1.0)
    (tmp_path / "bad.png").write_bytes(b"x")
    assert np.isnan(crop_metrics(str(tmp_path / "bad.png"))).all()
    assert np.isnan(crop_metrics(str(tmp_path / "missing.png"))).all()


def test_gate_is_per_type_and_pairs_use_top2():
    types = np.array([TYPE_ID["dress"], TYPE_ID["dress"], TYPE_ID["shoe"], TYPE_ID["dress"],
                      TYPE_ID["belt"], TYPE_ID["dress"], TYPE_ID["dress"], -1, TYPE_ID["dress"]])
    m = _metrics([(500, 1.0, 1.0, 0.9),          # clean dress
                  (500, 0.6, 1.0, 0.9),          # dress split in two (collage): fails
                  (500, 0.6, 1.0, 0.9, 0.5, 0.3),  # two shoes: a pair, passes on top2; gaps allowed
                  (50, 1.0, 1.0, 0.9),           # tiny dress: fails the default area
                  (50, 1.0, 1.0, 0.9),           # same size belt: passes its own lower bar
                  (500, 1.0, 1.0, 0.2),          # specks: fails fill
                  (np.nan,) * 4,                 # unreadable: fails
                  (500, 1.0, 1.0, 0.9),          # unknown type: fails
                  (100, 0.85, 0.9, 0.35)])       # exactly on every bar: passes
    assert crop_ok(m, types, GATE).tolist() == [True, False, True, False, True, False, False, False, True]
    more = _metrics([(500, 1.0, 1.0, 0.9, 0.1, 0.0),               # two pieces in opposite corners: low cover
                     (500, 1.0, 1.0, 0.9, 0.5, 0.3),               # hollowed out: too many holes
                     (500, 1.0, 1.0, 0.9, 0.2, 0.1),               # on both bars: passes
                     (500, 1.0, 1.0, 0.9, 0.5, 0.0, 0.01, 0.1),    # CLIP says it is not a dress: fails
                     (500, 1.0, 1.0, 0.9, 0.5, 0.0, 0.9, 0.8),     # caption text: fails
                     (500, 1.0, 1.0, 0.9, 0.5, 0.0, 0.05, 0.5),    # on both CLIP bars: passes
                     (500, 1.0, 1.0, 0.9, 0.5, 0.0, 0.01, 0.1),    # boots CLIP misreads: no label bar for shoes
                     (500, 1.0, 1.0, 0.9, 0.5, 0.0, 0.9, 0.1, 0.6),   # a blanket labelled "dress": fails
                     (500, 1.0, 1.0, 0.9, 0.5, 0.0, 0.9, 0.1, 0.6)])  # a belt CLIP reads as a rug: passes
    types = np.r_[np.full(6, TYPE_ID["dress"]), TYPE_ID["shoe"], TYPE_ID["dress"], TYPE_ID["belt"]]
    assert crop_ok(more, types, GATE).tolist() == [False, False, True, False, False, True, True, False, True]
    # product photos skip the CLIP checks, not the shape ones
    product = np.ones(len(types), bool)
    assert crop_ok(more, types, GATE, shape_only=product).tolist() == [False, False, True, True, True, True, True, True, True]


def test_clip_scores_own_group_and_text():
    texts, group = prompts()
    text_emb = np.eye(len(texts), 512, dtype=np.float32)          # one axis per prompt
    dress, gown, boots = (texts.index(f"a photo of {p}") for p in ("a dress", "a gown", "boots"))
    caption = texts.index("a photo of a caption written on an image")
    assert group[boots] == list(GROUP_PROMPTS).index("shoe") and group[caption] == TEXT
    emb = np.zeros((5, 512), np.float32)
    emb[0, boots] = 1.0                                           # boots, labelled shoe ...
    emb[1, boots] = 1.0                                           # ... and labelled "top"
    emb[2, caption] = 1.0                                         # a caption
    emb[3, dress] = 2.0                                           # unnormalised, unknown type
    emb[4, [dress, gown]] = 1.0                                   # between two dress prompts
    s = clip_scores(emb, np.array([TYPE_ID["shoe"], TYPE_ID["top"], TYPE_ID["dress"], -1, TYPE_ID["dress"]]),
                    text_emb, group, chunk=3)
    assert s["label_p"][0] > 0.99 and s["label_p"][1] < 0.01 and np.isnan(s["label_p"][3])
    assert s["label_p"][4] > 0.99                                 # the group's prompts add up
    assert s["text_p"][2] > 0.99 and s["text_p"][0] < 0.01
    junk = np.zeros((1, 512), np.float32)
    junk[0, texts.index("a photo of a pile of fabric scraps")] = 1.0
    j = clip_scores(junk, np.array([TYPE_ID["sweater"]]), text_emb, group)
    assert j["junk_p"][0] > 0.99 and s["junk_p"][0] < 0.01
