"""
test_store_mode_parity.py  (Module 4, Phase 0 step 0.2 guard)

Store mode must produce exactly what the Fashion144k feature store was built from. This re-runs a few
Fashion144k photos through the same per-photo path the Pinterest store mode uses
(Module2/01_segment_batch.segment_photos + relativise_crop_paths) into a temporary folder, rebuilds
their feature rows with Module 2's own node builder, and compares them with the existing store rows.

Re-running Module 1 is close but not bit-exact (fp16 GPU inference): on 40 photos, embeddings kept
cosine >= 0.996, areas within 0.4%, colour L1 within 0.01, formality within 3 points, about 5% of
garments flipped one zero-shot attribute and 1 outfit in 40 gained or lost a garment. So the checks are
statistical: they still fail at once on a different code path (other crops, missing attributes,
another embedding), which is what this guards against.

Needs the Fashion144k data and a GPU (skipped otherwise). Run: pytest test_store_mode_parity.py
"""
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
F144 = os.path.join(HERE, "..", "Fashion144k_v1")
KEYS = tuple(range(20))

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(F144, "feature_store", "index.json")),
                                reason="Fashion144k data not available")


def test_rerun_reproduces_fashion144k_store_rows(tmp_path):
    from pinterest_garment_segmentation import _load_segment_batch, relativise_crop_paths
    sb = _load_segment_batch()
    od = sb_od()
    with open(os.path.join(F144, "photos.txt"), encoding="utf-8", errors="replace") as f:
        photos = [ln.strip() for ln in f if ln.strip()]
    subset = [photos[k] for k in KEYS]
    out_root = str(tmp_path / "outputs")
    sb.segment_photos(subset, os.path.join(F144, "photos"), out_root, str(tmp_path / "ckpt.txt"),
                      after_each=relativise_crop_paths)

    with open(os.path.join(F144, "feature_store", "index.json"), encoding="utf-8") as f:
        index = json.load(f)["outfits"]
    feat = np.load(os.path.join(F144, "feature_store", "features.npy"), mmap_mode="r")
    scal = np.load(os.path.join(F144, "feature_store", "node_scalars.npy"), mmap_mode="r")
    same_set, emb, col, area, form, attr_flip = 0, [], [], [], [], []
    for i, key in enumerate(KEYS):
        d = os.path.join(out_root, str(i))
        meta = os.path.join(d, "metadata.json")
        if not os.path.exists(meta):
            continue
        with open(meta, encoding="utf-8") as f:
            garments = {g["file"]: g for g in json.load(f)}
        assert all(not os.path.isabs(g["crop_path"]) for g in garments.values())
        stored = {e["file"]: e for e in index[str(key)]}
        if set(garments) != set(stored):
            continue
        same_set += 1
        for name, ent in stored.items():
            node = od.build_node_from_meta(d, garments[name])
            r = ent["row"]
            a = node["embedding"] / np.linalg.norm(node["embedding"])
            b = feat[r, :512] / np.linalg.norm(feat[r, :512])
            emb.append(float(a @ b))
            col.append(float(np.abs(node["colour"] - feat[r, 512:544]).sum()))
            area.append(abs(node["area"] - scal[r, 2]) / max(float(scal[r, 2]), 1.0))
            form.append(abs(node["formality"] - scal[r, 1]))
            attr_flip.append(float(np.abs(node["attr_feat"] - feat[r, 544:]).max()) > 0.5)
    assert same_set >= 0.9 * len(KEYS), f"only {same_set}/{len(KEYS)} outfits kept the same garments"
    assert min(emb) > 0.99 and np.median(emb) > 0.999, "CLIP embeddings differ"
    assert max(col) < 0.05, "colour histograms differ"
    assert max(area) < 0.02, "garment areas differ"
    assert np.percentile(form, 95) < 3.0, "formality differs"
    assert np.mean(attr_flip) <= 0.15, "zero-shot attributes differ too often"


def sb_od():
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("outfit_dataset", os.path.join(HERE, "..", "Module2", "02_outfit_dataset.py"))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
