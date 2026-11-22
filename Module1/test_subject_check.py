"""
test_subject_check.py
=====================
Who is in the photo: a person, an animal, both or neither (Review 1: "evaluate how the pipeline
behaves on non-human subjects"). An animal with no person stops the pipeline; a pet next to a
person is masked out so its clothes are not counted as the person's.
"""

import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fashion_segmenter
import run_pipeline
from subject_check import SubjectCheck, animal_only_mask, decide

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _box(shape, y0, y1, x0, x1):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def test_decide():
    assert decide(persons=0, animals=["dog"]) == "animal_no_person"
    assert decide(persons=1, animals=["cat"]) == "person_with_animal"
    assert decide(persons=2, animals=[]) == "ok"
    assert decide(persons=0, animals=[]) == "ok"        # flat-lays and product shots are not blocked


def test_animal_only_mask_keeps_pixels_the_person_covers():
    shape = (10, 10)
    dog = _box(shape, 5, 10, 0, 6)
    person = _box(shape, 0, 10, 4, 10)
    m = animal_only_mask([person], [dog], shape)
    assert m.sum() == 5 * 4                               # dog pixels left of the person only
    assert not (m & person).any()
    assert animal_only_mask([person], [], shape) is None


def test_animal_mask_grows_past_its_outline_but_not_into_the_person():
    shape = (20, 20)
    dog = _box(shape, 5, 10, 5, 10)
    person = _box(shape, 0, 20, 12, 20)
    m = animal_only_mask([person], [dog], shape, grow=3)
    assert m[4, 7] and m[7, 4] and m[7, 11]               # outline pixels SegFormer leaks onto
    assert not m[7, 13] and not (m & person).any()
    assert not m[0, 0]


def _stub_models(monkeypatch, pred):
    labels = {0: "Background", 4: "Upper-clothes"}
    monkeypatch.setattr(fashion_segmenter, "segment_clothing", lambda image, device=None: (pred.copy(), labels))
    monkeypatch.setattr(fashion_segmenter, "get_models", lambda device=None: (None, None, None, None, "cpu"))
    monkeypatch.setattr(fashion_segmenter, "classify_crop_clip", lambda crop, segformer_label, device=None: ("sweater", 0.9))
    monkeypatch.setattr(fashion_segmenter, "analyze_garment", lambda crop, **kw: {})


def test_extract_drops_excluded_pixels(monkeypatch):
    pred = np.zeros((40, 40), np.int64)
    pred[0:40, 0:40] = 4                                  # one "Upper-clothes" region over the whole photo
    _stub_models(monkeypatch, pred)
    image = Image.new("RGB", (40, 40))
    full = fashion_segmenter.extract_garment_segments(image, min_pixel_area=1)
    pet = _box(pred.shape, 20, 40, 0, 40)
    kept = fashion_segmenter.extract_garment_segments(image, min_pixel_area=1, exclude_mask=pet)
    assert full[0]["pixel_count"] == 1600 and kept[0]["pixel_count"] == 800


def _stub_pipeline(monkeypatch, check):
    calls = {}
    monkeypatch.setattr(run_pipeline, "check_subjects", lambda image_path, device=None: check)

    def fake_process(**kw):
        calls["process"] = kw
        return {"output_dir": "x", "garments": [], "num_garments": 0}

    monkeypatch.setattr(run_pipeline, "process_image_segformer", fake_process)
    monkeypatch.setattr(run_pipeline, "extract_folder_embeddings", lambda **kw: [])
    return calls


def test_pipeline_stops_on_an_animal_with_no_person(monkeypatch, tmp_path):
    calls = _stub_pipeline(monkeypatch, SubjectCheck(persons=0, animals=["dog"], exclude_mask=None))
    res = run_pipeline.run_single_image("dog.jpg", image_id="dog", output_root=str(tmp_path))
    assert res["status"] == "animal_no_person" and res["num_garments"] == 0
    assert "an animal" in res["message"] and "dog" not in res["message"]   # COCO species names are unreliable
    assert "process" not in calls


def test_pipeline_masks_out_a_pet_next_to_a_person(monkeypatch, tmp_path):
    pet = np.ones((4, 4), bool)
    calls = _stub_pipeline(monkeypatch, SubjectCheck(persons=1, animals=["dog"], exclude_mask=pet))
    res = run_pipeline.run_single_image("walk.jpg", image_id="walk", output_root=str(tmp_path))
    assert res["status"] == "person_with_animal"
    assert calls["process"]["exclude_mask"] is pet


def test_pipeline_unchanged_without_an_animal(monkeypatch, tmp_path):
    calls = _stub_pipeline(monkeypatch, SubjectCheck(persons=1, animals=[], exclude_mask=None))
    res = run_pipeline.run_single_image("me.jpg", image_id="me", output_root=str(tmp_path))
    assert res["status"] == "ok" and calls["process"]["exclude_mask"] is None


@pytest.mark.parametrize("photo, status", [("samples/nonhuman/golden_retriever_grass.jpg", "animal_no_person"),
                                           ("samples/nonhuman/woman_walking_dog.jpg", "person_with_animal"),
                                           ("samples/image2.jpg", "ok")])
def test_real_photos(photo, status):
    path = os.path.join(REPO, photo)
    if not os.path.exists(path):
        pytest.skip(f"{photo} is local only")
    from subject_check import check_subjects
    assert decide(**{k: v for k, v in vars(check_subjects(path)).items() if k != "exclude_mask"}) == status
