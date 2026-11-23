"""
subject_check.py
================
Who is in the photo? A COCO instance-segmentation model (Ultralytics YOLO11m-seg) finds people and
animals before garments are parsed.

  - an animal and no person  -> "animal_no_person": the pipeline stops (SegFormer would otherwise read
                                fur and paws as clothes: an undressed golden retriever came out as a
                                "shirt_blouse" with harmony 100/100)
  - a person and an animal   -> "person_with_animal": pixels on the animal but not on a person are
                                excluded, so a pet's sweater is not counted as the owner's clothes
  - otherwise                -> "ok", unchanged (flat-lays and product shots are not blocked)

Measured 2026-11-22: fires on all 13 animal-only photos of samples/nonhuman/ and on 0 of 600 real
outfit photos (300 Fashion144k, 300 Pinterest). A whole-image CLIP "is this a person?" score was
rejected: it scores people in elaborate garments (saree, lehenga, sherwani) below 0.3.
Live path only (run_single_image); feature-store builds call process_image_segformer directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw

WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "yolo11m-seg.pt")
PERSON = 0
ANIMALS = {14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant",
           21: "bear", 22: "zebra", 23: "giraffe"}
CONF = 0.25
GROW = 0.01            # animal mask grown by 1% of the photo's longer side: SegFormer leaks past the outline

_MODEL = None


@dataclass
class SubjectCheck:
    persons: int
    animals: List[str]
    exclude_mask: Optional[np.ndarray]     # animal pixels no person covers; None when there is nothing to drop


def decide(persons: int, animals: List[str]) -> str:
    if animals and not persons:
        return "animal_no_person"
    if animals:
        return "person_with_animal"
    return "ok"


def animal_only_mask(person_masks, animal_masks, shape, grow: int = 0) -> Optional[np.ndarray]:
    if not animal_masks:
        return None
    animal = np.logical_or.reduce([np.asarray(m, bool) for m in animal_masks])
    if grow > 0:
        from scipy.ndimage import binary_dilation
        animal = binary_dilation(animal, iterations=grow)
    person = np.logical_or.reduce([np.asarray(m, bool) for m in person_masks]) if person_masks else np.zeros(shape, bool)
    return animal & ~person


def _get_model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
        _MODEL = YOLO(WEIGHTS)          # downloads the official weights on first use
    return _MODEL


def _polygon_mask(points, shape) -> np.ndarray:
    canvas = Image.new("L", (shape[1], shape[0]), 0)
    if len(points) >= 3:
        ImageDraw.Draw(canvas).polygon([tuple(p) for p in points], fill=1)
    return np.array(canvas, bool)


def check_subjects(image_path: str, device: Optional[str] = None) -> SubjectCheck:
    # CPU on purpose: the installed torchvision is a CPU build, so its NMS cannot take CUDA tensors;
    # one photo takes well under a second on CPU. `device` is accepted for a uniform call signature.
    res = _get_model().predict(image_path, conf=CONF, verbose=False, device="cpu")[0]
    shape = res.orig_shape
    classes = res.boxes.cls.int().tolist()
    polygons = res.masks.xy if res.masks is not None else [[] for _ in classes]
    persons = [_polygon_mask(p, shape) for c, p in zip(classes, polygons) if c == PERSON]
    animals = [(ANIMALS[c], _polygon_mask(p, shape)) for c, p in zip(classes, polygons) if c in ANIMALS]
    names = [n for n, _ in animals]
    grow = max(1, round(GROW * max(shape)))
    exclude = animal_only_mask(persons, [m for _, m in animals], shape, grow) if persons else None
    return SubjectCheck(persons=len(persons), animals=names, exclude_mask=exclude)
