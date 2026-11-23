"""Weak-label recall of Module 1 on Fashion144k titles: tokeniser, keyword map, scoring."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_weak_labels import evaluate, title_labels, tokens, wilson

NAME = "10000078429-tawny-stilettos-vintage-shoes-red-polyester-vintage-dress-beige-beaded-purse_400.jpg"


def test_tokens_drop_id_and_suffix_and_join_t_shirt():
    assert tokens(NAME)[:3] == ["tawny", "stilettos", "vintage"]
    assert tokens("1-white-t-shirt-black-jeans_400.jpg") == ["white", "tshirt", "black", "jeans"]
    assert tokens("6475795322-6475795322_400.jpg") == ["6475795322"]


def test_title_labels_map_words_and_ignore_brands_and_part_words():
    lab = title_labels(NAME)
    assert lab["types"] == {"shoe", "dress", "bag"} and lab["groups"] == {"shoe", "dress", "bag"}
    lab = title_labels("2-topshop-white-t-shirt-tights-handbags_400.jpg")
    assert lab["types"] == {"t_shirt"}                 # "topshop" is not "top"; "handbags" not listed
    assert lab["legwear"]
    assert title_labels("3-grey-hoodie_400.jpg") == {"types": set(), "groups": {"upper"}, "legwear": False}


def test_evaluate_exact_group_and_misses():
    names = ["1-black-jacket-jeans_400.jpg", "2-red-jacket_400.jpg", "3-blue-dress_400.jpg"]
    found = [["sweater", "pants"], ["jacket"], None]          # photo 3 had no Module 1 output
    r = evaluate(names, found)
    assert r["exact"]["jacket"]["n"] == 2 and r["exact"]["jacket"]["hits"] == 1
    assert r["group"]["upper"]["recall"] == 1.0
    assert r["exact"]["pants"]["recall"] == 1.0
    assert "dress" not in r["exact"]
    assert r["misses"]["jacket"] == {"sweater": 1}


def test_wilson_interval():
    lo, hi = wilson(50, 100)
    assert lo == pytest.approx(0.404, abs=1e-3) and hi == pytest.approx(0.596, abs=1e-3)
