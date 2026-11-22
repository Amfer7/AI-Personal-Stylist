"""Photo batch runner: one result row per photo and the summary table."""
from types import SimpleNamespace as NS

import pandas as pd

from photo_batch import result_row, summary


def _rec(verdict, swap=None, sugg=()):
    return NS(verdict=verdict, baseline={"harmony_pct": 71.234},
              swap_piece={"file": swap} if swap else None,
              suggestions=[NS(from_type=a, to_type=b) for a, b in sugg])


def test_result_row_for_a_recommendation():
    r = result_row("me.jpg", "ok", ["t_shirt_0.png", "pants_0.png"],
                   _rec("swap", "pants_0.png", [("pants", "skirt")]), 1.234)
    assert r == {"photo": "me.jpg", "status": "ok", "garments": "t_shirt, pants", "n_garments": 2,
                 "harmony": 71.2, "verdict": "swap", "swap_piece": "pants", "suggestions": "pants→skirt",
                 "seconds": 1.2, "note": ""}


def test_result_row_when_the_pipeline_stops():
    r = result_row("dog.jpg", "animal_no_person", [], None, 0.4, note="No person found, only an animal.")
    assert r["verdict"] == "" and r["harmony"] is None and r["n_garments"] == 0
    assert r["note"] == "No person found, only an animal."


def test_summary_counts_statuses_and_verdicts():
    rows = pd.DataFrame([{"status": "ok", "verdict": "swap"}, {"status": "ok", "verdict": "refresh"},
                         {"status": "ok", "verdict": "refresh"}, {"status": "no_garments", "verdict": ""}])
    s = summary(rows)
    assert s["photos"] == 4
    assert s["status"] == {"ok": 3, "no_garments": 1}
    assert s["verdict"] == {"refresh": 2, "swap": 1}
