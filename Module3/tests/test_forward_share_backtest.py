"""Unit tests for forward_share_backtest.py (no data needed). Run: pytest test_forward_share_backtest.py"""
import numpy as np
import pandas as pd

import os
import sys
_MODULE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Module3
sys.path.insert(0, os.path.join(_MODULE, "scripts"))
sys.path.insert(0, os.path.join(_MODULE, "src"))

from forward_share_backtest import decide, forward_window, shift_month


def test_months():
    assert shift_month("2026-01", 0) == "2026-01"
    assert shift_month("2025-12", 1) == "2026-01"
    assert shift_month("2026-03", -4) == "2025-11"
    assert forward_window("2026-05", 3, "2026-08") == ["2026-05", "2026-06", "2026-07"]
    assert forward_window("2026-05", 6, "2026-08") is None


def _blocks(rhos_by_weight, n=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n):
        for w, mu in rhos_by_weight.items():
            rows.append({"category": "top", "cutoff": f"c{b}", "horizon": 3, "weight": w,
                         "rho": mu + 0.05 * rng.standard_normal()})
    return pd.DataFrame(rows)


def test_clear_winner_is_chosen():
    d = decide(_blocks({0.6: 0.5, 0.3: 0.3, 0.0: 0.1, 1.0: 0.4}))
    assert d["best"] == 0.6 and d["chosen_growth_weight"] == 0.6 and not d["tie_with_recency_only"]


def test_tie_goes_to_recency_only():
    d = decide(_blocks({0.6: 0.101, 0.3: 0.1, 0.0: 0.1, 1.0: 0.0}, n=6))
    assert d["chosen_growth_weight"] == 0.0


def test_recency_only_best():
    d = decide(_blocks({0.6: -0.2, 0.3: 0.0, 0.0: 0.3, 1.0: -0.4}))
    assert d["best"] == 0.0 and d["chosen_growth_weight"] == 0.0
