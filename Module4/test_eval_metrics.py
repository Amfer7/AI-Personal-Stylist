import numpy as np
import pytest

from eval_metrics import (balanced_threshold, bootstrap_ci, grid_summary, mcnemar_p, pareto_weight,
                          severity_strata)
from selection import RULES


def test_balanced_threshold_separates_clean_scores():
    real = np.array([0.1, 0.2, 0.3])
    corr = np.array([0.6, 0.7, 0.8])
    tau, ba, fpr, tpr = balanced_threshold(real, corr)
    assert 0.3 < tau < 0.6
    assert ba == 1.0 and fpr == 0.0 and tpr == 1.0


def test_balanced_threshold_with_overlap_and_infinities():
    real = np.array([-np.inf, 0.0, 0.5, 0.9])
    corr = np.array([0.4, 0.8, 1.0, 1.2])
    tau, ba, fpr, tpr = balanced_threshold(real, corr)
    assert np.isfinite(tau)
    assert ba == pytest.approx((tpr + 1 - fpr) / 2)
    assert ba >= 0.75
    assert fpr == np.mean(real >= tau) and tpr == np.mean(corr >= tau)


def test_pareto_weight_picks_largest_weight_keeping_the_share():
    ws = np.array([0.0, 0.1, 0.2, 0.3])
    dh = np.array([1.0, 0.95, 0.91, 0.5])
    assert pareto_weight(ws, dh, keep=0.9) == 0.2
    assert pareto_weight(ws, np.array([1.0, 0.2, 0.95, 0.1]), keep=0.9) == 0.0   # first drop ends the run


def test_bootstrap_ci_brackets_the_mean_and_is_seeded():
    x = np.r_[np.zeros(50), np.ones(50)]
    lo, hi = bootstrap_ci(x, np.mean, n_boot=500, seed=1)
    assert lo < 0.5 < hi
    assert (lo, hi) == bootstrap_ci(x, np.mean, n_boot=500, seed=1)


def test_mcnemar_p():
    a = np.array([1] * 30 + [0] * 70, bool)
    assert mcnemar_p(a, a) == 1.0
    b = np.zeros(100, bool)
    assert mcnemar_p(a, b) < 1e-6


def test_severity_strata_are_thirds():
    s = severity_strata(np.arange(9.0))
    assert list(s) == [0, 0, 0, 1, 1, 1, 2, 2, 2]


class _PS:
    def __init__(self, d_h, d_t, rows):
        self.d_h, self.d_t, self.rows = np.asarray(d_h, float), np.asarray(d_t, float), np.asarray(rows)
        self.logits = self.d_h.copy()


def test_grid_summary_matches_hand_computation():
    pieces = [_PS([0.1, 0.5, 0.2], [1.0, -1.0, 0.0], [10, 11, 12]),
              _PS([0.4, 0.3], [0.0, 2.0], [20, 21]),
              _PS([], [], [])]
    ws = np.array([0.0, 1.0])
    out = grid_summary(pieces, ws, topk=2)
    r = RULES.index("max_gain")
    # w = 0: harmony only. piece 0 best 0.5, piece 1 best 0.4 -> piece 0, row 11
    assert out["chosen"][r, 0] == 0 and out["top_row"][r, 0] == 11
    assert out["s_star"][r, 0] == pytest.approx(0.5)
    assert out["top_dh"][r, 0] == pytest.approx(0.5) and out["top_dt"][r, 0] == pytest.approx(-1.0)
    # w = 1: trend only. piece 0 best 1.0, piece 1 best 2.0 -> piece 1, row 21
    assert out["chosen"][r, 1] == 1 and out["top_row"][r, 1] == 21
    k = RULES.index("topk_mean_gain")
    # w = 0, top-2 means: piece 0 (0.5+0.2)/2 = 0.35, piece 1 (0.4+0.3)/2 = 0.35 -> tie -> first
    assert out["chosen"][k, 0] == 0
    p = RULES.index("slot_percentile")
    # w = 1: share > 0: piece 0 1/3, piece 1 1/2 -> piece 1
    assert out["chosen"][p, 1] == 1 and out["s_star"][p, 1] == pytest.approx(0.5)
    assert out["piece_scores"].shape == (3, 2, 3)
    assert np.isneginf(out["piece_scores"][:, :, 2]).all()


def test_grid_summary_all_pieces_empty():
    out = grid_summary([_PS([], [], [])], np.array([0.0]), topk=5)
    assert (out["chosen"] == -1).all() and np.isneginf(out["s_star"]).all()
