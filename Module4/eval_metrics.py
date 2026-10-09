"""Pure helpers for the Module 4 evaluation (spec §6): per-outfit summaries over the (rule, weight)
grid, threshold calibration, the Pareto weight rule, bootstrap CIs and paired tests.

The trend weight is a mix: gain = (1 - w)·ΔH + w·ΔT, so w = 0 is harmony-only and w = 1 is
trend-only (the two ablations of spec §6.2 item 4).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest

from selection import RULES, combine, selection_score


def grid_summary(pieces, ws, topk: int) -> dict:
    """For every selection rule × weight: the piece decide() would pick, its selection score, and the
    best candidate of that piece (row, ΔH, ΔT, logit). Mirrors Recommender.decide(): stable ordering,
    so ties go to the first piece / first candidate. pieces: objects with d_h, d_t, rows, logits."""
    R, W, m = len(RULES), len(ws), len(pieces)
    scores = np.full((R, W, m), -np.inf)
    best_k = np.full((W, m), -1, np.int64)
    for j, ps in enumerate(pieces):
        if ps.d_h.size == 0:
            continue
        for wi, w in enumerate(ws):
            g = combine(ps.d_h, ps.d_t, 1.0 - w, w)
            best_k[wi, j] = int(np.argmax(g))
            for ri, rule in enumerate(RULES):
                scores[ri, wi, j] = selection_score(g, rule, topk)
    out = {k: np.full((R, W), np.nan) for k in ("top_dh", "top_dt", "top_logit")}
    out["chosen"] = np.full((R, W), -1, np.int64)
    out["top_row"] = np.full((R, W), -1, np.int64)
    out["s_star"] = np.full((R, W), -np.inf)
    for ri in range(R):
        for wi in range(W):
            s = scores[ri, wi]
            if m == 0 or not np.isfinite(s).any():
                continue
            j = int(np.argmax(s))
            k = best_k[wi, j]
            ps = pieces[j]
            out["chosen"][ri, wi] = j
            out["s_star"][ri, wi] = s[j]
            out["top_row"][ri, wi] = int(ps.rows[k])
            out["top_dh"][ri, wi] = ps.d_h[k]
            out["top_dt"][ri, wi] = ps.d_t[k]
            out["top_logit"][ri, wi] = ps.logits[k]
    out["piece_scores"] = scores
    return out


def balanced_threshold(real, corrupted):
    """τ maximising balanced accuracy for "swap iff score ≥ τ" (real outfits should not swap, corrupted
    ones should). Thresholds are midpoints between observed scores; among equally good thresholds the
    middle one is taken. Returns (tau, balanced_accuracy, false_alarm_rate, detection_rate)."""
    real = np.sort(np.asarray(real, np.float64))
    corr = np.sort(np.asarray(corrupted, np.float64))
    u = np.unique(np.concatenate([real, corr]))
    u = u[np.isfinite(u)]
    if u.size == 0:
        return 0.0, 0.5, 0.0, 0.0
    cands = np.concatenate([[u[0] - 1.0], (u[:-1] + u[1:]) / 2, [u[-1] + 1.0]])
    fpr = 1.0 - np.searchsorted(real, cands, "left") / real.size
    tpr = 1.0 - np.searchsorted(corr, cands, "left") / corr.size
    ba = (tpr + 1.0 - fpr) / 2
    best = np.flatnonzero(ba == ba.max())
    i = int(best[len(best) // 2])
    return float(cands[i]), float(ba[i]), float(fpr[i]), float(tpr[i])


def pareto_weight(ws, mean_dh, keep: float = 0.9) -> float:
    """Largest weight, walking up from w = 0, whose mean harmony gain stays ≥ keep × the w = 0 gain."""
    ws, mean_dh = np.asarray(ws), np.asarray(mean_dh)
    order = np.argsort(ws)
    base = mean_dh[order[0]]
    if not base > 0:
        return float(ws[order[0]])
    best = ws[order[0]]
    for i in order[1:]:
        if mean_dh[i] < keep * base:
            break
        best = ws[i]
    return float(best)


def bootstrap_ci(x, stat, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap over the first axis (outfits). For paired statistics pass a stacked array."""
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    vals = np.array([stat(x[i]) for i in idx], np.float64)
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def mcnemar_p(a, b) -> float:
    """Exact McNemar test on paired binary outcomes (two-sided)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(binomtest(n10, n01 + n10, 0.5).pvalue)


def severity_strata(severity) -> np.ndarray:
    """0 / 1 / 2 = subtle / medium / obvious thirds of the corruption's own harmony drop."""
    s = np.asarray(severity)
    ranks = np.argsort(np.argsort(s, kind="stable"), kind="stable")
    return (ranks * 3 // len(s)).astype(np.int64)
