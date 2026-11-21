"""Pure ranking helpers: combined gain (D3), piece selection rules (D2), style-diverse top-n (D8c)."""
from __future__ import annotations

import numpy as np

RULES = ("max_gain", "topk_mean_gain", "slot_percentile")


def combine(d_h, d_t, w_h: float, w_t: float) -> np.ndarray:
    return w_h * np.asarray(d_h, np.float64) + w_t * np.asarray(d_t, np.float64)


def selection_score(g: np.ndarray, rule: str, k: int = 5) -> float:
    """Summarise one piece's candidate gains. slot_percentile = share of candidates beating the
    current piece (gain > 0)."""
    if rule not in RULES:
        raise ValueError(f"unknown selection rule {rule!r}; expected one of {RULES}")
    if g.size == 0:
        return -np.inf
    if rule == "max_gain":
        return float(g.max())
    if rule == "topk_mean_gain":
        kk = min(k, g.size)
        return float(np.partition(g, g.size - kk)[g.size - kk:].mean())
    return float((g > 0).mean())


def cluster_keys(broad, cluster) -> np.ndarray:
    """One integer per (broad category, M3 cluster); unmatched garments share one bucket per broad."""
    return np.asarray(broad, np.int64) * 1000 + np.asarray(cluster, np.int64) + 1


def diverse_top(g: np.ndarray, keys: np.ndarray, qualifies: np.ndarray, n: int = 3,
                groups=None, max_per_group: int | None = None) -> list:
    """#1 = best gain; then the best qualifying candidates from not-yet-used clusters; then fill by gain.
    groups / max_per_group: at most that many picks per group (refresh: per worn piece) while other
    groups still have candidates; only once they run out is the cap lifted to fill the n slots."""
    order = np.argsort(-np.asarray(g), kind="stable")
    if order.size == 0:
        return []
    grp = np.zeros(order.size, np.int64) if groups is None else np.asarray(groups)
    cap = order.size if max_per_group is None else int(max_per_group)
    picks = [int(order[0])]
    seen = {int(keys[order[0]])}
    used = {int(grp[order[0]]): 1}
    room = lambda j: used.get(int(grp[j]), 0) < cap

    def take(j):
        picks.append(int(j))
        seen.add(int(keys[j]))
        used[int(grp[j])] = used.get(int(grp[j]), 0) + 1

    for j in order[1:]:
        if len(picks) == n:
            break
        if qualifies[j] and int(keys[j]) not in seen and room(j):
            take(j)
    for capped in (True, False):
        for j in order[1:]:
            if len(picks) == n:
                break
            if int(j) not in picks and (room(j) or not capped):
                take(j)
    return picks
