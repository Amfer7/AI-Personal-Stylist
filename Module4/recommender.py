"""Module 4 recommender: which one piece to swap, and what for.

score(state)  -> SwapTable : every allowed candidate for every piece through the GNN (D4, D5, D8b),
                             cached as standardised ΔH and ΔT (D3). The only GNN calls.
decide(table) -> Recommendation : selection rule (D2), threshold + verdicts (D7), style-diverse
                             top 3 (D8c), output shape (D6). Pure NumPy, so the trend-weight slider and
                             the evaluation sweeps never re-run the GNN.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from candidate_pool import CandidatePool, pool_dir
from dense_gnn import DenseGNN
from explain import what_changed
from harmony_reference import HarmonyReference
from m2_bridge import apply_vram_cap, load_gnn, load_trend_model, pick_device
from m4_settings import M4Settings
from outfit_state import load_outfit_dir
from recommendation import Recommendation, Suggestion
from selection import cluster_keys, combine, diverse_top, selection_score
from swap_groups import CooccurrenceFilter, replacement_types
from swap_scorer import SwapScorer


def _num(x):
    return None if x is None or not np.isfinite(x) else float(x)


@dataclass
class PieceSwaps:
    rows: np.ndarray           # candidate pool rows
    logits: np.ndarray         # raw GNN logit of each what-if outfit
    d_h: np.ndarray            # (logit - r0) / sigma_logit
    d_t: np.ndarray            # (trend(cand) - trend(piece)) / sigma_trend, 0 where unknown
    trend_known: np.ndarray
    flags: list


@dataclass
class SwapTable:
    state: object
    r0: float
    pieces: list               # pieces[i] belongs to state.pieces[i]
    mode: str


class Recommender:
    def __init__(self, settings, pool, scorer, reference, cooc, gnn=None, trend_model=None):
        self.settings, self.pool, self.scorer = settings, pool, scorer
        self.reference, self.cooc = reference, cooc
        self.gnn, self.trend_model = gnn, trend_model

    @classmethod
    def from_settings(cls, settings=None, device=None, pool=None, gnn=None) -> "Recommender":
        s = settings or M4Settings.load()
        dev = pick_device(device)
        apply_vram_cap(s.vram_gb, dev)
        gnn = gnn or DenseGNN(load_gnn(s.abs(s.checkpoint), dev), dev, s.vram_gb)
        pool = pool or CandidatePool.load(s, dev)
        d = pool_dir(s)
        return cls(s, pool, SwapScorer(pool, gnn), HarmonyReference.load(os.path.join(d, "reference.json")),
                   CooccurrenceFilter.load(os.path.join(d, "cooc.npy"), s.cooc_min), gnn=gnn,
                   trend_model=load_trend_model(s.abs(s.trend_model)))

    # ---------------- GNN stage ----------------
    def _candidates(self, state, i, mode, exclude, same_type, trend_floor):
        p, flags = state.pieces[i], []
        types = replacement_types(p.type, same_type)
        if not types:
            return self.pool.empty_rows(), [f"type '{p.type}' has no swap group: not swapped"]
        others = [q.type for j, q in enumerate(state.pieces) if j != i]
        types = [t for t in types if self.cooc.allowed(t, others)]
        rows = self.pool.candidate_rows(types, mode=mode, exclude=exclude)
        if len(rows):
            rows = rows[~self.pool.dup_mask(rows, p.emb, p.colour, p.area)]          # D8b: not your own item
        if trend_floor and not np.isnan(p.trend) and len(rows):
            t = self.pool.trend_t[rows]
            keep = torch.isnan(t) | (t >= p.trend)
            if bool(keep.any()):
                rows = rows[keep]
            else:
                flags.append("trend floor left no candidates: floor relaxed")
        if not len(rows):
            flags.append("no allowed candidates: piece skipped")
        return rows, flags

    def score(self, state, mode="demo", exclude=None, same_type=None, trend_floor=None) -> SwapTable:
        s = self.settings
        same_type = s.same_type if same_type is None else same_type
        trend_floor = s.trend_floor if trend_floor is None else trend_floor
        r0 = self.scorer.outfit_logit(state)
        out = []
        for i, p in enumerate(state.pieces):
            rows, flags = self._candidates(state, i, mode, exclude, same_type, trend_floor)
            logits = self.scorer.swap_logits(state, i, rows) if len(rows) else np.zeros(0, np.float32)
            rows_np = rows.cpu().numpy()
            ct = self.pool.trend[rows_np].astype(np.float64)
            known = ~np.isnan(ct) & (not np.isnan(p.trend))
            d_t = np.where(known, (ct - p.trend) / self.reference.sigma_trend, 0.0)
            if np.isnan(p.trend):
                flags.append("worn piece has no trend score: ΔT = 0")
            out.append(PieceSwaps(rows_np, logits, (logits - r0) / self.reference.sigma_logit, d_t, known, flags))
        return SwapTable(state, r0, out, mode)

    # ---------------- decision stage (no GNN) ----------------
    def _suggestion(self, table, i, k, gain) -> Suggestion:
        ps, p, pool = table.pieces[i], table.state.pieces[i], self.pool
        row = int(ps.rows[k])
        others = [q.describe() for j, q in enumerate(table.state.pieces) if j != i]
        return Suggestion(
            piece_file=p.file, pool_row=row, source=pool.source_name(row), outfit_id=pool.outfit_key(row),
            file=pool.files[row], crop_path=pool.crop_path(row), from_type=p.type, to_type=pool.type_name(row),
            harmony_pct_before=self.reference.pct(table.r0), harmony_pct_after=self.reference.pct(float(ps.logits[k])),
            trend_before=_num(p.trend), trend_after=_num(pool.trend[row]), trend_known=bool(ps.trend_known[k]),
            d_harmony_z=float(ps.d_h[k]), d_trend_z=float(ps.d_t[k]), gain=float(gain),
            what_changed=what_changed(p.describe(), pool.describe(row), others))

    def decide(self, table, w_t=None, rule=None) -> Recommendation:
        s = self.settings
        w_t = s.w_t if w_t is None else float(w_t)
        rule = rule or s.rule
        tau, eps = float(s.tau[rule]), float(s.eps)
        pieces = table.state.pieces
        G = [combine(ps.d_h, ps.d_t, s.w_h, w_t) for ps in table.pieces]
        S = [selection_score(g, rule, s.topk) for g in G]
        per_piece = {}
        for p, ps, g, sc in zip(pieces, table.pieces, G, S):
            per_piece[p.file] = {
                "type": p.type, "group": p.group, "n_candidates": int(g.size), "selection_score": _num(sc),
                "best_gain": _num(g.max()) if g.size else None,
                "topk_mean_gain": _num(selection_score(g, "topk_mean_gain", s.topk)),
                "slot_percentile": _num(selection_score(g, "slot_percentile")),
                "best_dH": _num(ps.d_h.max()) if g.size else None,
                "best_dT": _num(ps.d_t.max()) if g.size else None,
                "clears_tau": bool(np.isfinite(sc) and sc >= tau), "trend": _num(p.trend), "flags": list(ps.flags)}
        order = sorted(range(len(S)), key=lambda j: S[j], reverse=True)
        verdict, swap_piece, suggestions, also = "keep", None, [], []
        if order and np.isfinite(S[order[0]]) and S[order[0]] >= tau:
            verdict, j = "swap", order[0]
            g, ps, p = G[j], table.pieces[j], pieces[j]
            swap_piece = {"file": p.file, "category": p.type, "group": p.group}
            qualifies = g > 0 if rule == "slot_percentile" else g >= tau
            keys = cluster_keys(self.pool.broad[ps.rows], self.pool.cluster[ps.rows])
            suggestions = [self._suggestion(table, j, k, g[k]) for k in diverse_top(g, keys, qualifies)]
            for o in order[1:]:
                if np.isfinite(S[o]) and S[o] >= tau:
                    k = int(np.argmax(G[o]))
                    also.append({"piece": pieces[o].file, "selection_score": float(S[o]),
                                 "suggestion": self._suggestion(table, o, k, G[o][k])})
        else:
            js, ks, gs, rs = [], [], [], []
            for j, ps in enumerate(table.pieces):
                ok = np.nonzero(ps.trend_known & (ps.d_t > 0) & (ps.d_h >= -eps))[0]
                js.append(np.full(ok.size, j))
                ks.append(ok)
                gs.append(G[j][ok])
                rs.append(ps.rows[ok])
            if js and sum(k.size for k in ks):
                verdict = "refresh"
                js, ks, gs, rows = (np.concatenate(x) for x in (js, ks, gs, rs))
                keys = cluster_keys(self.pool.broad[rows], self.pool.cluster[rows])
                suggestions = [self._suggestion(table, int(js[x]), int(ks[x]), gs[x])
                               for x in diverse_top(gs, keys, np.ones(gs.size, bool))]
        known = [p.trend for p in pieces if not np.isnan(p.trend)]
        return Recommendation(
            verdict=verdict, selection_rule=rule, pool_mode=table.mode, weights={"w_h": s.w_h, "w_t": w_t},
            thresholds={"tau": tau, "eps": eps},
            baseline={"harmony_pct": self.reference.pct(table.r0), "harmony_logit": float(table.r0),
                      "trend_mean": float(np.mean(known)) if known else None},
            swap_piece=swap_piece, suggestions=suggestions, also_improve=also, per_piece=per_piece,
            flags=list(table.state.flags), provisional=s.provisional())

    def recommend(self, outfit_dir, mode="demo", photo_px=None, **score_kw) -> Recommendation:
        state = load_outfit_dir(outfit_dir, self.gnn, self.trend_model, photo_px=photo_px)
        return self.decide(self.score(state, mode=mode, **score_kw))
