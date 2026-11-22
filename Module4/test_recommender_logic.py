import numpy as np
import pytest
import torch

from candidate_pool import SPLIT_CODES, CandidatePool
from harmony_reference import HarmonyReference
from m2_bridge import BROAD_ID
from m4_settings import M4Settings, StoreSpec
from outfit_state import OutfitState, Piece
from recommendation import Recommendation, format_text
from recommender import Recommender
from swap_groups import BROAD_OF_TYPE, TYPE_ID, TYPES, CooccurrenceFilter

# pool row: (type, stub GNN logit, M3 cluster, trend)
ROWS = [
    ("t_shirt", 3.0, 5, 60.0),        # 0
    ("sweater", 2.9, 5, 40.0),        # 1  same cluster as row 0
    ("jacket", 2.0, 6, 70.0),         # 2
    ("top", 1.0, 7, np.nan),          # 3  no trend score
    ("shirt_blouse", 0.5, -1, 55.0),  # 4
    ("pants", 0.2, 1, 90.0),          # 5  blocked for the skirt slot (pants + leggings never co-occur)
    ("skirt", 0.3, 2, 45.0),          # 6
    ("skirt", 9.0, 2, 50.0),          # 7  the worn skirt itself: own-item exclusion
    ("leggings", -0.5, 3, 30.0),      # 8
    ("shoe", 0.1, 4, 80.0),           # 9
    ("shoe", -1.0, 4, 20.0),          # 10
    ("bag", 7.0, 8, 99.0),            # 11 no bag is worn: never a candidate
    ("skirt", -0.2, 2, 40.0),         # 12
]
WORN = [("shirt_blouse_0.png", "shirt_blouse"), ("skirt_0.png", "skirt"),
        ("leggings_0.png", "leggings"), ("shoe_0.png", "shoe")]


class StubScorer:
    def __init__(self, logits):
        self.logits, self.calls = logits, 0

    def outfit_logit(self, state):
        self.calls += 1
        return 0.0

    def swap_logits(self, state, i, rows):
        self.calls += 1
        return np.array([self.logits[int(r)] for r in rows.tolist()], np.float32)


def _unit(rng):
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def make(trend_worn=50.0, sigma_logit=1.0, train_rows=(), **overrides):
    rng = np.random.default_rng(0)
    worn_emb = [_unit(rng) for _ in WORN]
    n = len(ROWS)
    emb = np.stack([_unit(rng) for _ in ROWS])
    emb[7] = worn_emb[1]
    split = np.full(n, SPLIT_CODES["test"], np.uint8)
    split[list(train_rows)] = SPLIT_CODES["train"]
    meta = {"source": np.zeros(n, np.uint8), "split": split,
            "type": np.array([TYPE_ID[t] for t, *_ in ROWS], np.int8),
            "broad": np.array([BROAD_ID[BROAD_OF_TYPE[t]] for t, *_ in ROWS], np.int8),
            "store_row": np.arange(n, dtype=np.int32), "outfit_ord": np.arange(n, dtype=np.int32),
            "has_crop": np.ones(n, bool),
            "trend": np.array([r[3] for r in ROWS], np.float32),
            "cluster": np.array([r[2] for r in ROWS], np.int16),
            "rep_demo": np.arange(n, dtype=np.int32),
            "rep_eval": np.where(split == SPLIT_CODES["test"], np.arange(n), -1).astype(np.int32),
            "repost": np.arange(n, dtype=np.int32)}
    scal = np.column_stack([rng.uniform(0, 360, n), rng.uniform(0, 100, n), rng.uniform(500, 5000, n)])
    scal[7, 2] = 2000.0                      # row 7 is the worn top itself: same embedding, colour and size
    pool = CandidatePool(stores=[StoreSpec("fashion144k", "s", "o", "x.mat")], proj=np.zeros((n, 128)),
                         emb=emb, scalars=scal, meta=meta, files=[f"{t}_0.png" for t, *_ in ROWS],
                         outfit_keys={"fashion144k": [str(i) for i in range(n)]}, device="cpu")
    pieces = tuple(Piece(file=f, type=t, broad=BROAD_OF_TYPE[t], proj=torch.zeros(128), emb=e,
                         hue=10.0 * i, formality=50.0, area=2000.0, trend=trend_worn, cluster=-1)
                   for i, ((f, t), e) in enumerate(zip(WORN, worn_emb)))
    counts = np.full((len(TYPES), len(TYPES)), 100, np.int64)
    np.fill_diagonal(counts, 0)
    for a, b in [("pants", "leggings"), ("shorts", "leggings"), ("pants", "shorts")]:
        counts[TYPE_ID[a], TYPE_ID[b]] = counts[TYPE_ID[b], TYPE_ID[a]] = 0
    s = M4Settings.load().with_(w_h=1.0, w_t=0.0, rule="topk_mean_gain", topk=5, eps=0.25, t_dup=0.97,
                                tau={"max_gain": 0.0, "topk_mean_gain": 0.0, "slot_percentile": 0.5},
                                trend_floor=False, same_type=False)
    s = s.with_(**overrides)
    pool.dup_rule = (s.t_dup, s.dup_max_colour_l1, s.dup_max_area_ratio)
    scorer = StubScorer([r[1] for r in ROWS])
    ref = HarmonyReference(np.linspace(-5, 5, 101), sigma_logit=sigma_logit, sigma_trend=10.0)
    return Recommender(s, pool, scorer, ref, CooccurrenceFilter(counts, 20)), OutfitState(pieces), scorer


def rows_of(table):
    return [sorted(ps.rows.tolist()) for ps in table.pieces]


def test_candidates_respect_groups_cooccurrence_and_own_item():
    rec, st, _ = make()
    assert rows_of(rec.score(st)) == [[0, 1, 2, 3, 4], [6, 12], [5, 8], [9, 10]]


def test_topk_mean_swaps_upper_with_style_diverse_top3_and_also_improve():
    rec, st, _ = make()
    r = rec.decide(rec.score(st))
    assert r.verdict == "swap" and r.swap_piece["file"] == "shirt_blouse_0.png"
    assert [s.pool_row for s in r.suggestions] == [0, 2, 3]          # row 1 shares row 0's cluster
    assert [a["piece"] for a in r.also_improve] == ["skirt_0.png"]   # 0.05 >= tau; others below
    assert r.suggestions[0].from_type == "shirt_blouse" and r.suggestions[0].to_type == "t_shirt"
    assert r.per_piece["leggings_0.png"]["topk_mean_gain"] == pytest.approx(-0.15)


def test_selection_rules_are_switchable_without_rescoring():
    rec, st, scorer = make()
    table = rec.score(st)
    calls = scorer.calls
    r_max = rec.decide(table, rule="max_gain")
    r_pct = rec.decide(table, rule="slot_percentile")
    assert [p["selection_score"] for p in r_max.per_piece.values()] == pytest.approx([3.0, 0.3, 0.2, 0.1])
    assert [p["selection_score"] for p in r_pct.per_piece.values()] == pytest.approx([1.0, 0.5, 0.5, 0.5])
    assert r_pct.verdict == "swap" and len(r_pct.also_improve) == 3
    for w in np.linspace(0, 1, 11):
        rec.decide(table, w_t=w)
    assert scorer.calls == calls == 1 + len(st.pieces)               # decide() never calls the GNN


def test_gains_are_standardised():
    rec, st, _ = make(sigma_logit=2.0, w_t=0.5)
    table = rec.score(st)
    up = table.pieces[0]
    assert up.d_h.tolist() == pytest.approx([1.5, 1.45, 1.0, 0.5, 0.25])
    assert up.d_t.tolist() == pytest.approx([1.0, -1.0, 2.0, 0.0, 0.5])
    assert up.trend_known.tolist() == [True, True, True, False, True]
    assert rec.decide(table).per_piece["shirt_blouse_0.png"]["best_gain"] == pytest.approx(2.0)


def test_refresh_when_below_tau_but_trendier_options_keep_harmony():
    rec, st, _ = make(tau={"max_gain": 10.0, "topk_mean_gain": 10.0, "slot_percentile": 1.1})
    r = rec.decide(rec.score(st))
    assert r.verdict == "refresh" and r.swap_piece is None
    assert [s.pool_row for s in r.suggestions] == [0, 5, 9]          # one per worn piece: top, leggings, shoe
    assert all(s.d_trend_z > 0 and s.d_harmony_z >= -0.25 for s in r.suggestions)


def test_refresh_per_piece_cap_is_a_setting_and_fills_when_few_pieces_qualify():
    tau = {"max_gain": 10.0, "topk_mean_gain": 10.0, "slot_percentile": 1.1}
    rec, st, _ = make(tau=tau, refresh_per_piece=3)
    assert [s.pool_row for s in rec.decide(rec.score(st)).suggestions] == [0, 2, 4]
    rec, st, _ = make(tau=tau, refresh_per_piece=2)
    assert [s.pool_row for s in rec.decide(rec.score(st)).suggestions] == [0, 2, 5]
    rec, st, _ = make(trend_worn=85.0, tau=tau)                      # only the leggings slot has a trendier option
    assert [s.pool_row for s in rec.decide(rec.score(st)).suggestions] == [5]


def test_same_trend_score_is_no_trend_gain_despite_float32_storage():
    # pool trends are float32: 82.86 is stored as 82.86000061, the worn piece's score is a Python float
    rec, st, _ = make(trend_worn=82.86, tau={"max_gain": 10.0, "topk_mean_gain": 10.0, "slot_percentile": 1.1})
    rec.pool.trend[9] = 82.86
    table = rec.score(st)
    shoe = table.pieces[3]
    assert shoe.d_t[shoe.rows.tolist().index(9)] == 0.0
    assert 9 not in [s.pool_row for s in rec.decide(table).suggestions]


def test_refresh_never_offers_a_net_loss_to_fill_the_per_piece_cap():
    rec, st, _ = make(eps=2.0, tau={"max_gain": 10.0, "topk_mean_gain": 10.0, "slot_percentile": 1.1})
    for row, trend in ((9, 30.0), (10, 60.0), (5, 30.0), (8, 60.0), (12, 60.0)):
        rec.pool.trend[row] = trend    # each non-top slot's only trendier option lowers harmony (ΔH < 0)
    r = rec.decide(rec.score(st))
    assert r.verdict == "refresh"
    assert all(s.gain > 0 for s in r.suggestions)
    assert [s.pool_row for s in r.suggestions] == [0, 2, 4]


def test_keep_when_nothing_qualifies():
    rec, st, _ = make(trend_worn=100.0, tau={"max_gain": 10.0, "topk_mean_gain": 10.0, "slot_percentile": 1.1})
    r = rec.decide(rec.score(st))
    assert r.verdict == "keep" and r.suggestions == [] and r.also_improve == []


def test_trend_floor_drops_less_trendy_known_and_relaxes_when_empty():
    rec, st, _ = make(trend_floor=True)
    table = rec.score(st)
    assert rows_of(table) == [[0, 2, 3, 4], [6, 12], [5], [9]]
    assert any("relaxed" in f for f in table.pieces[1].flags)


def test_same_type():
    rec, st, _ = make(same_type=True)
    assert rows_of(rec.score(st)) == [[4], [6, 12], [8], [9, 10]]


def test_eval_mode_and_source_exclusion():
    rec, st, _ = make(train_rows=(0,))
    table = rec.score(st, mode="eval", exclude=("fashion144k", "2"))
    assert rows_of(table)[0] == [1, 3, 4]


def test_json_round_trip_and_text():
    rec, st, _ = make(w_t=0.5)
    r = rec.decide(rec.score(st))
    again = Recommendation.from_json(r.to_json())
    assert again == r
    assert "SWAP your shirt_blouse" in format_text(r)
    assert r.provisional == rec.settings.provisional()


def test_decide_takes_the_harmony_weight():
    rec, st, _ = make()
    table = rec.score(st)
    # shirt_blouse candidates: ΔH = 3, 2.9, 2, 1, 0.5; ΔT = 1, -1, 2, 0 (unknown), 0.5 (worn trend 50, σ_trend 10)
    r = rec.decide(table, w_t=0.5, w_h=0.5)
    assert r.weights == {"w_h": 0.5, "w_t": 0.5}
    assert r.per_piece["shirt_blouse_0.png"]["best_gain"] == pytest.approx(2.0)
    r_default = rec.decide(table, w_t=0.5)                 # w_h falls back to settings (1.0 here)
    assert r_default.weights == {"w_h": 1.0, "w_t": 0.5}
    assert r_default.per_piece["shirt_blouse_0.png"]["best_gain"] == pytest.approx(3.5)
