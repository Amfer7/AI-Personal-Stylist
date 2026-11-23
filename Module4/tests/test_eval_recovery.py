import numpy as np
import pytest

from eval_metrics import grid_summary
from eval_recovery import SOURCE, WS, eval_sets, evaluate_outfit
from outfit_state import state_from_pool
from selection import RULES
from swap_groups import swap_group

pytestmark = pytest.mark.real


@pytest.fixture(scope="module")
def sets(pool):
    return eval_sets(pool, 2000)


def test_sets_are_disjoint_seeded_test_outfits(pool, sets):
    dev, report = sets
    assert len(dev) == len(report) == 2000
    assert not set(dev) & set(report)
    assert (dev, report) == eval_sets(pool, 2000)
    for key in dev[:50] + report[:50]:
        rows = pool.outfit_rows(SOURCE, key)
        assert len(rows) >= 2 and (pool.split[rows] == 2).all()


def test_grid_summary_agrees_with_decide(recommender, sets):
    """The evaluation's fast summary must pick the same piece and top candidate as decide()."""
    rec = recommender
    for key in sets[0][:5]:
        state = state_from_pool(rec.pool, SOURCE, key)
        table = rec.score(state, mode="eval", exclude=(SOURCE, key), same_type=False, trend_floor=False)
        g = grid_summary(table.pieces, WS, rec.settings.topk)
        for ri, rule in enumerate(RULES):
            for wi in (0, 3, 10):
                w = float(WS[wi])
                saved = rec.settings
                rec.settings = saved.with_(w_h=1.0 - w, w_t=w, tau={**saved.tau, rule: -1e9})   # force "swap"
                try:
                    r = rec.decide(table, rule=rule)
                finally:
                    rec.settings = saved
                j = int(g["chosen"][ri, wi])
                assert r.swap_piece["file"] == state.pieces[j].file
                assert r.per_piece[state.pieces[j].file]["selection_score"] == pytest.approx(g["s_star"][ri, wi])
                assert r.suggestions[0].pool_row == g["top_row"][ri, wi]


def test_evaluate_outfit_record_is_consistent(recommender, sets):
    rec, pool = recommender, recommender.pool
    for key in sets[0][:5]:
        r = evaluate_outfit(rec, key, np.random.default_rng([0, int(key)]))
        assert r is not None
        own = set(pool.outfit_rows(SOURCE, key).tolist())
        assert r["corrupter_row"] not in own
        assert pool.split[r["corrupter_row"]] == 2
        state = state_from_pool(pool, SOURCE, key)
        orig = state.pieces[r["slot"]]
        ctype = pool.type_name(r["corrupter_row"])
        others = [p.type for j, p in enumerate(state.pieces) if j != r["slot"]]
        assert swap_group(ctype) == orig.group
        assert rec.cooc.allowed(ctype, others)
        c = pool.empty_rows().new_tensor([r["corrupter_row"]])
        assert not bool(pool.dup_mask(c, orig.emb, orig.colour, orig.area)[0])
        assert all(1 <= x <= r["n_rank"] for x in r["orig_rank"])
        assert 1 <= r["clip_rank"] <= r["n_rank"] and 1 <= r["rand_rank"] <= r["n_rank"]
        assert len(r["corr"]["chosen"]) == len(RULES) and len(r["corr"]["chosen"][0]) == len(WS)
        # same seed -> same corruption
        again = evaluate_outfit(rec, key, np.random.default_rng([0, int(key)]))
        assert again["corrupter_row"] == r["corrupter_row"] and again["slot"] == r["slot"]
