"""Domain check: do Module 2 (harmony) and Module 3 (trend) behave on H&M studio product shots?

The GNN and the trend model were fitted on garments cut out of street / Pinterest photos. Before the
H&M catalogue becomes the demo's suggestion pool, this compares an H&M sample (hm_catalogue.py
--per_type) with the existing stores on four things:

1. Type agreement: CLIP zero-shot label_p (crop_quality.clip_scores) on H&M items vs. store crops.
2. Embedding gap: each item's nearest same-type store garment (cosine), and how well a linear
   classifier tells H&M from store embeddings.
3. Trend: share of items with a confident M3 cluster match, and how many clusters they spread over.
4. GNN transfer (the main test). For test outfits of both catalogues, one random piece is replaced in
   turn by a set of H&M candidates and by an equally sized set of store candidates (same types,
   test split, never the outfit itself or its reposts). In each set the candidate nearest to the worn
   piece (CLIP) is a stand-in for "what the person actually wore"; its rank among the set by GNN
   score shows whether harmony judgements survive the change of photo style. The store set is the
   reference for what the GNN does in its own domain.

    python Module4/hm_domain_check.py --store "H&M Dataset/check_store" --n 1000
    python Module4/hm_domain_check.py --align mean    # H&M embeddings shifted onto the store's per-type mean

--align mean|coral runs every check on embeddings aligned the way hm_catalogue.py --align --method
aligns the full catalogue (mean: each type's H&M mean moved onto the street stores' mean; coral: mean
and covariance matched); results go to domain_check_<method>.json. The checks always start from the raw
embeddings (features_raw.npy when the store was aligned in place). CORAL needs a few hundred items per
type for its covariances, so run it on the full catalogue ("H&M Dataset/store"), not the check sample.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from candidate_pool import SPLIT_CODES, trend_scores  # noqa: E402
from crop_quality import clip_scores, encode_prompts, prompts  # noqa: E402
from harmony_reference import test_outfits  # noqa: E402
from hm_catalogue import ALIGN_METHODS, align_embeddings  # noqa: E402
from m2_bridge import BROAD_ID, BROAD_VOCAB, FEATURE_COLS  # noqa: E402
from m4_settings import M4Settings  # noqa: E402
from outfit_state import state_from_pool  # noqa: E402
from recommender import Recommender  # noqa: E402
from swap_groups import TYPE_ID, TYPES, replacement_types  # noqa: E402


def load_hm(store: str, gnn, trend_model, device, settings=None, align: str = "none") -> dict:
    d = os.path.join(store, "feature_store")
    raw = os.path.join(d, "features_raw.npy")     # a store aligned in place keeps its raw copy here
    feat = np.load(raw if os.path.exists(raw) else os.path.join(d, "features.npy"))
    scal = np.load(os.path.join(d, "node_scalars.npy"))
    with open(os.path.join(d, "index.json"), encoding="utf-8") as f:
        index = json.load(f)
    n = feat.shape[0]
    type_ = np.full(n, -1, np.int64)
    broad = np.full(n, -1, np.int64)
    key = [None] * n
    for k, es in index["outfits"].items():
        for e in es:
            type_[e["row"]], broad[e["row"]], key[e["row"]] = TYPE_ID[e["category"]], BROAD_ID[e["broad_category"]], k
    if align != "none":
        feat[:, :512] = align_embeddings(feat[:, :512], type_, settings, align)
    emb = feat[:, :512] / np.linalg.norm(feat[:, :512], axis=1, keepdims=True)
    trend, cluster = trend_scores(feat[:, :512], np.asarray(BROAD_VOCAB)[broad], trend_model)
    art = pd.read_csv(os.path.join(store, "articles.csv"), dtype={"key": str}).set_index("key")
    t = lambda x, dt=torch.float64: torch.as_tensor(np.asarray(x), dtype=dt, device=device)
    return {"n": n, "type": type_, "broad": broad, "key": key, "emb": emb.astype(np.float32),
            "gender": art.loc[key, "gender"].to_numpy(), "trend": trend, "cluster": cluster,
            "proj": gnn.project(torch.from_numpy(feat[:, FEATURE_COLS])),
            "hue": t(scal[:, 0]), "formality": t(scal[:, 1]), "area": t(scal[:, 2]),
            "broad_t": t(broad, torch.long)}


def store_cols(pool) -> dict:
    return {"proj": pool.proj, "hue": pool.hue, "formality": pool.formality, "area": pool.area,
            "broad_t": pool.broad_t}


def swap_logits(gnn, state, i, cols, rows) -> np.ndarray:
    """GNN logit of the outfit with piece i replaced by each row of a candidate table (pool or H&M)."""
    from swap_scorer import SwapScorer
    h, hue, form, area, broad = SwapScorer(None, gnn)._fixed(state)
    n = h.shape[0]
    rows = torch.as_tensor(rows, device=gnn.device)

    def make(a, b):
        r = rows[a:b]
        B = r.shape[0]
        H = h.expand(B, n, -1).clone()
        H[:, i] = cols["proj"][r]
        out = [H]
        for fixed, col in ((hue, cols["hue"]), (form, cols["formality"]), (area, cols["area"]), (broad, cols["broad_t"])):
            x = fixed.expand(B, n).clone()
            x[:, i] = col[r]
            out.append(x)
        return tuple(out)

    return gnn.logits_chunked(make, int(rows.shape[0]), n).numpy()


def pct_rank(scores, k) -> float:
    """Share of the set scoring above item k (ties half): 0 = top, 0.5 = chance."""
    s = scores[k]
    return float(((scores > s).sum() + 0.5 * ((scores == s).sum() - 1)) / max(len(scores) - 1, 1))


# ---------------- checks 1-3 ----------------
def type_agreement(hm, pool, store_rows, device) -> dict:
    text_emb, group = encode_prompts(device), prompts()[1]
    out = {}
    for name, emb, types in (("hm", hm["emb"], hm["type"]),
                             ("store", pool.emb[store_rows].cpu().numpy(), pool.type[store_rows])):
        s = clip_scores(emb, types, text_emb, group)
        out[name] = {TYPES[t]: {"median_label_p": float(np.median(s["label_p"][types == t])),
                                "pass_0.02": float(np.mean(s["label_p"][types == t] >= 0.02)),
                                "median_junk_p": float(np.median(s["junk_p"][types == t]))}
                     for t in np.unique(types)}
    return out


def embedding_gap(hm, pool, store_rows, rng) -> dict:
    """Nearest same-type store garment for H&M items, and for a store sample (excluding its own
    repost group); plus a linear H&M-vs-store classifier."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    ref = np.nonzero(pool.rep_eval == np.arange(pool.n))[0]
    out = {"hm": {}, "store": {}}
    for t in np.unique(hm["type"]):
        R = ref[pool.type[ref] == t]
        E = pool.emb[torch.as_tensor(R, device=pool.device)]
        h = torch.as_tensor(hm["emb"][hm["type"] == t], device=pool.device)
        out["hm"][TYPES[t]] = float((h @ E.T).max(1).values.median())
        s = store_rows[pool.type[store_rows] == t]
        sims = pool.emb[torch.as_tensor(s, device=pool.device)] @ E.T
        sims[torch.as_tensor(pool.repost[s][:, None] == pool.repost[R][None, :], device=pool.device)] = -1
        out["store"][TYPES[t]] = float(sims.max(1).values.median())
    X = np.r_[hm["emb"], pool.emb[store_rows].cpu().numpy()]
    y = np.r_[np.ones(hm["n"]), np.zeros(len(store_rows))]
    out["domain_classifier_acc"] = float(cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5).mean())
    return out


def trend_coverage(hm, pool, store_rows) -> dict:
    out = {}
    for name, trend, cluster, types in (("hm", hm["trend"], hm["cluster"], hm["type"]),
                                        ("store", pool.trend[store_rows], pool.cluster[store_rows],
                                         pool.type[store_rows])):
        per = {}
        for t in np.unique(types):
            m = types == t
            c = cluster[m][cluster[m] >= 0]
            p = np.bincount(c - c.min()) / len(c) if len(c) else np.zeros(1)
            p = p[p > 0]
            per[TYPES[t]] = {"matched": float(np.mean(~np.isnan(trend[m]))), "clusters": int(len(p)),
                             "eff_clusters": float(np.exp(-(p * np.log(p)).sum())) if len(c) else 0.0,
                             "mean_trend": float(np.nanmean(trend[m])) if (~np.isnan(trend[m])).any() else None}
        out[name] = per
    return out


# ---------------- check 4 ----------------
def gnn_transfer(rec, hm, n_outfits, rng) -> pd.DataFrame:
    pool, gnn = rec.pool, rec.gnn
    hm_cols, st_cols = hm, store_cols(pool)
    eval_rep = np.nonzero(pool.rep_eval == np.arange(pool.n))[0]
    outfits = test_outfits(pool)
    recs = []
    for src in range(len(pool.stores)):
        mine = [r for r in outfits if pool.source[r[0]] == src]
        for o in rng.permutation(len(mine))[:n_outfits]:
            rows = mine[o]
            key = pool.outfit_key(int(rows[0]))
            state = state_from_pool(pool, pool.stores[src].name, key)
            slots = [i for i, p in enumerate(state.pieces) if replacement_types(p.type)]
            if not slots:
                continue
            i = int(rng.choice(slots))
            p = state.pieces[i]
            others = [q.type for j, q in enumerate(state.pieces) if j != i]
            types = [t for t in replacement_types(p.type) if rec.cooc.allowed(t, others)]
            if not types:
                continue
            tid = [TYPE_ID[t] for t in types]
            h_rows = np.nonzero(np.isin(hm["type"], tid))[0]
            # equally many store candidates per type: test-split representatives, not this outfit's repost group
            s_rows = []
            for t in tid:
                k = int((hm["type"] == t).sum())
                c = eval_rep[(pool.type[eval_rep] == t) & (pool.repost[eval_rep] != pool.repost[rows[0]])]
                s_rows.append(rng.choice(c, min(k, len(c)), replace=False))
            s_rows = np.concatenate(s_rows)
            if len(h_rows) < 20 or len(s_rows) < 20:
                continue
            E = np.stack([q.emb for q in state.pieces])
            ctx = np.delete(E, i, 0).mean(0)
            rec_row = {"source": pool.stores[src].name, "key": key, "type": p.type, "r0": rec.scorer.outfit_logit(state)}
            for name, cols, cr, emb in (("hm", hm_cols, h_rows, hm["emb"][h_rows]),
                                        ("store", st_cols, s_rows, pool.emb[torch.as_tensor(s_rows, device=pool.device)].cpu().numpy())):
                logit = swap_logits(gnn, state, i, cols, cr)
                cos = emb @ p.emb
                k = int(np.argmax(cos))
                rec_row.update({f"{name}_n": len(cr), f"{name}_proxy_cos": float(cos[k]),
                                f"{name}_gnn_pct": pct_rank(logit, k),
                                f"{name}_clip_pct": pct_rank(emb @ ctx, k),
                                f"{name}_mean_logit": float(logit.mean()), f"{name}_top_logit": float(logit.max()),
                                f"{name}_share_gain": float(np.mean(logit > rec_row["r0"]))})
            recs.append(rec_row)
    return pd.DataFrame(recs)


def summarise_transfer(df: pd.DataFrame, sigma: float) -> dict:
    out = {}
    for src, g in [("all", df)] + list(df.groupby("source")):
        d = {"outfits": len(g)}
        for name in ("hm", "store"):
            d[name] = {"gnn_pct_median": float(g[f"{name}_gnn_pct"].median()),
                       "gnn_pct_mean": float(g[f"{name}_gnn_pct"].mean()),
                       "gnn_top10": float((g[f"{name}_gnn_pct"] <= 0.1).mean()),
                       "clip_pct_mean": float(g[f"{name}_clip_pct"].mean()),
                       "proxy_cos_median": float(g[f"{name}_proxy_cos"].median()),
                       "share_gain_mean": float(g[f"{name}_share_gain"].mean())}
        d["mean_logit_shift_sigma"] = float(((g.hm_mean_logit - g.store_mean_logit) / sigma).mean())
        d["top_logit_shift_sigma"] = float(((g.hm_top_logit - g.store_top_logit) / sigma).mean())
        out[src] = d
    # like-for-like: outfits where both stand-ins are about equally close to the worn piece
    bins = [0, 0.75, 0.8, 0.85, 0.9, 1.01]
    for name in ("hm", "store"):
        cut = pd.cut(df[f"{name}_proxy_cos"], bins)
        out[f"{name}_gnn_pct_by_proxy_cos"] = {str(k): [round(float(v), 3), int(c)] for k, v, c in
                                               zip(cut.cat.categories, df.groupby(cut, observed=False)[f"{name}_gnn_pct"].mean(),
                                                   df.groupby(cut, observed=False).size())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="H&M Dataset/check_store")
    ap.add_argument("--n", type=int, default=1000, help="test outfits per catalogue")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--align", choices=("none",) + ALIGN_METHODS, default="none")
    args = ap.parse_args()
    s = M4Settings.load()
    s = s.with_(demo_sources=[sp.name for sp in s.stores if not sp.product_photos])   # compare with street crops
    store = s.abs(args.store)
    rec = Recommender.from_settings(s)
    pool, rng = rec.pool, np.random.default_rng(args.seed)
    hm = load_hm(store, rec.gnn, rec.trend_model, pool.device, s, args.align)
    print(f"[check] {hm['n']} H&M items")
    demo = np.nonzero(pool.rep_demo == np.arange(pool.n))[0]
    pick = lambda c, k: rng.choice(c, min(k, len(c)), replace=False)   # the full catalogue can outnumber the stores
    store_rows = np.concatenate([pick(demo[pool.type[demo] == t], int((hm["type"] == t).sum()))
                                 for t in np.unique(hm["type"])])
    report = {"n_hm": hm["n"],
              "type_agreement": type_agreement(hm, pool, store_rows, str(pool.device)),
              "embedding_gap": embedding_gap(hm, pool, store_rows, rng),
              "trend": trend_coverage(hm, pool, store_rows)}
    df = gnn_transfer(rec, hm, args.n, rng)
    tag = "" if args.align == "none" else f"_{args.align}"
    df.to_csv(os.path.join(store, f"gnn_transfer{tag}.csv"), index=False)
    report["gnn_transfer"] = summarise_transfer(df, rec.reference.sigma_logit)
    with open(os.path.join(store, f"domain_check{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
