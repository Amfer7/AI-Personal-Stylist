"""
forward_share_backtest.py  (Module 4 decision D11, Phase 0 step 0.6)

Which trend formula should Module 4 act on? The trend score mixes two per-cluster signals, growth
and recency (trend_model._assign_trendiness_scores). temporal_holdout_validation.py showed growth
does not predict future *growth*. Module 4 needs something different: is a style that scores high
today still widely seen months later? This script asks exactly that.

For every garment category and every rolling cutoff month c (default 2026-01 ... 2026-05):
  1. fit the category's TrendModel on garments dated before c only (same k rule, same KMeans);
  2. score every cluster with each growth weight w (0.6 = current, 0.3, 0 = recency-only, 1 = growth-only
     reference), using the model's own scoring code;
  3. assign the held-out garments to the train clusters and measure each cluster's **forward share**:
     its share of the category's garments in the next h months (h = 1, 3, 6; windows past the last
     complete month are skipped);
  4. Spearman ρ between score and forward share over clusters with enough training garments.

Each (category, cutoff, horizon) is one block. A weight's result is the mean ρ over blocks, with a
bootstrap CI over blocks. Decision rule (D11): the weight with the best pooled ρ wins; if its
advantage over recency-only (w = 0) is not significant (paired bootstrap CI of the mean difference
includes 0), recency-only wins as the simpler formula.

Usage:
    python forward_share_backtest.py \
        --manifest_csv ../PinterestData/store/garment_manifest.csv \
        --embeddings   ../PinterestData/store/garment_embeddings.npy --end_month 2026-08
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from trend_model import (MIN_CLUSTER_IMAGES, TrendModel, _assign_trendiness_scores, _parse_ym,
                         k_for_category)

WEIGHTS = (0.6, 0.3, 0.0, 1.0)          # current, middle, recency-only, growth-only (reference)
CANDIDATES = (0.6, 0.3, 0.0)            # the decision is among these
HORIZONS = (1, 3, 6)
DEFAULT_CUTOFFS = ("2026-01", "2026-02", "2026-03", "2026-04", "2026-05")


def shift_month(ym: str, delta: int) -> str:
    y, m = _parse_ym(ym)
    t = y * 12 + (m - 1) + delta
    return f"{t // 12}-{t % 12 + 1:02d}"


def forward_window(cutoff: str, horizon: int, end_month: str):
    """The h months starting at the cutoff, or None if the window runs past end_month."""
    months = [shift_month(cutoff, i) for i in range(horizon)]
    return months if months[-1] <= end_month else None


def block_rhos(emb, dates, cat, cutoff, end_month, n_clusters_per_category, weights=WEIGHTS):
    """Spearman ρ per (horizon, weight) for one category and cutoff. Returns a list of dicts."""
    train = dates < cutoff
    n_train = int(train.sum())
    if n_train < 100:
        return []
    k = k_for_category(cat, n_train, n_clusters_per_category)
    model = TrendModel(n_clusters=k).fit(emb[train], list(dates[train]))
    labels = model.kmeans.predict(emb)
    trusted = [cid for cid, c in model.clusters.items() if c.n_images >= MIN_CLUSTER_IMAGES]
    if len(trusted) < 4:
        return []
    out = []
    scores = {}
    for w in weights:
        _assign_trendiness_scores(model.clusters, growth_weight=w)
        scores[w] = np.array([model.clusters[c].trendiness_score for c in trusted])
    for h in HORIZONS:
        window = forward_window(cutoff, h, end_month)
        if window is None:
            continue
        in_win = np.isin(dates, window)
        total = int(in_win.sum())
        if total == 0:
            continue
        share = np.array([np.sum(in_win & (labels == c)) / total for c in trusted])
        for w in weights:
            if np.ptp(scores[w]) == 0 or np.ptp(share) == 0:
                continue
            rho = spearmanr(scores[w], share).correlation
            out.append({"category": cat, "cutoff": cutoff, "horizon": h, "weight": w, "rho": float(rho),
                        "n_clusters": len(trusted), "n_forward": total})
    return out


def decide(blocks: pd.DataFrame, n_boot: int = 5000, seed: int = 0) -> dict:
    """Mean ρ per weight over (category, cutoff, horizon) blocks, bootstrap CIs, and the D11 rule."""
    piv = blocks.pivot_table(index=["category", "cutoff", "horizon"], columns="weight", values="rho")
    piv = piv.dropna(subset=[w for w in CANDIDATES if w in piv.columns])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(piv), size=(n_boot, len(piv)))
    res = {}
    for w in piv.columns:
        x = piv[w].to_numpy()
        boots = np.nanmean(x[idx], axis=1)
        res[float(w)] = {"mean_rho": float(np.nanmean(x)), "ci": [float(np.percentile(boots, 2.5)),
                                                                  float(np.percentile(boots, 97.5))]}
    best = max(CANDIDATES, key=lambda w: res[w]["mean_rho"])
    diff = (piv[best] - piv[0.0]).to_numpy()
    boots = diff[idx].mean(axis=1)
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    tie = best != 0.0 and lo <= 0.0
    chosen = 0.0 if tie or best == 0.0 else best
    return {"n_blocks": int(len(piv)), "weights": res, "best": best,
            "best_minus_recency": {"mean": float(diff.mean()), "ci": [lo, hi]},
            "tie_with_recency_only": bool(tie), "chosen_growth_weight": chosen}


def run(manifest_csv, embeddings, end_month="2026-08", cutoffs=DEFAULT_CUTOFFS,
        n_clusters_per_category=15, category_col="broad_category", min_images=100):
    man = pd.read_csv(manifest_csv)
    emb = np.load(embeddings, mmap_mode="r")
    assert len(man) == len(emb), "manifest/embeddings row count mismatch"
    keep = (man["date_ym"] <= end_month).to_numpy()
    man, emb = man[keep].reset_index(drop=True), np.asarray(emb[keep])
    rows = []
    for cat in sorted(man[category_col].unique()):
        mask = (man[category_col] == cat).to_numpy()
        if mask.sum() < min_images:
            continue
        dates = man.loc[mask, "date_ym"].to_numpy().astype(str)
        for c in cutoffs:
            rows += block_rhos(emb[mask], dates, cat, c, end_month, n_clusters_per_category)
    return pd.DataFrame(rows)


def main():
    for stream in (sys.stdout, sys.stderr):        # Windows console: Greek letters
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--end_month", default="2026-08", help="last complete month")
    ap.add_argument("--cutoffs", nargs="+", default=list(DEFAULT_CUTOFFS))
    ap.add_argument("--n_clusters", type=int, default=15, help="default k per category (overrides apply)")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    blocks = run(args.manifest_csv, args.embeddings, args.end_month, args.cutoffs, args.n_clusters)
    if blocks.empty:
        print("No usable blocks: too little data before the cutoffs.")
        return None
    table = blocks.pivot_table(index="horizon", columns="weight", values="rho", aggfunc="mean")
    print("Mean Spearman ρ (score vs forward share) by horizon and growth weight:")
    print(table.round(3).to_string())
    d = decide(blocks)
    print(f"\nPooled over {d['n_blocks']} blocks (category × cutoff × horizon):")
    for w, r in sorted(d["weights"].items()):
        label = {0.6: "current", 0.3: "middle", 0.0: "recency-only", 1.0: "growth-only (reference)"}.get(w, "")
        print(f"  growth weight {w:.1f} {label:24s} mean ρ {r['mean_rho']:+.3f}  (95% CI {r['ci'][0]:+.3f} to {r['ci'][1]:+.3f})")
    bm = d["best_minus_recency"]
    print(f"\nBest candidate: {d['best']:.1f}; minus recency-only: {bm['mean']:+.3f} "
          f"(95% CI {bm['ci'][0]:+.3f} to {bm['ci'][1]:+.3f})")
    print(f"D11 decision: growth weight = {d['chosen_growth_weight']:.1f}"
          + (" (statistical tie -> recency-only)" if d["tie_with_recency_only"] else ""))
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"decision": d, "blocks": blocks.to_dict("records")}, f, indent=2)
    return d


if __name__ == "__main__":
    main()
