"""Build Module 4's cache: candidate pool (projection, trend, dedupe), co-occurrence matrix and harmony
reference. Rebuilds only when the fingerprint (checkpoint, trend model, stores, t_dup) has changed.
Also measures every crop for the demo pool's quality gate (crop_quality.py) when the pool changed.

    python Module4/build_cache.py            # build if missing or stale
    python Module4/build_cache.py --force    # rebuild anyway
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from candidate_pool import CandidatePool, build_pool, gate_demo_pool, is_fresh, pool_dir, pool_fingerprint  # noqa: E402
from crop_quality import build_metrics, metrics_fresh  # noqa: E402
from dense_gnn import DenseGNN  # noqa: E402
from harmony_reference import build_reference, train_outfit_types  # noqa: E402
from m2_bridge import apply_vram_cap, load_gnn, load_trend_model, pick_device  # noqa: E402
from m4_settings import M4Settings  # noqa: E402
from swap_groups import SWAP_GROUPS, CooccurrenceFilter  # noqa: E402


def _raw_pool_arrays(s):
    from candidate_pool import load_store_arrays
    d = pool_dir(s)
    emb, scal = load_store_arrays(s)
    with open(os.path.join(d, "files.json"), encoding="utf-8") as f:
        files = json.load(f)
    with open(os.path.join(d, "outfit_keys.json"), encoding="utf-8") as f:
        keys = json.load(f)
    return dict(proj=np.load(os.path.join(d, "proj.npy")), emb=emb, scalars=scal,
                meta=gate_demo_pool(s, dict(np.load(os.path.join(d, "meta.npz"))), scal), files=files,
                outfit_keys=keys)


def _ensure_crop_metrics(s) -> None:
    if metrics_fresh(s, pool_dir(s)):
        return
    t = time.time()
    n = len(build_metrics(s, pool_dir(s))["area"])
    print(f"[m4-cache] crop metrics: {n} crops measured in {time.time() - t:.0f}s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--settings", default=None, help="another settings JSON (default: Module4/m4_settings.json)")
    args = ap.parse_args(argv)

    s = M4Settings.load(args.settings) if args.settings else M4Settings.load()
    dev = pick_device(args.device)
    apply_vram_cap(s.vram_gb, dev)
    if is_fresh(s) and not args.force:
        _ensure_crop_metrics(s)
        print(f"[m4-cache] up to date: {pool_dir(s)}")
        return
    fp = os.path.join(pool_dir(s), "fingerprint.json")
    if os.path.exists(fp):
        os.remove(fp)   # an interrupted rebuild must never look fresh
    gnn = DenseGNN(load_gnn(s.abs(s.checkpoint), dev), dev, s.vram_gb)
    t = time.time()
    build_pool(s, gnn, load_trend_model(s.abs(s.trend_model)))
    print(f"[m4-cache] pool built in {time.time() - t:.0f}s")
    _ensure_crop_metrics(s)
    pool = CandidatePool(stores=s.stores, **_raw_pool_arrays(s), device=dev)
    CooccurrenceFilter.from_outfits(train_outfit_types(pool), s.cooc_min).save(
        os.path.join(pool_dir(s), "cooc.npy"))
    ref = build_reference(pool, gnn)
    ref.save(os.path.join(pool_dir(s), "reference.json"))
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(pool_fingerprint(s), f, indent=2)
    idx = np.arange(pool.n)
    print(f"[m4-cache] {pool.n} rows; demo reps {(pool.rep_demo == idx).sum()}, "
          f"eval reps {(pool.rep_eval == idx).sum()}; trend-scored {(pool.trend == pool.trend).mean():.1%}")
    print(f"[m4-cache] reference: {len(ref.logits_sorted)} test outfits, sigma_logit {ref.sigma_logit:.4f}, "
          f"sigma_trend {ref.sigma_trend:.3f}")
    for g, types in SWAP_GROUPS.items():
        demo = len(pool.candidate_rows(list(types), "demo"))
        ev = len(pool.candidate_rows(list(types), "eval"))
        print(f"[m4-cache]   group {g:10s} demo {demo:>7}  eval {ev:>7}")

if __name__ == "__main__":
    main()
