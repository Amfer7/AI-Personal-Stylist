"""Module 4 evaluation (spec §6, decision D9) on test outfits of one source (Fashion144k by default).

Two disjoint, seeded outfit sets are drawn from the test split: **dev** (all tuning) and **report**
(final numbers, run once with frozen settings). For every outfit:

- **Corruption (group-consistent):** one random piece is replaced by a random garment from the same
  D4 swap group that passes the co-occurrence filter, drawn from the eval pool (test rows), never from
  the outfit itself or its reposts, nor a near-duplicate of the original. This is exactly the space M4 searches.
- **M4** scores the corrupted outfit and the real one (pool "eval", source outfit excluded) and
  summarises both over every selection rule × trend weight (eval_metrics.grid_summary). The trend
  weight is a mix: gain = (1 - w)·ΔH + w·ΔT.
- **Original recovery:** the garment actually worn is ranked against every candidate for the
  corrupted slot (mid-rank for ties). Reposts of the outfit are excluded, so the original can't be a candidate.
- **Baselines** on the same corruptions: random (piece + candidate), trend-only (w = 1) and CLIP
  coherence (weak link = lowest mean cosine to the other pieces; replacement = highest cosine to their mean).

Calibration on dev, in order: selection rule (best hit rate at w = 0) -> w (largest weight keeping
≥ 90% of the w = 0 harmony gain) -> τ per rule (balanced accuracy, real vs corrupted). ε = M4's
detection limit: on the dev set its hit rate is no better than chance for harmony drops below ~0.2σ
(table "Detection by harmony drop"), so a refresh option losing less than that "keeps the look".

    python Module4/src/eval_recovery.py --set dev --freeze     # calibrate, write m4_settings.json
    python Module4/src/eval_recovery.py --set report           # final numbers, once, frozen settings
    python Module4/src/eval_recovery.py --set report --source pinterest   # same, on Pinterest test outfits
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

for _s in (sys.stdout, sys.stderr):           # Windows console: Greek letters and arrows
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from eval_metrics import (balanced_threshold, bootstrap_ci, grid_summary, mcnemar_p,  # noqa: E402
                          pareto_weight, severity_strata)
from harmony_reference import test_outfits  # noqa: E402
from m4_settings import M4Settings  # noqa: E402
from outfit_state import piece_from_pool, state_from_pool  # noqa: E402
from selection import RULES  # noqa: E402

SOURCE = "fashion144k"
WS = np.round(np.linspace(0.0, 1.0, 11), 2)
KS = (1, 3, 10)
SEED = 0
PARETO_KEEP = 0.9
EPS_DETECTION_LIMIT = 0.2      # σ units; from the dev detection-by-drop table (owner decision 2026-10-09)
DROP_BINS = (-np.inf, 0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, np.inf)


# ---------------- outfit sets ----------------
def eval_sets(pool, n: int, seed: int = SEED, source: str = SOURCE):
    """Disjoint dev / report outfit keys: a seeded shuffle of `source` test outfits with ≥ 2 garments."""
    keys = sorted((pool.outfit_key(int(rows[0])) for rows in test_outfits(pool) if
                   pool.source_name(int(rows[0])) == source), key=int)
    order = np.random.default_rng(seed).permutation(len(keys))
    keys = [keys[i] for i in order]
    if 2 * n > len(keys):
        raise ValueError(f"asked for 2 × {n} outfits, only {len(keys)} test outfits available")
    return keys[:n], keys[n:2 * n]


# ---------------- one outfit ----------------
def _ranks(g_cand, g_orig):
    """1-based mid-rank of the original among candidates, and the list size. Ties count half: trend
    scores are per style cluster, so thousands of candidates can tie with the original exactly."""
    above = int((g_cand > g_orig).sum())
    ties = int((g_cand == g_orig).sum())
    return 1 + above + ties / 2, len(g_cand) + 1


def evaluate_outfit(rec, key: str, rng, source: str = SOURCE) -> dict | None:
    pool, s = rec.pool, rec.settings
    state = state_from_pool(pool, source, key)
    exclude = (source, key)
    cand = [rec._candidates(state, i, "eval", exclude, False, False)[0] for i in range(len(state.pieces))]
    slots = [i for i, c in enumerate(cand) if len(c)]
    if not slots:
        return None
    i = int(rng.choice(slots))
    c_row = int(cand[i][int(rng.integers(len(cand[i])))])
    orig = state.pieces[i]
    corrupt = state.with_swap(i, piece_from_pool(pool, c_row))

    kw = dict(mode="eval", exclude=exclude, same_type=False, trend_floor=False)
    t_real = rec.score(state, **kw)
    t_corr = rec.score(corrupt, **kw)
    sig_h, sig_t = rec.reference.sigma_logit, rec.reference.sigma_trend
    g_real = grid_summary(t_real.pieces, WS, s.topk)
    g_corr = grid_summary(t_corr.pieces, WS, s.topk)

    # original recovery on the corrupted slot
    ps = t_corr.pieces[i]
    rows_t = torch.as_tensor(ps.rows, device=pool.device)
    dh_o = (t_real.r0 - t_corr.r0) / sig_h
    cur = corrupt.pieces[i]
    dt_o = (orig.trend - cur.trend) / sig_t if not (np.isnan(orig.trend) or np.isnan(cur.trend)) else 0.0
    orig_rank = []
    for w in WS:
        r, n_list = _ranks((1 - w) * ps.d_h + w * ps.d_t, (1 - w) * dh_o + w * dt_o)
        orig_rank.append(r)

    # CLIP-coherence baseline (no GNN)
    E = np.stack([p.emb for p in corrupt.pieces]).astype(np.float64)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    n = len(E)
    mean_cos = (E @ E.T).sum(1) - 1.0
    clip_pick = int(np.argmin(mean_cos / (n - 1)))
    target = np.delete(E, i, axis=0).mean(0)
    target /= np.linalg.norm(target) + 1e-12
    clip_c = pool.cosine_to(rows_t, target).cpu().numpy() if len(ps.rows) else np.zeros(0)
    clip_rank, _ = _ranks(clip_c, float(orig.emb.astype(np.float64) @ target))

    return {
        "key": key, "n_pieces": n, "slot": i, "slot_type": orig.type, "slot_group": orig.group,
        "corrupter_row": c_row, "r_orig": float(t_real.r0), "r_corr": float(t_corr.r0),
        "severity": float(dh_o), "n_rank": n_list, "orig_rank": orig_rank,
        "clip_pick": clip_pick, "clip_rank": clip_rank,
        "rand_pick": int(rng.integers(n)), "rand_rank": int(rng.integers(1, n_list + 1)),
        "corr": {k: g_corr[k].tolist() for k in ("chosen", "s_star", "top_dh", "top_dt", "top_logit", "top_row")},
        "real": {k: g_real[k].tolist() for k in ("chosen", "s_star", "top_dh", "top_dt")},
    }


# ---------------- aggregation ----------------
def _arr(records, path):
    a, b = path
    return np.array([r[a][b] for r in records], np.float64)


def calibrate(records) -> dict:
    corr_chosen = np.array([r["corr"]["chosen"] for r in records])          # [N, R, W]
    slot = np.array([r["slot"] for r in records])
    hit = corr_chosen == slot[:, None, None]
    hit_w0 = hit[:, :, 0].mean(0)
    rule_i = int(np.argmax(hit_w0))
    top_dh = _arr(records, ("corr", "top_dh"))                            # [N, R, W]
    mean_dh = np.nanmean(top_dh[:, rule_i, :], axis=0)
    w = pareto_weight(WS, mean_dh, PARETO_KEEP)
    wi = int(np.flatnonzero(WS == w)[0])
    s_real = _arr(records, ("real", "s_star"))
    s_corr = _arr(records, ("corr", "s_star"))
    tau = {}
    for ri, rule in enumerate(RULES):
        t, ba, fpr, tpr = balanced_threshold(s_real[:, ri, wi], s_corr[:, ri, wi])
        tau[rule] = {"tau": t, "balanced_accuracy": ba, "false_alarm": fpr, "detection": tpr}
    return {"rule": RULES[rule_i], "w_t": float(w), "w_h": float(1 - w), "tau": tau, "eps": EPS_DETECTION_LIMIT,
            "rule_hit_rate_w0": dict(zip(RULES, hit_w0.tolist())), "sweep_mean_dh": mean_dh.tolist()}


def summarise(records, rule: str, w: float, tau: float, n_boot: int = 2000) -> dict:
    ri, wi = RULES.index(rule), int(np.flatnonzero(WS == w)[0])
    w1 = len(WS) - 1
    slot = np.array([r["slot"] for r in records])
    npieces = np.array([r["n_pieces"] for r in records])
    chosen = np.array([r["corr"]["chosen"] for r in records])
    nrank = np.array([r["n_rank"] for r in records], np.float64)
    orig_rank = np.array([r["orig_rank"] for r in records], np.float64)
    hits = {
        "M4": chosen[:, ri, wi] == slot,
        "trend-only": chosen[:, ri, w1] == slot,
        "CLIP coherence": np.array([r["clip_pick"] for r in records]) == slot,
        "random": np.array([r["rand_pick"] for r in records]) == slot,
    }
    ranks = {
        "M4": orig_rank[:, wi],
        "trend-only": orig_rank[:, w1],
        "CLIP coherence": np.array([r["clip_rank"] for r in records], np.float64),
        "random": np.array([r["rand_rank"] for r in records], np.float64),
    }
    out = {"n_outfits": len(records), "rule": rule, "w_t": w, "tau": tau,
           "chance_hit_rate": float(np.mean(1.0 / npieces)), "methods": {}}
    for m in hits:
        h, rk = hits[m], ranks[m]
        pct = rk / nrank
        d = {"hit_rate": float(h.mean()), "hit_rate_ci": bootstrap_ci(h, np.mean, n_boot),
             "median_pct_rank": float(np.median(pct)), "median_pct_rank_ci": bootstrap_ci(pct, np.median, n_boot)}
        for k in KS:
            rk_k = rk <= k
            d[f"recall@{k}"] = float(rk_k.mean())
            d[f"recall@{k}_ci"] = bootstrap_ci(rk_k, np.mean, n_boot)
        if m != "M4":
            d["hit_rate_vs_M4_p"] = mcnemar_p(hits["M4"], h)
            diff = np.stack([ranks["M4"] / nrank, pct], 1)
            d["pct_rank_diff_ci"] = bootstrap_ci(diff, lambda x: np.median(x[:, 0]) - np.median(x[:, 1]), n_boot)
        out["methods"][m] = d
    # severity strata and swap groups (M4)
    sev = np.array([r["severity"] for r in records])
    strata = severity_strata(sev)
    out["by_severity"] = {name: {"n": int((strata == k).sum()), "hit_rate": float(hits["M4"][strata == k].mean()),
                                 "mean_severity": float(sev[strata == k].mean())}
                          for k, name in enumerate(("subtle", "medium", "obvious"))}
    groups = np.array([r["slot_group"] for r in records])
    out["detection_by_drop"] = []
    for lo, hi in zip(DROP_BINS[:-1], DROP_BINS[1:]):
        sel = (sev >= lo) & (sev < hi)
        if sel.any():
            out["detection_by_drop"].append({"from": lo, "to": hi, "n": int(sel.sum()),
                                             "hit_rate": float(hits["M4"][sel].mean()),
                                             "hit_rate_ci": bootstrap_ci(hits["M4"][sel], np.mean, n_boot),
                                             "chance": float(np.mean(1.0 / npieces[sel]))})
    out["by_group"] = {g: {"n": int((groups == g).sum()), "hit_rate": float(hits["M4"][groups == g].mean()),
                           "recall@10": float((ranks["M4"][groups == g] <= 10).mean())}
                       for g in sorted(set(groups))}
    # harmony recovery (GNN-judged, circular): corrupted -> M4 fix vs corrupted -> original
    top_logit = _arr(records, ("corr", "top_logit"))[:, ri, wi]
    r_corr = np.array([r["r_corr"] for r in records])
    r_orig = np.array([r["r_orig"] for r in records])
    out["harmony_recovery_logit"] = {"fix_minus_corrupted": float(np.nanmean(top_logit - r_corr)),
                                     "original_minus_corrupted": float(np.mean(r_orig - r_corr)),
                                     "fix_at_least_original": float(np.nanmean(top_logit >= r_orig))}
    # verdict rates at τ
    s_real = _arr(records, ("real", "s_star"))[:, ri, wi]
    s_corr = _arr(records, ("corr", "s_star"))[:, ri, wi]
    out["verdict"] = {"false_alarm_on_real": float((s_real >= tau).mean()),
                      "detection_on_corrupted": float((s_corr >= tau).mean())}
    # weight sweep and rule ablation (no extra GNN calls)
    top_dh, top_dt = _arr(records, ("corr", "top_dh")), _arr(records, ("corr", "top_dt"))
    out["weight_sweep"] = [{"w_t": float(x), "mean_dH": float(np.nanmean(top_dh[:, ri, j])),
                            "mean_dT": float(np.nanmean(top_dt[:, ri, j])),
                            "hit_rate": float((chosen[:, ri, j] == slot).mean()),
                            "recall@10": float((orig_rank[:, j] <= 10).mean())} for j, x in enumerate(WS)]
    out["rule_ablation"] = {r: float((chosen[:, k, wi] == slot).mean()) for k, r in enumerate(RULES)}
    return out


def markdown(summary: dict, calib: dict | None, set_name: str, source: str = SOURCE) -> str:
    s = summary
    L = [f"# Module 4 evaluation: {set_name} set", "",
         f"{s['n_outfits']} {source} test outfits · rule `{s['rule']}` · w_t = {s['w_t']} · τ = {s['tau']:.4f}",
         "", "## Weak link and original recovery", "",
         "| Method | Hit rate (95% CI) | Recall@1 | Recall@3 | Recall@10 (95% CI) | Median % rank | p (hit vs M4) |",
         "|---|---|---|---|---|---|---|"]
    for m, d in s["methods"].items():
        p = "" if m == "M4" else f"{d['hit_rate_vs_M4_p']:.2g}"
        L.append(f"| {m} | {d['hit_rate']:.3f} ({d['hit_rate_ci'][0]:.3f}–{d['hit_rate_ci'][1]:.3f}) | "
                 f"{d['recall@1']:.3f} | {d['recall@3']:.3f} | {d['recall@10']:.3f} "
                 f"({d['recall@10_ci'][0]:.3f}–{d['recall@10_ci'][1]:.3f}) | {100 * d['median_pct_rank']:.1f}% | {p} |")
    L += ["", f"Chance hit rate (mean 1 / pieces): {s['chance_hit_rate']:.3f}.", "",
          "## By corruption severity (M4)", "", "| Stratum | n | Mean drop (σ) | Hit rate |", "|---|---|---|---|"]
    L += [f"| {k} | {v['n']} | {v['mean_severity']:.2f} | {v['hit_rate']:.3f} |" for k, v in s["by_severity"].items()]
    L += ["", "## Detection by harmony drop (M4)", "", "| Drop (σ) | n | Hit rate (95% CI) | Chance |", "|---|---|---|---|"]
    L += [f"| {x['from']:g} to {x['to']:g} | {x['n']} | {x['hit_rate']:.3f} ({x['hit_rate_ci'][0]:.2f}–{x['hit_rate_ci'][1]:.2f}) "
          f"| {x['chance']:.3f} |" for x in s["detection_by_drop"]]
    L += ["", "## By swap group (M4)", "", "| Group | n | Hit rate | Recall@10 |", "|---|---|---|---|"]
    L += [f"| {k} | {v['n']} | {v['hit_rate']:.3f} | {v['recall@10']:.3f} |" for k, v in s["by_group"].items()]
    h = s["harmony_recovery_logit"]
    v = s["verdict"]
    L += ["", "## Harmony recovery (GNN-judged, circular)", "",
          f"Mean logit gain of M4's fix: {h['fix_minus_corrupted']:.3f}; of the original: "
          f"{h['original_minus_corrupted']:.3f}; fix ≥ original in {100 * h['fix_at_least_original']:.1f}% of outfits.",
          "", "## Verdicts at τ", "",
          f"False alarms on real outfits: {100 * v['false_alarm_on_real']:.1f}% (upper bound: some real outfits "
          f"are improvable). Detection on corrupted outfits: {100 * v['detection_on_corrupted']:.1f}%.",
          "", "## Trend-weight sweep", "", "| w_t | Mean ΔH | Mean ΔT | Hit rate | Recall@10 |", "|---|---|---|---|---|"]
    L += [f"| {x['w_t']:.1f} | {x['mean_dH']:.3f} | {x['mean_dT']:.3f} | {x['hit_rate']:.3f} | {x['recall@10']:.3f} |"
          for x in s["weight_sweep"]]
    L += ["", "## Selection-rule ablation (hit rate)", ""]
    L += [f"- `{r}`: {x:.3f}" for r, x in s["rule_ablation"].items()]
    if calib:
        L += ["", "## Calibration (dev)", "",
              f"- Rule: `{calib['rule']}` (hit rate at w = 0: "
              + ", ".join(f"{k} {x:.3f}" for k, x in calib["rule_hit_rate_w0"].items()) + ")",
              f"- w_t = {calib['w_t']} (largest weight keeping ≥ {int(100 * PARETO_KEEP)}% of the w = 0 harmony gain)",
              f"- ε = {calib['eps']}σ (M4's detection limit; see \"Detection by harmony drop\")"]
        L += [f"- τ[{r}] = {x['tau']:.4f} (balanced accuracy {x['balanced_accuracy']:.3f}, false alarms "
              f"{100 * x['false_alarm']:.1f}%, detection {100 * x['detection']:.1f}%)" for r, x in calib["tau"].items()]
    return "\n".join(L) + "\n"


# ---------------- CLI ----------------
def run_records(rec, keys, seed, label, source: str = SOURCE):
    out, skipped, t0 = [], 0, time.time()
    for j, key in enumerate(keys):
        r = evaluate_outfit(rec, key, np.random.default_rng([seed, int(key)]), source)
        if r is None:
            skipped += 1
        else:
            out.append(r)
        if (j + 1) % 100 == 0 or j + 1 == len(keys):
            el = time.time() - t0
            print(f"[eval {label}] {j + 1}/{len(keys)}  {el:.0f}s elapsed, ~{el / (j + 1) * (len(keys) - j - 1):.0f}s left",
                  flush=True)
    if skipped:
        print(f"[eval {label}] skipped {skipped} outfits with no swappable piece")
    return out


def main(argv=None):
    from recommender import Recommender
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["dev", "report"], required=True)
    ap.add_argument("--n", type=int, default=2000, help="outfits per set")
    ap.add_argument("--limit", type=int, help="evaluate only the first N outfits of the set (smoke runs; never frozen)")
    ap.add_argument("--freeze", action="store_true", help="dev: write the calibrated values to m4_settings.json")
    ap.add_argument("--force", action="store_true", help="report: allow a re-run")
    ap.add_argument("--source", default=SOURCE, help="store whose test outfits are evaluated (candidates come from every store)")
    ap.add_argument("--device")
    args = ap.parse_args(argv)
    if args.source != SOURCE and args.freeze:
        ap.error("--freeze calibrates on the fashion144k dev set only")

    settings = M4Settings.load()
    out_dir = settings.abs(os.path.join(settings.cache_dir, "eval"))
    os.makedirs(out_dir, exist_ok=True)
    prefix = "" if args.source == SOURCE else f"{args.source}_"
    tag = prefix + args.set + (f"_limit{args.limit}" if args.limit else "")
    done = os.path.join(out_dir, f"{prefix}report_done.json")
    if args.set == "report" and not args.limit:
        if settings.provisional():
            sys.exit(f"Settings still provisional ({', '.join(settings.provisional())}): run --set dev --freeze first")
        if os.path.exists(done) and not args.force:
            sys.exit("The report set has already been run once; pass --force to run it again")

    rec = Recommender.from_settings(settings, args.device)
    dev_keys, report_keys = eval_sets(rec.pool, args.n, source=args.source)
    keys = dev_keys if args.set == "dev" else report_keys
    if args.limit:
        keys = keys[:args.limit]
    records = run_records(rec, keys, SEED, tag, args.source)
    with open(os.path.join(out_dir, f"{tag}_records.json"), "w", encoding="utf-8") as f:
        json.dump(records, f)

    calib = None
    if args.set == "dev":
        calib = calibrate(records)
        rule, w, tau = calib["rule"], calib["w_t"], calib["tau"][calib["rule"]]["tau"]
    else:
        rule, w, tau = settings.rule, settings.w_t, float(settings.tau[settings.rule])
        if abs((1 - w) - settings.w_h) > 1e-9:
            sys.exit(f"Settings w_h = {settings.w_h} but the evaluation mixes weights as w_h = 1 - w_t")
    summary = summarise(records, rule, w, tau)
    summary["calibration"] = calib
    with open(os.path.join(out_dir, f"{tag}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    md = markdown(summary, calib, tag, args.source)
    with open(os.path.join(out_dir, f"{tag}_report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)

    if args.set == "dev" and args.freeze and not args.limit:
        new = settings.with_(w_h=calib["w_h"], w_t=calib["w_t"], rule=calib["rule"],
                             tau={r: calib["tau"][r]["tau"] for r in RULES}, eps=calib["eps"],
                             status={**settings.status, "w_t": "calibrated", "rule": "calibrated",
                                     "tau": "calibrated", "eps": "calibrated"})
        new.save()
        print("[eval] calibrated values written to Module4/config/m4_settings.json")
    if args.set == "report" and not args.limit:
        with open(done, "w", encoding="utf-8") as f:
            json.dump({"n_outfits": len(records), "rule": rule, "w_t": w, "tau": tau}, f)


if __name__ == "__main__":
    main()
