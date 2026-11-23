"""Human study pilot (Review 3): do people prefer Module 4's swap to a random allowed swap?

    python Module4/scripts/human_study.py build [--n 20] [--seed 0]      # pair images, answer key, form text
    python Module4/scripts/human_study.py score --responses <form export.csv>

Design: N street test outfits for which Module 4 says "swap", half Fashion144k and half Pinterest,
every crop passing the crop quality gate and no outfit from an excluded search query. Each outfit is
shown twice: original vs Module 4's top suggestion, and original vs a random allowed replacement for
the same piece (drawn from the same candidate list, so the same H&M catalogue and photo style).
Raters see "Outfit A" over "Outfit B", sides randomised, and pick the one that goes together better,
or "Can't decide". The replacement is a studio cut-out in both conditions, so the random control
absorbs any liking for clean product photos; the result is the lift, the share preferring Module 4's
swap minus the share preferring the random swap, with a 95% bootstrap interval over raters.
Outputs go to <cache_dir>/human_study/ (gitignored): pairs/Qnn.png, key.csv, form.md, result.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

QUESTION = "Which outfit goes together better?"
CONDITIONS = ("m4", "random")


# ---------------- pair layout ----------------
def make_pairs(items: list, seed: int = 0) -> list:
    """items: one dict per (outfit, condition), two per outfit. Returns them in question order with
    "q" (Q01...) and "variant_side" (A/B, half each). The two versions of an outfit sit in different
    halves of the form and never back to back."""
    rng = np.random.default_rng(seed)
    by_outfit = {}
    for it in items:
        by_outfit.setdefault(it["outfit"], []).append(it)
    outfits = list(by_outfit)
    first_cond = {o: int(rng.integers(2)) for o in outfits}
    first = [by_outfit[outfits[i]][first_cond[outfits[i]]] for i in rng.permutation(len(outfits))]
    second = [by_outfit[outfits[i]][1 - first_cond[outfits[i]]] for i in rng.permutation(len(outfits))]
    if len(second) > 1 and second[0]["outfit"] == first[-1]["outfit"]:
        second = second[1:] + second[:1]
    ordered = first + second
    sides = np.array(["A"] * (len(ordered) // 2) + ["B"] * (len(ordered) - len(ordered) // 2))
    sides = sides[rng.permutation(len(sides))]
    return [{**it, "q": f"Q{i + 1:02d}", "variant_side": str(s)} for i, (it, s) in enumerate(zip(ordered, sides))]


# ---------------- scoring ----------------
def _long(responses: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    cols = {c: m.group(1) for c in responses.columns if (m := re.match(r"\s*(Q\d+)", str(c)))}
    rows = []
    for rater, (_, r) in enumerate(responses.iterrows()):
        for col, q in cols.items():
            a = str(r[col]).strip()
            side = a[-1] if a in ("Outfit A", "Outfit B") else None
            rows.append({"rater": rater, "q": q, "side": side})
    long = pd.DataFrame(rows).merge(key[["q", "condition", "variant_side"]], on="q")
    long["prefer"] = np.where(long["side"].isna(), np.nan, (long["side"] == long["variant_side"]).astype(float))
    return long


def _shares(long: pd.DataFrame) -> dict:
    return {c: long.loc[long["condition"] == c, "prefer"].mean() for c in CONDITIONS}


def score_responses(responses: pd.DataFrame, key: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    long = _long(responses, key)
    out = {"raters": int(long["rater"].nunique())}
    for c in CONDITIONS:
        sub = long[long["condition"] == c]
        out[c] = {"votes": int(sub["prefer"].notna().sum()), "undecided": int(sub["prefer"].isna().sum()),
                  "prefer_variant": float(sub["prefer"].mean())}
    out["lift"] = out["m4"]["prefer_variant"] - out["random"]["prefer_variant"]
    rng = np.random.default_rng(seed)
    raters = long["rater"].unique()
    groups = {r: g for r, g in long.groupby("rater")}
    boots = []
    for _ in range(n_boot):
        s = _shares(pd.concat([groups[r] for r in rng.choice(raters, len(raters))]))
        boots.append(s["m4"] - s["random"])
    boots = np.array(boots, float)
    boots = boots[~np.isnan(boots)]
    out["lift_ci95"] = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    return out


# ---------------- building the pairs (real data) ----------------
MIN_AREA = 4000      # px of garment in the crop: the slide figures' readability bar (render_recommendation)


def _readable(rec) -> np.ndarray:
    """Rows whose crop passes the crop gate (full CLIP checks for street photos), has at least
    MIN_AREA garment pixels, and whose outfit was not found by an excluded query."""
    from candidate_pool import excluded_by_query, pool_dir
    from crop_quality import crop_ok, load_metrics
    s, pool = rec.settings, rec.pool
    product = np.isin(pool.source, [i for i, st in enumerate(s.stores) if st.product_photos])
    metrics = load_metrics(s, pool_dir(s))
    ok = crop_ok(metrics, pool.type, s.crop_gate, shape_only=product) & (metrics["area"] >= MIN_AREA)
    meta = {"type": pool.type, "source": pool.source, "outfit_ord": pool.outfit_ord}
    return ok & pool.has_crop.astype(bool) & ~excluded_by_query(s, meta, pool.outfit_keys)


def select_outfits(rec, n: int, seed: int = 0) -> list:
    from harmony_reference import test_outfits
    from outfit_state import state_from_pool
    pool = rec.pool
    readable = _readable(rec)
    rng = np.random.default_rng(seed)
    cands = [rows for rows in test_outfits(pool) if len(rows) <= 5 and readable[rows].all()]
    cands = [cands[i] for i in rng.permutation(len(cands))]
    quota = {"fashion144k": n // 2, "pinterest": n - n // 2}
    chosen = []
    for rows in cands:
        src, key = pool.source_name(int(rows[0])), pool.outfit_key(int(rows[0]))
        if quota.get(src, 0) <= 0:
            continue
        state = state_from_pool(pool, src, key)
        table = rec.score(state, mode="demo", exclude=(src, key))
        r = rec.decide(table)
        if r.verdict != "swap" or not r.suggestions:
            continue
        j = [p.file for p in state.pieces].index(r.swap_piece["file"])
        ps = table.pieces[j]
        m4_row = r.suggestions[0].pool_row
        others = ps.rows[ps.rows != m4_row]
        if not len(others):
            continue
        k_rand = int(np.nonzero(ps.rows == others[int(rng.integers(len(others)))])[0][0])
        k_m4 = int(np.nonzero(ps.rows == m4_row)[0][0])
        file_row = {pool.files[int(x)]: int(x) for x in rows}
        chosen.append({"source": src, "key": key, "piece": j,
                       "originals": [file_row[p.file] for p in state.pieces],
                       "m4": (m4_row, float(ps.d_h[k_m4])), "random": (int(ps.rows[k_rand]), float(ps.d_h[k_rand]))})
        quota[src] -= 1
        print(f"  {src} {key}: swap {state.pieces[j].type}  ({len(chosen)}/{n})", flush=True)
        if len(chosen) == n:
            break
    return chosen


def _pair_image(rec, outfit, row, variant_side, out_path):
    from PIL import Image
    from outfit_board import outfit_board
    pool, s = rec.pool, rec.settings
    tiles = lambda rows: [(s.abs(pool.crop_path(r)), "", False) for r in rows]
    variant = list(outfit["originals"])
    variant[outfit["piece"]] = row
    boards = {}
    for side, rows in ((variant_side, variant), ("B" if variant_side == "A" else "A", outfit["originals"])):
        boards[side] = Image.open(outfit_board(tiles(rows), out_path + f".{side}.png", title=f"Outfit {side}",
                                                 upscale=True))
    w = max(b.width for b in boards.values())
    img = Image.new("RGB", (w, boards["A"].height + boards["B"].height), "white")
    img.paste(boards["A"], (0, 0))
    img.paste(boards["B"], (0, boards["A"].height))
    img.save(out_path)
    for side in "AB":
        os.remove(out_path + f".{side}.png")


def build(n: int, seed: int) -> str:
    from m4_settings import M4Settings
    from recommender import Recommender
    s = M4Settings.load()
    rec = Recommender.from_settings(s)
    out_dir = s.abs(os.path.join(s.cache_dir, "human_study"))
    os.makedirs(os.path.join(out_dir, "pairs"), exist_ok=True)
    outfits = select_outfits(rec, n, seed)
    items = [{"outfit": f"{o['source']}/{o['key']}", "condition": c, "idx": i}
             for i, o in enumerate(outfits) for c in CONDITIONS]
    pairs = make_pairs(items, seed)
    key_rows = []
    for p in pairs:
        o = outfits[p["idx"]]
        row, d_h = o[p["condition"]]
        _pair_image(rec, o, row, p["variant_side"], os.path.join(out_dir, "pairs", f"{p['q']}.png"))
        orig = o["originals"][o["piece"]]
        key_rows.append({"q": p["q"], "condition": p["condition"], "variant_side": p["variant_side"],
                         "source": o["source"], "outfit": o["key"], "from_type": rec.pool.type_name(orig),
                         "to_type": rec.pool.type_name(row), "to_item": rec.pool.outfit_key(row),
                         "d_harmony_z": round(d_h, 3)})
    pd.DataFrame(key_rows).to_csv(os.path.join(out_dir, "key.csv"), index=False)
    with open(os.path.join(out_dir, "form.md"), "w", encoding="utf-8") as f:
        f.write("# Form questions\n\nOne multiple-choice question per image, in this order. Title each "
                f"question exactly `Qnn. {QUESTION}`, attach `pairs/Qnn.png`, options: "
                "`Outfit A`, `Outfit B`, `Can't decide`. Do not share key.csv.\n\n")
        for p in pairs:
            f.write(f"- {p['q']}. {QUESTION}  (image pairs/{p['q']}.png)\n")
    print(f"{len(pairs)} questions in {out_dir}")
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--n", type=int, default=20)
    b.add_argument("--seed", type=int, default=0)
    sc = sub.add_parser("score")
    sc.add_argument("--responses", required=True)
    sc.add_argument("--key")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        build(args.n, args.seed)
        return
    from m4_settings import M4Settings
    s = M4Settings.load()
    out_dir = s.abs(os.path.join(s.cache_dir, "human_study"))
    key = pd.read_csv(args.key or os.path.join(out_dir, "key.csv"))
    res = score_responses(pd.read_csv(args.responses), key)
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
