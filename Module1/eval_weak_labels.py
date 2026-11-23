"""
eval_weak_labels.py
===================
Module 1 recall on Fashion144k, with the photo file names as weak labels.

Fashion144k file names carry the poster's own garment words, e.g.
`10000078429-tawny-stilettos-vintage-shoes-red-polyester-vintage-dress-beige-beaded-purse_400.jpg`.
A garment word in the title is taken as "this item is in the outfit"; recall per type is the share
of those photos where Module 1 found that type (exact) or any type of the same swap group (group:
upper / lower / dress / shoe / bag / hat / belt / scarf / sunglasses).

Caveats (weak labels):
  - Titles name items that may be out of frame, hidden (a coat over the arm, a bag behind the back)
    or only in the shop list, so recall is a lower bound.
  - Titles omit many items, so there is no precision.
  - Module 1 gives one garment per body region: a jacket over a shirt is one upper-body piece, so a
    title naming both can only be half right at the exact level. Group recall is the fairer number
    for upper-body clothes; the exact subtype comes from CLIP zero-shot.
  - Tights / stockings have no Module 1 type (SegFormer labels them as legs); reported separately.

Reads Module 1's existing outputs (outputs/<line index of photos.txt>/metadata.json); nothing is
re-segmented. Writes recall.csv, confusion.json and result.json to --out (gitignored).

    python Module1/eval_weak_labels.py [--root Fashion144k_v1] [--out Fashion144k_v1/eval_weak_labels]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict

# Title word -> Module 1 type. Conservative: brands, colours and ambiguous words are ignored.
KEYWORDS = {
    "dress": "dress", "dresses": "dress",
    "skirt": "skirt", "skirts": "skirt",
    "shorts": "shorts",
    "jeans": "pants", "pants": "pants", "trousers": "pants", "chinos": "pants", "slacks": "pants",
    "joggers": "pants",
    "leggings": "leggings",
    "jacket": "jacket", "blazer": "jacket", "coat": "jacket", "parka": "jacket", "trench": "jacket",
    "bomber": "jacket", "peacoat": "jacket",
    "sweater": "sweater", "jumper": "sweater", "cardigan": "sweater", "pullover": "sweater",
    "shirt": "shirt_blouse", "blouse": "shirt_blouse",
    "tshirt": "t_shirt", "tee": "t_shirt",
    "top": "top", "tank": "top", "cami": "top", "camisole": "top",
    "hat": "hat", "beanie": "hat", "cap": "hat", "fedora": "hat", "beret": "hat",
    "bag": "bag", "purse": "bag", "tote": "bag", "clutch": "bag", "backpack": "bag", "handbag": "bag",
    "satchel": "bag",
    "shoes": "shoe", "shoe": "shoe", "heels": "shoe", "boots": "shoe", "boot": "shoe", "booties": "shoe",
    "sneakers": "shoe", "trainers": "shoe", "sandals": "shoe", "flats": "shoe", "pumps": "shoe",
    "loafers": "shoe", "stilettos": "shoe", "wedges": "shoe", "oxfords": "shoe",
    "belt": "belt",
    "scarf": "scarf", "scarves": "scarf",
    "sunglasses": "sunglasses",
}
# Upper-body words with no clear Module 1 subtype: counted at the group level only.
GROUP_ONLY = {"vest": "upper", "sweatshirt": "upper", "hoodie": "upper", "tunic": "upper"}
LEGWEAR = {"tights", "stockings", "pantyhose"}

GROUPS = {"upper": ("t_shirt", "sweater", "shirt_blouse", "jacket", "top", "hoodie"),
          "lower": ("pants", "leggings", "shorts", "skirt"),
          "dress": ("dress",), "shoe": ("shoe",), "bag": ("bag",), "hat": ("hat",), "belt": ("belt",),
          "scarf": ("scarf",), "sunglasses": ("sunglasses",)}
GROUP_OF = {t: g for g, ts in GROUPS.items() for t in ts}


def tokens(filename: str) -> list:
    """Title words of a Fashion144k file name: drop the id and the `_400.jpg` suffix, split on '-',
    and join "t"+"shirt" into "tshirt"."""
    stem = os.path.splitext(os.path.basename(filename))[0].rsplit("_", 1)[0]
    words = [w for w in stem.lower().split("-")[1:] if w]
    out, i = [], 0
    while i < len(words):
        if words[i] == "t" and i + 1 < len(words) and words[i + 1] in ("shirt", "shirts"):
            out.append("tshirt")
            i += 2
            continue
        out.append(words[i])
        i += 1
    return out


def title_labels(filename: str) -> dict:
    """{"types": set of Module 1 types, "groups": set of swap groups, "legwear": bool}."""
    toks = tokens(filename)
    types = {KEYWORDS[t] for t in toks if t in KEYWORDS}
    groups = {GROUP_OF[t] for t in types} | {GROUP_ONLY[t] for t in toks if t in GROUP_ONLY}
    return {"types": types, "groups": groups, "legwear": any(t in LEGWEAR for t in toks)}


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def evaluate(names: list, found: list) -> dict:
    """names[i]: file name; found[i]: list of Module 1 types for that photo (None = no output)."""
    exact = defaultdict(lambda: [0, 0])          # type -> [hits, n]
    group = defaultdict(lambda: [0, 0])
    miss = defaultdict(Counter)                  # type -> what Module 1 had instead in its group
    legwear = [0, 0]
    for name, f in zip(names, found):
        if f is None:
            continue
        lab = title_labels(name)
        have = set(f)
        have_groups = {GROUP_OF.get(t) for t in have}
        for t in lab["types"]:
            exact[t][1] += 1
            if t in have:
                exact[t][0] += 1
            else:
                inside = sorted(x for x in have if GROUP_OF.get(x) == GROUP_OF[t])
                miss[t]["/".join(inside) if inside else ("nothing" if not have else "no " + GROUP_OF[t])] += 1
        for g in lab["groups"]:
            group[g][1] += 1
            group[g][0] += g in have_groups
        if lab["legwear"]:
            legwear[1] += 1
            legwear[0] += bool(have & {"leggings", "pants"})
    row = lambda k, n: {"hits": k, "n": n, "recall": k / n if n else float("nan"), "ci95": wilson(k, n)}
    return {"exact": {t: row(*v) for t, v in sorted(exact.items())},
            "group": {g: row(*v) for g, v in sorted(group.items())},
            "legwear_as_leggings_or_pants": row(*legwear),
            "misses": {t: dict(c.most_common(4)) for t, c in miss.items()}}


def load_found(root: str) -> tuple:
    with open(os.path.join(root, "photos.txt"), encoding="utf-8") as f:
        names = [l.strip() for l in f if l.strip()]
    found = []
    for i in range(len(names)):
        p = os.path.join(root, "outputs", str(i), "metadata.json")
        if not os.path.exists(p):
            found.append(None)
            continue
        with open(p, encoding="utf-8") as f:
            found.append([g.get("category") for g in json.load(f)])
    return names, found


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="Fashion144k_v1")
    ap.add_argument("--out", default=os.path.join("Fashion144k_v1", "eval_weak_labels"))
    args = ap.parse_args(argv)
    names, found = load_found(args.root)
    res = evaluate(names, found)
    res["photos"] = len(names)
    res["photos_with_output"] = sum(f is not None for f in found)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(args.out, "recall.csv"), "w", encoding="utf-8") as f:
        f.write("level,label,n,hits,recall,ci_low,ci_high\n")
        for level in ("group", "exact"):
            for k, v in res[level].items():
                f.write(f"{level},{k},{v['n']},{v['hits']},{v['recall']:.4f},{v['ci95'][0]:.4f},{v['ci95'][1]:.4f}\n")
    for level in ("group", "exact"):
        print(f"\n{level}")
        for k, v in res[level].items():
            print(f"  {k:13s} n {v['n']:6d}  recall {v['recall']:.3f}  ({v['ci95'][0]:.3f}-{v['ci95'][1]:.3f})")
    lw = res["legwear_as_leggings_or_pants"]
    print(f"\ntights/stockings -> leggings or pants: n {lw['n']}  {lw['recall']:.3f}")
    print("\nmain misses:", json.dumps(res["misses"], indent=1))


if __name__ == "__main__":
    main()
