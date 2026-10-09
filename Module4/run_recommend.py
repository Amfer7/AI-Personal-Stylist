"""Module 4 from the command line: photo (runs Module 1) or an M1 outfit folder -> recommendation.

    python Module4/run_recommend.py --photo samples/image2.jpg
    python Module4/run_recommend.py --outfit_dir <M1 output folder> --pool eval --json rec.json
"""
from __future__ import annotations

import argparse
import os
import sys
import time

for _s in (sys.stdout, sys.stderr):           # arrows/ticks on a Windows console
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from m4_settings import REPO, M4Settings  # noqa: E402
from outfit_state import load_outfit_dir  # noqa: E402
from recommendation import format_text  # noqa: E402
from recommender import Recommender  # noqa: E402
from selection import RULES  # noqa: E402


def segment_photo(photo: str, out_root: str, device=None) -> str:
    """Run Module 1 on one photo; returns its outfit folder (<out_root>/<photo stem>)."""
    m1 = os.path.join(REPO, "Module1")
    if m1 not in sys.path:
        sys.path.insert(0, m1)
    from run_pipeline import run_single_image
    image_id = os.path.splitext(os.path.basename(photo))[0]
    run_single_image(image_path=photo, image_id=image_id, output_root=out_root, device=device)
    return os.path.join(out_root, image_id)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--photo")
    src.add_argument("--outfit_dir")
    ap.add_argument("--pool", choices=["demo", "eval"], default="demo")
    ap.add_argument("--rule", choices=RULES)
    ap.add_argument("--w-t", type=float, dest="w_t")
    ap.add_argument("--same-type", action="store_true", help="strict like-for-like swaps")
    ap.add_argument("--no-trend-drop", action="store_true",
                    help="restore the hard trend floor (off by default, D5)")
    ap.add_argument("--json", help="write the Recommendation JSON here")
    ap.add_argument("--device")
    args = ap.parse_args(argv)

    s = M4Settings.load()
    rec = Recommender.from_settings(s, args.device)
    outfit_dir = args.outfit_dir or segment_photo(args.photo, s.abs(os.path.join(s.cache_dir, "runs")), args.device)
    t0 = time.time()
    state = load_outfit_dir(outfit_dir, rec.gnn, rec.trend_model)
    table = rec.score(state, mode=args.pool, same_type=args.same_type or None,
                      trend_floor=args.no_trend_drop or None)
    r = rec.decide(table, w_t=args.w_t, rule=args.rule)
    n = sum(len(p.rows) for p in table.pieces)
    print(format_text(r))
    print(f"[m4] {len(state.pieces)} pieces, {n} what-if outfits scored in {time.time() - t0:.2f}s")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(r.to_json())


if __name__ == "__main__":
    main()
