"""
fit_per_category.py

Fits the PerCategoryTrendModel from a garment manifest + embeddings produced by
pinterest_garment_segmentation.py, and saves it to a .pkl the demos load.

This is the previously-inline "step 3" of the Module 3 pipeline, made into a
reproducible script. It only reads the manifest/embeddings and calls the
already-tested build_per_category_from_manifest() — no new modelling logic.

Usage:
    python fit_per_category.py \
        --manifest_csv ../PinterestData/garment_data/garment_manifest.csv \
        --embeddings   ../PinterestData/garment_data/garment_embeddings.npy \
        --out          per_category_trend_model.pkl
"""

import argparse

from trend_model import build_per_category_from_manifest, PerCategoryTrendModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", default="../PinterestData/garment_data/garment_manifest.csv")
    ap.add_argument("--embeddings", default="../PinterestData/garment_data/garment_embeddings.npy")
    ap.add_argument("--out", default="per_category_trend_model.pkl")
    ap.add_argument("--n_clusters_per_category", type=int, default=15,
                    help="default k; per-category overrides live in trend_model.N_CLUSTERS_OVERRIDES")
    ap.add_argument("--end_month", default="2026-08",
                    help="drop garments after this YYYY-MM (the scrape ended 2026-09-11, so "
                         "2026-09 is partial). Pass '' to keep everything.")
    ap.add_argument("--drop_fragments", action="store_true",
                    help="drop likely SegFormer fragments (see trend_model.fragment_mask)")
    args = ap.parse_args()

    model = build_per_category_from_manifest(
        manifest_csv=args.manifest_csv,
        embeddings_path=args.embeddings,
        n_clusters_per_category=args.n_clusters_per_category,
        end_month=args.end_month or None,
        drop_fragments=args.drop_fragments,
    )
    model.save(args.out)
    print(f"\nSaved per-category trend model -> {args.out}")
    print(f"Categories fitted: {sorted(model.models.keys())}\n")

    summary = model.summary()
    # Show the top few trend clusters per category
    for cat in sorted(model.models.keys()):
        sub = summary[summary["category"] == cat].sort_values("trendiness_score", ascending=False)
        print(f"=== {cat}: {len(sub)} clusters ===")
        print(sub.head(3).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
