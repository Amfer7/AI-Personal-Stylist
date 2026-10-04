"""
fit_per_category.py

Fits the PerCategoryTrendModel from a garment manifest + embeddings produced by
pinterest_garment_segmentation.py, and saves it to a .pkl the demos load.

This is the previously-inline "step 3" of the Module 3 pipeline, made into a
reproducible script. It only reads the manifest/embeddings and calls the
already-tested build_per_category_from_manifest() — no new modelling logic.

Usage:
    python fit_per_category.py \
        --manifest_csv ../PinterestLatest/garment_data/garment_manifest.csv \
        --embeddings   ../PinterestLatest/garment_data/garment_embeddings.npy \
        --out          per_category_trend_model.pkl
"""

import argparse

from trend_model import build_per_category_from_manifest, PerCategoryTrendModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", default="../PinterestLatest/garment_data/garment_manifest.csv")
    ap.add_argument("--embeddings", default="../PinterestLatest/garment_data/garment_embeddings.npy")
    ap.add_argument("--out", default="per_category_trend_model.pkl")
    ap.add_argument("--n_clusters_per_category", type=int, default=15)
    args = ap.parse_args()

    model = build_per_category_from_manifest(
        manifest_csv=args.manifest_csv,
        embeddings_path=args.embeddings,
        n_clusters_per_category=args.n_clusters_per_category,
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
