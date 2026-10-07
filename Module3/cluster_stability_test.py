"""
cluster_stability_test.py

K-Means has no loss function or epochs to validate the way the GNN does —
there's nothing analogous to "training accuracy over epochs" for an
unsupervised clustering method. What IS a meaningful, standard test for a
clustering method is STABILITY: if you fit on a random 80% subset of the
data, do you get essentially the same clusters as fitting on 100%? If yes,
the clusters reflect real structure in the embeddings, not an artifact of
exactly which images happened to be included.

Method: fit KMeans on the FULL category dataset (the "production" clustering).
Then repeat K times: fit KMeans on a random 80% subset, predict cluster labels
for the held-out 20%, and compare those labels to the full model's labels for
the same points using Adjusted Rand Index (ARI). ARI = 1.0 means perfect
agreement; ARI near 0 means the clustering is no better than random.

Usage (after running pinterest_garment_segmentation.py):
    python cluster_stability_test.py \
        --manifest_csv ../PinterestData/garment_data/garment_manifest.csv \
        --embeddings ../PinterestData/garment_data/garment_embeddings.npy \
        --n_clusters_per_category 15 \
        --n_repeats 5

k per category follows trend_model.k_for_category (so dresses use the k=8
override), i.e. it tests exactly the clustering the saved model uses.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from trend_model import k_for_category


def stability_test_for_category(
    embeddings: np.ndarray,
    n_clusters: int,
    n_repeats: int = 5,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(embeddings)

    full_model = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(embeddings)
    full_labels = full_model.labels_

    ari_scores = []
    for repeat in range(n_repeats):
        perm = rng.permutation(n)
        train_size = int(n * train_fraction)
        train_idx = perm[:train_size]
        holdout_idx = perm[train_size:]

        subset_model = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed + repeat + 1).fit(
            embeddings[train_idx]
        )
        holdout_labels_from_subset = subset_model.predict(embeddings[holdout_idx])
        holdout_labels_from_full = full_labels[holdout_idx]

        ari = adjusted_rand_score(holdout_labels_from_full, holdout_labels_from_subset)
        ari_scores.append(ari)

    return {
        "n_images": n,
        "n_clusters": n_clusters,
        "mean_ari": float(np.mean(ari_scores)),
        "std_ari": float(np.std(ari_scores)),
        "ari_per_repeat": ari_scores,
    }


def run_all_categories(
    manifest_csv: str,
    embeddings_path: str,
    n_clusters_per_category: int,
    n_repeats: int,
    category_col: str = "broad_category",
    min_images: int = 100,
):
    manifest = pd.read_csv(manifest_csv)
    embeddings = np.load(embeddings_path)
    assert len(manifest) == len(embeddings), "manifest/embeddings row count mismatch"

    results = {}
    for cat in sorted(manifest[category_col].unique()):
        mask = (manifest[category_col] == cat).values
        n_images = int(mask.sum())
        if n_images < min_images:
            print(f"Skipping '{cat}': only {n_images} images (< {min_images})")
            continue

        k = k_for_category(cat, n_images, n_clusters_per_category)
        print(f"\nTesting category '{cat}' ({n_images} images, k={k})...")
        result = stability_test_for_category(embeddings[mask], n_clusters=k, n_repeats=n_repeats)
        results[cat] = result
        print(f"  Mean ARI: {result['mean_ari']:.3f} (std {result['std_ari']:.3f}) "
              f"across {n_repeats} random 80/20 splits")

    return results


def interpret_ari(ari: float) -> str:
    if ari >= 0.75:
        return "strong agreement — clusters are highly stable / reproducible"
    elif ari >= 0.5:
        return "moderate agreement — clusters are reasonably stable"
    elif ari >= 0.25:
        return "weak agreement — clusters are somewhat sensitive to sampling"
    else:
        return "little to no agreement — clustering may not reflect stable structure"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--n_clusters_per_category", type=int, default=15)
    parser.add_argument("--n_repeats", type=int, default=5)
    args = parser.parse_args()

    results = run_all_categories(
        args.manifest_csv, args.embeddings,
        n_clusters_per_category=args.n_clusters_per_category,
        n_repeats=args.n_repeats,
    )

    print("\n" + "=" * 70)
    print("CLUSTER STABILITY SUMMARY (80/20 split, Adjusted Rand Index)")
    print("=" * 70)
    for cat, r in results.items():
        print(f"{cat:12s}  mean ARI = {r['mean_ari']:.3f}  ->  {interpret_ari(r['mean_ari'])}")


if __name__ == "__main__":
    main()
