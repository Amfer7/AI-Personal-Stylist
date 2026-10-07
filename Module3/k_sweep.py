"""
k_sweep.py

Picks n_clusters for ONE garment category by sweeping k and reporting, per k:
  - cluster stability (mean ARI over 80/20 resamples — same test as
    cluster_stability_test.py, reused directly), and
  - silhouette score (on a random sample; full silhouette is O(n^2)).

Optionally drops fragmentary masks first (--drop_fragments), see
trend_model.fragment_mask() — e.g. a small "dress" mask that is really the
hem of a top SegFormer split in two.

Usage:
    python k_sweep.py --category dress --k 4 6 8 10 12 15 --drop_fragments
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from cluster_stability_test import stability_test_for_category
from trend_model import fragment_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", default="../PinterestData/garment_data/garment_manifest.csv")
    ap.add_argument("--embeddings", default="../PinterestData/garment_data/garment_embeddings.npy")
    ap.add_argument("--category", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[4, 6, 8, 10, 12, 15, 20])
    ap.add_argument("--n_repeats", type=int, default=3)
    ap.add_argument("--silhouette_sample", type=int, default=5000)
    ap.add_argument("--drop_fragments", action="store_true")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest_csv)
    embeddings = np.load(args.embeddings, mmap_mode="r")
    assert len(manifest) == len(embeddings), "manifest/embeddings row count mismatch"

    mask = (manifest["broad_category"] == args.category).values
    if args.drop_fragments:
        frag = fragment_mask(manifest)
        print(f"Dropping {int((mask & frag).sum())} fragmentary '{args.category}' masks")
        mask &= ~frag
    X = np.asarray(embeddings[mask], dtype=np.float32)
    print(f"Category '{args.category}': {len(X)} garments\n")

    rng = np.random.default_rng(0)
    sample = rng.choice(len(X), size=min(args.silhouette_sample, len(X)), replace=False)

    rows = []
    for k in args.k:
        r = stability_test_for_category(X, n_clusters=k, n_repeats=args.n_repeats)
        labels = KMeans(n_clusters=k, n_init=5, random_state=42).fit_predict(X)
        sil = float(silhouette_score(X[sample], labels[sample]))
        rows.append({"k": k, "mean_ari": round(r["mean_ari"], 3),
                     "std_ari": round(r["std_ari"], 3), "silhouette": round(sil, 4)})
        print(f"k={k:3d}  ARI={r['mean_ari']:.3f} (±{r['std_ari']:.3f})  silhouette={sil:.4f}", flush=True)

    print("\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
