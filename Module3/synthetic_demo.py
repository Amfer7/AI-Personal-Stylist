"""
synthetic_demo.py

Sanity-checks trend_model.py WITHOUT needing the real Pinterest scrape or
even CLIP/torch installed — it fabricates fake "style clusters" directly in
embedding space with different growth patterns (growing / flat / declining)
and checks that the model recovers the right ranking.

Run: python synthetic_demo.py

When the real data is ready, throw this file away (or keep it as a
regression test) and call trend_model.build_from_directory(metadata_csv=...)
instead — nothing in trend_model.py needs to change.
"""

import numpy as np

from trend_model import TrendModel

RNG = np.random.default_rng(42)
EMBED_DIM = 512


def random_unit_vector(dim: int) -> np.ndarray:
    v = RNG.normal(size=dim)
    return v / np.linalg.norm(v)


def make_synthetic_cluster(
    centroid: np.ndarray,
    monthly_image_counts: list[tuple[str, int]],
    noise: float = 0.05,
) -> tuple[np.ndarray, list[str]]:
    """monthly_image_counts: list of (year_month, n_images) tuples."""
    embeds = []
    months = []
    for ym, n in monthly_image_counts:
        for _ in range(n):
            v = centroid + RNG.normal(scale=noise, size=EMBED_DIM)
            v = v / np.linalg.norm(v)
            embeds.append(v)
            months.append(ym)
    return np.array(embeds, dtype=np.float32), months


def main():
    # Three synthetic style archetypes with clearly different trajectories.
    growing_centroid = random_unit_vector(EMBED_DIM)
    flat_centroid = random_unit_vector(EMBED_DIM)
    declining_centroid = random_unit_vector(EMBED_DIM)

    months_sequence = [f"2025-{m:02d}" for m in range(1, 13)] + ["2026-01", "2026-02"]

    # Growing: ramps up sharply in the last few months
    growing_counts = list(zip(months_sequence, [2, 3, 3, 4, 5, 5, 6, 8, 10, 14, 20, 28, 35, 40]))
    # Flat: roughly constant volume throughout
    flat_counts = list(zip(months_sequence, [15] * 14))
    # Declining: was popular, tapering off
    declining_counts = list(zip(months_sequence, [40, 38, 35, 30, 25, 20, 15, 12, 9, 7, 5, 4, 3, 2]))

    growing_emb, growing_months = make_synthetic_cluster(growing_centroid, growing_counts)
    flat_emb, flat_months = make_synthetic_cluster(flat_centroid, flat_counts)
    declining_emb, declining_months = make_synthetic_cluster(declining_centroid, declining_counts)

    all_embeddings = np.concatenate([growing_emb, flat_emb, declining_emb], axis=0)
    all_months = growing_months + flat_months + declining_months

    model = TrendModel(n_clusters=3)
    model.fit(all_embeddings, all_months)

    print("=== Cluster summary (sorted by Trendiness Score) ===")
    print(model.summary().to_string(index=False))

    print("\n=== Sanity check: scoring a fresh garment from each archetype ===")
    for name, centroid in [
        ("growing-style garment", growing_centroid),
        ("flat-style garment", flat_centroid),
        ("declining-style garment", declining_centroid),
    ]:
        test_vec = centroid + RNG.normal(scale=0.05, size=EMBED_DIM)
        test_vec = test_vec / np.linalg.norm(test_vec)
        result = model.score_embedding(test_vec)
        print(f"{name}: {result}")

    print("\n=== Sanity check: garment that matches nothing in the dataset ===")
    random_vec = random_unit_vector(EMBED_DIM)
    print(model.score_embedding(random_vec))

    model.save("synthetic_trend_model.pkl")
    print("\nSaved model to synthetic_trend_model.pkl")


if __name__ == "__main__":
    main()
