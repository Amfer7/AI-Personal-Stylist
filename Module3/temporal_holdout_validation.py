"""
temporal_holdout_validation.py

Backtests the trend model's growth signal: fit clusters and compute a
"predicted growth" score using ONLY data up to a cutoff date, then check
whether clusters flagged as high-growth actually kept growing in the
held-out months that follow — i.e. does the model's growth signal have any
real predictive validity, or is it just describing noise in retrospect.

This is the trend-model equivalent of Module 2's "real outfit > corrupted
outfit" ranking test: a known-answer check with a quantitative pass/fail
number, not just a qualitative sanity check.

Requires: pinterest_embeddings.npy + the metadata_csv used to generate it
(both already produced by build_from_directory() in a prior run — this
script does NOT re-encode anything, it only re-clusters and re-scores using
the cached embeddings).

Usage (Colab):
    python temporal_holdout_validation.py \
        --metadata_csv /content/pinterest_final_dataset/pinterest_final_dataset/pinterest_dataset/metadata.csv \
        --embeddings /content/drive/MyDrive/Poster/pinterest_embeddings.npy \
        --n_clusters 30
"""

import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import spearmanr

MIN_CLUSTER_IMAGES_FOR_TEST = 15  # exclude clusters too small to trust growth numbers from
WINDOW_MONTHS = 3                  # matches trend_model.py's GROWTH_WINDOW_MONTHS


def _parse_ym(ym: str):
    year, month = ym.split("-")
    return int(year), int(month)


def _month_diff(a, b) -> int:
    return (a[0] - b[0]) * 12 + (a[1] - b[1])


def _shift_months(ym: str, delta: int) -> str:
    year, month = _parse_ym(ym)
    total = year * 12 + (month - 1) + delta
    return f"{total // 12}-{(total % 12) + 1:02d}"


def load_aligned(metadata_csv: str, embeddings_path: str, image_col="image_path", date_col="date_ym"):
    """Reproduces the exact same filtering build_from_directory() used, so
    row i in the returned dataframe matches row i in the embeddings array."""
    df = pd.read_csv(metadata_csv)
    df = df.dropna(subset=[image_col, date_col]).reset_index(drop=True)
    embeddings = np.load(embeddings_path)
    assert len(df) == len(embeddings), (
        f"Row count mismatch: metadata has {len(df)} dated rows, embeddings "
        f"has {len(embeddings)}. Are you pointing at the same metadata_csv "
        f"that generated this embeddings file?"
    )
    return df, embeddings


def growth_ratio(prior_mean: float, recent_mean: float) -> float:
    if prior_mean < 1e-6:
        return float(recent_mean > 0)
    return float((recent_mean - prior_mean) / prior_mean)


def monthly_counts(months) -> dict:
    counts = {}
    for m in months:
        counts[m] = counts.get(m, 0) + 1
    return counts


def window_mean(counts: dict, months_in_window) -> float:
    return float(np.mean([counts.get(m, 0) for m in months_in_window])) if months_in_window else 0.0


def run_holdout_validation(df: pd.DataFrame, embeddings: np.ndarray, n_clusters: int, date_col="date_ym"):
    all_months = sorted(df[date_col].unique())
    if len(all_months) < WINDOW_MONTHS * 3:
        raise ValueError(
            f"Only {len(all_months)} distinct months in the data — need at "
            f"least {WINDOW_MONTHS * 3} (prior window + recent/train window "
            f"+ holdout window) for a meaningful backtest."
        )

    # Holdout = last WINDOW_MONTHS months. Cutoff = everything before that.
    holdout_months = all_months[-WINDOW_MONTHS:]
    cutoff_month = holdout_months[0]
    train_months = [m for m in all_months if m < cutoff_month]
    recent_train_months = train_months[-WINDOW_MONTHS:]
    prior_train_months = train_months[-2 * WINDOW_MONTHS : -WINDOW_MONTHS]

    print(f"Total months in data: {len(all_months)} ({all_months[0]} to {all_months[-1]})")
    print(f"  Prior window (train):        {prior_train_months}")
    print(f"  Recent window (train, 'now'): {recent_train_months}")
    print(f"  Holdout window (the 'future' we check against): {holdout_months}")
    print()

    train_mask = df[date_col] < cutoff_month
    holdout_mask = df[date_col] >= cutoff_month

    # Total dataset volume per month (ALL clusters combined) — used to compute
    # each cluster's SHARE of monthly volume, which cancels out any global
    # scrape-recency spike affecting every cluster equally.
    total_counts = monthly_counts(df[date_col].tolist())

    train_embeddings = embeddings[train_mask.values]
    print(f"Fitting KMeans(k={n_clusters}) on {len(train_embeddings)} train-only embeddings "
          f"(holdout data NOT used for clustering)...")
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(train_embeddings)

    # Assign every image (train AND holdout) to its nearest train-fitted centroid
    all_labels = km.predict(embeddings)
    df = df.copy()
    df["cluster"] = all_labels

    rows = []
    for cid in range(n_clusters):
        cluster_df = df[df["cluster"] == cid]
        train_cluster_months = cluster_df.loc[train_mask, date_col].tolist()
        holdout_cluster_months = cluster_df.loc[holdout_mask, date_col].tolist()
        n_train_images = len(train_cluster_months)

        counts = monthly_counts(train_cluster_months + holdout_cluster_months)

        prior_mean = window_mean(counts, prior_train_months)
        recent_mean = window_mean(counts, recent_train_months)
        holdout_mean = window_mean(counts, holdout_months)

        predicted_growth = growth_ratio(prior_mean, recent_mean)   # what the model would have said, back then
        actual_future_growth = growth_ratio(recent_mean, holdout_mean)  # what actually happened next

        # --- Share-normalized version: cluster's fraction of TOTAL monthly
        # volume, instead of raw counts. Cancels out a dataset-wide spike
        # (e.g. scrape-recency bias) that would otherwise inflate every
        # cluster's growth roughly equally and confound relative comparisons.
        prior_share = window_mean(counts, prior_train_months) / max(window_mean(total_counts, prior_train_months), 1e-6)
        recent_share = window_mean(counts, recent_train_months) / max(window_mean(total_counts, recent_train_months), 1e-6)
        holdout_share = window_mean(counts, holdout_months) / max(window_mean(total_counts, holdout_months), 1e-6)

        predicted_growth_norm = growth_ratio(prior_share, recent_share)
        actual_future_growth_norm = growth_ratio(recent_share, holdout_share)

        rows.append({
            "cluster_id": cid,
            "n_train_images": n_train_images,
            "prior_mean": prior_mean,
            "recent_mean": recent_mean,
            "holdout_mean": holdout_mean,
            "predicted_growth": predicted_growth,
            "actual_future_growth": actual_future_growth,
            "predicted_growth_norm": predicted_growth_norm,
            "actual_future_growth_norm": actual_future_growth_norm,
        })

    result_df = pd.DataFrame(rows)
    trusted = result_df[result_df["n_train_images"] >= MIN_CLUSTER_IMAGES_FOR_TEST].copy()
    excluded = len(result_df) - len(trusted)
    if excluded:
        print(f"Excluding {excluded} cluster(s) with < {MIN_CLUSTER_IMAGES_FOR_TEST} train images "
              f"(too sparse to trust growth numbers from).\n")

    corr, pval = spearmanr(trusted["predicted_growth"], trusted["actual_future_growth"])
    corr_norm, pval_norm = spearmanr(trusted["predicted_growth_norm"], trusted["actual_future_growth_norm"])

    trusted_sorted = trusted.sort_values("predicted_growth", ascending=False)
    n_top = max(1, len(trusted_sorted) // 5)  # top/bottom ~20%
    top_predicted = trusted_sorted.head(n_top)
    bottom_predicted = trusted_sorted.tail(n_top)

    top_actual_avg = top_predicted["actual_future_growth"].mean()
    bottom_actual_avg = bottom_predicted["actual_future_growth"].mean()

    trusted_sorted_norm = trusted.sort_values("predicted_growth_norm", ascending=False)
    top_predicted_norm = trusted_sorted_norm.head(n_top)
    bottom_predicted_norm = trusted_sorted_norm.tail(n_top)
    top_actual_avg_norm = top_predicted_norm["actual_future_growth_norm"].mean()
    bottom_actual_avg_norm = bottom_predicted_norm["actual_future_growth_norm"].mean()

    return {
        "result_df": trusted.sort_values("predicted_growth", ascending=False).reset_index(drop=True),
        "spearman_corr": corr,
        "spearman_pval": pval,
        "spearman_corr_norm": corr_norm,
        "spearman_pval_norm": pval_norm,
        "n_clusters_tested": len(trusted),
        "top_predicted_actual_growth_avg": top_actual_avg,
        "bottom_predicted_actual_growth_avg": bottom_actual_avg,
        "top_predicted_actual_growth_avg_norm": top_actual_avg_norm,
        "bottom_predicted_actual_growth_avg_norm": bottom_actual_avg_norm,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--n_clusters", type=int, default=30)
    args = parser.parse_args()

    df, embeddings = load_aligned(args.metadata_csv, args.embeddings)
    out = run_holdout_validation(df, embeddings, args.n_clusters)

    print("=" * 70)
    print("TEMPORAL HOLDOUT VALIDATION RESULTS")
    print("=" * 70)
    print(f"Clusters tested (sufficient data): {out['n_clusters_tested']}")
    print()
    print("--- RAW growth (susceptible to a dataset-wide volume spike) ---")
    print(f"Spearman correlation (predicted vs. actual future growth): "
          f"{out['spearman_corr']:.3f}  (p={out['spearman_pval']:.4f})")
    print(f"Top-predicted-growth clusters -> average ACTUAL future growth: "
          f"{out['top_predicted_actual_growth_avg']:+.2f}")
    print(f"Bottom-predicted-growth clusters -> average ACTUAL future growth: "
          f"{out['bottom_predicted_actual_growth_avg']:+.2f}")
    print()
    print("--- SHARE-NORMALIZED growth (cluster's share of total monthly volume) ---")
    print(f"Spearman correlation (predicted vs. actual future growth): "
          f"{out['spearman_corr_norm']:.3f}  (p={out['spearman_pval_norm']:.4f})")
    print(f"Top-predicted-growth clusters -> average ACTUAL future growth: "
          f"{out['top_predicted_actual_growth_avg_norm']:+.2f}")
    print(f"Bottom-predicted-growth clusters -> average ACTUAL future growth: "
          f"{out['bottom_predicted_actual_growth_avg_norm']:+.2f}")
    print()
    print("Full table (sorted by RAW predicted growth, highest first):")
    print(out["result_df"].to_string(index=False))

    return out


if __name__ == "__main__":
    main()
