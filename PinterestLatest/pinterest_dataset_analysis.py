"""
Pinterest Dataset — Detailed Analysis
=======================================
Full-range analysis of metadata.csv (not just the last 30 months),
covering every query, with charts saved as PNG files.

Usage:
    python pinterest_dataset_analysis.py --dataset-dir <data_root>\\pinterest_dataset

Outputs (saved into --dataset-dir by default, override with --output-dir):
    monthly_distribution.png   - bar chart, full date_ym range, every month
    query_breakdown.png        - horizontal bar chart, all 67 queries
    query_monthly_heatmap.png  - query x month heatmap (image counts)
    analysis_summary.txt       - full text report (every month, every query)

Requirements:
    pip install pandas matplotlib
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save files
import matplotlib.pyplot as plt


def load_metadata(dataset_dir: str) -> pd.DataFrame:
    csv_path = Path(dataset_dir) / "metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"metadata.csv not found at {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def full_monthly_distribution(df: pd.DataFrame, output_dir: Path):
    """Bar chart covering every month from the earliest to the latest date_ym, no truncation."""
    dated = df[df["date_ym"].notna()].copy()
    monthly = dated.groupby("date_ym").size().sort_index()

    # Reindex to include months with zero images too, so gaps are visible
    all_months = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M").astype(str)
    monthly = monthly.reindex(all_months, fill_value=0)

    fig_width = max(14, len(monthly) * 0.25)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    ax.bar(monthly.index, monthly.values, color="#4C72B0")
    ax.set_title(f"Monthly Image Counts — Full Range ({monthly.index[0]} to {monthly.index[-1]})")
    ax.set_xlabel("Month")
    ax.set_ylabel("Image count")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    fig.tight_layout()
    out_path = output_dir / "monthly_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return monthly, out_path


def query_breakdown(df: pd.DataFrame, output_dir: Path):
    """Horizontal bar chart of image count per query, ALL queries (not just top 10)."""
    counts = df.groupby("query").size().sort_values(ascending=False)

    fig_height = max(8, len(counts) * 0.22)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(counts.index[::-1], counts.values[::-1], color="#DD8452")
    ax.set_title(f"Image Count per Query — all {len(counts)} queries")
    ax.set_xlabel("Image count")
    fig.tight_layout()
    out_path = output_dir / "query_breakdown.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return counts, out_path


def query_monthly_heatmap(df: pd.DataFrame, output_dir: Path):
    """Heatmap: rows = query, columns = month, values = image count."""
    dated = df[df["date_ym"].notna()].copy()
    pivot = dated.pivot_table(index="query", columns="date_ym", values="pin_id",
                               aggfunc="count", fill_value=0)
    # sort rows by total count, descending
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    fig_width = max(14, pivot.shape[1] * 0.3)
    fig_height = max(10, pivot.shape[0] * 0.25)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title("Image Count — Query × Month")
    fig.colorbar(im, ax=ax, label="Image count")
    fig.tight_layout()
    out_path = output_dir / "query_monthly_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return pivot, out_path


def write_summary(df: pd.DataFrame, monthly: pd.Series, counts: pd.Series, output_dir: Path):
    lines = []
    lines.append("=" * 60)
    lines.append("PINTEREST DATASET — FULL ANALYSIS SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Total images:        {len(df)}")
    lines.append(f"Dated images:        {df['date_ym'].notna().sum()}")
    lines.append(f"Unique queries:      {df['query'].nunique()}")
    lines.append(f"Unique boards:       {df['board'].nunique()}")
    lines.append(f"Date range:          {monthly.index[0]} -> {monthly.index[-1]}")
    lines.append(f"Total months spanned:{len(monthly)}")
    lines.append("")

    lines.append("-" * 60)
    lines.append("FULL MONTHLY DISTRIBUTION (every month, no truncation)")
    lines.append("-" * 60)
    max_c = monthly.max()
    for ym, c in monthly.items():
        bar = "#" * max(1, int(40 * c / max_c)) if max_c > 0 else ""
        lines.append(f"  {ym}  {bar:<41} {c}")
    lines.append("")

    lines.append("-" * 60)
    lines.append(f"ALL {len(counts)} QUERIES BY IMAGE COUNT")
    lines.append("-" * 60)
    for q, c in counts.items():
        lines.append(f"  {c:>6}  {q}")

    out_path = output_dir / "analysis_summary.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Detailed Pinterest dataset analysis")
    parser.add_argument("--dataset-dir", required=True, help="Folder containing metadata.csv")
    parser.add_argument("--output-dir", default=None,
                         help="Where to save charts/summary (default: same as dataset-dir)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_metadata(str(dataset_dir))

    monthly, monthly_png = full_monthly_distribution(df, output_dir)
    counts, query_png = query_breakdown(df, output_dir)
    pivot, heatmap_png = query_monthly_heatmap(df, output_dir)
    summary_txt = write_summary(df, monthly, counts, output_dir)

    print("\n[analysis done]")
    print(f"  Monthly chart:   {monthly_png}")
    print(f"  Query chart:     {query_png}")
    print(f"  Heatmap:         {heatmap_png}")
    print(f"  Text summary:    {summary_txt}")


if __name__ == "__main__":
    main()
