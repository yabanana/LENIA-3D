#!/usr/bin/env python3
"""
Analyze lenia_search CSV output: phase diagrams, heatmaps, interesting runs.

Usage:
    python scripts/analyze_search.py sweep_quick.csv
    python scripts/analyze_search.py sweep_quick.csv --output-dir plots/
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns


# Class → color mapping
CLASS_COLORS = {
    "Extinct":  "#2c3e50",   # dark gray
    "Filled":   "#e74c3c",   # red
    "Pulsing":  "#f39c12",   # orange
    "Stable":   "#27ae60",   # green
    "Gliding":  "#3498db",   # blue
    "Chaotic":  "#9b59b6",   # purple
    "Unknown":  "#1abc9c",   # teal
}

CLASS_ORDER = ["Extinct", "Filled", "Stable", "Pulsing", "Gliding", "Chaotic", "Unknown"]


def load_csv(path: str) -> pd.DataFrame:
    """Load search CSV, skipping comment lines."""
    df = pd.read_csv(path, comment="#")
    # Round floats to avoid float precision issues in grouping
    df["sigma"] = df["sigma"].round(4)
    df["mu"] = df["mu"].round(4)
    return df


def plot_phase_diagram(df: pd.DataFrame, pattern: str, outdir: Path) -> None:
    """Heatmap of classification for sigma x mu, one per pattern."""
    sub = df[df["pattern"] == pattern].copy()
    if sub.empty:
        return

    # Map class names to integers for coloring
    class_to_int = {c: i for i, c in enumerate(CLASS_ORDER)}
    sub["class_int"] = sub["class"].map(class_to_int)

    pivot = sub.pivot_table(
        index="mu", columns="sigma", values="class_int",
        aggfunc="first"
    )

    # Sort axes
    pivot = pivot.sort_index(ascending=False)
    pivot = pivot[sorted(pivot.columns)]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Custom colormap from CLASS_COLORS
    colors = [CLASS_COLORS[c] for c in CLASS_ORDER]
    cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(-0.5, len(CLASS_ORDER) + 0.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    sns.heatmap(
        pivot, ax=ax, cmap=cmap, norm=norm,
        linewidths=0.5, linecolor="white",
        cbar_kws={"ticks": range(len(CLASS_ORDER))},
        annot=False, square=True,
    )

    # Fix colorbar labels
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(range(len(CLASS_ORDER)))
    cbar.set_ticklabels(CLASS_ORDER)

    ax.set_title(f"Phase Diagram — pattern={pattern}", fontsize=14, fontweight="bold")
    ax.set_xlabel("sigma", fontsize=12)
    ax.set_ylabel("mu", fontsize=12)

    # Format tick labels
    ax.set_xticklabels([f"{float(t.get_text()):.3f}" for t in ax.get_xticklabels()],
                       rotation=45, ha="right")
    ax.set_yticklabels([f"{float(t.get_text()):.2f}" for t in ax.get_yticklabels()],
                       rotation=0)

    plt.tight_layout()
    outpath = outdir / f"phase_{pattern}.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_mass_ratio_heatmap(df: pd.DataFrame, pattern: str, outdir: Path) -> None:
    """Heatmap of final mass ratio for sigma x mu."""
    sub = df[df["pattern"] == pattern].copy()
    if sub.empty:
        return

    pivot = sub.pivot_table(
        index="mu", columns="sigma", values="final_mass_ratio",
        aggfunc="first"
    )
    pivot = pivot.sort_index(ascending=False)
    pivot = pivot[sorted(pivot.columns)]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Log-scale coloring with clipping for readability
    data_clipped = pivot.clip(lower=1e-3)

    sns.heatmap(
        data_clipped, ax=ax, cmap="YlOrRd",
        norm=mcolors.LogNorm(vmin=1e-3, vmax=data_clipped.max().max()),
        linewidths=0.5, linecolor="white",
        annot=True, fmt=".2f", annot_kws={"fontsize": 7},
        square=True,
    )

    ax.set_title(f"Final Mass Ratio — pattern={pattern}", fontsize=14, fontweight="bold")
    ax.set_xlabel("sigma", fontsize=12)
    ax.set_ylabel("mu", fontsize=12)

    ax.set_xticklabels([f"{float(t.get_text()):.3f}" for t in ax.get_xticklabels()],
                       rotation=45, ha="right")
    ax.set_yticklabels([f"{float(t.get_text()):.2f}" for t in ax.get_yticklabels()],
                       rotation=0)

    plt.tight_layout()
    outpath = outdir / f"mass_ratio_{pattern}.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_compactness_heatmap(df: pd.DataFrame, pattern: str, outdir: Path) -> None:
    """Heatmap of final compactness for sigma x mu."""
    sub = df[df["pattern"] == pattern].copy()
    if sub.empty:
        return

    # Only plot non-extinct runs
    sub = sub[sub["class"] != "Extinct"].copy()
    if sub.empty:
        return

    pivot = sub.pivot_table(
        index="mu", columns="sigma", values="final_compactness",
        aggfunc="first"
    )
    pivot = pivot.sort_index(ascending=False)
    pivot = pivot[sorted(pivot.columns)]

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(
        pivot, ax=ax, cmap="viridis",
        linewidths=0.5, linecolor="white",
        annot=True, fmt=".2f", annot_kws={"fontsize": 7},
        square=True,
    )

    ax.set_title(f"Compactness (non-extinct) — pattern={pattern}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("sigma", fontsize=12)
    ax.set_ylabel("mu", fontsize=12)

    ax.set_xticklabels([f"{float(t.get_text()):.3f}" for t in ax.get_xticklabels()],
                       rotation=45, ha="right")
    ax.set_yticklabels([f"{float(t.get_text()):.2f}" for t in ax.get_yticklabels()],
                       rotation=0)

    plt.tight_layout()
    outpath = outdir / f"compactness_{pattern}.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_class_distribution(df: pd.DataFrame, outdir: Path) -> None:
    """Bar chart of class distribution per pattern."""
    fig, ax = plt.subplots(figsize=(10, 5))

    patterns = sorted(df["pattern"].unique())
    x = np.arange(len(patterns))
    width = 0.8 / len(CLASS_ORDER)

    for i, cls in enumerate(CLASS_ORDER):
        counts = []
        for pat in patterns:
            sub = df[(df["pattern"] == pat) & (df["class"] == cls)]
            counts.append(len(sub))
        if sum(counts) > 0:
            ax.bar(x + i * width, counts, width, label=cls,
                   color=CLASS_COLORS[cls], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Pattern", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Classification Distribution per Pattern",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x + 0.4)
    ax.set_xticklabels(patterns)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    outpath = outdir / "class_distribution.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")


def print_interesting_runs(df: pd.DataFrame) -> None:
    """Print runs classified as Stable, Gliding, Unknown, or localized Pulsing."""
    interesting = df[
        (df["class"].isin(["Stable", "Gliding", "Unknown"])) |
        ((df["class"] == "Pulsing") &
         (df["final_mass_ratio"] > 0.1) &
         (df["final_mass_ratio"] < 5.0) &
         (df["bbox_dx"] < df["bbox_dx"].max() * 0.8))
    ].copy()

    if interesting.empty:
        print("\n  No interesting (localized) runs found.")
        return

    print(f"\n  === {len(interesting)} Interesting Runs ===")
    cols = ["run_id", "sigma", "mu", "pattern", "class",
            "final_mass_ratio", "final_compactness", "final_rg",
            "bbox_dx", "bbox_dy", "bbox_dz", "mean_com_speed",
            "osc_amp"]
    print(interesting[cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze lenia_search CSV output")
    parser.add_argument("csv", help="Path to search results CSV")
    parser.add_argument("--output-dir", default="plots",
                        help="Directory for output plots (default: plots/)")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.csv}...")
    df = load_csv(args.csv)
    print(f"  {len(df)} runs loaded")

    # Summary stats
    print(f"\n  Class distribution:")
    for cls in CLASS_ORDER:
        n = len(df[df["class"] == cls])
        if n > 0:
            print(f"    {cls:10s}: {n:4d} ({100*n/len(df):.1f}%)")

    patterns = sorted(df["pattern"].unique())
    print(f"\n  Patterns: {', '.join(patterns)}")

    # Generate plots per pattern
    for pat in patterns:
        print(f"\n  --- Pattern: {pat} ---")
        plot_phase_diagram(df, pat, outdir)
        plot_mass_ratio_heatmap(df, pat, outdir)
        plot_compactness_heatmap(df, pat, outdir)

    plot_class_distribution(df, outdir)
    print_interesting_runs(df)

    print(f"\nAll plots saved to {outdir}/")


if __name__ == "__main__":
    main()
