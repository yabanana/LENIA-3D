#!/usr/bin/env python3
"""Plot time evolution of a deep per-step run."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_deep_run(csv_path: str, outdir: str = "plots") -> None:
    df = pd.read_csv(csv_path, comment="#")
    Path(outdir).mkdir(parents=True, exist_ok=True)

    sigma = df["sigma"].iloc[0]
    mu = df["mu"].iloc[0]
    R = df["radius"].iloc[0]
    T = df["T"].iloc[0]
    pat = df["pattern"].iloc[0]
    title = f"R={R} T={T} sigma={sigma:.4f} mu={mu:.2f} pattern={pat}"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Mass ratio
    ax = axes[0, 0]
    ax.plot(df["step"], df["mass_ratio"], "b-", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mass Ratio")
    ax.set_title("Mass Ratio over Time")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=df["mass_ratio"].iloc[-1], color="r", linestyle="--", alpha=0.5,
               label=f"final={df['mass_ratio'].iloc[-1]:.4f}")
    ax.legend()

    # Radius of gyration + compactness
    ax = axes[0, 1]
    ax.plot(df["step"], df["rg"], "g-", linewidth=1.5, label="Rg")
    ax.set_xlabel("Step")
    ax.set_ylabel("Radius of Gyration", color="g")
    ax.tick_params(axis="y", labelcolor="g")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(df["step"], df["compactness"], "r-", linewidth=1.5, label="Compactness")
    ax2.set_ylabel("Compactness", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    ax.set_title("Rg and Compactness")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    # Entropy
    ax = axes[1, 0]
    ax.plot(df["step"], df["entropy"], "m-", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Spatial Entropy (normalized)")
    ax.set_title("Spatial Entropy")
    ax.grid(True, alpha=0.3)

    # Surface/Volume ratio
    ax = axes[1, 1]
    ax.plot(df["step"], df["sv_ratio"], "c-", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Surface / Volume Ratio")
    ax.set_title("Surface/Volume Ratio")
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    stem = Path(csv_path).stem
    outpath = f"{outdir}/{stem}_dynamics.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved {outpath}")

if __name__ == "__main__":
    plot_deep_run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "plots")
