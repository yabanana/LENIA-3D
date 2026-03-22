#!/usr/bin/env python3
"""Plot dynamics of multiple runs from a per-step CSV, one subplot grid per run."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def plot_multi_runs(csv_path: str, outdir: str = "plots") -> None:
    df = pd.read_csv(csv_path, comment="#")
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Filter localized runs
    groups = list(df.groupby(["sigma", "mu"]))
    localized = []
    for (sigma, mu), g in groups:
        last = g.iloc[-1]
        if 0.05 < last["mass_ratio"] < 3.0:
            localized.append(((sigma, mu), g))

    if not localized:
        print("No localized runs found.")
        return

    n = len(localized)
    fig, axes = plt.subplots(n, 4, figsize=(18, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for row, ((sigma, mu), g) in enumerate(localized):
        R = g["radius"].iloc[0]
        T = g["T"].iloc[0]
        pat = g["pattern"].iloc[0]

        # Mass ratio
        ax = axes[row, 0]
        ax.plot(g["step"], g["mass_ratio"], "b-", linewidth=1.2)
        ax.set_ylabel("Mass Ratio")
        ax.set_title(f"s={sigma:.3f} m={mu:.2f} R={R} T={T} {pat}")
        ax.grid(True, alpha=0.3)
        final_mr = g["mass_ratio"].iloc[-1]
        ax.axhline(y=final_mr, color="r", linestyle="--", alpha=0.5,
                   label=f"{final_mr:.4f}")
        ax.legend(fontsize=8)

        # CoM displacement
        ax = axes[row, 1]
        ax.plot(g["step"], g["com_disp"], "r-", linewidth=1.2)
        ax.set_ylabel("CoM Displacement")
        ax.set_title("CoM drift")
        ax.grid(True, alpha=0.3)

        # Rg + Compactness
        ax = axes[row, 2]
        ax.plot(g["step"], g["rg"], "g-", linewidth=1.2, label="Rg")
        ax.set_ylabel("Rg", color="g")
        ax2 = ax.twinx()
        ax2.plot(g["step"], g["compactness"], "m-", linewidth=1.2, label="C")
        ax2.set_ylabel("Compactness", color="m")
        ax.set_title("Shape")
        ax.grid(True, alpha=0.3)

        # Entropy + SV ratio
        ax = axes[row, 3]
        ax.plot(g["step"], g["entropy"], "c-", linewidth=1.2, label="H")
        ax.plot(g["step"], g["sv_ratio"], "orange", linewidth=1.2, label="S/V")
        ax.set_ylabel("Value")
        ax.set_title("Entropy & S/V")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel("Step")

    fig.suptitle(f"Localized 3D Structures — {Path(csv_path).stem}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    stem = Path(csv_path).stem
    outpath = f"{outdir}/{stem}_localized.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved {outpath}")

if __name__ == "__main__":
    plot_multi_runs(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "plots")
