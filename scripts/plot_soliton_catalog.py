#!/usr/bin/env python3
"""
Catalog all localized 3D solitons found across multiple sweep CSVs.
Generates a summary scatter plot of sigma vs mu colored by structure type.
"""
import sys
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path


def load_all_summaries(csv_patterns: list[str]) -> pd.DataFrame:
    """Load and concatenate all summary-mode CSVs."""
    frames = []
    for pattern in csv_patterns:
        for f in glob.glob(pattern):
            try:
                df = pd.read_csv(f, comment="#")
                if "step" in df.columns:
                    continue  # skip per-step files
                df["source"] = Path(f).stem
                frames.append(df)
            except Exception:
                pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def extract_localized_from_perstep(csv_patterns: list[str]) -> pd.DataFrame:
    """Extract final-state summary from per-step CSVs."""
    frames = []
    for pattern in csv_patterns:
        for f in glob.glob(pattern):
            try:
                df = pd.read_csv(f, comment="#")
                if "step" not in df.columns:
                    continue  # skip summary files
                # Group by run parameters, take last row
                groups = df.groupby(["sigma", "mu", "radius", "T", "pattern"])
                records = []
                for (sigma, mu, R, T, pat), g in groups:
                    last = g.iloc[-1]
                    first_mass = g.iloc[0]["mass"]
                    records.append({
                        "sigma": sigma, "mu": mu, "radius": R, "T": T,
                        "pattern": pat,
                        "final_mass_ratio": last["mass_ratio"],
                        "final_rg": last["rg"],
                        "final_compactness": last["compactness"],
                        "final_entropy": last["entropy"],
                        "com_disp": last["com_disp"],
                        "final_sv_ratio": last["sv_ratio"],
                        "steps_run": int(last["step"]),
                        "source": Path(f).stem,
                    })
                if records:
                    frames.append(pd.DataFrame(records))
            except Exception:
                pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    outdir = Path("plots")
    outdir.mkdir(exist_ok=True)

    # Load all available data
    summaries = load_all_summaries(["sweep_*.csv"])
    perstep = extract_localized_from_perstep(["deep_*.csv"])

    all_data = pd.concat([summaries, perstep], ignore_index=True)
    if all_data.empty:
        print("No data found.")
        return

    # Filter localized structures
    if "bbox_dx" in all_data.columns:
        loc = all_data[
            (all_data["final_mass_ratio"] > 0.05) &
            (all_data["final_mass_ratio"] < 3.0) &
            ((all_data["bbox_dx"] < 40) | all_data["bbox_dx"].isna())
        ].copy()
    else:
        loc = all_data[
            (all_data["final_mass_ratio"] > 0.05) &
            (all_data["final_mass_ratio"] < 3.0)
        ].copy()

    # Also filter from perstep data
    if "com_disp" in loc.columns:
        pass  # keep all

    print(f"Total runs across all files: {len(all_data)}")
    print(f"Localized structures found: {len(loc)}")

    if loc.empty:
        print("No localized structures to catalog.")
        return

    # Print catalog
    cols = ["sigma", "mu", "radius", "T", "pattern", "final_mass_ratio",
            "final_compactness", "final_rg", "source"]
    if "com_disp" in loc.columns:
        cols.append("com_disp")
    if "mean_com_speed" in loc.columns:
        cols.append("mean_com_speed")

    available_cols = [c for c in cols if c in loc.columns]
    print("\n=== SOLITON CATALOG ===")
    print(loc[available_cols].sort_values(["pattern", "radius", "sigma"]).to_string(index=False))

    # --- Plot: sigma vs mu scatter, colored by pattern, sized by compactness ---
    fig, ax = plt.subplots(figsize=(12, 8))

    patterns = loc["pattern"].unique() if "pattern" in loc.columns else ["unknown"]
    colors = {"blob": "#e74c3c", "glider": "#3498db", "shell": "#27ae60",
              "multi": "#9b59b6", "dipole": "#f39c12"}

    for pat in sorted(patterns):
        sub = loc[loc["pattern"] == pat] if "pattern" in loc.columns else loc
        color = colors.get(pat, "#95a5a6")
        sizes = sub["final_compactness"].clip(0.5, 3.0) * 40 if "final_compactness" in sub.columns else 60

        # Mark moving structures differently
        if "com_disp" in sub.columns:
            moving = sub[sub["com_disp"] > 1.0]
            static = sub[sub["com_disp"] <= 1.0]

            ax.scatter(static["sigma"], static["mu"], s=sizes[static.index],
                      c=color, marker="o", alpha=0.7, label=f"{pat} (static)",
                      edgecolors="black", linewidth=0.5)
            if not moving.empty:
                ax.scatter(moving["sigma"], moving["mu"],
                          s=sizes[moving.index],
                          c=color, marker="*", alpha=0.9,
                          label=f"{pat} (MOVING)",
                          edgecolors="black", linewidth=1.0, zorder=10)
        else:
            ax.scatter(sub["sigma"], sub["mu"], s=sizes,
                      c=color, marker="o", alpha=0.7, label=pat,
                      edgecolors="black", linewidth=0.5)

    ax.set_xlabel("sigma (growth width)", fontsize=13)
    ax.set_ylabel("mu (growth center)", fontsize=13)
    ax.set_title("3D Lenia Soliton Catalog — sigma vs mu\n"
                 "Size ~ compactness, * = moving structure",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = outdir / "soliton_catalog.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"\nSaved {outpath}")

    # --- Plot: Compactness vs Mass Ratio ---
    fig, ax = plt.subplots(figsize=(10, 7))

    for pat in sorted(patterns):
        sub = loc[loc["pattern"] == pat] if "pattern" in loc.columns else loc
        color = colors.get(pat, "#95a5a6")
        ax.scatter(sub["final_mass_ratio"], sub["final_compactness"],
                  c=color, s=60, alpha=0.7, label=pat,
                  edgecolors="black", linewidth=0.5)

    ax.set_xlabel("Final Mass Ratio", fontsize=13)
    ax.set_ylabel("Compactness", fontsize=13)
    ax.set_title("3D Soliton Characterization — Compactness vs Mass Ratio",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = outdir / "soliton_compactness_vs_mass.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved {outpath}")


if __name__ == "__main__":
    main()
