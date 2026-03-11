"""
Plot Nuclear GFP Normalised by iRFP
====================================
Reads:  nuclear_gfp_mcherry_quantification.csv
        (output of quantify_gfp_mcherry_from_h2b.py)

Nuclei are segmented on iRFP (ch2).
GFP (ch0) and iRFP (ch2) are both measured within those nuclear ROIs.

Normalisation: GFP_norm = ch0_mean / ch2_mean  (per cell, per timepoint)

Produces the same style plots as normalize_fluo_by_h2b.py:
  1. gfp_raw.png              — raw GFP mean ± SEM over time
  2. irfp_reference.png       — iRFP reference signal over time
  3. gfp_normalised_by_irfp.png — GFP/iRFP ratio ± SEM over time

Run with: conda activate cellpose3 && python plot_nuclear_fluo_normalised.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

CSV_PATH = r"C:\Users\gz24763\OneDrive - University of Bristol\Documents\PC9_images\12dec25_data\output\P20\nuclear_quant_results\nuclear_gfp_mcherry_quantification.csv"

OUTPUT_DIR = os.path.join(os.path.dirname(CSV_PATH), "nuclear_fluo_plots")

# Y-axis limit for the normalised plot only (set None for auto)
Y_MIN = 0
Y_MAX = 8

# ============================================================================
# LOAD & NORMALISE
# ============================================================================

print("Loading data...")
df = pd.read_csv(CSV_PATH)
print(f"  {len(df)} rows | {df['timepoint'].nunique()} timepoints")

# Per-cell normalisation: GFP / iRFP and mCherry / iRFP
df["gfp_norm_by_irfp"]     = df["ch0_mean"] / df["ch2_mean"]
df["mcherry_norm_by_irfp"] = df["ch1_mean"] / df["ch2_mean"]

print(f"\ngfp_norm_by_irfp summary:")
print(df["gfp_norm_by_irfp"].describe().to_string())
print(f"\nmcherry_norm_by_irfp summary:")
print(df["mcherry_norm_by_irfp"].describe().to_string())

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Per-timepoint iRFP reference (mean across all cells) — for the reference plot
irfp_ref = df.groupby("timepoint")["ch2_mean"].mean()

# ============================================================================
# HELPER
# ============================================================================

def plot_errorbar(ax, df, col, color):
    stats = df.groupby("timepoint")[col].agg(["mean", "std", "count"])
    stats["sem"] = stats["std"] / np.sqrt(stats["count"])
    ax.errorbar(stats.index.values, stats["mean"].values, yerr=stats["sem"].values,
                marker='o', linestyle='-', linewidth=2.5, markersize=5,
                color=color, alpha=0.85, capsize=4, capthick=1.5)

# ============================================================================
# PLOT 1 — Raw GFP
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 5))
plot_errorbar(ax, df, "ch0_mean", "#00cc44")
ax.set_xlabel("Timepoint", fontsize=13, fontweight="bold")
ax.set_ylabel("Nuclear mean intensity ± SEM (background-corrected)", fontsize=13, fontweight="bold")
ax.set_title("Nuclear GFP (ch0) — Raw", fontsize=14, fontweight="bold")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gfp_raw.png"), dpi=300, bbox_inches="tight")
print("✓ Saved: gfp_raw.png")
plt.close()

# ============================================================================
# PLOT 2 — iRFP reference
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(irfp_ref.index.values, irfp_ref.values,
        marker='o', linewidth=2.5, markersize=5, color="#aa44ff")
ax.set_xlabel("Timepoint", fontsize=13, fontweight="bold")
ax.set_ylabel("Mean iRFP Intensity (background-corrected)", fontsize=13, fontweight="bold")
ax.set_title("Nuclear iRFP (ch2) Reference Signal Over Time", fontsize=14, fontweight="bold")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "irfp_reference.png"), dpi=300, bbox_inches="tight")
print("✓ Saved: irfp_reference.png")
plt.close()

# ============================================================================
# PLOT 3 — GFP & mCherry normalised by iRFP on same figure
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 6))

for col, label, color in [
    ("gfp_norm_by_irfp",     "GFP / iRFP (ch0/ch2)",     "#00cc44"),
    ("mcherry_norm_by_irfp", "mCherry / iRFP (ch1/ch2)", "#ff4444"),
]:
    plot_errorbar(ax, df, col, color)
    # Add label via a dummy plot for the legend
    ax.plot([], [], color=color, linewidth=2.5, marker='o', markersize=5, label=label)

if Y_MIN is not None or Y_MAX is not None:
    ax.set_ylim(Y_MIN, Y_MAX)
ax.set_xlabel("Timepoint", fontsize=13, fontweight="bold")
ax.set_ylabel("Normalised Intensity ± SEM", fontsize=13, fontweight="bold")
ax.set_title("Nuclear GFP & mCherry Normalised by iRFP Over Time", fontsize=14, fontweight="bold")
ax.legend(fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gfp_mcherry_normalised_by_irfp.png"), dpi=300, bbox_inches="tight")
print("✓ Saved: gfp_mcherry_normalised_by_irfp.png")
plt.close()

print(f"\n✓ Done! Plots saved to: {OUTPUT_DIR}")