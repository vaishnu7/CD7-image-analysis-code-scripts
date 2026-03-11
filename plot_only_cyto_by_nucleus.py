"""
Plot Cytoplasm / Nucleus Ratio for GFP and mCherry
===================================================
Computes the cytoplasmic-to-nuclear ratio (C/N) of GFP and mCherry over time.

Requires two CSVs:
  - cell_fluorescence_dic_mask.csv   → cytoplasmic GFP/mCherry (ch0_mean, ch1_mean, from DIC mask)
  - nuclear_gfp_mcherry_quantification.csv → nuclear GFP/mCherry (ch0_mean, ch1_mean, from iRFP/nuclear mask)

Produces 4 plots:
  1. cyto_vs_nuclear_gfp_raw.png         — raw cyto vs nuclear GFP ± SEM
  2. cyto_nuclear_gfp_ratio.png          — GFP C/N ratio ± SEM
  3. cyto_vs_nuclear_mcherry_raw.png     — raw cyto vs nuclear mCherry ± SEM
  4. cyto_nuclear_mcherry_ratio.png      — mCherry C/N ratio ± SEM
  5. cyto_nuclear_gfp_ratio_clean.png    — GFP C/N ratio, mean only (no error bars)
  6. cyto_nuclear_mcherry_ratio_clean.png — mCherry C/N ratio, mean only (no error bars)

Run with: conda activate cellpose3 && python plot_cyto_nuclear_ratio.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

CYTO_CSV    = r"C:\Users\gz24763\OneDrive - University of Bristol\Documents\PC9_images\12dec25_data\output\P14\cytoplasm_quant_results\cytoplasm_gfp_mcherry_quantification.csv"
NUCLEAR_CSV = r"C:\Users\gz24763\OneDrive - University of Bristol\Documents\PC9_images\12dec25_data\output\P14\nuclear_quant_results\nuclear_gfp_mcherry_quantification.csv"

OUTPUT_DIR  = r"C:\Users\gz24763\OneDrive - University of Bristol\Documents\PC9_images\12dec25_data\output\P14\cyto_by_nuclear_fluo_plots"

# Y-axis limits (set None for auto)
Y_MIN = 0
Y_MAX = None

# ============================================================================
# LOAD
# ============================================================================

print("Loading data...")
df_cyto    = pd.read_csv(CYTO_CSV)
df_nuclear = pd.read_csv(NUCLEAR_CSV)
print(f"  Cytoplasm (DIC mask):  {len(df_cyto)} rows | {df_cyto['timepoint'].nunique()} timepoints")
print(f"  Nuclear (iRFP mask):   {len(df_nuclear)} rows | {df_nuclear['timepoint'].nunique()} timepoints")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# HELPER — compute ratio and propagated SEM for one channel
# ============================================================================

def compute_ratio(df_cyto, df_nuclear, col):
    cyto_s    = df_cyto.groupby("timepoint")[col].agg(["mean", "std", "count"])
    nuclear_s = df_nuclear.groupby("timepoint")[col].agg(["mean", "std", "count"])

    shared    = cyto_s.index.intersection(nuclear_s.index)
    cyto_s    = cyto_s.loc[shared].copy()
    nuclear_s = nuclear_s.loc[shared].copy()

    floor = 1e-6
    cyto_s["mean"]    = cyto_s["mean"].clip(lower=floor)
    nuclear_s["mean"] = nuclear_s["mean"].clip(lower=floor)

    ratio     = cyto_s["mean"] / nuclear_s["mean"]
    cyto_sem  = cyto_s["std"]    / np.sqrt(cyto_s["count"])
    nuc_sem   = nuclear_s["std"] / np.sqrt(nuclear_s["count"])

    ratio_sem = ratio * np.sqrt(
        (cyto_sem  / cyto_s["mean"]).pow(2) +
        (nuc_sem   / nuclear_s["mean"]).pow(2)
    )
    ratio_sem = ratio_sem.clip(lower=0)

    return shared, cyto_s, nuclear_s, ratio, ratio_sem


# ============================================================================
# COMPUTE
# ============================================================================

tps_gfp,     cyto_gfp,     nuc_gfp,     ratio_gfp,     sem_gfp     = compute_ratio(df_cyto, df_nuclear, "ch0_mean")
tps_mcherry, cyto_mcherry, nuc_mcherry, ratio_mcherry, sem_mcherry = compute_ratio(df_cyto, df_nuclear, "ch1_mean")

print(f"\nGFP C/N ratio summary:")
print(ratio_gfp.describe().to_string())
print(f"\nmCherry C/N ratio summary:")
print(ratio_mcherry.describe().to_string())

# ============================================================================
# PLOT HELPERS
# ============================================================================

def plot_raw(cyto_s, nuc_s, cyto_color, nuc_color, cyto_label, nuc_label, title, fname):
    fig, ax = plt.subplots(figsize=(14, 6))
    for stats, label, color in [(cyto_s, cyto_label, cyto_color),
                                 (nuc_s,  nuc_label,  nuc_color)]:
        sem = stats["std"] / np.sqrt(stats["count"])
        ax.errorbar(stats.index.values, stats["mean"].values, yerr=sem.values,
                    marker='o', linestyle='-', linewidth=2.5, markersize=5,
                    alpha=0.85, capsize=4, capthick=1.5, color=color, label=label)
    ax.set_xlabel("Timepoint", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean intensity ± SEM (background-corrected)", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {fname}")
    plt.close()


def plot_ratio(tps, ratio, sem, color, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.errorbar(tps, ratio.values, yerr=sem.values,
                marker='o', linestyle='-', linewidth=2.5, markersize=5,
                color=color, alpha=0.85, capsize=4, capthick=1.5)
    ax.axhline(y=1, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, label="C/N = 1 (equal)")
    if Y_MIN is not None or Y_MAX is not None:
        ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("Timepoint", fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {fname}")
    plt.close()


def plot_ratio_clean(tps, ratio, color, ylabel, title, fname):
    """Mean line only — no error bars."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(tps, ratio.values,
            marker='o', linestyle='-', linewidth=2.5, markersize=5,
            color=color, alpha=0.85)
    ax.axhline(y=1, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, label="C/N = 1 (equal)")
    if Y_MIN is not None or Y_MAX is not None:
        ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("Timepoint", fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {fname}")
    plt.close()


# ============================================================================
# GENERATE ALL PLOTS
# ============================================================================

# GFP raw
plot_raw(cyto_gfp, nuc_gfp,
         "#00cc44", "#006622",
         "Cytoplasmic GFP (DIC mask)", "Nuclear GFP (iRFP mask)",
         "Cytoplasmic vs Nuclear GFP Over Time",
         "cyto_vs_nuclear_gfp_raw.png")

# GFP C/N ± SEM
plot_ratio(tps_gfp, ratio_gfp, sem_gfp,
           "#00cc44",
           "Cytoplasm GFP / Nucleus GFP ± SEM",
           "Cytoplasmic-to-Nuclear GFP Ratio Over Time",
           "cyto_nuclear_gfp_ratio.png")

# mCherry raw
plot_raw(cyto_mcherry, nuc_mcherry,
         "#ff4444", "#881111",
         "Cytoplasmic mCherry (DIC mask)", "Nuclear mCherry (iRFP mask)",
         "Cytoplasmic vs Nuclear mCherry Over Time",
         "cyto_vs_nuclear_mcherry_raw.png")

# mCherry C/N ± SEM
plot_ratio(tps_mcherry, ratio_mcherry, sem_mcherry,
           "#ff4444",
           "Cytoplasm mCherry / Nucleus mCherry ± SEM",
           "Cytoplasmic-to-Nuclear mCherry Ratio Over Time",
           "cyto_nuclear_mcherry_ratio.png")

# GFP C/N clean (no error bars)
plot_ratio_clean(tps_gfp, ratio_gfp,
                 "#00cc44",
                 "Cytoplasm GFP / Nucleus GFP",
                 "Cytoplasmic-to-Nuclear GFP Ratio Over Time (mean only)",
                 "cyto_nuclear_gfp_ratio_clean.png")

# mCherry C/N clean (no error bars)
plot_ratio_clean(tps_mcherry, ratio_mcherry,
                 "#ff4444",
                 "Cytoplasm mCherry / Nucleus mCherry",
                 "Cytoplasmic-to-Nuclear mCherry Ratio Over Time (mean only)",
                 "cyto_nuclear_mcherry_ratio_clean.png")

print(f"\n✓ Done! Plots saved to: {OUTPUT_DIR}")