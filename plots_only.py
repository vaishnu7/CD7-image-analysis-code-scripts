"""
Normalise GFP and mCherry Fluorescence by H2B
=============================================
Loads the two CSVs produced by analysis_fluo_cellpose.py:
  - cell_fluorescence_dic_mask.csv   (GFP ch0, mCherry ch1)
  - cell_fluorescence_h2b_mask.csv   (H2B ch2)

Strategy:
  - For each timepoint, compute the MEAN H2B (ch2_mean) across all H2B-masked
    cells → use this as a per-timepoint nuclear reference value.
  - Divide every cell's GFP and mCherry background-corrected mean by that value.

Outputs:
  - cell_fluorescence_dic_mask_normalised.csv  (adds ch0_norm, ch1_norm columns)
  - normalisation_plots/  (timeline plots of raw vs normalised signals)

Run with: conda activate cellpose3 && python normalize_fluo_by_h2b.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION — match paths from your main analysis script
# ============================================================================

RESULTS_DIR = r"C:\Users\gz24763\OneDrive - University of Bristol\Documents\PC9_images\12dec25_data\output\P19\segmentation_results"

DIC_CSV  = os.path.join(RESULTS_DIR, "cell_fluorescence_dic_mask.csv")
H2B_CSV  = os.path.join(RESULTS_DIR, "cell_fluorescence_h2b_mask.csv")

OUTPUT_CSV  = os.path.join(RESULTS_DIR, "cell_fluorescence_dic_mask_normalised.csv")
PLOT_DIR    = os.path.join(RESULTS_DIR, "normalisation_plots")

# Which columns to normalise (background-corrected means)
GFP_COL     = "ch0_mean"
MCHERRY_COL = "ch1_mean"
H2B_COL     = "ch2_mean"

# Normalised column names in output
GFP_NORM_COL     = "ch0_norm_by_h2b"
MCHERRY_NORM_COL = "ch1_norm_by_h2b"

# Y-axis limits for all normalised plots (change here to update all plots at once)
Y_MIN = 0
Y_MAX = 8

# ============================================================================
# LOAD DATA
# ============================================================================

def load_csvs(dic_csv, h2b_csv):
    print(f"Loading DIC mask data:  {dic_csv}")
    df_dic = pd.read_csv(dic_csv)
    print(f"  → {len(df_dic)} rows, timepoints: {sorted(df_dic['timepoint'].unique())[:5]}...")

    print(f"Loading H2B mask data:  {h2b_csv}")
    df_h2b = pd.read_csv(h2b_csv)
    print(f"  → {len(df_h2b)} rows, timepoints: {sorted(df_h2b['timepoint'].unique())[:5]}...")

    return df_dic, df_h2b

# ============================================================================
# COMPUTE PER-TIMEPOINT H2B REFERENCE
# ============================================================================

def compute_h2b_reference(df_h2b, h2b_col):
    """
    Compute the mean H2B intensity across all cells for each timepoint.
    Returns a Series indexed by timepoint.
    """
    if h2b_col not in df_h2b.columns:
        raise ValueError(f"Column '{h2b_col}' not found in H2B CSV. "
                         f"Available columns: {list(df_h2b.columns)}")

    h2b_ref = df_h2b.groupby("timepoint")[h2b_col].mean()
    print(f"\nH2B reference (mean per timepoint):")
    print(h2b_ref.describe().to_string())
    return h2b_ref

# ============================================================================
# NORMALISE
# ============================================================================

def normalise(df_dic, h2b_ref, gfp_col, mcherry_col, gfp_norm_col, mcherry_norm_col):
    """
    Divide GFP and mCherry means by the per-timepoint H2B reference.
    Adds new columns in-place; returns modified dataframe.
    """
    df = df_dic.copy()
    df["_h2b_ref"] = df["timepoint"].map(h2b_ref)

    missing_tps = df["_h2b_ref"].isna().sum()
    if missing_tps > 0:
        print(f"\n⚠  {missing_tps} rows have no matching H2B timepoint — "
              "they will have NaN normalised values.")

    for raw_col, norm_col in [(gfp_col, gfp_norm_col), (mcherry_col, mcherry_norm_col)]:
        if raw_col not in df.columns:
            print(f"⚠  Column '{raw_col}' not found in DIC CSV, skipping.")
            continue
        df[norm_col] = df[raw_col] / df["_h2b_ref"]
        print(f"\n{norm_col} summary (normalised):")
        print(df[norm_col].describe().to_string())

    df.drop(columns=["_h2b_ref"], inplace=True)
    return df

# ============================================================================
# PLOTTING
# ============================================================================

def plot_normalised(df, h2b_ref, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # ── Plot 1: GFP raw vs normalised ──────────────────────────────────────
    _plot_raw_vs_norm(df, "ch0_mean", GFP_NORM_COL, "GFP (ch0)",
                      "#00cc44", "#006622", output_dir, "gfp_raw_vs_norm.png")

    # ── Plot 2: mCherry raw vs normalised ──────────────────────────────────
    _plot_raw_vs_norm(df, "ch1_mean", MCHERRY_NORM_COL, "mCherry (ch1)",
                      "#ff4444", "#881111", output_dir, "mcherry_raw_vs_norm.png")

    # ── Plot 3: H2B reference over time (no ylim — raw signal, not normalised) ──
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(h2b_ref.index.values, h2b_ref.values,
            marker='o', linewidth=2.5, markersize=5, color="#4444ff")
    ax.set_xlabel("Timepoint", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean H2B Intensity (background-corrected)", fontsize=13, fontweight="bold")
    ax.set_title("H2B Reference Signal Over Time", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(output_dir, "h2b_reference.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: h2b_reference.png")
    plt.close()

    # ── Plot 4: Both normalised signals on one axes ─────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    for norm_col, label, color in [
        (GFP_NORM_COL,     "GFP / H2B",     "#00cc44"),
        (MCHERRY_NORM_COL, "mCherry / H2B", "#ff4444"),
    ]:
        if norm_col not in df.columns:
            continue
        stats = df.groupby("timepoint")[norm_col].agg(["mean", "std", "count"])
        stats["sem"] = stats["std"] / np.sqrt(stats["count"])
        ax.errorbar(stats.index.values, stats["mean"].values, yerr=stats["sem"].values,
                    marker='o', linestyle='-', linewidth=2.5, markersize=5,
                    color=color, alpha=0.85, capsize=4, capthick=1.5, label=label)

    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("Timepoint", fontsize=13, fontweight="bold")
    ax.set_ylabel("Normalised Intensity ± SEM", fontsize=13, fontweight="bold")
    ax.set_title("GFP & mCherry Normalised by H2B Over Time", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(output_dir, "gfp_mcherry_normalised.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: gfp_mcherry_normalised.png")
    plt.close()


def _plot_raw_vs_norm(df, raw_col, norm_col, label, raw_color, norm_color, output_dir, fname):
    if raw_col not in df.columns or norm_col not in df.columns:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for ax, col, color, ylabel, title_suffix, apply_ylim in [
        (ax1, raw_col,  raw_color,  "Mean Intensity (bg-corrected)", "Raw",               False),
        (ax2, norm_col, norm_color, "Normalised Intensity (÷ H2B)",  "Normalised by H2B", True),
    ]:
        stats = df.groupby("timepoint")[col].agg(["mean", "std", "count"])
        stats["sem"] = stats["std"] / np.sqrt(stats["count"])
        ax.errorbar(stats.index.values, stats["mean"].values, yerr=stats["sem"].values,
                    marker='o', linestyle='-', linewidth=2.5, markersize=5,
                    color=color, alpha=0.85, capsize=4, capthick=1.5)
        if apply_ylim:
            ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(f"{label} — {title_suffix}", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)

    ax2.set_xlabel("Timepoint", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(output_dir, fname)
    plt.savefig(p, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {fname}")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GFP & mCherry NORMALISATION BY H2B")
    print("=" * 70)

    # 1. Load
    df_dic, df_h2b = load_csvs(DIC_CSV, H2B_CSV)

    # 2. Compute H2B reference per timepoint
    h2b_ref = compute_h2b_reference(df_h2b, H2B_COL)

    # 3. Normalise
    print("\nNormalising DIC-mask data by H2B reference...")
    df_norm = normalise(df_dic, h2b_ref, GFP_COL, MCHERRY_COL,
                        GFP_NORM_COL, MCHERRY_NORM_COL)

    # 4. Save
    df_norm.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Saved normalised CSV: {OUTPUT_CSV}")
    print(f"  Columns added: {GFP_NORM_COL}, {MCHERRY_NORM_COL}")

    # 5. Plot
    print("\nGenerating plots...")
    plot_normalised(df_norm, h2b_ref, PLOT_DIR)

    print("\n✓ Done! Outputs written to:")
    print(f"  {OUTPUT_CSV}")
    print(f"  {PLOT_DIR}/")