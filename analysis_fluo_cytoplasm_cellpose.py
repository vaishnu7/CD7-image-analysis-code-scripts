"""
True Cytoplasm Fluorescence Quantification
==========================================
Computes genuine cytoplasmic fluorescence by subtracting the nuclear ROI
from the whole-cell (DIC) ROI, matched per cell using spatial overlap (IoU).

Requires:
  - DIC mask TIFs:   masks_dic_t****.tif       (whole cell, from analysis_fluo_cellpose.py)
  - Nuclear mask TIFs: h2b_mask_t****.tif      (nucleus, from quantify_gfp_mcherry_from_h2b.py)
  - Channel image folders for GFP (ch0) and mCherry (ch1)

Strategy per timepoint:
  1. Load DIC mask (whole cell) and nuclear mask
  2. Match each nucleus to its parent cell by finding maximum overlap
  3. Cytoplasm ROI = cell pixels that are NOT in the matched nucleus
  4. Quantify GFP and mCherry in the cytoplasm ROI
  5. Background correct using pixels outside all cells (DIC mask background)

Output:
  - cytoplasm_gfp_mcherry_quantification.csv
  - cyto_mean_per_cell.png
  - cyto_nuclear_ratio.png  (true C/N ratio)

Run with: conda activate cellpose3 && python quantify_true_cytoplasm.py
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import tifffile

# ============================================================================
# CONFIGURATION
# ============================================================================

PARENT_DIR      = r"C:\Users\gz24763\OneDrive - University of Bristol\Documents\PC9_images\12dec25_data\output\P14"
EXPERIMENT_NAME = "New-01-Scene-21-P14-B02"

# Where the DIC masks are saved (from analysis_fluo_cellpose.py)
DIC_MASK_DIR    = os.path.join(PARENT_DIR, "segmentation_results")

# Where the nuclear masks are saved (from quantify_gfp_mcherry_from_h2b.py)
NUC_MASK_DIR    = os.path.join(PARENT_DIR, "nuclear_quant_results")

# Channel image folders
CHANNEL_MAP = {
    "ch0": ("GFP",     os.path.join(PARENT_DIR, f"{EXPERIMENT_NAME}_ch0")),
    "ch1": ("mCherry", os.path.join(PARENT_DIR, f"{EXPERIMENT_NAME}_ch1")),
}

OUTPUT_DIR = os.path.join(PARENT_DIR, "cytoplasm_quant_results")

# Minimum overlap fraction (IoU) to accept a nucleus→cell match
MIN_OVERLAP = 0.01

# Y-axis limits for plots (None = auto)
Y_MIN = 0
Y_MAX = None

# ============================================================================
# HELPERS
# ============================================================================

def parse_timepoint(filename):
    m = re.search(r'_?t(\d+)\.tif', filename)
    return int(m.group(1)) if m else None


def load_image(path):
    try:
        return np.array(Image.open(path))
    except Exception as e:
        print(f"  ⚠ Could not load {path}: {e}")
        return None


def collect_masks_and_images(dic_mask_dir, nuc_mask_dir, channel_map):
    """
    Find all timepoints that have both DIC and nuclear masks, plus channel images.
    Returns dict: {timepoint: {dic_mask, nuc_mask, ch0, ch1}}
    """
    dic_masks = {parse_timepoint(os.path.basename(f)): f
                 for f in glob.glob(os.path.join(dic_mask_dir, "masks_dic_t*.tif"))
                 if parse_timepoint(os.path.basename(f)) is not None}

    nuc_masks = {parse_timepoint(os.path.basename(f)): f
                 for f in glob.glob(os.path.join(nuc_mask_dir, "h2b_mask_t*.tif"))
                 if parse_timepoint(os.path.basename(f)) is not None}

    channel_files = defaultdict(dict)
    for ch_key, (ch_name, folder) in channel_map.items():
        for fp in glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.tif")):
            tp = re.search(r'_t(\d+)\.', os.path.basename(fp))
            if tp:
                channel_files[int(tp.group(1))][ch_key] = fp

    shared = sorted(set(dic_masks) & set(nuc_masks) & set(channel_files))
    print(f"  DIC masks found:     {len(dic_masks)} timepoints")
    print(f"  Nuclear masks found: {len(nuc_masks)} timepoints")
    print(f"  Shared timepoints:   {len(shared)}")

    return {tp: {
        "dic_mask": dic_masks[tp],
        "nuc_mask": nuc_masks[tp],
        **channel_files[tp]
    } for tp in shared}


def match_nuclei_to_cells(cell_mask, nuc_mask, min_overlap=0.3):
    """
    For each nucleus, find the cell with maximum overlap.
    Returns dict: {nuc_id: cell_id} for accepted matches only.
    """
    matches = {}
    nuc_ids = np.unique(nuc_mask)
    nuc_ids = nuc_ids[nuc_ids > 0]

    for nuc_id in nuc_ids:
        nuc_roi = nuc_mask == nuc_id
        # Which cell IDs overlap with this nucleus?
        overlapping_cells = cell_mask[nuc_roi]
        overlapping_cells = overlapping_cells[overlapping_cells > 0]

        if overlapping_cells.size == 0:
            continue

        # Pick the cell with the most overlapping pixels
        cell_id, overlap_count = np.unique(overlapping_cells, return_counts=True)
        best_cell = cell_id[np.argmax(overlap_count)]
        best_count = overlap_count.max()

        # Compute IoU: intersection / (nuc_area + cell_area - intersection)
        nuc_area  = nuc_roi.sum()
        cell_area = (cell_mask == best_cell).sum()
        iou = best_count / (nuc_area + cell_area - best_count)

        if iou >= min_overlap:
            matches[int(nuc_id)] = int(best_cell)

    return matches


def quantify_cytoplasm(cell_mask, nuc_mask, matches, images, channels, bg_means):
    """
    For each matched nucleus→cell pair, compute cytoplasm = cell pixels minus nucleus pixels.
    Returns list of per-cell dicts.
    """
    records = []

    for nuc_id, cell_id in matches.items():
        cell_roi = cell_mask == cell_id
        nuc_roi  = nuc_mask  == nuc_id
        cyto_roi = cell_roi & ~nuc_roi   # cytoplasm = cell minus nucleus

        cyto_area = int(cyto_roi.sum())
        if cyto_area == 0:
            continue

        row = {
            "cell_id":    cell_id,
            "nuc_id":     nuc_id,
            "cell_area":  int(cell_roi.sum()),
            "nuc_area":   int(nuc_roi.sum()),
            "cyto_area":  cyto_area,
        }

        for ch in channels:
            if ch not in images:
                continue
            bg   = bg_means[ch]
            px   = images[ch][cyto_roi]

            row[f"{ch}_mean_raw"] = float(px.mean())
            row[f"{ch}_mean"]     = float(px.mean() - bg)
            row[f"{ch}_sum"]      = float(px.sum()  - bg * cyto_area)
            row[f"{ch}_std"]      = float(px.std())
            row[f"{ch}_bg_mean"]  = bg

        records.append(row)

    return records

# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "="*65)
    print("TRUE CYTOPLASM QUANTIFICATION (whole cell − nucleus)")
    print("="*65)

    # Collect file paths
    print("\n[1/3] Collecting masks and images...")
    timepoints = collect_masks_and_images(DIC_MASK_DIR, NUC_MASK_DIR, CHANNEL_MAP)
    if not timepoints:
        print("ERROR: No shared timepoints found. Check mask directories.")
        return

    channels = list(CHANNEL_MAP.keys())

    # Process each timepoint
    print("\n[2/3] Computing cytoplasm ROIs and quantifying...")
    all_records = []

    for idx, (tp, files) in enumerate(timepoints.items(), 1):
        print(f"\n  [{idx}/{len(timepoints)}] t{tp:04d}")

        # Load masks
        cell_mask = tifffile.imread(files["dic_mask"])
        nuc_mask  = tifffile.imread(files["nuc_mask"])

        # Load channel images
        imgs = {}
        for ch in channels:
            if ch in files:
                arr = load_image(files[ch])
                if arr is not None:
                    imgs[ch] = arr

        # Background: pixels outside all cells (DIC mask)
        bg_mask = cell_mask == 0
        bg_means = {}
        for ch in channels:
            if ch in imgs:
                bg_px = imgs[ch][bg_mask]
                bg_means[ch] = float(bg_px.mean()) if bg_px.size > 0 else 0.0
                print(f"    {CHANNEL_MAP[ch][0]} background mean: {bg_means[ch]:.1f}")

        # Match nuclei to cells
        matches = match_nuclei_to_cells(cell_mask, nuc_mask, MIN_OVERLAP)
        print(f"    Cells matched: {len(matches)} / {len(np.unique(nuc_mask)) - 1} nuclei")

        # Quantify cytoplasm
        records = quantify_cytoplasm(cell_mask, nuc_mask, matches, imgs, channels, bg_means)
        for r in records:
            r["timepoint"] = tp
        all_records.extend(records)

    # Save CSV
    print("\n[3/3] Saving results...")
    if not all_records:
        print("  ✗ No records — check mask files and overlap threshold.")
        return

    df = pd.DataFrame(all_records)
    front = ["timepoint", "cell_id", "nuc_id", "cell_area", "nuc_area", "cyto_area"]
    rest  = sorted([c for c in df.columns if c not in front])
    df    = df[front + rest]

    csv_path = os.path.join(OUTPUT_DIR, "cytoplasm_gfp_mcherry_quantification.csv")
    df.to_csv(csv_path, index=False)
    print(f"  ✓ CSV saved: cytoplasm_gfp_mcherry_quantification.csv")
    print(f"  → {len(df)} cells × {df['timepoint'].nunique()} timepoints")

    # Summary
    print("\n  Summary (background-corrected cytoplasm):")
    for ch, (name, _) in CHANNEL_MAP.items():
        col = f"{ch}_mean"
        if col in df.columns:
            print(f"    {name}: mean = {df[col].mean():.1f} ± {df[col].std():.1f}")

    # ============================================================================
    # PLOTS
    # ============================================================================

    ch_styles = {
        "ch0": ("GFP",     "#00cc44"),
        "ch1": ("mCherry", "#ff4444"),
    }

    # Plot 1 — cytoplasm mean ± SEM over time
    fig, ax = plt.subplots(figsize=(14, 6))
    for ch, (name, color) in ch_styles.items():
        col = f"{ch}_mean"
        if col not in df.columns:
            continue
        stats = df.groupby("timepoint")[col].agg(["mean", "std", "count"])
        stats["sem"] = stats["std"] / np.sqrt(stats["count"])
        ax.errorbar(stats.index.values, stats["mean"].values, yerr=stats["sem"].values,
                    marker='o', linestyle='-', linewidth=2.5, markersize=5,
                    color=color, alpha=0.85, capsize=4, capthick=1.5, label=f"{name} ({ch})")
    if Y_MIN is not None or Y_MAX is not None:
        ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("Timepoint", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cytoplasm mean intensity ± SEM (background-corrected)", fontsize=13, fontweight="bold")
    ax.set_title("True Cytoplasmic GFP & mCherry Over Time", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cyto_mean_per_cell.png"), dpi=300, bbox_inches="tight")
    print("  ✓ Saved: cyto_mean_per_cell.png")
    plt.close()

    print(f"\n✓ Done! All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()