"""
Nuclear Fluorescence Quantification via H2B Mask
=================================================
Segments nuclei using the H2B channel (ch2) with Cellpose's nuclei model,
then quantifies GFP (ch0) and mCherry (ch1) intensities within each nucleus.

Background correction: mean intensity of non-cell pixels subtracted per channel.

Run with:
    conda activate cellpose3 && python quantify_gfp_mcherry_from_h2b.py
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from PIL import Image
import tifffile
from cellpose import models

# ============================================================================
# CONFIGURATION — edit these paths and parameters
# ============================================================================

PARENT_DIR      = r"C:\Users\gz24763\OneDrive - University of Bristol\Documents\PC9_images\12dec25_data\output\P1"
EXPERIMENT_NAME = "New-01-Scene-02-P1-A02"
OUTPUT_DIR      = os.path.join(PARENT_DIR, "nuclear_quant_results")

# Cellpose nuclear segmentation settings
NUCLEAR_DIAMETER       = 30      # Approximate nuclear diameter in pixels — tune to your images
FLOW_THRESHOLD         = 0.4
CELLPROB_THRESHOLD     = 0.0
USE_GPU                = True

# Channel folder mapping:  channel_key -> (display_name, folder_suffix)
CHANNEL_MAP = {
    "ch0": ("GFP",     f"{EXPERIMENT_NAME}_ch0"),
    "ch1": ("mCherry", f"{EXPERIMENT_NAME}_ch1"),
    "ch2": ("H2B",     f"{EXPERIMENT_NAME}_ch2"),
}

# ============================================================================
# HELPERS
# ============================================================================

def parse_timepoint(filename: str):
    """Extract integer timepoint from filename, e.g. _t0070.png → 70."""
    m = re.search(r'_t(\d+)\.', filename)
    return int(m.group(1)) if m else None


def load_image(path: str) -> np.ndarray | None:
    try:
        return np.array(Image.open(path))
    except Exception as e:
        print(f"  ⚠ Could not load {path}: {e}")
        return None


def collect_timepoints(parent_dir: str, channel_map: dict) -> dict:
    """
    Returns: {timepoint_int: {channel_key: filepath, ...}, ...}
    """
    data = defaultdict(dict)
    for ch_key, (ch_name, folder) in channel_map.items():
        folder_path = os.path.join(parent_dir, folder)
        if not os.path.isdir(folder_path):
            print(f"  ⚠ Folder not found: {folder_path}")
            continue
        files = glob.glob(os.path.join(folder_path, "*.png",)) + glob.glob(os.path.join(folder_path, "*.tif"))
        print(f"  {ch_name} ({ch_key}): {len(files)} files in {folder}")
        for fp in files:
            tp = parse_timepoint(os.path.basename(fp))
            if tp is not None:
                data[tp][ch_key] = fp
    return dict(sorted(data.items()))


def segment_nuclei(h2b_image: np.ndarray, model, diameter, flow_thr, cellprob_thr) -> np.ndarray:
    """Run Cellpose nuclei model on H2B image. Returns labeled mask."""
    inp = h2b_image[np.newaxis, ...]          # add channel dim
    result = model.eval(
        inp,
        diameter=diameter,
        channels=[0, 0],                      # single-channel grayscale
        flow_threshold=flow_thr,
        cellprob_threshold=cellprob_thr,
    )
    masks = result[0] if isinstance(result, tuple) else result
    return masks.squeeze()


def quantify_per_nucleus(masks: np.ndarray, images: dict, channels: list) -> list:
    """
    For each nucleus in `masks`, extract mean/median/sum for each channel.
    Applies background correction (mean of background pixels subtracted).

    Args:
        masks    : Cellpose labeled mask (0 = background)
        images   : {channel_key: 2D numpy array}
        channels : list of channel keys to quantify

    Returns:
        List of per-cell dicts with raw + background-corrected measurements.
    """
    background_mask = masks == 0

    # Pre-compute one background mean per channel
    bg_means = {}
    for ch in channels:
        if ch in images:
            bg_px = images[ch][background_mask]
            bg_means[ch] = float(np.mean(bg_px)) if bg_px.size > 0 else 0.0

    records = []
    for cell_id in np.unique(masks):
        if cell_id == 0:
            continue
        roi = masks == cell_id
        row = {
            "cell_id":   int(cell_id),
            "cell_area": int(roi.sum()),
        }
        for ch in channels:
            if ch not in images:
                continue
            px  = images[ch][roi]
            bg  = bg_means[ch]
            row[f"{ch}_mean_raw"]    = float(px.mean())
            row[f"{ch}_median_raw"]  = float(np.median(px))
            row[f"{ch}_sum_raw"]     = float(px.sum())
            row[f"{ch}_std"]         = float(px.std())
            row[f"{ch}_bg_mean"]     = bg
            # background-corrected
            row[f"{ch}_mean"]        = float(px.mean()   - bg)
            row[f"{ch}_median"]      = float(np.median(px) - bg)
            row[f"{ch}_sum"]         = float(px.sum()    - bg * roi.sum())
        records.append(row)
    return records

# ============================================================================
# PLOTTING
# ============================================================================

def plot_results(df: pd.DataFrame, output_dir: str):
    """Two summary plots: per-cell mean ± SEM and total field intensity over time."""

    channel_style = {
        "ch0": ("GFP",     "#00cc44"),
        "ch1": ("mCherry", "#cc2200"),
    }

    # — Plot 1: mean per cell over time —
    fig, ax = plt.subplots(figsize=(12, 6))
    for ch, (name, color) in channel_style.items():
        col = f"{ch}_mean"
        if col not in df.columns:
            continue
        stats = df.groupby("timepoint")[col].agg(["mean", "std", "count"])
        stats["sem"] = stats["std"] / np.sqrt(stats["count"])
        ax.errorbar(stats.index, stats["mean"], yerr=stats["sem"],
                    label=f"{name} ({ch})", color=color,
                    marker="o", linewidth=2, capsize=4)
    ax.set_xlabel("Timepoint", fontsize=12)
    ax.set_ylabel("Nuclear mean intensity ± SEM\n(background-corrected)", fontsize=12)
    ax.set_title("GFP & mCherry — Mean Nuclear Intensity Over Time", fontsize=13)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    p1 = os.path.join(output_dir, "nuclear_mean_per_cell.png")
    plt.savefig(p1, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: nuclear_mean_per_cell.png")
    plt.close()

    # — Plot 2: total field intensity over time —
    fig, ax = plt.subplots(figsize=(12, 6))
    for ch, (name, color) in channel_style.items():
        col = f"{ch}_sum"
        if col not in df.columns:
            continue
        totals = df.groupby("timepoint")[col].sum()
        ax.plot(totals.index, totals.values,
                label=f"{name} ({ch})", color=color,
                marker="o", linewidth=2)
    ax.set_xlabel("Timepoint", fontsize=12)
    ax.set_ylabel("Total integrated intensity\n(background-corrected)", fontsize=12)
    ax.set_title("GFP & mCherry — Total Field Intensity Over Time", fontsize=13)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(output_dir, "nuclear_total_per_field.png")
    plt.savefig(p2, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: nuclear_total_per_field.png")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "="*65)
    print("NUCLEAR FLUORESCENCE QUANTIFICATION (H2B → GFP + mCherry)")
    print("="*65)
    print(f"Parent dir : {PARENT_DIR}")
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"Output dir : {OUTPUT_DIR}")
    print(f"GPU        : {USE_GPU}")
    print(f"Diameter   : {NUCLEAR_DIAMETER} px")

    # 1. Collect image paths
    print("\n[1/4] Scanning channel folders...")
    timepoints = collect_timepoints(PARENT_DIR, CHANNEL_MAP)
    if not timepoints:
        print("ERROR: No timepoints found. Check PARENT_DIR and EXPERIMENT_NAME.")
        return
    tps = list(timepoints.keys())
    print(f"  → {len(tps)} timepoints found  (t{tps[0]:04d} – t{tps[-1]:04d})")

    # 2. Load Cellpose nuclei model
    print("\n[2/4] Loading Cellpose nuclei model...")
    model = models.Cellpose(gpu=USE_GPU, model_type="cyto3")

    # 3. Process each timepoint
    print("\n[3/4] Segmenting and quantifying...")
    all_records = []

    for idx, (tp, ch_files) in enumerate(timepoints.items(), 1):
        print(f"\n  [{idx}/{len(timepoints)}] t{tp:04d}")

        # Load images
        imgs = {ch: load_image(fp) for ch, fp in ch_files.items()
                if load_image(fp) is not None}
        # reload cleanly (avoid double-load artefact from the dict comprehension)
        imgs = {}
        for ch, fp in ch_files.items():
            arr = load_image(fp)
            if arr is not None:
                imgs[ch] = arr

        if "ch2" not in imgs:
            print("    ✗ H2B (ch2) missing — skipping")
            continue

        # Segment on H2B
        masks = segment_nuclei(imgs["ch2"], model,
                               NUCLEAR_DIAMETER, FLOW_THRESHOLD, CELLPROB_THRESHOLD)
        n_cells = int(masks.max())
        print(f"    Nuclei detected: {n_cells}")

        # Save mask
        mask_path = os.path.join(OUTPUT_DIR, f"h2b_mask_t{tp:04d}.tif")
        tifffile.imwrite(mask_path, masks.astype(np.uint32))

        # Quantify GFP and mCherry within nuclear ROIs
        records = quantify_per_nucleus(masks, imgs, channels=["ch0", "ch1", "ch2"])
        for r in records:
            r["timepoint"] = tp
        all_records.extend(records)

        # Print background values for quick sanity check
        if records:
            for ch in ["ch0", "ch1", "ch2"]:
                bg_key = f"{ch}_bg_mean"
                if bg_key in records[0]:
                    ch_name = CHANNEL_MAP[ch][0]
                    print(f"    {ch_name} background mean: {records[0][bg_key]:.1f}")

    # 4. Save results
    print("\n[4/4] Saving results...")
    if not all_records:
        print("  ✗ No data collected — check image paths and channel availability.")
        return

    df = pd.DataFrame(all_records)

    # Reorder columns neatly
    front = ["timepoint", "cell_id", "cell_area"]
    rest  = sorted([c for c in df.columns if c not in front])
    df    = df[front + rest]

    csv_path = os.path.join(OUTPUT_DIR, "nuclear_gfp_mcherry_quantification.csv")
    df.to_csv(csv_path, index=False)
    print(f"  ✓ CSV saved: nuclear_gfp_mcherry_quantification.csv")
    print(f"  → {len(df)} cells × {df['timepoint'].nunique()} timepoints")

    # Summary stats
    print("\n  Summary (background-corrected):")
    for ch, (name, _) in [("ch0", ("GFP", None)), ("ch1", ("mCherry", None))]:
        col = f"{ch}_mean"
        if col in df.columns:
            print(f"    {name}: mean = {df[col].mean():.1f} ± {df[col].std():.1f}  "
                  f"(range {df[col].min():.1f} – {df[col].max():.1f})")

    # Plots
    plot_results(df, OUTPUT_DIR)

    print(f"\n✓ Done! All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()