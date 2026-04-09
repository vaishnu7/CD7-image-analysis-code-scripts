"""
Cytoplasm Quantification: Binary mask subtraction (no cell ID tracking)
=======================================================================
...
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tifffile
from PIL import Image
from scipy.ndimage import binary_erosion

# ============================================================================
# CONFIGURATION — edit these to match your setup
# ============================================================================

PARENT_DIR      = r"C:\Users\path to folder"
EXPERIMENT_NAME = "New-02-Scene-45-P45-C03"

MASK_DIR   = os.path.join(PARENT_DIR, "segmentation_results")
OUTPUT_DIR = MASK_DIR

CH0_DIR = os.path.join(PARENT_DIR, f"{EXPERIMENT_NAME}_ch1")  # GFP / AKT-KTR
CH1_DIR = os.path.join(PARENT_DIR, f"{EXPERIMENT_NAME}_ch2")  # mCherry / ERK-KTR
CH2_DIR = os.path.join(PARENT_DIR, f"{EXPERIMENT_NAME}_ch3")  # iRFP (normalisation)

Y_MIN = 0
Y_MAX = None  # None = auto-scale

VISUALISE_EVERY_N = 1

NUC_EROSION_PX  = 0
CYTO_EROSION_PX = 0

# ============================================================================
# HELPERS
# ============================================================================

def parse_timepoint(filename):
    m = re.search(r'_?t(\d+)\.', filename)
    return int(m.group(1)) if m else None

def load_img(path):
    try:
        return np.array(Image.open(path)).astype(np.float32)
    except Exception:
        return tifffile.imread(path).astype(np.float32)

def collect_masks(folder, pattern):
    files = {}
    for fp in glob.glob(os.path.join(folder, pattern)):
        tp = parse_timepoint(os.path.basename(fp))
        if tp is not None:
            files[tp] = fp
    return files

def collect_imgs(folder):
    files = {}
    for ext in ("*.png", "*.tif", "*.tiff"):
        for fp in glob.glob(os.path.join(folder, ext)):
            tp = parse_timepoint(os.path.basename(fp))
            if tp is not None:
                files[tp] = fp
    return files

# ============================================================================
# MASK VISUALISATION
# ============================================================================

def save_mask_overlay(cell_binary, nuc_binary, nuc_eroded,
                      cyto_binary, cyto_eroded,
                      ch1, ch2, tp, out_dir):
    def norm(img):
        p2, p98 = np.percentile(img, 2), np.percentile(img, 98)
        return np.clip((img - p2) / (p98 - p2 + 1e-6), 0, 1)

    h, w = cell_binary.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    nuc_edge  = nuc_binary  & ~nuc_eroded
    cyto_edge = cyto_binary & ~cyto_eroded

    rgb[cyto_edge,   0] = 0.8;  rgb[cyto_edge,   1] = 0.4
    rgb[nuc_edge,    1] = 0.5;  rgb[nuc_edge,    2] = 0.7
    rgb[cyto_eroded, 1] = 0.85
    rgb[nuc_eroded,  2] = 1.0

    fig, axes = plt.subplots(1, 7, figsize=(30, 5))
    fig.suptitle(
        f"Mask erosion visualisation — t{tp:04d}  "
        f"(nuc erosion={NUC_EROSION_PX}px  cyto erosion={CYTO_EROSION_PX}px)",
        fontsize=12, fontweight="bold"
    )

    panels = [
        (norm(ch1),    "Ch1 — GFP (AKT-KTR)",                       "gray"),
        (norm(ch2),    "Ch2 — mCherry (ERK-KTR)",                    "gray"),
        (cell_binary,  "Whole-cell mask (DIC)",                       "gray"),
        (nuc_binary,   "Nucleus mask — full",                         "Blues"),
        (nuc_eroded,   f"Nucleus mask — eroded {NUC_EROSION_PX}px",  "Blues"),
        (cyto_eroded,  f"Cytoplasm mask — eroded {CYTO_EROSION_PX}px","Greens"),
        (rgb,          "RGB overlay",                                  None),
    ]

    for ax, (data, title, cmap) in zip(axes, panels):
        if cmap is None:
            ax.imshow(data, interpolation="nearest")
        else:
            ax.imshow(data, cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.axis("off")

    legend_elements = [
        mpatches.Patch(facecolor="#00dd00", label="Cytoplasm used"),
        mpatches.Patch(facecolor="#cc6600", label="Cytoplasm edge excluded"),
        mpatches.Patch(facecolor="#0000ff", label="Nucleus used"),
        mpatches.Patch(facecolor="#008099", label="Nucleus edge excluded"),
        mpatches.Patch(facecolor="black",   label="Background", edgecolor="white"),
    ]
    axes[-1].legend(handles=legend_elements, loc="lower right",
                    fontsize=7, framealpha=0.7)

    plt.tight_layout()
    fname = os.path.join(out_dir, f"mask_overlay_t{tp:04d}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():

    print("\n" + "="*60)
    print("CYTOPLASM = WHOLE-CELL BINARY  minus  NUCLEUS BINARY")
    print("Population-level C/N ratio (no per-cell tracking)")
    print("="*60)
    print(f"  Nucleus erosion  : {NUC_EROSION_PX  if NUC_EROSION_PX  > 0 else 'DISABLED'} px")
    print(f"  Cytoplasm erosion: {CYTO_EROSION_PX if CYTO_EROSION_PX > 0 else 'DISABLED'} px")
    print()

    dic_masks = collect_masks(MASK_DIR, "masks_dic_t*.tif")
    h2b_masks = collect_masks(MASK_DIR, "masks_h2b_t*.tif")
    ch0_files = collect_imgs(CH0_DIR)
    ch1_files = collect_imgs(CH1_DIR)
    ch2_files = collect_imgs(CH2_DIR)  # iRFP

    shared_tps = sorted(set(dic_masks) & set(h2b_masks) & set(ch0_files) & set(ch1_files) & set(ch2_files))
    print(f"Found {len(shared_tps)} timepoints with all 5 inputs\n")

    if not shared_tps:
        print("ERROR: No shared timepoints found. Check your paths and filenames.")
        return

    records = []

    overlay_dir = os.path.join(OUTPUT_DIR, "mask_overlay")
    if VISUALISE_EVERY_N is not None:
        os.makedirs(overlay_dir, exist_ok=True)

    for idx, tp in enumerate(shared_tps, 1):
        print(f"  [{idx:3d}/{len(shared_tps)}] t{tp:04d}", end="  ")

        dic_mask = tifffile.imread(dic_masks[tp])
        h2b_mask = tifffile.imread(h2b_masks[tp])

        cell_binary = dic_mask > 0
        nuc_binary  = h2b_mask > 0
        cyto_binary = cell_binary & ~nuc_binary
        bg_mask = ~cell_binary

        nuc_eroded  = binary_erosion(nuc_binary,  iterations=NUC_EROSION_PX)  if NUC_EROSION_PX  > 0 else nuc_binary
        cyto_eroded = binary_erosion(cyto_binary, iterations=CYTO_EROSION_PX) if CYTO_EROSION_PX > 0 else cyto_binary

        nuc_area_full  = int(nuc_binary.sum())
        nuc_area_used  = int(nuc_eroded.sum())
        cyto_area_full = int(cyto_binary.sum())
        cyto_area_used = int(cyto_eroded.sum())
        cell_area      = int(cell_binary.sum())

        ch0  = load_img(ch0_files[tp])
        ch1  = load_img(ch1_files[tp])
        irfp = load_img(ch2_files[tp])  # iRFP

        # --- iRFP nucleus mean (background-subtracted) ---
        irfp_bg_mean  = float(np.median(irfp[bg_mask])) if bg_mask.any() else 0.0
        irfp_nuc_mean = float(np.median(irfp[nuc_eroded])) - irfp_bg_mean if nuc_area_used > 0 else np.nan

        row = {
            "timepoint":      tp,
            "cell_area_px":   cell_area,
            "nuc_area_full":  nuc_area_full,
            "nuc_area_used":  nuc_area_used,
            "cyto_area_full": cyto_area_full,
            "cyto_area_used": cyto_area_used,
            "irfp_nuc_mean":  irfp_nuc_mean,
        }

        for ch_name, img in [("ch1", ch0), ("ch2", ch1)]:
            bg_mean   = float(np.median(img[bg_mask])) if bg_mask.any() else 0.0
            nuc_mean  = float(np.median(img[nuc_eroded]))  - bg_mean if nuc_area_used  > 0 else np.nan
            cyto_mean = float(np.median(img[cyto_eroded])) - bg_mean if cyto_area_used > 0 else np.nan
            cn_ratio  = (cyto_mean / nuc_mean) if (nuc_mean and nuc_mean > 1e-6) else np.nan

            # C / (N / N_iRFP)  =  C * N_iRFP / N
            cn_irfp = (cyto_mean * irfp_nuc_mean / nuc_mean) \
                      if (nuc_mean and nuc_mean > 1e-6 and irfp_nuc_mean and irfp_nuc_mean > 1e-6) \
                      else np.nan

            row[f"{ch_name}_nuc_mean"]  = nuc_mean
            row[f"{ch_name}_cyto_mean"] = cyto_mean
            row[f"{ch_name}_cn_ratio"]  = cn_ratio
            row[f"{ch_name}_cn_irfp"]   = cn_irfp   # C / (N / N_iRFP)

        records.append(row)
        print(f"→ cell={cell_area}px  nuc_used={nuc_area_used}px  "
              f"cyto_used={cyto_area_used}px  "
              f"ch1_CN={row['ch1_cn_ratio']:.3f}  ch2_CN={row['ch2_cn_ratio']:.3f}  "
              f"ch1_CN_iRFP={row['ch1_cn_irfp']:.3f}  ch2_CN_iRFP={row['ch2_cn_irfp']:.3f}")

        if VISUALISE_EVERY_N is not None and (idx - 1) % VISUALISE_EVERY_N == 0:
            save_mask_overlay(cell_binary, nuc_binary, nuc_eroded,
                              cyto_binary, cyto_eroded,
                              ch0, ch1, tp, overlay_dir)
            print(f"           ↳ overlay saved: mask_overlay_t{tp:04d}.png")

    if not records:
        print("\nERROR: No records produced. Check mask files.")
        return

    df = pd.DataFrame(records)
    csv_path = os.path.join(OUTPUT_DIR, "cytoplasm_binary_quantification.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ CSV saved: cytoplasm_binary_quantification.csv  ({len(df)} rows)")

    print("\n── C/N ratio summary ──")
    for ch in ["ch1", "ch2"]:
        for col in [f"{ch}_cn_ratio", f"{ch}_cn_irfp"]:
            print(f"  {col}: mean={df[col].mean():.3f}  "
                  f"median={df[col].median():.3f}  "
                  f"range=[{df[col].min():.3f}, {df[col].max():.3f}]")

    ch_styles = {
        "ch1": ("GFP (AKT-KTR)",     "#00cc44"),
        "ch2": ("mCherry (ERK-KTR)", "#ff4444"),
    }

    for ch, (name, color) in ch_styles.items():
        tp_vals = df["timepoint"].values

        # --- Plot 1: C/N ratio ---
        col_cn = f"{ch}_cn_ratio"
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(tp_vals, df[col_cn].values,
                marker='o', linestyle='-', linewidth=2,
                markersize=5, color=color, label=name)
        ax.axhline(1, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, label="C/N = 1")
        if Y_MAX is not None:
            ax.set_ylim(Y_MIN, Y_MAX)
        else:
            ax.set_ylim(bottom=Y_MIN)
        ax.set_xlabel("Timepoint", fontsize=12)
        ax.set_ylabel("Cytoplasm / Nucleus (median intensity)", fontsize=12)
        ax.set_title(
            f"KTR C/N Ratio — {name}\n"
            f"(nuc eroded {NUC_EROSION_PX}px, cyto eroded {CYTO_EROSION_PX}px, population-level)",
            fontsize=13, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fname = f"cn_ratio_binary_{ch}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved: {fname}")
        plt.close()

        # --- Plot 2: C / (N / N_iRFP) ---
        col_cn_irfp = f"{ch}_cn_irfp"
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(tp_vals, df[col_cn_irfp].values,
                marker='o', linestyle='-', linewidth=2,
                markersize=5, color=color, label=name)
        ax.set_xlabel("Timepoint", fontsize=12)
        ax.set_ylabel("C / (N / N_iRFP)  (a.u.)", fontsize=12)
        ax.set_title(
            f"iRFP-normalised KTR ratio — {name}\n"
            f"C_fluo / (N_fluo / N_iRFP), population-level",
            fontsize=13, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        fname = f"cn_irfp_binary_{ch}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved: {fname}")
        plt.close()

        # --- Plot 3: raw compartment fluorescence ---
        col_cyto = f"{ch}_cyto_mean"
        col_nuc  = f"{ch}_nuc_mean"
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(tp_vals, df[col_cyto].values,
                marker='o', linestyle='-', linewidth=2,
                markersize=5, color=color, label="Cytoplasm")
        ax.plot(tp_vals, df[col_nuc].values,
                marker='s', linestyle='--', linewidth=1.5,
                markersize=4, color=color, alpha=0.45, label="Nucleus")
        ax.set_xlabel("Timepoint", fontsize=12)
        ax.set_ylabel("Mean fluorescence intensity (a.u.)", fontsize=12)
        ax.set_title(
            f"Compartment Fluorescence — {name}\n"
            f"(cytoplasm solid, nucleus dashed, population-level)",
            fontsize=13, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        fname = f"fluo_cyto_{ch}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved: {fname}")
        plt.close()

    print(f"\n✓ Done. All outputs saved to:\n  {OUTPUT_DIR}")
    if VISUALISE_EVERY_N is not None:
        print(f"  Mask overlays: {overlay_dir}")


if __name__ == "__main__":
    main()
