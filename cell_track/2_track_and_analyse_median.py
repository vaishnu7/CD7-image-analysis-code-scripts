"""
STEP 2 - Track + Measure Fluorescence
=======================================
Loads the masks saved by step1_segment.py, then:
  - Shows frame 0 with all detected cells coloured and numbered
  - You click the cell you want to track
  - Tracks that cell across all frames using last-known centroid search
  - Measures GFP (ch1) and mCherry (ch2) fluorescence:
      raw        : mean(compartment pixels) - background
      H2B-norm   : raw / H2B(ch3) in same compartment
      C/N ratio  : cytoplasmic / nuclear   (raw and H2B-normalised)
      t0-norm    : C/N(t) / mean(C/N over first BASELINE_FRAMES frames)
  - Saves results CSV and fluorescence plots

Prerequisites:
  Run step1_segment.py first to generate the masks folder.

Output (saved to OUTPUT_FOLDER):
  results.csv                  -- all measurements per frame
  plot_raw.png                 -- raw bg-subtracted fluorescence
  plot_normalised.png          -- H2B-normalised fluorescence
  plot_cn_ratio_raw.png        -- C/N ratio (raw, top row) and
                                  t0-normalised (bottom row)
  tracking_overlay.png         -- per-frame segmentation overlay
"""

import os
import re
import math
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("TkAgg")   # change to "Qt5Agg" if needed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_erosion


# ======================================================================
#                         USER SETTINGS
# ======================================================================

CH1_FOLDER   = r"C:\Users\gz24763\OneDrive - University of Bristol\Documents\PC9_images\18_03_2026\B2\P17\New-02-Scene-36-P17-B02_ch1"
CH2_FOLDER   = r"C:\Users\gz24763\OneDrive - University of Bristol\Documents\PC9_images\18_03_2026\B2\P17\New-02-Scene-36-P17-B02_ch2"
CH3_FOLDER   = r"C:\Users\gz24763\OneDrive - University of Bristol\Documents\PC9_images\18_03_2026\B2\P17\New-02-Scene-36-P17-B02_ch3"

# Folder produced by step1_segment.py
MASKS_FOLDER = r"C:\Users\gz24763\OneDrive - University of Bristol\Documents\PC9_images\18_03_2026\B2\P17\masks"

# Where to save results
OUTPUT_FOLDER = r"C:\Users\gz24763\OneDrive - University of Bristol\Documents\PC9_images\18_03_2026\B2\P17\results"

# Max centroid displacement (px) allowed between consecutive frames
MAX_DIST = 150

# Max frames to track (None = all frames with saved masks)
MAX_FRAMES = None

# How many frames to show in the tracking overlay (None = all)
OVERLAY_MAX_FRAMES = 20

# Verify tracking every N frames (shows result, asks y/n/s in terminal)
# Set to 1 to verify EVERY frame, 5 to verify every 5th frame, None to never verify
VERIFY_EVERY_N = 30

# Nucleus mask erosion in pixels applied before fluorescence measurement
NUC_EROSION_PX = 2

# Baseline normalisation
# Number of frames at the START of the timecourse to use as baseline.
# Used for:
#   1. Baseline-subtracted C/N : C/N(t) - mean(C/N[0:BASELINE_FRAMES])
#   2. t0-normalised C/N       : C/N(t) / mean(C/N[0:BASELINE_FRAMES])  <-- NEW
# Set to 0 to skip baseline subtraction (t0-norm will use frame 0 only).
BASELINE_FRAMES = 0

# ======================================================================


# ----------------------------------------------------------------------
# FILE DISCOVERY
# ----------------------------------------------------------------------

def _sorted_tifs(folder):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".tif"))
    return [os.path.join(folder, f) for f in files]


def _frame_index(path):
    m = re.search(r"_t(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def load_masks_and_fluo():
    """
    Loads saved masks from MASKS_FOLDER and matches them with
    ch1/ch2/ch3 fluorescence files by frame index.
    Returns sorted list of dicts, one per frame.
    """
    crop_coords = np.load(os.path.join(MASKS_FOLDER, "crop_coords.npy"))
    r1, r2, c1, c2 = int(crop_coords[0]), int(crop_coords[1]), \
                     int(crop_coords[2]), int(crop_coords[3])

    mask_files = [f for f in os.listdir(MASKS_FOLDER)
                  if f.startswith("cell_mask_t") and f.endswith(".npy")]
    frame_indices = sorted(
        int(re.search(r"t(\d+)", f).group(1)) for f in mask_files)

    if MAX_FRAMES is not None:
        frame_indices = frame_indices[:MAX_FRAMES]

    ch1_files = {_frame_index(p): p for p in _sorted_tifs(CH1_FOLDER)}
    ch2_files = {_frame_index(p): p for p in _sorted_tifs(CH2_FOLDER)}
    ch3_files = {_frame_index(p): p for p in _sorted_tifs(CH3_FOLDER)}

    records = []
    for t in frame_indices:
        if not all(t in d for d in [ch1_files, ch2_files, ch3_files]):
            print("  WARNING: frame {} missing fluorescence files, skipping.".format(t))
            continue
        records.append({
            "t":         t,
            "cell_mask": np.load(os.path.join(MASKS_FOLDER,
                                 "cell_mask_t{:04d}.npy".format(t))),
            "nuc_mask":  np.load(os.path.join(MASKS_FOLDER,
                                 "nuc_mask_t{:04d}.npy".format(t))),
            "ch0_crop":  np.load(os.path.join(MASKS_FOLDER,
                                 "ch0_crop_t{:04d}.npy".format(t))),
            "ch1_path":  ch1_files[t],
            "ch2_path":  ch2_files[t],
            "ch3_path":  ch3_files[t],
            "r1": r1, "r2": r2, "c1": c1, "c2": c2,
        })
    return records


# ----------------------------------------------------------------------
# CELL PICKER
# ----------------------------------------------------------------------

def get_centroids(mask):
    centroids = {}
    for lbl in np.unique(mask):
        if lbl == 0:
            continue
        rows, cols = np.where(mask == lbl)
        centroids[lbl] = (rows.mean(), cols.mean())
    return centroids


def pick_cell(ch0_crop, cell_mask):
    """
    Shows frame 0 with all cells coloured.
    Click on the cell you want to track.
    Returns chosen cellpose label (int) and its centroid (row, col).
    """
    lo, hi = np.percentile(ch0_crop, 1), np.percentile(ch0_crop, 99)
    display = np.clip((ch0_crop.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)

    n_labels = int(cell_mask.max())
    overlay  = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
    colours  = cm.tab20(np.linspace(0, 1, max(n_labels, 1)))
    for i, lbl in enumerate(range(1, n_labels + 1)):
        px = cell_mask == lbl
        overlay[px, :3] = colours[i % len(colours)][:3]
        overlay[px,  3] = 0.4

    centroids = get_centroids(cell_mask)
    chosen    = {"label": None}

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        col_i = int(np.clip(round(event.xdata), 0, cell_mask.shape[1] - 1))
        row_i = int(np.clip(round(event.ydata), 0, cell_mask.shape[0] - 1))
        lbl   = int(cell_mask[row_i, col_i])
        if lbl == 0:
            print("  Clicked background -- click inside a coloured cell.")
            return
        chosen["label"] = lbl
        for sc in ax.collections:
            sc.remove()
        yx = np.argwhere(cell_mask == lbl)
        ax.scatter(yx[:, 1], yx[:, 0], s=1, c="lime", alpha=0.6, linewidths=0)
        ax.set_title("Cell {} selected!  Close window to start tracking.".format(lbl),
                     color="lime", fontsize=12, fontweight="bold")
        fig.canvas.draw_idle()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(display, cmap="gray")
    ax.imshow(overlay)
    for lbl, (r, c) in centroids.items():
        ax.text(c, r, str(lbl), color="white", fontsize=7,
                ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5, lw=0))
    ax.set_title("Click the cell to track,  then close this window")
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()

    if chosen["label"] is None:
        raise RuntimeError("No cell selected. Please re-run.")
    lbl = chosen["label"]
    print("  Cell {} selected.  Centroid: ({:.1f}, {:.1f})".format(
        lbl, *centroids[lbl]))
    return lbl, centroids[lbl]


# ----------------------------------------------------------------------
# TRACKING
# ----------------------------------------------------------------------

def get_cell_area(mask, lbl):
    return int(np.sum(mask == lbl))


def find_cell(cell_mask, last_centroid, max_dist, last_area=None):
    """
    Find the closest cell within max_dist.
    Also detects division: if 2 cells are close AND closest is <70% of last area.
    Returns (label, centroid, area, division_candidates)
    """
    centroids = get_centroids(cell_mask)
    if not centroids:
        return None, None, None, []

    lr, lc = last_centroid
    ranked = sorted(
        [(lbl, c, math.sqrt((c[0]-lr)**2 + (c[1]-lc)**2))
         for lbl, c in centroids.items()],
        key=lambda x: x[2]
    )
    candidates = [(lbl, c, d) for lbl, c, d in ranked if d <= max_dist]

    if not candidates:
        return None, None, None, []

    best_lbl, best_centroid, _ = candidates[0]
    best_area = get_cell_area(cell_mask, best_lbl)

    division_candidates = []
    if len(candidates) >= 2 and last_area is not None:
        if best_area / last_area < 0.70:
            for lbl2, c2, d2 in candidates[1:]:
                division_candidates.append((lbl2, c2, get_cell_area(cell_mask, lbl2)))

    return best_lbl, best_centroid, best_area, division_candidates


# ----------------------------------------------------------------------
# VERIFY + MANUAL RECLICK
# ----------------------------------------------------------------------

def verify_and_reclick(ch0_crop, cell_mask, nuc_mask, auto_lbl, auto_centroid, frame_t):
    lo, hi = np.percentile(ch0_crop, 1), np.percentile(ch0_crop, 99)
    disp = np.clip((ch0_crop.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)

    n_lbl = int(cell_mask.max())
    overlay = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
    colours = cm.tab20(np.linspace(0, 1, max(n_lbl, 1)))
    for i, l in enumerate(range(1, n_lbl + 1)):
        px = cell_mask == l
        overlay[px, :3] = colours[i % len(colours)][:3]
        overlay[px,  3] = 0.25

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(disp, cmap="gray")
    ax.imshow(overlay)

    if auto_lbl is not None:
        cell_px = cell_mask == auto_lbl
        cho = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
        cho[cell_px] = [0.0, 1.0, 0.2, 0.55]
        ax.imshow(cho)
        nuc_px = cell_px & (nuc_mask > 0)
        if nuc_px.any():
            nuo = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
            nuo[nuc_px] = [0.0, 1.0, 1.0, 0.7]
            ax.imshow(nuo)
        cr, cc = auto_centroid
        ax.plot(cc, cr, "+", color="white", markersize=10, markeredgewidth=2)
        title_color = "lime"
        title_text  = ("Frame {:04d}  --  auto-detected cell (green)\n"
                       "Close this window, then answer y/n in the terminal").format(frame_t)
    else:
        title_color = "red"
        title_text  = ("Frame {:04d}  --  NO cell found automatically\n"
                       "Close this window, then answer y/n in the terminal").format(frame_t)

    ax.set_title(title_text, color=title_color, fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    while True:
        ans = input("  Frame {:04d}: correct? [y=yes / n=reclick / s=skip]  ".format(
            frame_t)).strip().lower()
        if ans in ("y", ""):
            return auto_lbl, auto_centroid
        elif ans == "s":
            print("  Skipped.")
            return None, auto_centroid
        elif ans == "n":
            break

    print("  Click the correct cell in the window that opens ...")
    centroids = get_centroids(cell_mask)
    chosen    = {"label": None}

    def on_click(event):
        if event.inaxes != ax2 or event.button != 1:
            return
        col_i = int(np.clip(round(event.xdata), 0, cell_mask.shape[1] - 1))
        row_i = int(np.clip(round(event.ydata), 0, cell_mask.shape[0] - 1))
        lbl   = int(cell_mask[row_i, col_i])
        if lbl == 0:
            print("  Clicked background -- click inside a cell.")
            return
        chosen["label"] = lbl
        for sc in ax2.collections:
            sc.remove()
        yx = np.argwhere(cell_mask == lbl)
        ax2.scatter(yx[:, 1], yx[:, 0], s=1, c="lime", alpha=0.6, linewidths=0)
        ax2.set_title("Cell {} selected!  Close to confirm.".format(lbl),
                      color="lime", fontsize=11, fontweight="bold")
        fig2.canvas.draw_idle()

    fig2, ax2 = plt.subplots(figsize=(9, 7))
    ax2.imshow(disp, cmap="gray")
    ax2.imshow(overlay)
    for lbl, (r, c) in centroids.items():
        ax2.text(c, r, str(lbl), color="white", fontsize=7,
                 ha="center", va="center", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5, lw=0))
    ax2.set_title("Frame {:04d}  --  click the CORRECT cell, then close".format(frame_t))
    ax2.axis("off")
    fig2.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()

    if chosen["label"] is None:
        print("  No cell clicked -- skipping frame.")
        return None, auto_centroid if auto_centroid is not None else (0, 0)

    new_lbl      = chosen["label"]
    new_centroid = centroids[new_lbl]
    print("  Manually selected cell {}.  Centroid: ({:.1f}, {:.1f})".format(
        new_lbl, *new_centroid))
    return new_lbl, new_centroid


# ----------------------------------------------------------------------
# DIVISION PICKER
# ----------------------------------------------------------------------

def pick_daughter(ch0_crop, cell_mask, auto_lbl, auto_centroid,
                  division_candidates, frame_t):
    lo, hi = np.percentile(ch0_crop, 1), np.percentile(ch0_crop, 99)
    disp = np.clip((ch0_crop.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)

    n_lbl = int(cell_mask.max())
    overlay = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
    colours = cm.tab20(np.linspace(0, 1, max(n_lbl, 1)))
    for i, l in enumerate(range(1, n_lbl + 1)):
        px = cell_mask == l
        overlay[px, :3] = colours[i % len(colours)][:3]
        overlay[px,  3] = 0.2

    if auto_lbl is not None:
        cho = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
        cho[cell_mask == auto_lbl] = [0.0, 1.0, 0.2, 0.6]
        overlay = np.clip(overlay + cho, 0, 1)

    for lbl2, c2, a2 in division_candidates:
        yel = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
        yel[cell_mask == lbl2] = [1.0, 1.0, 0.0, 0.6]
        overlay = np.clip(overlay + yel, 0, 1)

    chosen = {"label": None, "centroid": None}

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        col_i = int(np.clip(round(event.xdata), 0, cell_mask.shape[1] - 1))
        row_i = int(np.clip(round(event.ydata), 0, cell_mask.shape[0] - 1))
        lbl = int(cell_mask[row_i, col_i])
        if lbl == 0:
            print("  Clicked background -- click a cell.")
            return
        chosen["label"] = lbl
        chosen["centroid"] = get_centroids(cell_mask)[lbl]
        for sc in ax.collections:
            sc.remove()
        yx = np.argwhere(cell_mask == lbl)
        ax.scatter(yx[:, 1], yx[:, 0], s=1, c="cyan", alpha=0.7, linewidths=0)
        ax.set_title("Daughter {} selected!  Close to continue.".format(lbl),
                     color="cyan", fontsize=11, fontweight="bold")
        fig.canvas.draw_idle()

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(disp, cmap="gray")
    ax.imshow(overlay)

    for lbl_c, (r_c, c_c) in [(auto_lbl, auto_centroid)] + \
            [(l, c) for l, c, a in division_candidates]:
        if lbl_c is not None:
            ax.text(c_c, r_c, str(lbl_c), color="white", fontsize=8,
                    ha="center", va="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.6, lw=0))

    ax.set_title(
        "Frame {:04d}  --  DIVISION DETECTED!\n"
        "GREEN = auto cell,   YELLOW = other daughter(s)\n"
        "Click the daughter to track, then close".format(frame_t),
        color="yellow", fontsize=10)
    ax.axis("off")
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()

    if chosen["label"] is None:
        print("  No daughter selected -- keeping auto cell.")
        return auto_lbl, auto_centroid, get_cell_area(cell_mask, auto_lbl)

    new_lbl = chosen["label"]
    new_centroid = chosen["centroid"]
    print("  Daughter {} chosen.  Centroid ({:.1f},{:.1f})  Area {} px".format(
        new_lbl, *new_centroid, get_cell_area(cell_mask, new_lbl)))
    return new_lbl, new_centroid, get_cell_area(cell_mask, new_lbl)


# ----------------------------------------------------------------------
# FLUORESCENCE MEASUREMENT
# ----------------------------------------------------------------------

def measure(cell_mask, nuc_mask, ch1, ch2, ch3, chosen_lbl, r1, r2, c1, c2):

    nan = float("nan")
    empty = dict(
        bg_gfp=nan, bg_mcherry=nan, bg_h2b=nan, bg_n_pixels=nan,
        gfp_total=nan, gfp_nuclear=nan, gfp_cytoplasmic=nan,
        mcherry_total=nan, mcherry_nuclear=nan, mcherry_cytoplasmic=nan,
        h2b_nuclear=nan, h2b_cytoplasmic=nan,
        gfp_nuclear_norm=nan, gfp_cytoplasmic_norm=nan,
        mcherry_nuclear_norm=nan, mcherry_cytoplasmic_norm=nan,
        gfp_cn_ratio=nan, mcherry_cn_ratio=nan,
        gfp_cn_ratio_norm=nan, mcherry_cn_ratio_norm=nan,
    )
    if chosen_lbl is None:
        return empty

    ch1c = ch1[r1:r2, c1:c2]
    ch2c = ch2[r1:r2, c1:c2]
    ch3c = ch3[r1:r2, c1:c2]

    bg_mask = cell_mask == 0
    bg_n    = int(bg_mask.sum())
    bg1 = float(np.median(ch1c[bg_mask]))
    bg2 = float(np.median(ch2c[bg_mask]))
    bg3 = float(np.median(ch3c[bg_mask]))

    def median_px(arr, mask):
        return float(np.median(arr[mask])) if mask.any() else nan

    def safe_div(a, b):
        if math.isnan(a) or math.isnan(b) or b == 0:
            return nan
        return a / b

    cell_px = cell_mask == chosen_lbl

    if NUC_EROSION_PX > 0:
        nuc_eroded = binary_erosion(nuc_mask > 0, iterations=NUC_EROSION_PX)
        if not (cell_px & nuc_eroded).any():
            print(f"  WARNING: nucleus fully eroded at this frame — using uneroded mask")
            nuc_eroded = nuc_mask > 0
    else:
        nuc_eroded = nuc_mask > 0

    nuc_px  = cell_px & nuc_eroded
    cyto_px = cell_px & ~nuc_eroded

    print(f"  bg_gfp={bg1:.0f}  "
          f"raw_nuc_median={float(np.median(ch1c[nuc_px])) if nuc_px.any() else 'EMPTY':.0f}  "
          f"raw_cyto_median={float(np.median(ch1c[cyto_px])) if cyto_px.any() else 'EMPTY':.0f}  "
          f"nuc_px_count={nuc_px.sum()}  cyto_px_count={cyto_px.sum()}")

    gfp_tot  = median_px(ch1c, cell_px) - bg1
    gfp_nuc  = median_px(ch1c, nuc_px)  - bg1
    gfp_cyto = median_px(ch1c, cyto_px) - bg1
    mch_tot  = median_px(ch2c, cell_px) - bg2
    mch_nuc  = median_px(ch2c, nuc_px)  - bg2
    mch_cyto = median_px(ch2c, cyto_px) - bg2
    h2b_nuc  = median_px(ch3c, nuc_px)  - bg3
    h2b_cyto = median_px(ch3c, cyto_px) - bg3

    gfp_nuc_n  = safe_div(gfp_nuc,  h2b_nuc)
    gfp_cyto_n = safe_div(gfp_cyto, h2b_cyto)
    mch_nuc_n  = safe_div(mch_nuc,  h2b_nuc)
    mch_cyto_n = safe_div(mch_cyto, h2b_cyto)

    #gfp_nuc_h2b_norm = safe_div(gfp_nuc, h2b_nuc)
    #mch_nuc_h2b_norm = safe_div(mch_nuc, h2b_nuc)
    gfp_nuc_h2b_norm = gfp_nuc / h2b_nuc
    mch_nuc_h2b_norm = mch_nuc / h2b_nuc

    gfp_cn      = safe_div(gfp_cyto, gfp_nuc)
    mch_cn      = safe_div(mch_cyto, mch_nuc)

    gfp_cn_norm = safe_div(safe_div(gfp_cyto, h2b_cyto),
                           safe_div(gfp_nuc,  h2b_nuc))
    mch_cn_norm = safe_div(safe_div(mch_cyto, h2b_cyto),
                           safe_div(mch_nuc,  h2b_nuc))

    return dict(
        bg_gfp=bg1, bg_mcherry=bg2, bg_h2b=bg3, bg_n_pixels=bg_n,
        gfp_total=gfp_tot,        gfp_nuclear=gfp_nuc,       gfp_cytoplasmic=gfp_cyto,
        mcherry_total=mch_tot,    mcherry_nuclear=mch_nuc,    mcherry_cytoplasmic=mch_cyto,
        h2b_nuclear=h2b_nuc,      h2b_cytoplasmic=h2b_cyto,
        gfp_nuclear_norm=gfp_nuc_n,      gfp_cytoplasmic_norm=gfp_cyto_n,
        mcherry_nuclear_norm=mch_nuc_n,  mcherry_cytoplasmic_norm=mch_cyto_n,
        gfp_cn_ratio=gfp_cn,             mcherry_cn_ratio=mch_cn,
        gfp_cn_ratio_norm=gfp_cn_norm,   mcherry_cn_ratio_norm=mch_cn_norm,
    )


# ----------------------------------------------------------------------
# TRACKING OVERLAY
# ----------------------------------------------------------------------

def save_tracking_overlay(frame_records, output_path):
    n     = len(frame_records)
    ncols = min(5, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.2))
    fig.suptitle("Tracking overlay", fontsize=13, fontweight="bold")
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for idx, rec in enumerate(frame_records):
        ax        = axes_flat[idx]
        ch0       = rec["ch0_crop"]
        cell_mask = rec["cell_mask"]
        nuc_mask  = rec["nuc_mask"]
        lbl       = rec["chosen_lbl"]

        lo, hi = np.percentile(ch0, 1), np.percentile(ch0, 99)
        disp = np.clip((ch0.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
        ax.imshow(disp, cmap="gray", interpolation="nearest")

        n_lbl = int(cell_mask.max())
        if n_lbl > 0:
            ov = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
            cols = cm.tab20(np.linspace(0, 1, max(n_lbl, 1)))
            for i, l in enumerate(range(1, n_lbl + 1)):
                px = cell_mask == l
                ov[px, :3] = cols[i % len(cols)][:3]
                ov[px,  3] = 0.25
            ax.imshow(ov, interpolation="nearest")

        if lbl is not None:
            cell_px = cell_mask == lbl

            cho = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
            cho[cell_px] = [0.0, 1.0, 0.2, 0.55]
            ax.imshow(cho, interpolation="nearest")

            nuc_px = cell_px & (nuc_mask > 0)
            if nuc_px.any():
                nuo = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
                nuo[nuc_px] = [0.0, 1.0, 1.0, 0.7]
                ax.imshow(nuo, interpolation="nearest")

            border = cell_px & ~binary_erosion(cell_px)
            yx = np.argwhere(border)
            if len(yx):
                ax.scatter(yx[:, 1], yx[:, 0], s=0.3, c="white",
                           linewidths=0, alpha=0.9)
            cr, cc = rec["chosen_centroid"]
            ax.plot(cc, cr, "+", color="white", markersize=8, markeredgewidth=1.5)
            ax.set_title("Frame {:04d} [found]".format(rec["t"]),
                         fontsize=7, color="lime")
        else:
            ax.set_title("Frame {:04d} [lost]".format(rec["t"]),
                         fontsize=7, color="red")
        ax.axis("off")

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print("  Overlay saved --> {}".format(output_path))
    plt.show()


# ----------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------

def save_plots(df, n_found, n_total, output_folder):
    df_f = df.dropna(subset=["gfp_total"])
    title_suffix = "({}/{} frames found)".format(n_found, n_total)
    frames = df_f["frame"]

    def make_plot(specs, filename, suptitle, nrows, ncols, figsize):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        fig.suptitle("{}\n{}".format(suptitle, title_suffix),
                     fontsize=12, fontweight="bold")
        axes_flat = np.array(axes).flatten()
        for i, (col, title, colour) in enumerate(specs):
            ax = axes_flat[i]
            ax.plot(frames, df_f[col], color=colour,
                    linewidth=1.8, marker="o", markersize=4)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Frame")
            ax.set_ylabel("Intensity" if "norm" not in col and "ratio" not in col
                          else "Ratio")
            ax.grid(True, alpha=0.3)
        for i in range(len(specs), len(axes_flat)):
            axes_flat[i].axis("off")
        plt.tight_layout()
        path = os.path.join(output_folder, filename)
        plt.savefig(path, dpi=150)
        print("  Plot saved --> {}".format(path))
        plt.show()

    # --- Plot 1: Raw bg-subtracted ---
    make_plot([
        ("gfp_total",          "GFP total",           "green"),
        ("gfp_nuclear",        "GFP nuclear",         "limegreen"),
        ("gfp_cytoplasmic",    "GFP cytoplasmic",     "darkgreen"),
        ("mcherry_total",      "mCherry total",       "red"),
        ("mcherry_nuclear",    "mCherry nuclear",     "tomato"),
        ("mcherry_cytoplasmic","mCherry cytoplasmic", "darkred"),
        ("h2b_nuclear",        "H2B nuclear",         "slateblue"),
        ("h2b_cytoplasmic",    "H2B cytoplasmic",     "mediumpurple"),
    ], "plot_raw.png", "Raw fluorescence (bg subtracted)", 2, 4, (16, 8))

    # --- Plot 2: H2B-normalised ---
    make_plot([
        ("gfp_nuclear_norm",        "GFP nuclear / H2B nuc",     "limegreen"),
        ("gfp_cytoplasmic_norm",    "GFP cyto / H2B cyto",       "darkgreen"),
        ("mcherry_nuclear_norm",    "mCherry nuclear / H2B nuc", "tomato"),
        ("mcherry_cytoplasmic_norm","mCherry cyto / H2B cyto",   "darkred"),
    ], "plot_normalised.png", "H2B-normalised fluorescence", 2, 2, (10, 8))

    # --- Plot 3: C/N ratio ---
    # Top row    : raw C/N  (cyto / nuc_H2B-norm)
    # Bottom row : t0-normalised C/N = C/N(t) / mean(C/N[0:BASELINE_FRAMES])
    #              baseline = 1.0,  >1 = activation,  <1 = inhibition

    fig_cn, axes_cn = plt.subplots(2, 2, figsize=(14, 10))
    fig_cn.suptitle("C/N ratio\n" + title_suffix, fontsize=12, fontweight="bold")

    plot_specs = [
        (axes_cn[0, 0], df_f["gfp_cn_ratio"],
         "GFP  -  raw C/N\n(cyto / nuc - norm)",
         "green", "C/N"),
        (axes_cn[0, 1], df_f["mcherry_cn_ratio"],
         "mCherry  -  raw C/N\n(cyto / nuc - norm)",
         "red", "C/N"),
        (axes_cn[1, 0], df_f["gfp_cn_ratio_t0norm"],
         "GFP  -  t0-normalised C/N\n"
         "(C/N(t) / mean C/N[first {} frames])".format(BASELINE_FRAMES),
         "green", "C/N "),
        (axes_cn[1, 1], df_f["mcherry_cn_ratio_t0norm"],
         "mCherry  -  t0-normalised C/N\n"
         "(C/N(t) / mean C/N[first {} frames])".format(BASELINE_FRAMES),
         "red", "C/N "),
    ]

    for ax, vals, label, colour, ylabel in plot_specs:
        ax.plot(frames, vals, color=colour, linewidth=2.0, marker="o", markersize=5)
        ax.axhline(1.0, color="gray", linewidth=1.0, linestyle="--", label="= 1")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Frame")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    cn_path = os.path.join(output_folder, "plot_cn_ratio_raw.png")
    plt.savefig(cn_path, dpi=150)
    print("  Plot saved --> {}".format(cn_path))
    plt.show()


# ----------------------------------------------------------------------
# FLUORESCENCE DIAGNOSTIC
# ----------------------------------------------------------------------

def visualise_fluo_with_masks(ch1c, cell_mask, nuc_mask, chosen_lbl,
                               bg1, NUC_EROSION_PX, frame_t, output_folder):
    import matplotlib.patches as mpatches

    cell_px = cell_mask == chosen_lbl
    nuc_eroded = binary_erosion(nuc_mask > 0, iterations=NUC_EROSION_PX) \
                 if NUC_EROSION_PX > 0 else nuc_mask > 0
    if not (cell_px & nuc_eroded).any():
        nuc_eroded = nuc_mask > 0
    nuc_px  = cell_px & nuc_eroded
    cyto_px = cell_px & ~nuc_eroded

    raw_bg   = bg1
    raw_nuc  = float(ch1c[nuc_px].mean())  if nuc_px.any()  else float("nan")
    raw_cyto = float(ch1c[cyto_px].mean()) if cyto_px.any() else float("nan")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f"Frame {frame_t}  |  bg={raw_bg:.0f}  "
        f"raw_nuc={raw_nuc:.0f}  raw_cyto={raw_cyto:.0f}  "
        f"bg-sub nuc={raw_nuc-raw_bg:.0f}  bg-sub cyto={raw_cyto-raw_bg:.0f}  "
        f"C/N={(raw_cyto-raw_bg)/(raw_nuc-raw_bg):.3f}",
        fontsize=10, fontweight="bold"
    )

    p2, p98 = np.percentile(ch1c, 2), np.percentile(ch1c, 98)
    disp_auto = np.clip((ch1c - p2) / (p98 - p2 + 1e-6), 0, 1)
    axes[0].imshow(disp_auto, cmap="gray")
    axes[0].set_title("Auto contrast\n(misleading — full image stretch)", fontsize=8)

    cell_vals = ch1c[cell_px]
    vmin = raw_bg
    vmax = float(cell_vals.max())
    disp_true = np.clip((ch1c.astype(np.float32) - vmin) / (vmax - vmin + 1e-6), 0, 1)
    axes[1].imshow(disp_true, cmap="gray")
    axes[1].set_title(f"True contrast\n(min=bg={raw_bg:.0f}, max={vmax:.0f})", fontsize=8)

    axes[2].imshow(disp_true, cmap="gray")
    overlay = np.zeros((*ch1c.shape, 4), dtype=np.float32)
    overlay[nuc_px]  = [0.0, 1.0, 1.0, 0.4]
    overlay[cyto_px] = [0.0, 1.0, 0.0, 0.25]
    axes[2].imshow(overlay)
    axes[2].set_title("True contrast + masks\ncyan=nucleus  green=cytoplasm", fontsize=8)

    cell_region = ch1c.copy().astype(np.float32)
    cell_region[~cell_px] = np.nan
    im = axes[3].imshow(cell_region, cmap="hot", vmin=raw_bg, vmax=vmax)
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    axes[3].set_title("Heatmap (raw counts)\nbackground level = colour floor", fontsize=8)

    patches = [
        mpatches.Patch(color="cyan",  alpha=0.6,
                       label=f"Nucleus  mean={raw_nuc:.0f}  (bg-sub={raw_nuc-raw_bg:.0f})"),
        mpatches.Patch(color="green", alpha=0.6,
                       label=f"Cytoplasm mean={raw_cyto:.0f}  (bg-sub={raw_cyto-raw_bg:.0f})"),
        mpatches.Patch(color="gray",  alpha=0.6,
                       label=f"Background median={raw_bg:.0f}"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    path = os.path.join(output_folder, f"fluo_diagnostic_t{frame_t:04d}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  Diagnostic saved --> {path}")
    plt.show()
    plt.close()


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # -- Load masks ----------------------------------------------------
    print("\n[1/4] Loading masks from {} ...".format(MASKS_FOLDER))
    records = load_masks_and_fluo()
    print("      {} frames loaded.".format(len(records)))

    # -- Pick cell from frame 0 ----------------------------------------
    print("\n[2/4] Pick cell to track ...")
    chosen_lbl, last_centroid = pick_cell(records[0]["ch0_crop"],
                                          records[0]["cell_mask"])
    last_area = get_cell_area(records[0]["cell_mask"], chosen_lbl)

    # -- Track + measure -----------------------------------------------
    print("\n[3/4] Tracking and measuring ...")
    print("      MAX_DIST = {} px".format(MAX_DIST))

    all_rows   = []
    frame_recs = []

    for i, rec in enumerate(records):
        t         = rec["t"]
        cell_mask = rec["cell_mask"]
        nuc_mask  = rec["nuc_mask"]

        if i == 0:
            lbl, centroid, area = chosen_lbl, last_centroid, last_area
            division_candidates = []
        else:
            lbl, centroid, area, division_candidates = find_cell(
                cell_mask, last_centroid, MAX_DIST, last_area)

        if division_candidates:
            print("  Frame {:04d} ... DIVISION DETECTED -- opening picker".format(t))
            lbl, centroid, area = pick_daughter(
                rec["ch0_crop"], cell_mask,
                lbl, centroid if centroid is not None else last_centroid,
                division_candidates, t)

        elif VERIFY_EVERY_N is not None and i % VERIFY_EVERY_N == 0:
            lbl, centroid = verify_and_reclick(
                rec["ch0_crop"], cell_mask, nuc_mask,
                lbl, centroid if centroid is not None else last_centroid, t)
            if lbl is not None:
                area = get_cell_area(cell_mask, lbl)

        if lbl is not None:
            last_centroid = centroid
            last_area     = area
            status = "found ({:.0f},{:.0f})  area={}px".format(*centroid, area)
        else:
            status = "LOST"

        print("  Frame {:04d} ... {}".format(t, status))

        ch1 = tifffile.imread(rec["ch1_path"])
        ch2 = tifffile.imread(rec["ch2_path"])
        ch3 = tifffile.imread(rec["ch3_path"])

        m = measure(cell_mask, nuc_mask, ch1, ch2, ch3,
                    lbl, rec["r1"], rec["r2"], rec["c1"], rec["c2"])

        if i == 21 and lbl is not None:
            ch1c_diag = ch1[rec["r1"]:rec["r2"], rec["c1"]:rec["c2"]]
            visualise_fluo_with_masks(
                ch1c_diag, cell_mask, nuc_mask, lbl,
                m["bg_gfp"], NUC_EROSION_PX, t, OUTPUT_FOLDER
            )

        all_rows.append({
            "frame":          t,
            "centroid_row":   round(centroid[0], 2) if lbl is not None else float("nan"),
            "centroid_col":   round(centroid[1], 2) if lbl is not None else float("nan"),
            **m,
        })

        frame_recs.append({
            "t":               t,
            "ch0_crop":        rec["ch0_crop"],
            "cell_mask":       cell_mask,
            "nuc_mask":        nuc_mask,
            "chosen_lbl":      lbl,
            "chosen_centroid": centroid if lbl is not None else last_centroid,
        })

    # -- Save CSV ------------------------------------------------------
    print("\n[4/4] Saving results ...")

    col_order = [
        "frame", "centroid_row", "centroid_col",
        "bg_gfp", "bg_mcherry", "bg_h2b", "bg_n_pixels",
        "gfp_total", "gfp_nuclear", "gfp_cytoplasmic",
        "mcherry_total", "mcherry_nuclear", "mcherry_cytoplasmic",
        "h2b_nuclear", "h2b_cytoplasmic",
        "gfp_nuclear_norm", "gfp_cytoplasmic_norm",
        "mcherry_nuclear_norm", "mcherry_cytoplasmic_norm",
        "gfp_cn_ratio", "mcherry_cn_ratio",
        "gfp_cn_ratio_norm", "mcherry_cn_ratio_norm",
    ]
    df = pd.DataFrame(all_rows, columns=col_order)

    # ------------------------------------------------------------------
    # t0 normalisation
    # Divide each C/N trace by the mean of its first BASELINE_FRAMES
    # valid (non-NaN) values.
    #   Result: baseline = 1.0
    #           > 1  →  activation above baseline
    #           < 1  →  inhibition below baseline
    # ------------------------------------------------------------------
    n_bl = BASELINE_FRAMES if BASELINE_FRAMES > 0 else 1

    def t0_norm(series, n_baseline):
        baseline = series.dropna().iloc[:n_baseline].mean()
        if math.isnan(baseline) or baseline == 0:
            return series * float("nan")
        return series / baseline

    df["gfp_cn_ratio_t0norm"]     = t0_norm(df["gfp_cn_ratio"],     n_bl)
    df["mcherry_cn_ratio_t0norm"] = t0_norm(df["mcherry_cn_ratio"], n_bl)

    csv_path = os.path.join(OUTPUT_FOLDER, "results.csv")
    df.to_csv(csv_path, index=False)
    n_found = df["gfp_total"].notna().sum()
    print("  CSV saved --> {}".format(csv_path))
    print("  Cell found in {}/{} frames.".format(n_found, len(df)))

    # -- Tracking overlay ----------------------------------------------
    if OVERLAY_MAX_FRAMES is not None and len(frame_recs) > OVERLAY_MAX_FRAMES:
        idx = np.round(
            np.linspace(0, len(frame_recs) - 1, OVERLAY_MAX_FRAMES)
        ).astype(int)
        overlay_recs = [frame_recs[i] for i in idx]
    else:
        overlay_recs = frame_recs
    save_tracking_overlay(overlay_recs,
                          os.path.join(OUTPUT_FOLDER, "tracking_overlay.png"))

    # -- Plots ---------------------------------------------------------
    save_plots(df, n_found, len(df), OUTPUT_FOLDER)
    print("\nDone.\n")


if __name__ == "__main__":
    main()