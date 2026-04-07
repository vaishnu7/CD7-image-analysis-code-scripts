"""
Cell Tracker - Interactive Crop + Cellpose Segmentation + Fluorescence Tracking
================================================================================
Channels per timepoint (each channel lives in its OWN subfolder):
  ch0  Phase contrast  (whole-cell segmentation)
  ch1  GFP             (fluorescence measurement)
  ch2  mCherry         (fluorescence measurement)
  ch3  Nucleus / H2B   (nuclear segmentation)

HOW TO USE:
  1. Edit the USER SETTINGS block below.
  2. Activate your cellpose3 venv:
       Windows : cellpose3\\Scripts\\activate
       Mac/Linux: source /path/to/cellpose3/bin/activate
  3. Run:
       python cell_tracker.py

Output (saved to OUTPUT_FOLDER):
  results_all_cells.csv         -- fluorescence for every tracked cell
  results_cell_N.csv            -- fluorescence for the chosen cell only
  fluorescence_cell_N.png       -- 6-panel fluorescence plot
  tracking_overlay.png          -- per-frame overlay showing the chosen cell
"""

import os
import re
import math
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("TkAgg")   # change to "Qt5Agg" or "Agg" if TkAgg is unavailable
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from cellpose import models
from scipy.spatial.distance import cdist


# ======================================================================
#                         USER SETTINGS
# ======================================================================

CH0_FOLDER = r"C:\Users\gz24763\OneDrive - University of Bristol\Documents\PC9_images\18_03_2026\A6\P12\New-02-Scene-13-P12-A06_ch0"
CH1_FOLDER = r"C:\Users\gz24763\OneDrive - University of Bristol\Documents\PC9_images\18_03_2026\A6\P12\New-02-Scene-13-P12-A06_ch1"
CH2_FOLDER = r"C:\Users\gz24763\OneDrive - University of Bristol\Documents\PC9_images\18_03_2026\A6\P12\New-02-Scene-13-P12-A06_ch2"
CH3_FOLDER = r"C:\Users\gz24763\OneDrive - University of Bristol\Documents\PC9_images\18_03_2026\A6\P12\New-02-Scene-13-P12-A06_ch3"

OUTPUT_FOLDER = r"C:\Users\gz24763\OneDrive - University of Bristol\Documents\PC9_images\18_03_2026\A6\P12\results"

DIAMETER_CELL = 200    # expected whole-cell diameter in pixels
DIAMETER_NUC  = 50    # expected nucleus diameter in pixels

MAX_DIST = 150         # max centroid displacement (px) allowed between frames

# How many frames to show in the tracking overlay grid
# (set to None to show ALL frames — can make a very large image if you have 78)
OVERLAY_MAX_FRAMES = 20

# ======================================================================


# ----------------------------------------------------------------------
# 1.  FILE DISCOVERY
# ----------------------------------------------------------------------

def _sorted_tifs(folder):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".tif"))
    return [os.path.join(folder, f) for f in files]


def _frame_index(path):
    m = re.search(r"_t(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def discover_timepoints():
    channel_files = {}
    for ch, folder in enumerate([CH0_FOLDER, CH1_FOLDER, CH2_FOLDER, CH3_FOLDER]):
        channel_files[ch] = {_frame_index(p): p for p in _sorted_tifs(folder)}

    common_frames = sorted(
        set(channel_files[0]) & set(channel_files[1])
        & set(channel_files[2]) & set(channel_files[3])
    )
    if not common_frames:
        raise FileNotFoundError(
            "No matching frames found. Check CH0-CH3 folders and _tNNNN naming.")

    return [
        {"t": t,
         0: channel_files[0][t], 1: channel_files[1][t],
         2: channel_files[2][t], 3: channel_files[3][t]}
        for t in common_frames
    ]


# ----------------------------------------------------------------------
# 2.  INTERACTIVE CROP
# ----------------------------------------------------------------------

def interactive_crop(img_path):
    img = tifffile.imread(img_path)
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    display = np.clip((img.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)

    state = {"clicks": [], "patch": None}

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        cx, cy = event.xdata, event.ydata
        if len(state["clicks"]) >= 2:
            state["clicks"] = []
            if state["patch"] is not None:
                state["patch"].remove()
                state["patch"] = None
            for artist in ax.lines:
                artist.remove()
        state["clicks"].append((cx, cy))
        ax.plot(cx, cy, "r+", markersize=14, markeredgewidth=2)
        if len(state["clicks"]) == 1:
            ax.set_title("Point 1 set: (col={}, row={})\nNow click BOTTOM-RIGHT".format(
                int(cx), int(cy)))
        elif len(state["clicks"]) == 2:
            x1, y1 = state["clicks"][0]
            x2, y2 = state["clicks"][1]
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            if state["patch"] is not None:
                state["patch"].remove()
            rect = mpatches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor="red", facecolor="none")
            state["patch"] = ax.add_patch(rect)
            ax.set_title("Selection confirmed! cols {}:{}, rows {}:{}\n"
                         "Close to continue  (click again to reselect)".format(
                             int(xmin), int(xmax), int(ymin), int(ymax)))
        fig.canvas.draw_idle()

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(display, cmap="gray")
    ax.set_title("Click the TOP-LEFT corner of your region of interest")
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()

    if len(state["clicks"]) < 2:
        raise RuntimeError("Fewer than 2 points clicked. Please re-run.")

    x1, y1 = state["clicks"][0]
    x2, y2 = state["clicks"][1]
    r1, r2 = int(min(y1, y2)), int(max(y1, y2))
    c1, c2 = int(min(x1, x2)), int(max(x1, x2))
    print("  Crop confirmed -- rows {}:{},  cols {}:{}".format(r1, r2, c1, c2))
    return r1, r2, c1, c2


# ----------------------------------------------------------------------
# 3.  HELPERS
# ----------------------------------------------------------------------

def norm(x):
    lo, hi = float(x.min()), float(x.max())
    return ((x.astype(np.float32) - lo) / (hi - lo + 1e-6)).clip(0, 1)


def segment(img_crop, model, diameter):
    masks, flows, _, _ = model.eval(
        img_crop,
        diameter=diameter,
        channels=[0, 0],
        flow_threshold=0.4,
        cellprob_threshold=0.0,
    )
    return masks


def get_centroids(mask):
    centroids = {}
    for lbl in np.unique(mask):
        if lbl == 0:
            continue
        rows, cols = np.where(mask == lbl)
        centroids[lbl] = (rows.mean(), cols.mean())
    return centroids


# ----------------------------------------------------------------------
# 4.  NEAREST-NEIGHBOUR TRACKING
# ----------------------------------------------------------------------

def link_centroids(prev_centroids, curr_centroids, max_dist):
    if not prev_centroids or not curr_centroids:
        return {}
    prev_ids = list(prev_centroids.keys())
    curr_ids = list(curr_centroids.keys())
    prev_pts = np.array([prev_centroids[i] for i in prev_ids])
    curr_pts = np.array([curr_centroids[i] for i in curr_ids])
    dists     = cdist(curr_pts, prev_pts)
    assignment = {}
    used_prev  = set()
    for ci in np.argsort(dists.min(axis=1)):
        pi = int(np.argmin(dists[ci]))
        if dists[ci, pi] <= max_dist and pi not in used_prev:
            assignment[curr_ids[ci]] = prev_ids[pi]
            used_prev.add(pi)
    return assignment


# ----------------------------------------------------------------------
# 5.  FLUORESCENCE MEASUREMENT
# ----------------------------------------------------------------------

def measure_fluorescence(cell_mask, nuc_mask, ch_gfp, ch_mcherry):
    bg_gfp = float(np.median(ch_gfp    [cell_mask == 0]))
    bg_mch = float(np.median(ch_mcherry[cell_mask == 0]))

    def mean_px(arr, mask):
        return float(arr[mask].mean()) if mask.any() else float("nan")

    results = []
    for lbl in np.unique(cell_mask):
        if lbl == 0:
            continue
        cell_px = cell_mask == lbl
        nuc_px  = cell_px & (nuc_mask > 0)
        cyto_px = cell_px & (nuc_mask == 0)

# ----------------------------------------------------------------------
# 4.  NEAREST-NEIGHBOUR TRACKING  (for all cells, global linking)
# ----------------------------------------------------------------------

def link_centroids(prev_centroids, curr_centroids, max_dist):
    """
    Greedy nearest-neighbour linking.
    prev_centroids : {track_id   : (r, c)}
    curr_centroids : {cell_label : (r, c)}
    Returns {curr_label: matched_track_id} for linked cells only.
    """
    if not prev_centroids or not curr_centroids:
        return {}
    prev_ids = list(prev_centroids.keys())
    curr_ids = list(curr_centroids.keys())
    prev_pts = np.array([prev_centroids[i] for i in prev_ids])
    curr_pts = np.array([curr_centroids[i] for i in curr_ids])
    dists     = cdist(curr_pts, prev_pts)
    assignment = {}
    used_prev  = set()
    for ci in np.argsort(dists.min(axis=1)):
        pi = int(np.argmin(dists[ci]))
        if dists[ci, pi] <= max_dist and pi not in used_prev:
            assignment[curr_ids[ci]] = prev_ids[pi]
            used_prev.add(pi)
    return assignment


def find_chosen_cell_in_frame(cell_mask, last_known_centroid, max_dist):
    """
    Given a new frame's mask and the last known (row, col) centroid of
    the chosen cell, find the closest cell label within max_dist.
    Returns (label, centroid) or (None, None) if no cell is close enough.
    """
    centroids = get_centroids(cell_mask)
    if not centroids or last_known_centroid is None:
        return None, None

    last_r, last_c = last_known_centroid
    best_lbl  = None
    best_dist = float("inf")

    for lbl, (r, c) in centroids.items():
        d = math.sqrt((r - last_r) ** 2 + (c - last_c) ** 2)
        if d < best_dist:
            best_dist = d
            best_lbl  = lbl

    if best_dist <= max_dist:
        return best_lbl, centroids[best_lbl]
    return None, None


# ----------------------------------------------------------------------
# 5.  FLUORESCENCE MEASUREMENT
# ----------------------------------------------------------------------

def measure_fluorescence(cell_mask, nuc_mask, ch_gfp, ch_mcherry):
    bg_gfp = float(np.median(ch_gfp    [cell_mask == 0]))
    bg_mch = float(np.median(ch_mcherry[cell_mask == 0]))

    def mean_px(arr, mask):
        return float(arr[mask].mean()) if mask.any() else float("nan")

    results = []
    for lbl in np.unique(cell_mask):
        if lbl == 0:
            continue
        cell_px = cell_mask == lbl
        nuc_px  = cell_px & (nuc_mask > 0)
        cyto_px = cell_px & (nuc_mask == 0)
        results.append({
            "cell_label":          lbl,
            "gfp_total":           mean_px(ch_gfp,     cell_px) - bg_gfp,
            "gfp_nuclear":         mean_px(ch_gfp,     nuc_px)  - bg_gfp,
            "gfp_cytoplasmic":     mean_px(ch_gfp,     cyto_px) - bg_gfp,
            "mcherry_total":       mean_px(ch_mcherry, cell_px) - bg_mch,
            "mcherry_nuclear":     mean_px(ch_mcherry, nuc_px)  - bg_mch,
            "mcherry_cytoplasmic": mean_px(ch_mcherry, cyto_px) - bg_mch,
        })
    return results


# ----------------------------------------------------------------------
# 6.  CELL PICKER
# ----------------------------------------------------------------------

def pick_cell(ch0_crop, cell_mask):
    lo, hi = np.percentile(ch0_crop, 1), np.percentile(ch0_crop, 99)
    display = np.clip(
        (ch0_crop.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)

    n_labels = int(cell_mask.max())
    overlay  = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
    colours  = cm.tab20(np.linspace(0, 1, max(n_labels, 1)))
    for i, lbl in enumerate(range(1, n_labels + 1)):
        px = cell_mask == lbl
        overlay[px, :3] = colours[i % len(colours)][:3]
        overlay[px,  3] = 0.35

    centroids = get_centroids(cell_mask)
    chosen    = {"label": None}

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        col_i = int(np.clip(round(event.xdata), 0, cell_mask.shape[1] - 1))
        row_i = int(np.clip(round(event.ydata), 0, cell_mask.shape[0] - 1))
        lbl   = int(cell_mask[row_i, col_i])
        if lbl == 0:
            print("  Clicked background -- please click inside a coloured cell.")
            return
        chosen["label"] = lbl
        for sc in ax.collections:
            sc.remove()
        yx = np.argwhere(cell_mask == lbl)
        ax.scatter(yx[:, 1], yx[:, 0], s=1, c="lime", alpha=0.5, linewidths=0)
        ax.set_title(
            "Cell {} selected!  Close this window to start tracking.".format(lbl),
            color="lime", fontsize=12, fontweight="bold")
        fig.canvas.draw_idle()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(display, cmap="gray")
    ax.imshow(overlay)
    for lbl, (r, c) in centroids.items():
        ax.text(c, r, str(lbl), color="white", fontsize=7,
                ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5, lw=0))
    ax.set_title("Click on the cell you want to track,\nthen close this window to continue")
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()

    if chosen["label"] is None:
        raise RuntimeError("No cell was selected. Please re-run and click a cell.")
    print("  Cell {} selected for tracking.".format(chosen["label"]))
    return chosen["label"]


# ----------------------------------------------------------------------
# 7.  TRACKING OVERLAY PLOT
# ----------------------------------------------------------------------

def save_tracking_overlay(frame_records, output_path):
    """
    Each panel shows one frame with:
      - Grayscale phase contrast background
      - All cells semi-transparent coloured
      - Chosen cell = solid lime green (cell body) + cyan (nucleus)
      - White border + centroid crosshair on chosen cell
      - Green title if found, red if lost
    frame_records: list of dicts with keys:
        t, ch0, cell_mask, nuc_mask, chosen_lbl (int or None), chosen_centroid
    """
    n_frames = len(frame_records)
    ncols    = min(5, n_frames)
    nrows    = math.ceil(n_frames / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.2, nrows * 3.2))
    fig.suptitle("Tracking overlay  --  chosen cell", fontsize=13, fontweight="bold")

    axes_flat = np.array(axes).flatten() if n_frames > 1 else [axes]

    for idx, rec in enumerate(frame_records):
        ax         = axes_flat[idx]
        ch0        = rec["ch0"]
        cell_mask  = rec["cell_mask"]
        nuc_mask   = rec["nuc_mask"]
        chosen_lbl = rec["chosen_lbl"]

        lo, hi = np.percentile(ch0, 1), np.percentile(ch0, 99)
        display = np.clip(
            (ch0.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
        ax.imshow(display, cmap="gray", interpolation="nearest")

        # semi-transparent overlay for all cells
        n_labels = int(cell_mask.max())
        if n_labels > 0:
            overlay = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
            colours = cm.tab20(np.linspace(0, 1, max(n_labels, 1)))
            for i, lbl in enumerate(range(1, n_labels + 1)):
                px = cell_mask == lbl
                overlay[px, :3] = colours[i % len(colours)][:3]
                overlay[px,  3] = 0.25
            ax.imshow(overlay, interpolation="nearest")

        if chosen_lbl is not None:
            cell_px = cell_mask == chosen_lbl

            # lime green for chosen cell body
            cho = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
            cho[cell_px] = [0.0, 1.0, 0.2, 0.55]
            ax.imshow(cho, interpolation="nearest")

            # cyan for nucleus
            nuc_px = cell_px & (nuc_mask > 0)
            if nuc_px.any():
                nuo = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
                nuo[nuc_px] = [0.0, 1.0, 1.0, 0.7]
                ax.imshow(nuo, interpolation="nearest")

            # white border
            from scipy.ndimage import binary_erosion
            border = cell_px & ~binary_erosion(cell_px)
            yx = np.argwhere(border)
            if len(yx):
                ax.scatter(yx[:, 1], yx[:, 0], s=0.3,
                           c="white", linewidths=0, alpha=0.9)

            # centroid crosshair
            cr, cc = rec["chosen_centroid"]
            ax.plot(cc, cr, "+", color="white", markersize=8, markeredgewidth=1.5)

            ax.set_title("Frame {:04d}  [found]".format(rec["t"]),
                         fontsize=7, color="lime")
        else:
            ax.set_title("Frame {:04d}  [lost]".format(rec["t"]),
                         fontsize=7, color="red")

        ax.axis("off")

    for idx in range(n_frames, len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print("  Tracking overlay saved --> {}".format(output_path))
    plt.show()


# ----------------------------------------------------------------------
# 8.  MAIN PIPELINE
# ----------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # -- Discover files ------------------------------------------------
    print("\n[1/6] Discovering timepoints ...")
    timepoints = discover_timepoints()
    print("      Found {} complete timepoints.".format(len(timepoints)))

    # -- Interactive crop ----------------------------------------------
    print("\n[2/6] Select crop region ...")
    print("      INSTRUCTIONS: click TOP-LEFT corner, then BOTTOM-RIGHT corner.")
    print("      Close the window when happy with the red rectangle.")
    r1, r2, c1, c2 = interactive_crop(timepoints[0][0])

    def crop(img):
        return img[r1:r2, c1:c2]

    # -- Load Cellpose models ------------------------------------------
    print("\n[3/6] Loading Cellpose models ...")
    cell_model = models.Cellpose(gpu=False, model_type="cyto3")
    nuc_model  = models.Cellpose(gpu=False, model_type="nuclei")

    # -- Segment frame 0 and pick cell ---------------------------------
    print("\n[4/6] Segmenting first frame to identify cells ...")
    ch0_f0  = crop(tifffile.imread(timepoints[0][0]))
    mask_f0 = segment(norm(ch0_f0), cell_model, DIAMETER_CELL)
    print("      {} cells detected in frame 0.".format(int(mask_f0.max())))

    print("\n[5/6] Pick the cell to track ...")
    print("      Click on the cell you want, then close the window.")
    chosen_label_f0 = pick_cell(ch0_f0, mask_f0)

    # get initial centroid of the chosen cell
    centroids_f0    = get_centroids(mask_f0)
    last_centroid   = centroids_f0[chosen_label_f0]   # (row, col)
    print("      Initial centroid: row={:.1f}, col={:.1f}".format(*last_centroid))

    # -- Process all timepoints ----------------------------------------
    print("\n[6/6] Segmenting and measuring all frames ...")
    print("      Strategy: each frame, find the cell closest to the")
    print("      last known centroid within MAX_DIST={} px.".format(MAX_DIST))

    all_rows     = []
    frame_records = []

    for i, tp in enumerate(timepoints):
        t = tp["t"]
        print("  Frame {:04d} ...".format(t), end=" ", flush=True)

        ch0 = crop(tifffile.imread(tp[0]))
        ch1 = crop(tifffile.imread(tp[1]))
        ch2 = crop(tifffile.imread(tp[2]))
        ch3 = crop(tifffile.imread(tp[3]))

        cell_mask = mask_f0 if i == 0 else segment(norm(ch0), cell_model, DIAMETER_CELL)
        nuc_mask  = segment(norm(ch3), nuc_model, DIAMETER_NUC)

        # --- find chosen cell in this frame ---
        if i == 0:
            chosen_lbl      = chosen_label_f0
            chosen_centroid = last_centroid
        else:
            chosen_lbl, chosen_centroid = find_chosen_cell_in_frame(
                cell_mask, last_centroid, MAX_DIST)

        if chosen_lbl is not None:
            # update last known centroid for next frame
            last_centroid = chosen_centroid

            # pixel masks
            cell_px = cell_mask == chosen_lbl
            nuc_px  = cell_px & (nuc_mask > 0)
            cyto_px = cell_px & (nuc_mask == 0)

            # background = median of pixels outside ALL cells
            bg_gfp = float(np.median(ch1[cell_mask == 0]))
            bg_mch = float(np.median(ch2[cell_mask == 0]))
            bg_h2b = float(np.median(ch3[cell_mask == 0]))

            def mean_px(arr, mask):
                return float(arr[mask].mean()) if mask.any() else float("nan")

            # raw bg-subtracted values
            gfp_nuc  = mean_px(ch1, nuc_px)  - bg_gfp
            gfp_cyto = mean_px(ch1, cyto_px) - bg_gfp
            mch_nuc  = mean_px(ch2, nuc_px)  - bg_mch
            mch_cyto = mean_px(ch2, cyto_px) - bg_mch
            h2b_nuc  = mean_px(ch3, nuc_px)  - bg_h2b
            h2b_cyto = mean_px(ch3, cyto_px) - bg_h2b

            # safe division — returns NaN if denominator is 0 or NaN
            def safe_div(a, b):
                if math.isnan(a) or math.isnan(b) or b == 0:
                    return float("nan")
                return a / b

            all_rows.append({
                "frame":                    t,
                "centroid_row":             round(chosen_centroid[0], 2),
                "centroid_col":             round(chosen_centroid[1], 2),
                # raw bg-subtracted
                "gfp_total":                mean_px(ch1, cell_px) - bg_gfp,
                "gfp_nuclear":              gfp_nuc,
                "gfp_cytoplasmic":          gfp_cyto,
                "mcherry_total":            mean_px(ch2, cell_px) - bg_mch,
                "mcherry_nuclear":          mch_nuc,
                "mcherry_cytoplasmic":      mch_cyto,
                "h2b_nuclear":              h2b_nuc,
                "h2b_cytoplasmic":          h2b_cyto,
                # H2B-normalised
                "gfp_nuclear_norm":         safe_div(gfp_nuc,  h2b_nuc),
                "gfp_cytoplasmic_norm":     safe_div(gfp_cyto, h2b_cyto),
                "mcherry_nuclear_norm":     safe_div(mch_nuc,  h2b_nuc),
                "mcherry_cytoplasmic_norm": safe_div(mch_cyto, h2b_cyto),
            })
            print("found at ({:.0f},{:.0f})  dist={:.1f}px".format(
                chosen_centroid[0], chosen_centroid[1],
                math.sqrt((chosen_centroid[0]-last_centroid[0])**2 +
                          (chosen_centroid[1]-last_centroid[1])**2)
                if i > 0 else 0.0))
        else:
            # cell lost — record NaN row, do NOT update centroid
            nan = float("nan")
            all_rows.append({
                "frame":                    t,
                "centroid_row":             nan,
                "centroid_col":             nan,
                "gfp_total":                nan,
                "gfp_nuclear":              nan,
                "gfp_cytoplasmic":          nan,
                "mcherry_total":            nan,
                "mcherry_nuclear":          nan,
                "mcherry_cytoplasmic":      nan,
                "h2b_nuclear":              nan,
                "h2b_cytoplasmic":          nan,
                "gfp_nuclear_norm":         nan,
                "gfp_cytoplasmic_norm":     nan,
                "mcherry_nuclear_norm":     nan,
                "mcherry_cytoplasmic_norm": nan,
            })
            print("LOST  (last centroid: {:.0f},{:.0f})".format(*last_centroid))

        frame_records.append({
            "t":               t,
            "ch0":             ch0,
            "cell_mask":       cell_mask,
            "nuc_mask":        nuc_mask,
            "chosen_lbl":      chosen_lbl,
            "chosen_centroid": chosen_centroid if chosen_lbl is not None else last_centroid,
        })

    # -- Save CSV ------------------------------------------------------
    print("\nSaving results ...")
    col_order = [
        "frame", "centroid_row", "centroid_col",
        "gfp_total", "gfp_nuclear", "gfp_cytoplasmic",
        "mcherry_total", "mcherry_nuclear", "mcherry_cytoplasmic",
        "h2b_nuclear", "h2b_cytoplasmic",
        "gfp_nuclear_norm", "gfp_cytoplasmic_norm",
        "mcherry_nuclear_norm", "mcherry_cytoplasmic_norm",
    ]
    df = pd.DataFrame(all_rows, columns=col_order)
    csv_path = os.path.join(OUTPUT_FOLDER, "results_chosen_cell.csv")
    df.to_csv(csv_path, index=False)
    n_found = df["gfp_total"].notna().sum()
    print("  CSV saved --> {}".format(csv_path))
    print("  Cell found in {}/{} frames.\n".format(n_found, len(df)))

    # -- Tracking overlay ----------------------------------------------
    if OVERLAY_MAX_FRAMES is not None and len(frame_records) > OVERLAY_MAX_FRAMES:
        indices = np.round(
            np.linspace(0, len(frame_records) - 1, OVERLAY_MAX_FRAMES)
        ).astype(int)
        overlay_records = [frame_records[i] for i in indices]
        print("  Showing {} evenly-spaced frames in overlay.".format(OVERLAY_MAX_FRAMES))
    else:
        overlay_records = frame_records

    overlay_path = os.path.join(OUTPUT_FOLDER, "tracking_overlay.png")
    save_tracking_overlay(overlay_records, overlay_path)

    # -- Fluorescence plot  (2 rows: raw | H2B-normalised) -------------
    plot_path = os.path.join(OUTPUT_FOLDER, "fluorescence_chosen_cell.png")
    df_found  = df.dropna(subset=["gfp_total"])

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle(
        "Chosen cell  --  fluorescence over time  "
        "({}/{} frames found)".format(n_found, len(df)),
        fontsize=13, fontweight="bold")

    plot_specs = [
        # row 0: raw GFP
        (axes[0, 0], "gfp_total",                "GFP total (raw)",              "green"),
        (axes[0, 1], "gfp_nuclear",               "GFP nuclear (raw)",            "limegreen"),
        (axes[0, 2], "gfp_cytoplasmic",           "GFP cytoplasmic (raw)",        "darkgreen"),
        # row 1: raw mCherry + H2B
        (axes[1, 0], "mcherry_total",             "mCherry total (raw)",          "red"),
        (axes[1, 1], "mcherry_nuclear",           "mCherry nuclear (raw)",        "tomato"),
        (axes[1, 2], "mcherry_cytoplasmic",       "mCherry cytoplasmic (raw)",    "darkred"),
        (axes[1, 3], "h2b_nuclear",               "H2B nuclear (raw)",            "slateblue"),
        # row 2: H2B-normalised
        (axes[2, 0], "gfp_nuclear_norm",          "GFP nuclear / H2B nuclear",    "limegreen"),
        (axes[2, 1], "gfp_cytoplasmic_norm",      "GFP cyto / H2B cyto",         "darkgreen"),
        (axes[2, 2], "mcherry_nuclear_norm",      "mCherry nuclear / H2B nuclear","tomato"),
        (axes[2, 3], "mcherry_cytoplasmic_norm",  "mCherry cyto / H2B cyto",     "darkred"),
    ]

    # hide unused panel (axes[0,3])
    axes[0, 3].axis("off")

    for ax, col, title, colour in plot_specs:
        ax.plot(df_found["frame"], df_found[col], color=colour,
                linewidth=1.8, marker="o", markersize=4)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Intensity" if "norm" not in col else "Ratio (normalised)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print("  Fluorescence plot saved --> {}\n".format(plot_path))
    plt.show()


if __name__ == "__main__":
    main()