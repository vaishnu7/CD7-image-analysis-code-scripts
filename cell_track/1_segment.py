"""
STEP 1 - Crop + Tune Parameters + Segment
==========================================
Workflow:
  1. Crop your region of interest (two-click)
  2. TUNE MODE: test Cellpose parameters interactively on any frame
     until segmentation looks correct (especially for mitotic/elongated cells)
  3. Full segmentation of all frames using the tuned parameters
  4. Saves masks + preview for use by step2

Run this ONCE per experiment.  Re-run if you want a different crop
or different segmentation parameters.

Output (saved to MASKS_FOLDER):
  crop_coords.npy           -- (r1, r2, c1, c2) crop box
  cellpose_params.npy       -- the final parameters used
  cell_mask_t{N}.npy        -- whole-cell label mask per frame
  nuc_mask_t{N}.npy         -- nucleus label mask per frame
  ch0_crop_t{N}.npy         -- cropped phase-contrast image per frame
  segmentation_preview.png  -- overlay of masks on frame 0

"""

import os
import re
import numpy as np
import tifffile
import matplotlib
matplotlib.use("TkAgg")   # change to "Qt5Agg" if needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from cellpose import models


# ======================================================================
#                         USER SETTINGS
# ======================================================================

CH0_FOLDER = r"C:\Users\path to your file"
CH3_FOLDER = r"C:\Users\path to your file"

MASKS_FOLDER = r"C:\Users\path to your output"

# --- Starting parameters (will be tuned interactively) ----------------
DIAMETER_CELL     = 195   # expected whole-cell diameter in pixels
DIAMETER_NUC      = 40    # expected nucleus diameter in pixels

# Cellpose thresholds:
#   flow_threshold    : 0.0 - 1.0  higher = more/larger masks (try 0.4 - 0.9)
#   cellprob_threshold: -6.0 - 6.0 lower  = more permissive   (try 0.0 down to -3.0)
FLOW_THRESHOLD     = 0.4
CELLPROB_THRESHOLD = 0.0

# --- Run control ------------------------------------------------------
# How many frames to segment (None = ALL frames)
MAX_FRAMES = 78

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


def discover_timepoints():
    ch0_files = {_frame_index(p): p for p in _sorted_tifs(CH0_FOLDER)}
    ch3_files = {_frame_index(p): p for p in _sorted_tifs(CH3_FOLDER)}
    common = sorted(set(ch0_files) & set(ch3_files))
    if not common:
        raise FileNotFoundError("No matching frames found in CH0 and CH3 folders.")
    return [{"t": t, 0: ch0_files[t], 3: ch3_files[t]} for t in common]


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

def norm(x):
    lo, hi = float(x.min()), float(x.max())
    return ((x.astype(np.float32) - lo) / (hi - lo + 1e-6)).clip(0, 1)


def segment(img, model, diameter, flow_thresh, cellprob_thresh):
    masks, _, _, _ = model.eval(
        img,
        diameter=diameter,
        channels=[0, 0],
        flow_threshold=flow_thresh,
        cellprob_threshold=cellprob_thresh,
    )
    return masks


def show_masks(ch0_crop, cell_mask, nuc_mask, title):
    """Show segmentation result as a two-panel overlay."""
    lo, hi = np.percentile(ch0_crop, 1), np.percentile(ch0_crop, 99)
    disp = np.clip((ch0_crop.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)

    # cell overlay
    n_lbl = int(cell_mask.max())
    cell_ov = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
    colours  = cm.tab20(np.linspace(0, 1, max(n_lbl, 1)))
    for i, lbl in enumerate(range(1, n_lbl + 1)):
        px = cell_mask == lbl
        cell_ov[px, :3] = colours[i % len(colours)][:3]
        cell_ov[px,  3] = 0.45

    # nucleus overlay
    nuc_ov = np.zeros((*nuc_mask.shape, 4), dtype=np.float32)
    nuc_ov[nuc_mask > 0] = [0.0, 1.0, 1.0, 0.5]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=11, fontweight="bold")
    for ax, ov, subtitle in zip(axes,
                                [cell_ov, nuc_ov],
                                ["Whole-cell masks  ({} cells)".format(n_lbl),
                                 "Nucleus masks  ({} nuclei)".format(int(nuc_mask.max()))]):
        ax.imshow(disp, cmap="gray")
        ax.imshow(ov)
        ax.set_title(subtitle)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# INTERACTIVE CROP
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
            if state["patch"]:
                state["patch"].remove()
                state["patch"] = None
            for a in ax.lines:
                a.remove()
        state["clicks"].append((cx, cy))
        ax.plot(cx, cy, "r+", markersize=14, markeredgewidth=2)
        if len(state["clicks"]) == 1:
            ax.set_title("Point 1: (col={}, row={})\nNow click BOTTOM-RIGHT".format(
                int(cx), int(cy)))
        elif len(state["clicks"]) == 2:
            x1, y1 = state["clicks"][0]
            x2, y2 = state["clicks"][1]
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            if state["patch"]:
                state["patch"].remove()
            state["patch"] = ax.add_patch(mpatches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor="red", facecolor="none"))
            ax.set_title("Confirmed! rows {}:{} cols {}:{}\n"
                         "Close to continue".format(
                             int(ymin), int(ymax), int(xmin), int(xmax)))
        fig.canvas.draw_idle()

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(display, cmap="gray")
    ax.set_title("Click TOP-LEFT corner of your region of interest")
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()

    if len(state["clicks"]) < 2:
        raise RuntimeError("Fewer than 2 points selected. Please re-run.")
    x1, y1 = state["clicks"][0]
    x2, y2 = state["clicks"][1]
    r1, r2 = int(min(y1, y2)), int(max(y1, y2))
    c1, c2 = int(min(x1, x2)), int(max(x1, x2))
    print("  Crop: rows {}:{}, cols {}:{}".format(r1, r2, c1, c2))
    return r1, r2, c1, c2


# ----------------------------------------------------------------------
# PARAMETER TUNING
# ----------------------------------------------------------------------

def tune_parameters(timepoints, crop_fn, cell_model, nuc_model,
                    init_flow, init_cellprob, init_diam_cell, init_diam_nuc):
    """
    Interactive loop: lets user test Cellpose parameters on any frame
    until satisfied.  Returns final (flow_thresh, cellprob_thresh,
    diam_cell, diam_nuc).
    """
    flow_thresh    = init_flow
    cellprob_thresh = init_cellprob
    diam_cell      = init_diam_cell
    diam_nuc       = init_diam_nuc

    print("\n" + "="*60)
    print("  PARAMETER TUNING MODE")
    print("  Current values:")
    print("    flow_threshold     = {}  (range 0.0 - 1.0)".format(flow_thresh))
    print("    cellprob_threshold = {}  (range -6.0 - 6.0)".format(cellprob_thresh))
    print("    diameter_cell      = {} px".format(diam_cell))
    print("    diameter_nuc       = {} px".format(diam_nuc))
    print("  Available frames:  0  to  {}".format(len(timepoints) - 1))
    print("="*60)

    while True:
        print("\nOptions:")
        print("  t <frame_number>   -- test on that frame  (e.g. t 6)")
        print("  f <value>          -- set flow_threshold  (e.g. f 0.7)")
        print("  p <value>          -- set cellprob_threshold (e.g. p -2.0)")
        print("  dc <value>         -- set diameter_cell  (e.g. dc 60)")
        print("  dn <value>         -- set diameter_nuc   (e.g. dn 40)")
        print("  done               -- accept current parameters and segment all frames")

        cmd = input("\n  > ").strip().lower()

        if cmd == "done":
            break

        parts = cmd.split()
        if len(parts) < 2:
            print("  Unrecognised command.")
            continue

        key, val_str = parts[0], parts[1]

        try:
            val = float(val_str)
        except ValueError:
            print("  Invalid value: {}".format(val_str))
            continue

        if key == "f":
            flow_thresh = val
            print("  flow_threshold set to {}".format(flow_thresh))

        elif key == "p":
            cellprob_thresh = val
            print("  cellprob_threshold set to {}".format(cellprob_thresh))

        elif key == "dc":
            diam_cell = val
            print("  diameter_cell set to {}".format(diam_cell))

        elif key == "dn":
            diam_nuc = val
            print("  diameter_nuc set to {}".format(diam_nuc))

        elif key == "t":
            frame_i = int(val)
            if frame_i < 0 or frame_i >= len(timepoints):
                print("  Frame index out of range (0 - {})".format(
                    len(timepoints) - 1))
                continue

            tp = timepoints[frame_i]
            t  = tp["t"]
            print("  Segmenting frame {:04d} with current parameters ...".format(t),
                  end=" ", flush=True)

            ch0_crop = crop_fn(tifffile.imread(tp[0]))
            ch3_crop = crop_fn(tifffile.imread(tp[3]))

            cell_mask = segment(norm(ch0_crop), cell_model,
                                diam_cell, flow_thresh, cellprob_thresh)
            nuc_mask  = segment(norm(ch3_crop), nuc_model,
                                diam_nuc,  flow_thresh, cellprob_thresh)

            print("{} cells,  {} nuclei".format(
                int(cell_mask.max()), int(nuc_mask.max())))

            show_masks(
                ch0_crop, cell_mask, nuc_mask,
                "Frame {:04d}  |  flow={:.2f}  cellprob={:.1f}  "
                "diam_cell={}  diam_nuc={}".format(
                    t, flow_thresh, cellprob_thresh, diam_cell, diam_nuc)
            )
        else:
            print("  Unrecognised command.")

    print("\n  Final parameters:")
    print("    flow_threshold     = {}".format(flow_thresh))
    print("    cellprob_threshold = {}".format(cellprob_thresh))
    print("    diameter_cell      = {}".format(diam_cell))
    print("    diameter_nuc       = {}".format(diam_nuc))
    return flow_thresh, cellprob_thresh, diam_cell, diam_nuc


# ----------------------------------------------------------------------
# SEGMENTATION PREVIEW  (saved to file)
# ----------------------------------------------------------------------

def save_preview(ch0_crop, cell_mask, nuc_mask, output_path):
    lo, hi = np.percentile(ch0_crop, 1), np.percentile(ch0_crop, 99)
    display = np.clip((ch0_crop.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)

    n_labels = int(cell_mask.max())
    cell_overlay = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
    colours = cm.tab20(np.linspace(0, 1, max(n_labels, 1)))
    for i, lbl in enumerate(range(1, n_labels + 1)):
        px = cell_mask == lbl
        cell_overlay[px, :3] = colours[i % len(colours)][:3]
        cell_overlay[px,  3] = 0.4

    nuc_overlay = np.zeros((*nuc_mask.shape, 4), dtype=np.float32)
    nuc_overlay[nuc_mask > 0] = [0.0, 1.0, 1.0, 0.5]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Segmentation preview -- frame 0", fontsize=12, fontweight="bold")
    for ax, ov, title in zip(axes,
                             [cell_overlay, nuc_overlay],
                             ["Whole-cell masks (ch0)", "Nucleus masks (ch3)"]):
        ax.imshow(display, cmap="gray")
        ax.imshow(ov)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print("  Preview saved --> {}".format(output_path))
    plt.show()


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    os.makedirs(MASKS_FOLDER, exist_ok=True)

    # -- Discover timepoints -------------------------------------------
    print("\n[1/5] Discovering timepoints ...")
    timepoints = discover_timepoints()
    if MAX_FRAMES is not None:
        timepoints = timepoints[:MAX_FRAMES]
    print("      {} timepoints to process.".format(len(timepoints)))

    # -- Crop ----------------------------------------------------------
    print("\n[2/5] Select crop region ...")
    r1, r2, c1, c2 = interactive_crop(timepoints[0][0])
    np.save(os.path.join(MASKS_FOLDER, "crop_coords.npy"),
            np.array([r1, r2, c1, c2]))
    print("      Crop coordinates saved.")

    def crop(img):
        return img[r1:r2, c1:c2]

    # -- Load models ---------------------------------------------------
    print("\n[3/5] Loading Cellpose models ...")
    cell_model = models.Cellpose(gpu=True, model_type="cyto3")
    nuc_model  = models.Cellpose(gpu=True, model_type="cyto3")

    # -- Tune parameters -----------------------------------------------
    print("\n[4/5] Parameter tuning ...")
    print("      TIP: start by testing frame 0 (type: t 0)")
    print("      Then test a difficult frame like a mitotic one (e.g.: t 6)")
    print("      Adjust parameters until both frames look good, then type: done")

    flow_thresh, cellprob_thresh, diam_cell, diam_nuc = tune_parameters(
        timepoints, crop, cell_model, nuc_model,
        FLOW_THRESHOLD, CELLPROB_THRESHOLD, DIAMETER_CELL, DIAMETER_NUC
    )

    # save final parameters so step2 can report them
    np.save(os.path.join(MASKS_FOLDER, "cellpose_params.npy"),
            np.array([flow_thresh, cellprob_thresh, diam_cell, diam_nuc]))

    # -- Full segmentation ---------------------------------------------
    print("\n[5/5] Segmenting all {} frames ...".format(len(timepoints)))
    for i, tp in enumerate(timepoints):
        t = tp["t"]
        print("  Frame {:04d} ...".format(t), end=" ", flush=True)

        ch0_crop = crop(tifffile.imread(tp[0]))
        ch3_crop = crop(tifffile.imread(tp[3]))

        cell_mask = segment(norm(ch0_crop), cell_model,
                            diam_cell, flow_thresh, cellprob_thresh)
        nuc_mask  = segment(norm(ch3_crop), nuc_model,
                            diam_nuc,  flow_thresh, cellprob_thresh)

        np.save(os.path.join(MASKS_FOLDER,
                "cell_mask_t{:04d}.npy".format(t)), cell_mask)
        np.save(os.path.join(MASKS_FOLDER,
                "nuc_mask_t{:04d}.npy".format(t)),  nuc_mask)
        np.save(os.path.join(MASKS_FOLDER,
                "ch0_crop_t{:04d}.npy".format(t)),  ch0_crop)

        print("{} cells,  {} nuclei".format(
            int(cell_mask.max()), int(nuc_mask.max())))

        if i == 0:
            save_preview(ch0_crop, cell_mask, nuc_mask,
                         os.path.join(MASKS_FOLDER, "segmentation_preview.png"))

    print("\nDone. Masks saved to: {}".format(MASKS_FOLDER))
    print("Parameters used:")
    print("  flow_threshold     = {}".format(flow_thresh))
    print("  cellprob_threshold = {}".format(cellprob_thresh))
    print("  diameter_cell      = {}".format(diam_cell))
    print("  diameter_nuc       = {}".format(diam_nuc))
    print("\nNow run step2_track_and_measure.py\n")


if __name__ == "__main__":
    main()
