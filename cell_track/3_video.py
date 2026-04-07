"""
STEP 3 - Generate Tracking Video
==================================
Creates a side-by-side video for each frame showing:
  LEFT  : Phase contrast image with cell mask overlay (lime green = chosen cell,
           cyan = nucleus, white border + crosshair)
  RIGHT : Live C/N ratio plots for GFP (AKT-KTR) and mCherry (ERK-KTR),
           with a moving red dot showing the current frame

Prerequisites:
  - Run step1_segment.py  (generates masks folder)
  - Run step2_track_and_measure.py  (generates results.csv)
  - ffmpeg must be installed:
      Windows : https://ffmpeg.org/download.html  (add to PATH)
      Mac     : brew install ffmpeg
      Linux   : sudo apt install ffmpeg

Output (saved to OUTPUT_FOLDER):
  tracking_video.mp4   -- the final video

HOW TO USE:
  1. Edit USER SETTINGS below
  2. Activate your cellpose3 venv (or any env with matplotlib, numpy, pandas, tifffile)
  3. Run:
       python step3_make_video.py
"""

import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for video rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import cv2
from scipy.ndimage import binary_erosion


# ======================================================================
#                         USER SETTINGS
# ======================================================================

# Folder produced by step1_segment.py
MASKS_FOLDER  = r"C:\Users\gz24763\OneDrive - University of Bristol\Documents\PC9_images\18_03_2026\B2\P17\masks"

# results.csv produced by step2_track_and_measure.py
RESULTS_CSV   = r"C:\Users\gz24763\OneDrive - University of Bristol\Documents\PC9_images\18_03_2026\B2\P17\results\results.csv"

# Where to save the video
OUTPUT_FOLDER = r"C:\Users\gz24763\OneDrive - University of Bristol\Documents\PC9_images\18_03_2026\B2\P17\results"

# Video settings
FPS           = 5       # frames per second in the output video (5 = slow, 10 = faster)
DPI           = 150     # resolution — lower = smaller file (100-150 is fine)
VIDEO_NAME    = "tracking_video.mp4"  # change to .avi if mp4 fails

# Time between frames in minutes (for x-axis label)
FRAME_INTERVAL_MIN = 16

# ======================================================================


# ----------------------------------------------------------------------
# FILE LOADING
# ----------------------------------------------------------------------

def _frame_index(path):
    m = re.search(r"_t(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def load_data():
    """
    Loads:
      - results.csv  (fluorescence + C/N per frame)
      - mask files   (cell_mask, nuc_mask, ch0_crop per frame)
    Returns (df, frame_records) sorted by frame index.
    """
    df = pd.read_csv(RESULTS_CSV)
    df = df.sort_values("frame").reset_index(drop=True)

    # find mask frames
    mask_files = [f for f in os.listdir(MASKS_FOLDER)
                  if f.startswith("cell_mask_t") and f.endswith(".npy")]
    frame_indices = sorted(
        int(re.search(r"t(\d+)", f).group(1)) for f in mask_files)

    # only keep frames present in both CSV and masks
    csv_frames = set(df["frame"].tolist())
    frame_indices = [t for t in frame_indices if t in csv_frames]

    records = []
    for t in frame_indices:
        records.append({
            "t":         t,
            "cell_mask": np.load(os.path.join(MASKS_FOLDER,
                         "cell_mask_t{:04d}.npy".format(t))),
            "nuc_mask":  np.load(os.path.join(MASKS_FOLDER,
                         "nuc_mask_t{:04d}.npy".format(t))),
            "ch0_crop":  np.load(os.path.join(MASKS_FOLDER,
                         "ch0_crop_t{:04d}.npy".format(t))),
        })

    return df, records


# ----------------------------------------------------------------------
# FRAME RENDERER
# ----------------------------------------------------------------------

def render_frame(fig, ax_img, ax_gfp, ax_mch,
                 rec, df, frame_idx,
                 gfp_cn_all, mch_cn_all, time_axis):
    """
    Renders one video frame into the provided axes.
    """
    t         = rec["t"]
    ch0       = rec["ch0_crop"]
    cell_mask = rec["cell_mask"]
    nuc_mask  = rec["nuc_mask"]

    row = df[df["frame"] == t]
    chosen_lbl    = None
    chosen_centroid = None

    if not row.empty and not np.isnan(row["centroid_row"].values[0]):
        cr = row["centroid_row"].values[0]
        cc = row["centroid_col"].values[0]
        chosen_centroid = (cr, cc)

        # find the cell label closest to recorded centroid
        best_lbl, best_d = None, float("inf")
        for lbl in np.unique(cell_mask):
            if lbl == 0:
                continue
            rows_px, cols_px = np.where(cell_mask == lbl)
            r_c, c_c = rows_px.mean(), cols_px.mean()
            d = math.sqrt((r_c - cr) ** 2 + (c_c - cc) ** 2)
            if d < best_d:
                best_d, best_lbl = d, lbl
        chosen_lbl = best_lbl

    # ── re-apply dark styling after clear() ─────────────────────────
    for ax in [ax_img, ax_gfp, ax_mch]:
        ax.clear()
        ax.set_facecolor("#111111")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    # ── LEFT: cell image ──────────────────────────────────────────────
    lo, hi = np.percentile(ch0, 1), np.percentile(ch0, 99)
    disp = np.clip((ch0.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
    ax_img.imshow(disp, cmap="gray", interpolation="nearest")

    # all cells semi-transparent
    n_lbl = int(cell_mask.max())
    if n_lbl > 0:
        ov = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
        colours = cm.tab20(np.linspace(0, 1, max(n_lbl, 1)))
        for i, lbl in enumerate(range(1, n_lbl + 1)):
            px = cell_mask == lbl
            ov[px, :3] = colours[i % len(colours)][:3]
            ov[px,  3] = 0.25
        ax_img.imshow(ov, interpolation="nearest")

    if chosen_lbl is not None:
        cell_px = cell_mask == chosen_lbl

        # lime green for chosen cell
        cho = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
        cho[cell_px] = [0.0, 1.0, 0.2, 0.55]
        ax_img.imshow(cho, interpolation="nearest")

        # cyan for nucleus
        nuc_px = cell_px & (nuc_mask > 0)
        if nuc_px.any():
            nuo = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
            nuo[nuc_px] = [0.0, 1.0, 1.0, 0.7]
            ax_img.imshow(nuo, interpolation="nearest")

        # white border
        border = cell_px & ~binary_erosion(cell_px)
        yx = np.argwhere(border)
        if len(yx):
            ax_img.scatter(yx[:, 1], yx[:, 0], s=0.3,
                           c="white", linewidths=0, alpha=0.9)

        # centroid crosshair
        ax_img.plot(chosen_centroid[1], chosen_centroid[0],
                    "+", color="white", markersize=10, markeredgewidth=2)

    # time label
    time_min = t * FRAME_INTERVAL_MIN
    ax_img.set_title("Frame {:04d}  |  t = {} min".format(t, time_min),
                     fontsize=9, color="white", pad=3,
                     bbox=dict(fc="black", alpha=0.5, pad=2, lw=0))
    ax_img.axis("off")

    # ── RIGHT TOP: GFP C/N ────────────────────────────────────────────
    # plot full trace in light colour
    ax_gfp.plot(time_axis, gfp_cn_all, color="green",
                linewidth=1.2, alpha=0.3, zorder=1)
    # plot up to current frame in full colour
    past_mask = df["frame"] <= t
    past_time = df[past_mask]["frame"] * FRAME_INTERVAL_MIN
    past_gfp  = df[past_mask]["gfp_cn_ratio"]
    ax_gfp.plot(past_time, past_gfp, color="green",
                linewidth=2.0, zorder=2)
    # current point
    if not row.empty and not np.isnan(row["gfp_cn_ratio"].values[0]):
        ax_gfp.scatter([t * FRAME_INTERVAL_MIN],
                       [row["gfp_cn_ratio"].values[0]],
                       color="red", s=60, zorder=3)
    ax_gfp.axhline(1.0, color="gray", linewidth=1.0,
                   linestyle="--", alpha=0.7, label="C/N = 1")
    ax_gfp.set_xlim(time_axis.min() - 10, time_axis.max() + 10)
    ax_gfp.set_ylabel("C/N ratio", fontsize=8)
    ax_gfp.set_title("GFP  (AKT-KTR)  cyto / nuclear", fontsize=9)
    ax_gfp.legend(fontsize=7, loc="upper right")
    ax_gfp.grid(True, alpha=0.3)
    ax_gfp.tick_params(labelsize=7)
    # shade x label off for top plot
    ax_gfp.set_xticklabels([])

    # ── RIGHT BOTTOM: mCherry C/N ─────────────────────────────────────
    ax_mch.plot(time_axis, mch_cn_all, color="red",
                linewidth=1.2, alpha=0.3, zorder=1)
    past_mch = df[past_mask]["mcherry_cn_ratio"]
    ax_mch.plot(past_time, past_mch, color="red",
                linewidth=2.0, zorder=2)
    if not row.empty and not np.isnan(row["mcherry_cn_ratio"].values[0]):
        ax_mch.scatter([t * FRAME_INTERVAL_MIN],
                       [row["mcherry_cn_ratio"].values[0]],
                       color="darkred", s=60, zorder=3)
    ax_mch.axhline(1.0, color="gray", linewidth=1.0,
                   linestyle="--", alpha=0.7, label="C/N = 1")
    ax_mch.set_xlim(time_axis.min() - 10, time_axis.max() + 10)
    ax_mch.set_xlabel("Time (min)", fontsize=8)
    ax_mch.set_ylabel("C/N ratio", fontsize=8)
    ax_mch.set_title("mCherry  (ERK-KTR)  cyto / nuclear", fontsize=9)
    ax_mch.legend(fontsize=7, loc="upper right")
    ax_mch.grid(True, alpha=0.3)
    ax_mch.tick_params(labelsize=7)


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("\n[1/3] Loading masks and results ...")
    df, records = load_data()
    print("      {} frames loaded.".format(len(records)))

    if len(records) == 0:
        raise RuntimeError("No frames found. Check MASKS_FOLDER and RESULTS_CSV paths.")

    # pre-compute full time axis and C/N traces for background line
    time_axis  = df["frame"].values * FRAME_INTERVAL_MIN
    gfp_cn_all = df["gfp_cn_ratio"].values
    mch_cn_all = df["mcherry_cn_ratio"].values

    # ── Set up video writer ──────────────────────────────────────────
    video_path = os.path.join(OUTPUT_FOLDER, VIDEO_NAME)

    print("\n[2/3] Rendering {} frames at {} FPS ...".format(len(records), FPS))
    print("      Output --> {}".format(video_path))

    from io import BytesIO
    import PIL.Image

    frames_buffer = []
    for i, rec in enumerate(records):
        # create a fresh figure each frame -- guaranteed clean capture
        fig_f = plt.figure(figsize=(14, 6), facecolor="black")
        gs_f  = gridspec.GridSpec(
            2, 2,
            width_ratios=[1, 1.6], height_ratios=[1, 1],
            hspace=0.35, wspace=0.1,
            left=0.04, right=0.97, top=0.90, bottom=0.10,
        )
        ax_img_f = fig_f.add_subplot(gs_f[:, 0])
        ax_gfp_f = fig_f.add_subplot(gs_f[0, 1])
        ax_mch_f = fig_f.add_subplot(gs_f[1, 1])
        fig_f.suptitle("AKT & ERK activity  |  single cell tracking",
                       fontsize=11, color="white", y=0.97)

        render_frame(fig_f, ax_img_f, ax_gfp_f, ax_mch_f,
                     rec, df, i,
                     gfp_cn_all, mch_cn_all, time_axis)

        # save to BytesIO buffer and read back as numpy array
        buf = BytesIO()
        fig_f.savefig(buf, format="png", dpi=DPI,
                      facecolor="black", bbox_inches="tight")
        buf.seek(0)
        frame_img = np.array(PIL.Image.open(buf).convert("RGB"))
        frames_buffer.append(frame_img)
        plt.close(fig_f)   # free memory

        if (i + 1) % 10 == 0 or i == 0:
            print("  Frame {}/{} rendered.".format(i + 1, len(records)))

    print("  Writing video file ...")
    # Use OpenCV VideoWriter -- most reliable on Windows
    h, w = frames_buffer[0].shape[:2]
    # ensure dimensions are even (required by mp4 codec)
    w = w if w % 2 == 0 else w - 1
    h = h if h % 2 == 0 else h - 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(video_path, fourcc, FPS, (w, h))

    if not vw.isOpened():
        # fallback to avi with XVID if mp4v fails
        video_path = video_path.replace(".mp4", ".avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        vw = cv2.VideoWriter(video_path, fourcc, FPS, (w, h))

    for frame_img in frames_buffer:
        # cv2 expects BGR, matplotlib gives RGB — convert
        frame_bgr = cv2.cvtColor(frame_img[:h, :w], cv2.COLOR_RGB2BGR)
        vw.write(frame_bgr)

    vw.release()

    print("\n[3/3] Done.")
    print("  Video saved --> {}\n".format(video_path))


if __name__ == "__main__":
    main()