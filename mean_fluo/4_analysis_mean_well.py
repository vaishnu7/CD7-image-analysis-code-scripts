import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_DIR  = r"C:\Users\gz24763\OneDrive - University of Bristol\Documents\PC9_images\18_03_2026"
WELLS     = ["C4", "C5"]
CSV_NAME  = "pooled_results/pooled_mean_cn.csv"

OUTPUT_DIR   = os.path.join(BASE_DIR, "well_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)
INTERVAL_MIN = 16

# ── Channel styles ─────────────────────────────────────────────────────────────
ch_styles = {
    "ch1_cn_ratio": ("GFP (AKT-KTR)",                "#00cc44"),
    "ch2_cn_ratio": ("mCherry (ERK-KTR)",             "#ff4444"),
    "ch1_cn_irfp":  ("GFP (AKT-KTR) / iRFP norm",    "#009933"),
    "ch2_cn_irfp":  ("mCherry (ERK-KTR) / iRFP norm", "#cc0000"),
}

well_colors  = {WELLS[0]: "#1f77b4", WELLS[1]: "#ff7f0e"}
well_markers = {WELLS[0]: "o",       WELLS[1]: "s"}

# ── Load CSVs ──────────────────────────────────────────────────────────────────
well_data = {}
for well in WELLS:
    path = os.path.join(BASE_DIR, well, CSV_NAME)
    df   = pd.read_csv(path)
    df["time_min"] = df["timepoint"] * INTERVAL_MIN
    well_data[well] = df
    print(f"Loaded {well}: {len(df)} timepoints")

# ── Compute grand mean across wells ───────────────────────────────────────────
def grand_mean(col):
    """
    Stack the per-well means and compute the grand mean ± STD across wells.
    Returns (time_array, grand_mean_array, grand_std_array).
    """
    means = np.stack([well_data[w][f"{col}_mean"].values for w in WELLS], axis=0)
    t     = well_data[WELLS[0]]["time_min"].values
    return t, np.mean(means, axis=0), np.std(means, axis=0)

# ── Plotting function ──────────────────────────────────────────────────────────
def plot_well_comparison(cols, ylabel, title_suffix, fname):
    fig, axes = plt.subplots(1, len(cols), figsize=(7 * len(cols), 5))
    if len(cols) == 1:
        axes = [axes]
    fig.suptitle(f"Well comparison ({', '.join(WELLS)}) — {title_suffix}",
                 fontsize=13, fontweight="bold")

    for ax, col in zip(axes, cols):
        ch_name, ch_color = ch_styles[col]
        ax.set_title(ch_name, fontsize=11, fontweight="bold")

        # ── Individual wells (thin, semi-transparent) ──────────────────────
        for well in WELLS:
            df   = well_data[well]
            t    = df["time_min"].values
            mean = df[f"{col}_mean"].values
            std  = df[f"{col}_std"].values
            wc   = well_colors[well]
            wm   = well_markers[well]
            ax.plot(t, mean, marker=wm, linewidth=1.5, markersize=3,
                    color=wc, alpha=0.6, label=well)
            ax.fill_between(t, mean - std, mean + std,
                            alpha=0.10, color=wc)

        # ── Grand mean across wells (bold, channel colour) ─────────────────
        t_gm, gm, gs = grand_mean(col)
        ax.plot(t_gm, gm, linewidth=2.5, markersize=5, marker="D",
                color=ch_color, label=f"Mean ({', '.join(WELLS)})", zorder=5)
        ax.fill_between(t_gm, gm - gs, gm + gs,
                        alpha=0.25, color=ch_color)

        if "cn_ratio" in col:
            ax.axhline(1, color="gray", linestyle="--", linewidth=1.2,
                       alpha=0.7, label="C/N = 1")

        ax.set_xlabel("Time (min)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {fname}")
    plt.close()

# ── Generate plots ─────────────────────────────────────────────────────────────
plot_well_comparison(
    ["ch1_cn_ratio", "ch2_cn_ratio"],
    "C/N ratio", "C/N ratio", "well_cn_ratio.png"
)
plot_well_comparison(
    ["ch1_cn_irfp", "ch2_cn_irfp"],
    "C / (N / N_iRFP)  (a.u.)", "iRFP-normalised", "well_cn_irfp.png"
)

# ── Save summary CSV ───────────────────────────────────────────────────────────
combined = []
for well in WELLS:
    df = well_data[well].copy()
    df.insert(0, "well", well)
    combined.append(df)

# Append grand mean rows
t_ref = well_data[WELLS[0]]["time_min"].values
gm_rows = {"well": f"grand_mean_({'_'.join(WELLS)})",
            "time_min": t_ref}
for col in ch_styles:
    _, gm, gs = grand_mean(col)
    gm_rows[f"{col}_mean"] = gm
    gm_rows[f"{col}_std"]  = gs

combined.append(pd.DataFrame(gm_rows))
pd.concat(combined, ignore_index=True).to_csv(
    os.path.join(OUTPUT_DIR, "well_comparison_summary.csv"), index=False)
print("✓ Saved: well_comparison_summary.csv")