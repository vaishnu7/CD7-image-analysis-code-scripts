import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"C:\Users\path to folder"
POSITIONS = ["P41", "P42", "P43"]
CSV_NAME  = "segmentation_results/cytoplasm_binary_quantification.csv"

OUTPUT_DIR = os.path.join(BASE_DIR, "pooled_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
INTERVAL_MIN = 16  # minutes per timepoint

# ── Load all CSVs ─────────────────────────────────────────────────────────────
dfs = []
for pos in POSITIONS:
    path = os.path.join(BASE_DIR, pos, CSV_NAME)
    df = pd.read_csv(path)
    df["position"] = pos
    dfs.append(df)
    print(f"Loaded {pos}: {len(df)} timepoints")

all_data = pd.concat(dfs, ignore_index=True)

# ── Compute mean ± STD across positions per timepoint ─────────────────────────
ch_styles = {
    "ch1_cn_ratio": ("GFP (AKT-KTR)",              "#00cc44"),
    "ch2_cn_ratio": ("mCherry (ERK-KTR)",           "#ff4444"),
    "ch1_cn_irfp":  ("GFP (AKT-KTR) / iRFP norm",  "#009933"),
    "ch2_cn_irfp":  ("mCherry (ERK-KTR) / iRFP norm","#cc0000"),
}

grouped = all_data.groupby("timepoint")

# ── Plot C/N ratio ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Mean C/N ratio across positions (P41, P42, P43)", fontsize=13, fontweight="bold")

for ax, (col, (name, color)) in zip(axes, list(ch_styles.items())[:2]):
    stats = grouped[col].agg(["mean", "std"])
    stats.index = stats.index * INTERVAL_MIN  # convert to minutes
    ax.plot(stats.index, stats["mean"], marker='o', linewidth=2, markersize=4, color=color, label=name)
    ax.fill_between(stats.index, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                    alpha=0.25, color=color)
    ax.axhline(1, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, label="C/N = 1")
    ax.set_xlabel("Time (min)", fontsize=11)
    ax.set_ylabel("C/N ratio", fontsize=11)
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "mean_cn_ratio.png"), dpi=300, bbox_inches="tight")
print("✓ Saved: mean_cn_ratio.png")
plt.close()

# ── Plot iRFP-normalised ratio ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Mean C/(N/N_iRFP) across positions (P41, P42, P43)", fontsize=13, fontweight="bold")

for ax, (col, (name, color)) in zip(axes, list(ch_styles.items())[2:]):
    stats = grouped[col].agg(["mean", "std"])
    stats.index = stats.index * INTERVAL_MIN  # convert to minutes
    ax.plot(stats.index, stats["mean"], marker='o', linewidth=2, markersize=4, color=color, label=name)
    ax.fill_between(stats.index, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                    alpha=0.25, color=color)
    ax.set_xlabel("Time (min)", fontsize=11)
    ax.set_ylabel("C / (N / N_iRFP)  (a.u.)", fontsize=11)
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "mean_cn_irfp.png"), dpi=300, bbox_inches="tight")
print("✓ Saved: mean_cn_irfp.png")
plt.close()

# ── Save pooled summary CSV ────────────────────────────────────────────────────
summary = grouped[list(ch_styles.keys())].agg(["mean", "std"]).reset_index()
summary.columns = ["_".join(c).strip("_") for c in summary.columns]
summary.to_csv(os.path.join(OUTPUT_DIR, "pooled_mean_cn.csv"), index=False)
print("✓ Saved: pooled_mean_cn.csv")
