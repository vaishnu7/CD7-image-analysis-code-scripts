"""
Multi-Channel Cellpose Segmentation with Fluorescence Quantification
Dual segmentation approach:
  - DIC (ch3) mask → quantify GFP (ch0) and mCherry (ch1)
  - H2B (ch2) mask → quantify H2B (ch2)
Background correction: mean of non-cell pixels subtracted per channel per timepoint

Run with: conda activate cellpose3 && python analysis_fluo_cellpose.py
"""

import os
import numpy as np
import pandas as pd
import glob
from pathlib import Path
from cellpose import models
from PIL import Image
from scipy import ndimage
import tifffile
import re
from collections import defaultdict
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

# Parent directory containing all channel folders
PARENT_DIR = r"C:\Users\gz24763\OneDrive - University of Bristol\Documents\PC9_images\12dec25_data\output\P20"

# Experiment name (used to find channel folders)
EXPERIMENT_NAME = "New-01-Scene-16-P20-B04"

# Output directory for results
OUTPUT_DIR = os.path.join(PARENT_DIR, "segmentation_results")

# Channel information and folder names
CHANNELS = {
    "ch0": ("GFP", f"{EXPERIMENT_NAME}_ch0"),
    "ch1": ("mCherry", f"{EXPERIMENT_NAME}_ch1"), 
    "ch2": ("H2B", f"{EXPERIMENT_NAME}_ch2"),
    "ch3": ("DIC", f"{EXPERIMENT_NAME}_ch3")
}

# ---- Dual segmentation configuration ----
# Each entry: segmentation channel, model, diameter, and which channels to quantify
SEGMENTATION_CONFIGS = {
    "dic_mask": {
        "seg_channel": "ch3",           # Segment on DIC
        "model_type": "cyto3",          # Cytoplasm model for DIC
        "diameter": 35,                 # Adjust to your cell size
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
        "quantify_channels": ["ch0", "ch1"],  # Measure GFP and mCherry
        "mask_prefix": "masks_dic",
    },
    "h2b_mask": {
        "seg_channel": "ch2",           # Segment on H2B
        "model_type": "cyto3",         # Nuclear model for H2B
        "diameter": 30,                 # Nuclear diameter (likely smaller than whole cell)
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
        "quantify_channels": ["ch2"],   # Measure H2B
        "mask_prefix": "masks_h2b",
    },
}

# GPU usage
USE_GPU = True

# ============================================================================
# IMAGE LOADING AND ORGANIZATION
# ============================================================================

def parse_filename(filename):
    """
    Parse filename to extract timepoint
    Example: New-01-Scene-02-P1-A02_ch0_t0070.png -> 0070
    """
    match = re.search(r'_t(\d+)\.', filename)
    if match:
        return int(match.group(1))
    return None

def organize_images_by_timepoint(parent_dir, experiment_name, channels_dict):
    """
    Organize all images by timepoint and channel from separate channel folders
    
    Returns: dict with structure {timepoint: {channel: filepath}}
    """
    timepoint_data = defaultdict(dict)
    
    for channel_key, (channel_name, folder_name) in channels_dict.items():
        channel_dir = os.path.join(parent_dir, folder_name)
        
        if not os.path.exists(channel_dir):
            print(f"⚠ Warning: Channel folder not found: {channel_dir}")
            continue
        
        channel_files = glob.glob(os.path.join(channel_dir, "*.png")) + glob.glob(os.path.join(channel_dir, "*.tif"))
        print(f"Found {len(channel_files)} files in {folder_name}")
        
        for filepath in channel_files:
            filename = os.path.basename(filepath)
            timepoint = parse_filename(filename)
            if timepoint is None:
                continue
            timepoint_data[timepoint][channel_key] = filepath
    
    sorted_timepoints = sorted(timepoint_data.keys())
    
    if not sorted_timepoints:
        print("ERROR: No timepoints found!")
        return {}
    
    print(f"\nFound {len(sorted_timepoints)} unique timepoints")
    print(f"Timepoint range: t{sorted_timepoints[0]:04d} to t{sorted_timepoints[-1]:04d}")
    
    print("\nChannel availability:")
    for tp in sorted_timepoints[:3]:
        channels_present = list(timepoint_data[tp].keys())
        print(f"  t{tp:04d}: {channels_present}")
    
    return {t: timepoint_data[t] for t in sorted_timepoints}

def load_image(filepath):
    """Load image and return as numpy array"""
    try:
        img = Image.open(filepath)
        return np.array(img)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

# ============================================================================
# SEGMENTATION
# ============================================================================

def segment_image(seg_image, model, diameter, flow_threshold, cellprob_threshold):
    """
    Segment using Cellpose
    Returns: masks (labeled image where each cell has unique ID)
    """
    if seg_image.ndim == 2:
        seg_image = np.expand_dims(seg_image, axis=0)
    
    result = model.eval(
        seg_image,
        diameter=diameter,
        channels=[0, 0],
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold
    )
    
    if isinstance(result, tuple):
        masks = result[0]
    else:
        masks = result
    
    return masks.squeeze()

# ============================================================================
# FLUORESCENCE QUANTIFICATION (WITH BACKGROUND CORRECTION)
# ============================================================================

def quantify_fluorescence(masks, images_dict, channels_to_analyze, mask_label=""):
    """
    Quantify fluorescence for each cell in specified channels,
    with background correction (mean of non-cell pixels subtracted).
    
    Args:
        masks: Labeled segmentation mask
        images_dict: Dict with channel names as keys and image arrays as values
        channels_to_analyze: List of channel names to quantify
        mask_label: Label to identify which mask was used (e.g., 'dic_mask' or 'h2b_mask')
    
    Returns:
        List of dicts with cell statistics (both raw and background-corrected)
    """
    cell_data = []
    
    cell_ids = np.unique(masks)
    cell_ids = cell_ids[cell_ids > 0]
    
    # Background mask: all pixels not belonging to any cell
    background_mask = masks == 0
    
    # Pre-compute background mean for each channel (once per timepoint)
    background_means = {}
    for channel in channels_to_analyze:
        if channel not in images_dict:
            continue
        image = images_dict[channel]
        bg_pixels = image[background_mask]
        background_means[channel] = float(np.mean(bg_pixels))
    
    for cell_id in cell_ids:
        cell_mask = masks == cell_id
        
        cell_info = {
            "cell_id": int(cell_id),
            "cell_area": int(np.sum(cell_mask)),
            "mask_source": mask_label,
        }
        
        for channel in channels_to_analyze:
            if channel not in images_dict:
                continue
            
            image = images_dict[channel]
            bg_mean = background_means[channel]
            cell_pixels = image[cell_mask]
            
            # Raw statistics
            cell_info[f"{channel}_mean_raw"] = float(np.mean(cell_pixels))
            cell_info[f"{channel}_median_raw"] = float(np.median(cell_pixels))
            cell_info[f"{channel}_sum_raw"] = float(np.sum(cell_pixels))
            cell_info[f"{channel}_std"] = float(np.std(cell_pixels))
            cell_info[f"{channel}_max_raw"] = float(np.max(cell_pixels))
            cell_info[f"{channel}_min_raw"] = float(np.min(cell_pixels))
            
            # Background-corrected statistics
            cell_info[f"{channel}_mean"] = float(np.mean(cell_pixels) - bg_mean)
            cell_info[f"{channel}_median"] = float(np.median(cell_pixels) - bg_mean)
            cell_info[f"{channel}_sum"] = float(np.sum(cell_pixels) - bg_mean * np.sum(cell_mask))
            cell_info[f"{channel}_max"] = float(np.max(cell_pixels) - bg_mean)
            cell_info[f"{channel}_min"] = float(np.min(cell_pixels) - bg_mean)
            
            # Background reference
            cell_info[f"{channel}_bg_mean"] = bg_mean
        
        cell_data.append(cell_info)
    
    return cell_data

# ============================================================================
# PLOTTING
# ============================================================================

# Channel plot settings: (channel_key, display_name, color, source_df_key)
CHANNEL_PLOT_INFO = [
    ("ch0", "GFP",      "#00ff00", "dic_mask"),
    ("ch1", "mCherry",   "#ff0000", "dic_mask"),
    ("ch2", "H2B",       "#0000ff", "h2b_mask"),
]


def plot_mean_per_cell_timeline(exported_dfs, output_dir):
    """
    Plot background-corrected mean fluorescence per cell over time
    for all 3 channels on a single figure.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for ch_key, ch_name, ch_color, df_key in CHANNEL_PLOT_INFO:
        if df_key not in exported_dfs:
            print(f"  ⚠ Skipping {ch_name}: no data from {df_key}")
            continue
        
        df = exported_dfs[df_key]
        col = f"{ch_key}_mean"
        
        if col not in df.columns:
            print(f"  ⚠ Skipping {ch_name}: column {col} not found")
            continue
        
        # Mean ± SEM per timepoint
        stats = df.groupby("timepoint")[col].agg(["mean", "std", "count"])
        stats["sem"] = stats["std"] / np.sqrt(stats["count"])
        
        ax.errorbar(stats.index.values, stats["mean"].values, yerr=stats["sem"].values,
                     marker='o', linestyle='-', linewidth=2.5, markersize=6,
                     color=ch_color, alpha=0.85, capsize=4, capthick=1.5,
                     label=f"{ch_name} ({ch_key})")
    
    ax.set_xlabel("Timepoint", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean Intensity ± SEM (background-corrected)", fontsize=13, fontweight="bold")
    ax.set_title("Mean Fluorescence Per Cell Over Time — All Channels", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12, loc="best")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "all_channels_mean_per_cell.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved plot: all_channels_mean_per_cell.png")
    plt.close()


def plot_total_fluorescence_per_field(exported_dfs, output_dir):
    """
    Plot total fluorescence per field of view over time.
    For each timepoint: sum of ch*_sum across all cells.
    If total stays flat while per-cell mean drops → dilution through division.
    If total also drops → photobleaching or genuine expression loss.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for ch_key, ch_name, ch_color, df_key in CHANNEL_PLOT_INFO:
        if df_key not in exported_dfs:
            print(f"  ⚠ Skipping {ch_name}: no data from {df_key}")
            continue
        
        df = exported_dfs[df_key]
        col = f"{ch_key}_sum"
        
        if col not in df.columns:
            print(f"  ⚠ Skipping {ch_name}: column {col} not found")
            continue
        
        # Total fluorescence per field = sum of all cells' integrated intensity
        total_per_tp = df.groupby("timepoint")[col].sum()
        
        ax.plot(total_per_tp.index.values, total_per_tp.values,
                marker='o', linestyle='-', linewidth=2.5, markersize=6,
                color=ch_color, alpha=0.85,
                label=f"{ch_name} ({ch_key})")
    
    ax.set_xlabel("Timepoint", fontsize=13, fontweight="bold")
    ax.set_ylabel("Total Integrated Intensity (background-corrected)", fontsize=13, fontweight="bold")
    ax.set_title("Total Fluorescence Per Field Over Time — All Channels", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12, loc="best")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "all_channels_total_per_field.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved plot: all_channels_total_per_field.png")
    plt.close()

# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def analyze_timelapse(parent_dir, experiment_name, channels_dict, output_dir,
                      seg_configs, use_gpu):
    """
    Main analysis pipeline with dual segmentation masks
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Organize images
    print("\n" + "="*70)
    print("ORGANIZING IMAGES")
    print("="*70)
    timepoint_data = organize_images_by_timepoint(parent_dir, experiment_name, channels_dict)
    
    if not timepoint_data:
        print("ERROR: No images found!")
        return
    
    # Load Cellpose models (one per segmentation config)
    print("\n" + "="*70)
    print("LOADING CELLPOSE MODELS")
    print("="*70)
    
    models_dict = {}
    for config_name, config in seg_configs.items():
        model_type = config["model_type"]
        print(f"Loading {model_type} model for {config_name} (GPU={use_gpu})...")
        models_dict[config_name] = models.Cellpose(gpu=use_gpu, model_type=model_type)
    
    # Process each timepoint
    print("\n" + "="*70)
    print("DUAL SEGMENTATION AND QUANTIFICATION")
    print("="*70)
    
    results_by_mask = {name: [] for name in seg_configs}
    
    for tp_idx, (timepoint, channel_files) in enumerate(timepoint_data.items(), 1):
        print(f"\n[{tp_idx}/{len(timepoint_data)}] Processing timepoint t{timepoint:04d}")
        
        try:
            # Pre-load all channel images for this timepoint
            loaded_images = {}
            for ch_key in channel_files:
                img = load_image(channel_files[ch_key])
                if img is not None:
                    loaded_images[ch_key] = img
            
            # Run each segmentation config
            for config_name, config in seg_configs.items():
                seg_ch = config["seg_channel"]
                quant_channels = config["quantify_channels"]
                mask_prefix = config["mask_prefix"]
                
                if seg_ch not in loaded_images:
                    print(f"  ✗ Missing {seg_ch} for {config_name}, skipping")
                    continue
                
                missing_quant = [ch for ch in quant_channels if ch not in loaded_images]
                if missing_quant:
                    print(f"  ⚠ Missing quantification channels {missing_quant} for {config_name}")
                
                # Segment
                seg_label = f"{CHANNELS[seg_ch][0]} ({seg_ch})"
                print(f"  → [{config_name}] Segmenting on {seg_label}...")
                masks = segment_image(
                    loaded_images[seg_ch],
                    models_dict[config_name],
                    config["diameter"],
                    config["flow_threshold"],
                    config["cellprob_threshold"]
                )
                
                num_cells = masks.max()
                print(f"    Found {num_cells} cells")
                
                # Quantify
                quant_labels = ", ".join([f"{CHANNELS[ch][0]}({ch})" for ch in quant_channels])
                print(f"    Quantifying: {quant_labels}")
                cell_data = quantify_fluorescence(
                    masks, loaded_images, quant_channels,
                    mask_label=config_name
                )
                
                # Report background
                if cell_data:
                    for ch in quant_channels:
                        bg_key = f"{ch}_bg_mean"
                        if bg_key in cell_data[0]:
                            print(f"    {ch} background mean: {cell_data[0][bg_key]:.2f}")
                
                # Add timepoint
                for cell_stat in cell_data:
                    cell_stat["timepoint"] = timepoint
                
                results_by_mask[config_name].extend(cell_data)
                
                # Save mask
                mask_path = os.path.join(output_dir, f"{mask_prefix}_t{timepoint:04d}.tif")
                tifffile.imwrite(mask_path, masks.astype(np.uint32))
                print(f"    ✓ Saved: {mask_prefix}_t{timepoint:04d}.tif")
            
        except Exception as e:
            print(f"  ✗ Error processing t{timepoint:04d}: {str(e)}")
            continue
    
    # Export results
    print("\n" + "="*70)
    print("EXPORTING RESULTS")
    print("="*70)
    
    exported_dfs = {}
    
    for config_name, results in results_by_mask.items():
        if not results:
            print(f"⚠ No results for {config_name}")
            continue
        
        df = pd.DataFrame(results)
        
        # Reorder columns
        cols = ["timepoint", "cell_id", "cell_area", "mask_source"]
        other_cols = [c for c in df.columns if c not in cols]
        df = df[cols + sorted(other_cols)]
        
        csv_name = f"cell_fluorescence_{config_name}.csv"
        csv_path = os.path.join(output_dir, csv_name)
        df.to_csv(csv_path, index=False)
        
        quant_channels = seg_configs[config_name]["quantify_channels"]
        ch_labels = ", ".join([f"{CHANNELS[ch][0]}({ch})" for ch in quant_channels])
        print(f"✓ {csv_name}")
        print(f"  Mask: {config_name} | Channels: {ch_labels}")
        print(f"  {len(df)} cells across {df['timepoint'].nunique()} timepoints")
        
        exported_dfs[config_name] = df
    
    # Also export a combined CSV for convenience
    if len(exported_dfs) > 1:
        combined = pd.concat(exported_dfs.values(), ignore_index=True)
        combined_path = os.path.join(output_dir, "cell_fluorescence_data.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\n✓ Combined file: cell_fluorescence_data.csv ({len(combined)} total rows)")
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    plot_mean_per_cell_timeline(exported_dfs, output_dir)
    plot_total_fluorescence_per_field(exported_dfs, output_dir)
    
    return exported_dfs

# ============================================================================
# ANALYSIS AND VISUALIZATION HELPERS
# ============================================================================

def analyze_results_csv(csv_path):
    """Load and display summary statistics from results CSV"""
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*70)
    print("RESULT SUMMARY")
    print("="*70)
    print(f"\nTotal cells analyzed: {len(df)}")
    print(f"Timepoints: {df['timepoint'].min():.0f} to {df['timepoint'].max():.0f}")
    print(f"Mean cells per timepoint: {df.groupby('timepoint').size().mean():.1f}")
    
    if 'mask_source' in df.columns:
        print(f"\nResults by mask:")
        for mask, group in df.groupby('mask_source'):
            print(f"  {mask}: {len(group)} cells")
    
    print("\nChannel-specific statistics (background-corrected):")
    for channel in ["ch0", "ch1", "ch2"]:
        col = f"{channel}_mean"
        col_raw = f"{channel}_mean_raw"
        bg_col = f"{channel}_bg_mean"
        if col in df.columns:
            subset = df.dropna(subset=[col])
            if len(subset) == 0:
                continue
            ch_name = CHANNELS.get(channel, (channel,))[0]
            print(f"\n{ch_name} ({channel}):")
            print(f"  Corrected mean intensity: {subset[col].mean():.1f} ± {subset[col].std():.1f}")
            if col_raw in subset.columns:
                print(f"  Raw mean intensity:       {subset[col_raw].mean():.1f} ± {subset[col_raw].std():.1f}")
            if bg_col in subset.columns:
                print(f"  Mean background:          {subset[bg_col].mean():.1f} ± {subset[bg_col].std():.1f}")
            print(f"  Corrected range: {subset[col].min():.1f} - {subset[col].max():.1f}")
    
    return df

# ============================================================================
# RUN ANALYSIS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MULTI-CHANNEL CELLPOSE ANALYSIS — DUAL SEGMENTATION")
    print("Background correction: mean of non-cell pixels subtracted per channel")
    print("="*70)
    print(f"\nParent directory: {PARENT_DIR}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nSegmentation strategy:")
    for name, cfg in SEGMENTATION_CONFIGS.items():
        seg_label = CHANNELS[cfg['seg_channel']][0]
        quant_labels = ", ".join([CHANNELS[ch][0] for ch in cfg['quantify_channels']])
        print(f"  {name}: segment on {seg_label} ({cfg['seg_channel']}) "
              f"→ quantify {quant_labels} | model={cfg['model_type']}, diameter={cfg['diameter']}")
    
    # Run main analysis
    results = analyze_timelapse(
        PARENT_DIR,
        EXPERIMENT_NAME,
        CHANNELS,
        OUTPUT_DIR,
        seg_configs=SEGMENTATION_CONFIGS,
        use_gpu=USE_GPU
    )
    
    # Summary statistics
    if results:
        for config_name in results:
            csv_path = os.path.join(OUTPUT_DIR, f"cell_fluorescence_{config_name}.csv")
            if os.path.exists(csv_path):
                analyze_results_csv(csv_path)
        
        print("\n✓ Analysis complete!")
        print(f"Results saved to: {OUTPUT_DIR}")