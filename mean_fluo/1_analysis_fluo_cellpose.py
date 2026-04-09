"""
Multi-Channel Cellpose Segmentation (Segmentation Only)
Dual segmentation approach:
  - DIC (ch0) mask
  - H2B (ch3) mask

Run with: conda activate cellpose3 && python analysis_seg_only.py
"""

import os
import numpy as np
import glob
from pathlib import Path
from cellpose import models
from PIL import Image
import tifffile
import re
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

PARENT_DIR = r"C:\Users\path to the .tif folder"
EXPERIMENT_NAME = "New-02-Scene-45-P45-C03" #this is a sample
OUTPUT_DIR = os.path.join(PARENT_DIR, "segmentation_results")

CHANNELS = {
    "ch1": ("GFP",     f"{EXPERIMENT_NAME}_ch1"),
    "ch2": ("mCherry", f"{EXPERIMENT_NAME}_ch2"),
    "ch3": ("H2B",     f"{EXPERIMENT_NAME}_ch3"),
    "ch0": ("DIC",     f"{EXPERIMENT_NAME}_ch0"),
}

SEGMENTATION_CONFIGS = {
    "dic_mask": {
        "seg_channel":        "ch0",
        "model_type":         "cyto3",
        "diameter":           190,
        "flow_threshold":     0.4,
        "cellprob_threshold": 0.0,
        "mask_prefix":        "masks_dic",
    },
    "h2b_mask": {
        "seg_channel":        "ch3",
        "model_type":         "cyto3", #for our mask we chose cyto3 model instead of nuclei but you can change as per your need
        "diameter":           80,
        "flow_threshold":     0.4,
        "cellprob_threshold": 0.0,
        "mask_prefix":        "masks_h2b",
    },
}

USE_GPU = True

# ============================================================================
# IMAGE LOADING AND ORGANIZATION
# ============================================================================

def parse_filename(filename):
    match = re.search(r'_t(\d+)\.', filename)
    if match:
        return int(match.group(1))
    return None


def organize_images_by_timepoint(parent_dir, experiment_name, channels_dict):
    timepoint_data = defaultdict(dict)

    for channel_key, (channel_name, folder_name) in channels_dict.items():
        channel_dir = os.path.join(parent_dir, folder_name)

        if not os.path.exists(channel_dir):
            print(f"⚠ Warning: Channel folder not found: {channel_dir}")
            continue

        channel_files = (glob.glob(os.path.join(channel_dir, "*.png")) +
                         glob.glob(os.path.join(channel_dir, "*.tif")))
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
        print(f"  t{tp:04d}: {list(timepoint_data[tp].keys())}")

    return {t: timepoint_data[t] for t in sorted_timepoints}


def load_image(filepath):
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
    if seg_image.ndim == 2:
        seg_image = np.expand_dims(seg_image, axis=0)

    result = model.eval(
        seg_image,
        diameter=diameter,
        channels=[0, 0],
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )

    masks = result[0] if isinstance(result, tuple) else result
    return masks.squeeze()

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_segmentation(parent_dir, experiment_name, channels_dict, output_dir,
                     seg_configs, use_gpu):

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("ORGANIZING IMAGES")
    print("="*70)
    timepoint_data = organize_images_by_timepoint(parent_dir, experiment_name, channels_dict)

    if not timepoint_data:
        print("ERROR: No images found!")
        return

    print("\n" + "="*70)
    print("LOADING CELLPOSE MODELS")
    print("="*70)

    models_dict = {}
    for config_name, config in seg_configs.items():
        model_type = config["model_type"]
        print(f"Loading {model_type} model for {config_name} (GPU={use_gpu})...")
        models_dict[config_name] = models.Cellpose(gpu=use_gpu, model_type=model_type)

    print("\n" + "="*70)
    print("SEGMENTATION")
    print("="*70)

    for tp_idx, (timepoint, channel_files) in enumerate(timepoint_data.items(), 1):
        print(f"\n[{tp_idx}/{len(timepoint_data)}] Processing timepoint t{timepoint:04d}")

        try:
            # Load only the channels needed for segmentation
            loaded_images = {}
            seg_channels_needed = {cfg["seg_channel"] for cfg in seg_configs.values()}
            for ch_key in seg_channels_needed:
                if ch_key in channel_files:
                    img = load_image(channel_files[ch_key])
                    if img is not None:
                        loaded_images[ch_key] = img

            for config_name, config in seg_configs.items():
                seg_ch = config["seg_channel"]
                mask_prefix = config["mask_prefix"]

                if seg_ch not in loaded_images:
                    print(f"  ✗ Missing {seg_ch} for {config_name}, skipping")
                    continue

                seg_label = f"{CHANNELS[seg_ch][0]} ({seg_ch})"
                print(f"  → [{config_name}] Segmenting on {seg_label}...")

                masks = segment_image(
                    loaded_images[seg_ch],
                    models_dict[config_name],
                    config["diameter"],
                    config["flow_threshold"],
                    config["cellprob_threshold"],
                )

                print(f"    Found {masks.max()} cells")

                mask_path = os.path.join(output_dir, f"{mask_prefix}_t{timepoint:04d}.tif")
                tifffile.imwrite(mask_path, masks.astype(np.uint32))
                print(f"    ✓ Saved: {mask_prefix}_t{timepoint:04d}.tif")

        except Exception as e:
            print(f"  ✗ Error processing t{timepoint:04d}: {str(e)}")
            continue

    print("\n✓ Segmentation complete!")
    print(f"Masks saved to: {output_dir}")


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MULTI-CHANNEL CELLPOSE — SEGMENTATION ONLY")
    print("="*70)
    print(f"\nParent directory: {PARENT_DIR}")
    print(f"Experiment:       {EXPERIMENT_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")

    print(f"\nSegmentation strategy:")
    for name, cfg in SEGMENTATION_CONFIGS.items():
        seg_label = CHANNELS[cfg['seg_channel']][0]
        print(f"  {name}: segment on {seg_label} ({cfg['seg_channel']}) | "
              f"model={cfg['model_type']}, diameter={cfg['diameter']}")

    run_segmentation(
        PARENT_DIR,
        EXPERIMENT_NAME,
        CHANNELS,
        OUTPUT_DIR,
        seg_configs=SEGMENTATION_CONFIGS,
        use_gpu=USE_GPU,
    )
