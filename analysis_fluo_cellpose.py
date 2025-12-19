"""
Multi-Channel Cellpose Segmentation with Fluorescence Quantification
Segments cells using H2B (ch2) nuclear marker and quantifies fluorescence 
from GFP (ch0), mCherry (ch1), and H2B (ch2) for each cell across time-lapse

Run with: conda activate cellpose3 && python cellpose_multichannel_analysis.py
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

# ============================================================================
# CONFIGURATION
# ============================================================================

# Parent directory containing all channel folders
PARENT_DIR = r"C:\Users\gz24763\OneDrive - University of Bristol\Documents\PC9_images\12dec25_data\output\P1"

# Experiment name (used to find channel folders)
EXPERIMENT_NAME = "New-01-Scene-02-P1-A02"

# Output directory for results
OUTPUT_DIR = os.path.join(PARENT_DIR, "segmentation_results")

# Channel information and folder names
CHANNELS = {
    "ch0": ("GFP", f"{EXPERIMENT_NAME}_ch0"),
    "ch1": ("mCherry", f"{EXPERIMENT_NAME}_ch1"), 
    "ch2": ("H2B", f"{EXPERIMENT_NAME}_ch2"),
    "ch3": ("DIC", f"{EXPERIMENT_NAME}_ch3")  # Phase contrast/DIC - used only for reference
}

# Segmentation settings
SEGMENTATION_CHANNEL = "ch3"  # Use H2B for nuclear segmentation
MODEL_TYPE = "cyto"  # nuclei model is best for H2B
DIAMETER = 30  # Adjust based on nuclear size in your images
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0

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
    
    Expected structure:
    parent_dir/
    ├── New-01-Scene-02-P1-A02_ch0/
    │   ├── New-01-Scene-02-P1-A02_ch0_t0000.png
    │   ├── New-01-Scene-02-P1-A02_ch0_t0001.png
    │   └── ...
    ├── New-01-Scene-02-P1-A02_ch1/
    ├── New-01-Scene-02-P1-A02_ch2/
    └── New-01-Scene-02-P1-A02_ch3/
    
    Returns: dict with structure {timepoint: {channel: filepath}}
    """
    timepoint_data = defaultdict(dict)
    
    # Iterate through each channel
    for channel_key, (channel_name, folder_name) in channels_dict.items():
        channel_dir = os.path.join(parent_dir, folder_name)
        
        if not os.path.exists(channel_dir):
            print(f"⚠ Warning: Channel folder not found: {channel_dir}")
            continue
        
        # Find all PNG files in this channel folder
        channel_files = glob.glob(os.path.join(channel_dir, "*.png"))
        print(f"Found {len(channel_files)} files in {folder_name}")
        
        for filepath in channel_files:
            filename = os.path.basename(filepath)
            
            # Extract timepoint
            timepoint = parse_filename(filename)
            if timepoint is None:
                continue
            
            timepoint_data[timepoint][channel_key] = filepath
    
    # Sort by timepoint
    sorted_timepoints = sorted(timepoint_data.keys())
    
    if not sorted_timepoints:
        print("ERROR: No timepoints found!")
        return {}
    
    print(f"\nFound {len(sorted_timepoints)} unique timepoints")
    print(f"Timepoint range: t{sorted_timepoints[0]:04d} to t{sorted_timepoints[-1]:04d}")
    
    # Verify all channels present
    print("\nChannel availability:")
    for tp in sorted_timepoints[:3]:  # Check first 3 timepoints
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

def segment_nuclei(seg_image, model, diameter, flow_threshold, cellprob_threshold):
    """
    Segment nuclei using Cellpose
    Returns: masks (labeled image where each cell has unique ID)
    Handles different Cellpose versions that may return different outputs
    """
    # Ensure image is proper format
    if seg_image.ndim == 2:
        seg_image = np.expand_dims(seg_image, axis=0)
    
    # Run segmentation
    # Different Cellpose versions return different numbers of outputs
    result = model.eval(
        seg_image,
        diameter=diameter,
        channels=[0, 0],  # Single channel
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold
    )
    
    # Handle different return formats
    if isinstance(result, tuple):
        if len(result) >= 3:
            # Standard format: masks, flows, styles
            masks = result[0]
        elif len(result) == 2:
            # Some versions return: masks, flows
            masks = result[0]
        else:
            masks = result[0]
    else:
        # Just masks returned
        masks = result
    
    return masks.squeeze()

# ============================================================================
# FLUORESCENCE QUANTIFICATION
# ============================================================================

def quantify_fluorescence(masks, images_dict, channels_to_analyze):
    """
    Quantify fluorescence for each cell in specified channels
    
    Args:
        masks: Labeled segmentation mask
        images_dict: Dict with channel names as keys and image arrays as values
        channels_to_analyze: List of channel names to quantify (e.g., ['ch0', 'ch1', 'ch2'])
    
    Returns:
        List of dicts with cell statistics
    """
    cell_data = []
    
    # Get unique cell IDs (excluding 0 which is background)
    cell_ids = np.unique(masks)
    cell_ids = cell_ids[cell_ids > 0]
    
    for cell_id in cell_ids:
        # Create binary mask for this cell
        cell_mask = masks == cell_id
        
        # Basic morphology
        cell_info = {
            "cell_id": int(cell_id),
            "cell_area": int(np.sum(cell_mask)),  # Number of pixels
        }
        
        # Quantify fluorescence in each channel
        for channel in channels_to_analyze:
            if channel not in images_dict:
                continue
            
            image = images_dict[channel]
            
            # Get pixel intensities within cell
            cell_pixels = image[cell_mask]
            
            # Calculate statistics
            cell_info[f"{channel}_mean"] = float(np.mean(cell_pixels))
            cell_info[f"{channel}_median"] = float(np.median(cell_pixels))
            cell_info[f"{channel}_sum"] = float(np.sum(cell_pixels))
            cell_info[f"{channel}_std"] = float(np.std(cell_pixels))
            cell_info[f"{channel}_max"] = float(np.max(cell_pixels))
            cell_info[f"{channel}_min"] = float(np.min(cell_pixels))
        
        cell_data.append(cell_info)
    
    return cell_data

# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def analyze_timelapse(parent_dir, experiment_name, channels_dict, output_dir, seg_channel, channels_to_quantify,
                      model_type, diameter, flow_threshold, cellprob_threshold, use_gpu):
    """
    Main analysis pipeline for multi-channel time-lapse
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Organize images
    print("\n" + "="*70)
    print("ORGANIZING IMAGES")
    print("="*70)
    timepoint_data = organize_images_by_timepoint(parent_dir, experiment_name, channels_dict)
    
    if not timepoint_data:
        print("ERROR: No images found!")
        return
    
    # Initialize Cellpose model
    print("\n" + "="*70)
    print("LOADING CELLPOSE MODEL")
    print("="*70)
    print(f"Loading {model_type} model (GPU={use_gpu})...")
    model = models.Cellpose(gpu=use_gpu, model_type=model_type)
    
    # Process each timepoint
    print("\n" + "="*70)
    print("SEGMENTATION AND QUANTIFICATION")
    print("="*70)
    
    all_results = []
    
    for tp_idx, (timepoint, channel_files) in enumerate(timepoint_data.items(), 1):
        print(f"\n[{tp_idx}/{len(timepoint_data)}] Processing timepoint t{timepoint:04d}")
        
        try:
            # Load segmentation channel
            if seg_channel not in channel_files:
                print(f"  ✗ Missing {seg_channel}, skipping timepoint")
                continue
            
            seg_img = load_image(channel_files[seg_channel])
            if seg_img is None:
                continue
            
            # Segment
            print(f"  → Segmenting {seg_channel}...")
            masks = segment_nuclei(seg_img, model, diameter, 
                                  flow_threshold, cellprob_threshold)
            
            num_cells = masks.max()
            print(f"  → Found {num_cells} cells")
            
            # Load all analysis channels
            print(f"  → Loading channels for quantification...")
            images_dict = {}
            for channel in channels_to_quantify:
                if channel in channel_files:
                    img = load_image(channel_files[channel])
                    if img is not None:
                        images_dict[channel] = img
            
            # Quantify fluorescence
            print(f"  → Quantifying fluorescence...")
            cell_data = quantify_fluorescence(masks, images_dict, 
                                             channels_to_quantify)
            
            # Add timepoint information
            for cell_stat in cell_data:
                cell_stat["timepoint"] = timepoint
            
            all_results.extend(cell_data)
            
            # Save mask for this timepoint
            mask_path = os.path.join(output_dir, f"masks_t{timepoint:04d}.tif")
            tifffile.imwrite(mask_path, masks.astype(np.uint32))
            
            print(f"  ✓ Saved mask: masks_t{timepoint:04d}.tif")
            
        except Exception as e:
            print(f"  ✗ Error processing t{timepoint:04d}: {str(e)}")
            continue
    
    # Export to CSV
    if all_results:
        print("\n" + "="*70)
        print("EXPORTING RESULTS")
        print("="*70)
        
        df = pd.DataFrame(all_results)
        
        # Reorder columns for readability
        cols = ["timepoint", "cell_id", "cell_area"]
        # Add all other columns (channel quantifications)
        other_cols = [c for c in df.columns if c not in cols]
        df = df[cols + sorted(other_cols)]
        
        csv_path = os.path.join(output_dir, "cell_fluorescence_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"✓ Results exported: cell_fluorescence_data.csv")
        print(f"  Total measurements: {len(df)} cells across {len(timepoint_data)} timepoints")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(df.groupby("timepoint").size())
        
        return df
    
    else:
        print("ERROR: No results to export!")
        return None

# ============================================================================
# ANALYSIS AND VISUALIZATION HELPERS
# ============================================================================

def analyze_results_csv(csv_path):
    """
    Load and display summary statistics from results CSV
    """
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*70)
    print("RESULT SUMMARY")
    print("="*70)
    print(f"\nTotal cells analyzed: {len(df)}")
    print(f"Timepoints: {df['timepoint'].min():.0f} to {df['timepoint'].max():.0f}")
    print(f"Mean cells per timepoint: {df.groupby('timepoint').size().mean():.1f}")
    
    print("\nChannel-specific statistics:")
    for channel in ["ch0", "ch1", "ch2"]:
        col = f"{channel}_mean"
        if col in df.columns:
            print(f"\n{CHANNELS.get(channel, channel)}:")
            print(f"  Mean intensity: {df[col].mean():.1f} ± {df[col].std():.1f}")
            print(f"  Range: {df[col].min():.1f} - {df[col].max():.1f}")
    
    return df

# ============================================================================
# RUN ANALYSIS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MULTI-CHANNEL CELLPOSE ANALYSIS WITH FLUORESCENCE QUANTIFICATION")
    print("="*70)
    print(f"Parent directory: {PARENT_DIR}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Segmentation channel: {SEGMENTATION_CHANNEL} (H2B nuclei)")
    print(f"Analysis channels: {', '.join(['ch0 (GFP)', 'ch1 (mCherry)', 'ch2 (H2B)'])}")
    print(f"Model: {MODEL_TYPE}, Diameter: {DIAMETER} pixels")
    
    # Run main analysis
    results_df = analyze_timelapse(
        PARENT_DIR,
        EXPERIMENT_NAME,
        CHANNELS,
        OUTPUT_DIR,
        seg_channel=SEGMENTATION_CHANNEL,
        channels_to_quantify=["ch0", "ch1", "ch2"],
        model_type=MODEL_TYPE,
        diameter=DIAMETER,
        flow_threshold=FLOW_THRESHOLD,
        cellprob_threshold=CELLPROB_THRESHOLD,
        use_gpu=USE_GPU
    )
    
    # Summary statistics
    if results_df is not None:
        csv_path = os.path.join(OUTPUT_DIR, "cell_fluorescence_data.csv")
        analyze_results_csv(csv_path)
        
        print("\n✓ Analysis complete!")
        print(f"Results saved to: {OUTPUT_DIR}")