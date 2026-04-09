"""
Cell Counting Script - Quantify nuclei from H2B (ch2) images
Counts cells across all timepoints and exports results to CSV

Run with: conda activate cellpose3 && python cellpose_cell_counter.py
"""

import os
import numpy as np
import pandas as pd
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

# Parent directory containing all channel folders
PARENT_DIR = r"C:\Users\path to folder"

# Experiment name (used to find channel folders)
EXPERIMENT_NAME = "New-01-Scene-10-P9-A06"

# Output directory for results
OUTPUT_DIR = os.path.join(PARENT_DIR, "cell_counts")

# H2B channel settings
SEGMENTATION_CHANNEL = "ch2"  # Always use H2B for cell counting
MODEL_TYPE = "cyto3"  # nuclei model for H2B
DIAMETER = 36.7  # Adjust based on nuclear size - IMPORTANT!
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
    Example: New-01-Scene-02-P1-A02_ch2_t0070.png -> 0070
    """
    match = re.search(r'_t(\d+)\.', filename)
    if match:
        return int(match.group(1))
    return None

def organize_h2b_images(parent_dir, experiment_name):
    """
    Organize H2B (ch2) images by timepoint
    
    Expected structure:
    parent_dir/
    └── New-01-Scene-02-P1-A02_ch2/
        ├── New-01-Scene-02-P1-A02_ch2_t0000.png
        ├── New-01-Scene-02-P1-A02_ch2_t0001.png
        └── ...
    
    Returns: sorted list of (timepoint, filepath) tuples
    """
    h2b_folder = os.path.join(parent_dir, f"{experiment_name}_ch2")
    
    if not os.path.exists(h2b_folder):
        print(f"ERROR: H2B folder not found: {h2b_folder}")
        return []
    
    # Find all PNG files in H2B folder
    h2b_files = glob.glob(os.path.join(h2b_folder, "*.png"))
    print(f"Found {len(h2b_files)} H2B images")
    
    timepoint_files = []
    for filepath in h2b_files:
        filename = os.path.basename(filepath)
        timepoint = parse_filename(filename)
        if timepoint is not None:
            timepoint_files.append((timepoint, filepath))
    
    # Sort by timepoint
    timepoint_files.sort(key=lambda x: x[0])
    
    if timepoint_files:
        print(f"Timepoint range: t{timepoint_files[0][0]:04d} to t{timepoint_files[-1][0]:04d}")
        print(f"Total timepoints: {len(timepoint_files)}\n")
    
    return timepoint_files

def load_image(filepath):
    """Load image and return as numpy array"""
    try:
        img = Image.open(filepath)
        return np.array(img)
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return None

# ============================================================================
# CELL SEGMENTATION AND COUNTING
# ============================================================================

def segment_and_count_nuclei(image, model, diameter, flow_threshold, cellprob_threshold):
    """
    Segment H2B image and count nuclei
    
    Returns:
        - num_cells: number of nuclei detected
        - masks: labeled segmentation image
    """
    # Ensure image is proper format
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)
    
    # Run segmentation
    result = model.eval(
        image,
        diameter=None,
        channels=[0, 0],  # Single channel
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold
    )
    
    # Handle different return formats
    if isinstance(result, tuple):
        if len(result) >= 3:
            masks = result[0]
        elif len(result) == 2:
            masks = result[0]
        else:
            masks = result[0]
    else:
        masks = result
    
    masks = masks.squeeze()
    num_cells = masks.max()
    
    return int(num_cells), masks

# ============================================================================
# MAIN COUNTING PIPELINE
# ============================================================================

def count_cells_across_timelapse(parent_dir, experiment_name, output_dir, 
                                 diameter, flow_threshold, cellprob_threshold, use_gpu):
    """
    Main pipeline to count cells across all timepoints
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Organize images
    print("="*70)
    print("H2B CELL COUNTING")
    print("="*70)
    print(f"Parent directory: {parent_dir}")
    print(f"Experiment: {experiment_name}")
    print(f"Output directory: {output_dir}\n")
    
    timepoint_files = organize_h2b_images(parent_dir, experiment_name)
    
    if not timepoint_files:
        print("ERROR: No H2B images found!")
        return None
    
    # Initialize Cellpose model
    print("="*70)
    print("LOADING CELLPOSE MODEL")
    print("="*70)
    print(f"Loading {MODEL_TYPE} model (GPU={use_gpu})...")
    model = models.Cellpose(gpu=use_gpu, model_type=MODEL_TYPE)
    print("✓ Model loaded\n")
    
    # Process each timepoint
    print("="*70)
    print("CELL COUNTING")
    print("="*70)
    
    cell_counts = []
    
    for tp_idx, (timepoint, filepath) in enumerate(timepoint_files, 1):
        try:
            # Load image
            image = load_image(filepath)
            if image is None:
                continue
            
            print(f"[{tp_idx}/{len(timepoint_files)}] t{timepoint:04d}: ", end="", flush=True)
            
            # Segment and count
            num_cells, masks = segment_and_count_nuclei(
                image, model, diameter, 
                flow_threshold, cellprob_threshold
            )
            
            print(f"✓ {num_cells} cells detected")
            
            # Save mask for quality control
            mask_path = os.path.join(output_dir, f"h2b_mask_t{timepoint:04d}.tif")
            tifffile.imwrite(mask_path, masks.astype(np.uint32))
            
            # Save original image
            img_path = os.path.join(output_dir, f"h2b_original_t{timepoint:04d}.tif")
            tifffile.imwrite(img_path, image.astype(np.uint16))
            
            # Store results
            cell_counts.append({
                "timepoint": timepoint,
                "cell_count": num_cells,
                "image_file": os.path.basename(filepath)
            })
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            continue
    
    # Export to CSV
    if cell_counts:
        print("\n" + "="*70)
        print("EXPORTING RESULTS")
        print("="*70)
        
        df = pd.DataFrame(cell_counts)
        
        csv_path = os.path.join(output_dir, "cell_counts.csv")
        df.to_csv(csv_path, index=False)
        print(f"✓ Results exported: cell_counts.csv")
        
        # Print summary statistics
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        print(f"\nTimepoints processed: {len(df)}")
        print(f"Total cells detected: {df['cell_count'].sum()}")
        print(f"Mean cells per timepoint: {df['cell_count'].mean():.1f} ± {df['cell_count'].std():.1f}")
        print(f"Min: {df['cell_count'].min()}, Max: {df['cell_count'].max()}")
        
        print(f"\nCell count by timepoint:")
        print(df.to_string(index=False))
        
        return df
    
    else:
        print("ERROR: No results to export!")
        return None

# ============================================================================
# VISUALIZATION AND ANALYSIS HELPERS
# ============================================================================

def analyze_cell_counts(csv_path):
    """
    Load and analyze cell count data
    """
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*70)
    print("CELL COUNT ANALYSIS")
    print("="*70)
    
    print(f"\nTotal timepoints: {len(df)}")
    print(f"Total cells across experiment: {df['cell_count'].sum()}")
    
    # Temporal changes
    if len(df) > 1:
        first_count = df['cell_count'].iloc[0]
        last_count = df['cell_count'].iloc[-1]
        total_change = last_count - first_count
        percent_change = (total_change / first_count) * 100 if first_count > 0 else 0
        
        print(f"\nTemporal changes:")
        print(f"  Start (t{df['timepoint'].iloc[0]:04d}): {first_count} cells")
        print(f"  End (t{df['timepoint'].iloc[-1]:04d}): {last_count} cells")
        print(f"  Change: {total_change:+d} cells ({percent_change:+.1f}%)")
        
        # Find timepoint with max/min cells
        max_idx = df['cell_count'].idxmax()
        min_idx = df['cell_count'].idxmin()
        
        print(f"\n  Maximum: {df.loc[max_idx, 'cell_count']} cells at t{df.loc[max_idx, 'timepoint']:04d}")
        print(f"  Minimum: {df.loc[min_idx, 'cell_count']} cells at t{df.loc[min_idx, 'timepoint']:04d}")
    
    print(f"\nStatistics:")
    print(f"  Mean: {df['cell_count'].mean():.1f}")
    print(f"  Median: {df['cell_count'].median():.1f}")
    print(f"  Std Dev: {df['cell_count'].std():.1f}")
    
    return df

# ============================================================================
# RUN ANALYSIS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CELLPOSE H2B CELL COUNTER")
    print("="*70)
    print(f"Model: {MODEL_TYPE}")
    print(f"Diameter: {DIAMETER} pixels")
    print(f"Flow threshold: {FLOW_THRESHOLD}")
    print(f"GPU: {USE_GPU}\n")
    
    # Run cell counting
    results_df = count_cells_across_timelapse(
        PARENT_DIR,
        EXPERIMENT_NAME,
        OUTPUT_DIR,
        diameter=DIAMETER,
        flow_threshold=FLOW_THRESHOLD,
        cellprob_threshold=CELLPROB_THRESHOLD,
        use_gpu=USE_GPU
    )
    
    # Analysis
    if results_df is not None:
        csv_path = os.path.join(OUTPUT_DIR, "cell_counts.csv")
        analyze_cell_counts(csv_path)
        
        print("\n✓ Cell counting complete!")
        print(f"Results saved to: {OUTPUT_DIR}")
        print(f"\nOutput files:")
        print(f"  - cell_counts.csv (summary)")
        print(f"  - h2b_mask_tXXXX.tif (segmentation masks for QC)")
        print(f"  - h2b_original_tXXXX.tif (original H2B images)")
