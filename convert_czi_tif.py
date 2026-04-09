#!/usr/bin/env python3
"""
CZI to Image Converter with Frame Extraction and Organized Output
Extracts individual frames from CZI files as TIFF or PNG and organizes them by channel and timepoint
Handles complex multi-dimensional data (Z-stacks, series, etc.)

To run use: C:/path to your python exe/AppData/Local/Programs/Python/Python311/python.exe 
  "C:/path to this script/convert_czi_tif.py" 
 "C:/path to your output"
(paste as one line)

"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from tifffile import imwrite
from PIL import Image

try:
    from czifile import CziFile
    HAS_CZIFILE = True
except ImportError:
    HAS_CZIFILE = False
    print("Warning: czifile not installed. Install with: pip install czifile")


def read_czi_file(filepath):
    """Read CZI file and return metadata and image data"""
    if not HAS_CZIFILE:
        raise ImportError("czifile library required. Install with: pip install czifile")
    
    with CziFile(filepath) as czi:
        img_array = czi.asarray()
        metadata = {
            'shape': czi.shape,
            'axes': czi.axes,
            'dtype': czi.dtype
        }
        return img_array, metadata


def get_channel_count(shape, axes):
    """Determine number of channels from shape and axes"""
    if 'C' in axes:
        c_idx = axes.index('C')
        return shape[c_idx]
    return 1


def get_timepoint_count(shape, axes):
    """Determine number of timepoints from shape and axes"""
    if 'T' in axes:
        t_idx = axes.index('T')
        return shape[t_idx]
    return 1


def extract_channel(img_array, axes, channel_num):
    """Extract a single channel from the full image array"""
    if 'C' not in axes:
        return img_array
    
    c_idx = axes.index('C')
    slices = [slice(None)] * len(img_array.shape)
    slices[c_idx] = channel_num
    
    return img_array[tuple(slices)]


def extract_timepoint(img_array, axes, timepoint_num):
    """Extract a single timepoint from the full image array"""
    if 'T' not in axes:
        return img_array
    
    t_idx = axes.index('T')
    slices = [slice(None)] * len(img_array.shape)
    slices[t_idx] = timepoint_num
    
    return img_array[tuple(slices)]


def flatten_to_2d(img_array):
    """
    Flatten multi-dimensional image to 2D by:
    - Taking first slice of Z-stack
    - Taking first serie if multiple series
    - Squeezing singleton dimensions
    """
    # Remove singleton dimensions
    img = np.squeeze(img_array)
    
    # If still more than 2D, take first slice of extra dimensions
    while len(img.shape) > 2:
        img = img[0]
    
    return img


def normalize_image(img):
    """Normalize image to 8-bit"""
    if img.dtype == np.uint8:
        return img
    
    img_min = img.min()
    img_max = img.max()
    
    if img_max > img_min:
        img_normalized = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        img_normalized = img.astype(np.uint8)
    
    return img_normalized


def merge_channels_rgb(channel1, channel2, channel3=None):
    """Merge channels into RGB image"""
    ch1 = flatten_to_2d(channel1)
    ch2 = flatten_to_2d(channel2)
    
    ch1 = normalize_image(ch1)
    ch2 = normalize_image(ch2)
    
    if channel3 is not None:
        ch3 = flatten_to_2d(channel3)
        ch3 = normalize_image(ch3)
        rgb = np.stack([ch1, ch2, ch3], axis=-1)
    else:
        ch3_empty = np.zeros_like(ch1)
        rgb = np.stack([ch1, ch2, ch3_empty], axis=-1)
    
    return rgb


def save_image(img_array, output_path, format='tif'):
    """Save image in specified format"""
    # Ensure 2D for PNG
    if format.lower() == 'png':
        img_2d = flatten_to_2d(img_array)
        img_2d = normalize_image(img_2d)
        
        if len(img_2d.shape) == 3 and img_2d.shape[2] == 3:
            # RGB image
            pil_img = Image.fromarray(img_2d, 'RGB')
        else:
            # Grayscale image
            pil_img = Image.fromarray(img_2d, 'L')
        pil_img.save(output_path)
    else:
        # TIFF can handle multi-dimensional
        imwrite(output_path, img_array)


def convert_czi_to_frames(input_czi, output_dir, merge_channels=None, format='tif'):
    """
    Convert CZI file to individual frame images organized by channel/timepoint
    
    Args:
        input_czi: Path to input CZI file
        output_dir: Base directory for output files
        merge_channels: List of channel indices to merge (e.g., [0, 2])
        format: Output format ('tif' or 'png')
    """
    
    filename = Path(input_czi).stem
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"Output format: {format.upper()}")
    print(f"{'='*60}")
    
    print(f"Reading CZI file: {input_czi}")
    img_array, metadata = read_czi_file(input_czi)
    
    axes = metadata['axes']
    shape = metadata['shape']
    
    print(f"Image shape: {shape}")
    print(f"Axes: {axes}")
    
    num_channels = get_channel_count(shape, axes)
    num_timepoints = get_timepoint_count(shape, axes)
    
    print(f"Channels found: {num_channels}")
    print(f"Timepoints found: {num_timepoints}")
    print(f"Note: Taking first Z-slice and first series for each frame")
    
    # Create base output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    ext = format.lower() if format.lower() in ['png', 'tif'] else 'tif'
    
    if merge_channels:
        # Create merged output directory
        merged_dir = os.path.join(output_dir, f"{filename}_merged_channels")
        Path(merged_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"\nMerging channels {merge_channels}...")
        print(f"Output directory: {merged_dir}")
        
        count = 0
        for t in range(num_timepoints):
            channels_data = []
            
            for c in merge_channels:
                if c >= num_channels:
                    print(f"Warning: Channel {c} does not exist (only {num_channels} channels)")
                    continue
                
                ch_data = extract_channel(img_array, axes, c)
                ch_data = extract_timepoint(ch_data, axes, t)
                channels_data.append(ch_data)
            
            if len(channels_data) >= 2:
                merged = merge_channels_rgb(
                    channels_data[0], 
                    channels_data[1],
                    channels_data[2] if len(channels_data) > 2 else None
                )
                
                output_file = os.path.join(merged_dir, f"{filename}_t{t:04d}.{ext}")
                save_image(merged, output_file, format=ext)
                count += 1
                
                if (t + 1) % max(1, num_timepoints // 10) == 0 or t == 0:
                    print(f"  Frame {t}: {output_file}")
        
        print(f"✓ Extracted {count} merged frames")
    
    else:
        # Extract all channels separately
        print(f"\nExtracting all channels as individual frames...")
        
        for c in range(num_channels):
            # Create directory for each channel
            channel_dir = os.path.join(output_dir, f"{filename}_ch{c}")
            Path(channel_dir).mkdir(parents=True, exist_ok=True)
            
            print(f"\nChannel {c}: {channel_dir}")
            
            for t in range(num_timepoints):
                ch_data = extract_channel(img_array, axes, c)
                ch_data = extract_timepoint(ch_data, axes, t)
                ch_data = flatten_to_2d(ch_data)
                ch_data = normalize_image(ch_data)
                
                output_file = os.path.join(channel_dir, f"{filename}_ch{c}_t{t:04d}.{ext}")
                save_image(ch_data, output_file, format=ext)
                
                if (t + 1) % max(1, num_timepoints // 10) == 0 or t == 0:
                    print(f"  Frame {t}: {output_file}")
            
            print(f"  ✓ Extracted {num_timepoints} frames for channel {c}")
    
    print(f"\n{'='*60}")
    print("✓ Conversion complete!")
    print(f"{'='*60}\n")


def process_directory(input_dir, output_dir, merge_channels=None, format='tif'):
    """Process all CZI files in a directory"""
    czi_files = list(Path(input_dir).glob("*.czi"))
    
    if not czi_files:
        print(f"No CZI files found in: {input_dir}")
        return
    
    print(f"Found {len(czi_files)} CZI files")
    
    for czi_file in sorted(czi_files):
        try:
            convert_czi_to_frames(str(czi_file), output_dir, merge_channels, format)
        except Exception as e:
            print(f"✗ Error processing {czi_file.name}: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='Extract frames from Zeiss CZI microscopy files to individual image files'
    )
    parser.add_argument('input', help='Path to CZI file or directory with CZI files')
    parser.add_argument('-o', '--output', default='./czi_output',
                       help='Output base directory (default: ./czi_output)')
    parser.add_argument('-m', '--merge', type=int, nargs='+',
                       help='Channel indices to merge (0-indexed). E.g., -m 0 2 for channels 1 and 3')
    parser.add_argument('-f', '--format', choices=['tif', 'png'], default='tif',
                       help='Output image format: tif or png (default: tif)')
    
    args = parser.parse_args()
    
    input_path = args.input
    
    if not os.path.exists(input_path):
        print(f"Error: Path not found: {input_path}")
        sys.exit(1)
    
    try:
        if os.path.isfile(input_path):
            # Single file
            if input_path.lower().endswith('.czi'):
                convert_czi_to_frames(input_path, args.output, args.merge, args.format)
            else:
                print(f"Error: File is not a CZI file: {input_path}")
                sys.exit(1)
        else:
            # Directory
            process_directory(input_path, args.output, args.merge, args.format)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
