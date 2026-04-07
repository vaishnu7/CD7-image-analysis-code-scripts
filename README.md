# CD7 Image Analysis Code Scripts

Python scripts for automated fluorescence image analysis of live-cell timelapse microscopy data, including cell segmentation, fluorescence quantification, and KTR (Kinase Translocation Reporter) C/N ratio calculation.

---

## Repository Structure

```
CD7-image-analysis-code-scripts/
├── convert_czi_tif.py          # Convert .czi microscopy files to .tif
├── cell_track/                 # Cell tracking across timepoints
└── mean_fluo/                  # Mean fluorescence quantification scripts
```

---

## Scripts Overview

### `convert_czi_tif.py`
Converts raw `.czi` (Zeiss) microscopy files into individual channel `.tif` or `.png` images, organised by channel and timepoint — the expected input format for all downstream scripts.

### `mean_fluo/`
Scripts for multi-channel fluorescence quantification using a **dual segmentation approach**:
- **DIC channel** → whole-cell mask
- **H2B channel** → nuclear mask
- Cytoplasm is derived as: `whole-cell mask − nuclear mask`
- Fluorescence (median intensity, background-corrected) is measured in nucleus and cytoplasm for GFP (AKT-KTR) and mCherry (ERK-KTR) channels
- Outputs C/N ratios and optional iRFP-normalised C/N ratios

### `cell_track/`
Scripts for tracking individual cells across timeframes in timelapse experiments.

---

## Pipeline

```
Raw .czi files
      │
      ▼
convert_czi_tif.py          → per-channel .png/.tif images
      │
      ▼
Cellpose segmentation        → masks_dic_t****.tif
(analysis_seg_only.py)       → masks_h2b_t****.tif
      │
      ▼
mean_fluo / cell_track       → C/N ratios, fluorescence CSVs, plots
```

---

## Requirements

```
cellpose
numpy
pandas
scipy
matplotlib
tifffile
Pillow
```

Install dependencies:
```bash
pip install cellpose numpy pandas scipy matplotlib tifffile Pillow
```

It is recommended to run segmentation scripts inside the `cellpose3` conda environment:
```bash
conda activate cellpose3 && python your_script.py
```

---

## Configuration

Each script contains a `CONFIGURATION` block at the top. Key parameters to set:

| Parameter | Description |
|---|---|
| `PARENT_DIR` | Root directory containing channel image folders |
| `EXPERIMENT_NAME` | Prefix used to find channel subfolders |
| `OUTPUT_DIR` | Where masks and results are saved |
| `USE_GPU` | `True`/`False` for Cellpose GPU acceleration |
| `NUC_EROSION_PX` | Pixels to erode from nuclear mask edge (0 = disabled) |
| `CYTO_EROSION_PX` | Pixels to erode from cytoplasm mask edge (0 = disabled) |

---

## Outputs

- `masks_dic_t****.tif` — whole-cell segmentation masks per timepoint
- `masks_h2b_t****.tif` — nuclear segmentation masks per timepoint
- `cytoplasm_binary_quantification.csv` — C/N ratios and compartment intensities per timepoint
- `cn_ratio_binary_ch*.png` — C/N ratio plots over time
- `cn_irfp_binary_ch*.png` — iRFP-normalised C/N ratio plots
- `mask_overlay/` — visual QC overlays of masks per timepoint
