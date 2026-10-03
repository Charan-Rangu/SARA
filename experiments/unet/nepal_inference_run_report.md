# Small U-Net Nepal SAR Water Segmentation Inference Experiment Report

**Date:** October 3, 2026  
**Project:** SARA (Satellite-Assisted Rescue & Assessment)  
**Experiment Run ID:** `nepal_unet_eval_20261003`  
**Run Output Directory:** `data/output/run_nepal_unet_eval_20261003/`  
**Target Corrected Input:** `data/processed/unet_inputs/post_calibrated_sigma0_unet_3band_corrected.tif`  
**Model Configuration:** `config/primary_segmentation_model.json`  
**Model Checkpoint:** `experiments/unet/best_model.pt`  

---

## 1. Executive Summary

A controlled, reproducible Small U-Net inference experiment was executed on the corrected 3-band Nepal Sentinel-1 SAR input raster (`post_calibrated_sigma0_unet_3band_corrected.tif`).

The primary Small U-Net model checkpoint (`experiments/unet/best_model.pt`, 1.93 million parameters) performed windowed inference ($256 \times 256$ block size) across the $690,157,368$-pixel scene in **606.82 seconds** on an NVIDIA GPU (`cuda`). 

All independent post-inference output verification checks passed:
* Binary output values strictly contain only `{0 (land), 1 (water), 255 (nodata)}`.
* Valid pixel probabilities fall strictly in $[0.0000, 1.0000]$ (mean `0.0904`, std `0.2447`).
* Out of **379,266,320 valid pixels** ($54.95\%$), the model predicted **32,309,359 water pixels**, yielding a predicted preliminary water coverage of **8.5189%**.

---

## 2. Preconditions & Execution Environment

| Parameter | Execution Setting / Verified Value | Status |
| :--- | :--- | :---: |
| **Input Raster Path** | `data/processed/unet_inputs/post_calibrated_sigma0_unet_3band_corrected.tif` | **VERIFIED** |
| **Input Channels** | `[VV_dB, VH_dB, VV_dB - VH_dB]` (Bands 1, 2, 3) | **VERIFIED** |
| **Model Checkpoint** | `experiments/unet/best_model.pt` (PyTorch `SmallUNet`, 1.93M parameters) | **VERIFIED** |
| **Z-Score Normalization** | Means: `[-11.0, -18.0, 7.0]`, Standard Deviations: `[3.0, 3.5, 2.5]` | **VERIFIED** |
| **Normalization Frequency** | Applied exactly once per window chip in memory before model forward pass | **VERIFIED** |
| **Decision Threshold** | `0.50` probability threshold | **VERIFIED** |
| **Inference Hardware** | NVIDIA GPU (`cuda`) | **VERIFIED** |
| **Software Versions** | Python `3.13.5`, PyTorch `2.6.0+cu124`, Rasterio `1.4.3` | **VERIFIED** |

---

## 3. Quantitative Scene Statistics & Results

### 3.1 Pixel Distribution
* **Total Scene Pixels:** 690,157,368 ($32,072 \times 21,519$ pixels across 3 channels)
* **Valid Observation Pixels:** 379,266,320 pixels (**54.9536%**)
* **Nodata Pixels:** 310,891,048 pixels (45.0464%, nodata value `255.0`)

### 3.2 Water Prediction & Probability Statistics

| Metric | Value | Denominator / Basis |
| :--- | :---: | :--- |
| **Predicted Water Pixels** | **32,309,359** | Pixels with probability $\ge 0.50$ |
| **Predicted Land Pixels** | **346,956,961** | Pixels with probability $< 0.50$ |
| **Predicted Water Percentage** | **8.5189%** | Denominator = 379,266,320 valid pixels |
| **Mean Water Probability** | **0.0904** | Valid pixels only |
| **Standard Deviation** | **0.2447** | Valid pixels only |
| **Minimum Probability** | **0.0000** | Valid pixels only |
| **Maximum Probability** | **1.0000** | Valid pixels only |
| **Inference Runtime** | **606.82 s** (~10.1 minutes) | Full-scene GPU evaluation |

---

## 4. Independent Output Verification Checks

Following inference completion, all generated output rasters were reopened and independently audited:

| Verification Item | Result / Finding | Status |
| :--- | :--- | :---: |
| **Spatial Dimensions** | $32,072 \times 21,519$ pixels (matches input raster exactly) | **PASS** |
| **CRS & Geotransform** | `EPSG:4326`, Affine `(8.98315e-05, 0.0, 83.9258, 0.0, -8.98315e-05, 29.5243)` | **PASS** |
| **Binary Mask Values** | `np.unique` returns strictly `{0, 1, 255}` | **PASS** |
| **Probability Range** | All valid pixels are finite and lie strictly within $[0.0000, 1.0000]$ | **PASS** |
| **Nodata Exclusion** | Nodata pixels (`255.0`) are assigned `-1.0` in probability map and `255` in binary mask | **PASS** |
| **Denominator Integrity** | Water percentage (8.5189%) correctly uses valid pixel count ($379.27\text{M}$), not total scene area | **PASS** |
| **Non-Trivial Output** | Output contains realistic water body extent without empty or all-nodata artifacts | **PASS** |

---

## 5. Artifacts Generated

The run artifacts were saved in `data/output/run_nepal_unet_eval_20261003/`:

1. **`predicted_water_mask.tif`**: Georeferenced uint8 GeoTIFF binary water mask (`0=land`, `1=water`, `255=nodata`).
2. **`water_probability_map.tif`**: Georeferenced float32 GeoTIFF water probability map (`-1.0=nodata`, `[0.0, 1.0]=probability`).
3. **`water_prediction_overlay.png`**: RGBA PNG overlay for map visualization (`#0284c7` blue for water, transparent for land/nodata).
4. **`water_probability_overlay.png`**: RGBA PNG probability heatmap overlay for Leaflet map rendering.
5. **`nepal_unet_run_metrics.json`**: Machine-readable JSON summary of metrics, runtime, hardware, and configuration.

---

## 6. Scientific Limitations & Disclaimers

1. **Preliminary SAR Extent Only:** Model predictions represent preliminary probabilistic SAR water extent based on Sentinel-1 backscatter.
2. **No Life-Safety Claims:** Predictions must NOT be used directly to claim road closures, settlement access limitations, structural damage, or area safety.
3. **Strict Isolation from EMSR927:** EMSR927 activation data and change masks were kept strictly isolated and were not used for training, tuning, threshold selection, or evaluation in this experiment.
4. **Accuracy Disclaimer:** No accuracy metrics (IoU, F1, precision, recall) are reported for this Nepal scene because no independent, ground-truth label mask appropriate for this event was used.
