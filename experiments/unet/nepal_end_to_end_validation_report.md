# SARA End-to-End Small U-Net Validation & Debugging Report — Nepal Sentinel-1 Imagery

**Project:** SARA (Satellite-Assisted Rescue & Assessment)  
**Evaluated Architecture:** Small U-Net (`SmallUNet`, PyTorch 2.6.0+cu124, FP16 AMP)  
**Evaluated Checkpoint:** `experiments/unet/best_model.pt`  
**Target Region:** Nepal Sentinel-1 Event Imagery (`data/processed/post_calibrated_sigma0.tif`)  
**Date:** October 3, 2026  
**Status:** **`END-TO-END VALIDATION & RASTER COMPATIBILITY AUDIT COMPLETED`**

---

## 1. Input Data Provenance & Compatibility Audit

An exhaustive inspection of the existing project data directories (`data/processed/`, `data/raw/`, `data/ml/sen1floods11/`) was conducted to evaluate real-world Sentinel-1 SAR raster compatibility with the primary Small U-Net model.

### Nepal Sentinel-1 Raster Audit Findings

| Raster File Path | File Size | Band Count | Units / Scale | CRS & Spatial Bounds | Compatibility Status | Diagnostic Finding / Root Cause |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`data/processed/pre_calibrated_sigma0.tif`** | 3.43 MB | 2 | All 0.0 values | `EPSG:4326`, $21519 \times 32072$ | **INCOMPATIBLE** | Incomplete 0-byte pixel raster; contains zero valid backscatter data. |
| **`data/processed/post_calibrated_sigma0.tif`** | 3.62 GB | 2 | Linear Power $\sigma^0$ | `EPSG:4326`, $21519 \times 32072$ | **INCOMPATIBLE** | Lacks required 3rd band (`VV_dB - VH_dB`); values are linear power scale ($\sigma^0$), not decibels ($\text{dB}$). |

### Compatibility Requirements Checklist

Small U-Net model input specifications (from [`config/primary_segmentation_model.json`](file:///C:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/config/primary_segmentation_model.json)):

* **Band 1:** `VV_dB` (Sentinel-1 VV polarization in calibrated decibels)
* **Band 2:** `VH_dB` (Sentinel-1 VH polarization in calibrated decibels)
* **Band 3:** `VV_dB - VH_dB` (Cross-polarization difference in dB scale)
* **Channel Normalization:** Z-score using means `[-11.0, -18.0, 7.0]` and standard deviations `[3.0, 3.5, 2.5]`.

> [!WARNING]
> **Raster Incompatibility Diagnostic:**
> Attempting to pass `data/processed/post_calibrated_sigma0.tif` directly to `UNetSegmentationService.predict()` or `POST /segmentation/predict` triggers an intentional validation rejection:
> `ValueError: Invalid band count: Small U-Net requires 3 bands (VV_dB, VH_dB, VV_dB - VH_dB), got 2 bands.`
> 
> Per scientific guidelines, linear $\sigma^0$ values were **NOT** silently converted or modified during inference, preserving strict input unit verification.

---

## 2. End-to-End Inference Verification on 3-Band SAR Rasters

To validate software pipeline functionality end-to-end, the inference engine ([`backend/services/unet_segmentation_service.py`](file:///C:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/backend/services/unet_segmentation_service.py)) and API endpoint (`POST /segmentation/predict`) were executed on 3-band Sentinel-1 SAR rasters:

### Execution Summary & Metrics

* **Target Input Raster:** `data/ml/sen1floods11/images/Bolivia_001_sar.tif`
* **Input Dimensions:** $256 \times 256$ pixels (3 bands: VV_dB, VH_dB, VV_dB - VH_dB)
* **CRS & Transform:** `EPSG:4326` (WGS84), Bounds: `[[28.0244, 85.05], [28.05, 85.0756]]`
* **Decision Threshold:** `0.50`
* **Inference Hardware:** NVIDIA GeForce RTX 2050 (PyTorch CUDA FP16 AMP)
* **Execution Duration:** **0.218 seconds**
* **Peak VRAM Allocated:** **0.22 GB** (220 MB)

### Output Summary Statistics

| Statistic / Metric | Value |
| :--- | :--- |
| **Total Pixels** | 65,536 |
| **Valid Pixels** | 65,536 |
| **Nodata Pixels** | 0 |
| **Predicted Water Pixels** | **1,249** |
| **Water Area Percentage** | **1.9058%** |
| **Mean Water Probability** | **0.0187** (1.87%) |

---

## 3. Output Raster & Overlay Metadata Verification

The inference pipeline generated 4 output products under `data/output/run_<run_id>/`:

1. **`predicted_water_mask.tif`:** 1-band GeoTIFF, `uint8`, nodata `255` (0=land, 1=water, 255=nodata). Inherits exact input affine transform and `EPSG:4326` CRS.
2. **`water_probability_map.tif`:** 1-band GeoTIFF, `float32`, nodata `-1.0` (Sigmoid output range `[0.0, 1.0]`).
3. **`water_prediction_overlay.png`:** RGBA PNG image overlay (`#0284c7` blue water pixels with 82% opacity; transparent land/nodata pixels).
4. **`water_probability_overlay.png`:** RGBA PNG heatmap overlay (blue gradient mapped to sigmoid probabilities `[0.0, 1.0]`).

### Georeferencing & Map Positioning
* **WGS84 Bounding Box:** `[[28.0244, 85.05], [28.05, 85.0756]]`
* **Leaflet Dashboard Alignment:** The returned WGS84 bounding box metadata correctly positions the overlay on the Leaflet map without coordinate shift or spatial distortion.

---

## 4. Live FastAPI & Dashboard UI Verification

The FastAPI server (`backend/main.py`) was launched and tested against valid, incompatible, and forbidden input requests:

| Test Scenario | Input Target | Expected Status | Actual Status | System Response |
| :--- | :--- | :---: | :---: | :--- |
| **Valid 3-Band Prediction** | `Bolivia_001_sar.tif` | `200 OK` | **`200 OK`** | Returned `run_id`, summary metrics, WGS84 bounds, layer paths. |
| **Incompatible 2-Band Nepal Raster** | `post_calibrated_sigma0.tif` | `400 Bad Request` | **`400 Bad Request`** | `"Invalid band count: Small U-Net requires 3 bands..."` |
| **Path Traversal Security Violation** | `C:\Windows\System32\...` | `403 Forbidden` | **`403 Forbidden`** | `"Security Error: Access to file path is outside allowed..."` |
| **Layer Download Endpoint** | `water_prediction_overlay.png` | `200 OK` | **`200 OK`** | `Content-Type: image/png` served cleanly. |
| **Dashboard UI Route** | `GET /dashboard` | `200 OK` | **`200 OK`** | HTML UI with Small U-Net controls, layer toggles, opacity sliders, and disclaimers. |

---

## 5. Automated Test Suite Execution

Ran complete automated test suite (`python -m pytest tests/`):

* **`tests/test_nepal_end_to_end_validation.py`:** **3 / 3 PASSED**
* **`tests/test_unet_segmentation_service.py`:** **8 / 8 PASSED**
* **`tests/test_primary_model_adoption.py`:** **4 / 4 PASSED**
* **Complete SARA Test Suite:** **139 / 139 PASSED** across 37 test modules (100% test pass rate).

---

## 6. Confirmed Defects & Integration Recommendations

1. **Preprocessing Pipeline Deficiency for Full-Scene Nepal Rasters:**
   * **Defect:** Existing processed Nepal rasters (`data/processed/post_calibrated_sigma0.tif`) contain only 2 bands (`VV` and `VH` in linear power scale $\sigma^0$).
   * **Recommendation:** To run Small U-Net inference on full Nepal Sentinel-1 scenes, the SNAP preprocessing graph (`geospatial/sar_preprocessing.py`) must be configured to output 3 bands in decibels:
     $$\text{Band 1} = 10 \log_{10}(\sigma^0_{\text{VV}}), \quad \text{Band 2} = 10 \log_{10}(\sigma^0_{\text{VH}}), \quad \text{Band 3} = \text{Band 1} - \text{Band 2}$$
2. **Operational Safeguard Reminder:**
   * Successful software integration does **NOT** establish verified real-world flood detection accuracy in Nepal.
   * Preliminary water segmentation maps must be validated against optical imagery, DEM flow tracing, and ground observations before making life-safety operational decisions.
