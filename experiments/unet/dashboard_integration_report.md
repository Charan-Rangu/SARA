# SARA Small U-Net Dashboard Integration Report

**Project:** SARA (Satellite-Assisted Rescue & Assessment)  
**Primary Deep-Learning Model:** Small U-Net (`SmallUNet`, PyTorch 2.6.0+cu124)  
**Evaluated Checkpoint:** `experiments/unet/best_model.pt`  
**Configuration File:** `config/primary_segmentation_model.json`  
**Service File:** `backend/services/unet_segmentation_service.py`  
**Date:** October 3, 2026  
**Status:** **`DASHBOARD INTEGRATION & END-TO-END INFERENCE VERIFIED`**

---

## 1. Existing Components Reused

To maintain architectural integrity and avoid duplication, the following existing project components were reused:

1. **Model Architecture (`ml/model.py`):** Reused `SmallUNet` 4-stage encoder/decoder PyTorch model definition (1,927,009 trainable parameters).
2. **Model Checkpoint (`experiments/unet/best_model.pt`):** Reused trained state dict saved at Epoch 24.
3. **Model Configuration (`config/primary_segmentation_model.json`):** Reused primary model configuration specifying 3-band input structure, channel statistics, nodata encoding (255), and probability threshold (0.50).
4. **FastAPI Application (`backend/main.py`):** Reused main API service, routing infrastructure, static layer serving, and interactive HTML dashboard UI.
5. **Geospatial Pipeline Utilities (`rasterio` & `rasterio.warp`):** Reused windowed raster processing pattern and WGS84 bounding box coordinate transformer for Leaflet map overlay alignment.

---

## 2. Files Created or Modified

| File Path | Status | Purpose / Modification Summary |
| :--- | :---: | :--- |
| **`backend/services/unet_segmentation_service.py`** | **CREATED** | Core inference service (`UNetSegmentationService`) handling path security validation, raster verification, windowed FP16/CPU evaluation, GeoTIFF output generation, Leaflet RGBA PNG overlay generation, and metric calculation. |
| **`backend/main.py`** | **MODIFIED** | Added `POST /segmentation/predict` endpoint, updated `GET /dashboard/layers/{run_id}/{layer_name}` to serve PNG/TIFF overlays, and added Small U-Net controls, probability toggles, opacity sliders, and disclaimers to the dashboard HTML UI. |
| **`tests/test_unet_segmentation_service.py`** | **CREATED** | Focused unit & integration test suite (8 tests) covering checkpoint loading, inference, invalid band count handling, nodata preservation, georeferencing, API endpoints, and path security. |
| **`experiments/unet/dashboard_integration_report.md`** | **CREATED** | Comprehensive deliverable report detailing integration architecture, API specification, dashboard controls, test results, and scientific safeguards. |

---

## 3. API Endpoint Specification

### `POST /segmentation/predict`

Executes Small U-Net water segmentation inference on a target 3-band Sentinel-1 SAR raster.

#### Request Payload (JSON)
```json
{
  "sar_raster_path": "data/raw/Sen1Floods11/Somalia_028_SAR.tif",
  "probability_threshold": 0.50,
  "run_id": "custom_run_01"
}
```

#### Response Structure (JSON 200 OK)
```json
{
  "status": "ok",
  "run_id": "custom_run_01",
  "model_name": "Small U-Net",
  "probability_threshold": 0.50,
  "sar_raster_path": "C:\\Users\\chara\\...\\Somalia_028_SAR.tif",
  "output_directory": "C:\\Users\\chara\\...\\data\\output\\run_custom_run_01",
  "binary_mask_path": "C:\\Users\\chara\\...\\predicted_water_mask.tif",
  "probability_map_path": "C:\\Users\\chara\\...\\water_probability_map.tif",
  "binary_overlay_png_path": "C:\\Users\\chara\\...\\water_prediction_overlay.png",
  "probability_overlay_png_path": "C:\\Users\\chara\\...\\water_probability_overlay.png",
  "wgs84_bounds": [[2.0, 45.0], [2.256, 45.256]],
  "summary": {
    "total_pixels": 65536,
    "valid_pixels": 65024,
    "nodata_pixels": 512,
    "predicted_water_pixels": 3098,
    "water_percentage": 4.7644,
    "mean_water_probability": 0.0892
  },
  "disclaimer": "Preliminary SAR-based water prediction — independent confirmation required."
}
```

#### Security & Error Codes
* **`403 Forbidden`:** Path traversal attack or accessing files outside allowed workspace directories (`data/`, `experiments/`, `scratch/`, `validation/`, `models/`).
* **`400 Bad Request`:** Invalid band count (not equal to 3), unplausible dB values, non-finite inputs, missing CRS, or raster dimensions < 16x16.
* **`404 Not Found`:** SAR raster file or primary model checkpoint missing.

---

## 4. Model Loading & Preprocessing Details

* **Checkpoint:** `experiments/unet/best_model.pt` loaded via `torch.load` with `map_location` hardware autodetect (CUDA GPU / CPU).
* **Input Band Specification (3 Channels):**
  1. `VV_dB`: Sentinel-1 VV backscatter in decibels.
  2. `VH_dB`: Sentinel-1 VH backscatter in decibels.
  3. `VV_dB - VH_dB`: Cross-polarization log-ratio difference in decibels.
* **Unit Verification:** Validates that input rasters are in calibrated dB scale before applying Z-score normalization. If values are already in dB, conversion is not repeated.
* **Channel Normalization:**
  * Means: `[-11.0, -18.0, 7.0]`
  * Standard Deviations: `[3.0, 3.5, 2.5]`
* **Evaluation Mode:** Runs under `model.eval()` with `torch.no_grad()` enabled to disable gradient computation and conserve memory.

---

## 5. Raster Georeferencing, Nodata, & Windowed Processing

* **Windowed Tile Processing:** Processes rasters in memory-efficient `256 x 256` spatial windows, avoiding loading entire giant satellite rasters into RAM.
* **Metadata & Georeferencing Preservation:** Output GeoTIFF rasters (`predicted_water_mask.tif` and `water_probability_map.tif`) strictly inherit the exact width, height, Coordinate Reference System (CRS), and Affine Transform from the input raster.
* **Nodata Mask Preservation:** Pixels flagged with nodata (value `255` in label masks or input nodata values) are strictly preserved as `255` in binary outputs and `-1.0` in probability maps. Invalid pixels are **never** converted into predicted land or water.
* **Map-Ready RGBA PNG Overlays:** Generates transparent RGBA PNG overlays (`water_prediction_overlay.png` and `water_probability_overlay.png`) paired with `wgs84_bounds` (`[[min_lat, min_lon], [max_lat, max_lon]]`) for Leaflet `L.imageOverlay` map rendering.

---

## 6. Dashboard Controls Implemented

Integrated into the interactive SARA web dashboard served at `GET /dashboard`:

1. **SAR Raster Path Selection Input (`#sarPath`):** Allows selecting local SAR GeoTIFF rasters.
2. **Decision Threshold Selector (`#probThreshold`):** Configures water prediction decision threshold (default `0.50`).
3. **Inference Trigger Button:** Executes `runSegmentation()` calling `POST /segmentation/predict`.
4. **Predicted Water Extent Toggle & Opacity Slider:** Toggles binary water layer and adjusts opacity (0% to 100%).
5. **Water Probability Map Toggle & Opacity Slider:** Toggles continuous sigmoid probability heatmap overlay.
6. **Legend & Summary Metrics Card:** Displays total valid pixels, predicted water pixels, water percentage, and mean probability.
7. **Mandatory Scientific Labels & Disclaimers:**
   * Prominently displays: **"Preliminary SAR-based water prediction — independent confirmation required."**
   * Disclaims operational life-safety claims (road closures, settlement access, safety status).
   * Includes required attributions: `© Copernicus Sentinel Data (2024) | © OpenStreetMap contributors`.

---

## 7. Test Commands & Verification Results

### Test Execution Commands
```bash
# Run focused Small U-Net segmentation service test suite
.\.venv\Scripts\python.exe -m pytest tests/test_unet_segmentation_service.py

# Run complete SARA test suite
.\.venv\Scripts\python.exe -m pytest tests/
```

### Actual Test Results
* **`tests/test_unet_segmentation_service.py`:** **8 / 8 PASSED**
  1. `test_checkpoint_loading_success`: PASSED
  2. `test_valid_input_inference`: PASSED
  3. `test_missing_checkpoint_error`: PASSED
  4. `test_invalid_band_count_error`: PASSED
  5. `test_nodata_preservation`: PASSED
  6. `test_output_crs_geotransform_bounds`: PASSED
  7. `test_api_segmentation_endpoint`: PASSED
  8. `test_unauthorized_filesystem_access_prevention`: PASSED
* **Complete SARA Test Suite:** **136 / 136 PASSED** across 36 test modules (100% test pass rate).

---

## 8. Known Limitations & Remaining Work

1. **Window Boundary Padding:** Windowed tile inference pads edge tiles smaller than 256x256 to 256x256 before inference. Overlapping tile stitching (e.g. 32-pixel stride overlap) can be explored for edge smoothing in future releases.
2. **Multi-Scene Mosaicking:** Large multi-scene regional mosaics can be pre-tiled into memory-efficient blocks using the existing windowed pipeline.
3. **Operational Safeguards:** Model predictions represent preliminary probabilistic evidence. Operational deployment requires contextual validation with optical satellite data, DEM terrain flow tracing, and OpenStreetMap infrastructure layers before emergency response decisions.
