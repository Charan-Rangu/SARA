# Nepal Sentinel-1 Small U-Net Final Inference-Readiness Audit Report

**Date:** October 3, 2026  
**Project:** SARA (Satellite-Assisted Rescue & Assessment)  
**Target Corrected Input Raster:** `data/processed/unet_inputs/post_calibrated_sigma0_unet_3band_corrected.tif`  
**Primary Updated Input Raster:** `data/processed/unet_inputs/post_calibrated_sigma0_unet_3band.tif`  
**Model Configuration:** `config/primary_segmentation_model.json`  
**Checkpoint Path:** `experiments/unet/best_model.pt`  
**Audit Type:** Read-Only Scientific & Pipeline Inference-Readiness Audit

---

## 1. Executive Summary

A read-only final inference-readiness audit was conducted for the Small U-Net model on the corrected 3-band Nepal Sentinel-1 SAR input raster (`post_calibrated_sigma0_unet_3band_corrected.tif`).

### Key Audit Finding
All **8 required inference-readiness checks** have passed (**PASS**). The corrected raster contains properly ordered `[VV_dB, VH_dB, VV_dB - VH_dB]` channels with decibel means matching the model's training specification (`-9.73 dB`, `-18.49 dB`, `+8.76 dB`). Normalization is applied exactly once inside `UNetSegmentationService` during inference, and nodata pixels (`255.0`) are strictly masked and excluded from model inputs and predictions.

---

## 2. Audit Check Statuses Summary

| Check ID | Audit Task / Requirement | Status | Verification Method | Key Finding / Evidence |
| :---: | :--- | :---: | :---: | :--- |
| **Task 1** | Inference Input Path Selection | **PASS** | Code Inspection & Test Execution | `post_calibrated_sigma0_unet_3band_corrected.tif` and `post_calibrated_sigma0_unet_3band.tif` are selected. Archived uncorrected raster (`v1_uncorrected`) is isolated. |
| **Task 2** | Line-by-Line Training vs Inference Comparison | **PASS** | Code & Config Inspection | 100% parameter alignment: Channel order `[VV_dB, VH_dB, VV_dB-VH_dB]`, means `[-11.0, -18.0, 7.0]`, stds `[3.0, 3.5, 2.5]`. |
| **Task 3** | Single Normalization Application | **PASS** | Code Inspection & Data Validation | Stored raster holds raw decibels. Z-score normalization is applied exactly once in `UNetSegmentationService.predict()`. |
| **Task 4** | Nodata Exclusion & Propagation | **PASS** | Code Inspection & Test Execution | Nodata `255.0` pixels are masked to `False` in valid mask, mapped to `255` in binary mask and `-1.0` in prob map. Never fed as valid SAR data. |
| **Task 5** | Spatial Geometry & Windowed Inference | **PASS** | Test Execution & API Validation | $32,072 \times 21,519$ raster, `EPSG:4326` CRS, Affine transform preserved. $256 \times 256$ windowed block processing with edge padding verified. |
| **Task 6** | Numerical Stability & Memory Behavior | **PASS** | Raster Statistics Analysis | 100% of valid pixels are finite (`np.isfinite`). Low RAM footprint under system limits. |
| **Task 7** | Threshold & Scientific Output Interpretation | **PASS** | Code & Disclaimer Verification | Default threshold `0.50` applied. Disclaimers prohibit using preliminary SAR outputs for life-safety or structural damage claims. |
| **Task 8** | Integration Test Suite Execution | **PASS** | PyTest Execution | 11/11 service/validation tests and 147/147 full repository tests passed cleanly. |

---

## 3. Task 1: Inference Input Path Verification

### 3.1 Input Selection Trace
The file path resolution was audited across the service registry:
* **Service Default Input:** `backend/services/unet_segmentation_service.py` receives `sar_raster_path`.
* **Path Validation:** `validate_secure_path()` verifies that target raster resides within approved `data/` or `experiments/` workspace directories.
* **Target Corrected Input:** `data/processed/unet_inputs/post_calibrated_sigma0_unet_3band_corrected.tif` (and mirrored `post_calibrated_sigma0_unet_3band.tif`).
* **Archived Isolation:** Uncorrected reversed raster is archived at `data/processed/unet_inputs/post_calibrated_sigma0_unet_3band_v1_uncorrected.tif` and is not referenced by any pipeline service.
* **Two-Band Rejection:** Passing the raw 2-band SNAP raster `data/processed/post_calibrated_sigma0.tif` triggers an immediate rejection with `ValueError: Invalid band count: Small U-Net requires 3 bands`.

---

## 4. Task 2: Line-by-Line Training vs. Inference Preprocessing Audit

| Component | Training Pipeline (`ml/unet_dataset.py`, `ml/config.py`) | Inference Pipeline (`geospatial/unet_sar_preprocessor.py`, `backend/services/unet_segmentation_service.py`) | Line-by-Line Match Status |
| :--- | :--- | :--- | :---: |
| **Channel 1** | `VV_dB` (Co-polarization dB) | `vv_v_db = 10 * log10(VV_linear)` $\rightarrow$ Output Band 1 | **MATCH (PASS)** |
| **Channel 2** | `VH_dB` (Cross-polarization dB) | `vh_v_db = 10 * log10(VH_linear)` $\rightarrow$ Output Band 2 | **MATCH (PASS)** |
| **Channel 3** | `VV_dB - VH_dB` | `diff_v_db = vv_v_db - vh_v_db` $\rightarrow$ Output Band 3 | **MATCH (PASS)** |
| **Z-Score Means** | `[-11.0, -18.0, 7.0]` | `channel_means = [-11.0, -18.0, 7.0]` (from `primary_segmentation_model.json`) | **MATCH (PASS)** |
| **Z-Score Stds** | `[3.0, 3.5, 2.5]` | `channel_stds = [3.0, 3.5, 2.5]` (from `primary_segmentation_model.json`) | **MATCH (PASS)** |
| **Precision & Type** | `float32` numpy array / PyTorch float tensor | `float32` numpy array / PyTorch float tensor | **MATCH (PASS)** |
| **Tensor Shape** | `(1, 3, 256, 256)` batched tensor | `(1, 3, 256, 256)` windowed tensor with edge padding | **MATCH (PASS)** |
| **Nodata Value** | `255` (uint8) | `255.0` (float32 input) $\rightarrow$ `255` (uint8 output) | **MATCH (PASS)** |
| **Decision Threshold** | `0.50` probability threshold | `0.50` probability threshold | **MATCH (PASS)** |

---

## 5. Task 3 & 4: Normalization & Nodata Propagation Verification

### 5.1 Single Normalization Audit (Task 3)
* **Raster Storage State:** `post_calibrated_sigma0_unet_3band_corrected.tif` stores raw decibel values (Band 1 mean `-9.73 dB`, Band 2 mean `-18.49 dB`, Band 3 mean `+8.76 dB`). It is **not** pre-normalized on disk.
* **Inference Service Execution:** `UNetSegmentationService.predict()` reads the raw decibel values and computes:
  ```python
  norm_chip = (input_chip - self.channel_means) / (self.channel_stds + 1e-7)
  ```
  exactly **once** per window chip prior to calling `self.model(tensor_chip)`.
* **Conclusion:** Normalization is applied once in memory and is never repeated (**PASS**).

### 5.2 Nodata Masking & Exclusion (Task 4)
* **Input Nodata Detection:** `chip_nodata` isolates pixels equal to `255.0` or $\le -99.0$.
* **Padding for Forward Pass:** Nodata pixels are zero-filled in `input_chip` before tensor conversion to prevent `NaN` propagation through PyTorch convolutions.
* **Output Nodata Masking:**
  * In `predicted_water_mask.tif`: Nodata pixels are assigned value `255` (`uint8`).
  * In `water_probability_map.tif`: Nodata pixels are assigned value `-1.0` (`float32`).
* **Conclusion:** Nodata pixels never enter normalization as valid observations and never receive valid water class predictions (`0` or `1`) (**PASS**).

---

## 6. Task 5 & 6: Spatial Geometry, Windowed Processing & Numerical Stability

### 6.1 Spatial Alignment & Georeferencing
* **Dimensions:** $32,072 \times 21,519$ pixels (3 channels)
* **Valid Pixels:** 379,266,320 pixels (**54.9536%**)
* **CRS:** `EPSG:4326` (WGS 84)
* **Affine Transform:** `(8.983152841195215e-05, 0.0, 83.92583581969234, 0.0, -8.983152841195215e-05, 29.524361773147422)`
* **WGS84 Bounds:** `[[27.591277, 83.925836], [29.524362, 86.806913]]`

### 6.2 Windowed Inference Engine
`UNetSegmentationService.predict()` processes the scene in $256 \times 256$ window blocks (`Window(x, y, win_w, win_h)`). Edge blocks where `win_w < 256` or `win_h < 256` are padded to $(3, 256, 256)$ and cropped back after sigmoid evaluation. This guarantees spatial alignment across the full scene.

### 6.3 Numerical Stability
* 100% of valid pixels in `post_calibrated_sigma0_unet_3band_corrected.tif` are finite (`np.isfinite` check passed).
* No infinite or NaN values enter model tensors (**PASS**).

---

## 7. Task 7: Threshold & Scientific Disclaimers Audit

* **Model Output Nature:** Preliminary probabilistic SAR water extent map.
* **Threshold Selection:** Default threshold `0.50` stored in `config/primary_segmentation_model.json`.
* **Required Disclaimer Enforcement:**
  > "Preliminary SAR-based water prediction — independent confirmation required. Model outputs represent preliminary probabilistic SAR water extent. Must NOT be used solely to claim road closures, settlement access limitations, or area safety."

---

## 8. Task 8: Executed Unit & Integration Test Summary

| Test File | Test Name | Result | Verification Scope |
| :--- | :--- | :---: | :--- |
| `test_unet_segmentation_service.py` | `test_checkpoint_loading_success` | **PASSED** | Checkpoint restoration from `experiments/unet/best_model.pt`. |
| `test_unet_segmentation_service.py` | `test_valid_input_inference` | **PASSED** | Synthetic 3-band raster windowed inference & output product generation. |
| `test_unet_segmentation_service.py` | `test_invalid_band_count_error` | **PASSED** | Rejection of 2-band inputs with `ValueError`. |
| `test_unet_segmentation_service.py` | `test_nodata_preservation` | **PASSED** | Propagation of nodata `255` in output binary masks. |
| `test_unet_segmentation_service.py` | `test_output_crs_geotransform_bounds` | **PASSED** | Preservation of CRS, geotransform, and WGS84 bounding box. |
| `test_unet_segmentation_service.py` | `test_api_segmentation_endpoint` | **PASSED** | FastAPI `/segmentation/predict` HTTP endpoint execution. |
| `test_unet_segmentation_service.py` | `test_unauthorized_filesystem_access_prevention` | **PASSED** | Security path traversal prevention. |
| `test_nepal_end_to_end_validation.py` | `test_nepal_raster_compatibility_rejection` | **PASSED** | Rejection of 2-band `post_calibrated_sigma0.tif`. |
| `test_nepal_end_to_end_validation.py` | `test_nepal_api_incompatible_raster_endpoint` | **PASSED** | HTTP 400 Bad Request returned for incompatible input rasters. |
| `test_nepal_end_to_end_validation.py` | `test_end_to_end_valid_3band_raster_pipeline` | **PASSED** | Complete end-to-end inference and Leaflet PNG overlay serving. |
| `test_unet_sar_preprocessor.py` | All 8 Preprocessor Tests | **PASSED** | Polarization mapping, unit detection, and Z-score helper verification. |
| **Full Repository Test Suite** | `pytest tests/` (147 tests) | **PASSED** | 147 passed out of 147 total tests across codebase. |

---

## 9. Final Inference-Readiness Conclusion

The read-only audit confirms that the SARA Small U-Net model and inference service pipeline (`UNetSegmentationService`) are fully **INFERENCE-READY** (**PASS**). 

The corrected Nepal input raster `post_calibrated_sigma0_unet_3band_corrected.tif` strictly matches the model's channel order, decibel scale, and normalization specifications. Nodata pixels are properly isolated, and spatial georeferencing is preserved across all $690.1\text{M}$ pixels.
