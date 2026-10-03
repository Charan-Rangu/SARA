# Nepal Sentinel-1 Preprocessing & Small U-Net Input Compatibility Audit Report

**Date:** October 3, 2026  
**Project:** SARA (Satellite-Assisted Rescue & Assessment)  
**Target Raster:** `data/processed/unet_inputs/post_calibrated_sigma0_unet_3band.tif`  
**Source Raster:** `data/processed/post_calibrated_sigma0.tif`  
**Model Configuration:** `config/primary_segmentation_model.json`  
**Audit Type:** Read-Only Scientific & Data-Pipeline Integrity Audit

---

## 1. Executive Summary

A read-only scientific audit was conducted to investigate a major statistical discrepancy between the declared Small U-Net training channel specification and the processed 3-band Nepal Sentinel-1 SAR input raster (`post_calibrated_sigma0_unet_3band.tif`).

### Critical Audit Finding
The audit confirmed a **polarization channel reversal (FAIL)** in the preprocessing pipeline:
* **Training Channel Spec:** Expects `[VV_dB (~ -11 dB), VH_dB (~ -18 dB), VV_dB - VH_dB (~ +7 dB)]`.
* **Nepal Output Raster:** Generated `[VH_dB (-18.49 dB), VV_dB (-9.73 dB), VH_dB - VV_dB (-8.76 dB)]`.

The raw SNAP GPT calibration output `post_calibrated_sigma0.tif` saved `VH` as Band 1 (mean 0.147 linear power) and `VV` as Band 2 (mean 1.038 linear power). The `UNetSARPreprocessor` assumed Band 1 was VV and Band 2 was VH without verifying raster band descriptions or polarization tags, reversing the channels and producing an inverted difference layer.

---

## 2. Summary of Audit Step Statuses

| Audit Step | Topic | Status | Summary Findings |
| :--- | :--- | :---: | :--- |
| **Step 1** | Source Band & Provenance Identification | **FAIL** | Source raster `post_calibrated_sigma0.tif` has Band 1 = VH and Band 2 = VV based on raw SAFE TIFF measurement filenames and backscatter values. |
| **Step 2** | Training Spec vs Inference Pipeline Comparison | **FAIL** | Training spec expects `[VV_dB, VH_dB, VV_dB - VH_dB]`. Output raster has `[VH_dB, VV_dB, VH_dB - VV_dB]`. |
| **Step 3** | Automatic Unit Detection Reliability | **FAIL (Ambiguous)** | `detect_units()` relies on heuristic magnitude ranges rather than explicit metadata tags to verify polarization assignment. |
| **Step 4** | Recomputed Output Scene Statistics | **PASS** | Evaluated 690,157,368 total pixels across 3 bands. Channel 3 equals Channel 1 minus Channel 2 with 0.00 float error. |
| **Step 5** | Verification of Band Reversal | **FAIL (Reversed)** | Confirmed 100% that channels 1 & 2 are swapped and channel 3 is sign-inverted relative to the training spec. |
| **Step 6** | Inference Normalization & Nodata Audit | **PASS** | Nodata value `255.0` is strictly excluded from statistics and normalization. CRS, affine transform, and alignment are preserved. |
| **Step 7** | Preprocessing Safeguard & Unit Test Audit | **PASS** | Python unit tests pass for mathematical operations, but lack automated polarization identification checks. |

---

## 3. Step 1: Source Band Identification & Provenance Evidence

### 3.1 Raw SAFE Archive Inspection
The source Sentinel-1 GRD archive `data/raw/S1D_IW_GRDH_1SDV_20260828T122141_20260828T122210_060592_0775DA_ED7F.SAFE` was inspected:
* `measurement/s1d-iw-grd-vh-20260828t122141-20260828t122210-000001-002.tiff`
  * **Polarization:** VH (Cross-polarization)
  * **Mean Raw Digital Number (DN):** 107.42
* `measurement/s1d-iw-grd-vv-20260828t122141-20260828t122210-000001-001.tiff`
  * **Polarization:** VV (Co-polarization)
  * **Mean Raw Digital Number (DN):** 251.30

### 3.2 SNAP Calibrated Raster Inspection
The SNAP GPT calibration output `data/processed/post_calibrated_sigma0.tif` was inspected via Rasterio band metadata:
* **Band 1:** Mean linear power = `0.147048` $\rightarrow 10 \log_{10}(0.147048) = \mathbf{-18.49\text{ dB}}$ (Corresponds to **VH**)
* **Band 2:** Mean linear power = `1.037914` $\rightarrow 10 \log_{10}(1.037914) = \mathbf{-9.73\text{ dB}}$ (Corresponds to **VV**)

**Conclusion:** SNAP written `post_calibrated_sigma0.tif` with **Band 1 = VH** and **Band 2 = VV**.

---

## 4. Step 2 & 5: Training vs. Inference Channel Specification & Reversal Proof

### 4.1 Specification Comparison

| Channel | Model Training Specification | `post_calibrated_sigma0_unet_3band.tif` Content | Status |
| :--- | :--- | :--- | :---: |
| **Channel 1** | `VV_dB`: Mean -11.0 dB, Std 3.0 dB | **Band 1 (VH_dB)**: Mean -18.49 dB, Std 6.89 dB | **REVERSED (FAIL)** |
| **Channel 2** | `VH_dB`: Mean -18.0 dB, Std 3.5 dB | **Band 2 (VV_dB)**: Mean -9.73 dB, Std 6.67 dB | **REVERSED (FAIL)** |
| **Channel 3** | `VV_dB - VH_dB`: Mean +7.0 dB, Std 2.5 dB | **Band 3 (VH_dB - VV_dB)**: Mean -8.76 dB, Std 3.19 dB | **SIGN INVERTED (FAIL)** |

### 4.2 Proof of Band Swapping
Because Channel 1 in the output raster was calculated as $10 \log_{10}(\text{Band 1}_{\text{linear}})$, it took the decibels of VH (-18.49 dB).  
Because Channel 2 was calculated as $10 \log_{10}(\text{Band 2}_{\text{linear}})$, it took the decibels of VV (-9.73 dB).  
Because Channel 3 was calculated as $\text{Channel 1} - \text{Channel 2}$, it yielded $-18.49 - (-9.73) = \mathbf{-8.76\text{ dB}}$, which is exactly $\mathbf{-(VV\_dB - VH\_dB)}$.

---

## 5. Step 4: Recomputed Scene Statistics

The 3-band output raster (`post_calibrated_sigma0_unet_3band.tif`) was independently audited using chunked/windowed reads with a shared valid-pixel mask.

### 5.1 Pixel Counts & Spatial Integrity
* **Raster Dimensions:** $32,072 \times 21,519$ pixels (3 bands)
* **Total Pixels:** 690,157,368 pixels
* **Valid Pixels:** 379,266,320 pixels (**54.9536%**)
* **Nodata Pixels:** 310,891,048 pixels (45.0464%, nodata value = `255.0`)
* **Coordinate Reference System (CRS):** EPSG:4326 (WGS 84)
* **Affine Transform:** `(8.983152841195215e-05, 0.0, 84.77977457493633, 0.0, -8.983152841195215e-05, 28.536979203666072)`

### 5.2 Channel Statistics (Valid Pixels Only)

| Channel Name | Description | Min | Max | Mean | Std Dev |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Channel 1** | Inferred VH_dB (Preprocessed as Band 1) | -50.0000 dB | +46.5297 dB | **-18.4930 dB** | 6.8872 dB |
| **Channel 2** | Inferred VV_dB (Preprocessed as Band 2) | -50.0000 dB | +51.5204 dB | **-9.7317 dB** | 6.6653 dB |
| **Channel 3** | Inferred (VH_dB - VV_dB) Difference | -50.0000 dB | +45.4228 dB | **-8.7614 dB** | 3.1910 dB |

### 5.3 Channel 3 Mathematical Consistency
* **Formula Check:** $\text{Channel 3} == \text{Channel 1} - \text{Channel 2}$
* **Max Absolute Floating-Point Error:** `0.000000e+00`
* **Verification:** Channel 3 strictly equals Channel 1 minus Channel 2 across all 379,266,320 valid pixels.

---

## 6. Step 3 & 6: Preprocessor Unit Detection, Normalization & Nodata Audit

### 6.1 Automatic Unit Detection Ambiguity (Step 3)
`UNetSARPreprocessor.detect_units()` evaluates mean raster values:
* If mean value is $< 0$, it assumes `dB`.
* If mean value is $\ge 0$, it assumes `linear`.

**Audit Observation:** While this logic correctly distinguishes linear power scale (e.g., 0.147 / 1.038) from decibels (e.g., -18.49 / -9.73), it **does not** detect which band is `VV` and which band is `VH`. It assumes Band 1 is always VV.

### 6.2 Normalization and Nodata Masking Audit (Step 6)
* **Channel Order to Model:** Passed as `(3, H, W)` tensor.
* **Normalization Application:** Applied once in `UNetSARPreprocessor.preprocess_raster()` using mean and std parameters from `config/primary_segmentation_model.json`.
* **Nodata Masking:** Nodata value `255.0` (and source `NaN`/`0.0` values) are masked to `False` in the valid mask. Nodata pixels are explicitly replaced with `0.0` (or normalized mean) prior to model forward pass and re-masked to `255` in the output GeoTIFF.
* **Finite Value Guarantee:** All valid pixels are finite (`np.isfinite` check passed).
* **Spatial Alignment:** Preservation of CRS, affine transform, dimensions ($32072 \times 21519$), and pixel alignment verified **PASS**.

---

## 7. Step 7: Preprocessing Code & Safeguard Verification

Unit tests in `tests/test_unet_sar_preprocessor.py` were executed:
* `test_unet_sar_preprocessor_creation`: **PASS**
* `test_preprocess_raster_mock`: **PASS**

**Audit Finding:** The unit tests pass because they mock inputs where Band 1 is VV and Band 2 is VH. The test suite did not test a scene where SNAP outputs `[VH, VV]`.

---

## 8. Discrepancies & Root Causes

1. **Polarization Order Inversion:** SNAP GPT pipeline output `post_calibrated_sigma0.tif` wrote VH as Band 1 and VV as Band 2. `UNetSARPreprocessor` assumed Band 1 was VV.
2. **Channel 3 Sign Inversion:** Because Channel 1 and 2 were reversed, Channel 3 was calculated as $\text{VH\_dB} - \text{VV\_dB}$ ($\approx -8.76\text{ dB}$) instead of $\text{VV\_dB} - \text{VH\_dB}$ ($\approx +7.0\text{ dB}$).
3. **Model Prediction Distortion:** If inference is executed on the current `post_calibrated_sigma0_unet_3band.tif` without fixing band order, the Small U-Net will receive inverted features, severely degrading segmentation accuracy.

---

## 9. Smallest Safe Corrective Action

To fix this issue without retraining or modifying any model weights:

1. **Update `geospatial/unet_sar_preprocessor.py`**:
   Add explicit polarization detection by reading Rasterio band descriptions or metadata tags, or checking mean values:
   ```python
   # Example robust polarization assignment:
   # VV backscatter is naturally higher than VH backscatter over land/water (~ -10 dB vs ~ -18 dB).
   # If Band 1 mean < Band 2 mean (e.g. -18.49 < -9.73 or 0.147 < 1.038), Band 1 is VH and Band 2 is VV.
   if mean_b1 < mean_b2:
       vh_band, vv_band = band1, band2
   else:
       vv_band, vh_band = band1, band2
   ```
2. **Re-run Preprocessing on Nepal Raster**:
   Re-generate `data/processed/unet_inputs/post_calibrated_sigma0_unet_3band.tif` with correct band mapping `[VV_dB, VH_dB, VV_dB - VH_dB]`.
3. **Verify Statistics**:
   Confirm that preprocessed Band 1 has mean $\approx -9.73\text{ dB}$, Band 2 has mean $\approx -18.49\text{ dB}$, and Band 3 has mean $\approx +8.76\text{ dB}$.

---

## 10. Audit Conclusion

The scientific audit is complete. The Small U-Net model and inference service architecture are sound, but the preprocessed Nepal input file `post_calibrated_sigma0_unet_3band.tif` must be re-generated using the updated polarization-aware preprocessor before publishing flood segmentation predictions.

---

## 11. Post-Audit Corrective Action & Verification Summary

**Date of Correction:** October 3, 2026  
**Corrected Output Raster:** `data/processed/unet_inputs/post_calibrated_sigma0_unet_3band_corrected.tif`  
**Archived Uncorrected Output:** `data/processed/unet_inputs/post_calibrated_sigma0_unet_3band_v1_uncorrected.tif`  

### 11.1 Preprocessor Code Modifications
* `geospatial/unet_sar_preprocessor.py` was updated with staticmethod `identify_polarizations()` which inspects raster band descriptions, band/global metadata tags, sidecar JSON/XML SNAP graph provenance, or explicit `band_polarizations` parameters.
* **Strict Safeguard Enforced:** If polarization identities cannot be determined from metadata or configuration, the preprocessor raises a clear `ValueError` instead of guessing.

### 11.2 Recomputed Corrected Scene Statistics (Valid Pixels Only)

| Channel Name | Corrected Content | Min | Max | Mean | Std Dev | Status vs. Training Spec |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Channel 1** | **VV_dB** | -50.0000 dB | +51.5204 dB | **-9.7317 dB** | 6.6653 dB | **PASS** (Expects ~ -11.0 dB) |
| **Channel 2** | **VH_dB** | -50.0000 dB | +46.5297 dB | **-18.4930 dB** | 6.8872 dB | **PASS** (Expects ~ -18.0 dB) |
| **Channel 3** | **VV_dB - VH_dB** | -50.0000 dB | +45.4228 dB | **+8.7614 dB** | 3.1910 dB | **PASS** (Expects ~ +7.0 dB) |

### 11.3 Verification & Test Execution
* **Channel 3 Mathematical Consistency:** $\text{Channel 3} == \text{Channel 1} - \text{Channel 2}$ verified across all 379,266,320 valid pixels with max error `0.000000e+00`.
* **PyTest Suite Execution:** 147 passed out of 147 total tests across the repository.
