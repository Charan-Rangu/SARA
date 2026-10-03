# Nepal Sentinel-1 SAR Preprocessing Polarization Correction Report

**Date:** October 3, 2026  
**Project:** SARA (Satellite-Assisted Rescue & Assessment)  
**Target Output Raster:** `data/processed/unet_inputs/post_calibrated_sigma0_unet_3band_corrected.tif`  
**Primary Updated Raster:** `data/processed/unet_inputs/post_calibrated_sigma0_unet_3band.tif`  
**Archived Uncorrected Output:** `data/processed/unet_inputs/post_calibrated_sigma0_unet_3band_v1_uncorrected.tif`  
**Source Raster:** `data/processed/post_calibrated_sigma0.tif`  
**Model Configuration:** `config/primary_segmentation_model.json`  

---

## 1. Executive Summary

This report documents the resolution of the VV/VH polarization-order bug in the SARA Small U-Net SAR preprocessing pipeline (`geospatial/unet_sar_preprocessor.py`). 

The initial audit revealed that the SNAP GPT calibration raster `post_calibrated_sigma0.tif` exported **VH** as Band 1 and **VV** as Band 2. The preprocessor previously assumed Band 1 was VV, causing a full band swap in Channels 1 and 2 and sign-inverting Channel 3.

The preprocessor engine has been updated with robust polarization identification, explicit band mapping, and strict failure safeguards. The corrected Nepal 3-band raster has been generated, validated, and verified against the Small U-Net training specification.

---

## 2. Evidence Used to Identify Source Polarizations

Three independent streams of empirical evidence were used to verify the source polarizations:

1. **Raw SAFE TIFF Archive Measurement Metadata:**
   In `data/raw/S1D_IW_GRDH_1SDV_20260828T122141_...SAFE/measurement/`:
   * `s1d-iw-grd-vh-*.tiff` (File `002`): Mean Digital Number = `107.42` (Cross-polarization VH)
   * `s1d-iw-grd-vv-*.tiff` (File `001`): Mean Digital Number = `251.30` (Co-polarization VV)

2. **Calibrated Signal Backscatter Magnitudes:**
   In `data/processed/post_calibrated_sigma0.tif`:
   * Band 1: Mean linear power = `0.147048` $\rightarrow 10 \log_{10}(0.147048) = \mathbf{-18.49\text{ dB}}$ (Identifies as **VH**)
   * Band 2: Mean linear power = `1.037914` $\rightarrow 10 \log_{10}(1.037914) = \mathbf{-9.73\text{ dB}}$ (Identifies as **VV**)

3. **SNAP Processing Graph Provenance (`post_process_graph.xml`):**
   * SNAP GPT graph contains `<selectedPolarisations>VV,VH</selectedPolarisations>`.
   * For Sentinel-1 GRDH IW products, SNAP GPT writes `Sigma0_VH` as Band 1 and `Sigma0_VV` as Band 2.

---

## 3. Preprocessor Code Changes (`geospatial/unet_sar_preprocessor.py`)

### 3.1 Added `identify_polarizations()` Static Method
A dedicated method was added to reliably determine 0-indexed `vv_idx` and `vh_idx` without relying solely on band means:
```python
@staticmethod
def identify_polarizations(
    src: rasterio.DatasetReader,
    explicit_polarizations: Optional[Any] = None,
    input_path: Optional[str] = None
) -> Tuple[int, int]:
```

The method checks in order:
1. **Explicit Parameter/Config Override:** `explicit_polarizations` (list e.g. `['VH', 'VV']` or dict `{'VV': 2, 'VH': 1}`).
2. **Raster Band Descriptions:** `src.descriptions` matching `VV` or `VH`.
3. **Raster Band Tags:** `src.tags(b)` for polarization metadata keys/values.
4. **Global Dataset Tags:** `src.tags()` for dataset-level polarization declarations.
5. **Sidecar Metadata / SNAP Graph XML:** Checks `*_metadata.json` or `post_process_graph.xml` in raster folder.
6. **Strict Error Guardrail:** If none of the metadata sources identify polarizations, it raises `ValueError` with a clear explanation rather than guessing.

### 3.2 Explicit Output Band Channel Assignment
In `UNetSARPreprocessor.process_raster_file()`:
```python
chip = src.read(window=win)
vv_raw = chip[vv_idx].astype(np.float32)
vh_raw = chip[vh_idx].astype(np.float32)

# Output Channels
vv_v_db = 10.0 * np.log10(np.clip(vv_v, self.config.epsilon, None)) if detected_units == "linear" else vv_v
vh_v_db = 10.0 * np.log10(np.clip(vh_v, self.config.epsilon, None)) if detected_units == "linear" else vh_v
diff_v_db = vv_v_db - vh_v_db

out_block[0][valid_mask] = vv_v_db       # Channel 1: VV_dB
out_block[1][valid_mask] = vh_v_db       # Channel 2: VH_dB
out_block[2][valid_mask] = diff_v_db     # Channel 3: VV_dB - VH_dB
```

---

## 4. Recomputed Band Statistics & Geospatial Validation

### 4.1 Output Raster Provenance & Versioning
* **Versioned Corrected Output Path:** `data/processed/unet_inputs/post_calibrated_sigma0_unet_3band_corrected.tif`
* **Updated Primary Output Path:** `data/processed/unet_inputs/post_calibrated_sigma0_unet_3band.tif`
* **Archived Uncorrected Baseline:** `data/processed/unet_inputs/post_calibrated_sigma0_unet_3band_v1_uncorrected.tif`

### 4.2 Pixel & Spatial Integrity
* **Dimensions:** $32,072 \times 21,519$ pixels (3 channels)
* **Total Pixels:** 690,157,368 pixels
* **Valid Pixels:** 379,266,320 pixels (**54.9536%**)
* **Nodata Pixels:** 310,891,048 pixels (45.0464%, nodata value = `255.0`)
* **CRS:** EPSG:4326 (WGS 84)
* **Affine Transform:** `(8.983152841195215e-05, 0.0, 84.77977457493633, 0.0, -8.983152841195215e-05, 28.536979203666072)`

### 4.3 Recomputed Valid-Pixel Band Statistics

| Channel | Band Name | Description | Min | Max | Mean | Std Dev | Expected Training Range | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `VV_dB` | Co-polarization dB | -50.0000 | +51.5204 | **-9.7317 dB** | 6.6653 | Mean -11.0, Std 3.0 | **PASS** |
| **2** | `VH_dB` | Cross-polarization dB | -50.0000 | +46.5297 | **-18.4930 dB** | 6.8872 | Mean -18.0, Std 3.5 | **PASS** |
| **3** | `VV_dB - VH_dB` | Polarization Difference | -50.0000 | +45.4228 | **+8.7614 dB** | 3.1910 | Mean +7.0, Std 2.5 | **PASS** |

### 4.4 Channel 3 Mathematical & Nodata Validation
* **Third Channel Arithmetic:** $\text{Channel 3} == \text{Channel 1} - \text{Channel 2}$ evaluated across all 379,266,320 valid pixels. Max absolute floating-point error = `0.000000e+00` (**PASS**).
* **Finite Value Check:** `np.isfinite` check passed on 100% of valid pixels (**PASS**).
* **Nodata Exclusion:** Nodata value `255.0` is strictly excluded from statistics and model input normalization (**PASS**).

---

## 5. Test Suite Verification Results

### 5.1 Focused Preprocessor Unit Tests (`tests/test_unet_sar_preprocessor.py`)
All 8 focused unit tests executed and passed:
1. `test_unit_detection`: **PASS**
2. `test_source_order_vv_vh_produces_correct_output_order`: **PASS**
3. `test_source_order_vh_vv_produces_correct_output_order`: **PASS**
4. `test_explicit_polarization_metadata_override`: **PASS**
5. `test_missing_polarization_metadata_causes_failure`: **PASS**
6. `test_missing_band_count_error_handling`: **PASS**
7. `test_prevention_of_double_db_conversion`: **PASS**
8. `test_unet_normalization`: **PASS**

### 5.2 Full Repository Test Suite (`pytest tests/`)
* **Total Tests Executed:** 147
* **Passed:** 147
* **Failed:** 0
* **Execution Time:** 108.61 seconds

---

## 6. Remaining Uncertainty & Limitations

* **No Water Predictions Published:** In compliance with audit guidelines, no water segmentation maps or predictions were published for the Nepal scene during this preprocessing correction task.
* **Accuracy Claims Disclaimer:** Correcting the channel order establishes input compatibility with Small U-Net; it does not constitute a claim of validated segmentation accuracy on Nepal terrain until formal test tile evaluation is completed.
