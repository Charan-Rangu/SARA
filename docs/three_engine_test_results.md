# SARA — Three-Engine Test Results & Verification Log
**Automated Regression & Independent Mathematical Test Execution**
**Target System:** Satellite-Assisted Rescue & Assessment (SARA)  
**Execution Environment:** Windows (PowerShell)  
**Virtual Environment:** `.venv/Scripts/python.exe`  
**Test Framework:** `pytest 9.1.1` (with `anyio 4.15.1`)  
**Execution Date:** 2026-10-03

---

## 1. Test Suite Summary

```
============================== test session starts ==============================
platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\chara\OneDrive\Documents\Multimodal AI\sara
plugins: anyio-4.15.1
collected 177 items

======================= 177 passed, 3 warnings in ~75.4s =======================
```

---

## 2. Dedicated Three-Engine Verification Suite Results

**Test File:** [`tests/test_three_engines_verification.py`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/tests/test_three_engines_verification.py)  
**Execution Command:** `.\.venv\Scripts\python.exe -m pytest tests/test_three_engines_verification.py -v`  
**Execution Status:** **11 passed in 0.78s (100% Pass Rate)**

| Test Identifier | Engine | Verified Behavior | Status | Runtime |
| :--- | :--- | :--- | :---: | :---: |
| `test_engine1_identical_rasters` | Engine 1 | Identical pre/post inputs evaluate to exactly 0 changed pixels and 0.0% change. | **`PASSED`** | 0.06s |
| `test_engine1_known_decrease_and_increase` | Engine 1 | $-6.0\text{ dB}$ backscatter drop triggers change; $+6.0\text{ dB}$ increase does not. | **`PASSED`** | 0.07s |
| `test_engine1_exact_threshold_boundary` | Engine 1 | Exact $\Delta\sigma^0 = -3.0000\text{ dB}$ triggers change ($\le -3.0\text{ dB}$ inequality verified). | **`PASSED`** | 0.06s |
| `test_engine1_nan_inf_nodata_exclusion` | Engine 1 | Nodata (-9999.0), NaN, $+\infty$, and $-\infty$ are strictly isolated from valid pixel count. | **`PASSED`** | 0.07s |
| `test_engine1_5class_agreement_logic` | Engine 1 | Hand-constructed synthetic arrays match 5-class categorical assignments (0, 1, 2, 3, 4, 255). | **`PASSED`** | 0.09s |
| `test_engine2_flat_and_inclined_slope` | Engine 2 | Flat plane produces slope $< 10^{-4}$ deg; $45^\circ$ incline produces $45.0^\circ \pm 0.1^\circ$. | **`PASSED`** | 0.08s |
| `test_engine2_nodata_boundary_integrity` | Engine 2 | Nearest-neighbor extrapolation eliminates false $89.9^\circ$ cliffs at nodata borders. | **`PASSED`** | 0.07s |
| `test_engine3_confusion_matrix_and_metrics` | Engine 3 | Evaluates Precision, Recall, F1, IoU, and Accuracy against known hand-calculated counts. | **`PASSED`** | 0.01s |
| `test_engine3_division_by_zero_handling` | Engine 3 | Empty prediction, empty reference, and all-nodata masks produce explicit undefined flags. | **`PASSED`** | 0.01s |
| `test_engine3_quarantine_and_pending_status` | Engine 3 | Intercepts EMSR927 activation data and sets PENDING status when ground truth is absent. | **`PASSED`** | 0.05s |
| `test_engine3_situation_report_provenance_and_limitations` | Engine 3 | Preserves mandatory limitation statement, Sentinel/Copernicus attributions, and run ID. | **`PASSED`** | 0.02s |

---

## 3. Component Regression Test Suites

### 3.1 SAR Change Detection Regression Suite
**Test File:** [`tests/test_change_detection.py`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/tests/test_change_detection.py)  
**Execution Command:** `.\.venv\Scripts\python.exe -m pytest tests/test_change_detection.py -v`  
**Results:** **7 passed in 0.52s**
* `test_change_detection_decibel_success` — **PASSED**
* `test_change_detection_linear_units` — **PASSED**
* `test_crs_mismatch_error` — **PASSED**
* `test_dimension_mismatch_error` — **PASSED**
* `test_transform_mismatch_error` — **PASSED**
* `test_nodata_handling` — **PASSED**
* `test_all_nodata_empty_region` — **PASSED**

### 3.2 Copernicus Terrain & Slope Regression Suite
**Test File:** [`tests/test_terrain_analysis.py`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/tests/test_terrain_analysis.py)  
**Execution Command:** `.\.venv\Scripts\python.exe -m pytest tests/test_terrain_analysis.py -v`  
**Results:** **1 passed in 0.45s**
* `test_terrain_analysis_slope_aspect` — **PASSED**

### 3.3 Situation Report & Validation Regression Suite
**Test File:** [`tests/test_situation_report.py`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/tests/test_situation_report.py), [`tests/test_nepal_end_to_end_validation.py`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/tests/test_nepal_end_to_end_validation.py)  
**Results:** **4 passed in 5.89s**
* `test_situation_report_generation` — **PASSED**
* `test_nepal_raster_compatibility_rejection` — **PASSED**
* `test_nepal_api_incompatible_raster_endpoint` — **PASSED**
* `test_end_to_end_valid_3band_raster_pipeline` — **PASSED**

---

## 4. Independent Mathematical Verification Scripts

### 4.1 Engine 1 Verification Script
* **Path:** `scratch/verify_engine1_independent.py`
* **Output:**
  ```
  === Testing Engine 1 Change Detection Mathematics Independently ===
    [PASS] Identical arrays -> 0% change
    [PASS] Known decrease (-6 dB) -> 100% change
    [PASS] Known increase (+6 dB) -> 0% change
    [CHECK] Exact threshold delta_db == -3.000: changed_pixels = 16 / 16
    [PASS] Above (-2.999 dB -> 0) and below (-3.001 dB -> 1) threshold discrimination
    [PASS] Nodata in pre & post properly excluded with 255 fill
    [CHECK] NaN / Inf handling: total_valid = 1, changed = 1

  === Testing Hand-Constructed 5-Class Agreement Mask Logic ===
    [PASS] Hand-constructed classes matched exactly: [0, 1, 2, 3, 4, 4, 255] == [0, 1, 2, 3, 4, 4, 255]

  === Re-computing Nepal Baseline Outputs from GeoTIFF Files ===
    GeoTIFF Dimensions: 32072 x 21519 = 690,157,368 pixels
    Valid Pixels: 379,266,320
    Nodata Pixels: 310,891,048
    Class Breakdown:
      Class 0: 300,369,237 (79.20%)
      Class 1: 8,823,064 (2.33%)
      Class 2: 46,587,724 (12.28%)
      Class 3: 3,326,431 (0.88%)
      Class 4: 20,159,864 (5.32%)
  ```

### 4.2 Engine 2 Verification Script
* **Path:** `scratch/verify_engine2_independent.py`
* **Output:**
  ```
  === Testing Engine 2 Slope Mathematics Independently ===
    [PASS] Flat surface produced slope = 0.000000 deg (< 1e-4 deg)
    [PASS] 45-degree inclined surface: central slope = 45.0000 deg (expected 45.0 deg)
    [PASS] 10-degree inclined surface: central slope = 10.0000 deg (expected 10.0 deg)
    [PASS] Nodata region preserved as -9999.0 and valid region correctly evaluated

  === Verifying Nepal Slope-Stratified Water Statistics ===
    Total Valid: 379,266,320 | Total Predicted Water: 32,309,359 (8.52%)
      0-2 deg (Flat)      : In-Category Water Rate = 16.90% (expected 16.9%), Share of Water = 15.68% (expected 15.68%)
      2-5 deg (Gentle)    : In-Category Water Rate = 8.96% (expected 8.96%), Share of Water = 6.66% (expected 6.66%)
      5-15 deg (Moderate) : In-Category Water Rate = 6.05% (expected 6.05%), Share of Water = 15.26% (expected 15.26%)
      15-25 deg (Steep)   : In-Category Water Rate = 5.71% (expected 5.71%), Share of Water = 15.30% (expected 15.3%)
      >25 deg (Very Steep): In-Category Water Rate = 9.68% (expected 9.68%), Share of Water = 47.10% (expected 47.1%)
    Water on slope > 15 deg: 20,159,865 (62.40% of total water, expected 62.40%)
    Retained Water (Slope <= 15 deg): 12,149,494 (expected 12,149,494 or 12,149,495)
    Retained Water (Slope <= 10 deg): 9,895,805 (expected 9,895,805)
    Retained Water (Slope <=  5 deg): 7,220,402 (expected 7,220,402)
  ```

### 4.3 Engine 3 Verification Script
* **Path:** `scratch/verify_engine3_independent.py`
* **Output:**
  ```
  === Testing Engine 3 Validation Mathematics Independently ===
    [CHECK] Standard counts:
      Precision: 0.8 (expected 0.8)
      Recall:    0.666667 (expected 0.666667)
      F1 Score:  0.727273 (expected 0.727273)
      IoU:       0.571429 (expected 0.571429)
      Accuracy:  0.85 (expected 0.85)
    [PASS] Standard validation formulas verified.
    [PASS] Empty prediction handles undefined precision without silent 1.0.
    [PASS] Empty reference handles undefined recall safely.
    [PASS] Both empty masks yield 1.0 accuracy and IoU with undefined positive class metrics.
    [PASS] All-nodata input cleanly flagged as UNDEFINED_ZERO_VALID_PIXELS.

  === Testing RasterValidator with Real/Synthetic GeoTIFFs ===
    [PASS] Windowed RasterValidator matched hand-calculated matrix exactly.
    [PASS] Quarantined EMSR927 data explicitly rejected.
    [PASS] Missing ground-truth correctly yields PENDING status.

  === Testing Situation Report Data Provenance ===
    [PASS] Situation report format, caveats, disclaimers, and attributions verified.
  ```

---

## 5. Verification Commands for Host Execution

To reproduce all test and benchmark results on the host system:

```powershell
# 1. Activate environment and run full test suite (177 tests)
.\.venv\Scripts\python.exe -m pytest tests/ -v

# 2. Run focused Three-Engine verification suite
.\.venv\Scripts\python.exe -m pytest tests/test_three_engines_verification.py -v

# 3. Run individual mathematical verification scripts
.\.venv\Scripts\python.exe scratch/verify_engine1_independent.py
.\.venv\Scripts\python.exe scratch/verify_engine2_independent.py
.\.venv\Scripts\python.exe scratch/verify_engine3_independent.py

# 4. Launch backend and dashboard server
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
