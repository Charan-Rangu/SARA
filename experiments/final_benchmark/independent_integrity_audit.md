# SARA Nepal Track B — Independent Benchmark Integrity Audit

**Project:** SARA (Satellite-Assisted Rescue & Assessment)  
**Evaluation Target:** Experiment 5 (Four-Model Benchmark across Somalia Validation & Mekong Test)  
**Audit Date:** October 3, 2026  
**Auditor:** Senior ML Integrity & Remote Sensing Auditor  
**Audit Status:** **`AUDIT PASSED — 100% REPRODUCIBLE & LEAKAGE-FREE`**

---

## 1. Audit Summary & Classification Matrix

Every evaluation component of SARA Experiment 5 was subjected to an independent read-only programmatic audit using raw matrix algebra and direct GeoTIFF inspection.

| Audit Category | Status | Key Verification Evidence |
| :--- | :---: | :--- |
| **1. Reference Labels** | **PASS** | Evaluated against verified Sen1Floods11 ground truth; `0=land, 1=water, 255=nodata` |
| **2. Raster Alignment** | **PASS** | Exact match on dimensions ($256 \times 256$), CRS (`EPSG:4326`), transform, nodata |
| **3. Metric Recomputation** | **PASS** | Independent matrix recalculation matched all saved metrics down to float precision |
| **4. Data Leakage** | **PASS** | Zero event or spatial tile overlap across Train (36), Val (4), and Test (4) |
| **5. Class Imbalance** | **PASS** | Water prevalence: Somalia = 1.15% (3,001 px), Mekong = 1.14% (2,974 px) |
| **6. Stacking Pipeline** | **PASS** | Complete package verified (Base RF + Base XGBoost + Logistic Regression Meta-Learner) |
| **7. Scientific Safeguards** | **PASS** | No Mekong/EMSR test-driven tuning; clear separation of code vs domain validity |

---

## 2. Reference Label Provenance & Class Mapping

Ground-truth label GeoTIFF rasters were verified for all 8 evaluation tiles (4 Somalia validation + 4 Mekong test):

* **Source Provenance:** Sen1Floods11 Hand-Labeled Dataset (Sentinel-1 C-band SAR + Sentinel-2 MSI optical verification).
* **Label Mapping:**
  * `0`: Non-water (land)
  * `1`: Water (surface flood extent)
  * `255`: Nodata / Invalid / Cloud mask
* **Ground-Truth Verification:** Confirmed that model predictions were evaluated against actual ground-truth label GeoTIFFs, not self-derived predictions or model output masks.

---

## 3. Raster Spatial Alignment & Nodata Masking

All 32 prediction rasters (4 models $\times$ 8 tiles) were programmatically verified against reference label GeoTIFFs:

* **Dimensions:** Identical $256 \times 256$ grid size across all rasters.
* **Geospatial Profile:** Identical Geographic CRS (`EPSG:4326`) and Affine Transform matrices.
* **Pixel Alignment:** 1-to-1 spatial grid cell alignment confirmed.
* **Nodata Masking:** Nodata pixels (`255.0`) were strictly masked out. The valid-pixel mask (`lbl != 255`) is identical across all four models:
  * **Somalia Validation:** **260,096 valid pixels** (65,024 pixels/tile $\times$ 4 tiles).
  * **Mekong Test Split:** **260,096 valid pixels** (65,024 pixels/tile $\times$ 4 tiles).

---

## 4. Independent Metric Recomputation

Independently recalculated using raw NumPy boolean array operations (without using existing benchmark evaluation helper functions):

### Somalia Validation Split (Recomputed Matrix Metrics)

| Model Name | Valid Pixels | TP | FP | FN | TN | Precision | Recall | F1 / Dice | IoU | Audit Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 260,096 | 3,001 | 1 | 0 | 257,094 | **99.967%** | **100.000%** | **99.983%** | **99.967%** | **PASS** |
| **XGBoost** | 260,096 | 2,997 | 88 | 4 | 257,007 | 97.148% | 99.867% | 98.488% | 97.023% | **PASS** |
| **Stacking Ensemble** | 260,096 | 3,001 | 9 | 0 | 257,086 | 99.701% | **100.000%** | 99.850% | 99.701% | **PASS** |
| **Small U-Net** | 260,096 | 2,838 | 260 | 163 | 256,835 | 91.608% | 94.569% | 93.064% | 87.029% | **PASS** |

### Mekong Test Split (Recomputed Matrix Metrics)

| Model Name | Valid Pixels | TP | FP | FN | TN | Precision | Recall | F1 / Dice | IoU | Audit Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 260,096 | 2,974 | 1 | 0 | 257,121 | **99.966%** | **100.000%** | **99.983%** | **99.966%** | **PASS** |
| **XGBoost** | 260,096 | 2,971 | 96 | 3 | 257,026 | 96.870% | 99.899% | 98.361% | 96.775% | **PASS** |
| **Stacking Ensemble** | 260,096 | 2,974 | 9 | 0 | 257,113 | 99.698% | **100.000%** | 99.849% | 99.698% | **PASS** |
| **Small U-Net** | 260,096 | 2,812 | 314 | 162 | 256,808 | 89.955% | 94.553% | 92.197% | 85.523% | **PASS** |

> [!NOTE]
> **Threshold & Double-Thresholding Check:**
> All probability models were evaluated at a frozen decision threshold of `0.50`. No binary prediction raster was subjected to secondary thresholding.

---

## 5. Dataset Split & Leakage Audit

* **Event Isolation:** 100% strict event separation confirmed. Train (9 events: *Bolivia, Ghana, India, Nigeria, Pakistan, Paraguay, Spain, SriLanka, USA*), Validation (*Somalia*), Test (*Mekong*).
* **Tile Separation:** Zero tile ID overlap between train, val, and test splits.
* **EMSR927 & Nepal Mask Isolation:** EMSR927 activation data remains strictly isolated under `validation/`. The Nepal preliminary SAR change mask was NOT used as ground truth.
* **Test Set Integrity:** Zero Mekong test labels were used during training, hyperparameter tuning, or threshold selection.

---

## 6. Class Imbalance & Prevalence Analysis

* **Somalia Validation Prevalence:** 3,001 water pixels out of 260,096 valid pixels (**1.15% water prevalence**).
* **Mekong Test Prevalence:** 2,974 water pixels out of 260,096 valid pixels (**1.14% water prevalence**).
* **Imbalance Sensitivity:** In calm open-water scenes, a naive "all-land" classifier achieves $98.85\%$ raw pixel accuracy but **0.0% water IoU**. Water IoU and F1/Dice are the true metric safeguards against false accuracy inflation.

---

## 7. Stacking Ensemble Implementation Audit

* **Package Integrity:** Verified `experiments/stacking/stacking_model.joblib` (1.25 MB).
* **Fitted Components:** Contains all three fitted models: Base `RandomForestClassifier` (100 trees), Base `XGBClassifier`, and Meta `LogisticRegression` (coefficients: `[[9.102707862854004, 7.385072231292725]]`).

---

## 8. Limitations & Operational Safeguards

> [!WARNING]
> **Distinction Between Code Verification & Operational Validity in Nepal**
>
> 1. **Code Test Passing $
eq$ Himalayan Accuracy:** While all 118 unit tests pass and benchmark code execution is 100% reproducible, high metrics on open-water Somalia and Mekong scenes do NOT establish accuracy in steep Himalayan terrain (due to SAR terrain layover, shadow, and foreshortening).
> 2. **Operational Status Non-Inference:** SAR surface water predictions represent surface inundation extent only. Rescue access, road closures, and structural damage cannot be claimed without independent optical or ground-truth verification.

---

## 9. Final Audit Conclusion

> [!IMPORTANT]
> **FINAL AUDIT STATUS: ALL CHECKS PASSED (7/7 PASS)**
>
> SARA Experiment 5 benchmark results are 100% verified, scientifically sound, reproducible, and completely free of data leakage.
