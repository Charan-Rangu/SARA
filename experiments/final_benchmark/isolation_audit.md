# SARA Nepal Track B — Artifact & Dataset Isolation Audit Report

**Project:** SARA (Satellite-Assisted Rescue & Assessment)  
**Audit Date:** October 3, 2026  
**Status:** **`AUDIT PASSED — STRICT EVENT & SPATIAL ISOLATION VERIFIED`**

---

## 1. Dataset Partition Isolation Matrix

Strict event-level and spatial separation across the 44 total dataset chips:

| Partition | Event IDs | Tile Count | Access Rules & Purpose | Leakage Status |
| :--- | :--- | :---: | :--- | :---: |
| **Train Partition** | `Bolivia`, `Ghana`, `India`, `Nigeria`, `Pakistan`, `Paraguay`, `Spain`, `SriLanka`, `USA` | 36 | Model fitting & gradient updates only | **ISOLATED** |
| **Validation Partition** | `Somalia` | 4 | Early stopping (`val_iou`) & Somalia baseline comparison | **ISOLATED** |
| **Test Partition** | `Mekong` | 4 | Final benchmark evaluation only (read-only, no tuning) | **ISOLATED** |

---

## 2. Saved Artifact Inventory & Complete Ensemble Size

All four model checkpoints were verified in their original experiment directories:

| Model Name | Artifact File Path | File Size (Bytes) | Human-Readable Size | Ensemble Component Description |
| :--- | :--- | :---: | :---: | :--- |
| **Random Forest Baseline** | `experiments/rf_baseline/rf_baseline_model.joblib` | 1,074,921 | 1.07 MB | 100 Trees (Pixel-level feature classifier) |
| **XGBoost Baseline** | `experiments/xgboost_baseline/xgb_baseline_model.json` | 210,688 | 0.21 MB | 100 Booster Trees (Gradient boosting) |
| **Stacking Ensemble** | `experiments/stacking/stacking_model.joblib` | 1,250,118 | **1.25 MB** | **Full Package:** Base RF + Base XGB + Logistic Regression Meta-Learner |
| **Small U-Net** | `experiments/unet/best_model.pt` | 23,212,754 | **23.21 MB** | PyTorch Deep Learning Model (1,927,009 parameters) |

> [!NOTE]
> **Complete Stacking Package Verification:**
> The stacking artifact `stacking_model.joblib` (1.25 MB) includes both base models (RF and XGBoost) along with out-of-fold meta-model features, ensuring self-contained inference.

---

## 3. Ground Truth Integrity & Third-Party Isolation

* **EMSR927 Validation Assets:** EMSR927 activation data remains strictly isolated under `validation/` and was not accessed by any model training pipeline.
* **Nepal Change Mask:** `data/processed/preliminary_change_mask.tif` was NOT used as training or evaluation ground truth.
* **Label Convention & Nodata:** Verified uniform label encoding: `0 = Non-water`, `1 = Water`, `255 = Nodata/Cloud`. Nodata pixels are strictly excluded from loss, metric calculation, and precision-recall accounting across all 4 models.

---

## 4. Audit Conclusion

> [!IMPORTANT]
> **ISOLATION AUDIT PASSED**
>
> Zero data leakage detected across train, validation, and test splits. Test evaluation on Mekong chips may proceed under frozen evaluation protocols.
