# SARA Primary Model Adoption Report — Small U-Net (Experiment 4)

**Project:** SARA (Satellite-Assisted Rescue & Assessment)  
**Primary Model:** Small U-Net (`SmallUNet`, PyTorch 2.6.0+cu124)  
**Checkpoint Path:** `experiments/unet/best_model.pt`  
**Configuration File:** `config/primary_segmentation_model.json`  
**Date:** October 3, 2026  
**Status:** **`PRIMARY DEEP LEARNING MODEL ADOPTED & VERIFIED`**

---

## 1. Artifacts Verified

All 5 core Small U-Net artifacts from Experiment 4 were inspected, verified, and validated for adoption without re-training, threshold tuning, or modifying existing experiment files:

| Artifact Path | Size / Details | Verification Status | Key Attributes Verified |
| :--- | :--- | :---: | :--- |
| **`experiments/unet/best_model.pt`** | 23,212,754 bytes | **VERIFIED** | 1,927,009 trainable parameters; saved state dict at Epoch 24 |
| **`experiments/unet/unet_config.json`** | 1,161 bytes | **VERIFIED** | `in_channels=3`, `init_features=32`, `batch_size=2`, `seed=42`, AMP FP16 |
| **`experiments/unet/unet_metrics.json`** | 2,660 bytes | **VERIFIED** | 260,096 valid Somalia val pixels; F1 = 93.06%, IoU = 87.03% |
| **`experiments/unet/training_history.csv`** | 1,988 bytes | **VERIFIED** | 30 completed training epochs; peak VRAM 0.22 GB; 46.34 s duration |
| **`experiments/unet/sara_unet_independent_audit.md`** | 5,299 bytes | **VERIFIED** | 10/10 independent audit checks PASSED with zero data leakage |

### Model Architecture & Parameter Verification
* **Class Name:** `SmallUNet` ([`ml/model.py`](file:///C:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/ml/model.py))
* **Parameter Count:** **1,927,009** (1.93 million parameters)
* **Input Band Structure:** 3 channels:
  1. `VV_dB` (Sentinel-1 VV polarization backscatter in decibels)
  2. `VH_dB` (Sentinel-1 VH polarization backscatter in decibels)
  3. `VV_dB - VH_dB` (Cross-polarization difference / log-ratio in dB)
* **Channel Normalization:** Z-score normalization using dataset statistics:
  * Means: `[-11.0, -18.0, 7.0]`
  * Stds: `[3.0, 3.5, 2.5]`
* **Output Specification:** Single-channel probability map `(B, 1, 256, 256)` via sigmoid activation. Default decision threshold: **0.50**.
* **Nodata Handling:** Value **`255`** ignored during loss computation (`MaskedBCEDiceLoss`), evaluation metrics, and raster output metadata.

---

## 2. Configuration Created & Updated

1. **New Primary Model Configuration Created:**
   * Path: [`config/primary_segmentation_model.json`](file:///C:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/config/primary_segmentation_model.json)
   * Formatted using SARA configuration standards.
   * Formally specifies Small U-Net as the primary deep-learning preliminary flood segmentation model.
   * Explicitly documents Somalia validation metrics, input band definitions, decision threshold, nodata value, and scientific disclaimers.

2. **ML Path Configuration Updated:**
   * Updated [`ml/config.py`](file:///C:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/ml/config.py) to point `TrainingPaths.best_model_path` to `"experiments/unet/best_model.pt"` as the primary model checkpoint location.
   * Added `load_primary_model_config()` utility to load and parse `config/primary_segmentation_model.json`.

---

## 3. Application Integration Performed

* **Preserved Multi-Model Comparison & Reproducibility:**
  * Random Forest (`experiments/rf_baseline/`), XGBoost (`experiments/xgboost_baseline/`), Stacking Ensemble (`experiments/stacking/`), SegFormer Baseline (`experiments/segformer_b0/`), and Retrained SegFormer (`experiments/segformer_retrain/`) remain untouched and accessible for benchmark comparisons.
  * Benchmark evaluation script [`ml/benchmark_eval.py`](file:///C:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/ml/benchmark_eval.py) continues to evaluate all 4 model families independently without modification.
* **Default Deep-Learning Model:** Small U-Net is designated as the default deep-learning segmentation model across SARA configuration registries.

---

## 4. Tests Executed & Results

1. **Focused Adoption Unit Test Suite Created:**
   * File: [`tests/test_primary_model_adoption.py`](file:///C:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/tests/test_primary_model_adoption.py)
   * Verification items:
     * `test_primary_model_config_file_exists`: Validates structure of `config/primary_segmentation_model.json`.
     * `test_primary_model_checkpoint_loading_and_inference`: Loads state dict into `SmallUNet`, verifies 1,927,009 parameters, runs forward pass on `(1, 3, 256, 256)` dummy tensor, checks `(1, 1, 256, 256)` output shape and probability bounds `[0.0, 1.0]`.
     * `test_primary_model_training_paths`: Verifies default checkpoint path resolution in `TrainingPaths`.
     * `test_primary_model_somalia_validation_metrics_recorded`: Confirms accuracy of recorded Somalia metrics.

2. **Full Repository PyTest Suite Execution:**
   * Command: `python -m pytest tests/`
   * Result: **125 passed, 0 failed** across all test modules.

---

## 5. Unresolved Issues

* **None.** All artifacts loaded cleanly, architecture parameters matched expected values exactly, forward passes executed without error, and all unit and integration tests passed.

---

## 6. Remaining Validation Required Before Operational Claims

Before asserting operational deployment claims or using model predictions in active disaster response workflows, the following scientific safeguards and remaining validation steps must be observed:

> [!WARNING]
> **Scientific Disclaimers & Operational Limits:**
> 1. **Preliminary Probabilistic Evidence Only:** Model predictions represent preliminary probabilistic SAR water extent maps, not definitive proof of ground flooding, infrastructure damage, or structural collapse.
> 2. **No Life-Safety Claims:** Model predictions MUST NOT be used as the sole basis to claim that a road is closed, a settlement is inaccessible, or an area is safe.
> 3. **Validation Provenance:** Reported Somalia metrics (Precision: 91.61%, Recall: 94.57%, F1: 93.06%, IoU: 87.03%) represent validation results on the Sen1Floods11 benchmark split and DO NOT represent Nepal operational event accuracy.
> 4. **Data Isolation Guaranteed:** EMSR927 activation data and Nepal event change masks remain strictly isolated from training, tuning, and model selection.
> 5. **Attribution:** Sentinel-1 SAR data: © Copernicus Sentinel Data (2024). OpenStreetMap infrastructure: © OpenStreetMap contributors. DEM data: © Copernicus 30m Global DEM.
