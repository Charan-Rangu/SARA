# SARA Nepal Track B — Small U-Net Independent Audit Report

**Project:** SARA (Satellite-Assisted Rescue & Assessment)  
**Model Architecture:** Small U-Net (PyTorch 2.6.0+cu124, FP16 AMP)  
**Evaluated Checkpoint:** `experiments/unet/best_model.pt` (Epoch 24)  
**Audit Date:** October 3, 2026  
**Status:** **`AUDIT PASSED — RECOMPUTED METRICS VERIFIED & ACCEPTED`**

---

## 1. Executive Summary & Verification Matrix

Independent verification confirmed that the trained Small U-Net model (`experiments/unet/best_model.pt`) and exported prediction GeoTIFFs (`experiments/unet/predictions/`) strictly adhere to all dataset isolation, metadata preservation, and evaluation protocols.

| Audit Requirement | Status | Verification Detail |
| :--- | :---: | :--- |
| **1. Best Checkpoint Load** | PASSED | Loaded `best_model.pt` (1,927,009 trainable parameters) successfully |
| **2. Recomputed Metrics** | PASSED | Exact match: 260,096 valid Somalia pixels, F1 = 93.06%, IoU = 87.03% |
| **3. Nodata Exclusion** | PASSED | Label 255 strictly excluded from loss, evaluation, and exported GeoTIFFs |
| **4. Label & Channels** | PASSED | Output binary (0=non-water, 1=water), 3 channels (VV, VH, VV-VH) |
| **5. Manifest & History** | PASSED | 36 training tiles (9 events), 30 completed epochs verified in CSV |
| **6. Config Integrity** | PASSED | `unet_config.json` matches dataset parameters & random seed 42 |
| **7. Mekong & EMSR Isolation** | PASSED | Zero Mekong or EMSR927 leakage detected in train or val sets |
| **8. Artifact Preservation** | PASSED | RF, XGBoost, and Stacking model artifacts remain intact & un-modified |
| **9. GeoTIFF Compliance** | PASSED | $256 \times 256$, `uint8`, `nodata=255.0`, `EPSG:4326`, exact affine transform |
| **10. Scientific Integrity** | PASSED | Somalia validation used solely for early stopping (`val_iou`) |

---

## 2. Recomputed Somalia Validation Metrics

Evaluation performed over 4 Somalia validation tiles (260,096 valid pixels, excluding label 255):

### Overall Confusion Matrix & Metrics
* **Total Valid Pixels:** 260,096
* **True Positives (TP):** 2,838
* **False Positives (FP):** 260
* **False Negatives (FN):** 163
* **True Negatives (TN):** 256,835
* **Water Precision:** 91.61% (0.916075)
* **Water Recall:** 94.57% (0.945685)
* **Water F1 / Dice Score:** 93.06% (0.930644)
* **Water Intersection over Union (IoU):** 87.03% (0.870285)

### Per-Tile Detailed Breakdown

| Tile ID | Valid Pixels | TP | FP | FN | TN | Precision | Recall | F1 / Dice | IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Somalia_028` | 65,024 | 739 | 60 | 35 | 64,190 | 0.9249 | 0.9548 | 0.9396 | 0.8861 |
| `Somalia_029` | 65,024 | 727 | 70 | 47 | 64,180 | 0.9122 | 0.9393 | 0.9255 | 0.8614 |
| `Somalia_030` | 65,024 | 680 | 63 | 32 | 64,249 | 0.9152 | 0.9551 | 0.9347 | 0.8774 |
| `Somalia_031` | 65,024 | 692 | 67 | 49 | 64,216 | 0.9117 | 0.9339 | 0.9227 | 0.8564 |

---

## 3. Four-Model Comparison Across SARA Experiments

| Metric / Attribute | Experiment 1: RF Baseline | Experiment 2: XGBoost Baseline | Experiment 3: Stacking Ensemble | Experiment 4: Small U-Net |
| :--- | :---: | :---: | :---: | :---: |
| **Model Type** | Pixel Random Forest | Pixel XGBoost | RF + XGB Meta-Learner | Deep PyTorch U-Net |
| **Parameter Count** | 100 Trees (N/A) | 100 Trees (N/A) | Stacking Meta-LR | **1,927,009** |
| **Training Time** | 0.81 s | 0.91 s | 1.84 s | **46.34 s** |
| **Peak GPU VRAM** | 0.00 GB (CPU) | 0.00 GB (CPU) | 0.00 GB (CPU) | **0.22 GB** |
| **Validation Water Precision** | **99.967%** | 97.148% | 99.701% | 91.608% |
| **Validation Water Recall** | **100.000%** | 99.867% | **100.000%** | 94.569% |
| **Validation Water F1 / Dice** | **99.983%** | 98.488% | 99.850% | 93.064% |
| **Validation Water IoU** | **99.967%** | 97.023% | 99.701% | 87.029% |

### Analysis of U-Net Performance vs. Classical Baselines
1. **Spatial Context vs Pointwise Backscatter:** Classical pixel-level models (RF/XGBoost) exploit the stark specular reflection threshold in SAR VV/VH backscatter directly. U-Net introduces spatial convolutional kernels which produce spatial boundary smoothing along land-water transitions, leading to slight boundary false positives (260 FP) on Somalia tiles.
2. **Data Scale:** Small U-Net trained on 36 chips ($256 \times 256$) without extensive spatial augmentations achieves strong $87.03\%$ IoU, demonstrating solid spatial semantic learning, though classical pixel-wise decision trees remain highly effective for point backscatter segmentation.

---

## 4. Resource & Computational Footprint

* **Peak GPU Memory Allocated:** 0.22 GB (220 MB) VRAM on NVIDIA RTX 2050 (4.29 GB total capacity).
* **Completed Training Epochs:** 30 / 30 (Best checkpoint saved at Epoch 24).
* **Total Training Duration:** 46.34 seconds (~1.54 s/epoch).
* **Inference Runtime (4 Tiles):** 0.214 seconds.

---

## 5. Independent Audit Conclusion

> [!IMPORTANT]
> **AUDIT STATUS: PASSED**
>
> The Small U-Net implementation in Experiment 4 is fully verified, scientifically sound, reproducible, and ready for baseline synthesis. All 10 independent verification checks passed with zero errors or data leakage.
