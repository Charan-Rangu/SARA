# SARA Nepal Track B — Final Four-Model Benchmark Report

**Project:** SARA (Satellite-Assisted Rescue & Assessment)  
**Evaluation Date:** October 3, 2026  
**Hardware Platform:** NVIDIA GeForce RTX 2050 (4.29 GB VRAM), 16 GB System RAM, AMD Ryzen / Intel CPU  
**Status:** **`BENCHMARK COMPLETED — REPRODUCIBLE FOUR-MODEL COMPARISON`**

---

## 1. Executive Summary

This report presents the final read-only, leakage-resistant benchmark across all four Machine Learning experiments implemented for Sentinel-1 flood-water segmentation:

1. **Random Forest Baseline** (100 Decision Trees, Pixel Backscatter Features)
2. **XGBoost Baseline** (100 Gradient-Boosted Trees, Pixel Backscatter Features)
3. **Stacking Ensemble** (Base RF + Base XGBoost + Logistic Regression Meta-Learner)
4. **Small U-Net** (1.93 M Parameter Deep PyTorch Convolutional Neural Network)

All evaluation protocols, feature extraction rules, nodata masking logic (`nodata=255`), and decision thresholds (`0.50`) were strictly frozen prior to evaluating the **Mekong Test Split**. No model retraining, hyperparameter tuning, or threshold selection occurred on Mekong test labels.

---

## 2. Model Artifact & Resource Inventory

| Model Name | Artifact File Path | File Size | Model Type / Architecture | Training Time | Peak VRAM |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **Random Forest Baseline** | `experiments/rf_baseline/rf_baseline_model.joblib` | 1.07 MB | Sklearn RandomForestClassifier (100 trees) | 0.81 s | 0.00 GB (CPU) |
| **XGBoost Baseline** | `experiments/xgboost_baseline/xgb_baseline_model.json` | 0.21 MB | XGBoost Booster JSON (100 trees) | 0.91 s | 0.00 GB (CPU) |
| **Stacking Ensemble** | `experiments/stacking/stacking_model.joblib` | **1.25 MB** | **Complete Package:** RF + XGB + Meta-LR | 1.84 s | 0.00 GB (CPU) |
| **Small U-Net** | `experiments/unet/best_model.pt` | **23.21 MB** | PyTorch Small U-Net (1,927,009 params) | 46.34 s | 0.22 GB (GPU) |

---

## 3. Independently Recomputed Somalia Validation Metrics

Evaluated on the **4 Somalia Validation Tiles** (260,096 valid pixels, excluding label 255):

| Metric | Random Forest | XGBoost | Stacking Ensemble | Small U-Net |
| :--- | :---: | :---: | :---: | :---: |
| **Valid Pixels** | 260,096 | 260,096 | 260,096 | 260,096 |
| **True Positives (TP)** | 3,001 | 2,997 | 3,001 | 2,838 |
| **False Positives (FP)** | 1 | 88 | 9 | 260 |
| **False Negatives (FN)** | 0 | 4 | 0 | 163 |
| **True Negatives (TN)** | 257,094 | 257,007 | 257,086 | 256,835 |
| **Water Precision** | **99.967%** | 97.148% | 99.701% | 91.608% |
| **Water Recall** | **100.000%** | 99.867% | **100.000%** | 94.569% |
| **Water F1 / Dice** | **99.983%** | 98.488% | 99.850% | 93.064% |
| **Water IoU** | **99.967%** | 97.023% | 99.701% | 87.029% |

---

## 4. Final Mekong Test Evaluation Results

Evaluated on the **4 Mekong Test Tiles** (260,096 valid pixels, excluding label 255):

### Overall Test Metrics

| Model Name | Valid Pixels | TP | FP | FN | TN | Precision | Recall | F1 / Dice | IoU | Inference Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 260,096 | 2,974 | 1 | 0 | 257,121 | 99.97% | 100.00% | 99.98% | 99.97% | 0.224 s |
| **XGBoost** | 260,096 | 2,971 | 96 | 3 | 257,026 | 96.87% | 99.90% | 98.36% | 96.78% | 0.100 s |
| **Stacking Ensemble** | 260,096 | 2,974 | 9 | 0 | 257,113 | 99.70% | 100.00% | 99.85% | 99.70% | 0.313 s |
| **Small U-Net** | 260,096 | 2,812 | 314 | 162 | 256,808 | 89.96% | 94.55% | 92.20% | 85.52% | 0.066 s |

### Per-Tile Detailed Breakdown (Mekong Test Split)

#### Random Forest
| Tile ID | Valid Pixels | TP | FP | FN | TN | Precision | Recall | F1 / Dice | IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Mekong_012` | 65,024 | 775 | 0 | 0 | 64,249 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `Mekong_013` | 65,024 | 763 | 1 | 0 | 64,260 | 0.9987 | 1.0000 | 0.9993 | 0.9987 |
| `Mekong_014` | 65,024 | 755 | 0 | 0 | 64,269 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `Mekong_015` | 65,024 | 681 | 0 | 0 | 64,343 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

#### XGBoost
| Tile ID | Valid Pixels | TP | FP | FN | TN | Precision | Recall | F1 / Dice | IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Mekong_012` | 65,024 | 774 | 27 | 1 | 64,222 | 0.9663 | 0.9987 | 0.9822 | 0.9651 |
| `Mekong_013` | 65,024 | 763 | 22 | 0 | 64,239 | 0.9720 | 1.0000 | 0.9858 | 0.9720 |
| `Mekong_014` | 65,024 | 754 | 21 | 1 | 64,248 | 0.9729 | 0.9987 | 0.9856 | 0.9716 |
| `Mekong_015` | 65,024 | 680 | 26 | 1 | 64,317 | 0.9632 | 0.9985 | 0.9805 | 0.9618 |

#### Stacking Ensemble
| Tile ID | Valid Pixels | TP | FP | FN | TN | Precision | Recall | F1 / Dice | IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Mekong_012` | 65,024 | 775 | 2 | 0 | 64,247 | 0.9974 | 1.0000 | 0.9987 | 0.9974 |
| `Mekong_013` | 65,024 | 763 | 3 | 0 | 64,258 | 0.9961 | 1.0000 | 0.9980 | 0.9961 |
| `Mekong_014` | 65,024 | 755 | 2 | 0 | 64,267 | 0.9974 | 1.0000 | 0.9987 | 0.9974 |
| `Mekong_015` | 65,024 | 681 | 2 | 0 | 64,341 | 0.9971 | 1.0000 | 0.9985 | 0.9971 |

#### Small U-Net
| Tile ID | Valid Pixels | TP | FP | FN | TN | Precision | Recall | F1 / Dice | IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Mekong_012` | 65,024 | 740 | 100 | 35 | 64,149 | 0.8810 | 0.9548 | 0.9164 | 0.8457 |
| `Mekong_013` | 65,024 | 727 | 63 | 36 | 64,198 | 0.9203 | 0.9528 | 0.9363 | 0.8801 |
| `Mekong_014` | 65,024 | 706 | 80 | 49 | 64,189 | 0.8982 | 0.9351 | 0.9163 | 0.8455 |
| `Mekong_015` | 65,024 | 639 | 71 | 42 | 64,272 | 0.9000 | 0.9383 | 0.9188 | 0.8497 |

---

## 5. Comparative Performance & Trade-Off Analysis

1. **Specular Reflection vs Pointwise Feature Power:** In Sentinel-1 C-band SAR imagery, calm open water acts as a specular reflector, resulting in very low backscatter ($VV < -16\text{ dB}$, $VH < -24\text{ dB}$). Pixel-level tree ensembles (Random Forest, XGBoost, Stacking) directly partition this feature space with zero spatial overhead, achieving near-perfect metrics across both Somalia validation and Mekong test scenes.
2. **Spatial Convolution Overhead & Data Requirements:** Small U-Net spatial convolutions provide contextual region awareness. However, when trained on 36 chips without extensive domain-specific spatial augmentation, U-Net exhibits slight spatial boundary smoothing along shorelines and river channels.
3. **Efficiency vs Accuracy:** Random Forest and Stacking achieve top overall test metrics with sub-second inference runtime and 0.00 GB GPU RAM overhead, making them exceptionally suited for rapid operational deployment.

---

## 6. Limitations & Scientific Safeguards

* **Regional Transferability:** High performance on Somalia and Mekong chips does NOT guarantee identical quantitative accuracy in Nepal's steep Himalayan terrain (due to SAR layover, shadow, and foreshortening).
* **Ground Truth Boundaries:** Sen1Floods11 labels are training ground truth only to the extent supported by their actual sensor resolution and provenance.
* **Non-Inference of Disaster Operational Status:** SAR flood predictions represent surface water extent only. Emergency rescue access, road closures, and structural damage require independent ground-truth or optical/vector verification.

---

## 7. Benchmark Conclusion

> [!IMPORTANT]
> **FINAL BENCHMARK COMPLETE**
>
> All 4 models have been evaluated in a fully reproducible, leakage-safe environment. All artifacts, predictions, and metrics are permanently archived under `experiments/final_benchmark/`.
