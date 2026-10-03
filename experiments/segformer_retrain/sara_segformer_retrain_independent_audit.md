# SARA Nepal Track B — Controlled SegFormer Retraining Independent Audit Report

**Project:** SARA (Satellite-Assisted Rescue & Assessment)  
**Model Architecture:** SegFormer-B0 (MiT-B0 + All-MLP Decoder Head, FP16 AMP)  
**Evaluated Checkpoint:** `experiments/segformer_retrain/best_model.pt`  
**Audit Date:** October 3, 2026  
**Status:** **`AUDIT PASSED — RECOMPUTED METRICS VERIFIED & ACCEPTED`**

---

## 1. Executive Summary & Verification Matrix

Independent verification confirmed that the retrained SegFormer-B0 model (`experiments/segformer_retrain/best_model.pt`) and exported prediction GeoTIFFs (`experiments/segformer_retrain/predictions/`) strictly adhere to all dataset isolation, metadata preservation, and evaluation protocols.

| Audit Requirement | Status | Verification Detail |
| :--- | :---: | :--- |
| **1. Best Checkpoint Load** | PASSED | Loaded `best_model.pt` (3,714,401 trainable parameters) successfully |
| **2. Recomputed Metrics** | PASSED | Exact match: 260,096 valid Somalia pixels, F1 = 3.13%, IoU = 1.59% |
| **3. Nodata Exclusion** | PASSED | Label 255 strictly excluded from loss, evaluation, and exported GeoTIFFs |
| **4. Label & Channels** | PASSED | Output binary (0=non-water, 1=water), 3 channels (VV, VH, VV-VH) |
| **5. Manifest & History** | PASSED | 36 training tiles (9 events), completed epochs verified in CSV |
| **6. Config Integrity** | PASSED | `config.json` matches dataset parameters & random seed 42 |
| **7. Mekong & EMSR Isolation** | PASSED | Zero Mekong or EMSR927 leakage detected in train or val sets |
| **8. Artifact Preservation** | PASSED | RF, XGBoost, Stacking, U-Net, and SegFormer B0 baseline intact & un-modified |
| **9. GeoTIFF Compliance** | PASSED | $256 \times 256$, `uint8`, `nodata=255.0`, `EPSG:4326`, exact affine transform |
| **10. Scientific Safeguards** | PASSED | Somalia validation used solely for early stopping (`val_iou`) |

---

## 2. Recomputed Somalia Validation Metrics

Evaluation performed over 4 Somalia validation tiles (260,096 valid pixels, excluding label 255):

### Overall Confusion Matrix & Metrics
* **Total Valid Pixels:** 260,096
* **True Positives (TP):** 307
* **False Positives (FP):** 16,300
* **False Negatives (FN):** 2,694
* **True Negatives (TN):** 240,795
* **Water Precision:** 1.85% (0.018486)
* **Water Recall:** 10.23% (0.102299)
* **Water F1 / Dice Score:** 3.13% (0.031314)
* **Water Intersection over Union (IoU):** 1.59% (0.015906)

### Per-Tile Detailed Breakdown

| Tile ID | Valid Pixels | TP | FP | FN | TN | Precision | Recall | F1 / Dice | IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Somalia_028` | 65,024 | 99 | 4,907 | 675 | 59,343 | 0.0198 | 0.1279 | 0.0343 | 0.0174 |
| `Somalia_029` | 65,024 | 61 | 3,217 | 713 | 61,033 | 0.0186 | 0.0788 | 0.0301 | 0.0153 |
| `Somalia_030` | 65,024 | 69 | 3,920 | 643 | 60,392 | 0.0173 | 0.0969 | 0.0294 | 0.0149 |
| `Somalia_031` | 65,024 | 78 | 4,256 | 663 | 60,027 | 0.0180 | 0.1053 | 0.0307 | 0.0156 |

---

## 3. Audit Conclusion

> [!IMPORTANT]
> **AUDIT STATUS: PASSED**
>
> The controlled SegFormer retraining experiment is fully verified, reproducible, and cleanly archived under `experiments/segformer_retrain/`.
