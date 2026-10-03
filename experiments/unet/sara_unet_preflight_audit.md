# SARA Nepal Track B — Small U-Net Preflight Audit Report

**Project:** SARA (Satellite-Assisted Rescue & Assessment)  
**Target Architecture:** Small U-Net (PyTorch 2.6.0+cu124)  
**Hardware Target:** NVIDIA GeForce RTX 2050 (4.29 GB VRAM), 16 GB System RAM  
**Audit Date:** October 3, 2026  
**Status:** **`PREFLIGHT PASSED — DESIGN APPROVED FOR TRAINING`**

---

## 1. Dataset & Infrastructure Alignment

| Attribute | Verified Value / Property | Compliance Note |
| :--- | :--- | :--- |
| **Input Shape** | $3 \times 256 \times 256$ float32 tensor | Channel 1: $VV_{\text{dB}}$, Channel 2: $VH_{\text{dB}}$, Channel 3: $VV_{\text{dB}} - VH_{\text{dB}}$ ratio |
| **Label Encoding** | $0 = \text{Non-water}$, $1 = \text{Water}$, $255 = \text{Nodata/Cloud}$ | $255$ is strictly masked in loss and metrics |
| **Normalization** | Channel means `[-11.0, -18.0, 7.0]`, stds `[3.0, 3.5, 2.5]` | Normalization applied per channel in PyTorch Dataset |
| **Batch Size & AMP** | Batch size = 2, FP16 Automatic Mixed Precision (AMP) enabled | Optimized for 4 GB VRAM limit |
| **Loss Function** | `MaskedBCEDiceLoss` (BCE weight = 0.5, Dice weight = 0.5) | Ignores label 255 completely |

---

## 2. Manifest & Split Isolation Audit

* **Train Partition (`train_manifest.json`):** 36 tiles across 9 events (*Bolivia, Ghana, India, Nigeria, Pakistan, Paraguay, Spain, SriLanka, USA*). Used exclusively for gradient updates.
* **Validation Partition (`val_manifest.json`):** 4 tiles (*Somalia* event). Evaluated in `torch.no_grad()` mode for early stopping (`val_iou`) and checkpoint selection. No gradient updates occur on validation data.
* **Test Partition (`test_manifest.json`):** 4 tiles (*Mekong* event) — **STRICTLY ISOLATED & UNTOUCHED**.
* **EMSR927 & Anomaly Mask Isolation:** EMSR927 assets remain isolated under `validation/`. Nepal preliminary anomaly mask is NOT used as ground truth.

---

## 3. Preflight Conclusion

> [!IMPORTANT]
> **PREFLIGHT AUDIT PASSED — DESIGN APPROVED FOR TRAINING**
>
> All raster shapes, dtypes, channel orders, nodata values, split manifests, and PyTorch CUDA GPU prerequisites have been verified. Model training may proceed.
