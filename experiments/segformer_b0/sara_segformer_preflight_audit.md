# SARA Nepal Track B — SegFormer-B0 Preflight Audit Report

**Project:** SARA (Satellite-Assisted Rescue & Assessment)  
**Target Architecture:** SegFormer-B0 (Hierarchical Mix Transformer MiT-B0 + All-MLP Decoder Head)  
**Software Environment:** PyTorch 2.6.0+cu124 (CUDA 12.4 enabled)  
**Hardware Target:** NVIDIA GeForce RTX 2050 (4.29 GB VRAM), 16 GB System RAM  
**Audit Date:** October 3, 2026  
**Status:** **`PREFLIGHT PASSED — NATIVE SEGFORMER-B0 DESIGN APPROVED FOR TRAINING`**

---

## 1. Environment & Framework Audit

| Dependency / Tool | Verified Version | Capability / Compliance |
| :--- | :--- | :--- |
| **PyTorch** | `2.6.0+cu124` | Native PyTorch implementation (`torch.nn`, `torch.amp.autocast`) |
| **CUDA Device** | `NVIDIA GeForce RTX 2050` | 4.29 GB VRAM capacity (Target VRAM budget: $< 2.0\text{ GB}$) |
| **Mixed Precision** | `torch.cuda.amp.GradScaler` | FP16 Automatic Mixed Precision enabled for CUDA |
| **Raster I/O** | `rasterio 1.4.3` | GeoTIFF spatial metadata preservation (`EPSG:4326`, transform) |

---

## 2. Model Architecture & Data Alignment

| Attribute | Verified Value / Property | Compliance Note |
| :--- | :--- | :--- |
| **Encoder** | Hierarchical MiT-B0 ($C_1=32, C_2=64, C_3=160, C_4=256$) | 4 stages with Overlapped Patch Embedding & Efficient Self-Attention |
| **Decoder Head** | Lightweight All-MLP Decoder Head ($E=256$) | Unifies multiscale features to $H/4 \times W/4$ before fusion |
| **Input Tensor** | $3 \times 256 \times 256$ float32 tensor | Channel 1: $VV_{\text{dB}}$, Channel 2: $VH_{\text{dB}}$, Channel 3: $VV_{\text{dB}} - VH_{\text{dB}}$ ratio |
| **Output Logits** | $1 \times 256 \times 256$ float32 logit tensor | Binary water logit per pixel, upsampled $4\times$ to full resolution |
| **Label Encoding** | $0 = \text{Non-water}$, $1 = \text{Water}$, $255 = \text{Nodata/Cloud}$ | Label $255$ is strictly masked in loss and metrics |
| **Normalization** | Channel means `[-11.0, -18.0, 7.0]`, stds `[3.0, 3.5, 2.5]` | Standardized per channel in PyTorch Dataset |
| **Loss Function** | `MaskedBCEDiceLoss` (BCE weight = 0.5, Dice weight = 0.5) | Completely ignores label $255$ |

---

## 3. Manifest & Dataset Isolation Audit

* **Train Partition (`train_manifest.json`):** 36 tiles across 9 events (*Bolivia, Ghana, India, Nigeria, Pakistan, Paraguay, Spain, SriLanka, USA*). Used exclusively for gradient updates.
* **Validation Partition (`val_manifest.json`):** 4 tiles (*Somalia* event). Evaluated in `torch.no_grad()` mode for early stopping (`val_iou`) and checkpoint selection. Zero gradient updates on validation data.
* **Test Partition (`test_manifest.json`):** 4 tiles (*Mekong* event) — **STRICTLY ISOLATED & UNTOUCHED**.
* **EMSR927 & Anomaly Mask Isolation:** EMSR927 assets remain isolated under `validation/`. Nepal preliminary anomaly mask is NOT used as ground truth.

---

## 4. Preflight Audit Conclusion

> [!IMPORTANT]
> **PREFLIGHT AUDIT PASSED — DESIGN APPROVED FOR TRAINING**
>
> All raster shapes, dtypes, channel orders, nodata values, split manifests, and PyTorch CUDA GPU prerequisites have been verified. SegFormer-B0 model implementation and training may proceed.
