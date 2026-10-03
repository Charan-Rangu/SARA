# SARA — Secondary Disk-Space Optimization Audit Report

**Project:** SARA (Satellite-Assisted Rescue & Assessment)  
**Audit Timestamp:** 2026-10-04T00:30:00+05:30  
**Project Path:** `C:\Users\chara\OneDrive\Documents\Multimodal AI\sara`  
**Audit Objective:** Rigorously investigate `target.data/` (SNAP intermediate workspace) and `data/output/run_custom_*` (ephemeral test output runs) to identify safe space recovery opportunities while strictly protecting Nepal baseline outputs, model checkpoints, and core source code.

---

## 1. Executive Summary & Recoverable Space Overview

| Asset Group | Directory Path | File Count | Exact Size | Recommended Action |
|---|---|:---:|:---:|:---:|
| **SNAP Intermediate Raster Workspace** | `target.data/` + `target.dim` | 5 files | **5.142 GB** | **RECOMMEND REMOVAL (OR ARCHIVE)** |
| **Custom Test Run Outputs** | `data/output/run_custom_*` | 207 dirs | **6.584 GB** | **RECOMMEND REMOVAL (OR ARCHIVE)** |
| **Nepal Baseline U-Net Evaluation** | `data/output/run_nepal_unet_eval_20261003/` | 10 files | **1.943 GB** | **MUST_KEEP (PROTECTED BASELINE)** |
| **Nepal Baseline Change Detection** | `data/output/run_nepal_multitemporal_change_20261003/` | 10 files | **0.049 GB** | **MUST_KEEP (PROTECTED BASELINE)** |

- **Total Potential Recoverable Storage:** **11.726 GB (12,590,808,690 bytes)**
- **Post-Recovery Estimated Project Size:** **~48.525 GB** (down from 60.251 GB)

---

## 2. Phase 1 — Detailed Investigation of `target.data/` and `target.dim`

### 2.1 File Inventory & Format Analysis

```
Multimodal AI/sara/
├── target.dim                     (7.84 MB XML)   - ESA SNAP BEAM-DIMAP product metadata
└── target.data/                   (5.14 GB Total) - ENVI binary raster band folder
    ├── Amplitude_VH.hdr           (715 B)         - ENVI header for VH band
    ├── Amplitude_VH.img           (2,632.74 MB)   - Raw float32 raster data (32,072 × 21,519 pixels)
    ├── Amplitude_VV.hdr           (715 B)         - ENVI header for VV band
    ├── Amplitude_VV.img           (2,632.74 MB)   - Raw float32 raster data (32,072 × 21,519 pixels)
    └── vector_data/               (0 B)           - Empty metadata directory
```

### 2.2 Technical Provenance & Relationship to `data/processed/`
- **Origin:** `target.dim` and `target.data/` were created as intermediate outputs by the ESA SNAP Sentinel Application Platform `gpt` tool during the initial execution of Range-Doppler Terrain Correction and Calibration on the raw Sentinel-1 GRD archive `data/raw/S1D_IW_GRDH_1SDV_20260816T122141_...SAFE.zip`.
- **Processed GeoTIFFs Status:** The final, calibrated, and co-registered 3-band Float32 GeoTIFFs derived from this step already exist and are strictly preserved in `data/processed/`:
  - `data/processed/unet_inputs/post_calibrated_sigma0_unet_3band.tif` (6.45 GB)
  - `data/processed/unet_inputs/post_calibrated_sigma0_prepared_3band.tif` (6.45 GB)
  - `data/processed/post_calibrated_sigma0.tif` (3.46 GB)

### 2.3 Code, Test, and Dependency Audit
- **Code References:** Zero production modules in `backend/`, `geospatial/`, `ml/`, or `validation/` read from `target.data/` or `target.dim`.
- **Inference Requirements:** Runtime inference (`SmallUNet` / `backend.services.unet_segmentation_service`) directly reads GeoTIFFs from `data/processed/` or user-uploaded `.tif` files.
- **Automated Tests:** Zero pytest test cases depend on `target.data/` or `target.dim`.
- **Regeneration Capability:** 100% regeneratable from the original raw satellite zip `data/raw/S1D_IW_GRDH_1SDV_20260816T122141_...SAFE.zip` using `geospatial/snap_graph.py` and ESA SNAP `gpt.exe`.

### 2.4 Risk Assessment & Recommendation
- **Risk Level:** **Very Low (Zero runtime impact)**.
- **Recommendation:** **`REMOVE` (or move to external archive)** to instantly recover **5.14 GB** of disk space.

---

## 3. Phase 2 — Detailed Investigation of `data/output/run_custom_*`

### 3.1 Directory Analysis & Classification
The `data/output/` directory contains **207 custom test run folders** generated dynamically during automated testing and interactive web-app testing.

#### Top Space-Consuming Custom Runs (> 100 MB):
1. `data/output/run_custom_20261003_194849_0f8040/` — **1,686.02 MB** (ad-hoc full-scene Nepal test upload)
2. `data/output/run_custom_20261003_200721_f49d0c/` — **1,686.02 MB** (ad-hoc full-scene Nepal test upload)
3. `data/output/run_custom_20261003_205143_7aad8c/` — **1,686.02 MB** (ad-hoc full-scene Nepal test upload)
4. `data/output/run_custom_20261003_205928_6fb327/` — **968.65 MB** (ad-hoc sub-scene test upload)
5. `data/output/run_custom_*` (203 remaining folders) — **~557.34 MB** (small $256 \times 256$ mock/fixture outputs generated during pytest execution)

### 3.2 Comparison with Protected Baseline Nepal Runs

| Run Directory | Size (MB) | Role / Artifacts | Status |
|---|:---:|---|:---:|
| `data/output/run_nepal_unet_eval_20261003/` | **1,989.95 MB** | Official Nepal Small U-Net baseline prediction raster (`water_probability_map.tif`, `predicted_water_mask.tif`, `water_probability_overlay.png`, evaluation metrics) | **MUST_KEEP (DO NOT DELETE)** |
| `data/output/run_nepal_multitemporal_change_20261003/` | **49.18 MB** | Official Nepal multi-temporal SAR change & terrain slope agreement overlay (`nepal_water_and_change_agreement_mask.tif`, `agreement_overlay.png`, `slope_overlay.png`) | **MUST_KEEP (DO NOT DELETE)** |
| `data/output/run_custom_*` (207 dirs) | **6,584.05 MB** | Ephemeral, dynamically generated test and evaluation outputs | **SAFE_TO_REMOVE (UPON APPROVAL)** |

### 3.3 Code, Test, and Dependency Audit
- **Code References:** Production application default routes in `backend/main.py` and `docs/demo_script.md` link to `run_nepal_multitemporal_change_20261003` and `run_nepal_unet_eval_20261003`.
- **Test Dependencies:** Test suites dynamically generate fresh `run_custom_<timestamp>` directories when executed and do not depend on existing historical ones.
- **Regeneration Capability:** Any custom run can be re-executed on demand via `POST /segmentation/predict` or `POST /segmentation/upload`.

### 3.4 Risk Assessment & Recommendation
- **Risk Level:** **Very Low (Zero regression impact)**.
- **Recommendation:** **`REMOVE` (or move to external archive)** all 207 `run_custom_*` directories to recover **6.58 GB** of disk space while keeping `run_nepal_unet_eval_20261003` and `run_nepal_multitemporal_change_20261003` intact.

---

## 4. Protected Assets Confirmation

The following assets remain **strictly protected and will NOT be touched**:
1. `data/output/run_nepal_unet_eval_20261003/` (1.94 GB baseline)
2. `data/output/run_nepal_multitemporal_change_20261003/` (49.18 MB baseline)
3. `data/raw/*.SAFE.zip` (3.30 GB raw Sentinel-1 archives)
4. `data/processed/unet_inputs/*.tif` (calibrated Nepal rasters)
5. `experiments/unet/best_model.pt` (Small U-Net model checkpoint)
6. `dist/SARA_competition_submission.zip` (authoritative submission ZIP)
7. `submission_package/` (uncompressed submission tree)
8. All source code (`backend/`, `geospatial/`, `ml/`, `validation/`, `config/`, `tests/`, `docs/`)
9. Copernicus, WorldDEM-30, and OpenStreetMap legal attributions
10. EMSR927 quarantine logic in `validation/`

---

## 5. Candidate Action Summary Table

| Proposed Cleanup Candidate | Exact Location | Size | Regeneration Source | Risk | Recommended Action |
|---|---|:---:|---|:---:|:---:|
| **SNAP Intermediate Product** | `target.data/` + `target.dim` | **5.142 GB** | `data/raw/*.SAFE.zip` via SNAP GPT | None | **REMOVE (Pending User Approval)** |
| **Custom Test Run Outputs** | `data/output/run_custom_*` (207 dirs) | **6.584 GB** | Re-run via API or test suite | None | **REMOVE (Pending User Approval)** |
| **Nepal Baseline U-Net Evaluation** | `data/output/run_nepal_unet_eval_20261003/` | **1.943 GB** | Baseline evaluation | Critical | **MUST_KEEP** |
| **Nepal Baseline Change Detection** | `data/output/run_nepal_multitemporal_change_20261003/` | **0.049 GB** | Baseline change analysis | Critical | **MUST_KEEP** |

**Total Recoverable Space:** **11.726 GB**
