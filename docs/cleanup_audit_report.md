# SARA — Safe Project Cleanup & File Optimization Audit Report

**Project:** SARA (Satellite-Assisted Rescue & Assessment)  
**Audit Timestamp:** 2026-10-04T00:18:30+05:30  
**Project Root Directory:** `C:\Users\chara\OneDrive\Documents\Multimodal AI\sara`  
**Overall Goal:** Safely identify and prune unneeded development artifacts, regeneratable bytecode caches, and temporary test directories while strictly preserving all trained models, raw satellite imagery, baseline GeoTIFFs, test suites, and the authoritative submission archive.

---

## 1. Project Inventory & Baseline Metrics (Before Cleanup)

| Directory | File Count | Size (MB) | Purpose / Role | Classification |
|---|:---:|:---:|---|:---:|
| `data/` | 1,972 | 51,050.51 MB | Raw Sentinel-1 SAFE archives, processed Nepal 3-band rasters, OSM vectors, training sets | **MUST_KEEP** |
| `target.data/` | 6 | 5,265.48 MB | Intermediate SNAP ESA raster exports (`Amplitude_VH.img`, `Amplitude_VV.img`) | **REVIEW_REQUIRED** |
| `.venv/` | 33,596 | 5,187.08 MB | Local Python 3.13 virtual environment with PyTorch CUDA & GDAL bindings | **MUST_KEEP** *(Separate Env Decision)* |
| `experiments/` | 93 | 112.89 MB | Primary Small U-Net model checkpoint (`best_model.pt`), metrics, evaluation logs | **MUST_KEEP** |
| `scratch/` | 275 | 33.02 MB | Scratch test scripts, temporary extraction audits (`scratch/clean_extract_audit/`) | **SAFE_TO_REMOVE** *(Temp Extractions)* / **REVIEW_REQUIRED** *(Scripts)* |
| `submission_package/` | 124 | 31.57 MB | Uncompressed competition submission package folder | **MUST_KEEP** |
| `dist/` | 1 | 29.07 MB | Authoritative competition submission ZIP archive (`SARA_competition_submission.zip`) | **MUST_KEEP** |
| `docs/` | 19 | 7.91 MB | Architecture, technical summaries, methodology, demo script, and readiness reports | **MUST_KEEP** |
| `tests/` | 85 | 0.85 MB | Comprehensive 177-test automated regression and integration test suite | **MUST_KEEP** |
| `ml/` | 48 | 0.53 MB | Machine learning architectures, dataset loaders, loss functions, and trainer modules | **MUST_KEEP** |
| `backend/` | 26 | 0.49 MB | FastAPI REST API routes, Leaflet dashboard UI, background service workers | **MUST_KEEP** |
| `geospatial/` | 24 | 0.25 MB | SAR preprocessing, multi-temporal change detection, D8 flow tracing, accessibility | **MUST_KEEP** |
| `scripts/` | 11 | 0.09 MB | Submission packaging tools, preflight verification, benchmark runners | **MUST_KEEP** |
| `validation/` | 3 | 0.02 MB | Independent validation safeguards and EMSR927 data isolation engine | **MUST_KEEP** |
| `.pytest_cache/` | 5 | 0.01 MB | Pytest execution cache | **SAFE_TO_REMOVE** |
| Root (`.`) | 9 | 7.50 MB | `README.md`, `requirements*.txt`, `.env.example`, `.gitignore`, `LICENSE` | **MUST_KEEP** |

- **Total Files Before Cleanup:** 36,299 files
- **Total Project Size Before Cleanup:** 60.281 GB (64,725,771,208 bytes)

---

## 2. Protected Assets Policy (MUST_KEEP)

The following assets are strictly protected and **must never be deleted or modified**:

1. **Trained Neural Network Checkpoint:**
   - `experiments/unet/best_model.pt` (22.14 MB, Small U-Net, 1.93M parameters).
2. **Raw Satellite Imagery & SAFE Archives:**
   - `data/raw/S1D_IW_GRDH_1SDV_20260816T122141_...SAFE.zip` (1.65 GB)
   - `data/raw/S1D_IW_GRDH_1SDV_20260828T122141_...SAFE.zip` (1.65 GB)
3. **Calibrated & Co-Registered Nepal Rasters:**
   - `data/processed/unet_inputs/post_calibrated_sigma0_prepared_3band.tif` (6.45 GB)
   - `data/processed/unet_inputs/post_calibrated_sigma0_unet_3band.tif` (6.45 GB)
   - `data/processed/unet_inputs/post_calibrated_sigma0_unet_3band_corrected.tif` (6.45 GB)
   - `data/processed/post_calibrated_sigma0.tif` (3.46 GB)
4. **Historical Nepal Production GeoTIFFs:**
   - `data/output/run_custom_*/water_probability_map.tif` (1.68 GB each)
   - `data/output/run_nepal_unet_eval_20261003/water_probability_map.tif` (1.68 GB)
5. **Training Datasets & Labels:**
   - `data/ml/sen1floods11/` and `data/ml/train_manifest.json`
6. **Core Codebase & Test Suites:**
   - `backend/`, `geospatial/`, `ml/`, `validation/`, `config/`, `tests/`
7. **Official Submission Files:**
   - `dist/SARA_competition_submission.zip` (29.07 MB)
   - `submission_package/` (124 files, 31.57 MB)
8. **EMSR927 Quarantine Safeguards & Attributions:**
   - `validation/raster_validator.py`, Copernicus Sentinel (2026), Copernicus WorldDEM-30, and OpenStreetMap notices.

---

## 3. Duplicate Files & Redundancy Analysis

We identified **245 duplicate hash groups** (excluding `.venv/`), primarily originating from:
1. **Submission Staging Copies:** Identical copies of root configuration and documentation files mirrored in `submission_package/` and `scratch/clean_extract_audit/` (e.g. `README.md`, `requirements*.txt`, `.env.example`, `.gitignore`).
   - *Policy:* Retain `submission_package/` as the official uncompressed release manifest. Prune the temporary scratch copy `scratch/clean_extract_audit/`.
2. **Intermediate Pytest Cache Files:** `.pytest_cache/` copies in root and scratch subfolders.
   - *Policy:* Safe to remove.

---

## 4. Proposed Action Classification

### Category A: SAFE_TO_REMOVE (Actionable in Phase 5)
These files are 100% regeneratable caches or temporary test extraction folders created during earlier audit verification:
1. **Python Bytecode Caches:** All `__pycache__/` directories and `*.pyc`/`*.pyo` files across `backend/`, `geospatial/`, `ml/`, `validation/`, `tests/`, `scripts/`, `scratch/` (~189 files, ~2.5 MB).
2. **Pytest Caches:** `.pytest_cache/` folder in root (~5 files, ~0.01 MB).
3. **Scratch Temporary Extractions:** `scratch/clean_extract_audit/` (124 files, ~31.57 MB, temporary folder created solely for verifying clean zip extraction).
4. **Temporary Log / Temp Files:** Any `*.tmp` or `*.log.tmp` leftover files.

*Total Immediate Safe Savings:* **~318 files, ~34.1 MB**.

### Category B: REVIEW_REQUIRED (User Approval Needed Before Removal)
1. **Intermediate SNAP Raster Workspace (`target.data/`):**
   - Files: `target.data/Amplitude_VH.img` (2.63 GB), `target.data/Amplitude_VV.img` (2.63 GB), total **5.27 GB**.
   - *Rationale:* These are intermediate ENVI binary raster outputs generated by ESA SNAP during raw `.SAFE` terrain correction. The final co-registered 3-band GeoTIFFs in `data/processed/` are already preserved. Removing these would reclaim 5.27 GB, but requires explicit confirmation.
2. **Historical Custom Run Duplicate Outputs (`data/output/run_custom_*`):**
   - Multiple ad-hoc web-app upload test runs (`~1.68 GB` each).
   - *Rationale:* Baseline benchmark run `data/output/run_nepal_unet_eval_20261003/` is preserved. Ad-hoc run folders can be archived or cleaned upon user consent.
3. **Historical Scratch Investigation Scripts (`scratch/*.py`):**
   - *Rationale:* Test/audit scripts used to verify various project components.

### Category C: MUST_KEEP (Strictly Retained)
- All files in `backend/`, `geospatial/`, `ml/`, `validation/`, `config/`, `tests/`, `docs/`, `scripts/`, `experiments/unet/best_model.pt`, `data/raw/`, `data/processed/`, `dist/SARA_competition_submission.zip`, and `submission_package/`.

---

## 5. Potential Dependency & Compatibility Risks

1. **Risk of breaking tests:** Removing `__pycache__` or `.pytest_cache` carries zero risk; pytest and Python automatically recompile bytecode on next invocation.
2. **Risk of breaking submission archive:** `dist/SARA_competition_submission.zip` is completely separate from cache files and remains untouched.
3. **Risk of breaking imports:** Removing `scratch/clean_extract_audit/` does not affect runtime imports, as production code resides in `backend/`, `geospatial/`, and `ml/`.

---

## 6. Execution & Quarantine Plan

1. **Backup/Quarantine Location:** `quarantine_cleanup_backup/` (timestamped and verified readable).
2. **Execution Scope:** Prune only **Category A (SAFE_TO_REMOVE)** files.
3. **Post-Cleanup Verification:** Execute full test suite `pytest tests/ -v` to ensure 100% pass rate (177 tests) and verify FastAPI server `/health` and `/dashboard`.
