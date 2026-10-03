# SARA — Competition Submission & Packaging Checklist
**Project:** Satellite-Assisted Rescue & Assessment (SARA)  
**Submission Target:** Multi-Modal AI / Space Track Challenge  
**Date:** 2026-10-03  

---

## 1. Pre-Submission Verification Checklist

### A. Code & Test Integrity
- [x] **Full Regression Test Suite:** All 177 automated tests pass (`pytest tests/ -v`).
- [x] **Dedicated Three-Engine Tests:** All 11 three-engine verification tests pass (`pytest tests/test_three_engines_verification.py -v`).
- [x] **No Fabricated Benchmarks:** U-Net timing, FLOPs, and parameter counts reflect real measurements (Full Nepal scene inference: $22.72\text{ s}$; Small U-Net: $1.93\text{M}$ parameters, $69.7\text{ GFLOPs}$).
- [x] **No Circular Reasoning:** Model agreement layers and slope filters are never presented as independent ground-truth validation.
- [x] **Zero Division Protection:** All validation formulas handle empty masks and all-nodata scenarios cleanly.

### B. Security, Credentials & Clean Repository
- [x] **No Plaintext API Keys:** Checked `.env.example` — contains only placeholders.
- [x] **Gitignore Validation:** `.gitignore` excludes `.env`, `*.log`, `__pycache__`, `data/raw/*`, `data/processed/*`, and large multi-gigabyte `.tif` outputs while preserving `.gitkeep`.
- [x] **No Massive Files in Commit History:** Verified that 6.45GB GeoTIFFs and raw SAFE archives are kept in local `data/` and excluded from version control.

### C. Scientific & Legal Compliance
- [x] **Copernicus Attribution:** Preserved mandatory Copernicus Sentinel-1 notice:
  > *"Contains modified Copernicus Sentinel data 2026."*
- [x] **Copernicus WorldDEM-30 Attribution:** Preserved mandatory DEM copyright notice:
  > *"Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved."*
- [x] **OpenStreetMap Attribution:** Preserved OpenStreetMap notice:
  > *"© OpenStreetMap contributors."*
- [x] **EMSR927 Data Quarantine:** Copernicus EMS / EMSR927 activation data is strictly isolated from inference, training, threshold tuning, and evaluation.
- [x] **Caveat Visibility:** System limitation statements are prominently displayed in the Dashboard UI and generated Situation Reports:
  > *"SARA produces preliminary geospatial assessments. Backscatter changes and flow paths are not, by themselves, proof of confirmed flooding, infrastructure destruction, or road closure."*

---

## 2. Sample Data & Offline Demo Packages

The repository includes lightweight sample fixtures so judges can test all components immediately without downloading multi-gigabyte satellite products:

| Sample Asset Path | Asset Type | Purpose | Size |
| :--- | :--- | :--- | :--- |
| `data/ml/sen1floods11/images/Bolivia_001_sar.tif` | 3-Band Calibrated SAR | Small U-Net live segmentation demo | ~500 KB |
| `data/output/run_nepal_multitemporal_change_20261003/agreement_overlay.png` | RGBA PNG | Nepal 5-class agreement map overlay | ~4.1 MB |
| `data/output/run_nepal_multitemporal_change_20261003/slope_overlay.png` | RGBA PNG | Nepal Copernicus slope overlay | ~7.5 MB |
| `data/output/run_nepal_unet_eval_20261003/water_prediction_overlay.png` | RGBA PNG | Nepal Small U-Net water overlay | ~2.5 MB |
| `data/output/run_nepal_multitemporal_change_20261003/situation_report_full.md` | Markdown SitRep | Evidence-grounded situation report | ~3.2 KB |

---

## 3. Host System Execution Instructions for Judges

### 1. Environment Setup
```powershell
# Navigate to project root
cd "C:\Users\chara\OneDrive\Documents\Multimodal AI\sara"

# Activate Python virtual environment
.\.venv\Scripts\Activate.ps1
```

### 2. Run Complete Automated Test Suite
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```
*(Expected Result: 177 passed, 3 warnings)*

### 3. Launch Backend & Dashboard
```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
Open your browser and navigate to:
```
http://localhost:8000/dashboard
```

---

## 4. Manual Pre-Submission Steps for Charan

1. **Verify Git Status:**
   ```powershell
   git status
   ```
   Ensure no untracked `.env` files with real credentials or large `.tif` binaries are staged.
2. **Review Dashboard Live in Browser:**
   * Open `http://localhost:8000/dashboard`.
   * Verify the 5-Class Agreement Layer, Slope Heatmap, and Small U-Net Water Layer toggle and render smoothly with opacity sliders.
   * Open the **Operational SitRep** tab and inspect the evidence-grounded report.
   * Open the **Model Zoo & Validation** tab and confirm cross-disaster metrics (Somalia) and the `PENDING` status for Nepal ground truth.
3. **Zip/Commit Project:**
   * Ensure `docs/`, `backend/`, `geospatial/`, `ml/`, `validation/`, and `tests/` are included.
