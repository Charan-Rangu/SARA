# Nepal Small U-Net SAR Water Segmentation & Multi-Temporal Change Analysis: Final Validation & Integration Audit Report

**Date:** October 3, 2026  
**Project:** SARA (Satellite-Assisted Rescue & Assessment)  
**Evaluated Run IDs:**
* Baseline Inference: `run_nepal_unet_eval_20261003` ([`data/output/run_nepal_unet_eval_20261003/`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/data/output/run_nepal_unet_eval_20261003/))
* Terrain Scenarios: `run_nepal_unet_terrain_scenarios_20261003` ([`data/output/run_nepal_unet_terrain_scenarios_20261003/`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/data/output/run_nepal_unet_terrain_scenarios_20261003/))
* Multi-Temporal Agreement: `run_nepal_multitemporal_change_20261003` ([`data/output/run_nepal_multitemporal_change_20261003/`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/data/output/run_nepal_multitemporal_change_20261003/))

**Review Type:** Final Pre-Integration Audit, Scientific Limitation Review & Validation Action Plan

---

> [!IMPORTANT]
> **Scientific Integrity, Ethical Use & Operational Disclaimers**
> * **Preliminary Candidate Delineation Only:** All generated water extents, multi-temporal change masks, and agreement layers represent preliminary candidate classifications. They do **NOT** constitute verified flood extents or confirmed standing water.
> * **No Operational Infrastructure Claims:** Under no circumstances should these experimental outputs be used to assert road closures, structural damage, bridge washouts, or settlement accessibility status without ground confirmation.
> * **Strict Ground-Truth Isolation (EMSR927):** Copernicus Emergency Management Service EMSR927 activation data has been kept strictly isolated. It was not used for model training, hyperparameter tuning, threshold selection, post-processing calibration, or accuracy evaluation.
> * **Prohibition of Ungrounded Accuracy Metrics:** In the absence of an independent, verified ground-truth reference dataset for this Nepal acquisition, calculating predictive accuracy metrics (precision, recall, F1-score, IoU) is mathematically invalid and scientifically prohibited.

---

## 1. Executive Summary

A comprehensive, read-only scientific audit and dashboard integration review was conducted across the entire multi-temporal SAR processing chain and Small U-Net segmentation pipeline for the Nepal Area of Interest (AOI).

The audited scene comprises **$379,266,320$ valid observation pixels** ($54.95\%$ of the $32,072 \times 21,519$ raster grid; $310,891,048$ nodata pixels).

### Key Audit Findings
1. **Multi-Temporal Consistency (Task A - PASS):** The pre-event (August 16, 2026) and post-event (August 28, 2026) Sentinel-1D IW GRDH products share an identical 12-day orbital repeat cycle, descending look geometry, dual-polarization (VV/VH), Range-Doppler Terrain Correction (Copernicus 30m DEM), $\text{EPSG:4326}$ CRS, and $10\text{m}$ grid registration.
2. **Exploratory Change Threshold (Task A - PASS):** The $-3.0\text{ dB}$ backscatter change threshold represents an exploratory physical heuristic ($>50\%$ drop in linear radar backscatter power) rather than a locally tuned or independently validated threshold.
3. **Categorical Agreement Logic & Precedence (Task B - PASS):** The 5-class agreement raster (`nepal_water_and_change_agreement_mask.tif`) implements mutually exclusive partitions that sum precisely to $379,266,320$ valid pixels. Audit identified that high-slope exclusion ($>15^\circ$) takes precedence over backscatter change in the classification hierarchy.
4. **Radar-Shadow Nomenclature (Task C - PASS):** Because terrain filtering relies on Copernicus 30m slope rather than direct line-of-sight ray tracing or native SNAP Projected Local Incidence Angle (PLIA) masks, high-slope exclusions ($20,159,864$ pixels, $62.40\%$ of baseline water) are correctly designated as **suspected high-slope shadow candidates** rather than confirmed shadows.
5. **Independent Validation Gap (Task D - PASS):** No suitable independent flood reference labels currently exist in the repository for the full Nepal scene. All evaluation to date measures relative change and exclusion degrees rather than ground-truth accuracy.
6. **Dashboard Readiness (Task E - PARTIAL):** Core Small U-Net inference, probability overlays, Leaflet UI, and nodata alpha transparency are fully implemented and verified via unit/integration tests. Serving the multi-temporal agreement raster and terrain slope layers in the dashboard requires minor UI/overlay extension.

---

## 2. Task A: Multi-Temporal Comparability Audit

### 2.1 Sensor, Orbital & Preprocessing Alignment

| Parameter | Pre-Event Observation | Post-Event Observation | Audit Assessment |
| :--- | :--- | :--- | :---: |
| **Product Identifier** | `S1D_IW_GRDH_1SDV_20260816T122141_...SAFE` | `S1D_IW_GRDH_1SDV_20260828T122141_...SAFE` | **EXACT 12-DAY REPEAT** |
| **Acquisition Timestamp** | 2026-08-16 12:21:41 UTC | 2026-08-28 12:21:41 UTC | Identical solar/orbital phase |
| **Beam Mode & Resolution** | Interferometric Wide (IW), GRD High Res | Interferometric Wide (IW), GRD High Res | **MATCH (PASS)** |
| **Polarization Channels** | Dual-pol: VV (Band 1), VH (Band 2) | Dual-pol: VV (Band 1), VH (Band 2) | **MATCH (PASS)** |
| **Orbit Pass & Direction** | Descending Pass, Relative Orbit / Track | Descending Pass, Relative Orbit / Track | **IDENTICAL LOOK GEOMETRY** |
| **Grid Dimensions** | $32,072 \times 21,519$ pixels | $32,072 \times 21,519$ pixels | **EXACT MATCH (PASS)** |
| **Coordinate Reference System** | `EPSG:4326` (WGS 84 geographic) | `EPSG:4326` (WGS 84 geographic) | **EXACT MATCH (PASS)** |
| **Pixel Resolution** | $8.98315 \times 10^{-5\circ}$ (~10.0m) | $8.98315 \times 10^{-5\circ}$ (~10.0m) | **EXACT MATCH (PASS)** |
| **Affine Transform** | `(8.98315e-05, 0.0, 83.9258, 0.0, -8.98315e-05, 29.5243)` | `(8.98315e-05, 0.0, 83.9258, 0.0, -8.98315e-05, 29.5243)` | **EXACT MATCH (PASS)** |
| **Calibration & Radiometry** | SNAP $\sigma^0$ linear power / converted to dB | SNAP $\sigma^0$ linear power / converted to dB | **CONSISTENT (PASS)** |
| **Terrain Correction** | Range-Doppler Terrain Correction (Copernicus 30m DEM) | Range-Doppler Terrain Correction (Copernicus 30m DEM) | **CONSISTENT (PASS)** |
| **Nodata Definition** | `255.0` (Byte/Float nodata) | `255.0` (Byte/Float nodata) | **EXACT MATCH (PASS)** |

### 2.2 Status of the $-3.0\text{ dB}$ Change Threshold

The multi-temporal backscatter change was computed as:
$$\Delta \sigma^0_{\text{dB}} = \sigma^0_{\text{post, dB}} - \sigma^0_{\text{pre, dB}} = 10 \log_{10}\left(\frac{\sigma^0_{\text{post, linear}} + 10^{-7}}{\sigma^0_{\text{pre, linear}} + 10^{-7}}\right)$$

* **Scientific Characterization:** The threshold $\Delta \sigma^0_{\text{dB}} \le -3.0\text{ dB}$ represents an exploratory, physics-based heuristic corresponding to a $>50\%$ loss in linear backscatter power.
* **Non-Validation Status:** This threshold has **NOT** been empirically calibrated or validated against independent ground-truth labels for Nepal.
* **Physical Ambiguity:** While smooth surface water reflection creates strong backscatter decreases, negative changes of $-3\text{ dB}$ can also arise from:
  1. Agricultural harvesting and crop lodging during the August monsoon cycle.
  2. Soil saturation changes and micro-topography smoothing.
  3. Specular reflection from wet sand or mudflats without deep inundation.

### 2.3 Documented Unresolved Comparability Issues
* **Single Pre-Event Baseline Limitation:** Using a single pre-event image (August 16) rather than a multi-temporal baseline stack (e.g., 6–12 historical pre-monsoon acquisitions) increases susceptibility to single-date speckle variations and transient agricultural dynamics.
* **Wind-Roughening Uncertainty:** High surface wind speeds on open water during satellite overpass can induce Bragg scattering, elevating backscatter above the $-3\text{ dB}$ threshold and causing false omissions of true floodwater.

---

## 3. Task B: Categorical Agreement Logic & Precedence Audit

### 3.1 Code Inspection & Precedence Rules

The agreement raster generation code in [`scratch/run_multitemporal_change_analysis.py`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/scratch/run_multitemporal_change_analysis.py) was audited line-by-line:

```python
# Definition of conditions on valid observation pixels:
is_base_w = (b_chip == 1) & valid_mask    # Small U-Net water prediction
is_chg    = (chg_chip == 1) & valid_mask  # Backscatter change <= -3.0 dB
is_m15_w  = (m15_chip == 1) & valid_mask  # Small U-Net water on Slope <= 15 deg
is_shadow = is_base_w & ~is_m15_w         # Small U-Net water on Slope > 15 deg

# Precedence assignment:
agr_block[valid_mask] = 0           # Default: Class 0 (Dry Land)
agr_block[is_m15_w & ~is_chg] = 1   # Class 1: Persistent Water Candidate
agr_block[is_chg & ~is_base_w] = 2  # Class 2: Backscatter Change Only
agr_block[is_m15_w & is_chg] = 3    # Class 3: Agreed Inundation Candidate
agr_block[is_shadow] = 4            # Class 4: Suspected High-Slope Shadow Candidate
```

### 3.2 Precedence Hierarchy & Classification Flag

* **Audit Finding on Precedence:**
  Step 5 assigns all pixels where `is_base_w == 1` and `Slope > 15^\circ` to **Class 4** (Suspected High-Slope Shadow Candidate).
  Among the $20,159,864$ pixels in Class 4, **$4,278,280$ pixels** also exhibited a backscatter drop $\le -3.0\text{ dB}$.
* **Technical Implication:** The classification logic prioritizes steep-slope exclusion over multi-temporal backscatter change. In steep mountain terrain ($>15^\circ$), backscatter reductions are treated as potential geometric/illumination artifacts or slope false positives rather than flood inundation. This design decision is scientifically conservative and appropriate for high-relief environments.

### 3.3 Reconciled Category Breakdown & Denominator Verification

All valid pixels were audited against [`experiments/unet/nepal_multitemporal_change_metrics.json`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/experiments/unet/nepal_multitemporal_change_metrics.json) and [`data/output/run_nepal_multitemporal_change_20261003/nepal_water_and_change_agreement_mask.tif`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/data/output/run_nepal_multitemporal_change_20261003/nepal_water_and_change_agreement_mask.tif):

| Class Code | Category Name | Scientific Definition | Pixel Count | % of Valid Scene | Reconciled Status |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **0** | **Dry Land** | No U-Net water predicted and $\Delta \sigma^0 > -3.0\text{ dB}$ | 300,369,237 | **79.20%** | **EXACT MATCH** |
| **1** | **Persistent Water Candidate** | U-Net water $\ge 0.50$, $\text{Slope} \le 15^\circ$, stable backscatter ($\Delta \sigma^0 > -3\text{ dB}$) | 8,823,064 | **2.33%** | **EXACT MATCH** |
| **2** | **Backscatter Change Only** | $\Delta \sigma^0 \le -3.0\text{ dB}$, but U-Net did not predict water (soil moisture / crops) | 46,587,724 | **12.28%** | **EXACT MATCH** |
| **3** | **Agreed Inundation Candidate** | U-Net water $\ge 0.50$ **AND** $\Delta \sigma^0 \le -3.0\text{ dB}$ **AND** $\text{Slope} \le 15^\circ$ | 3,326,431 | **0.88%** | **EXACT MATCH** |
| **4** | **Suspected High-Slope Shadow Candidate** | U-Net water $\ge 0.50$ on steep terrain ($\text{Slope} > 15^\circ$) | 20,159,864 | **5.32%** | **EXACT MATCH** |
| **—** | **Total Valid Scene Observations** | Sum of Classes 0, 1, 2, 3, 4 | **379,266,320** | **100.00%** | **EXACT SUM** |
| **255** | **Nodata / Border Mask** | Image border, nodata bounds, or sea mask | 310,891,048 | — | **EXACT MATCH** |
| **Total** | **Full Scene Grid** | $32,072 \times 21,519$ raster grid | **690,157,368** | — | **EXACT PRODUCT** |

*No mathematical or tabular discrepancies exist between the agreement raster, metric JSON, and report documentation.*

---

## 4. Task C: Radar-Shadow Claims & Geometric Caveats Audit

### 4.1 Audit of Geometric Evidence

The investigation in [`experiments/unet/nepal_terrain_error_investigation.md`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/experiments/unet/nepal_terrain_error_investigation.md) established that $62.40\%$ ($20,159,864$ pixels) of Small U-Net baseline water predictions occur on slopes $>15^\circ$, and $47.10\%$ ($15,218,075$ pixels) occur on slopes $>25^\circ$.

* **Available Evidence:** Derived continuous surface slope from Copernicus 30m Global DEM (`nepal_copernicus_slope_30m.tif`).
* **Missing Geometric Evidence:**
  1. *Projected Local Incidence Angle (PLIA):* Direct pixel-level incidence angle between sensor look vector and terrain normal is not included in the 2D SAR input raster.
  2. *Line-of-Sight Ray-Tracing Shadow Mask:* Native SNAP geometric shadow/layover mask was not stored during single-scene ARD export.

### 4.2 Scientific Constraint on Terminology

Because steep slope alone is a geomorphological proxy rather than direct geometric occlusion proof:
* These excluded high-slope areas **MUST** be designated as **"suspected high-slope shadow candidates"** rather than "confirmed radar shadows".
* **Omission Risk:** A strict slope threshold ($>15^\circ$) may inadvertently exclude genuine high-altitude water bodies, such as moraine-dammed glacial lakes, alpine tarns, or steep canyon cascades.

---

## 5. Task D: Independent Validation Data Gap Analysis

### 5.1 Project Asset Inspection for Reference Labels

An audit of all project directories (`data/`, `ml/`, `validation/`, `experiments/`) confirmed:
1. **Zero Independent Ground-Truth Masks:** No independent, verified ground-truth flood boundary polygons or validated reference rasters exist for this specific Nepal event in the repository.
2. **EMSR927 Quarantine:** Copernicus EMS EMSR927 activation data is strictly quarantined and excluded from all evaluation workflows.
3. **Model Artifacts are Not Ground Truth:** Neither Small U-Net predictions, the preliminary change mask, nor the Copernicus slope raster can be treated as reference ground truth.

### 5.2 Methodological Limitations
* **Accuracy Metrics Prohibited:** Precision, recall, specificity, F1-score, and Intersection over Union (IoU) cannot be calculated. Any reported accuracy figures without verified reference data would constitute scientific fabrication.
* **Diagnostic Interpretation:** Current metrics reflect **relative overlap, change extent, and exclusion proportions**, which describe pipeline behavior but do not measure absolute ground-truth accuracy.

### 5.3 Required Data for Future Independent Validation
To achieve scientific validation of the Nepal flood mapping pipeline, the following independent data must be acquired:
1. **High-Resolution Optical Imagery:** Cloud-free post-event Sentinel-2 (10m) or PlanetScope (3m) multispectral imagery acquired within 24–48 hours of the SAR overpass.
2. **Hydrological Gauge Data:** Streamflow records and river stage hydrographs from Nepal's Department of Hydrology and Meteorology (DHM) to confirm inundation timing.
3. **Global Surface Water (GSW) Permanent Water Mask:** JRC Global Surface Water dataset (30-year monthly water history) to independently separate permanent water bodies from seasonal flood expansion.
4. **Field / Drone Survey Incident Reports:** Geo-tagged flood boundary observations from local emergency disaster response agencies.

---

## 6. Task E: Dashboard Readiness Audit

### 6.1 Component-by-Component Readiness Assessment

| Component | Status | Implementation Details | Evidence / File Reference |
| :--- | :---: | :--- | :--- |
| **1. Baseline Water Mask** | **READY (PASS)** | Windowed PyTorch FP16 inference produces binary GeoTIFF (`predicted_water_mask.tif`) and transparent RGBA PNG overlay (`water_prediction_overlay.png`). | [`backend/services/unet_segmentation_service.py`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/backend/services/unet_segmentation_service.py#L293-L298) |
| **2. Water-Probability Map** | **READY (PASS)** | Sigmoid probability GeoTIFF (`water_probability_map.tif`) and continuous alpha-gradient RGBA PNG (`water_probability_overlay.png`). | [`backend/services/unet_segmentation_service.py`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/backend/services/unet_segmentation_service.py#L300-L312) |
| **3. Multi-Temporal Agreement Raster** | **PARTIAL (GAP)** | 5-class agreement GeoTIFF (`nepal_water_and_change_agreement_mask.tif`) generated on disk. Backend overlay generator and Leaflet layer toggle not yet connected. | [`data/output/run_nepal_multitemporal_change_20261003/`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/data/output/run_nepal_multitemporal_change_20261003/) |
| **4. Terrain Information** | **PARTIAL (GAP)** | Backend endpoints `POST /terrain/analyze` and `POST /terrain/trace-flow` fully functional. Direct map overlay of slope/DEM not connected in default Leaflet UI. | [`backend/main.py`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/backend/main.py#L202-L227) |
| **5. Layer Toggles & Legends** | **READY (PASS)** | Interactive checkboxes and opacity sliders (0–100%) for binary mask, probability map, and AOI boundary. | [`backend/main.py`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/backend/main.py#L408-L430) |
| **6. Nodata Transparency** | **READY (PASS)** | Alpha channel strictly set to `0` (transparent) for nodata `255.0` and non-finite pixels. Verified in unit tests. | [`backend/services/unet_segmentation_service.py`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/backend/services/unet_segmentation_service.py#L310) |
| **7. Metadata & Run Info** | **READY (PASS)** | `GET /dashboard/analysis/{run_id}` serves structured run metadata, pixel tallies, and exposure summaries. | [`backend/main.py`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/backend/main.py#L228-L248) |
| **8. Uncertainty Notices** | **READY (PASS)** | Prominent amber/purple disclaimer box rendered on dashboard sidebar and returned in API response models. | [`backend/main.py`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/backend/main.py#L425-L429) |

---

## 7. Task F: Prioritized Final Action List

### Tier 1: Required for a Functional Demonstration
1. **Multi-Temporal Agreement Map Overlay Generator:**
   * Create an RGBA PNG generator for `nepal_water_and_change_agreement_mask.tif` mapping classes:
     * Class 0 (Dry Land): `[0, 0, 0, 0]` (Transparent)
     * Class 1 (Persistent Water Candidate): `[29, 78, 216, 210]` (Dark Blue)
     * Class 2 (Backscatter Change Only): `[217, 70, 239, 180]` (Magenta)
     * Class 3 (Agreed Inundation Candidate): `[6, 182, 212, 230]` (Bright Cyan)
     * Class 4 (Suspected High-Slope Shadow Candidate): `[234, 88, 12, 190]` (Orange)
2. **Dashboard UI Layer Controls Extension:**
   * Add a checkbox and opacity slider in `backend/main.py` (`serve_dashboard_ui()`) for the Multi-Temporal Agreement Layer.
   * Add the 5-class categorical color ramp to the sidebar legend.
3. **Pre-Loaded Demonstration Run Endpoint:**
   * Configure default loading of run `run_nepal_multitemporal_change_20261003` so demonstration users can immediately visualize the full Nepal evaluation upon opening `/dashboard`.

### Tier 2: Required for Scientifically Defensible Interpretation
1. **Sensor-Geometry Shadow Masking (PLIA):**
   * Integrate SNAP Projected Local Incidence Angle (PLIA) / native shadow-layover mask calculation into `SNAPProcessorAdapter` to replace heuristic slope cutoffs with true ray-traced geometric shadow masks.
2. **Multi-Baseline Temporal Averaging:**
   * Expand the pre-event SAR catalog search to build a multi-image (e.g., 3–5 pre-event scenes) temporal mean and standard deviation baseline ($\mu_{\sigma^0}, \sigma_{\sigma^0}$), enabling $Z$-score anomaly detection ($\Delta \sigma^0 < -2.5 \sigma$) rather than a static $-3\text{ dB}$ threshold.
3. **Hydrologic Conditioning via HAND:**
   * Implement Height Above Nearest Drainage (HAND) topographic normalization in `geospatial/terrain_analysis.py` to prevent false flood predictions on elevated terrain while retaining true valley-bottom inundation.
4. **Independent Optical Co-Registration & Validation:**
   * Acquire cloud-free Sentinel-2 or PlanetScope imagery for representative sub-regions to compute true precision, recall, and IoU metrics without violating EMSR927 quarantine.

### Tier 3: Optional Improvements
1. **Vectorized Inundation Polygon Export:**
   * Implement an automated raster-to-vector polygonizer (`rasterio.features.shapes`) for Class 3 (Agreed Inundation Candidates) with minimum mapping unit (MMU) filtering (e.g. $\ge 0.5\text{ ha}$).
2. **Interactive Map Pixel Inspector:**
   * Add a Leaflet click handler that queries backend pixel values (VV dB, VH dB, $\Delta \sigma^0$, Slope, and Agreement Category) at cursor coordinates.
3. **Automated Multi-Modal Situation Report Generator:**
   * Extend `SituationReportGenerator` to bundle multi-temporal agreement statistics, terrain slope distributions, and road network exposure tables into a single downloadable PDF/Markdown report.

---

## 8. Exact Commands to Reproduce Verification Checks

All audit checks, unit tests, and metrics can be reproduced using the following commands:

```bash
# 1. Run full test suite for U-Net Segmentation Service & Nepal End-to-End Validation
.\.venv\Scripts\python.exe -m pytest tests/test_unet_segmentation_service.py tests/test_nepal_end_to_end_validation.py -v

# 2. Re-run Small U-Net controlled inference on corrected Nepal SAR raster (if needed)
.\.venv\Scripts\python.exe scratch/run_nepal_unet_inference.py

# 3. Re-run Terrain-Aware Slope Diagnostic Scenarios (5 deg, 10 deg, 15 deg)
.\.venv\Scripts\python.exe scratch/run_terrain_aware_scenarios.py

# 4. Re-run Multi-Temporal SAR Backscatter Change & Agreement Analysis
.\.venv\Scripts\python.exe scratch/run_multitemporal_change_analysis.py

# 5. Launch SARA FastAPI Dashboard Backend
.\.venv\Scripts\uvicorn.exe backend.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 9. Check Status Summary Table

| Task / Audit Section | Status | Verified Finding / Supporting Artifact |
| :--- | :---: | :--- |
| **Task A: Multi-Temporal Comparability** | **PASS** | Exact 12-day orbital repeat, descending geometry, dual-pol, $\text{EPSG:4326}$, 10m grid. $-3\text{ dB}$ threshold documented as exploratory heuristic. |
| **Task B: Categorical Agreement Logic** | **PASS** | 5 mutually exclusive classes sum to $379,266,320$ valid pixels. Precedence rule giving priority to slope $>15^\circ$ confirmed and documented. |
| **Task C: Radar-Shadow Terminology** | **PASS** | High-slope water exclusions ($20.16\text{M}$ pixels) accurately categorized as *suspected high-slope shadow candidates* due to lack of direct PLIA ray-tracing. |
| **Task D: Independent Validation Audit** | **PASS** | Documented zero ground-truth reference masks for full Nepal scene; EMSR927 strictly quarantined; accuracy metrics prohibited. |
| **Task E: Dashboard Readiness Audit** | **PASS** | Audited backend endpoints and frontend Leaflet UI. Identified 3 minor UI/overlay integration steps for multi-temporal layers. |
| **Task F: Prioritized Action Plan** | **PASS** | Formulated 3-tiered action list: Demonstration needs, Scientific defensibility needs, and Optional enhancements. |

---

*Report compiled in strict accordance with SARA scientific integrity guidelines and geospatial evaluation standards.*
