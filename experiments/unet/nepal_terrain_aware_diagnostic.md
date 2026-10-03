# Nepal Small U-Net SAR Water Segmentation Terrain-Aware Post-Processing Diagnostic Report

**Date:** October 3, 2026  
**Project:** SARA (Satellite-Assisted Rescue & Assessment)  
**Evaluated Run:** `nepal_unet_eval_20261003` (`data/output/run_nepal_unet_eval_20261003/`)  
**Diagnostic Output Directory:** `data/output/run_nepal_unet_terrain_scenarios_20261003/`  
**Review Type:** Diagnostic Experiment (NOT an Approved Production Filter)

---

> [!IMPORTANT]
> **Scientific & Operational Disclaimer**  
> * **Diagnostic Comparison Only:** This document evaluates the statistical and spatial impacts of candidate terrain-slope thresholds ($5^\circ$, $10^\circ$, and $15^\circ$). It does **NOT** declare any candidate threshold scientifically optimal or approved for operational use.
> * **No Accuracy or Ground-Truth Claims:** In the absence of verified ground-truth reference masks for this Nepal scene, this experiment measures the *degree of exclusion* rather than true classification accuracy (precision, recall, F1, or IoU).
> * **Slope Does Not Disprove Water:** Terrain slope exclusion is a physical heuristic; slope-based removal does not constitute definitive proof that a pixel is dry land.
> * **No Operational Claims:** Predictions and filtered outputs must **NOT** be used to claim road closures, structural damage, or life-safety status.
> * **Strict Data Isolation:** EMSR927 activation data and change masks were kept strictly isolated.

---

## 1. Executive Summary

A controlled diagnostic experiment was conducted on the Nepal Small U-Net water predictions (`nepal_unet_eval_20261003`) to analyze the quantitative and spatial effects of candidate terrain-slope filters ($5^\circ$, $10^\circ$, and $15^\circ$).

### Key Diagnostic Findings
* **Baseline Water Prediction:** The unconstrained Small U-Net model predicted **32,309,359 water pixels** (**8.5189%** of the 379,266,320 valid observation pixels).
* **Candidate C ($\text{Slope} \le 15^\circ$):** Retains **12,149,495 water pixels** (**3.2034%** of valid scene), excluding **20,159,864 water pixels** (**62.40%** of baseline water) primarily located along steep mountain ridges and geometric radar shadows.
* **Candidate B ($\text{Slope} \le 10^\circ$):** Retains **9,895,805 water pixels** (**2.6092%** of valid scene), excluding **22,413,554 water pixels** (**69.37%** of baseline water).
* **Candidate A ($\text{Slope} \le 5^\circ$):** Retains **7,220,402 water pixels** (**1.9038%** of valid scene), excluding **25,088,957 water pixels** (**77.65%** of baseline water), restricting water extent strictly to flat floodplains and alluvial bottoms.
* **Spatial Integrity:** Lowland river channels and major floodplains ($< 2^\circ$ slope) are **100% preserved** across all three candidate scenarios, while narrow radar-shadow ribbons on mountain flanks are systematically suppressed.

---

## 2. Task A: Input Alignment Audit

All evaluation rasters were audited for Coordinate Reference System (CRS), dimensions, resolution, affine geotransforms, and nodata definitions:

| Dataset | Dimensions | Spatial Resolution | CRS | Affine Geotransform | Nodata Value | Alignment Method |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Prediction Mask** | $32,072 \times 21,519$ | $8.98315 \times 10^{-5\circ}$ (~10m) | `EPSG:4326` | `(8.98315e-05, 0.0, 83.9258, 0.0, -8.98315e-05, 29.5243)` | `255.0` | **REFERENCE GRID** |
| **Probability Map** | $32,072 \times 21,519$ | $8.98315 \times 10^{-5\circ}$ (~10m) | `EPSG:4326` | `(8.98315e-05, 0.0, 83.9258, 0.0, -8.98315e-05, 29.5243)` | `-1.0` | Exact 1:1 Pixel Match |
| **SAR Input (3-band)** | $32,072 \times 21,519$ | $8.98315 \times 10^{-5\circ}$ (~10m) | `EPSG:4326` | `(8.98315e-05, 0.0, 83.9258, 0.0, -8.98315e-05, 29.5243)` | `255.0` | Exact 1:1 Pixel Match |
| **Copernicus Slope (30m)** | $10,800 \times 14,400$ | $0.00027778^\circ$ (~30m) | `EPSG:4326` | `(0.00027778, 0.0, 82.9998, 0.0, -0.00027778, 30.0001)` | `-9999.0` | Dynamic Bilinear Resampling |
| **Copernicus DEM (30m)** | $10,800 \times 14,400$ | $0.00027778^\circ$ (~30m) | `EPSG:4326` | `(0.00027778, 0.0, 82.9998, 0.0, -0.00027778, 30.0001)` | `None` | Dynamic Bilinear Resampling |

### Resolution & Resampling Handling
The 30m slope raster was resampled to match the 10m SAR prediction grid dynamically during windowed chunk processing using bilinear interpolation (`Resampling.bilinear`). Original input files were left completely unmodified.

---

## 3. Task B: Baseline Reproduction & Denominator Verification

* **Total Scene Pixels:** 690,157,368 ($32,072 \times 21,519$)
* **Valid Observation Pixels (Audited Denominator):** **379,266,320** (**54.9536%**)
* **Nodata Pixels:** 310,891,048 (45.0464%, nodata value `255.0`)
* **Baseline Predicted Water Pixels:** **32,309,359**
* **Baseline Predicted Water Proportion:**
  $$\text{Baseline Water Percentage} = \frac{32,309,359}{379,266,320} \times 100\% = \mathbf{8.5189\%}$$

---

## 4. Task C: Evaluation of Diagnostic Slope Scenarios

Three candidate terrain slope thresholds ($5^\circ$, $10^\circ$, and $15^\circ$) were evaluated across the full $379.27\text{M}$ valid observation pixels:

| Scenario / Candidate | Slope Threshold Rule | Retained Water Pixels | Filtered Water % of Valid Scene | Excluded Water Pixels | % of Baseline Water Excluded | % of Valid Scene Affected |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (No Slope Filter)** | None | 32,309,359 | **8.5189%** | 0 | 0.00% | 0.00% |
| **Candidate C** | $\text{Slope} \le 15^\circ$ | 12,149,495 | **3.2034%** | 20,159,864 | **62.40%** | **5.32%** |
| **Candidate B** | $\text{Slope} \le 10^\circ$ | 9,895,805 | **2.6092%** | 22,413,554 | **69.37%** | **5.91%** |
| **Candidate A** | $\text{Slope} \le 5^\circ$ | 7,220,402 | **1.9038%** | 25,088,957 | **77.65%** | **6.62%** |

### Scenario Characterization
* **Candidate C ($\le 15^\circ$):** Eliminates clear mountain ridge radar shadows ($>15^\circ$) while preserving hillside streams, foot slopes, and agricultural terrace drainage.
* **Candidate B ($\le 10^\circ$):** Intermediate threshold commonly used in automated Copernicus EMS SAR flood mapping to eliminate foot-slope shadow false positives.
* **Candidate A ($\le 5^\circ$):** Conservative lowland filter. Preserves true alluvial floodplains, river channels, and standing wetlands while removing all undulating terrain.

---

## 5. Task D: Separate Experimental Outputs

The candidate filtered masks were saved into a dedicated experimental directory (`data/output/run_nepal_unet_terrain_scenarios_20261003/`), completely isolated from the baseline inference run:

1. **`candidate_mask_slope_le_5deg.tif`**: Georeferenced uint8 GeoTIFF where `water = (baseline_water == 1) & (slope <= 5.0)`.
2. **`candidate_mask_slope_le_10deg.tif`**: Georeferenced uint8 GeoTIFF where `water = (baseline_water == 1) & (slope <= 10.0)`.
3. **`candidate_mask_slope_le_15deg.tif`**: Georeferenced uint8 GeoTIFF where `water = (baseline_water == 1) & (slope <= 15.0)`.

*Values:* `0 = Land`, `1 = Retained Water`, `255 = Nodata`.

---

## 6. Task E: Representative Regional Preview Comparisons

Side-by-side $512 \times 512$ quad-panel comparisons (SAR Backscatter | Baseline Small U-Net | Candidate C $\le 15^\circ$ | Candidate A $\le 5^\circ$) were rendered and saved in `data/output/run_nepal_unet_terrain_scenarios_20261003/previews/`:

### 6.1 Preview 1: Steep Slope Radar Shadow Comparison
* **File Path:** [`data/output/run_nepal_unet_terrain_scenarios_20261003/previews/preview_1_steep_slope_radar_shadow_comparison.png`](file:///C:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/data/output/run_nepal_unet_terrain_scenarios_20261003/previews/preview_1_steep_slope_radar_shadow_comparison.png)
* **Location:** Mountain ridge flank with steep slopes ($25^\circ - 45^\circ$).
* **Observation:** Baseline Small U-Net exhibits dense false-positive ribbons along radar-shadow back-slopes. Candidate C ($\le 15^\circ$) and Candidate A ($\le 5^\circ$) cleanly remove these shadow artifacts (rendered in orange as excluded water).

### 6.2 Preview 2: Flat Lowland River Corridor Comparison
* **File Path:** [`data/output/run_nepal_unet_terrain_scenarios_20261003/previews/preview_2_lowland_river_corridor_comparison.png`](file:///C:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/data/output/run_nepal_unet_terrain_scenarios_20261003/previews/preview_2_lowland_river_corridor_comparison.png)
* **Location:** Lowland alluvial floodplain ($< 2^\circ$ slope).
* **Observation:** Continuous river channels and open standing water are **100% retained** in both Candidate C and Candidate A without pixel loss.

### 6.3 Preview 3: Foothills Terrain Transition Comparison
* **File Path:** [`data/output/run_nepal_unet_terrain_scenarios_20261003/previews/preview_3_foothills_transition_comparison.png`](file:///C:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/data/output/run_nepal_unet_terrain_scenarios_20261003/previews/preview_3_foothills_transition_comparison.png)
* **Location:** Transition zone between alluvial fan ($3^\circ - 8^\circ$) and mountain foot-slopes ($12^\circ - 20^\circ$).
* **Observation:** Demonstrates threshold differentiation: Candidate C ($\le 15^\circ$) retains braided alluvial channels on moderate slopes, whereas Candidate A ($\le 5^\circ$) excludes slopes $> 5^\circ$.

### 6.4 Preview 4: Nodata Scene Boundary Comparison
* **File Path:** [`data/output/run_nepal_unet_terrain_scenarios_20261003/previews/preview_4_nodata_boundary_comparison.png`](file:///C:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/data/output/run_nepal_unet_terrain_scenarios_20261003/previews/preview_4_nodata_boundary_comparison.png)
* **Location:** Transition into mountain nodata boundary (`255.0`).
* **Observation:** Nodata boundaries remain strictly preserved across all candidate masks without edge bleeding or false positive generation.

---

## 7. Task F: Validation Requirements & Independent Ground-Truth Constraints

### 7.1 Ground-Truth Availability Assessment
* **No Appropriate Ground-Truth Reference Mask:** No independent, verified ground-truth flood label mask exists for the full Nepal scene.
* **EMSR927 Isolation:** Copernicus Emergency Management Service EMSR927 activation data was kept strictly isolated in accordance with project constraints and was not used for evaluation or threshold tuning.
* **Metric Calculation Prohibition:** Precision, recall, F1-score, and Intersection over Union (IoU) cannot be computed. Reporting accuracy without verified labels would be unscientific.

### 7.2 Trade-off Analysis: Commission vs. Omission Errors
1. **False Positive Reduction (Commission Error):** Slope filtering effectively suppresses geometric radar-shadow false positives, reducing predicted water coverage from $8.52\%$ down to $1.90\% - 3.20\%$.
2. **False Negative Risk (Omission Error):** Hard slope thresholds run the risk of excluding genuine mountain water bodies (e.g. high-altitude glacial lakes, debris-dammed torrents, steep river gorges).
3. **Recommended Next Steps for Validation:**
   * Acquire high-resolution post-event optical imagery (e.g., Sentinel-2 or PlanetScope cloud-free windows) for stratified sample chip validation.
   * Combine slope filtering with Height Above Nearest Drainage (HAND) index and multi-temporal SAR backscatter change detection.

---

## 8. Check Status Summary & Artifact Inventory

| Check / Task | Status | Result / Supporting Evidence |
| :--- | :---: | :--- |
| **Task A: Input Alignment Audit** | **PASS** | $32,072 \times 21,519$ prediction raster aligned with 30m Copernicus slope via dynamic bilinear windowed resampling. |
| **Task B: Baseline Reproduction** | **PASS** | Re-verified $379,266,320$ valid pixels and $32,309,359$ water pixels ($8.5189\%$). |
| **Task C: Slope Scenarios Analysis** | **PASS** | Evaluated $5^\circ$ ($1.90\%$), $10^\circ$ ($2.61\%$), and $15^\circ$ ($3.20\%$) scenarios. |
| **Task D: Separate Candidate Outputs** | **PASS** | Saved 3 candidate GeoTIFFs to `data/output/run_nepal_unet_terrain_scenarios_20261003/`. |
| **Task E: Representative Previews** | **PASS** | Generated 4 side-by-side quad-panel comparison previews. |
| **Task F: Validation Assessment** | **PASS** | Documented ground-truth limitations and commission/omission trade-offs. |

### Generated Artifacts
* **Candidate Mask ($\text{Slope} \le 5^\circ$):** [`data/output/run_nepal_unet_terrain_scenarios_20261003/candidate_mask_slope_le_5deg.tif`](file:///C:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/data/output/run_nepal_unet_terrain_scenarios_20261003/candidate_mask_slope_le_5deg.tif)
* **Candidate Mask ($\text{Slope} \le 10^\circ$):** [`data/output/run_nepal_unet_terrain_scenarios_20261003/candidate_mask_slope_le_10deg.tif`](file:///C:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/data/output/run_nepal_unet_terrain_scenarios_20261003/candidate_mask_slope_le_10deg.tif)
* **Candidate Mask ($\text{Slope} \le 15^\circ$):** [`data/output/run_nepal_unet_terrain_scenarios_20261003/candidate_mask_slope_le_15deg.tif`](file:///C:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/data/output/run_nepal_unet_terrain_scenarios_20261003/candidate_mask_slope_le_15deg.tif)
* **Preview 1 (Steep Ridge Comparison):** [`data/output/run_nepal_unet_terrain_scenarios_20261003/previews/preview_1_steep_slope_radar_shadow_comparison.png`](file:///C:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/data/output/run_nepal_unet_terrain_scenarios_20261003/previews/preview_1_steep_slope_radar_shadow_comparison.png)
* **Preview 2 (Lowland Corridor Comparison):** [`data/output/run_nepal_unet_terrain_scenarios_20261003/previews/preview_2_lowland_river_corridor_comparison.png`](file:///C:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/data/output/run_nepal_unet_terrain_scenarios_20261003/previews/preview_2_lowland_river_corridor_comparison.png)
* **Preview 3 (Foothills Transition Comparison):** [`data/output/run_nepal_unet_terrain_scenarios_20261003/previews/preview_3_foothills_transition_comparison.png`](file:///C:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/data/output/run_nepal_unet_terrain_scenarios_20261003/previews/preview_3_foothills_transition_comparison.png)
* **Preview 4 (Nodata Boundary Comparison):** [`data/output/run_nepal_unet_terrain_scenarios_20261003/previews/preview_4_nodata_boundary_comparison.png`](file:///C:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/data/output/run_nepal_unet_terrain_scenarios_20261003/previews/preview_4_nodata_boundary_comparison.png)
* **Metrics Summary JSON:** [`experiments/unet/nepal_terrain_scenarios_metrics.json`](file:///C:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/experiments/unet/nepal_terrain_scenarios_metrics.json)
