# Nepal Small U-Net SAR Water Segmentation Spatial Quality Review

**Date:** October 3, 2026  
**Project:** SARA (Satellite-Assisted Rescue & Assessment)  
**Evaluated Run Directory:** `data/output/run_nepal_unet_eval_20261003/`  
**Target Corrected Input:** `data/processed/unet_inputs/post_calibrated_sigma0_unet_3band_corrected.tif`  
**Generated Previews Directory:** `data/output/run_nepal_unet_eval_20261003/previews/`  
**Review Type:** Read-Only Spatial Quality Review (NOT an Accuracy Assessment)

---

> [!IMPORTANT]
> **Scientific & Operational Disclaimer**  
> This document presents a **read-only spatial quality review** of model predictions from the Small U-Net architecture.  
> * **No Accuracy Assessment:** No accuracy, precision, recall, F1, or IoU metrics are reported because no independent, ground-truth reference mask appropriate for this Nepal scene was used.  
> * **No Life-Safety Claims:** Preliminary SAR model predictions must **NOT** be interpreted as confirmed ground flooding, structural damage, road closures, settlement access limitations, or area safety.  
> * **Strict Data Isolation:** EMSR927 activation data and change masks were kept strictly isolated and were not loaded or used in this evaluation.

---

## 1. Executive Summary

A read-only spatial quality review was conducted on the outputs of the Small U-Net inference run (`nepal_unet_eval_20261003`). 

The review confirmed **complete spatial georeferencing and alignment** across all input and output GeoTIFF rasters ($32,072 \times 21,519$ pixels, `EPSG:4326` CRS). Windowed inference ($256 \times 256$ block size) produced continuous probability and binary prediction fields across block boundaries without grid edge artifacts (boundary probability step error: `0.00717` vs internal noise `0.14804`). 

Out of **379,266,320 valid pixels** ($54.9536\%$ valid scene area), the model predicted **32,309,359 water pixels** ($8.5189\%$ water coverage). Representative 512x512 preview chips confirmed coherent river channel and open water delineation, while nodata mountain regions were cleanly excluded.

---

## 2. Review Methodology

The spatial quality review employed windowed raster processing to audit full-scene predictions without exceeding memory constraints:

1. **Geospatial & Spatial Alignment Verification:** Automated comparison of dimensions, CRS, affine geotransforms, and nodata definitions across input and output rasters.
2. **Windowed Sampling & Preview Generation:** Extracted $512 \times 512$ representative preview chips targeting dense water bodies, river channels, and mountain nodata boundaries.
3. **Artifact Analysis:** Computed pixel-to-pixel horizontal probability gradients across $256 \times 256$ window block boundaries to test for grid discontinuities.
4. **OSM Vector Contextual Analysis:** Overlaid project OSM road network snapshot ($38,286$ road segments) onto predictions to identify road-water intersection candidates.

---

## 3. Spatial Alignment & Georeferencing Audit

Automated verification confirmed identical spatial geometry and georeferencing parameters across all evaluation files:

| File Name | Dimensions | Coordinate Reference System | Affine Geotransform | Nodata Value | Alignment Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `post_calibrated_sigma0_unet_3band_corrected.tif` | $32,072 \times 21,519$ | `EPSG:4326` (WGS 84) | `(8.98315e-05, 0.0, 83.9258, 0.0, -8.98315e-05, 29.5243)` | `255.0` | **REFERENCE** |
| `predicted_water_mask.tif` | $32,072 \times 21,519$ | `EPSG:4326` (WGS 84) | `(8.98315e-05, 0.0, 83.9258, 0.0, -8.98315e-05, 29.5243)` | `255` | **MATCH (PASS)** |
| `water_probability_map.tif` | $32,072 \times 21,519$ | `EPSG:4326` (WGS 84) | `(8.98315e-05, 0.0, 83.9258, 0.0, -8.98315e-05, 29.5243)` | `-1.0` | **MATCH (PASS)** |

---

## 4. Visual Inspection & Representative Map Previews

Preview chips were generated and saved in `data/output/run_nepal_unet_eval_20261003/previews/`:

### 4.1 Preview 1: Dense Water & River Floodplain
* **File Path:** [`data/output/run_nepal_unet_eval_20261003/previews/preview_1_dense_water_floodplain.png`](file:///C:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/data/output/run_nepal_unet_eval_20261003/previews/preview_1_dense_water_floodplain.png)
* **Location:** Window $(x=17500, y=7500, w=512, h=512)$
* **Visual Observation:** The model highlights major river channels and low-backscatter open water bodies (62,872 water pixels in preview window). Water probability transitions smoothly from high confidence ($>0.90$) along deep channels to moderate confidence ($0.50 - 0.70$) along channel edges.

### 4.2 Preview 2: Nodata & Mountainous Edge Boundary
* **File Path:** [`data/output/run_nepal_unet_eval_20261003/previews/preview_2_nodata_boundary.png`](file:///C:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/data/output/run_nepal_unet_eval_20261003/previews/preview_2_nodata_boundary.png)
* **Location:** Window $(x=10000, y=2500, w=512, h=512)$
* **Visual Observation:** Captures the transition between valid SAR observations and masked mountain shadow / scene edge nodata pixels (`255.0`). Nodata pixels are cleanly masked without probability spillover or false positive edge artifacts along invalid boundaries.

---

## 5. Artifact Analysis

### 5.1 Window Boundary Grid Discontinuity Test
To verify that windowed inference ($256 \times 256$ block size with edge padding) did not introduce artificial grid lines or seam artifacts:
* **Internal Pixel Horizontal Difference:** Mean probability step $\Delta P_{\text{internal}} = \mathbf{0.14804}$
* **Block Boundary Column Difference:** Mean probability step $\Delta P_{\text{boundary}} = \mathbf{0.14087}$
* **Boundary Error Delta:** $|\Delta P_{\text{boundary}} - \Delta P_{\text{internal}}| = \mathbf{0.00717} \ll 0.05000$
* **Result:** **PASS**. No window boundary seam artifacts or block grid discontinuities were detected.

### 5.2 Speckle Noise & Terrain Shadow Risk
* **Speckle Noise:** Micro-scale speckle noise is suppressed due to refined Lee speckle filtering during SNAP preprocessing and spatial convolutions in Small U-Net.
* **Topographic Shadow Risk:** In steep Himalayan terrain, radar shadow produces low backscatter similar to calm water. While nodata masking handles sea/border areas, steep mountain shadow slopes require DEM height/hand-slope filtering for operational deployment.

---

## 6. Mapped Exposure & OSM Contextual Review

The project OSM road network snapshot (`data/output/run_ae24c1db/potentially_affected_roads.geojson`, 38,286 road segments) was evaluated against the predicted water mask.

* **Intersections:** Road line segments intersecting predicted water pixels ($\ge 0.50$ probability) were categorized strictly as **Review Candidates**.
* **Operational Caution:** Road-water intersections indicate potential inundated segments under SAR backscatter mapping, but must **NOT** be reported as confirmed road closures, structural washouts, or unpassable routes without ground confirmation.

---

## 7. Inference Metrics & Denominator Verification

### 7.1 Scene Metric Summary
* **Total Scene Area:** 690,157,368 pixels ($32,072 \times 21,519$)
* **Valid Pixels:** 379,266,320 pixels (**54.9536%**)
* **Nodata Pixels:** 310,891,048 pixels (45.0464%, nodata value `255.0`)
* **Predicted Water Pixels:** 32,309,359 pixels
* **Predicted Land Pixels:** 346,956,961 pixels

### 7.2 Denominator Verification
$$\text{Predicted Water Percentage} = \frac{\text{Predicted Water Pixels}}{\text{Valid Observation Pixels}} = \frac{32,309,359}{379,266,320} \times 100\% = \mathbf{8.5189\%}$$

**Verification:** The denominator strictly excludes 310,891,048 nodata pixels (**PASS**). Using total scene area as denominator would erroneously inflate land area and report a distorted $4.68\%$ proportion.

---

## 8. Categorized Summary of Findings

To ensure scientific rigor, findings are explicitly categorized by evidence type:

### A. Executed Automated Checks (Verified Facts)
1. **Spatial Geometry:** $32,072 \times 21,519$ raster shape, `EPSG:4326` CRS, and affine transform match 100% across inputs and outputs.
2. **Binary & Probability Values:** Binary mask values are strictly $\{0, 1, 255\}$; probability values are strictly within $[0.0, 1.0]$ for valid pixels.
3. **Grid Seam Continuity:** Window boundary continuity test passed ($\Delta P = 0.00717$).

### B. Visual Observations (Qualitative Patterns)
1. **Water Delineation:** Coherent delineation of major rivers and open water bodies in low-backscatter valleys.
2. **Nodata Handling:** Clean exclusion of scene edges and mountain nodata zones without border leakage.

### C. Hypotheses Requiring Independent Verification
1. **Topographic Shadow Confusion:** Low-backscatter mountain shadows in steep terrain may be misclassified as water and require DEM-slope filtering.
2. **Seasonal Vegetation Inundation:** Flooded vegetation with double-bounce scattering may be under-detected by pure intensity thresholding.

### D. Unestablished Findings (Cannot Be Established Without Ground Truth)
1. **Classification Accuracy / Metrics:** Precision, recall, F1-score, and IoU cannot be established without independent ground-truth reference masks.
2. **Physical Infrastructure Impacts:** Actual road closures, bridge damage, or flooded settlement counts cannot be confirmed from SAR predictions alone.
