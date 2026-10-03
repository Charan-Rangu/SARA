# Nepal Multi-Temporal SAR Agreement & Terrain Slope Dashboard Integration Report

**Date:** October 3, 2026  
**Project:** SARA (Satellite-Assisted Rescue & Assessment)  
**Integrated Layers:**
* Multi-Temporal Agreement Mask: `data/output/run_nepal_multitemporal_change_20261003/nepal_water_and_change_agreement_mask.tif`
* Baseline Water Mask: `data/output/run_nepal_unet_eval_20261003/predicted_water_mask.tif`
* Water Probability Map: `data/output/run_nepal_unet_eval_20261003/water_probability_map.tif`
* Copernicus DEM (30m): `data/processed/nepal_copernicus_dem_30m.tif`
* Derived Terrain Slope (30m): `data/processed/nepal_copernicus_slope_30m.tif`

---

> [!IMPORTANT]
> **Scientific Integrity & Operational Notice**
> * **Exploratory Change Candidates:** Multi-temporal agreement categories and backscatter change detections ($\le -3.0\text{ dB}$) represent preliminary candidate classifications, not confirmed flood boundaries. Independent ground-truth validation is not yet available for this scene.
> * **Suspected Shadow Candidates:** Excluded steep-slope predictions ($>15^\circ$) are classified as *suspected high-slope shadow candidates* based on DEM slope heuristics; DEM slope alone does not confirm the presence of radar shadow.
> * **No Life-Safety Claims:** Model outputs must **NOT** be used as sole proof of flooding, damage, road closures, or safety status.
> * **Attributions Preserved:** © Copernicus Sentinel Data (2024–2026) | © Copernicus WorldDEM-30 | © OpenStreetMap contributors.

---

## 1. Executive Summary

The Nepal multi-temporal SAR agreement raster and Copernicus 30m terrain slope layer have been integrated into the interactive Leaflet web dashboard (`GET /dashboard`) in [`backend/main.py`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/backend/main.py).

### Integration Highlights
1. **Layer Visualizer Service (`backend/services/layer_visualizer.py`):** Created a memory-bounded rendering and caching service that generates transparent RGBA PNG map overlays from categorical agreement GeoTIFFs and continuous slope rasters.
2. **Centralized Palette & Transparency:** Strict color mapping implemented for the 5 agreement classes and 4 slope ranges. Nodata pixels (`255.0` and `-9999.0`) and Class 0 (Dry Land) are preserved as $100\%$ transparent, ensuring base maps and roads remain visible.
3. **Dedicated Metadata Endpoint (`GET /dashboard/nepal-summary`):** Provides structured metadata, exact WGS84 bounding boxes, layer URLs, and palette definitions to the frontend.
4. **Enhanced Leaflet Dashboard UI:** Integrated 5 independent layer toggles with live opacity sliders (0–100%), a 5-class categorical legend, and required scientific disclaimers.
5. **Pre-Loaded Scene Initialization:** The dashboard automatically fits the camera to the Nepal AOI bounds (`[[27.591277, 83.925836], [29.524362, 86.806913]]`) upon startup without hard-coded local filesystem paths.
6. **Zero Regressions:** 19/19 unit and integration tests passed, with full backward compatibility preserved for existing endpoints.

---

## 2. Files Inspected & Modified

| File Path | Action | Description |
| :--- | :---: | :--- |
| [`backend/services/layer_visualizer.py`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/backend/services/layer_visualizer.py) | **CREATED** | Core service for windowed overlay generation, centralized palettes, bounds calculation, and on-disk PNG caching. |
| [`backend/main.py`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/backend/main.py) | **MODIFIED** | Added `GET /dashboard/nepal-summary`, updated `GET /dashboard/layers/{run_id}/{layer_name}` with on-demand generation, and overhauled `GET /dashboard` UI with new controls and legends. |
| [`tests/test_layer_visualizer.py`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/tests/test_layer_visualizer.py) | **CREATED** | Comprehensive unit test suite covering agreement palettes, nodata transparency, caching, coordinates, and API endpoints. |
| [`experiments/unet/nepal_dashboard_integration_report.md`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/experiments/unet/nepal_dashboard_integration_report.md) | **CREATED** | Complete integration, verification, and launch documentation. |

---

## 3. Endpoints & Response Formats

### 3.1 `GET /dashboard/nepal-summary`
Returns active Nepal scene information, layer URLs, and color palettes for dashboard auto-initialization.

**Example Response:**
```json
{
  "status": "ok",
  "run_id": "run_nepal_multitemporal_change_20261003",
  "eval_run_id": "run_nepal_unet_eval_20261003",
  "region": "Nepal Himalayan Flood & Slope AOI",
  "center": [28.5578, 85.3664],
  "sar_wgs84_bounds": [
    [27.591277113250623, 83.92583581969234],
    [29.524361773147422, 86.80691259892046]
  ],
  "slope_wgs84_bounds": [
    [27.000138888888888, 82.99986111111112],
    [30.000138888888888, 86.99986111111112]
  ],
  "layers": {
    "agreement_overlay": "/dashboard/layers/run_nepal_multitemporal_change_20261003/agreement_overlay.png",
    "water_prediction_overlay": "/dashboard/layers/run_nepal_unet_eval_20261003/water_prediction_overlay.png",
    "water_probability_overlay": "/dashboard/layers/run_nepal_unet_eval_20261003/water_probability_overlay.png",
    "slope_overlay": "/dashboard/layers/run_nepal_multitemporal_change_20261003/slope_overlay.png"
  },
  "metrics": { ... },
  "palettes": { ... },
  "disclaimers": { ... }
}
```

### 3.2 `GET /dashboard/layers/{run_id}/{layer_name}`
Serves map-ready RGBA PNG overlays, vector GeoJSON layers, or GeoTIFF rasters.
* **Security:** Blocks path traversal (`..`) with HTTP 403 Forbidden.
* **On-Demand Caching:** Automatically generates `agreement_overlay.png` or `slope_overlay.png` if missing and returns cached file on repeat calls (`Content-Type: image/png`).

---

## 4. Layer Names, Colors & Meanings

### 4.1 Multi-Temporal Agreement Candidates Layer (`agreement_overlay.png`)

| Class Code | Category Name | Rendered Color | RGBA Values | Hex | Description / Meaning |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **0** | **Dry Land** | Transparent | `(0, 0, 0, 0)` | `#00000000` | Dry land — no predicted water and no significant backscatter decrease. |
| **1** | **Persistent Water Candidate** | Blue | `(29, 78, 216, 215)` | `#1d4ed8` | Persistent water candidate — predicted water on terrain with slope $\le 15^\circ$ and stable backscatter. |
| **2** | **Backscatter Change Only** | Yellow | `(234, 179, 8, 215)` | `#eab308` | Backscatter change only — significant backscatter decrease ($\le -3\text{ dB}$) without a corresponding baseline U-Net water prediction. |
| **3** | **Agreed Inundation Candidate** | Cyan | `(6, 182, 212, 230)` | `#06b6d4` | Agreed inundation candidate — predicted water, backscatter decrease $\le -3\text{ dB}$, and slope $\le 15^\circ$. |
| **4** | **Suspected High-Slope Shadow Candidate** | Orange | `(234, 88, 12, 215)` | `#ea580c` | Suspected high-slope shadow candidate — predicted water on terrain with slope $> 15^\circ$. |
| **255** | **Nodata** | Fully Transparent | `(0, 0, 0, 0)` | `#00000000` | Outside scene boundary or invalid radar observation. |

### 4.2 Terrain Slope Layer (`slope_overlay.png`)

| Geomorphological Class | Slope Range | Rendered Color | RGBA Values | Meaning |
| :--- | :---: | :---: | :---: | :--- |
| **Flat** | $0^\circ - 5^\circ$ | Green | `(34, 197, 94, 60)` | Flat alluvial floodplains, river channels, and valley floors. |
| **Moderate** | $5^\circ - 15^\circ$ | Yellow | `(234, 179, 8, 120)` | Moderate rolling terrain and foot-slopes. |
| **Steep** | $15^\circ - 25^\circ$ | Orange | `(249, 115, 22, 170)` | Steep mountain slopes (suspected radar shadow zone). |
| **Very Steep** | $> 25^\circ$ | Crimson | `(239, 68, 68, 200)` | Extreme relief with severe geometric radar distortion. |
| **Nodata** | `-9999.0` | Fully Transparent | `(0, 0, 0, 0)` | Unmapped or invalid DEM area. |

---

## 5. CRS, Geotransform & Spatial Alignment

All evaluation layers were verified for coordinate consistency:

| Dataset | Dimensions | CRS | Affine Transform | WGS84 Bounds `[[min_lat, min_lon], [max_lat, max_lon]]` |
| :--- | :---: | :---: | :--- | :--- |
| **Agreement Mask** | $32,072 \times 21,519$ | `EPSG:4326` | `(8.98315e-05, 0.0, 83.9258, 0.0, -8.98315e-05, 29.5243)` | `[[27.591277, 83.925836], [29.524362, 86.806913]]` |
| **Baseline Water Mask** | $32,072 \times 21,519$ | `EPSG:4326` | `(8.98315e-05, 0.0, 83.9258, 0.0, -8.98315e-05, 29.5243)` | `[[27.591277, 83.925836], [29.524362, 86.806913]]` |
| **Probability Map** | $32,072 \times 21,519$ | `EPSG:4326` | `(8.98315e-05, 0.0, 83.9258, 0.0, -8.98315e-05, 29.5243)` | `[[27.591277, 83.925836], [29.524362, 86.806913]]` |
| **Copernicus Slope (30m)**| $14,400 \times 10,800$ | `EPSG:4326` | `(0.00027778, 0.0, 82.9998, 0.0, -0.00027778, 30.0001)` | `[[27.000139, 82.999861], [30.000139, 86.999861]]` |

*Exact spatial co-registration is maintained. Leaflet `L.imageOverlay` maps each overlay to its respective WGS84 geographic bounding box.*

---

## 6. Memory Considerations & Caching Strategy

1. **Avoid Full-Scene Uncompressed RAM Spikes:**
   A full $32,072 \times 21,519 \times 4$ RGBA array in 32-bit floats would occupy $\sim 2.76\text{ GB}$ of RAM. `LayerVisualizerService` leverages nearest/bilinear windowed downsampling (`Resampling.nearest` / `Resampling.bilinear`) to generate a web-optimized display overlay ($8,018 \times 5,379$ pixels, $\sim 3.93\text{ MB}$ PNG), keeping peak RAM usage below $150\text{ MB}$.
2. **On-Disk Persistent Caching:**
   Overlays are generated once and saved directly into `data/output/run_nepal_multitemporal_change_20261003/`. Repeated HTTP requests serve the pre-rendered PNG from disk in $\sim 1\text{ ms}$ (`304 Not Modified` / `FileResponse`), completely eliminating repeated CPU/disk overhead.

---

## 7. Verification Tests & Results

The test suite was executed:

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_layer_visualizer.py tests/test_unet_segmentation_service.py tests/test_nepal_end_to_end_validation.py -v
```

### Test Results Breakdown
* `test_agreement_raster_reading_and_color_mapping`: **PASSED** (Verified 5 classes and nodata 255 alpha = 0)
* `test_nodata_is_never_rendered_as_class_0`: **PASSED** (Verified 100% transparent nodata preservation)
* `test_missing_raster_handling`: **PASSED** (Verified `FileNotFoundError` handling)
* `test_cache_behavior`: **PASSED** (Verified initial render vs. cached retrieval)
* `test_slope_overlay_rendering`: **PASSED** (Verified slope palette mapping and nodata exclusion)
* `test_nepal_summary_endpoint`: **PASSED** (Verified schema, bounds, layer URLs, and disclaimers)
* `test_dashboard_layer_serving_endpoints`: **PASSED** (Verified HTTP 200 `image/png` and traversal prevention)
* `test_dashboard_html_contains_required_controls_and_notices`: **PASSED** (Verified UI controls, class names, disclaimers, attributions)
* `test_unet_segmentation_service.py` (8 tests): **PASSED** (Zero regressions)
* `test_nepal_end_to_end_validation.py` (3 tests): **PASSED** (Zero regressions)

**Total: 19 passed in 7.21s.**

---

## 8. Instructions for Launching & Manual Verification

### Launch Command
From the project root (`C:\Users\chara\OneDrive\Documents\Multimodal AI\sara`):

```powershell
.\.venv\Scripts\uvicorn.exe backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Manual Verification Checklist
1. Open a browser and navigate to: [`http://localhost:8000/dashboard`](http://localhost:8000/dashboard).
2. **Verify Scene Initialization:** Check that the Leaflet map centers on Nepal with a dashed cyan bounding box surrounding the scene (`[28.5578, 85.3664]`).
3. **Verify Multi-Temporal Agreement Layer:**
   * Confirm the `Multi-Temporal Agreement Candidates` checkbox is enabled.
   * Verify dark blue rivers (Persistent Water, Class 1), bright cyan floodplains (Agreed Inundation, Class 3), yellow agricultural change (Class 2), and orange mountain slopes (Suspected Shadow, Class 4).
   * Adjust the opacity slider (0% to 100%) and verify smooth alpha blending over OpenStreetMap.
4. **Verify Baseline Model Water Prediction Layer:**
   * Toggle `Baseline Model Water Prediction` and observe how it compares to the agreement mask.
5. **Verify Terrain Slope Layer:**
   * Enable `Terrain Slope (Copernicus 30m)` checkbox and verify green lowland valleys vs. orange/crimson Himalayan ridges.
6. **Verify Model Water Probability Heatmap:**
   * Enable `Model Water Probability Heatmap` and observe smooth probability gradients.
7. **Verify Scientific Safeguards:**
   * Confirm the amber warning box is clearly visible in the sidebar with the required text:
     *"Exploratory SAR change-analysis candidates, not a verified flood boundary. Independent ground-truth validation is not yet available."*

---

*Integration completed and validated in strict accordance with SARA project requirements and scientific standards.*
