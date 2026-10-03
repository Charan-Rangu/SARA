# SARA — Final Three-Engine Consistency & Scientific Integrity Audit
**Target System:** Satellite-Assisted Rescue & Assessment (SARA)  
**Audit Date:** 2026-10-03  
**Audited Runtime:** Python 3.13.5 (`.venv/Scripts/python.exe`)  
**Test Suite:** 177 passed across unit, component, API, and integration suites

---

## 1. Executive Summary

This document presents the definitive scientific and code consistency audit across SARA's three core processing engines:
1. **Engine 1:** Multi-Temporal SAR Change Detection & 5-Class Agreement Fusion
2. **Engine 2:** Copernicus 30m Topographic Slope & Shadow Disambiguation Engine
3. **Engine 3:** Independent Validation & Evidence-Grounded Situation Reporting Engine

Every equation, threshold condition, coordinate transform, nodata exclusion, and metric denominator has been mathematically verified.

---

## 2. Engine 1: Multi-Temporal SAR Change Detection & Agreement Engine

### 2.1 Mathematical Formulation & Sign Convention
The logarithmic radar backscatter change is defined as:
$$\Delta\sigma^0_{\text{dB}} = \sigma^0_{\text{post}}(\text{dB}) - \sigma^0_{\text{pre}}(\text{dB})$$
For linear-power inputs:
$$\Delta\sigma^0_{\text{dB}} = 10.0 \cdot \log_{10}\left(\frac{\max(P_{\text{post}}, \epsilon)}{\max(P_{\text{pre}}, \epsilon)}\right), \quad \epsilon = 10^{-7}$$

* **Threshold Condition:** Significant backscatter reduction (exploratory flood water candidate) is triggered when:
  $$\Delta\sigma^0_{\text{dB}} \le -3.000\text{ dB}$$
* **Boundary Validation:** Verified that exact boundary values ($\Delta\sigma^0 = -3.0000\text{ dB}$) evaluate to `True`, while values immediately above ($-2.9999\text{ dB}$) evaluate to `False`.

### 2.2 Invalid Observation & Extreme Value Isolation
* Both pre-event and post-event window reads apply `np.isfinite(block)` and `block != nodata`.
* Nodata sentinels (`255`, `-9999.0`), NaNs, $+\infty$, and $-\infty$ are strictly excluded from the valid pixel denominator and preserved as `255` in the output mask.

### 2.3 5-Class Agreement Fusion Formulation
Valid pixels ($\text{Water} \ne 255 \land \text{Change} \ne 255 \land \text{Slope} \ne 255$) are partitioned into mutually exclusive categories:
* **Class 0 (Dry Land / No Change):** $\text{Water} = 0 \land \Delta\sigma^0 > -3.0\text{ dB}$
* **Class 1 (Persistent Water Candidate):** $\text{Water} = 1 \land \text{Slope} \le 15.0^\circ \land \Delta\sigma^0 > -3.0\text{ dB}$
* **Class 2 (Backscatter Change Only):** $\text{Water} = 0 \land \Delta\sigma^0 \le -3.0\text{ dB}$
* **Class 3 (Agreed Inundation Candidate):** $\text{Water} = 1 \land \text{Slope} \le 15.0^\circ \land \Delta\sigma^0 \le -3.0\text{ dB}$
* **Class 4 (Suspected High-Slope Shadow Candidate):** $\text{Water} = 1 \land \text{Slope} > 15.0^\circ$
* **Class 255 (Nodata):** Invalid observation in any input layer.

### 2.4 Resolution of the 1-Pixel Discrepancy
An earlier report noted a 1-pixel difference:
* Class 4 Suspected Shadow: $20,159,864\text{ pixels}$
* 5-Bin Continuous Stratification Water above $15^\circ$: $20,159,865\text{ pixels}$ ($4,941,790 + 15,218,075$)

**Audit Finding:**
1. Direct evaluation across production GeoTIFFs (`predicted_water_mask.tif`, `candidate_mask_slope_le_15deg.tif`, `nepal_water_and_change_agreement_mask.tif`) confirmed **exact equality**:
   $$\text{Total Water on Slope } > 15^\circ = 20,159,864 = \text{Total Class 4 Pixels} = 20,159,864 \quad (\Delta = 0)$$
2. The 1-pixel discrepancy in earlier continuous table stratification arose from floating-point bin edge inclusivity: a pixel with slope precisely at $15.0000^\circ$ was included in the discrete binary mask `slope <= 15.0`, but placed into the steep bin $[15.0, 25.0)$ in the floating-point binning routine.
3. The discrete binary GeoTIFF raster outputs are completely consistent down to the single pixel.

---

## 3. Engine 2: Copernicus 30m Topographic Slope & Shadow Engine

### 3.1 Provenance & Legal Attribution
* **Elevation Source:** Copernicus WorldDEM-30 (Global 30m DEM, 1-arc-second resolution).
* **Required Legal Attribution:**
  > *"Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved."*

### 3.2 Gradient Calculation & Latitude-Aware Metric Scaling
Slope is derived using central finite differences:
$$S = \arctan\left(\sqrt{\left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2}\right) \cdot \frac{180^\circ}{\pi}$$
* For geographic coordinates (`EPSG:4326`), the metric cell dimensions are dynamically computed:
  $$\Delta y_{\text{meters}} = \text{res\_y} \cdot 111,320.0$$
  $$\Delta x_{\text{meters}} = \text{res\_x} \cdot 111,320.0 \cdot \cos(\text{center\_lat})$$
* At Nepal central latitude ($28.5^\circ\text{ N}$), $\cos(28.5^\circ) \approx 0.8788$, scaling $0.000278^\circ \approx 27.18\text{ m}$.

### 3.3 Nodata Boundary Protection
* **Artifact Eliminated:** Raw difference calculations against nodata pixels ($-9999.0$) previously caused false $89.9^\circ$ vertical cliffs along nodata borders.
* **Solution:** Euclidean distance transform extrapolation (`scipy.ndimage.distance_transform_edt`) populates invalid regions with nearest valid values prior to gradient computation, ensuring border pixels calculate slope only against valid adjacent terrain.
* **Output Nodata:** All invalid pixels are restored to `-9999.0` with full alpha transparency ($A=0$) in overlay PNGs.

### 3.4 Slope-Stratified Water Statistics (Nepal Scene)
* **Valid Scene Observations:** $379,266,320\text{ pixels}$
* **Predicted Water:** $32,309,359\text{ pixels}$ ($8.52\%$)
* **Slope Breakdown:**
  * **$0^\circ - 2^\circ$ (Flat):** In-category water rate = **$16.90\%$**; Share of water = **$15.68\%$** ($5,067,233\text{ px}$)
  * **$2^\circ - 5^\circ$ (Gentle):** In-category water rate = **$8.96\%$**; Share of water = **$6.66\%$** ($2,153,169\text{ px}$)
  * **$5^\circ - 15^\circ$ (Moderate):** In-category water rate = **$6.05\%$**; Share of water = **$15.26\%$** ($4,929,092\text{ px}$)
  * **$15^\circ - 25^\circ$ (Steep):** In-category water rate = **$5.71\%$**; Share of water = **$15.30\%$** ($4,941,790\text{ px}$)
  * **$>25^\circ$ (Very Steep):** In-category water rate = **$9.68\%$**; Share of water = **$47.10\%$** ($15,218,075\text{ px}$)
* **Diagnostic Interpretation:** Slope $>15^\circ$ is a geometric heuristic proxy for suspected radar shadow, not a sensor-calibrated shadow detector. High slopes do not mathematically prove false positives.

---

## 4. Engine 3: Independent Validation & Situation Reporting Engine

### 4.1 Independent Binary Metrics Formulation
For binary water classification against ground truth:
$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$
$$F_1 = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}, \quad \text{IoU} = \frac{TP}{TP + FP + FN}$$
$$\text{Accuracy} = \frac{TP + TN}{TP + FP + FN + TN}$$

### 4.2 Zero-Denominator Conventions
* If $TP + FP = 0$: $\text{Precision} = \text{None}$ (`precision_undefined=True`).
* If $TP + FN = 0$: $\text{Recall} = \text{None}$ (`recall_undefined=True`).
* If $TP + FP + FN = 0 \land TN > 0$: $\text{IoU} = 1.0$ (perfect background agreement).
* If total valid $= 0$: Status is `UNDEFINED_ZERO_VALID_PIXELS`.

### 4.3 Strict Data Quarantine & Ground Truth Lineage
* **EMSR927 Quarantine:** Copernicus EMS activation data is strictly isolated. `RasterValidator` immediately blocks any reference matching `EMSR927` or `EMS` with status `QUARANTINED_REFERENCE_DATA`.
* **Nepal Validation Status:**
  $$\textbf{INDEPENDENT VALIDATION: PENDING — suitable independent reference masks are unavailable.}$$
  No fabricated accuracy, precision, recall, or confusion matrices are reported for the Nepal event.
* **Situation Report Provenance:** [`backend/services/situation_report.py`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/backend/services/situation_report.py) binds metrics to the exact `run_id`, distinguishes simulated scenarios, labels modeled flow paths as topological gradients, and includes mandatory system limitation notices.

---

## 5. Summary Status

* **Engine 1 (SAR Change Detection):** **`WORKING AND VERIFIED`**
* **Engine 2 (Copernicus Slope & Shadow):** **`WORKING AND VERIFIED`**
* **Engine 3 (Validation & Situation Reporting):** **`WORKING AND VERIFIED`**
* **Full Automated Test Suite:** **177 passed, 3 warnings in 67.01s**
