# SARA — Live Technical Demonstration Script
**Satellite-Assisted Rescue & Assessment**
*Target Duration: 5 to 7 Minutes | Live Technical & Academic Review*

---

## Pre-Demonstration Setup & Launch Commands

Ensure your terminal is situated at the project root directory on Windows:
```powershell
# Navigate to repository root
cd "C:\Users\chara\OneDrive\Documents\Multimodal AI\sara"

# Verify Python virtual environment
.\.venv\Scripts\python.exe --version

# Launch FastAPI Application & Visualization Server
.\.venv\Scripts\uvicorn.exe backend.main:app --host 0.0.0.0 --port 8000 --reload
```

* **Dashboard Web URL**: `http://localhost:8000/dashboard`
* **Swagger API Docs**: `http://localhost:8000/docs`
* **Pre-flight Sanity Check**: Confirm browser renders the dark-theme Leaflet dashboard centered over Nepal with the summary analytics panel loaded on the left.

---

## Step-by-Step Demonstration Walkthrough

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                           5-7 MINUTE DEMO TIMELINE                            │
├─────────┬───────────────────────────────┬─────────────────────────────────────┤
│ 0:00    │ Introduction & Problem Context│ Monsoon flooding in high-relief     │
│ 0:45    │ System Architecture           │ Dual-engine: AI + SAR change + DEM  │
│ 1:30    │ Sentinel-1 SAR Input Data     │ 3-band calibrated GeoTIFF           │
│ 2:15    │ Baseline U-Net Prediction     │ Binary mask & Probability heatmap   │
│ 3:15    │ Multi-Temporal Agreement Map  │ 5-Class fusion overlay & analytics  │
│ 4:15    │ Terrain Slope Disambiguation  │ Copernicus 30m DEM & Shadow proxy   │
│ 5:15    │ API & Automated Test Results  │ FastAPI docs & 121 passing tests    │
│ 6:15    │ Scientific Integrity & Wrap   │ Disclaimers, ground truth, future   │
└─────────┴───────────────────────────────┴─────────────────────────────────────┘
```

---

### Step 1: Introduction & Problem Statement (0:00 – 0:45)
* **What to Show**: Dashboard main view at `http://localhost:8000/dashboard`.
* **What to Click**: Leave default view loaded; point to the title banner **"SARA — Satellite-Assisted Rescue & Assessment"**.
* **What to Say**:
  > *"Good morning. During major disaster events—such as the September 2024 Nepal monsoon deluge—optical satellites are blinded by dense cloud cover and heavy rainfall. First responders urgently need rapid, all-weather spatial intelligence to understand where water has inundated terrain.
  > SARA is an open-source geospatial platform combining Copernicus Sentinel-1 Synthetic Aperture Radar (SAR), Copernicus 30m Digital Elevation Models, and Deep Convolutional Neural Networks to rapidly map standing water, isolate flood changes, and filter out false alarms caused by mountainous radar shadow."*

---

### Step 2: System Architecture & Dual-Engine Concept (0:45 – 1:30)
* **What to Show**: Point to the left sidebar showing **"Summary Analytics"** and the layer toggle controls.
* **What to Explain**:
  > *"SARA does not rely on deep learning alone. Standard segmentation models struggle in complex terrain like the Himalayas because mountain radar shadows look identical to calm water. SARA implements a dual-engine architecture:
  > - Engine A uses a PyTorch Small U-Net to segment water from multi-polarization SAR backscatter ($VV, VH, VV-VH$).
  > - Engine B performs multi-temporal change detection against a pre-event baseline and integrates Copernicus 30m DEM slope analysis.
  > These two engines converge into an exploratory 5-class agreement raster that provides actionable, disambiguated intelligence."*

---

### Step 3: Sentinel-1 SAR Input Data & Radiometric Calibration (1:30 – 2:15)
* **What to Show**: In the **"Model Inference Trigger"** section of the sidebar, point out the pre-populated SAR raster path:
  `data/processed/unet_inputs/post_calibrated_sigma0_unet_3band_corrected.tif`
* **What to Explain**:
  > *"Our primary post-event dataset was acquired by Sentinel-1A on September 27, 2024 over central Nepal. The raw radar data was processed through ESA SNAP 10.0 using precise orbit files, thermal noise removal, calibration to Sigma-0, speckle filtering, and Range-Doppler Terrain Correction to WGS84.
  > The resulting raster spans 32,072 by 21,519 pixels—over 690 million pixels. It features three calibrated channels: $VV$ backscatter in dB, $VH$ in dB, and the polarization difference $VV - VH$."*

---

### Step 4: Baseline U-Net Water Segmentation & Probability Heatmap (2:15 – 3:15)
* **What to Click**:
  1. Uncheck **"5-Class Agreement Mask"**.
  2. Check **"Baseline Model Water Mask (Binary)"** $\to$ slide opacity to **80%**.
  3. Check **"Model Water Probability (0.0 - 1.0)"** $\to$ slide opacity to **60%**.
* **What to Show**: The blue binary water overlay and the purple-to-cyan continuous probability heatmap over Nepal.
* **What to Explain**:
  > *"Here is the raw output from our Small U-Net model. It processed the entire 690-million-pixel scene using windowed 512x512 inference.
  > Notice that while it accurately identifies permanent lakes, river valleys, and inundated lowlands (8.52% of the scene), you can see substantial false alarms in the northern mountain ranges. Because steep back-slopes cast radar shadows that return zero signal, the U-Net naturally flags these dark shadows as water. This is why raw AI predictions alone are insufficient in mountainous regions."*

---

### Step 5: Multi-Temporal Agreement Map & 5-Class Breakdown (3:15 – 4:15)
* **What to Click**:
  1. Uncheck **"Baseline Model Water Mask"** and **"Model Water Probability"**.
  2. Check **"5-Class Agreement Mask"** $\to$ slide opacity to **90%**.
  3. Point to the **"Multi-Temporal Agreement Classes"** in the sidebar.
* **What to Show**: The 5-class color-coded overlay on the Leaflet map (Blue, Yellow, Cyan, Orange).
* **What to Explain**:
  > *"This is SARA's synthesized 5-Class Agreement Layer:
  > - **Class 0 (Transparent, 79.2%)**: Stable dry land.
  > - **Class 1 (Blue, 2.33%)**: Persistent water candidates—rivers and lakes present before and after the storm.
  > - **Class 2 (Yellow, 12.28%)**: Backscatter drop only ($\le -3\text{ dB}$)—representing agricultural changes, wet soil, or plowing where the model did not predict standing water.
  > - **Class 3 (Cyan, 0.88%)**: **Agreed Inundation Candidates**—pixels where the U-Net predicted water, backscatter dropped by at least 3 dB, and slope is $\le 15^\circ$. This isolates the most probable flood zones.
  > - **Class 4 (Orange, 5.32%)**: **Suspected High-Slope Radar Shadow**—water predictions on mountain slopes $> 15^\circ$, successfully flagging false alarms."*

---

### Step 6: Terrain Slope Disambiguation (4:15 – 5:00)
* **What to Click**:
  1. Check **"Terrain Slope (Copernicus 30m)"** $\to$ slide opacity to **50%**.
  2. Zoom into a high-relief mountain ridge showing Orange (Class 4) coinciding with steep red/orange slope gradients.
* **What to Explain**:
  > *"By overlaying the Copernicus 30m DEM slope gradient (Green for flat land $<5^\circ$, Red for steep cliffs $>25^\circ$), we can visually confirm that the Class 4 orange pixels align exactly with rugged mountain ridges. Liquid water cannot pool on a 25-degree cliff. Our terrain disambiguation engine isolates over 20 million false-alarm pixels without discarding real valley inundation."*

---

### Step 7: Eight Navigation Tabs & Operational Situation Report (5:00 – 6:00)
* **What to Click**: Click through the top navigation tabs:
  1. **Map Explorer:** Live Leaflet map with multi-layer overlays and telemetry.
  2. **SAR Preprocessing:** SNAP pipeline status and graph verification.
  3. **Discovery & Ingest:** Copernicus CDSE catalog search with storage safeguards.
  4. **Multi-Date Change:** Dual-date SAR change detection engine.
  5. **Terrain & Flow:** DEM slope, aspect, and D8 flow path tracing.
  6. **Exposure & Roads:** Vector OSM infrastructure exposure and NetworkX reachability.
  7. **Operational SitRep:** Deterministic, evidence-grounded Markdown situation report generation.
  8. **Model Zoo & Validation:** Model benchmark comparison and Sen1Floods11 validation metrics.
* **What to Explain**:
  > *"SARA generates structured Markdown situation reports bound to exact run IDs. Notice that when external data is unavailable, SARA explicitly outputs `NOT_AVAILABLE` or `PENDING` rather than inventing synthetic numbers. We strictly isolate quarantined Copernicus EMS data and display prominent limitation notices."*

---

### Step 8: API Architecture & Automated Test Verification (6:00 – 6:45)
* **What to Show**:
  1. Open a new tab to `http://localhost:8000/docs` (Swagger UI).
  2. Switch to terminal and run the test suite:
     ```powershell
     .\.venv\Scripts\python.exe -m pytest tests/ -v
     ```
* **What to Say**:
  > *"The entire system is powered by a modular FastAPI backend with streaming raster decimation, disk-cached PNG rendering, and REST endpoints for catalog discovery, routing exposure, and situation reporting.
  > Our automated test suite executes 177 tests covering unit segmentation, raster math, coordinate transforms, dedicated three-engine verification, and API endpoints with 100% passing results."*

---

### Step 9: Scientific Limitations, Ethics & Conclusion (6:45 – 7:15)
* **What to Show**: Point to the **"Scientific & Operational Notice"** disclaimer box at the bottom of the dashboard sidebar.
* **What to Say**:
  > *"In conclusion, SARA demonstrates how combining deep learning with domain-specific physical radar models can overcome satellite AI limitations in extreme topography.
  > We maintain strict scientific transparency: independent Nepal ground-truth flood masks are not currently available, so we treat these outputs as exploratory decision-support candidates, not verified ground truth. The Copernicus EMSR927 activation data remains strictly quarantined.
  > Thank you, and I welcome any technical questions."*

