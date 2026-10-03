# SARA — Project Review Presentation Deck Outline
**Satellite-Assisted Rescue & Assessment: High-Resolution SAR Flood Inundation Screening & Terrain Disambiguation**

---

## Slide 1: Title & Project Overview

* **Slide Title**: SARA: Satellite-Assisted Rescue & Assessment
* **Subtitle**: Dual-Engine Synthetic Aperture Radar (SAR) Flood Inundation Screening & Mountain Terrain Disambiguation
* **Team Information**:
  - *Presenter / Developer*: [Student / Developer Name]
  - *Department / Institution*: Department of Computer Science & Engineering / Multimodal AI Laboratory
  - *Academic Year / Date*: 2026 / Project Review
* **Recommended Visual**:
  - SARA system logo and dashboard map preview showing Nepal Sentinel-1 multi-temporal agreement overlay.
* **Speaker Notes**:
  > "Welcome, members of the evaluation panel. Today I am presenting SARA—Satellite-Assisted Rescue & Assessment—an open-source geospatial platform combining deep convolutional neural networks and physical radar change detection for rapid disaster screening in high-relief mountainous environments."

---

## Slide 2: Problem Statement & Motivation

* **Slide Title**: The Disaster Challenge: Flood Screening Under Adverse Conditions
* **Key Bullet Points**:
  - **Cloud Occlusion in Optical Satellites**: Optical sensors (Sentinel-2, Landsat) cannot see through heavy monsoon clouds or rain storms when floods occur.
  - **SAR All-Weather Capability**: Synthetic Aperture Radar (Sentinel-1 C-band) penetrates clouds, smoke, and nighttime conditions.
  - **The High-Relief Mountain Dilemma**:
    - Calm floodwater causes specular reflection (dark radar returns).
    - Mountain back-slopes cast severe **radar shadows** (zero radar illumination, also dark).
    - Naive deep learning models produce massive false-positive water predictions on dry mountain cliffs.
  - **Operational Need**: Automated, terrain-aware screening to isolate real candidate flood inundation without false alarms.
* **Recommended Visual**:
  - Comparison graphic: Optical cloud blockage vs. SAR microwave penetration vs. diagram of mountain radar shadow.
* **Speaker Notes**:
  > "During extreme flood events like the September 2024 Nepal disaster, optical satellites are completely blind due to cloud cover. While SAR penetrates clouds, its side-looking geometry creates dark radar shadows on mountain slopes that trick standard AI models into predicting water on 30-degree cliffs. Our motivation is to solve this shadow-water ambiguity."

---

## Slide 3: Project Objectives

* **Slide Title**: Key Project Objectives
* **Key Bullet Points**:
  - **Reproducible Data Pipeline**: Automated discovery, download, and SNAP-based radiometric/geometric calibration of Sentinel-1 Level-1 GRD imagery.
  - **Multi-Band Feature Engineering**: Generate 3-band calibrated inputs ($VV_{\text{dB}}, VH_{\text{dB}}, VV-VH_{\text{dB}}$) with statistical normalization.
  - **Deep Semantic Segmentation**: Deploy an optimized Small U-Net for large-scale windowed tile inference ($32,072 \times 21,519$ scene).
  - **Physical Disambiguation Engine**: Implement multi-temporal backscatter log-ratio difference ($\Delta\sigma^0 \le -3\text{ dB}$) and Copernicus 30m DEM slope filtering ($> 15^\circ$).
  - **5-Class Consensus Synthesis**: Formulate an exploratory agreement raster classifying dry land, persistent water, backscatter drop, candidate inundation, and suspected shadow.
  - **Interactive Decision-Support Platform**: Deliver a FastAPI backend and Leaflet web dashboard with real-time layer blending.
* **Recommended Visual**:
  - Objective icon flow: Data Ingestion $\to$ ML Segmentation $\to$ Multi-Temporal Change $\to$ DEM Disambiguation $\to$ Web Dashboard.
* **Speaker Notes**:
  > "We established six core objectives: automating SAR data pipelines, deploying an efficient Small U-Net, implementing multi-temporal backscatter change detection, filtering mountain shadows using Copernicus DEMs, and presenting the results in an interactive web GIS dashboard."

---

## Slide 4: System Architecture

* **Slide Title**: End-to-End System Architecture
* **Key Bullet Points**:
  - **Modular Architecture**: Decoupled data ingestion, machine learning, geospatial analysis, and web presentation layers.
  - **Dual-Engine Fusion**:
    - *Engine A (AI)*: Small U-Net predicting per-pixel water probability $p \in [0.0, 1.0]$.
    - *Engine B (Physical Radar & Topography)*: Multi-temporal $\Delta\sigma^0$ change detection + Copernicus 30m DEM Horn's slope.
  - **High-Performance Serving**: FastAPI REST backend with Rasterio windowed decimation and disk-cached RGBA PNG delivery.
  - **Frontend UI**: Single-page Leaflet application with independent layer toggles and live opacity sliders.
* **Recommended Visual**:
  - SARA full-system architecture diagram (Mermaid flow from `docs/architecture.md`).
* **Speaker Notes**:
  > "Our architecture separates data ingestion, machine learning inference, geospatial terrain analysis, and web visualization. The core innovation is the dual-engine design where deep learning predictions are cross-validated against physical radar backscatter drops and DEM slope constraints."

---

## Slide 5: Data Sources & SAR Preprocessing Pipeline

* **Slide Title**: Copernicus Sentinel-1 Ingestion & Radiometric Calibration
* **Key Bullet Points**:
  - **Data Assets**:
    - Sentinel-1A Pre-Event: 2024-09-15 (Track 85, Descending, IW, VV+VH).
    - Sentinel-1A Post-Event: 2024-09-27 (Track 85, Descending, IW, VV+VH).
    - Topography: Copernicus GLO-30 30-meter DEM (`EPSG:4326`).
  - **ESA SNAP Preprocessing Steps**:
    1. Apply Precise Orbit Ephemerides (POD).
    2. Thermal Noise Removal.
    3. Radiometric Calibration to Sigma Naught ($\sigma^0$).
    4. Refined Lee $7 \times 7$ Speckle Filtering.
    5. Range-Doppler Terrain Correction to WGS84 ($0.00018^\circ$ pixel resolution).
  - **Feature Stack**: $VV_{\text{dB}}$, $VH_{\text{dB}}$, and Difference Band ($VV_{\text{dB}} - VH_{\text{dB}}$) normalized via fixed training statistics.
* **Recommended Visual**:
  - Preprocessing flow diagram and side-by-side display of raw SAR amplitude vs. terrain-corrected $\sigma^0$ in dB.
* **Speaker Notes**:
  > "We ingest co-registered Sentinel-1 acquisitions from before and during the disaster. The raw radar is calibrated using ESA SNAP with precise orbits, thermal noise removal, Refined Lee speckle filtering, and Range-Doppler terrain correction with the Copernicus 30m DEM, yielding a 32,000 by 21,000 pixel 3-channel input."

---

## Slide 6: Small U-Net Segmentation Workflow

* **Slide Title**: Deep Learning Water Segmentation (Small U-Net)
* **Key Bullet Points**:
  - **Model Architecture**:
    - 4-level convolutional encoder-decoder with double convolution blocks (Conv2d, BatchNorm2d, ReLU).
    - Feature channel progression: $3 \to 32 \to 64 \to 128 \to 256 \to \text{Bottleneck } 512$.
    - Skip connections concatenating multi-scale spatial details into decoder blocks.
    - $1 \times 1$ Conv + Sigmoid head outputting probability $p \in [0.0, 1.0]$.
  - **Inference Engine**:
    - Chunked $512 \times 512$ sliding window execution via `backend/services/unet_segmentation_service.py`.
    - Explicit nodata tracking preventing edge and out-of-swath false alarms.
  - **Inference Results (Nepal AOI)**:
    - Processed $379,266,320$ valid pixels ($54.95\%$ valid, $45.05\%$ nodata).
    - Predicted $32,309,359$ water candidate pixels ($8.52\%$, threshold $\tau = 0.50$).
* **Recommended Visual**:
  - Small U-Net network topology diagram + sample water probability heatmap snippet.
* **Speaker Notes**:
  > "Our segmentation engine is a custom Small U-Net with four downsampling stages and skip connections. It performs chunked windowed inference over 690 million pixels with strict nodata boundary handling. At a 0.50 threshold, it identifies 32.3 million candidate water pixels across Nepal."

---

## Slide 7: Multi-Temporal Change & Terrain Disambiguation

* **Slide Title**: Physical Multi-Temporal Analysis & 5-Class Agreement
* **Key Bullet Points**:
  - **Differential Backscatter ($\Delta\sigma^0$)**:
    - Compares 2024-09-27 vs. 2024-09-15: $\Delta\sigma^0_{\text{VV}} = VV_{\text{dB, post}} - VV_{\text{dB, pre}}$.
    - Exploratory threshold: $\Delta\sigma^0 \le -3.0\text{ dB}$ ($50\%$ backscatter power drop).
  - **DEM Slope Masking**: Horn's slope $> 15.0^\circ$ flags mountain radar shadow proxies.
  - **Quantified 5-Class Distribution**:
    - *Class 0 (Dry Land)*: $300.4\text{M}$ px ($79.20\%$) — Stable ground.
    - *Class 1 (Persistent Water)*: $8.8\text{M}$ px ($2.33\%$) — Lakes/rivers.
    - *Class 2 (Backscatter Drop Only)*: $46.6\text{M}$ px ($12.28\%$) — Agricultural/soil moisture.
    - *Class 3 (Agreed Inundation)*: $3.3\text{M}$ px ($0.88\%$) — High-confidence flood candidates.
    - *Class 4 (Suspected Shadow)*: $20.2\text{M}$ px ($5.32\%$) — Mountain false alarms isolated.
* **Recommended Visual**:
  - 5-Class agreement decision tree + Nepal agreement raster map screenshot (`nepal_change_detection_map.png`).
* **Speaker Notes**:
  > "By subtracting pre-event from post-event radar backscatter and applying a -3 dB threshold alongside a 15-degree DEM slope filter, we synthesize our 5-class agreement raster. Most importantly, Class 3 isolates 3.3 million pixels of agreed flood candidates, while Class 4 successfully isolates 20.2 million pixels of mountain shadow false alarms."

---

## Slide 8: Interactive Web GIS Dashboard & System Demo

* **Slide Title**: Interactive Leaflet Dashboard & API Infrastructure
* **Key Bullet Points**:
  - **Modern UI / UX**: Embedded Leaflet 1.9.4 dashboard with dark-mode glassmorphism styling.
  - **Dynamic Multi-Layer Controls**:
    - Real-time layer visibility toggles for 5-Class Agreement, Binary Mask, Probability, and DEM Slope.
    - Continuous opacity sliders ($0\% - 100\%$) for intuitive layer blending.
  - **Live Analytics Panel**: Dynamic KPI cards reporting exact pixel counts and scene distribution percentages.
  - **Decimated Tile Service**: Rasterio-powered decimation delivering cached $1024 \times 1024$ RGBA overlays with sub-10ms latency.
* **Recommended Visual**:
  - High-resolution screenshot of the SARA Leaflet dashboard at `http://localhost:8000/dashboard`.
* **Speaker Notes**:
  > "Here is our live Leaflet dashboard. Emergency planners can toggle each layer independently, adjust opacities to blend radar probability with terrain slope, and review real-time summary statistics. Rasterio decimation ensures full-scene overlays load instantaneously without choking client memory."

---

## Slide 9: Automated Testing & Scientific Limitations

* **Slide Title**: Validation, Automated Testing & Scientific Boundaries
* **Key Bullet Points**:
  - **Rigorous Test Suite**:
    - **121 automated unit and integration tests passing (0 failures)**.
    - Full coverage across segmentation service, layer visualizer, DEM slope, and FastAPI endpoints.
  - **Transparent Scientific Boundaries**:
    - **No Ground Truth Fabrication**: Independent Nepal flood ground-truth masks are not currently available in the public domain; ungrounded F1/IoU metrics are strictly withheld.
    - **Quarantined Validation Data**: Copernicus EMS activation EMSR927 is kept strictly isolated.
    - **Exploratory Status**: Class 3 represents candidate inundation screening, not legally verified flood extent.
* **Recommended Visual**:
  - Terminal screenshot showing `121 passed in 68.65s` + table of evidence levels (Model vs. Change vs. Ground Truth).
* **Speaker Notes**:
  > "Our software reliability is backed by 121 automated tests covering every pipeline component. We adhere to strict scientific transparency: because independent ground truth is unavailable for this event, we disclose that these are exploratory candidate layers and do not invent artificial accuracy metrics. EMSR927 data remains strictly quarantined."

---

## Slide 10: Future Scope & Conclusion

* **Slide Title**: Future Directions & Conclusion
* **Key Bullet Points**:
  - **Multi-Pass SAR Fusion**: Integrating ascending and descending Sentinel-1 orbit passes to eliminate shadow blind spots on opposing mountain faces.
  - **Hydrological Network Routing**: Coupling candidate water masks with OpenStreetMap road graphs to automate safe evacuation corridor calculations.
  - **Multi-Sensor Integration**: Fusing SAR screening with post-event cloud-free optical imagery (Sentinel-2 / PlanetScope) when available.
  - **Edge / Cloud Scalability**: Containerized deployment on Kubernetes for global real-time disaster activations.
  - **Key Takeaway**: SARA demonstrates that combining deep learning with domain-specific physical radar models provides robust, terrain-resilient flood screening.
* **Recommended Visual**:
  - Summary visual of future roadmap: Multi-orbit fusion $\to$ Road graph re-routing $\to$ Global disaster response.
* **Speaker Notes**:
  > "To conclude, SARA bridges the gap between deep learning and physical radar science, overcoming the mountain shadow challenge for flood screening. Next steps include dual-orbit ascending/descending fusion and automated road graph evacuation routing. Thank you for your time, and I am ready for your questions."
