# SARA — Nepal Independent Validation Data Discovery & Planning Audit
**Satellite-Assisted Rescue & Assessment: Comprehensive Reference Data Audit, Availability Table & Validation Protocol**

---

## 1. Existing SARA Predictions & Candidate Rasters

To ensure rigorous validation without circular reasoning, SARA explicitly separates its model outputs and exploratory candidate layers from independent ground truth. The existing prediction artifacts for the Nepal study area (`NEPAL_BHOTE_KOSHI_AOI`: `[85.25, 28.10, 85.42, 28.30]` / Full SAR extent: `[[27.591277, 83.925836], [29.524362, 86.806913]]`) are summarized below:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 EXISTING SARA PREDICTION RASTERS                                 │
├──────────────────────────────────────────────────────────────────┬───────────┬───────────────────┤
│ File Path                                                        │ Dtype/Dim │ Scientific Role   │
├──────────────────────────────────────────────────────────────────┼───────────┼───────────────────┤
│ data/output/run_nepal_unet_eval_20261003/                        │ uint8     │ Baseline Semantic │
│   predicted_water_mask.tif                                       │ 32k x 21k │ Water Mask (p>=0.5│
├──────────────────────────────────────────────────────────────────┼───────────┼───────────────────┤
│ data/output/run_nepal_multitemporal_change_20261003/             │ uint8     │ 5-Class Synthesize│
│   nepal_water_and_change_agreement_mask.tif                      │ 32k x 21k │ Agreement Raster  │
├──────────────────────────────────────────────────────────────────┼───────────┼───────────────────┤
│ data/processed/                                                  │ float32   │ Copernicus 30m DEM│
│   nepal_copernicus_dem_30m.tif                                   │ 14k x 10k │ Elevation/Slope   │
└──────────────────────────────────────────────────────────────────┴───────────┴───────────────────┘
```

### 1.1 Detailed Raster Technical Specifications

1. **Baseline U-Net Binary Water Mask (`predicted_water_mask.tif`)**:
   - *Dimensions*: $32,072 \times 21,519$ pixels ($690,157,368$ total grid cells).
   - *Coordinate Reference System (CRS)*: Geographic WGS84 (`EPSG:4326`).
   - *Pixel Resolution*: $8.98315 \times 10^{-5}$ degrees ($\approx 10.0\text{ m}$ spatial resolution).
   - *Pixel Values*: `1` = Predicted Water Candidate ($p \ge 0.50$, $32,309,359$ pixels / $8.52\%$ of valid scene), `0` = Predicted Non-Water, `255` = Nodata ($310,891,048$ pixels / $45.05\%$).
   - *Scientific Interpretation*: Raw probabilistic semantic segmentation inference from the Small U-Net model applied to post-event SAR backscatter ($VV_{\text{dB}}, VH_{\text{dB}}, VV-VH_{\text{dB}}$). Subject to false alarms in mountainous radar shadows.

2. **5-Class Multi-Temporal Agreement Mask (`nepal_water_and_change_agreement_mask.tif`)**:
   - *Dimensions & Grid*: Identical $32,072 \times 21,519$ raster grid in `EPSG:4326`.
   - *Pixel Values (Classes 0–4)*:
     - **Class 0 (Dry Land / No Change)**: $300,369,237$ px ($79.20\%$, Transparent) — Model = 0, $\Delta\sigma^0 > -3\text{ dB}$.
     - **Class 1 (Persistent Water Candidate)**: $8,823,064$ px ($2.33\%$, Blue `#1d4ed8`) — Model = 1, Slope $\le 15^\circ$, stable backscatter ($\Delta\sigma^0 > -3\text{ dB}$).
     - **Class 2 (Backscatter Change Only)**: $46,587,724$ px ($12.28\%$, Yellow `#eab308`) — Model = 0, $\Delta\sigma^0 \le -3\text{ dB}$ (agricultural/soil moisture drop).
     - **Class 3 (Agreed Inundation Candidate)**: $3,326,431$ px ($0.88\%$, Cyan `#06b6d4`) — Model = 1, $\Delta\sigma^0 \le -3\text{ dB}$, Slope $\le 15^\circ$.
     - **Class 4 (Suspected High-Slope Shadow Candidate)**: $20,159,864$ px ($5.32\%$, Orange `#ea580c`) — Model = 1 on steep mountain slopes ($> 15^\circ$).
     - **Class 255 (Nodata)**: $310,891,048$ px ($45.05\%$).
   - *Scientific Interpretation*: Exploratory multi-source synthesis layer isolating probable inundation candidates from permanent water and mountain shadow false alarms.

3. **Copernicus Digital Elevation Model (`nepal_copernicus_dem_30m.tif`)**:
   - *Dimensions*: $14,400 \times 10,800$ pixels in `EPSG:4326`.
   - *Spatial Resolution*: $0.00027778^\circ$ ($1\text{ arc-second} \approx 30.87\text{ m}$).
   - *Scientific Role*: Provides topographic baseline for Horn's slope derivation and terrain elevation disambiguation.

> [!CRITICAL]
> **Baseline predictions and agreement rasters are candidate hypothesis layers, NOT reference ground truth.** They cannot be used to validate themselves.

---

## 2. Type A: Independent Satellite Imagery Audit (Sentinel-2 L2A)

To evaluate SARA's predictions against independent optical observations, a complete catalog query was executed against the European Space Agency (ESA) Copernicus Data Space Ecosystem (CDSE) OData API for Sentinel-2 MSI Level-2A (`S2MSI2A`) products covering Sentinel-2 Tile **`45RUM`** (Central Nepal / Rasuwa / Bhote Koshi–Trishuli Corridor).

### 2.1 Verified Sentinel-2 Candidate Acquisitions

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             VERIFIED SENTINEL-2 L2A ACQUISITIONS                                │
├──────────────────────────────────────┬─────────────┬───────────┬──────────────┬─────────────────┤
│ Scene Identifier (CDSE)              │ Date (UTC)  │ Cloud (%) │ Valley Vis.  │ Usability       │
├──────────────────────────────────────┼─────────────┼───────────┼──────────────┼─────────────────┤
│ S2C_MSIL2A_20260812T045701_N0512_... │ 2026-08-12  │ 18.75%    │ High (Clear) │ Optimal Pre-Base│
│ S2B_MSIL2A_20260814T044659_N0512_... │ 2026-08-14  │ 62.63%    │ Moderate     │ Pre-Event Gap   │
│ S2B_MSIL2A_20260824T044659_N0512_... │ 2026-08-24  │ 38.57%    │ Moderate     │ Pre-Peak Event  │
│ S2B_MSIL2A_20260827T045659_N0512_... │ 2026-08-27  │ 78.47%    │ Localized    │ Post-Peak Event │
│ S2C_MSIL2A_20260829T044701_N0512_... │ 2026-08-29  │ 82.62%    │ Localized    │ Post-SAR Event  │
└──────────────────────────────────────┴─────────────┴───────────┴──────────────┴─────────────────┘
```

* **Data Provider**: European Space Agency (ESA) / Copernicus Data Space Ecosystem (CDSE).
* **Direct Access URL**: `https://catalogue.dataspace.copernicus.eu/odata/v1/Products`
* **Licensing & Attribution**: Free, open, and full access under the Copernicus Sentinel Data Policy (CC-BY 4.0). Mandatory citation: *"Copernicus Sentinel data [2026]"*.
* **Bands & Resolution**:
  - Blue (B2, $490\text{ nm}$), Green (B3, $560\text{ nm}$), Red (B4, $665\text{ nm}$), NIR (B8, $842\text{ nm}$): $10\text{ m}$ GSD.
  - SWIR-1 (B11, $1610\text{ nm}$), SWIR-2 (B12, $2190\text{ nm}$): $20\text{ m}$ GSD.
  - Scene Classification Layer (SCL): $20\text{ m}$ categorical quality map (identifying cloud, cloud shadow, water, vegetation, bare soil, snow/ice).

### 2.2 Independent Optical Water Mapping Suitability
The Sentinel-2 imagery can support rigorous independent water extraction via the **Modified Normalized Difference Water Index (MNDWI)**:
$$\text{MNDWI} = \frac{\text{Band 3 (Green)} - \text{Band 11 (SWIR)}}{\text{Band 3 (Green)} + \text{Band 11 (SWIR)}}$$
- **Independent Delineation Rule**: A pixel is classified as *Independent Reference Water* if $\text{MNDWI} > 0.0$ and $\text{SCL} \notin \{\text{Cloud (High Prob)}, \text{Cloud (Med Prob)}, \text{Cloud Shadow}\}$.
- **Visual Cross-Check**: False color shortwave infrared composite (Bands 12-8A-4) where standing water absorbs SWIR and appears deep navy/black, sharply contrasting with bright green vegetation.

### 2.3 Environmental & Physical Limitations of Optical Imagery
1. **Persistent Monsoon Cloud Obscuration**: Cloud cover exceeds $78\% - 83\%$ during the peak event window (26–29 August 2026). Wall-to-wall full-scene optical validation is physically impossible.
2. **Topographic Gorge Shadows**: Deep Himalayan canyons cast cast-shadows on steep north-facing slopes during morning optical passes (04:57 UTC / 10:42 AM local time), reducing NIR/SWIR reflectance and mimicking water.
3. **Glacial Snow and Ice**: High-altitude snowfields ($> 4,500\text{ m}$) exhibit high green reflectance that can confound thresholding unless combined with elevation masks.
4. **Turbidity & Suspended Sediment**: Heavy monsoon runoff carries dense silt loads, elevating Red (B4) and Green (B3) reflectance and lowering MNDWI contrast.
5. **Optical vs. SAR Sensor Differences**:
   - SAR responds to surface roughness and dielectric constant (calm water = specular black).
   - Optical responds to molecular absorption and spectral reflectance.
   - Emergent flooded vegetation shows double-bounce bright SAR return but may show submerged water signatures in optical SWIR.

---

## 3. Type B: Independent Flood Maps & Reference Records

Four authoritative, non-satellite external datasets and monitoring systems were investigated:

### 3.1 Nepal Department of Hydrology and Meteorology (DHM) & BIPAD Portal
* **Official Portals**:
  - `https://hydrology.gov.np` (Real-Time River Watch / Flood Early Warning)
  - `https://dhm.gov.np` (Hydrometric & Weather Station Network)
  - `https://bipadportal.gov.np` (National Disaster Risk Reduction & Management Portal)
* **Verified Available Data**:
  - In-situ river level gauge observations ($m$) and discharge time series ($m^3/s$) for hydrometric stations on the Trishuli River (Station 447 at Betrawati; Station 445 at Dhunche) and Bhote Koshi / Sun Koshi basins.
  - 24-hour cumulative rainfall records from automated weather stations in Rasuwa, Nuwakot, and Sindhupalchok.
* **Access Requirements**: Real-time river hydrographs are publicly viewable; high-frequency archived time series require formal academic data request through DHM.
* **Role in Validation**: High-confidence physical temporal ground truth confirming that the river hydrograph crested on **26 August 2026**, verifying that the Sentinel-1 post-event acquisition (28 August 2026) captured recession-phase inundation.

### 3.2 Copernicus Global Flood Monitoring (GFM)
* **Official Portals**:
  - GloFAS Map Viewer: `https://global-flood.emergency.copernicus.eu/`
  - openEO GFM Collection: `https://openeo.cloud/`
  - Documentation: `https://extwiki.eodc.eu/`
* **Characteristics**:
  - Near-real-time global automated flood mapping component of Copernicus Emergency Management Service (CEMS).
  - Ingests all Sentinel-1 GRD imagery and runs an ensemble of three independent algorithms (LIST, DLR, TU Wien) using majority voting ($> 0.50$ consensus) to generate Observed Flood Extent ($20\text{ m}$ GSD).
* **Access & Coverage**: Global coverage, free and open access via openEO and GloFAS viewer.
* **Role in Validation**: Useful as a benchmark for comparison with other automated SAR pipelines, but GFM itself is an algorithm ensemble (not field ground truth) and is subject to terrain shadow exclusion masks in the Himalayas.

### 3.3 EC JRC Global Surface Water (GSW) Dataset
* **Citation**: Pekel, J.-F. et al. (2016). *High-resolution mapping of global surface water and its long-term changes.* Nature, 540, 418–422.
* **Direct Access URL**: `https://global-surface-water.appspot.com/download` | Tile: `80E_30N` (30m spatial resolution).
* **Verified Available Layers**:
  - *Water Occurrence (1984–2021)*: Long-term percentage of valid observations where surface water was present.
  - *Water Seasonality*: Intra-annual permanent vs. seasonal water presence.
  - *Maximum Water Extent*: Binary map of all historical water detections.
* **Attribution Requirement**: *"Source: EC JRC/Google"*.
* **Role in Validation**: Invaluable for independently establishing **permanent water baselines (Class 1)**, but **cannot** serve as dynamic ground truth for August 2026 event-specific inundation.

### 3.4 Dartmouth Flood Observatory (DFO) & UNOSAT Global Archives
* **Portals**: `https://floodobservatory.colorado.edu/` | `https://unosat.org/`
* **Characteristics**: Regional flood archive records and MODIS/VIIRS regional inundation vectors.
* **Role in Validation**: Confirms macro-scale regional flooding across central Nepal, but spatial resolution ($250\text{ m} - 500\text{ m}$) is too coarse to validate narrow Himalayan river gorge flooding ($< 50\text{ m}$ width).

---

## 4. Comprehensive Data Availability Table

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                             DATA AVAILABILITY & AUDIT TABLE                                                 │
├───────────────────────────────┬─────────────────────┬──────────────┬──────────────┬───────────────────┬─────────────────────┤
│ Dataset / Source              │ Verification Status │ Resolution   │ Coverage     │ Relevant Dates    │ Access / Portal     │
├───────────────────────────────┼─────────────────────┼──────────────┼──────────────┼───────────────────┼─────────────────────┤
│ Sentinel-2 L2A (CDSE)         │ VERIFIED AVAILABLE  │ 10m / 20m    │ Tile 45RUM   │ 2026-08-12 (18.7%)│ CDSE OData API      │
│   Optical Multi-Spectral      │                     │              │ (Nepal AOI)  │ 2026-08-27 (78.5%)│ dataspace.copernicus│
├───────────────────────────────┼─────────────────────┼──────────────┼──────────────┼───────────────────┼─────────────────────┤
│ Nepal DHM Hydrometric Gauges  │ PARTIALLY VERIFIED  │ Point        │ Trishuli /   │ 2026-08-24 to     │ hydrology.gov.np /  │
│   River Level & Rain Gauges   │                     │ Stations     │ Bhote Koshi  │ 2026-08-30        │ dhm.gov.np          │
├───────────────────────────────┼─────────────────────┼──────────────┼──────────────┼───────────────────┼─────────────────────┤
│ Copernicus Global Flood Mon.  │ VERIFIED AVAILABLE  │ 20m          │ Global /     │ Continuous        │ global-flood.       │
│   (GFM Ensemble SAR Flood)    │                     │              │ Nepal AOI    │ Sentinel-1 passes │ emergency.copernicus│
├───────────────────────────────┼─────────────────────┼──────────────┼──────────────┼───────────────────┼─────────────────────┤
│ EC JRC Global Surface Water   │ VERIFIED AVAILABLE  │ 30m          │ Tile 80E_30N │ 1984–2021         │ global-surface-     │
│   (GSW Occurrence/Seasonality)│                     │              │ (Nepal AOI)  │ Long-term Base    │ water.appspot.com   │
├───────────────────────────────┼─────────────────────┼──────────────┼──────────────┼───────────────────┼─────────────────────┤
│ Dartmouth Flood Obs. (DFO)    │ PARTIALLY VERIFIED  │ 250m         │ Regional     │ Event Record      │ floodobservatory.   │
│   Macro Flood Extents         │                     │ Macro-scale  │ Nepal        │ August 2026       │ colorado.edu        │
├───────────────────────────────┼─────────────────────┼──────────────┼──────────────┼───────────────────┼─────────────────────┤
│ Copernicus EMS Activation     │ STRICTLY QUARANTINED│ N/A          │ N/A          │ N/A               │ ISOLATED            │
│   EMSR927 Rapid Mapping       │ (DO NOT USE)        │              │              │                   │ (Zero Data Leakage) │
└───────────────────────────────┴─────────────────────┴──────────────┴──────────────┴───────────────────┴─────────────────────┘
```

---

## 5. Recommended Validation Dataset Design

Given the $78\%$ monsoon cloud cover that precludes full-scene wall-to-wall optical digitization, SARA adopts a **stratified multi-chip validation design**:

### 5.1 Stratified Validation Chips
* **Number of Chips**: 10 geographically distributed validation chips of size $512 \times 512$ pixels ($10\text{ m}$ GSD $\approx 5.12\text{ km} \times 5.12\text{ km}$).
* **Stratification Strata**:
  1. *Stratum A (Alluvial Floodplain — Lower Trishuli River, Nuwakot/Dhading)*: 3 chips covering flat riverbanks and braided gravel bars.
  2. *Stratum B (Deep Mountain Gorge — Upper Bhote Koshi / Rasuwa)*: 3 chips covering high-relief canyons prone to radar shadow.
  3. *Stratum C (Terraced Agriculture / Tributary Inundation)*: 2 chips covering valley-floor agricultural plots.
  4. *Stratum D (High-Slope Forest / Mountain Ridge)*: 2 chips on slopes $> 20^\circ$ to strictly quantify false alarm rejection of Class 4.
* **Source Images for Chip Delineation**:
  - Pre-Event Baseline: `S2C_MSIL2A_20260812T045701...` (18.75% cloud).
  - Post-Event Flood: `S2B_MSIL2A_20260827T045659...` (78.47% cloud, using localized cloud gaps over target river reaches).
  - Permanent Water Reference: JRC Global Surface Water Occurrence tile `80E_30N` (Water Occurrence $> 80\%$).

---

## 6. Validation Alignment & Masking Workflow

To guarantee spatial fidelity and prevent resampling artifacts, reference labels must be aligned strictly to the SARA prediction raster grid:

```mermaid
flowchart TD
    S2[Sentinel-2 L2A B3, B4, B8, B11 & SCL] --> Crop[Extract 10 Stratified 512x512 Validation Chips]
    Crop --> MNDWI[Compute MNDWI = B3-B11 / B3+B11]
    MNDWI --> SCL_Mask[Mask Clouds & Shadows using SCL Layer]
    SCL_Mask --> RefLabels["Generate 3-Class Reference Mask<br/>1 = Water | 0 = Non-Water | 255 = Nodata/Uncertain"]
    
    RefLabels --> Reproject["Warp/Align to SARA Prediction Grid<br/>EPSG:4326 | Res=8.983e-5 deg | Nearest Neighbor"]
    
    SARAPred["SARA Prediction Raster<br/>(Baseline Mask or 5-Class Agreement)"] --> Overlap{"Spatial Overlap Mask<br/>Ref != 255 AND SARA != 255?"}
    Reproject --> Overlap
    
    Overlap -->|Yes (Valid Overlap Pixels)| Eval[Compute Confusion Matrix & Metrics]
    Overlap -->|No (Nodata or Cloud)| Discard[Strictly Exclude from Metrics]
```

### 6.1 Reference Raster Encoding Standard
Reference validation chips will be encoded into single-band GeoTIFFs:
- **Value 1 (Confirmed Water)**: $\text{MNDWI} > 0.0$ on cloud-free pixels, cross-verified against false-color SWIR imagery.
- **Value 0 (Confirmed Non-Water / Dry Land)**: $\text{MNDWI} \le -0.10$ on cloud-free dry terrain.
- **Value 255 (Nodata / Cloud / Cloud Shadow / Ambiguous)**: High/medium probability cloud, cloud shadow, deep mountain optical shadow, or unresolved mixed pixels.

### 6.2 Strict Overlap Evaluation Rule
Metrics are computed **strictly and exclusively** over pixels where both:
$$\text{Reference}(x, y) \in \{0, 1\} \quad \text{AND} \quad \text{SARA\_Prediction}(x, y) \in \{0, 1\}$$
Any pixel where either layer contains `255 (Nodata/Uncertain)` is masked out and completely excluded from the denominator.

---

## 7. Quantitative Evaluation Metrics Protocol

Once the 10 independent reference chips are delineated and aligned, SARA will calculate the following standard statistical metrics, treating **Water** as the positive class ($1$) and **Non-Water** as the negative class ($0$):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CONFUSION MATRIX                                 │
├──────────────────────────┬────────────────────────┬─────────────────────────┤
│                          │ Reference: Water (1)   │ Reference: Non-Water (0)│
├──────────────────────────┼────────────────────────┼─────────────────────────┤
│ SARA Prediction: Water(1)│ True Positive (TP)     │ False Positive (FP)     │
├──────────────────────────┼────────────────────────┼─────────────────────────┤
│ SARA Prediction: Non-W(0)│ False Negative (FN)    │ True Negative (TN)      │
└──────────────────────────┴────────────────────────┴─────────────────────────┘
```

### 7.1 Mathematical Formulations

$$\text{Precision} = \frac{TP}{TP + FP} \quad \text{(Exactness of water prediction)}$$

$$\text{Recall} = \frac{TP}{TP + FN} \quad \text{(Completeness / Detection rate)}$$

$$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \quad \text{(Harmonic mean)}$$

$$\text{Intersection over Union (IoU)} = \frac{TP}{TP + FP + FN} \quad \text{(Jaccard Index)}$$

$$\text{Overall Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

### 7.2 Class Imbalance Handling
In mountainous terrain, water typically covers $< 2\%$ of the total landscape. Overall Accuracy is heavily distorted by the massive true negative land class ($TN$). SARA strictly prioritizes **Precision, Recall, F1-Score, and IoU** over overall accuracy to evaluate genuine flood detection performance.

### 7.3 Dual Pipeline Benchmarking
The evaluation will be performed separately for:
1. **Pipeline 1: Baseline Small U-Net Mask** (`predicted_water_mask.tif` at threshold $\tau = 0.50$).
2. **Pipeline 2: SARA 5-Class Synthesized Agreement Layer** (`Class 3: Agreed Inundation Candidate`).
3. **Shadow False-Alarm Isolation Metric**:
   $$\text{Shadow Rejection Rate} = \frac{\text{Pixels in Class 4 overlapping Reference Non-Water}}{\text{Total Pixels in Class 4}} \times 100\%$$

---

## 8. Scientific Limitations & Uncertainty Disclosures

1. **Spatial Discontinuity**: Chip-based validation evaluates representative sample reaches rather than wall-to-wall continuous flood extent across the entire Himalayan basin.
2. **Optical-SAR Acquisition Time Offset**: Sentinel-2 (post-event 2026-08-27 04:57 UTC) was acquired approximately 31 hours prior to Sentinel-1 (post-event 2026-08-28 12:21 UTC). Rapid flood recession in steep Himalayan rivers may cause minor discrepancies in river stage between optical and SAR acquisitions.
3. **Emergent Vegetation Ambiguity**: Dense tree canopy bordering flooded rivers obscures water from optical top-of-atmosphere sensors while producing mixed double-bounce/specular signatures in SAR.
4. **Strict Isolation of EMSR927**: Copernicus EMS activation EMSR927 remains quarantined; zero data from EMSR927 will be used for validation.

---

## 9. Summary & Exact Next Steps

### Current Status Summary
* **Available & Verified**:
  - Sentinel-2 L2A Optical Scenes from CDSE (2026-08-12 pre-event baseline and 2026-08-27 post-event flood window).
  - JRC Global Surface Water 30m Occurrence dataset (Tile `80E_30N`).
  - Nepal DHM river gauge peak flood timing (26 August 2026).
  - Copernicus GFM operational Sentinel-1 ensemble product.
* **Currently Missing**: Delineated and digitized 2D raster validation chips for the August 2026 Nepal event.

### Exact Next Steps
1. **Acquire Optical Scenes**: Download Sentinel-2 products `S2C_MSIL2A_20260812...` ($\approx 1.1\text{ GB}$) and `S2B_MSIL2A_20260827...` ($\approx 1.1\text{ GB}$) from CDSE.
2. **Acquire GSW Baseline**: Download JRC Global Surface Water Occurrence tile `80E_30N` ($\approx 45\text{ MB}$).
3. **Extract 10 Stratified Chips**: Crop 10 geographically distributed $512 \times 512$ pixel chips across the Trishuli and Bhote Koshi river corridors.
4. **Generate Reference Masks**: Compute MNDWI and SCL cloud masking to create independent binary reference labels (`0` = Land, `1` = Water, `255` = Nodata/Cloud).
5. **Run Evaluation Benchmark**: Execute the strict overlap evaluation script to compute the Confusion Matrix, Precision, Recall, F1, and IoU.
