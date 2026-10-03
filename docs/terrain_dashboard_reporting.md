# SARA — Terrain Analysis, Web Dashboard & Situation Reporting Engineering Guide

This document details the engineering specifications, mathematical algorithms, dashboard architecture, and report generation engines for **Phase 5 of SARA (Satellite-Assisted Rescue & Assessment)**.

---

## ⛰️ 1. Copernicus DEM Terrain Analysis & Attribute Provenance

SARA processes **Copernicus WorldDEM-30** 30-meter global elevation rasters.

### Required Legal Attribution Statement:
> *"Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved."*

### Derived Terrain Rasters (`geospatial/terrain_analysis.py`):
1. **Slope ($\theta$ in degrees):**
   $$\text{Slope} = \arctan \left( \sqrt{\left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2} \right) \cdot \frac{180}{\pi}$$
2. **Aspect ($\phi$ in degrees 0–360°):**
   $$\text{Aspect} = \left( 270^\circ - \arctan2\left(\frac{\partial z}{\partial y}, -\frac{\partial z}{\partial x}\right) \cdot \frac{180}{\pi} \right) \pmod{360^\circ}$$

---

## 🌊 2. D8 Downstream Flow-Path Tracing (`geospatial/flow_tracing.py`)

Generates preliminary terrain-based flow trajectories from analyst-specified starting points using the D8 steepest-descent algorithm:
- Evaluates elevation drop ($\Delta z = z_{\text{curr}} - z_{\text{neighbor}}$) across 8 surrounding cells.
- Iteratively steps to the neighbor with maximum downward gradient.
- Enforces maximum step limit (500 steps) and visited pixel sets to prevent infinite loops.
- Outputs GeoJSON LineString features tagged `MODELED_TERRAIN_FLOW_PATH`.

> [!CAUTION]
> **Modeled Flow-Path Disclaimer:**
> Modeled D8 flow paths indicate surface water drainage directions based on elevation topology. They do **NOT** predict downstream water depth, flood timing, or inundation boundaries.

---

## 🖥️ 3. Interactive Web Dashboard (`/dashboard`)

Access the web dashboard in your browser:
- **URL:** `http://127.0.0.1:8000/dashboard`
- **Features:**
  - Interactive Leaflet map viewer with OpenStreetMap basemap.
  - AOI bounding box & event date inputs.
  - Real-time CDSE OData satellite discovery triggering (`POST /satellite/search`).
  - Layer toggle controls (AOI, Preliminary SAR Change Mask, Infrastructure Exposure, Settlement Accessibility, Copernicus DEM & Modeled Flow Paths).
  - Live status indicators (Catalogue, Preprocessing, Exposure, Accessibility, Terrain).

---

## 📄 4. Evidence-Grounded Situation Report Generator (`backend/services/situation_report.py`)

Generates deterministic, structured Markdown situation reports from analysis metrics:
- **Endpoint:** `POST /reports/situation`
- **Sections:** Executive Overview, Satellite Discovery, Preliminary SAR Change, Infrastructure Exposure, Road Accessibility, Copernicus DEM Terrain Analysis, Key Uncertainties, Mandatory Limitation Statement, and Copyright Attributions.

> [!IMPORTANT]
> **Mandatory System Limitation Statement:**
> *"SARA produces preliminary geospatial assessments to support further investigation. Satellite backscatter changes and terrain-derived flow paths are not, by themselves, proof of flooding, infrastructure damage, road closure, or safe accessibility."*

---

## 🧪 5. Reproducing All Automated Offline Tests

Run the complete 60+ offline unit test suite:
```powershell
sara\.venv\Scripts\python.exe -m pytest tests/ -v
```
