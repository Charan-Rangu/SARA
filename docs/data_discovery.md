# SARA — Satellite Data Discovery & Inventory Module

This module provides automated discovery, spatial validation, geometric compatibility analysis, and storage budget planning for **Sentinel-1 Synthetic Aperture Radar (SAR)** satellite imagery.

---

## 📡 Catalogue Access Engine

SARA interfaces with the **Copernicus Data Space Ecosystem (CDSE)** STAC API endpoint:
- **API Endpoint:** `https://stac.dataspace.copernicus.eu/v1/search`
- **Authentication:** Metadata search is open, free, and anonymous.
- **Collection:** `SENTINEL-1` (Ground Range Detected / GRD products in Interferometric Wide / IW mode).

---

## 🗺️ Area of Interest (AOI) Specification & Bounds

Spatial queries require an Area of Interest represented in WGS84 (`EPSG:4326`) coordinates.

### AOI Validation Criteria:
1. **Bounding Box:** `(min_lon, min_lat, max_lon, max_lat)` coordinates must satisfy:
   - $-180.0 \le \text{longitude} \le 180.0$
   - $-90.0 \le \text{latitude} \le 90.0$
   - $\text{min\_lon} < \text{max\_lon}$ and $\text{min\_lat} < \text{max\_lat}$
2. **GeoJSON Polygon:** Geometry must be OGC-valid (`geom.is_valid`) and non-empty.
3. **Area Limit:** Bounding box area must not exceed $25.0 \text{ deg}^2$ (approx. $5^\circ \times 5^\circ$) to prevent unmanageable satellite scene queries.

---

## 🛰️ Sentinel-1 Geometric Compatibility Matching

SAR flood detection relies on comparing pre-event radar backscatter intensity with post-event backscatter. Arbitrary scene comparisons produce severe geometric and radiometric errors.

### Compatibility Criteria:
1. **Orbit Direction (Ascending vs Descending):**
   - **Requirement:** Must match (`ASCENDING` with `ASCENDING` or `DESCENDING` with `DESCENDING`).
   - **Rationale:** SAR is a side-looking active radar sensor. Ascending (south-to-north) and Descending (north-to-south) passes view ground topography from opposite directions, creating opposite terrain layover and shadow. Comparing opposite orbit directions introduces false flood detection artifacts.
2. **Relative Orbit Number:**
   - **Requirement:** Matching relative orbit preferred (score bonus +30 points).
   - **Rationale:** Scenes from the same relative orbit repeat the exact same flight path and look angle every 6 to 12 days, maintaining consistent ground geometry.
3. **Sensor Mode & Polarization:**
   - **Requirement:** `IW` (Interferometric Wide) mode required; `VV` or `VH` dual-polarization matching required.
4. **Spatial Overlap:**
   - **Requirement:** Both pre and post scenes must cover at least 50% of the target AOI.

> [!IMPORTANT]
> **Co-Registration Note:** Matching relative orbits and orbit directions provides consistent viewing geometry, but does **NOT** guarantee pixel-level alignment. Terrain correction, DEM-assisted Doppler geocoding, and spatial co-registration in Phase 3 are required.

---

## 💾 Storage Budget Estimation & Download Planning

Before acquiring satellite products, SARA computes an estimated download budget:
- Evaluates compressed SAFE product download sizes (~1.5 GB to 2.5 GB per scene).
- Checks total download size against user-configured storage budget (`storage_budget_gb`).
- Emits warnings if storage budget is exceeded or if size metadata is missing.
- **No satellite files are downloaded during the discovery phase.**

---

## 🧪 Testing Guidelines

### 1. Run Offline Unit Tests (No Internet Required)
```powershell
sara\.venv\Scripts\python.exe -m pytest tests/ -v
```

### 2. Run Optional Live STAC API Smoke Test (Requires Internet)
```powershell
sara\.venv\Scripts\python.exe tests/smoke_test_live_catalog.py
```
