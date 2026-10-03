# SARA — OpenStreetMap Infrastructure Overlay & Road Accessibility Engineering Guide

This document details the engineering specifications, vector validation data contracts, raster-vector exposure overlays, NetworkX graph construction, and accessibility analysis modes for **SARA (Satellite-Assisted Rescue & Assessment)**.

---

## 🗺️ 1. OpenStreetMap Vector Infrastructure Acquisition

SARA supports two methods for obtaining pre-event vector infrastructure:

1. **Local Pre-Event GeoJSON Snapshot (Recommended for Disaster Operations):**
   - Analyst supplies local GeoJSON files (`roads.geojson`, `buildings.geojson`, `facilities.geojson`, `settlements.geojson`).
   - Ensures strict temporal compliance by using verified pre-disaster snapshots.
2. **Live Overpass API Queries:**
   - Endpoint: `https://overpass-api.de/api/interpreter`
   - Queries `highway=*`, `building=*`, `amenity=hospital|clinic`, and `place=town|village|hamlet`.
   - **Important:** Live Overpass queries return current OpenStreetMap data and are explicitly tagged as `LIVE_OVERPASS_API` with a disclaimer. Live responses must **NOT** be represented as historical pre-event snapshots.

---

## 🛠️ 2. Vector Data Validation Contract (`geospatial/infrastructure.py`)

All vector layers pass through a 6-stage cleaning pipeline:
1. **CRS Reprojection:** Standardizes layers to `EPSG:4326` (or `EPSG:3857` for metric distance calculations).
2. **Empty Geometry Filtering:** Removes empty or missing geometry records.
3. **Invalid Geometry Repair:** Applies `shapely.make_valid()` to repair self-intersecting polygons or corrupted lines.
4. **Geometry Type Enforcement:** Filters out non-matching geometry types (e.g. keeping `LineString`/`MultiLineString` for roads).
5. **Spatial AOI Filtering:** Retains features intersecting the target AOI bounding box.
6. **Deduplication:** Drops duplicate records based on unique feature identifiers (e.g. `osm_id`).

---

## 🌊 3. Flood-Change Exposure Overlay Analysis (`geospatial/infrastructure_exposure.py`)

Intersects validated infrastructure layers with preliminary SAR change masks (`preliminary_change_mask.tif`).

### Evidence Labels:
To avoid alarmist misclassification, SARA uses cautious, objective evidence labels:
- `potential_change_overlap`: Feature intersects pixels flagged with preliminary SAR backscatter reduction.
- `near_preliminary_change`: Feature is within proximity distance ($30\text{ m}$) of a change pixel.
- `no_detected_overlap`: No backscatter change detected in feature footprint.
- `not_assessed`: Change mask raster not provided or outside spatial extent.

> [!CAUTION]
> **Strict Labeling Rule:** Features must **NEVER** be labeled `destroyed`, `flooded`, or `impassable` based solely on SAR backscatter change overlap.

---

## 🛣️ 4. Road Network Graph Topology (`geospatial/road_network.py`)

Converts vector road geometries into a weighted NetworkX graph (`nx.Graph`):
- **Metric Projection:** Reprojects roads to `EPSG:3857` so edge weights represent real ground distances in **meters**.
- **Node Snapping Tolerance:** Snaps line endpoints within a $2.0\text{ m}$ tolerance to connect adjacent road segments without topological distortion.
- **Topology Metadata:** Tracks nodes, edges, connected components, and isolated subgraphs.

---

## 🏥 5. Accessibility Analysis Modes (`geospatial/accessibility.py`)

Evaluates network reachability from populated settlements to nearest healthcare facilities/hospitals using Dijkstra's shortest-path algorithm (`nx.dijkstra_path_length`).

### Mode A: Baseline Connectivity
Calculates shortest-path network distances along the entire mapped pre-event road network without applying satellite closures.

### Mode B: Scenario Connectivity (Hypothetical Closures)
Analyst specifies candidate closed road edges (e.g. roads marked `potential_change_overlap`). The engine temporarily excludes selected edges from the graph and re-evaluates settlement reachability.
- If no alternative route exists in the mapped network, the settlement is tagged `HYPOTHETICALLY_ISOLATED`.
- **Note:** An unmapped ground route or unmapped rural dirt track may still exist. Scenario isolation indicates *network disruption along mapped infrastructure*, not absolute physical isolation.

---

## 📁 6. Output Files & Run Organization

Run-specific outputs are saved under `data/output/run_<run_id>/`:
- `potentially_affected_roads.geojson`
- `potentially_exposed_buildings.geojson`
- `potentially_exposed_facilities.geojson`
- `settlements_exposure.geojson`
- `exposure_summary.json`
- `processing_metadata.json`

---

## 🧪 7. Reproducing Unit Tests

Run all 50+ automated offline tests in PowerShell:
```powershell
sara\.venv\Scripts\python.exe -m pytest tests/ -v
```
