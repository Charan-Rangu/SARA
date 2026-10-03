# SARA — Competition Evidence & Implementation Audit Report
**Satellite-Assisted Rescue & Assessment Platform**
*Audit Date: October 3, 2026 | Comprehensive Evidence Verification*

---

## 1. Executive Summary & Audit Protocol

This audit provides a factual, evidence-grounded review of every major functional, scientific, and performance claim made regarding the SARA (Satellite-Assisted Rescue & Assessment) platform. 

Every claim is evaluated against five audit criteria:
1. **Source Code & API Contract Evidence**: Verifiable file paths, class names, and endpoint signatures.
2. **Data Provenance**: Distinction between real-world satellite/vector datasets, local test fixtures, and synthetic graphs.
3. **Empirical Execution Logs**: Direct command outputs and hardware telemetry.
4. **Physical & Methodological Integrity**: Strict adherence to scientific uncertainty boundaries (e.g. distinguishing candidate radar backscatter change from ground-truth verified flooding).
5. **Reproducibility & Safety**: Absence of credential leaks, proper data isolation, and deterministic test suite execution.

---

## 2. Claim-by-Claim Evidence Matrix

| # | Major Claim | Evidence File / Command | Test Method | Actual Result | Data Provenance | Limitations & Boundaries | Verification Status |
| :- | :--- | :--- | :--- | :--- | :---: | :--- | :---: |
| **1** | **Interactive 8-View Web Dashboard** | [`backend/main.py`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/backend/main.py#L640-L840), `http://localhost:8000/dashboard` | DOM inspection & TestClient API validation | 8 views switch dynamically; 4 top KPI cards reflect active Nepal run metrics; layer toggles and opacity sliders work. | Real Web DOM & REST API | Requires active FastAPI backend server and internet connection for OSM Carto tiles. | **VERIFIED** |
| **2** | **New-AOI Catalogue Discovery** | [`backend/services/sentinel1_catalog.py`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/backend/services/sentinel1_catalog.py), `scratch/test_new_aoi_workflow.py` | Live query against CDSE OData API for Valencia, Spain | Found 9 candidate Sentinel-1 scenes (6 pre, 3 post) with exact dates, relative orbits (30, 8, 110), and ~1.68 GB sizes. | Real CDSE Satellite Catalogue | Multi-GB scene downloads require explicit user confirmation and CDSE credentials. | **VERIFIED** |
| **3** | **New-AOI 1.92% Water Provenance** | `data/output/run_new_aoi_valencia_1791043956/run_metrics.json` | JSON metadata and execution trace audit | 1.92% water ($315/16,384\text{ px}$) was generated from local test fixture `valid_custom_sar_2band_prepared_3band.tif` ($128 \times 128$), NOT newly downloaded Valencia imagery. | **Existing Local SAR Fixture (B)** | Download safeguards halted unconfirmed 3.3GB download; inference executed on fallback fixture. | **VERIFIED (With Provenance Clarification)** |
| **4** | **ESA SNAP 14.0.0 Availability** | `C:\Program Files\esa-snap\bin\gpt.exe`, [`geospatial/sar_preprocessing.py`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/geospatial/sar_preprocessing.py) | Subprocess execution diagnostic (`gpt -h`) | Host GPT binary detected; exited code 0 in 2.23s; all 8 operators (`Read` to `Terrain-Correction`) available. | Real Host Environment | Execution of full-scene GPT graphs takes 20–40 min per 1.7GB SAFE archive. | **VERIFIED** |
| **5** | **SNAP XML Graph Generation** | [`geospatial/snap_graph.py`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/geospatial/snap_graph.py), `tests/test_phase6_snap_and_download.py` | Unit test & string XML hierarchy check | Generated 4,128-char validated XML graph with Range-Doppler Terrain Correction to `EPSG:4326` using Copernicus 30m DEM. | Deterministic XML Builder | XML generation validates graph syntax; does not execute full raster processing. | **VERIFIED** |
| **6** | **U-Net Sub-Scene Performance** | `scratch/detailed_benchmark_audit.py` | Component-level profiling on $2048 \times 2048$ SAR slice (4.19M px) | Total runtime: $2.40\text{ s}$ (Validation: 27ms, Read: 214ms, PyTorch GPU: 812ms, Write: 248ms, PNG: 1049ms). | Real Sentinel-1 Calibrated SAR | Measured on 64 tiles ($256 \times 256$) with batch size 16 on NVIDIA RTX 2050 GPU. | **VERIFIED** |
| **7** | **U-Net Full-Scene Timing Reality** | `experiments/unet/nepal_multitemporal_change_metrics.json`, `scratch/audit_full_scene_benchmark.py` | Full-scene ($32,072 \times 21,519$, 690M px, 10,710 tiles) trace audit | Offline full-scene execution required **22.72 seconds** (not 2 seconds). Sub-2s claims apply only to crops $\le 2048 \times 2048$. | Real Nepal Sentinel-1 Scene (6.45 GB) | Full 6.45 GB GeoTIFF disk I/O requires ~15–20s; ~5,910 valid tiles require ~370 GPU batches (~6s). | **VERIFIED (With Timing Boundary Clarification)** |
| **8** | **Infrastructure Exposure Analysis** | [`geospatial/infrastructure_exposure.py`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/geospatial/infrastructure_exposure.py) | Raster-vector spatial intersection testing | Correctly identifies intersecting road lengths and building counts; outputs labeled `potential_change_overlap`. | Real Vector Formats & Spatial Logic | Uses cautious terminology; does not classify assets as confirmed destroyed. | **VERIFIED** |
| **9** | **Settlement Accessibility Routing** | [`geospatial/accessibility.py`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/geospatial/accessibility.py), [`geospatial/road_network.py`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/geospatial/road_network.py) | NetworkX Dijkstra shortest-path reachability | Baseline: 3/3 settlements connected ($17.5\text{ km}, 11.2\text{ km}, 3.2\text{ km}$). Scenario (HWY_02 closed): 2 settlements isolated. | Metric `EPSG:3857` Graph Routing | Scratch and unit tests use deterministic synthetic fixtures to avoid Overpass rate limits. | **VERIFIED (Fixture-Labeled)** |
| **10** | **Evidence-Grounded Situation Reports** | [`backend/services/situation_report.py`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/backend/services/situation_report.py), `scratch/verify_populated_situation_report.py` | Verbatim text and metric cross-check | Generated markdown report reflects exact JSON metrics; unexecuted modules explicitly marked `NOT_AVAILABLE / NOT_RUN`. | Real Computed Analysis Artifacts | No LLM hallucination; all numbers match active run metadata verbatim. | **VERIFIED** |
| **11** | **Independent Validation Status** | [`backend/main.py`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/backend/main.py#L1509), [`docs/methodology_and_limitations.md`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/docs/methodology_and_limitations.md) | Codebase string & UI inspection | Displays `"Pending"` status; EMSR927 quarantined; no ungrounded F1/IoU metrics reported for Nepal. | Real Policy Enforcement | Independent reference labels not publicly available for full Nepal AOI. | **VERIFIED** |
| **12** | **Submission Safety & Secrets Exclusion** | [`.gitignore`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/.gitignore), [`.env.example`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/.env.example), [`.env`](file:///c:/Users/chara/OneDrive/Documents/Multimodal%20AI/sara/.env) | Filesystem inspection & regex check | `.env` ignored; `.env.example` contains only sanitized placeholders; large raw `*.SAFE` and `*.tif` ignored. | Git Config & Local Filesystem | Submission archive must omit local `.env` and `data/raw/*.SAFE.zip`. | **VERIFIED** |
| **13** | **Automated Pytest Suite** | `tests/`, `pytest tests/ -v` | Full test runner execution | **166 passed, 0 failed, 3 warnings in 73.22s.** | All 43 test modules in repository | Deprecation warning from starlette testclient httpx import. | **VERIFIED** |

---

## 3. Deep-Dive Findings on Critical Questions

### 3.1 Task 1: The Valencia New-AOI Workflow & 1.92% Water Provenance
* **Catalogue Search & Discovery**: Real-world CDSE OData API integration succeeded. For the Valencia bbox `[-0.55, 39.30, -0.25, 39.60]` across the October 2024 flood window, 9 candidate Sentinel-1 acquisitions were discovered with exact scene titles, acquisition timestamps, orbit directions, and relative orbits (30, 8, 110).
* **Download Safeguards**: The download planner estimated the raw payload at $3.28\text{ GB}$ (2 scenes). SARA's automated safety gate prevented unconfirmed multi-gigabyte downloading.
* **Inference Provenance**: Because raw Valencia SAFE archives were not downloaded to disk, the U-Net inference step in `scratch/test_new_aoi_workflow.py` executed on the existing local test fixture `data/processed/unet_inputs/valid_custom_sar_2band_prepared_3band.tif` ($128 \times 128$).
* **Conclusion**: The reported **1.92% candidate-water extent** originates from **Category B (Previously Existing Local Test Fixture)**, while the **satellite search and download planning** were executed against **live real-world CDSE catalogue data**.

### 3.2 Task 2 & 3: SNAP Execution & Full-Scene Timing Reality
* **SNAP Execution**: ESA SNAP 14.0.0 is verified installed on the host at `C:\Program Files\esa-snap\bin\gpt.exe`. Pre-event and post-event calibrated GeoTIFFs (`pre_calibrated_sigma0.tif` and `post_calibrated_sigma0.tif`, $32,072 \times 21,519$ pixels) are verified present in `data/processed/`.
* **U-Net Timing Reality**:
  - For small crops ($128 \times 128$ to $2048 \times 2048$, 16K to 4.19M pixels), end-to-end execution takes **$0.32\text{ s}$ to $2.40\text{ s}$**.
  - For the **complete 690-million-pixel scene** ($32,072 \times 21,519$), reading and writing 6.45 GB GeoTIFFs from disk and running ~370 GPU batches of size 16 across 5,910 valid tiles requires **22.72 seconds**.
  - Sub-2-second claims must not be extrapolated to the full 690M pixel scene.

### 3.3 Task 4: Infrastructure & Accessibility Graph Provenance
* **Exposure & Routing**: The mathematical algorithms for spatial overlap (`InfrastructureExposureAnalyzer`) and Dijkstra routing (`AccessibilityAnalyzer`) operate on standard GeoPandas GeoDataFrames and NetworkX graphs.
* **Kerala Example Provenance**: The Aluva, Kalamassery, and Perumbavoor accessibility example in `scratch/verify_infrastructure_and_accessibility.py` was executed on a **deterministic synthetic test fixture** situated at real Kerala coordinates (`[76.15, 9.85, 76.45, 10.15]`) with projected metric distances in `EPSG:3857`.
* **Safeguards**: All accessibility findings are labeled as scenario-based simulations (`"HYPOTHETICALLY_ISOLATED"`), with clear statements that candidate radar change does not prove ground road destruction.

### 3.4 Task 5: Situation Report Grounding
* **Evidence Grounding**: The `SituationReportGenerator` compiles reports strictly from provided dictionaries. When sections are omitted, it renders `NOT_AVAILABLE / NOT_RUN`. When summaries are provided from active runs, it inserts exact numbers (e.g. $379,266,320$ valid pixels, $8.52\%$ water, $14.29\%$ backscatter drop). Zero figures are invented or hallucinated by LLMs.

---

## 4. Remaining Blockers & Pre-Submission Action Checklist

1. **Submission Archive Packaging**:
   - Ensure local `.env` file is excluded from the final ZIP/tarball.
   - Verify `.env.example` is present with placeholder values.
   - Exclude raw multi-gigabyte files (`data/raw/*.SAFE`, `data/raw/*.SAFE.zip`, and temporary `.part` files) to keep submission package within competition size limits (~50–100 MB).
2. **Preserve Required Attributions**:
   - *"Contains modified Copernicus Sentinel data 2026."*
   - *"Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved."*
   - *"© OpenStreetMap contributors."*
3. **Independent Validation Status Transparency**:
   - Maintain the honest disclosure that independent ground-truth flood boundary masks for the full Nepal scene are pending, and that EMSR927 activation data remains strictly quarantined to prevent data leakage.

---

## 5. Final Automated Test Summary

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

```
================= 166 passed, 3 warnings in 73.22s (0:01:13) ==================
```
* **Total Tests**: **166**
* **Passing Tests**: **166 (100.0%)**
* **Failed / Errors**: **0**
