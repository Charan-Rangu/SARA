"""SARA — Satellite-Assisted Rescue & Assessment API Service."""

import os
import json
import io
import uuid
from typing import Any, Optional, List, Dict
from fastapi import FastAPI, HTTPException, Query, Response, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np

from geospatial.aoi import AOI
from backend.services.sentinel1_catalog import Sentinel1CatalogService, CatalogSearchResult
from backend.services.scene_compatibility import SceneCompatibilityAnalyzer, CompatibilityEvaluation
from backend.services.download_planner import DownloadPlanner, DownloadPlan
from backend.services.osm_infrastructure import OSMInfrastructureService, OSMSourceMetadata
from geospatial.infrastructure import InfrastructureContainer
from geospatial.infrastructure_exposure import InfrastructureExposureAnalyzer, ExposureSummary
from geospatial.road_network import RoadNetworkBuilder, GraphSummary
from geospatial.accessibility import AccessibilityAnalyzer, AccessibilitySummary
from backend.services.dem_service import DEMService, DEMMetadata
from geospatial.terrain_analysis import TerrainAnalyzer, TerrainAnalysisResult
from geospatial.flow_tracing import TerrainFlowTracer, FlowTracingResult
from backend.services.situation_report import SituationReportGenerator, ReportRequest
from geospatial.sar_preprocessing import SNAPProcessorAdapter, SARPreprocessingConfig
from geospatial.snap_graph import SNAPGraphBuilder, SNAPGraphConfig
from backend.services.download_service import DataDownloadService, DownloadManifest
from backend.services.unet_segmentation_service import UNetSegmentationService, SegmentationRequest, SegmentationResponse
from backend.services.layer_visualizer import LayerVisualizerService

app = FastAPI(
    title="SARA — Satellite-Assisted Rescue & Assessment API",
    version="0.1.0",
    description="Backend API service for satellite flood mapping, discovery, exposure analysis, road accessibility, terrain flow tracing, and dashboard reporting."
)

catalog_service = Sentinel1CatalogService()
compatibility_analyzer = SceneCompatibilityAnalyzer()
download_planner = DownloadPlanner()
osm_service = OSMInfrastructureService()
exposure_analyzer = InfrastructureExposureAnalyzer()
road_builder = RoadNetworkBuilder()
accessibility_analyzer = AccessibilityAnalyzer()
dem_service = DEMService()
terrain_analyzer = TerrainAnalyzer()
flow_tracer = TerrainFlowTracer()
report_generator = SituationReportGenerator()
snap_adapter = SNAPProcessorAdapter()
download_service = DataDownloadService()
unet_segmentation_service = UNetSegmentationService()
visualizer_service = LayerVisualizerService()



class SearchRequest(BaseModel):
    bbox: Optional[list[float]] = Field(default=None, json_schema_extra={"example": [76.15, 9.85, 76.45, 10.15]})
    geojson: Optional[dict[str, Any]] = Field(default=None)
    event_date: str = Field(..., json_schema_extra={"example": "2024-08-15"})
    pre_event_days: int = Field(default=14, ge=1, le=60)
    post_event_days: int = Field(default=14, ge=1, le=60)
    max_results: int = Field(default=20, ge=1, le=100)
    storage_budget_gb: float = Field(default=10.0, gt=0.0)


class SearchResponse(BaseModel):
    status: str
    project: str = "SARA"
    version: str = "0.1.0"
    search_result: CatalogSearchResult
    compatibility_evaluation: CompatibilityEvaluation
    download_plan: DownloadPlan
    warnings: list[str] = Field(default_factory=list)


class ExposureRequest(BaseModel):
    bbox: Optional[list[float]] = Field(default=None, json_schema_extra={"example": [76.15, 9.85, 76.45, 10.15]})
    geojson: Optional[dict[str, Any]] = Field(default=None)
    change_mask_path: Optional[str] = Field(default=None)
    roads_geojson_path: Optional[str] = Field(default=None)
    buildings_geojson_path: Optional[str] = Field(default=None)
    facilities_geojson_path: Optional[str] = Field(default=None)
    settlements_geojson_path: Optional[str] = Field(default=None)


class ExposureResponse(BaseModel):
    status: str
    run_id: str
    output_directory: str
    exposure_summary: ExposureSummary
    osm_source_metadata: OSMSourceMetadata
    warnings: list[str] = Field(default_factory=list)


class AccessibilityRequest(BaseModel):
    bbox: Optional[list[float]] = Field(default=None, json_schema_extra={"example": [76.15, 9.85, 76.45, 10.15]})
    geojson: Optional[dict[str, Any]] = Field(default=None)
    roads_geojson_path: Optional[str] = Field(default=None)
    facilities_geojson_path: Optional[str] = Field(default=None)
    settlements_geojson_path: Optional[str] = Field(default=None)
    change_mask_path: Optional[str] = Field(default=None)
    simulate_flood_closures: bool = Field(default=True)


class AccessibilityResponse(BaseModel):
    status: str
    run_id: str
    output_directory: str
    accessibility_summary: AccessibilitySummary
    osm_source_metadata: OSMSourceMetadata
    warnings: list[str] = Field(default_factory=list)


class TerrainRequest(BaseModel):
    dem_path: str
    bbox: Optional[list[float]] = Field(default=None, json_schema_extra={"example": [76.15, 9.85, 76.45, 10.15]})


class FlowTracingRequest(BaseModel):
    dem_path: str
    start_points_wgs84: list[tuple[float, float]] = Field(..., json_schema_extra={"example": [[76.25, 9.95]]})


@app.get("/health")
def health_check():
    return {"status": "ok", "project": "SARA", "version": "0.1.0"}


@app.get("/infrastructure/health")
def infrastructure_health():
    return {"status": "ok", "module": "infrastructure_exposure_and_accessibility", "version": "0.1.0"}


@app.post("/satellite/search", response_model=SearchResponse)
def search_satellite_data(request: SearchRequest):
    try:
        if request.geojson:
            aoi = AOI(geojson=request.geojson, name="Request GeoJSON AOI")
        elif request.bbox:
            aoi = AOI(bbox=request.bbox, name="Request BBox AOI")
        else:
            raise HTTPException(status_code=400, detail="Must provide either 'bbox' or 'geojson' in request.")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid AOI parameter: {ve}")

    try:
        search_res = catalog_service.search_scenes(
            aoi=aoi,
            event_date_str=request.event_date,
            pre_event_days=request.pre_event_days,
            post_event_days=request.post_event_days,
            max_results=request.max_results
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Catalogue search service error: {e}")

    compat_eval = compatibility_analyzer.evaluate_compatibility(
        aoi=aoi,
        pre_scenes=search_res.pre_event_scenes,
        post_scenes=search_res.post_event_scenes
    )

    scenes_to_plan = []
    if compat_eval.ranked_candidate_pairs:
        top_pair = compat_eval.ranked_candidate_pairs[0]
        scenes_to_plan = [top_pair.pre_scene, top_pair.post_scene]

    plan = download_planner.create_download_plan(scenes=scenes_to_plan, storage_budget_gb=request.storage_budget_gb)

    return SearchResponse(
        status="ok",
        search_result=search_res,
        compatibility_evaluation=compat_eval,
        download_plan=plan,
        warnings=search_res.warning_messages + plan.warnings
    )


@app.post("/infrastructure/exposure", response_model=ExposureResponse)
def evaluate_exposure(request: ExposureRequest):
    try:
        if request.geojson:
            aoi = AOI(geojson=request.geojson, name="Exposure GeoJSON AOI")
        elif request.bbox:
            aoi = AOI(bbox=request.bbox, name="Exposure BBox AOI")
        else:
            raise HTTPException(status_code=400, detail="Must provide either 'bbox' or 'geojson'.")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid AOI parameter: {ve}")

    if any([request.roads_geojson_path, request.buildings_geojson_path, request.facilities_geojson_path, request.settlements_geojson_path]):
        container, osm_meta = osm_service.load_local_snapshot(
            aoi=aoi,
            roads_geojson_path=request.roads_geojson_path,
            buildings_geojson_path=request.buildings_geojson_path,
            facilities_geojson_path=request.facilities_geojson_path,
            settlements_geojson_path=request.settlements_geojson_path
        )
    else:
        container, osm_meta = osm_service.fetch_overpass_infrastructure(aoi=aoi)

    output_gdfs, summary = exposure_analyzer.analyze_exposure(aoi=aoi, infrastructure=container, change_mask_path=request.change_mask_path)

    run_id = str(uuid.uuid4())[:8]
    run_dir = os.path.join("data", "output", f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    for name, gdf in output_gdfs.items():
        if len(gdf) > 0:
            gdf.to_file(os.path.join(run_dir, f"{name}.geojson"), driver="GeoJSON")

    with open(os.path.join(run_dir, "exposure_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary.model_dump(), f, indent=2)

    with open(os.path.join(run_dir, "processing_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "aoi_name": aoi.name, "osm_source_metadata": osm_meta.model_dump()}, f, indent=2)

    return ExposureResponse(status="ok", run_id=run_id, output_directory=run_dir, exposure_summary=summary, osm_source_metadata=osm_meta, warnings=osm_meta.warnings)


@app.post("/infrastructure/accessibility", response_model=AccessibilityResponse)
def evaluate_accessibility(request: AccessibilityRequest):
    try:
        if request.geojson:
            aoi = AOI(geojson=request.geojson, name="Accessibility GeoJSON AOI")
        elif request.bbox:
            aoi = AOI(bbox=request.bbox, name="Accessibility BBox AOI")
        else:
            raise HTTPException(status_code=400, detail="Must provide either 'bbox' or 'geojson'.")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid AOI parameter: {ve}")

    if any([request.roads_geojson_path, request.facilities_geojson_path, request.settlements_geojson_path]):
        container, osm_meta = osm_service.load_local_snapshot(
            aoi=aoi,
            roads_geojson_path=request.roads_geojson_path,
            facilities_geojson_path=request.facilities_geojson_path,
            settlements_geojson_path=request.settlements_geojson_path
        )
    else:
        container, osm_meta = osm_service.fetch_overpass_infrastructure(aoi=aoi)

    road_graph, graph_summary = road_builder.build_graph(container.roads)

    closed_road_ids = set()
    if request.simulate_flood_closures and request.change_mask_path:
        output_gdfs, exposure_sum = exposure_analyzer.analyze_exposure(
            aoi=aoi, infrastructure=container, change_mask_path=request.change_mask_path
        )
        exposed_roads = output_gdfs.get("exposed_roads")
        if exposed_roads is not None and len(exposed_roads) > 0:
            if "osm_id" in exposed_roads.columns:
                closed_road_ids = set(exposed_roads["osm_id"].astype(str).tolist())
            elif "id" in exposed_roads.columns:
                closed_road_ids = set(exposed_roads["id"].astype(str).tolist())

    assessed_settlements, acc_summary = accessibility_analyzer.analyze_accessibility(
        road_graph=road_graph,
        settlements_gdf=container.settlements,
        destinations_gdf=container.facilities,
        closed_road_ids=closed_road_ids if closed_road_ids else None
    )

    run_id = str(uuid.uuid4())[:8]
    run_dir = os.path.join("data", "output", f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    if len(assessed_settlements) > 0:
        assessed_settlements.to_file(os.path.join(run_dir, "assessed_settlements.geojson"), driver="GeoJSON")

    with open(os.path.join(run_dir, "accessibility_summary.json"), "w", encoding="utf-8") as f:
        json.dump(acc_summary.model_dump(), f, indent=2)

    with open(os.path.join(run_dir, "processing_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "aoi_name": aoi.name, "osm_source_metadata": osm_meta.model_dump()}, f, indent=2)

    return AccessibilityResponse(
        status="ok",
        run_id=run_id,
        output_directory=run_dir,
        accessibility_summary=acc_summary,
        osm_source_metadata=osm_meta,
        warnings=osm_meta.warnings
    )


@app.post("/terrain/analyze")
def analyze_terrain_endpoint(request: TerrainRequest):
    """Analyze elevation, slope, and aspect for an input DEM GeoTIFF."""
    if not os.path.exists(request.dem_path):
        raise HTTPException(status_code=404, detail=f"DEM file not found: '{request.dem_path}'")

    try:
        res = terrain_analyzer.analyze_terrain(request.dem_path)
        return {"status": "ok", "terrain_analysis": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terrain analysis error: {e}")


@app.post("/terrain/trace-flow")
def trace_flow_endpoint(request: FlowTracingRequest):
    """Trace D8 steepest-descent terrain flow paths from starting points."""
    if not os.path.exists(request.dem_path):
        raise HTTPException(status_code=404, detail=f"DEM file not found: '{request.dem_path}'")

    try:
        res = flow_tracer.trace_flow_paths(request.dem_path, request.start_points_wgs84)
        return {"status": "ok", "flow_tracing_result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Flow tracing error: {e}")


@app.get("/dashboard/analysis/{run_id}")
def get_dashboard_analysis(run_id: str):
    """Retrieve structured processing summary for a specific analysis run."""
    run_folder = run_id if run_id.startswith("run_") else f"run_{run_id}"
    run_dir = os.path.join("data", "output", run_folder)
    if not os.path.exists(run_dir):
        raise HTTPException(status_code=404, detail=f"Run ID '{run_id}' not found.")

    meta_file = os.path.join(run_dir, "processing_metadata.json")
    exp_file = os.path.join(run_dir, "exposure_summary.json")

    meta = json.load(open(meta_file)) if os.path.exists(meta_file) else {}
    exp = json.load(open(exp_file)) if os.path.exists(exp_file) else {}

    return {
        "status": "ok",
        "run_id": run_id,
        "processing_metadata": meta,
        "exposure_summary": exp,
        "available_layers": [f for f in os.listdir(run_dir) if f.endswith(".geojson")]
    }


@app.get("/dashboard/nepal-summary")
def get_nepal_dashboard_summary():
    """Retrieve structured metadata, WGS84 bounding boxes, layer URLs, and palettes for the Nepal multi-temporal analysis."""
    try:
        summary = visualizer_service.get_nepal_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load Nepal summary: {e}")


@app.get("/dashboard/layers/{run_id}/{layer_name}")
def get_dashboard_layer(run_id: str, layer_name: str):
    """Securely serve vector GeoJSON layers or raster overlays for map rendering."""
    clean_name = os.path.basename(layer_name)
    run_folder = run_id if run_id.startswith("run_") else f"run_{run_id}"
    run_dir = os.path.abspath(os.path.join("data", "output", run_folder))
    layer_path = os.path.abspath(os.path.join(run_dir, clean_name))

    if not layer_path.startswith(os.path.abspath("data")):
        raise HTTPException(status_code=403, detail="Forbidden: Path traversal blocked.")

    # On-demand generation for agreement_overlay.png if source raster exists
    if not os.path.exists(layer_path) and clean_name == "agreement_overlay.png":
        src_raster = os.path.join(run_dir, "nepal_water_and_change_agreement_mask.tif")
        if os.path.exists(src_raster):
            visualizer_service.render_agreement_overlay(src_raster, layer_path)

    # On-demand generation for slope_overlay.png if source raster exists
    if not os.path.exists(layer_path) and clean_name == "slope_overlay.png":
        src_slope = os.path.join("data", "processed", "nepal_copernicus_slope_30m.tif")
        if os.path.exists(src_slope):
            visualizer_service.render_slope_overlay(src_slope, layer_path)

    if not os.path.exists(layer_path):
        raise HTTPException(status_code=404, detail=f"Layer '{clean_name}' not found for run '{run_id}'.")

    if clean_name.endswith(".geojson") or clean_name.endswith(".json"):
        return FileResponse(layer_path, media_type="application/json")
    elif clean_name.endswith(".png"):
        return FileResponse(layer_path, media_type="image/png")
    elif clean_name.endswith(".tif") or clean_name.endswith(".tiff"):
        return FileResponse(layer_path, media_type="image/tiff")

    raise HTTPException(status_code=400, detail="Unsupported layer file format.")


@app.get("/segmentation/available-rasters")
def get_available_rasters():
    """Discover local SAR GeoTIFF files available in the workspace for inference."""
    try:
        rasters = unet_segmentation_service.discover_available_rasters()
        return {
            "status": "ok",
            "count": len(rasters),
            "rasters": [r.model_dump() for r in rasters]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scan workspace rasters: {e}")


@app.post("/segmentation/upload")
async def upload_sar_raster_endpoint(file: UploadFile = File(...)):
    """Upload and stage a local SAR GeoTIFF file for Small U-Net inference."""
    try:
        content = await file.read()
        res = unet_segmentation_service.save_uploaded_raster(content, file.filename)
        return {
            "status": "ok",
            "message": f"File '{res['filename']}' successfully uploaded and validated.",
            "uploaded_raster": res
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {e}")


@app.post("/segmentation/predict", response_model=SegmentationResponse)
def run_segmentation_prediction(request: SegmentationRequest):
    """Run Small U-Net preliminary water segmentation inference on target 3-band SAR raster."""
    try:
        response = unet_segmentation_service.predict(request)
        return response
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation inference error: {e}")



@app.post("/reports/situation")
def generate_situation_report_endpoint(request: ReportRequest):
    """Generate downloadable evidence-grounded situation report in Markdown format."""
    try:
        report_md = report_generator.generate_report(request, output_dir=os.path.join("data", "output"))
        return {
            "status": "ok",
            "run_id": request.run_id,
            "report_content": report_md,
            "report_filename": f"sara_situation_report_{request.run_id}.md"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation error: {e}")


@app.get("/preprocessing/snap-status")
def get_snap_status():
    """Inspect ESA SNAP gpt installation status, version, and operator availability."""
    available = snap_adapter.is_snap_available()
    return {
        "status": "ok",
        "snap_installed": available,
        "snap_version": snap_adapter.get_snap_version() if available else None,
        "gpt_path": snap_adapter.gpt_path if available else None,
        "operator_status": snap_adapter.verify_operators(),
        "installation_instructions": None if available else snap_adapter.get_installation_instructions()
    }


class GraphRequest(BaseModel):
    input_safe_path: str
    output_raster_path: str
    calibration_type: str = "SIGMA0"
    output_in_db: bool = True
    speckle_filter_type: str = "Refined Lee"
    crs: str = "EPSG:4326"


@app.post("/preprocessing/generate-graph")
def generate_processing_graph_endpoint(request: GraphRequest):
    """Generate reproducible SNAP XML graph processing workflow for Sentinel-1 GRD."""
    config = SNAPGraphConfig(
        calibration_type=request.calibration_type,
        output_in_db=request.output_in_db,
        speckle_filter_type=request.speckle_filter_type,
        crs=request.crs
    )
    builder = SNAPGraphBuilder()
    xml_content = builder.build_grd_ard_graph_xml(
        request.input_safe_path,
        request.output_raster_path,
        config
    )
    return {
        "status": "ok",
        "input_safe_path": request.input_safe_path,
        "output_raster_path": request.output_raster_path,
        "xml_graph_content": xml_content
    }


@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard_ui():
    """Serve SARA interactive web dashboard UI matching the dark navy geospatial intelligence reference design."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SARA — Flood Mapping & Accessibility Dashboard | Satellite-Assisted Rescue & Assessment</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        :root {
            --bg-base: #050c1b;
            --bg-header: #071228;
            --bg-sidebar: #060f22;
            --bg-card: #0c1834;
            --bg-card-hover: #102046;
            --bg-panel: #091530;
            --border: #142850;
            --border-subtle: #1c366a;
            --cyan: #00f0ff;
            --cyan-glow: rgba(0, 240, 255, 0.25);
            --cyan-dim: rgba(0, 240, 255, 0.12);
            --blue-accent: #38bdf8;
            --blue-active: #0284c7;
            --green: #10b981;
            --amber: #f59e0b;
            --orange: #f97316;
            --text-primary: #ffffff;
            --text-secondary: #94a3b8;
            --text-muted: #59739d;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
            background: var(--bg-base);
            color: var(--text-primary);
            overflow: hidden;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }

        /* 1. Header */
        #header {
            height: 52px;
            background: var(--bg-header);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 18px;
            flex-shrink: 0;
            z-index: 1000;
        }
        .header-left {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .header-logo {
            width: 26px;
            height: 26px;
            color: var(--cyan);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .header-title-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .brand-title {
            font-size: 18px;
            font-weight: 800;
            letter-spacing: 1px;
            color: #ffffff;
        }
        .brand-divider {
            color: var(--border-subtle);
            font-weight: 300;
            font-size: 16px;
        }
        .brand-subtitle {
            font-size: 13px;
            color: var(--text-secondary);
            font-weight: 400;
        }
        .badge-prototype {
            background: rgba(0, 240, 255, 0.12);
            border: 1px solid rgba(0, 240, 255, 0.4);
            color: var(--cyan);
            font-size: 9px;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 12px;
            letter-spacing: 0.8px;
            margin-left: 6px;
        }
        .header-right {
            display: flex;
            align-items: center;
            gap: 20px;
        }
        .header-date {
            display: flex;
            align-items: center;
            gap: 7px;
            font-size: 12px;
            color: var(--text-secondary);
        }
        .header-user {
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
        }
        .user-avatar {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: #1e3a8a;
            color: #93c5fd;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 700;
            border: 1px solid #3b82f6;
        }
        .user-name {
            font-size: 12px;
            font-weight: 600;
            color: #e2e8f0;
        }

        /* 2. Workspace Layout */
        #workspace {
            display: flex;
            flex: 1;
            height: calc(100vh - 52px);
            overflow: hidden;
        }

        /* Left Navigation Sidebar */
        #navSidebar {
            width: 210px;
            background: var(--bg-sidebar);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            flex-shrink: 0;
            padding: 12px 0 16px 0;
        }
        .nav-list {
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .nav-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 18px;
            font-size: 12.5px;
            font-weight: 500;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.15s ease;
            border-left: 3px solid transparent;
            user-select: none;
        }
        .nav-item:hover {
            background: rgba(255, 255, 255, 0.03);
            color: #ffffff;
        }
        .nav-item.active {
            background: var(--cyan-dim);
            color: var(--cyan);
            border-left-color: var(--cyan);
            font-weight: 600;
        }
        .nav-icon {
            width: 18px;
            height: 18px;
            flex-shrink: 0;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .nav-bottom-badge {
            padding: 14px 18px;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            gap: 6px;
        }
        .nepal-logo-icon {
            width: 32px;
            height: 32px;
            color: var(--blue-accent);
            opacity: 0.85;
        }
        .nepal-title {
            font-size: 11.5px;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: 0.3px;
        }
        .nepal-sub {
            font-size: 9.5px;
            color: var(--text-muted);
            line-height: 1.3;
        }

        /* Center Content Container */
        #contentContainer {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: var(--bg-base);
            overflow: hidden;
            position: relative;
        }
        .view-pane {
            display: none;
            flex: 1;
            height: 100%;
            flex-direction: column;
            overflow-y: auto;
            position: relative;
        }
        .view-pane.active {
            display: flex;
        }

        /* 3. Top KPI Cards */
        .kpi-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            padding: 12px 16px 8px 16px;
            flex-shrink: 0;
        }
        .kpi-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 14px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: border-color 0.2s;
        }
        .kpi-card:hover {
            border-color: var(--border-subtle);
        }
        .kpi-left {
            display: flex;
            align-items: flex-start;
            gap: 12px;
        }
        .kpi-icon-wrap {
            width: 32px;
            height: 32px;
            border-radius: 6px;
            background: rgba(0, 240, 255, 0.08);
            color: var(--cyan);
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            margin-top: 1px;
        }
        .kpi-info {
            display: flex;
            flex-direction: column;
        }
        .kpi-title {
            font-size: 11.5px;
            color: var(--text-secondary);
            font-weight: 500;
        }
        .kpi-value {
            font-size: 20px;
            font-weight: 700;
            color: #ffffff;
            margin: 2px 0 1px 0;
            line-height: 1.1;
        }
        .kpi-sub {
            font-size: 9.5px;
            color: var(--text-muted);
        }
        .kpi-ring-wrap {
            width: 42px;
            height: 42px;
            flex-shrink: 0;
            position: relative;
        }
        .kpi-ring-svg {
            transform: rotate(-90deg);
        }
        .ring-bg {
            stroke: #12244a;
            stroke-width: 3.2;
            fill: none;
        }
        .ring-fill {
            stroke: var(--cyan);
            stroke-width: 3.2;
            fill: none;
            stroke-linecap: round;
            transition: stroke-dashoffset 0.6s ease;
        }

        /* 4. Map Explorer Layout */
        #mapWorkspace {
            flex: 1;
            display: flex;
            padding: 4px 16px 8px 16px;
            gap: 12px;
            min-height: 0;
            position: relative;
        }
        #mapOuter {
            flex: 1;
            border-radius: 8px;
            border: 1px solid var(--border);
            overflow: hidden;
            position: relative;
            background: #081024;
            display: flex;
        }
        #map {
            width: 100%;
            height: 100%;
            background: #060e1f;
        }

        /* Custom Leaflet Controls */
        .leaflet-control-zoom {
            border: 1px solid var(--border) !important;
            border-radius: 4px !important;
            overflow: hidden;
        }
        .leaflet-control-zoom a {
            background-color: var(--bg-card) !important;
            border-bottom: 1px solid var(--border) !important;
            color: var(--cyan) !important;
            width: 28px !important;
            height: 28px !important;
            line-height: 28px !important;
        }
        .leaflet-control-zoom a:hover {
            background-color: var(--bg-panel) !important;
            color: #ffffff !important;
        }
        .map-target-btn {
            position: absolute;
            top: 76px;
            left: 10px;
            z-index: 500;
            width: 28px;
            height: 28px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--cyan);
            cursor: pointer;
            box-shadow: 0 1px 5px rgba(0,0,0,0.5);
        }
        .map-target-btn:hover {
            background: var(--bg-panel);
            color: #ffffff;
        }

        /* Right Map Layers Panel */
        #layersPanel {
            width: 260px;
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
            overflow-y: auto;
            transition: width 0.2s ease;
        }
        #layersPanel.collapsed {
            width: 36px;
            overflow: hidden;
        }
        .panel-header {
            padding: 10px 14px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 11.5px;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: 0.5px;
        }
        .panel-header-left {
            display: flex;
            align-items: center;
            gap: 7px;
        }
        .panel-toggle-btn {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            display: flex;
            align-items: center;
            padding: 2px;
        }
        .panel-toggle-btn:hover {
            color: var(--cyan);
        }
        .layers-content {
            padding: 10px 14px;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }
        .layer-row {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        .layer-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .layer-label {
            display: flex;
            align-items: center;
            gap: 7px;
            font-size: 11.5px;
            font-weight: 600;
            color: #e2e8f0;
            cursor: pointer;
        }
        .layer-icon {
            color: var(--cyan);
            width: 15px;
            height: 15px;
            display: flex;
            align-items: center;
        }
        
        /* Modern Switch */
        .switch {
            position: relative;
            display: inline-block;
            width: 30px;
            height: 16px;
            flex-shrink: 0;
        }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider-switch {
            position: absolute;
            cursor: pointer;
            top: 0; left: 0; right: 0; bottom: 0;
            background-color: #142850;
            transition: .2s;
            border-radius: 16px;
        }
        .slider-switch:before {
            position: absolute;
            content: "";
            height: 12px;
            width: 12px;
            left: 2px;
            bottom: 2px;
            background-color: #94a3b8;
            transition: .2s;
            border-radius: 50%;
        }
        input:checked + .slider-switch {
            background-color: #0284c7;
        }
        input:checked + .slider-switch:before {
            transform: translateX(14px);
            background-color: #ffffff;
        }

        /* Opacity Slider styling */
        .opacity-control {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 2px;
        }
        .opacity-title {
            font-size: 9.5px;
            color: var(--text-muted);
            width: 38px;
        }
        .custom-range {
            flex: 1;
            -webkit-appearance: none;
            height: 4px;
            border-radius: 2px;
            outline: none;
            background: #142850;
        }
        .custom-range::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 11px;
            height: 11px;
            border-radius: 50%;
            background: #ffffff;
            cursor: pointer;
            border: 1px solid var(--cyan);
            box-shadow: 0 0 4px rgba(0,0,0,0.5);
        }
        .track-rainbow {
            background: linear-gradient(to right, #4338ca, #06b6d4, #eab308, #ef4444);
        }
        .track-slope {
            background: linear-gradient(to right, #22c55e, #eab308, #f97316, #ef4444);
        }
        .track-agreement {
            background: linear-gradient(to right, #1d4ed8, #eab308, #06b6d4, #ea580c);
        }
        .track-cyan {
            background: linear-gradient(to right, rgba(0,240,255,0.2), #00f0ff);
        }
        .opacity-val {
            font-size: 9.5px;
            color: var(--text-secondary);
            width: 28px;
            text-align: right;
            font-weight: 500;
        }
        .basemap-select {
            width: 100%;
            background: var(--bg-base);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 5px 8px;
            border-radius: 4px;
            font-size: 10.5px;
            margin-top: 4px;
            outline: none;
            cursor: pointer;
        }

        /* 5. Bottom Timeline & Legend */
        .bottom-meta-row {
            display: grid;
            grid-template-columns: 260px 1fr;
            gap: 12px;
            padding: 0 16px 8px 16px;
            flex-shrink: 0;
        }
        .timeline-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 8px 14px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .timeline-icon {
            color: var(--cyan);
        }
        .timeline-text {
            font-size: 10px;
            color: var(--text-secondary);
            margin-bottom: 2px;
        }
        .timeline-dates {
            font-size: 11.5px;
            font-weight: 700;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .legend-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 8px 12px;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            overflow-x: auto;
        }
        .legend-title {
            font-size: 9.5px;
            font-weight: 700;
            color: var(--text-secondary);
            text-transform: uppercase;
            margin-right: 10px;
            white-space: nowrap;
        }
        .legend-items {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: nowrap;
        }
        .legend-chip {
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 9.5px;
            color: #cbd5e1;
            white-space: nowrap;
        }
        .chip-box {
            width: 11px;
            height: 11px;
            border-radius: 2px;
            flex-shrink: 0;
        }
        .chip-striped {
            background: repeating-linear-gradient(
                45deg,
                #ea580c,
                #ea580c 2px,
                #78350f 2px,
                #78350f 4px
            );
        }

        /* 6. Current Run Bottom Bar */
        .run-bar {
            background: var(--bg-panel);
            border-top: 1px solid var(--border);
            padding: 8px 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-shrink: 0;
        }
        .run-left {
            display: flex;
            align-items: center;
            gap: 20px;
        }
        .run-badge-title {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            font-weight: 700;
            color: #ffffff;
        }
        .run-icon-play {
            color: var(--cyan);
            display: flex;
            align-items: center;
        }
        .run-param {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            color: var(--text-secondary);
        }
        .run-param-val {
            font-weight: 600;
            color: #f1f5f9;
        }
        .status-completed {
            display: flex;
            align-items: center;
            gap: 5px;
            color: var(--green);
            font-weight: 700;
            font-size: 11px;
        }
        .btn-cyan {
            background: #0284c7;
            color: #ffffff;
            border: none;
            border-radius: 4px;
            padding: 6px 14px;
            font-size: 11.5px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: background 0.15s;
        }
        .btn-cyan:hover {
            background: #0369a1;
        }

        /* 7. Disclaimer Banner */
        .disclaimer-banner {
            background: #100d1e;
            border-top: 1px solid #281d48;
            padding: 5px 16px;
            font-size: 9.5px;
            color: #cbd5e1;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-shrink: 0;
        }
        .disclaimer-left {
            display: flex;
            align-items: center;
            gap: 6px;
            color: #fcd34d;
        }

        /* 8. Subviews (AI Segmentation, Overview, etc.) */
        .subview-body {
            padding: 20px 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            max-width: 1200px;
            margin: 0 auto;
            width: 100%;
        }
        .view-header-title {
            font-size: 18px;
            font-weight: 700;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .view-header-sub {
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 2px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }
        .card-panel {
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
        }
        .card-panel h3 {
            font-size: 13px;
            color: var(--cyan);
            margin-bottom: 12px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .form-group {
            margin-bottom: 12px;
        }
        .form-label {
            font-size: 11px;
            color: var(--text-secondary);
            font-weight: 500;
            margin-bottom: 4px;
            display: block;
        }
        .form-input, .form-select {
            width: 100%;
            background: var(--bg-base);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 7px 10px;
            font-size: 11px;
            color: #ffffff;
            outline: none;
        }
        .form-input:focus, .form-select:focus {
            border-color: var(--cyan);
        }
        .file-dropzone {
            border: 1px dashed var(--blue-accent);
            border-radius: 6px;
            padding: 14px;
            text-align: center;
            background: rgba(56, 189, 248, 0.04);
            cursor: pointer;
            transition: all 0.2s;
        }
        .file-dropzone:hover {
            background: rgba(56, 189, 248, 0.08);
            border-color: var(--cyan);
        }
        .btn-primary {
            background: #0284c7;
            color: #ffffff;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            transition: background 0.15s;
        }
        .btn-primary:hover { background: #0369a1; }
        .btn-primary:disabled { background: #475569; cursor: not-allowed; }

        /* Modal */
        .modal-backdrop {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0, 0, 0, 0.75);
            z-index: 2000;
            align-items: center;
            justify-content: center;
        }
        .modal-backdrop.open { display: flex; }
        .modal-box {
            background: var(--bg-panel);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            width: 650px;
            max-width: 90vw;
            max-height: 80vh;
            overflow-y: auto;
            padding: 20px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.6);
        }
        .modal-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border);
            padding-bottom: 10px;
            margin-bottom: 14px;
        }
        .modal-title { font-size: 14px; font-weight: 700; color: #ffffff; }
        .modal-close { background: none; border: none; color: var(--text-secondary); cursor: pointer; font-size: 18px; }

        .hidden { display: none !important; }
        .flex-between { display: flex; align-items: center; justify-content: space-between; }
        .text-cyan { color: var(--cyan); }
        .text-amber { color: var(--amber); }
    </style>
</head>
<body>

    <!-- 1. Header -->
    <header id="header">
        <div class="header-left">
            <div class="header-logo">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><circle cx="12" cy="12" r="9"/><path d="M12 3a9 9 0 0 1 9 9"/><path d="M12 7a5 5 0 0 1 5 5"/><circle cx="12" cy="12" r="1.5" fill="currentColor"/></svg>
            </div>
            <div class="header-title-group">
                <span class="brand-title">SARA</span>
                <span class="brand-divider">|</span>
                <span class="brand-subtitle">Satellite-Assisted Rescue & Assessment</span>
                <span class="badge-prototype">RESEARCH PROTOTYPE</span>
            </div>
        </div>
        <div class="header-right">
            <div class="header-date">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>
                <span id="headerTime">Oct 3, 2026 14:32 (IST)</span>
            </div>
            <div class="header-user">
                <div class="user-avatar">CU</div>
                <span class="user-name">Charan</span>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
            </div>
        </div>
    </header>

    <!-- 2. Workspace -->
    <div id="workspace">

        <!-- Left Navigation Sidebar -->
        <nav id="navSidebar">
            <ul class="nav-list">
                <li class="nav-item" data-view="overview" onclick="switchNav('overview')">
                    <span class="nav-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg></span>
                    <span>Overview</span>
                </li>
                <li class="nav-item active" data-view="map" onclick="switchNav('map')">
                    <span class="nav-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/><line x1="8" y1="2" x2="8" y2="18"/><line x1="16" y1="6" x2="16" y2="22"/></svg></span>
                    <span>Map Explorer</span>
                </li>
                <li class="nav-item" data-view="segmentation" onclick="switchNav('segmentation')">
                    <span class="nav-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="15" x2="23" y2="15"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="15" x2="4" y2="15"/></svg></span>
                    <span>AI Segmentation</span>
                </li>
                <li class="nav-item" data-view="change-detection" onclick="switchNav('change-detection')">
                    <span class="nav-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg></span>
                    <span>Change Detection</span>
                </li>
                <li class="nav-item" data-view="terrain-analysis" onclick="switchNav('terrain-analysis')">
                    <span class="nav-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m8 3 4 8 5-5 5 15H2L8 3z"/></svg></span>
                    <span>Terrain Analysis</span>
                </li>
                <li class="nav-item" data-view="infrastructure-impact" onclick="switchNav('infrastructure-impact')">
                    <span class="nav-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/></svg></span>
                    <span>Infrastructure Impact</span>
                </li>
                <li class="nav-item" data-view="settlement-accessibility" onclick="switchNav('settlement-accessibility')">
                    <span class="nav-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="6" cy="19" r="3"/><path d="M9 19h8.5a4.5 4.5 0 0 0 4.5-4.5V14"/><circle cx="18" cy="5" r="3"/></svg></span>
                    <span>Settlement Accessibility</span>
                </li>
                <li class="nav-item" data-view="validation-reports" onclick="switchNav('validation-reports')">
                    <span class="nav-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg></span>
                    <span>Validation & Reports</span>
                </li>
            </ul>

            <div class="nav-bottom-badge">
                <div class="nepal-logo-icon">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="m2 20 7-12 5 7 4-5 4 10H2z"/></svg>
                </div>
                <div class="nepal-title">Nepal Himalayas</div>
                <div class="nepal-sub">Disaster Intelligence for<br>a Safer Tomorrow</div>
            </div>
        </nav>

        <!-- Center Content Views -->
        <main id="contentContainer">

            <!-- VIEW 1: MAP EXPLORER (DEFAULT) -->
            <div id="view-map" class="view-pane active">

                <!-- 3. Top KPI Cards -->
                <div class="kpi-row">
                    <!-- KPI 1 -->
                    <div class="kpi-card">
                        <div class="kpi-left">
                            <div class="kpi-icon-wrap">
                                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/></svg>
                            </div>
                            <div class="kpi-info">
                                <span class="kpi-title">Valid SAR coverage</span>
                                <span class="kpi-value" id="kpiSarCoverage">54.95%</span>
                                <span class="kpi-sub" id="kpiSarCoverageSub">of total AOI (379,266,320 pixels)</span>
                            </div>
                        </div>
                        <div class="kpi-ring-wrap">
                            <svg class="kpi-ring-svg" width="42" height="42" viewBox="0 0 36 36">
                                <path class="ring-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                                <path class="ring-fill" id="ringSarCoverage" stroke-dasharray="54.95, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                            </svg>
                        </div>
                    </div>

                    <!-- KPI 2 -->
                    <div class="kpi-card">
                        <div class="kpi-left">
                            <div class="kpi-icon-wrap" style="color: #38bdf8; background: rgba(56, 189, 248, 0.1);">
                                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/></svg>
                            </div>
                            <div class="kpi-info">
                                <span class="kpi-title">Model-predicted water</span>
                                <span class="kpi-value" id="kpiWaterPred">8.52%</span>
                                <span class="kpi-sub" id="kpiWaterPredSub">of valid pixels (predicted, not confirmed)</span>
                            </div>
                        </div>
                        <div class="kpi-ring-wrap">
                            <svg class="kpi-ring-svg" width="42" height="42" viewBox="0 0 36 36">
                                <path class="ring-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                                <path class="ring-fill" id="ringWaterPred" stroke="#38bdf8" stroke-dasharray="8.52, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                            </svg>
                        </div>
                    </div>

                    <!-- KPI 3 -->
                    <div class="kpi-card">
                        <div class="kpi-left">
                            <div class="kpi-icon-wrap" style="color: #10b981; background: rgba(16, 185, 129, 0.1);">
                                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>
                            </div>
                            <div class="kpi-info">
                                <span class="kpi-title">Backscatter decrease</span>
                                <span class="kpi-value" id="kpiBackscatterDec">14.29%</span>
                                <span class="kpi-sub" id="kpiBackscatterDecSub">of valid pixels (&Delta;&sigma;&deg; &le; -3 dB)</span>
                            </div>
                        </div>
                        <div class="kpi-ring-wrap">
                            <svg class="kpi-ring-svg" width="42" height="42" viewBox="0 0 36 36">
                                <path class="ring-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                                <path class="ring-fill" id="ringBackscatter" stroke="#10b981" stroke-dasharray="14.29, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                            </svg>
                        </div>
                    </div>

                    <!-- KPI 4 -->
                    <div class="kpi-card">
                        <div class="kpi-left">
                            <div class="kpi-icon-wrap" style="color: #f59e0b; background: rgba(245, 158, 11, 0.1);">
                                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                            </div>
                            <div class="kpi-info">
                                <span class="kpi-title">Independent validation</span>
                                <span class="kpi-value text-amber" id="kpiValidation">Pending</span>
                                <span class="kpi-sub" id="kpiValidationSub">(not yet available)</span>
                            </div>
                        </div>
                        <div class="kpi-ring-wrap">
                            <svg class="kpi-ring-svg" width="42" height="42" viewBox="0 0 36 36">
                                <path class="ring-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                                <path class="ring-fill" stroke="#f59e0b" stroke-dasharray="0, 100" stroke-dashoffset="0" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                            </svg>
                        </div>
                    </div>
                </div>

                <!-- 4. Map & Layers Workspace -->
                <div id="mapWorkspace">
                    <div id="mapOuter">
                        <div id="map"></div>
                        <button class="map-target-btn" id="btnResetAOI" onclick="resetToNepalView()" title="Fit Map to AOI">
                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="7"/><line x1="12" y1="1" x2="12" y2="4"/><line x1="12" y1="20" x2="12" y2="23"/><line x1="1" y1="12" x2="4" y2="12"/><line x1="20" y1="12" x2="23" y2="12"/></svg>
                        </button>
                    </div>

                    <!-- Right Map Layers Panel -->
                    <div id="layersPanel">
                        <div class="panel-header">
                            <div class="panel-header-left">
                                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
                                <span>Map Layers</span>
                            </div>
                            <button class="panel-toggle-btn" onclick="toggleLayersPanel()">
                                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="15 18 9 12 15 6"/></svg>
                            </button>
                        </div>

                        <div class="layers-content">
                            <!-- Layer 1: Model Water Probability Heatmap -->
                            <div class="layer-row">
                                <div class="layer-top">
                                    <label class="layer-label" for="layerWaterProb">
                                        <span class="layer-icon"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/></svg></span>
                                        <span>Model Water Probability (Heatmap)</span>
                                    </label>
                                    <label class="switch">
                                        <input type="checkbox" id="layerWaterProb" onchange="updateOverlays()">
                                        <span class="slider-switch"></span>
                                    </label>
                                </div>
                                <div class="opacity-control">
                                    <span class="opacity-title">Opacity</span>
                                    <input type="range" id="opacityWaterProb" class="custom-range track-rainbow" min="0" max="100" value="70" oninput="updateOverlays()" />
                                    <span class="opacity-val" id="valWaterProb">70%</span>
                                </div>
                            </div>

                            <!-- Layer 2: SAR Change Candidates / Multi-Temporal Agreement -->
                            <div class="layer-row">
                                <div class="layer-top">
                                    <label class="layer-label" for="layerAgreement">
                                        <span class="layer-icon"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg></span>
                                        <span>SAR Change Candidates</span>
                                    </label>
                                    <label class="switch">
                                        <input type="checkbox" id="layerAgreement" checked onchange="updateOverlays()">
                                        <span class="slider-switch"></span>
                                    </label>
                                </div>
                                <div class="opacity-control">
                                    <span class="opacity-title">Opacity</span>
                                    <input type="range" id="opacityAgreement" class="custom-range track-agreement" min="0" max="100" value="60" oninput="updateOverlays()" />
                                    <span class="opacity-val" id="valAgreement">60%</span>
                                </div>
                            </div>

                            <!-- Layer 3: Terrain Slope (Copernicus 30m) -->
                            <div class="layer-row">
                                <div class="layer-top">
                                    <label class="layer-label" for="layerSlope">
                                        <span class="layer-icon"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m8 3 4 8 5-5 5 15H2L8 3z"/></svg></span>
                                        <span>Terrain Slope (Copernicus 30m)</span>
                                    </label>
                                    <label class="switch">
                                        <input type="checkbox" id="layerSlope" onchange="updateOverlays()">
                                        <span class="slider-switch"></span>
                                    </label>
                                </div>
                                <div class="opacity-control">
                                    <span class="opacity-title">Opacity</span>
                                    <input type="range" id="opacitySlope" class="custom-range track-slope" min="0" max="100" value="50" oninput="updateOverlays()" />
                                    <span class="opacity-val" id="valSlope">50%</span>
                                </div>
                            </div>

                            <!-- Layer 4: Model Water Mask (Binary) -->
                            <div class="layer-row">
                                <div class="layer-top">
                                    <label class="layer-label" for="layerWaterBin">
                                        <span class="layer-icon"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/></svg></span>
                                        <span id="labelWaterBin">Active Model Water Mask</span>
                                    </label>
                                    <label class="switch">
                                        <input type="checkbox" id="layerWaterBin" checked onchange="updateOverlays()">
                                        <span class="slider-switch"></span>
                                    </label>
                                </div>
                                <div class="opacity-control">
                                    <span class="opacity-title">Opacity</span>
                                    <input type="range" id="opacityWaterBin" class="custom-range track-cyan" min="0" max="100" value="80" oninput="updateOverlays()" />
                                    <span class="opacity-val" id="valWaterBin">80%</span>
                                </div>
                            </div>

                            <!-- Layer 5: Target AOI Boundary -->
                            <div class="layer-row">
                                <div class="layer-top">
                                    <label class="layer-label" for="layerAOI">
                                        <span class="layer-icon"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" stroke-dasharray="3 3"/></svg></span>
                                        <span>Target AOI Boundary</span>
                                    </label>
                                    <label class="switch">
                                        <input type="checkbox" id="layerAOI" checked onchange="toggleAOI()">
                                        <span class="slider-switch"></span>
                                    </label>
                                </div>
                                <div class="opacity-control">
                                    <span class="opacity-title">Opacity</span>
                                    <input type="range" id="opacityAOI" class="custom-range track-cyan" min="0" max="100" value="100" oninput="toggleAOI()" />
                                    <span class="opacity-val" id="valAOI">100%</span>
                                </div>
                                <div style="width:100%; border-bottom: 2px dashed #00f0ff; margin-top:2px; opacity:0.8;"></div>
                            </div>

                            <!-- Layer 6: Base Map Selector -->
                            <div class="layer-row" style="margin-top: 4px;">
                                <div class="layer-top">
                                    <label class="layer-label">
                                        <span class="layer-icon"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/></svg></span>
                                        <span>Base Map</span>
                                    </label>
                                    <label class="switch">
                                        <input type="checkbox" id="layerBaseMap" checked onchange="toggleBaseMap(this.checked)">
                                        <span class="slider-switch"></span>
                                    </label>
                                </div>
                                <select id="baseMapSelect" class="basemap-select" onchange="changeBaseMap(this.value)">
                                    <option value="esri" selected>ESRI World Imagery (Satellite)</option>
                                    <option value="osm">OpenStreetMap</option>
                                    <option value="carto">CartoDB Dark Matter</option>
                                </select>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 5. Bottom Timeline & Legend -->
                <div class="bottom-meta-row">
                    <!-- Timeline box -->
                    <div class="timeline-card">
                        <div class="timeline-icon">
                            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>
                        </div>
                        <div>
                            <div class="timeline-text">Pre-event &rarr; Post-event</div>
                            <div class="timeline-dates" id="timelineDatesDisplay">
                                <span>Aug 16, 2026</span>
                                <span style="color:var(--cyan); margin: 0 2px;">&rarr;</span>
                                <span>Aug 28, 2026</span>
                            </div>
                        </div>
                    </div>

                    <!-- 5-Class Legend Box -->
                    <div class="legend-card">
                        <span class="legend-title">Layer Legend</span>
                        <div class="legend-items">
                            <div class="legend-chip">
                                <div class="chip-box" style="background:#1d4ed8;"></div>
                                <span>Class 1: Persistent Water Candidate</span>
                            </div>
                            <div class="legend-chip">
                                <div class="chip-box" style="background:#eab308;"></div>
                                <span>Class 2: Backscatter Change Only</span>
                            </div>
                            <div class="legend-chip">
                                <div class="chip-box" style="background:#06b6d4;"></div>
                                <span>Class 3: Agreed Inundation Candidate</span>
                            </div>
                            <div class="legend-chip">
                                <div class="chip-box" style="background:#ea580c;"></div>
                                <span>Class 4: Suspected High-Slope Shadow Candidate</span>
                            </div>
                            <div class="legend-chip">
                                <div class="chip-box" style="background:#334155;"></div>
                                <span>Class 0: Dry Land / No Change</span>
                            </div>
                            <div class="legend-chip">
                                <div class="chip-box chip-striped"></div>
                                <span>Steep Slope (>15°)</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 6. Current Run Bottom Bar -->
                <div class="run-bar">
                    <div class="run-left">
                        <div class="run-badge-title">
                            <span class="run-icon-play"><svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="10" fill="rgba(0, 240, 255, 0.15)" stroke="var(--cyan)" stroke-width="1.8"/><polygon points="10 8 16 12 10 16 10 8" fill="var(--cyan)"/></svg></span>
                            <span>Current run</span>
                        </div>
                        <div class="run-param">
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/></svg>
                            <span>Run ID:</span>
                            <span class="run-param-val" id="curRunId">run_nepal_multitemporal_change_20261003</span>
                        </div>
                        <div class="run-param">
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"/></svg>
                            <span>Model:</span>
                            <span class="run-param-val" id="curModelName">Small U-Net</span>
                        </div>
                        <div class="run-param">
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="22" y1="12" x2="18" y2="12"/><line x1="6" y1="12" x2="2" y2="12"/><line x1="12" y1="6" x2="12" y2="2"/><line x1="12" y1="22" x2="12" y2="18"/></svg>
                            <span>Threshold:</span>
                            <span class="run-param-val" id="curThreshold">0.50</span>
                        </div>
                        <div class="status-completed" id="curStatusContainer">
                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                            <span id="curStatusText">Completed</span>
                        </div>
                        <span id="statusBadge" class="hidden">Nepal Baseline Active</span>
                    </div>
                    <div>
                        <button class="btn-cyan" onclick="openAnalysisModal()">
                            <span>Open analysis</span>
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                        </button>
                    </div>
                </div>

                <!-- 7. Bottom Scientific Disclaimer Banner -->
                <div class="disclaimer-banner">
                    <div class="disclaimer-left">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                        <span>Preliminary model candidates only &mdash; independent reference validation pending. Not confirmed flood extent or safety status.</span>
                    </div>
                    <div style="color: #64748b;">
                        &copy; Copernicus Sentinel Data (2024&ndash;2026) | &copy; Copernicus WorldDEM-30 | &copy; OpenStreetMap contributors
                    </div>
                </div>
            </div>

            <!-- VIEW 2: AI SEGMENTATION RUNNER -->
            <div id="view-segmentation" class="view-pane">
                <div class="subview-body">
                    <div>
                        <h2 class="view-header-title">
                            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--cyan)" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/></svg>
                            <span>Small U-Net AI Segmentation Control Center</span>
                        </h2>
                        <div class="view-header-sub">Run windowed FP16/FP32 inference on 3-channel SAR GeoTIFF images [VV_dB, VH_dB, VV_dB - VH_dB].</div>
                    </div>

                    <div class="form-grid">
                        <!-- Left Panel: Input Configuration -->
                        <div class="card-panel">
                            <h3>1. Input SAR GeoTIFF Selection</h3>

                            <div class="form-group">
                                <label class="form-label">Select Workspace Preset GeoTIFF:</label>
                                <select id="sarSelectPreset" class="form-select" onchange="handlePresetSelect()">
                                    <option value="">-- Scanning Workspace GeoTIFFs... --</option>
                                </select>
                            </div>

                            <div class="form-group">
                                <label class="form-label">Or Pick / Upload Local SAR GeoTIFF File:</label>
                                <input type="file" id="sarFileInput" accept=".tif,.tiff" style="display:none;" onchange="handleFileSelect(event)" />
                                <div class="file-dropzone" onclick="document.getElementById('sarFileInput').click()">
                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--cyan)" stroke-width="2" style="margin-bottom:4px;"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                                    <div style="font-size:12px; font-weight:600; color:#ffffff;">Click to Browse Local SAR GeoTIFF</div>
                                    <div style="font-size:10px; color:var(--text-secondary); margin-top:2px;">Supports calibrated Sentinel-1 3-band [VV, VH, VV-VH] or dual-pol .tif</div>
                                </div>
                                <div id="uploadStatus" style="font-size:11px; color:var(--cyan); margin-top:6px;"></div>
                            </div>

                            <div class="form-group">
                                <label class="form-label">Target SAR File Path:</label>
                                <input id="sarPath" class="form-input" value="data/processed/unet_inputs/post_calibrated_sigma0_unet_3band_corrected.tif" />
                            </div>

                            <div class="form-group">
                                <div class="flex-between">
                                    <label class="form-label">Decision Threshold (&tau;):</label>
                                    <span class="text-cyan" style="font-size:11px; font-weight:700;" id="valProbThresh">0.50</span>
                                </div>
                                <input type="range" id="probThreshRange" class="custom-range track-cyan" style="width:100%; margin-top:4px;" min="0.05" max="0.95" step="0.05" value="0.50" oninput="syncThresh(this.value, 'range')" />
                            </div>

                            <button id="btnRunInference" class="btn-primary" onclick="runSegmentation()">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                                <span>Run Small U-Net Inference</span>
                            </button>
                        </div>

                        <!-- Right Panel: Inference Results & Telemetry -->
                        <div class="card-panel">
                            <h3>2. Execution Telemetry & Results</h3>
                            <div id="segResults" style="font-size:12px; color:var(--text-secondary); line-height:1.5;">
                                <div style="padding: 16px; background: var(--bg-base); border-radius: 6px; border: 1px solid var(--border);">
                                    <div style="color:#ffffff; font-weight:600; margin-bottom:4px;">Nepal Baseline Run Active</div>
                                    <div>Pre-loaded baseline predictions are currently rendered in Map Explorer. Select a new SAR GeoTIFF to run customized windowed inference.</div>
                                </div>
                            </div>

                            <div style="margin-top:14px; padding:12px; background: rgba(0, 240, 255, 0.05); border: 1px solid rgba(0, 240, 255, 0.2); border-radius: 6px;">
                                <div style="font-size:11px; font-weight:700; color:var(--cyan); margin-bottom:4px;">Architecture & Normalization Standards:</div>
                                <ul style="font-size:10px; color:var(--text-secondary); margin-left:16px; line-height:1.4;">
                                    <li><b>Input Specification:</b> Band 1 = VV_dB, Band 2 = VH_dB, Band 3 = VV_dB - VH_dB</li>
                                    <li><b>Model Checkpoint:</b> <code>experiments/unet/best_model.pt</code> (1.93M parameters)</li>
                                    <li><b>Normalizations:</b> Mean = [-11.0, -18.0, 7.0], Std = [3.0, 3.5, 2.5]</li>
                                    <li><b>Nodata Handling:</b> Label 255 strictly excluded from computation and preserved in output GeoTIFF.</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- VIEW 3: OVERVIEW -->
            <div id="view-overview" class="view-pane">
                <div class="subview-body">
                    <h2 class="view-header-title">SARA System Overview & Event Telemetry</h2>
                    <div class="view-header-sub">Emergency Flood Inundation & Topological Accessibility Assessment Platform</div>

                    <div class="card-panel">
                        <h3>Active Scenario: Nepal Himalayan Monsoon Flood (August 2026)</h3>
                        <p style="font-size:12px; color:var(--text-secondary); line-height:1.6;">
                            The SARA platform integrates deep-learning microwave SAR water segmentation with multi-temporal change detection and Copernicus 30m topographic slope stratification. It isolates steep mountain radar shadow candidates from flat alluvial plain inundations, and assesses road network reachability to isolated settlements and healthcare facilities.
                        </p>
                    </div>

                    <div class="form-grid">
                        <div class="card-panel">
                            <h3>Area of Interest (AOI) Specifications</h3>
                            <ul style="font-size:11px; color:var(--text-secondary); margin-left:16px; line-height:1.6;">
                                <li><b>Corridor:</b> Bhote Koshi &ndash; Trishuli River Basin (Rasuwa, Nuwakot, Kathmandu Valley)</li>
                                <li><b>SAR Scene Dimension:</b> 32,072 &times; 21,519 pixels (~690 Million pixels)</li>
                                <li><b>Valid Earth Pixels:</b> 379,266,320 px (54.95% coverage)</li>
                                <li><b>Repeat Orbit:</b> Track 85 (Ascending) | Pre: Aug 16, 2026 &rarr; Post: Aug 28, 2026</li>
                            </ul>
                        </div>
                        <div class="card-panel">
                            <h3>Benchmark Model Performance</h3>
                            <ul style="font-size:11px; color:var(--text-secondary); margin-left:16px; line-height:1.6;">
                                <li><b>Small U-Net F1 (Dice):</b> 93.06% on Sen1Floods11 validation split</li>
                                <li><b>IoU (Jaccard):</b> 87.03% | Pixel Accuracy: 99.84%</li>
                                <li><b>Precision:</b> 91.61% | Recall: 94.57%</li>
                                <li><b>Inference Runtime:</b> Windowed 256&times;256 tiles (~0.22 GB VRAM)</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>

            <!-- VIEW 4: CHANGE DETECTION -->
            <div id="view-change-detection" class="view-pane">
                <div class="subview-body">
                    <h2 class="view-header-title">Multi-Temporal SAR Change Detection Engine</h2>
                    <div class="view-header-sub">Logarithmic Backscatter Decrease Analysis (&Delta;&sigma;&deg; &le; -3.0 dB)</div>
                    <div class="card-panel">
                        <h3>Physics-Based Microwave Backscatter Analysis</h3>
                        <p style="font-size:12px; color:var(--text-secondary); line-height:1.6;">
                            Calm floodwater causes specular reflection of radar energy away from the satellite, producing a sharp drop in observed backscatter compared to dry soil or vegetation. SARA compares co-registered pre-event (2026-08-16) and post-event (2026-08-28) Sentinel-1 VV/VH backscatter grids.
                        </p>
                    </div>
                    <div class="card-panel">
                        <h3>Nepal Corridor 5-Class Categorical Breakdown</h3>
                        <div style="font-size:11px; color:var(--text-secondary); line-height:1.8;">
                            <div>🟦 <b>Class 1: Persistent Water Candidate</b> &mdash; 3,514,642 px (0.93% of valid SAR)</div>
                            <div>🟨 <b>Class 2: Backscatter Change Only</b> &mdash; 54,198,201 px (14.29% of valid SAR)</div>
                            <div>🟩 <b>Class 3: Agreed Inundation Candidate</b> &mdash; 1,842,109 px (0.49% of valid SAR)</div>
                            <div>🟧 <b>Class 4: Suspected High-Slope Shadow Candidate</b> &mdash; 27,841,902 px (7.34% of valid SAR)</div>
                            <div>⬛ <b>Class 0: Dry Land / No Change</b> &mdash; 291,869,466 px (76.95% of valid SAR)</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- VIEW 5: TERRAIN ANALYSIS -->
            <div id="view-terrain-analysis" class="view-pane">
                <div class="subview-body">
                    <h2 class="view-header-title">Copernicus 30m Topographic Slope & Shadow Disambiguation</h2>
                    <div class="view-header-sub">DEM-Derived Slope Stratification (&le;5&deg;, &le;10&deg;, &le;15&deg;, >15&deg;) & Downstream D8 Flow Tracing</div>
                    <div class="form-grid">
                        <div class="card-panel">
                            <h3>Steep Slope Radar Shadow Disambiguation</h3>
                            <p style="font-size:12px; color:var(--text-secondary); line-height:1.6;">
                                Steep mountain back-slopes produce geometric radar shadow that mimics water backscatter. Topographic slope masking filters out false positives on mountain slopes (>15&deg;) while preserving flood candidates in low-lying river valleys.
                            </p>
                        </div>
                        <div class="card-panel">
                            <h3>Elevation-Based Flow Path Tracer</h3>
                            <div class="form-group">
                                <label class="form-label">Starting Coordinates (Lat, Lon):</label>
                                <input id="flowStartCoord" class="form-input" value="28.15, 85.25" />
                            </div>
                            <button class="btn-primary" onclick="runTerrainFlow()">
                                <span>Trace Downstream D8 Flow Path</span>
                            </button>
                            <div id="flowResults" style="font-size:11px; color:var(--text-secondary); margin-top:8px;"></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- VIEW 6: INFRASTRUCTURE IMPACT -->
            <div id="view-infrastructure-impact" class="view-pane">
                <div class="subview-body">
                    <h2 class="view-header-title">Pre-Event OpenStreetMap Infrastructure Impact Assessment</h2>
                    <div class="view-header-sub">Spatial Intersection Analysis of Candidate Inundation Zones with Transport and Settlement Assets</div>
                    <div class="form-grid">
                        <div class="card-panel">
                            <h3>1. Infrastructure Exposure Assessment</h3>
                            <p style="font-size:11.5px; color:var(--text-secondary); line-height:1.5; margin-bottom:12px;">
                                Calculates candidate spatial overlap across pre-event OSM roads, buildings, bridges, and healthcare facilities.
                            </p>
                            <button id="btnRunExposure" class="btn-primary" onclick="runInfrastructureExposure()">
                                <span>Run Exposure Assessment</span>
                            </button>
                            <div id="exposureResults" style="margin-top:12px; font-size:11.5px; color:var(--text-secondary);">
                                Click button to execute spatial overlap calculation against pre-event snapshot.
                            </div>
                        </div>
                        <div class="card-panel">
                            <h3>2. Scientific Damage Categorization</h3>
                            <ul style="font-size:11px; color:var(--text-secondary); margin-left:16px; line-height:1.6;">
                                <li><b>Candidate Spatial Overlap:</b> Feature geometry intersects preliminary model flood candidate mask.</li>
                                <li><b>Satellite-Observed Change:</b> Statistically significant backscatter drop (&Delta;&sigma;&deg; &le; -3 dB).</li>
                                <li><b>Independently Confirmed Damage:</b> Requires ground-truth validation (DHM, UN-SPIDER, or local verification).</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>

            <!-- VIEW 7: SETTLEMENT ACCESSIBILITY -->
            <div id="view-settlement-accessibility" class="view-pane">
                <div class="subview-body">
                    <h2 class="view-header-title">Topological Road Network Accessibility & Disconnected Settlements</h2>
                    <div class="view-header-sub">NetworkX Dijkstra Shortest-Path Hospital Reachability & Flood Severance Simulation</div>
                    <div class="form-grid">
                        <div class="card-panel">
                            <h3>1. Scenario Reachability Routing</h3>
                            <p style="font-size:11.5px; color:var(--text-secondary); line-height:1.5; margin-bottom:10px;">
                                Evaluates whether populated settlements retain road access to the nearest healthcare facility when flooded road segments are modeled as impassable.
                            </p>
                            <button id="btnRunAccessibility" class="btn-primary" onclick="runAccessibilityRouting()">
                                <span>Execute Network Accessibility Routing</span>
                            </button>
                            <div id="accessibilityResults" style="margin-top:12px; font-size:11.5px; color:var(--text-secondary);">
                                Click button to calculate baseline and scenario hospital reachability.
                            </div>
                        </div>
                        <div class="card-panel">
                            <h3>2. Modeling Safeguards & Assumptions</h3>
                            <p style="font-size:11px; color:var(--text-secondary); line-height:1.6;">
                                <b>Important:</b> Excluding a road edge models a <i>hypothetical scenario</i> based on preliminary SAR change; it does <b>NOT</b> prove physical destruction or that unmapped alternate access paths do not exist.
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- VIEW 8: VALIDATION & REPORTS -->
            <div id="view-validation-reports" class="view-pane">
                <div class="subview-body">
                    <h2 class="view-header-title">Independent Validation & Evidence-Grounded Situation Reports</h2>
                    <div class="view-header-sub">Scientific Safeguards, EMSR927 Data Isolation, Benchmark Metrics, and Downloadable Reports</div>
                    <div class="form-grid">
                        <div class="card-panel">
                            <h3>1. One-Page Situation Report Generator</h3>
                            <p style="font-size:11.5px; color:var(--text-secondary); line-height:1.5; margin-bottom:10px;">
                                Generates an evidence-grounded Markdown/JSON situation report compiled strictly from actual computed metrics and spatial denominators.
                            </p>
                            <button class="btn-primary" onclick="runGenerateReport()">
                                <span>Generate Official Situation Report</span>
                            </button>
                            <div id="reportPreviewContainer" style="margin-top:12px; font-size:11px; color:var(--text-secondary);"></div>
                        </div>
                        <div class="card-panel">
                            <h3>2. Benchmark Model Validation</h3>
                            <div style="font-size:11px; color:var(--text-secondary); line-height:1.6;">
                                <div><b>Small U-Net F1 (Dice):</b> 93.06%</div>
                                <div><b>IoU (Jaccard):</b> 87.03% | <b>Accuracy:</b> 99.84%</div>
                                <div><b>Precision:</b> 91.61% | <b>Recall:</b> 94.57%</div>
                                <div style="margin-top:8px; font-size:10px; color:#fcd34d;">
                                    <b>Exploratory SAR change-analysis candidates, not a verified flood boundary. Independent ground-truth validation is not yet available.</b>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

        </main>
    </div>

    <!-- Modal for Open Analysis -->
    <div id="analysisModal" class="modal-backdrop">
        <div class="modal-box">
            <div class="modal-head">
                <div class="modal-title">SARA Run Analysis Breakdown</div>
                <button class="modal-close" onclick="closeAnalysisModal()">&times;</button>
            </div>
            <div id="modalContent" style="font-size:11px; color:var(--text-secondary); line-height:1.6;">
                Loading analysis details...
            </div>
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        // Initialize Leaflet Map centered on Nepal
        const map = L.map('map', {zoomControl: true}).setView([28.15, 85.25], 9);
        
        let baseLayers = {
            esri: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', { 
                maxZoom: 18, 
                attribution: '&copy; ESRI &amp; Earthstar Geographics' 
            }),
            osm: L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { 
                maxZoom: 19, 
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" style="color:#38bdf8;">OpenStreetMap contributors</a>' 
            }),
            carto: L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
                maxZoom: 19,
                attribution: '&copy; CartoDB'
            })
        };

        // Default to ESRI Satellite for the authentic mountain satellite look
        baseLayers.esri.addTo(map);
        let activeBaseLayer = baseLayers.esri;

        let aoiLayer = null;
        let agreementOverlay = null;
        let waterBinOverlay = null;
        let waterProbOverlay = null;
        let slopeOverlay = null;

        let currentSarBounds = null;
        let currentSlopeBounds = null;
        let defaultNepalBounds = [[27.591277, 83.925836], [29.524362, 86.806913]];
        let currentRunId = "run_nepal_multitemporal_change_20261003";
        let evalRunId = "run_nepal_unet_eval_20261003";
        let availableRastersList = [];

        // View Navigation Switcher
        function switchNav(viewName) {
            document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
            const activeNav = document.querySelector(`.nav-item[data-view="${viewName}"]`);
            if (activeNav) activeNav.classList.add('active');

            document.querySelectorAll('.view-pane').forEach(el => el.classList.remove('active'));
            const activePane = document.getElementById(`view-${viewName}`);
            if (activePane) activePane.classList.add('active');

            if (viewName === 'map') {
                setTimeout(() => { map.invalidateSize(); }, 150);
            }
        }

        function toggleLayersPanel() {
            const panel = document.getElementById('layersPanel');
            panel.classList.toggle('collapsed');
            setTimeout(() => { map.invalidateSize(); }, 200);
        }

        function toggleBaseMap(enabled) {
            if (enabled) {
                if (!map.hasLayer(activeBaseLayer)) map.addLayer(activeBaseLayer);
            } else {
                if (map.hasLayer(activeBaseLayer)) map.removeLayer(activeBaseLayer);
            }
        }

        function changeBaseMap(key) {
            if (activeBaseLayer && map.hasLayer(activeBaseLayer)) {
                map.removeLayer(activeBaseLayer);
            }
            if (baseLayers[key]) {
                activeBaseLayer = baseLayers[key];
                if (document.getElementById('layerBaseMap').checked) {
                    activeBaseLayer.addTo(map);
                }
            }
        }

        async function initDashboard() {
            L.control.scale({position: 'bottomleft', imperial: false, maxWidth: 100}).addTo(map);

            // Add valley place labels
            const places = [
                { name: 'Rasuwa', lat: 28.18, lng: 85.20, isPark: false },
                { name: 'Langtang National Park', lat: 28.21, lng: 85.50, isPark: true },
                { name: 'Trisuli', lat: 27.98, lng: 85.23, isPark: false },
                { name: 'Nuwakot', lat: 27.91, lng: 85.16, isPark: false },
                { name: 'Kathmandu', lat: 27.72, lng: 85.32, isPark: false }
            ];

            places.forEach(p => {
                const iconHtml = p.isPark 
                    ? `<div style="display:flex; align-items:center; gap:4px; background:rgba(6,15,34,0.85); padding:2px 6px; border-radius:4px; border:1px solid #10b981; color:#ffffff; font-size:10px; font-weight:600; white-space:nowrap; box-shadow:0 2px 5px rgba(0,0,0,0.5);"><span style="color:#10b981;">⛰</span> ${p.name}</div>`
                    : `<div style="background:rgba(6,15,34,0.85); padding:2px 6px; border-radius:4px; border:1px solid #142850; color:#e2e8f0; font-size:10px; font-weight:600; white-space:nowrap; box-shadow:0 2px 5px rgba(0,0,0,0.5);">${p.name}</div>`;
                
                L.marker([p.lat, p.lng], {
                    icon: L.divIcon({
                        html: iconHtml,
                        className: 'place-marker-wrap',
                        iconSize: [80, 20],
                        iconAnchor: [40, 10]
                    }),
                    interactive: false
                }).addTo(map);
            });

            await loadAvailableRasters();
            try {
                const res = await fetch('/dashboard/nepal-summary');
                if(!res.ok) throw new Error("Failed to fetch Nepal summary");
                const data = await res.json();

                currentSarBounds = data.sar_wgs84_bounds;
                defaultNepalBounds = data.sar_wgs84_bounds;
                currentSlopeBounds = data.slope_wgs84_bounds;
                currentRunId = data.run_id;
                evalRunId = data.eval_run_id;

                map.setView([28.08, 85.30], 10);

                const corridorPolygon = [
                    [28.25, 85.06],
                    [28.32, 85.20],
                    [28.30, 85.45],
                    [28.24, 85.64],
                    [28.12, 85.65],
                    [27.96, 85.52],
                    [27.86, 85.32],
                    [27.89, 85.12],
                    [28.04, 85.02],
                    [28.18, 85.03]
                ];

                if(aoiLayer) map.removeLayer(aoiLayer);
                aoiLayer = L.polygon(corridorPolygon, {color: "#00f0ff", weight: 1.8, dashArray: "5, 5", fill: false}).addTo(map);

                updateOverlays();
            } catch(e) {
                console.warn("Init fallback:", e);
                currentSarBounds = defaultNepalBounds;
                currentSlopeBounds = [[27.000139, 82.999861], [30.000139, 86.999861]];
                map.setView([28.08, 85.30], 10);
                updateOverlays();
            }
        }

        async function loadAvailableRasters() {
            try {
                const res = await fetch('/segmentation/available-rasters');
                if(!res.ok) return;
                const data = await res.json();
                availableRastersList = data.rasters || [];
                
                const select = document.getElementById('sarSelectPreset');
                if (!select) return;
                select.innerHTML = '<option value="">-- Choose Workspace SAR GeoTIFF --</option>';
                
                availableRastersList.forEach(r => {
                    const opt = document.createElement('option');
                    opt.value = r.path;
                    const readyTag = r.is_ready_for_unet ? ' [3-band Ready]' : ` [${r.num_bands}-band]`;
                    opt.textContent = `${r.filename} (${r.size_mb} MB)${readyTag}`;
                    select.appendChild(opt);
                });

                const def = availableRastersList.find(r => r.filename.includes('corrected') || r.is_ready_for_unet);
                if(def) {
                    select.value = def.path;
                    document.getElementById('sarPath').value = def.path;
                }
            } catch(e) {
                console.warn("Could not load available rasters:", e);
            }
        }

        function handlePresetSelect() {
            const select = document.getElementById('sarSelectPreset');
            if(select.value) {
                document.getElementById('sarPath').value = select.value;
                document.getElementById('uploadStatus').innerText = `Selected preset: ${select.options[select.selectedIndex].text}`;
            }
        }

        async function handleFileSelect(event) {
            const file = event.target.files[0];
            if(!file) return;

            const statusDiv = document.getElementById('uploadStatus');
            statusDiv.innerHTML = `Uploading & validating <b>${file.name}</b> (${(file.size/(1024*1024)).toFixed(2)} MB)...`;

            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch('/segmentation/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                if(!res.ok) {
                    statusDiv.innerHTML = `<span style="color:#ef4444;">Upload error: ${data.detail || 'Failed'}</span>`;
                    return;
                }

                const up = data.uploaded_raster;
                document.getElementById('sarPath').value = up.path;
                statusDiv.innerHTML = `<span style="color:#10b981;">✔ Uploaded: ${up.filename} (${up.width}&times;${up.height}, ${up.bands} bands, ${up.crs})</span>`;
                
                await loadAvailableRasters();
                document.getElementById('sarSelectPreset').value = up.path;
            } catch(e) {
                statusDiv.innerHTML = `<span style="color:#ef4444;">Upload failed: ${e}</span>`;
            }
        }

        function syncThresh(val, source) {
            const num = parseFloat(val).toFixed(2);
            document.getElementById('probThreshRange').value = num;
            document.getElementById('valProbThresh').innerText = num;
            document.getElementById('curThreshold').innerText = num;
        }

        function resetToNepalView() {
            if(defaultNepalBounds) map.fitBounds(defaultNepalBounds);
        }

        function updateOverlays() {
            if(!currentSarBounds) return;

            // 1. SAR Change Candidates / Multi-Temporal Agreement Layer
            const showAgr = document.getElementById('layerAgreement').checked;
            const opAgr = parseFloat(document.getElementById('opacityAgreement').value) / 100.0;
            document.getElementById('valAgreement').innerText = `${Math.round(opAgr * 100)}%`;

            if(agreementOverlay) { map.removeLayer(agreementOverlay); agreementOverlay = null; }
            if(showAgr && currentRunId) {
                const urlAgr = `/dashboard/layers/${currentRunId}/agreement_overlay.png`;
                agreementOverlay = L.imageOverlay(urlAgr, defaultNepalBounds, {opacity: opAgr}).addTo(map);
            }

            // 2. Active Model Water Mask (Binary)
            const showBin = document.getElementById('layerWaterBin').checked;
            const opBin = parseFloat(document.getElementById('opacityWaterBin').value) / 100.0;
            document.getElementById('valWaterBin').innerText = `${Math.round(opBin * 100)}%`;

            if(waterBinOverlay) { map.removeLayer(waterBinOverlay); waterBinOverlay = null; }
            if(showBin && evalRunId) {
                const urlBin = `/dashboard/layers/${evalRunId}/water_prediction_overlay.png`;
                waterBinOverlay = L.imageOverlay(urlBin, currentSarBounds, {opacity: opBin}).addTo(map);
            }

            // 3. Model Water Probability Heatmap
            const showProb = document.getElementById('layerWaterProb').checked;
            const opProb = parseFloat(document.getElementById('opacityWaterProb').value) / 100.0;
            document.getElementById('valWaterProb').innerText = `${Math.round(opProb * 100)}%`;

            if(waterProbOverlay) { map.removeLayer(waterProbOverlay); waterProbOverlay = null; }
            if(showProb && evalRunId) {
                const urlProb = `/dashboard/layers/${evalRunId}/water_probability_overlay.png`;
                waterProbOverlay = L.imageOverlay(urlProb, currentSarBounds, {opacity: opProb}).addTo(map);
            }

            // 4. Terrain Slope (Copernicus 30m)
            const showSlope = document.getElementById('layerSlope').checked;
            const opSlope = parseFloat(document.getElementById('opacitySlope').value) / 100.0;
            document.getElementById('valSlope').innerText = `${Math.round(opSlope * 100)}%`;

            if(slopeOverlay) { map.removeLayer(slopeOverlay); slopeOverlay = null; }
            if(showSlope) {
                const bounds = currentSlopeBounds || defaultNepalBounds;
                const urlSlope = `/dashboard/layers/${currentRunId}/slope_overlay.png`;
                slopeOverlay = L.imageOverlay(urlSlope, bounds, {opacity: opSlope}).addTo(map);
            }
        }

        function toggleAOI() {
            const showAOI = document.getElementById('layerAOI').checked;
            const opAOI = parseFloat(document.getElementById('opacityAOI').value) / 100.0;
            document.getElementById('valAOI').innerText = `${Math.round(opAOI * 100)}%`;

            if(aoiLayer) {
                if(showAOI) {
                    if (!map.hasLayer(aoiLayer)) map.addLayer(aoiLayer);
                    aoiLayer.setStyle({opacity: opAOI});
                } else {
                    if (map.hasLayer(aoiLayer)) map.removeLayer(aoiLayer);
                }
            }
        }

        async function runSegmentation() {
            const sarPath = document.getElementById('sarPath').value.trim();
            const threshold = parseFloat(document.getElementById('probThreshRange').value);
            const btn = document.getElementById('btnRunInference');
            const segDiv = document.getElementById('segResults');

            if(!sarPath) {
                segDiv.innerHTML = '<span style="color:#ef4444;">Please select or enter a valid SAR GeoTIFF path.</span>';
                return;
            }

            btn.disabled = true;
            segDiv.innerHTML = `<span style="color:var(--cyan);">⏳ Running Small U-Net FP16 windowed inference (&tau;=${threshold.toFixed(2)})...</span>`;

            try {
                const res = await fetch('/segmentation/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({sar_raster_path: sarPath, probability_threshold: threshold, auto_prepare_2band: true})
                });
                const data = await res.json();

                if(!res.ok) {
                    segDiv.innerHTML = `<span style="color:#ef4444;"><b>Inference Error:</b> ${data.detail || 'Inference failed'}</span>`;
                    btn.disabled = false;
                    return;
                }

                evalRunId = data.run_id;
                currentSarBounds = data.wgs84_bounds;

                // Update bottom current run bar
                document.getElementById('curRunId').innerText = data.run_id;
                document.getElementById('curModelName').innerText = "Small U-Net";
                document.getElementById('curThreshold').innerText = threshold.toFixed(2);
                document.getElementById('curStatusText').innerText = "Completed";
                document.getElementById('labelWaterBin').innerText = `Run ${data.run_id} Water Mask`;
                
                // Update KPI Cards for custom run
                document.getElementById('kpiWaterPred').innerText = `${data.summary.water_percentage}%`;
                document.getElementById('kpiWaterPredSub').innerText = `of ${data.summary.valid_pixels.toLocaleString()} valid pixels`;
                const ringWater = document.getElementById('ringWaterPred');
                if (ringWater) ringWater.setAttribute('stroke-dasharray', `${Math.min(100, data.summary.water_percentage)}, 100`);

                document.getElementById('kpiSarCoverage').innerText = 'Custom';
                document.getElementById('kpiSarCoverageSub').innerText = `${data.summary.valid_pixels.toLocaleString()} valid px`;

                // Update results pane
                segDiv.innerHTML = `
                    <div style="padding:14px; background:var(--bg-base); border-radius:6px; border:1px solid var(--cyan);">
                        <div style="color:var(--cyan); font-weight:700; margin-bottom:6px;">✔ Run ${data.run_id} Completed</div>
                        <div><b>Input File:</b> ${data.input_filename}</div>
                        <div><b>Valid Earth Pixels:</b> ${data.summary.valid_pixels.toLocaleString()} px</div>
                        <div><b>Predicted Water Pixels:</b> <span style="color:var(--cyan); font-weight:700;">${data.summary.predicted_water_pixels.toLocaleString()} px (${data.summary.water_percentage}%)</span></div>
                        <div><b>Mean Probability:</b> ${(data.summary.mean_water_probability * 100).toFixed(1)}%</div>
                        <div style="font-size:10px; color:var(--text-muted); margin-top:8px;">${data.disclaimer}</div>
                        <button class="btn-cyan" style="margin-top:10px;" onclick="switchNav('map')">View Predictions in Map Explorer &rarr;</button>
                    </div>
                `;

                // Automatically enable water mask layer
                document.getElementById('layerWaterBin').checked = true;

                // Center map on new raster
                map.fitBounds(currentSarBounds);

                // Update AOI rectangle
                if(aoiLayer) map.removeLayer(aoiLayer);
                aoiLayer = L.rectangle(currentSarBounds, {color: "#00f0ff", weight: 2, dashArray: "6, 6", fill: false}).addTo(map);

                updateOverlays();

            } catch(e) {
                segDiv.innerHTML = `<span style="color:#ef4444;">Inference request failed: ${e}</span>`;
            } finally {
                btn.disabled = false;
            }
        }

        async function runInfrastructureExposure() {
            const btn = document.getElementById('btnRunExposure');
            const resDiv = document.getElementById('exposureResults');
            btn.disabled = true;
            resDiv.innerHTML = '<span style="color:var(--cyan);">⏳ Running spatial overlap exposure assessment on pre-event OSM data...</span>';

            try {
                const res = await fetch('/infrastructure/exposure', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        bbox: [83.925, 27.591, 86.806, 29.524],
                        change_mask_path: 'data/output/run_nepal_multitemporal_change_20261003/nepal_water_and_change_agreement_mask.tif'
                    })
                });
                const data = await res.json();
                if(!res.ok) {
                    resDiv.innerHTML = `<span style="color:#ef4444;">Exposure error: ${data.detail || 'Failed'}</span>`;
                    return;
                }

                const s = data.exposure_summary;
                resDiv.innerHTML = `
                    <div style="padding:10px; background:var(--bg-base); border-radius:4px; border:1px solid var(--border);">
                        <div style="color:var(--cyan); font-weight:700; margin-bottom:4px;">✔ Exposure Run: ${data.run_id}</div>
                        <div><b>Candidate Exposed Roads:</b> ${s.exposed_road_length_km.toFixed(2)} km (${s.exposed_road_percentage.toFixed(1)}% of network)</div>
                        <div><b>Candidate Impacted Buildings:</b> ${s.exposed_building_count} buildings</div>
                        <div><b>Candidate Impacted Healthcare:</b> ${s.exposed_healthcare_facility_count} facilities</div>
                        <div style="font-size:10px; color:var(--text-muted); margin-top:6px;">${s.disclaimer}</div>
                    </div>
                `;
            } catch(e) {
                resDiv.innerHTML = `<span style="color:#ef4444;">Exposure request failed: ${e}</span>`;
            } finally {
                btn.disabled = false;
            }
        }

        async function runAccessibilityRouting() {
            const btn = document.getElementById('btnRunAccessibility');
            const resDiv = document.getElementById('accessibilityResults');
            btn.disabled = true;
            resDiv.innerHTML = '<span style="color:var(--cyan);">⏳ Building Dijkstra road network graph and calculating reachability...</span>';

            try {
                const res = await fetch('/infrastructure/accessibility', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        bbox: [83.925, 27.591, 86.806, 29.524],
                        change_mask_path: 'data/output/run_nepal_multitemporal_change_20261003/nepal_water_and_change_agreement_mask.tif',
                        simulate_flood_closures: true
                    })
                });
                const data = await res.json();
                if(!res.ok) {
                    resDiv.innerHTML = `<span style="color:#ef4444;">Accessibility error: ${data.detail || 'Failed'}</span>`;
                    return;
                }

                const a = data.accessibility_summary;
                resDiv.innerHTML = `
                    <div style="padding:10px; background:var(--bg-base); border-radius:4px; border:1px solid var(--border);">
                        <div style="color:var(--cyan); font-weight:700; margin-bottom:4px;">✔ Accessibility Run: ${data.run_id}</div>
                        <div><b>Total Evaluated Settlements:</b> ${a.total_settlements}</div>
                        <div><b>Baseline Connected Settlements:</b> <span style="color:#10b981; font-weight:700;">${a.baseline_connected_count}</span></div>
                        <div><b>Hypothetically Isolated Settlements:</b> <span style="color:#f59e0b; font-weight:700;">${a.scenario_isolated_count || 0}</span></div>
                        <div style="font-size:10px; color:var(--text-muted); margin-top:6px;">${a.disclaimer}</div>
                    </div>
                `;
            } catch(e) {
                resDiv.innerHTML = `<span style="color:#ef4444;">Accessibility request failed: ${e}</span>`;
            } finally {
                btn.disabled = false;
            }
        }

        async function runTerrainFlow() {
            const coordStr = document.getElementById('flowStartCoord').value.trim();
            const resDiv = document.getElementById('flowResults');
            const parts = coordStr.split(',').map(s => parseFloat(s.trim()));
            if(parts.length < 2 || isNaN(parts[0]) || isNaN(parts[1])) {
                resDiv.innerHTML = '<span style="color:#ef4444;">Enter valid Lat, Lon coordinates.</span>';
                return;
            }

            resDiv.innerHTML = '<span style="color:var(--cyan);">⏳ Tracing D8 steepest-descent elevation flow path...</span>';
            try {
                const res = await fetch('/terrain/trace-flow', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        dem_path: 'data/processed/nepal_copernicus_slope_30m.tif',
                        start_points_wgs84: [[parts[1], parts[0]]]
                    })
                });
                const data = await res.json();
                if(!res.ok) {
                    resDiv.innerHTML = `<span style="color:#ef4444;">Flow tracing error: ${data.detail || 'Failed'}</span>`;
                    return;
                }
                const fr = data.flow_tracing_result;
                resDiv.innerHTML = `
                    <div style="color:#10b981; font-weight:600;">✔ Trace Completed (${fr.paths ? fr.paths.length : 0} path points generated)</div>
                    <div style="font-size:10px; color:var(--text-muted);">${fr.caveats || 'Non-hydrodynamic steepest-descent path.'}</div>
                `;
            } catch(e) {
                resDiv.innerHTML = `<span style="color:#ef4444;">Flow trace failed: ${e}</span>`;
            }
        }

        async function runGenerateReport() {
            const container = document.getElementById('reportPreviewContainer');
            container.innerHTML = '<span style="color:var(--cyan);">⏳ Generating one-page situation report from actual analysis metrics...</span>';

            try {
                const res = await fetch('/reports/situation', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        run_id: currentRunId || 'run_nepal_multitemporal_change_20261003',
                        aoi_name: 'Bhote Koshi - Trishuli Corridor, Nepal',
                        event_date: '2026-08-26'
                    })
                });
                const data = await res.json();
                if(!res.ok) {
                    container.innerHTML = `<span style="color:#ef4444;">Report error: ${data.detail || 'Failed'}</span>`;
                    return;
                }

                container.innerHTML = `
                    <div style="padding:10px; background:var(--bg-base); border-radius:4px; border:1px solid var(--cyan); margin-top:8px;">
                        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
                            <b style="color:var(--cyan);">✔ ${data.report_filename}</b>
                            <button class="btn-cyan" onclick="downloadReportContent('${encodeURIComponent(data.report_content)}', '${data.report_filename}')">Download Report (.md)</button>
                        </div>
                        <pre style="max-height:220px; overflow-y:auto; font-size:10px; color:#cbd5e1; white-space:pre-wrap; background:#071228; padding:8px; border-radius:4px;">${data.report_content}</pre>
                    </div>
                `;
            } catch(e) {
                container.innerHTML = `<span style="color:#ef4444;">Report generation failed: ${e}</span>`;
            }
        }

        function downloadReportContent(encodedContent, filename) {
            const content = decodeURIComponent(encodedContent);
            const blob = new Blob([content], {type: 'text/markdown'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }

        function openAnalysisModal() {
            const modal = document.getElementById('analysisModal');
            const content = document.getElementById('modalContent');
            modal.classList.add('open');
            content.innerHTML = `
                <div style="margin-bottom:12px;">
                    <b>Active Run ID:</b> <code>${currentRunId}</code><br>
                    <b>Evaluation Run ID:</b> <code>${evalRunId}</code><br>
                    <b>Decision Threshold:</b> <code>${document.getElementById('curThreshold').innerText}</code>
                </div>
                <div style="margin-bottom:12px;">
                    <b style="color:#ffffff;">Available Raster Layers for this Run:</b>
                    <ul style="margin-left:18px; margin-top:4px;">
                        <li><code>/dashboard/layers/${currentRunId}/agreement_overlay.png</code></li>
                        <li><code>/dashboard/layers/${evalRunId}/water_prediction_overlay.png</code></li>
                        <li><code>/dashboard/layers/${evalRunId}/water_probability_overlay.png</code></li>
                        <li><code>/dashboard/layers/${currentRunId}/slope_overlay.png</code></li>
                    </ul>
                </div>
                <div style="margin-bottom:12px; font-size:10px; color:#fcd34d;">
                    <b>Scientific Notice:</b> Exploratory SAR change-analysis candidates, not a verified flood boundary. Independent ground-truth validation is not yet available.
                </div>
                <div style="display:flex; justify-content:flex-end;">
                    <button class="btn-cyan" onclick="closeAnalysisModal()">Close</button>
                </div>
            `;
        }

        function closeAnalysisModal() {
            document.getElementById('analysisModal').classList.remove('open');
        }

        window.onload = initDashboard;
    </script>
</body>
</html>"""
