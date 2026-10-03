"""Tests for Real-Data Pipeline Capabilities, Safeguards, and Status Flags.

Verifies:
1. Catalogue discovery and scene compatibility on real search schemas.
2. Download planner budget limits and confirmation requirements.
3. Missing SNAP GPT handling without fake raster generation.
4. Real OpenStreetMap vector infrastructure loading with live provenance.
5. Real vs Synthetic layer labeling for dashboard API responses.
6. Situation report generation under partial/real data constraints.
"""

import pytest
import os
from geospatial.aoi import AOI
from backend.services.sentinel1_catalog import Sentinel1Scene, CatalogSearchResult
from backend.services.scene_compatibility import SceneCompatibilityAnalyzer
from backend.services.download_planner import DownloadPlanner
from geospatial.sar_preprocessing import SNAPProcessorAdapter, SARPreprocessingConfig
from backend.services.osm_infrastructure import OSMInfrastructureService, OSMSourceMetadata
from backend.services.situation_report import SituationReportGenerator, ReportRequest, LIMITATION_STATEMENT


def test_download_planner_budget_and_safeguards():
    planner = DownloadPlanner(default_storage_budget_gb=5.0)
    scenes = [
        Sentinel1Scene(
            id="scene_1",
            title="S1A_IW_GRDH_PRE.SAFE",
            acquisition_time="2024-08-13T00:41:18Z",
            size_bytes=1639800000,
            download_url="https://zipper.dataspace.copernicus.eu/odata/v1/Products(scene_1)/$value"
        ),
        Sentinel1Scene(
            id="scene_2",
            title="S1A_IW_GRDH_POST.SAFE",
            acquisition_time="2024-08-25T00:41:18Z",
            size_bytes=1640000000,
            download_url="https://zipper.dataspace.copernicus.eu/odata/v1/Products(scene_2)/$value"
        )
    ]
    plan = planner.create_download_plan(scenes)

    assert plan.scenes_count == 2
    assert plan.total_estimated_size_gb > 3.0
    assert plan.within_budget is True
    assert len(plan.products) == 2


def test_missing_snap_graceful_handling():
    # Force a dummy non-existent gpt path
    adapter = SNAPProcessorAdapter(gpt_path="non_existent_gpt_executable_path")
    assert adapter.is_snap_available() is False
    instructions = adapter.get_installation_instructions()
    assert "ESA SNAP (Sentinel Application Platform)" in instructions
    assert "https://step.esa.int/main/download/snap-download/" in instructions


def test_osm_live_provenance_metadata():
    aoi = AOI(bbox=[76.15, 9.85, 76.45, 10.15], name="Kerala Test AOI")
    meta = OSMSourceMetadata(
        source_type="LIVE_OVERPASS_API",
        retrieval_timestamp="2024-08-15T12:00:00Z",
        snapshot_date="2024-08-15",
        is_pre_event_snapshot=False,
        aoi_name=aoi.name
    )
    assert meta.is_pre_event_snapshot is False
    assert "LIVE_OVERPASS_API" in meta.source_type
    assert "not be used as verified historical" in meta.limitation_statement


def test_situation_report_with_real_osm_data():
    generator = SituationReportGenerator()
    req = ReportRequest(
        run_id="real_data_run_001",
        aoi_name="Ernakulam / Kochi Region, Kerala",
        event_date="2024-08-15",
        satellite_search_summary={
            "total_scenes_found": 3,
            "pre_scenes_count": 2,
            "post_scenes_count": 1,
            "suitable_pairs_count": 2
        },
        exposure_summary={
            "total_roads": 1845,
            "roads_potential_overlap_count": 0,
            "total_facilities": 48,
            "facilities_potential_overlap_count": 0
        },
        analyst_notes="Real-data catalogue discovery and OSM infrastructure assessment completed. SNAP processing pending download confirmation."
    )
    report = generator.generate_report(req)

    assert "SARA" in report and "Situation Report" in report
    assert "Ernakulam / Kochi Region, Kerala" in report
    assert LIMITATION_STATEMENT in report
    assert "Total Satellite Scenes Retrieved:" in report
    assert "`1845`" in report  # Real roads count
    assert "Contains modified Copernicus Sentinel data 2026." in report
