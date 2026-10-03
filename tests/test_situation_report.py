import pytest
from backend.services.situation_report import (
    SituationReportGenerator,
    ReportRequest,
    LIMITATION_STATEMENT
)

def test_situation_report_generation():
    generator = SituationReportGenerator()
    req = ReportRequest(
        run_id="test_run_123",
        aoi_name="Ernakulam / Kochi Region, Kerala",
        event_date="2024-08-15",
        satellite_search_summary={"total_scenes_found": 3, "pre_scenes_count": 2, "post_scenes_count": 1, "suitable_pairs_count": 2},
        exposure_summary={"total_roads": 10, "roads_potential_overlap_count": 2, "total_buildings": 50, "buildings_potential_overlap_count": 5},
        accessibility_summary={"total_settlements": 5, "baseline_connected_count": 4, "baseline_unconnected_count": 1},
        terrain_summary={"min_elevation_m": 2.0, "max_elevation_m": 120.0, "mean_elevation_m": 35.0, "mean_slope_deg": 4.5},
        analyst_notes="Preliminary report generated during Phase 5 verification."
    )

    report_md = generator.generate_report(req)

    assert "SARA" in report_md and "Situation Report" in report_md and "test_run_123" in report_md
    assert "Ernakulam / Kochi Region, Kerala" in report_md
    assert LIMITATION_STATEMENT in report_md
    assert "Contains modified Copernicus Sentinel data 2026." in report_md
    assert "Produced using Copernicus WorldDEM-30" in report_md
    assert "OpenStreetMap contributors." in report_md
    assert "Total Satellite Scenes Retrieved:" in report_md and "`3`" in report_md
