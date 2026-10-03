import sys
import os
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.main import app

client = TestClient(app)

from unittest.mock import patch
import geopandas as gpd
import shapely.geometry
from geospatial.infrastructure import InfrastructureContainer
from backend.services.osm_infrastructure import OSMSourceMetadata

def create_mock_container():
    pt_settlement = shapely.geometry.Point(76.15, 9.85)
    pt_hospital = shapely.geometry.Point(76.35, 10.05)
    line_r1 = shapely.geometry.LineString([[76.15, 9.85], [76.25, 9.95]])
    line_r2 = shapely.geometry.LineString([[76.25, 9.95], [76.35, 10.05]])
    roads = gpd.GeoDataFrame([{"id": "R1", "name": "Road 1", "geometry": line_r1}, {"id": "R2", "name": "Road 2", "geometry": line_r2}], crs="EPSG:4326")
    settlements = gpd.GeoDataFrame([{"id": "S1", "name": "Village A", "geometry": pt_settlement}], crs="EPSG:4326")
    hospitals = gpd.GeoDataFrame([{"id": "H1", "name": "City Hospital", "geometry": pt_hospital}], crs="EPSG:4326")
    meta = OSMSourceMetadata(
        source_type="SYNTHETIC_TEST",
        is_pre_event_snapshot=True,
        snapshot_date="2026-08-01",
        retrieval_timestamp="2026-08-01T00:00:00Z",
        aoi_name="Synthetic Test AOI"
    )
    return InfrastructureContainer(roads=roads, settlements=settlements, facilities=hospitals, crs="EPSG:4326"), meta

def test_infrastructure_health_endpoint():
    response = client.get("/infrastructure/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert data.get("module") == "infrastructure_exposure_and_accessibility"

@patch("backend.main.osm_service.fetch_overpass_infrastructure")
def test_infrastructure_exposure_endpoint_bbox_only(mock_fetch):
    mock_fetch.return_value = create_mock_container()
    payload = {
        "bbox": [76.15, 9.85, 76.45, 10.15]
    }
    response = client.post("/infrastructure/exposure", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert "run_id" in data
    assert "exposure_summary" in data
    assert "osm_source_metadata" in data

@patch("backend.main.osm_service.fetch_overpass_infrastructure")
def test_infrastructure_accessibility_endpoint(mock_fetch):
    mock_fetch.return_value = create_mock_container()
    payload = {
        "bbox": [76.15, 9.85, 76.45, 10.15],
        "simulate_flood_closures": False
    }
    response = client.post("/infrastructure/accessibility", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert "run_id" in data
    assert "accessibility_summary" in data
    assert "total_settlements" in data["accessibility_summary"]
    assert "baseline_connected_count" in data["accessibility_summary"]
    assert "osm_source_metadata" in data

