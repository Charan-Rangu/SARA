import sys
import os
import pytest
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.main import app
from tests.test_dem_service import create_synthetic_dem

client = TestClient(app)

def test_terrain_analyze_endpoint(tmp_path):
    dem_path = str(tmp_path / "dem_api_test.tif")
    data = np.linspace(10, 100, 100).reshape((10, 10)).astype(np.float32)
    create_synthetic_dem(dem_path, data)

    payload = {"dem_path": dem_path}
    response = client.post("/terrain/analyze", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "ok"
    assert "terrain_analysis" in res
    assert res["terrain_analysis"]["min_elevation_m"] == 10.0

def test_terrain_trace_flow_endpoint(tmp_path):
    dem_path = str(tmp_path / "dem_flow_api.tif")
    data = np.linspace(100, 10, 100).reshape((10, 10)).astype(np.float32)
    create_synthetic_dem(dem_path, data)

    payload = {
        "dem_path": dem_path,
        "start_points_wgs84": [[76.15, 10.15]]
    }
    response = client.post("/terrain/trace-flow", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "ok"
    assert "flow_tracing_result" in res

def test_dashboard_ui_endpoint():
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "SARA — Flood Mapping & Accessibility Dashboard" in response.text

def test_situation_report_api_endpoint():
    payload = {
        "run_id": "api_test_99",
        "aoi_name": "API Test AOI",
        "event_date": "2024-08-15"
    }
    response = client.post("/reports/situation", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "ok"
    assert "sara_situation_report_api_test_99.md" in res["report_filename"]
