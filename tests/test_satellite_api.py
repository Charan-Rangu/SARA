import sys
import os
from unittest.mock import patch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.main import app
from tests.test_catalog import MOCK_STAC_RESPONSE

client = TestClient(app)

def test_satellite_search_api_success():
    payload = {
        "bbox": [76.15, 9.85, 76.45, 10.15],
        "event_date": "2026-08-15",
        "pre_event_days": 14,
        "post_event_days": 14,
        "max_results": 20,
        "storage_budget_gb": 10.0
    }

    # Patch catalog_service.search_scenes to use mock response offline
    with patch("backend.main.catalog_service.search_scenes") as mock_search:
        from backend.services.sentinel1_catalog import Sentinel1CatalogService
        service = Sentinel1CatalogService()
        mock_result = service.search_scenes(
            aoi=None,
            event_date_str="2026-08-15",
            mock_response=MOCK_STAC_RESPONSE
        )
        mock_search.return_value = mock_result

        response = client.post("/satellite/search", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data.get("status") == "ok"
        assert "search_result" in data
        assert "compatibility_evaluation" in data
        assert "download_plan" in data
        assert data["compatibility_evaluation"]["status"] == "SUITABLE_PAIRS_FOUND"

def test_satellite_search_api_invalid_aoi():
    payload = {
        "bbox": [190.0, 9.85, 76.45, 10.15],  # Invalid longitude > 180
        "event_date": "2026-08-15"
    }
    response = client.post("/satellite/search", json=payload)
    assert response.status_code == 400
    assert "Invalid AOI parameter" in response.json()["detail"]
