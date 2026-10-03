"""End-to-End Nepal SAR Validation and Debugging Test Suite for Small U-Net."""

import os
import pytest
import rasterio
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.unet_segmentation_service import UNetSegmentationService, SegmentationRequest


@pytest.fixture
def test_client():
    return TestClient(app)


def test_nepal_raster_compatibility_rejection():
    """Verify that 2-band linear power scale Nepal rasters are rejected with clear error."""
    nepal_raster_path = "data/processed/post_calibrated_sigma0.tif"
    if not os.path.exists(nepal_raster_path):
        pytest.skip(f"Nepal raster not found at: {nepal_raster_path}")

    service = UNetSegmentationService()
    with pytest.raises(ValueError, match="Invalid band count: Small U-Net requires 3 bands"):
        service.validate_input_raster(os.path.abspath(nepal_raster_path))


def test_nepal_api_incompatible_raster_endpoint(test_client):
    """Verify API endpoint returns 400 Bad Request for 2-band Nepal post_calibrated_sigma0.tif."""
    nepal_raster_path = "data/processed/post_calibrated_sigma0.tif"
    if not os.path.exists(nepal_raster_path):
        pytest.skip(f"Nepal raster not found at: {nepal_raster_path}")

    payload = {"sar_raster_path": nepal_raster_path}
    response = test_client.post("/segmentation/predict", json=payload)
    assert response.status_code == 400
    assert "Invalid band count: Small U-Net requires 3 bands" in response.json()["detail"]


def test_end_to_end_valid_3band_raster_pipeline(test_client):
    """Verify end-to-end inference, GeoTIFF output generation, PNG overlays, and bounds for valid 3-band raster."""
    sar_path = "data/ml/sen1floods11/images/Bolivia_001_sar.tif"
    if not os.path.exists(sar_path):
        pytest.skip(f"3-band SAR raster not found at: {sar_path}")

    payload = {"sar_raster_path": sar_path, "probability_threshold": 0.50}
    response = test_client.post("/segmentation/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ok"
    assert "run_id" in data
    assert len(data["wgs84_bounds"]) == 2
    assert data["summary"]["valid_pixels"] > 0
    assert 0.0 <= data["summary"]["mean_water_probability"] <= 1.0

    # Layer serving check
    run_id = data["run_id"]
    layer_res = test_client.get(f"/dashboard/layers/{run_id}/water_prediction_overlay.png")
    assert layer_res.status_code == 200
    assert layer_res.headers["content-type"] == "image/png"
