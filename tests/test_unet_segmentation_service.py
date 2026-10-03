"""Comprehensive unit and integration tests for Small U-Net Segmentation Service & API Endpoint."""

import os
import json
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.unet_segmentation_service import (
    UNetSegmentationService,
    SegmentationRequest,
    validate_secure_path
)


@pytest.fixture
def test_client():
    return TestClient(app)


@pytest.fixture
def synthetic_sar_raster(tmp_path):
    """Fixture generating a synthetic 3-band SAR raster (256x256) with CRS EPSG:4326."""
    raster_path = str(tmp_path / "synthetic_sar_chip.tif")
    width = 256
    height = 256
    crs = "EPSG:4326"
    transform = from_origin(45.0, 2.0, 0.001, 0.001)

    # Band 1: VV_dB (-25.0 to 0.0)
    vv = np.random.uniform(-25.0, 0.0, (height, width)).astype(np.float32)
    # Band 2: VH_dB (-30.0 to -5.0)
    vh = np.random.uniform(-30.0, -5.0, (height, width)).astype(np.float32)
    # Band 3: VV_dB - VH_dB
    ratio = vv - vh

    # Add nodata region (label 255)
    vv[:20, :20] = 255.0
    vh[:20, :20] = 255.0
    ratio[:20, :20] = 255.0

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": rasterio.float32,
        "crs": crs,
        "transform": transform,
        "nodata": 255.0
    }

    with rasterio.open(raster_path, "w", **profile) as dst:
        dst.write(vv, 1)
        dst.write(vh, 2)
        dst.write(ratio, 3)

    return raster_path


@pytest.fixture
def invalid_single_band_raster(tmp_path):
    """Fixture generating an invalid 1-band SAR raster."""
    raster_path = str(tmp_path / "invalid_single_band.tif")
    profile = {
        "driver": "GTiff",
        "height": 64,
        "width": 64,
        "count": 1,
        "dtype": rasterio.float32,
        "crs": "EPSG:4326",
        "transform": from_origin(45.0, 2.0, 0.001, 0.001)
    }
    with rasterio.open(raster_path, "w", **profile) as dst:
        dst.write(np.zeros((64, 64), dtype=np.float32), 1)
    return raster_path


def test_checkpoint_loading_success():
    """1. Test successful model checkpoint loading."""
    service = UNetSegmentationService()
    service._load_model_if_needed()
    assert service.model is not None
    assert next(service.model.parameters()).device is not None


def test_missing_checkpoint_error(tmp_path):
    """3. Test missing or incompatible checkpoint error handling."""
    fake_config = tmp_path / "fake_config.json"
    fake_config.write_text(json.dumps({
        "architecture": {"checkpoint_path": "non_existent_checkpoint.pt", "class_name": "SmallUNet"},
        "input_specification": {"num_channels": 3, "normalization": {"channel_means": [0,0,0], "channel_stds": [1,1,1]}},
        "inference_parameters": {"nodata_value": 255, "probability_threshold": 0.50}
    }))
    
    service = UNetSegmentationService(config_path=str(fake_config))
    with pytest.raises(FileNotFoundError, match="Primary model checkpoint not found"):
        service._load_model_if_needed()


def test_valid_input_inference(synthetic_sar_raster):
    """2 & 7. Test valid input inference, output shapes, and probability bounds [0.0, 1.0]."""
    service = UNetSegmentationService()
    req = SegmentationRequest(sar_raster_path=synthetic_sar_raster, probability_threshold=0.50)
    res = service.predict(req)

    assert res.status == "ok"
    assert os.path.exists(res.binary_mask_path)
    assert os.path.exists(res.probability_map_path)
    assert os.path.exists(res.binary_overlay_png_path)
    assert os.path.exists(res.probability_overlay_png_path)

    # Check summary metrics
    assert res.summary.total_pixels == 256 * 256
    assert res.summary.valid_pixels > 0
    assert 0.0 <= res.summary.mean_water_probability <= 1.0


def test_invalid_band_count_error(invalid_single_band_raster):
    """4. Test invalid band count error handling."""
    service = UNetSegmentationService()
    # Mock secure path validation for temp directory
    abs_path = os.path.abspath(invalid_single_band_raster)
    with pytest.raises(ValueError, match="Invalid band count"):
        service.validate_input_raster(abs_path)


def test_nodata_preservation(synthetic_sar_raster):
    """5. Test nodata preservation (pixels with value 255 remain 255 in binary mask)."""
    service = UNetSegmentationService()
    req = SegmentationRequest(sar_raster_path=synthetic_sar_raster, probability_threshold=0.50)
    res = service.predict(req)

    with rasterio.open(res.binary_mask_path) as src:
        bin_mask = src.read(1)
        # Verify the top-left 20x20 block (nodata fixture) contains 255
        assert (bin_mask[:20, :20] == 255).all()


def test_output_crs_geotransform_bounds(synthetic_sar_raster):
    """6. Test output CRS, geotransform, dimensions, and WGS84 bounds."""
    service = UNetSegmentationService()
    req = SegmentationRequest(sar_raster_path=synthetic_sar_raster)
    res = service.predict(req)

    with rasterio.open(synthetic_sar_raster) as src_in, rasterio.open(res.binary_mask_path) as src_out:
        assert src_out.crs == src_in.crs
        assert src_out.transform == src_in.transform
        assert src_out.width == src_in.width
        assert src_out.height == src_in.height

    assert len(res.wgs84_bounds) == 2
    assert res.wgs84_bounds[0][0] < res.wgs84_bounds[1][0]  # min_lat < max_lat


def test_api_segmentation_endpoint(test_client, synthetic_sar_raster):
    """8 & 9. Test API segmentation endpoint POST /segmentation/predict and layer serving GET."""
    payload = {
        "sar_raster_path": synthetic_sar_raster,
        "probability_threshold": 0.50
    }
    response = test_client.post("/segmentation/predict", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "ok"
    assert "run_id" in data
    assert "wgs84_bounds" in data

    # Test layer download endpoint for generated PNG overlay
    run_id = data["run_id"]
    layer_res = test_client.get(f"/dashboard/layers/{run_id}/water_prediction_overlay.png")
    assert layer_res.status_code == 200
    assert layer_res.headers["content-type"] == "image/png"


def test_unauthorized_filesystem_access_prevention(test_client):
    """10. Test security path validation preventing unauthorized filesystem access."""
    forbidden_payload = {
        "sar_raster_path": "C:\\Windows\\System32\\drivers\\etc\\hosts",
        "probability_threshold": 0.50
    }
    response = test_client.post("/segmentation/predict", json=forbidden_payload)
    assert response.status_code == 403
    assert "Security Error" in response.json()["detail"]
