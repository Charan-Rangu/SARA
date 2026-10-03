"""Unit and integration tests for Custom SAR Inference, File Picker, Band Validation, and Nodata Handling."""

import os
import io
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
def client():
    return TestClient(app)


@pytest.fixture
def valid_3band_sar_raster(tmp_path):
    """Generate a standard 3-band SAR GeoTIFF (VV_dB, VH_dB, VV-VH) with CRS EPSG:4326."""
    raster_path = str(tmp_path / "valid_custom_sar_3band.tif")
    width, height = 128, 128
    crs = "EPSG:4326"
    transform = from_origin(85.0, 28.0, 0.0001, 0.0001)

    vv = np.random.uniform(-22.0, -5.0, (height, width)).astype(np.float32)
    vh = np.random.uniform(-28.0, -12.0, (height, width)).astype(np.float32)
    diff = vv - vh

    # Introduce a 16x16 nodata block (value 255.0)
    vv[:16, :16] = 255.0
    vh[:16, :16] = 255.0
    diff[:16, :16] = 255.0

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
        dst.write(diff, 3)
        dst.set_band_description(1, "VV_dB")
        dst.set_band_description(2, "VH_dB")
        dst.set_band_description(3, "VV_dB - VH_dB")

    return raster_path


@pytest.fixture
def valid_2band_sar_raster(tmp_path):
    """Generate a valid 2-band calibrated SAR GeoTIFF with explicit descriptions."""
    raster_path = str(tmp_path / "valid_custom_sar_2band.tif")
    width, height = 128, 128
    crs = "EPSG:4326"
    transform = from_origin(85.0, 28.0, 0.0001, 0.0001)

    # Linear sigma0 values
    vv = np.random.uniform(0.005, 0.25, (height, width)).astype(np.float32)
    vh = np.random.uniform(0.001, 0.05, (height, width)).astype(np.float32)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 2,
        "dtype": rasterio.float32,
        "crs": crs,
        "transform": transform,
        "nodata": 0.0
    }

    with rasterio.open(raster_path, "w", **profile) as dst:
        dst.write(vv, 1)
        dst.write(vh, 2)
        dst.set_band_description(1, "Sigma0_VV_linear")
        dst.set_band_description(2, "Sigma0_VH_linear")

    return raster_path


@pytest.fixture
def ambiguous_2band_sar_raster(tmp_path):
    """Generate a 2-band raster with no descriptions or metadata (ambiguous band order)."""
    raster_path = str(tmp_path / "ambiguous_2band_sar.tif")
    width, height = 64, 64
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 2,
        "dtype": rasterio.float32,
        "crs": "EPSG:4326",
        "transform": from_origin(85.0, 28.0, 0.0001, 0.0001)
    }
    with rasterio.open(raster_path, "w", **profile) as dst:
        dst.write(np.ones((64, 64), dtype=np.float32) * 0.01, 1)
        dst.write(np.ones((64, 64), dtype=np.float32) * 0.005, 2)
    return raster_path


@pytest.fixture
def reversed_3band_sar_raster(tmp_path):
    """Generate an incompatible 3-band raster where Band 1 is VH and Band 2 is VV."""
    raster_path = str(tmp_path / "reversed_3band_sar.tif")
    width, height = 64, 64
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": rasterio.float32,
        "crs": "EPSG:4326",
        "transform": from_origin(85.0, 28.0, 0.0001, 0.0001)
    }
    with rasterio.open(raster_path, "w", **profile) as dst:
        dst.write(np.ones((64, 64), dtype=np.float32) * -20.0, 1)
        dst.write(np.ones((64, 64), dtype=np.float32) * -10.0, 2)
        dst.write(np.ones((64, 64), dtype=np.float32) * 10.0, 3)
        dst.set_band_description(1, "VH_dB")
        dst.set_band_description(2, "VV_dB")
        dst.set_band_description(3, "Diff")
    return raster_path


def test_available_rasters_discovery(client):
    """1. Test discovering available local GeoTIFFs via GET /segmentation/available-rasters."""
    response = client.get("/segmentation/available-rasters")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "rasters" in data
    assert isinstance(data["rasters"], list)


def test_upload_sar_raster_endpoint(client, valid_3band_sar_raster):
    """2. Test uploading a local GeoTIFF via POST /segmentation/upload."""
    with open(valid_3band_sar_raster, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/segmentation/upload",
        files={"file": ("test_upload.tif", io.BytesIO(file_bytes), "image/tiff")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "uploaded_raster" in data
    up = data["uploaded_raster"]
    assert up["filename"] == "test_upload.tif"
    assert up["bands"] == 3
    assert os.path.exists(up["path"])


def test_upload_invalid_extension_rejected(client):
    """3. Test rejecting invalid non-GeoTIFF upload files."""
    response = client.post(
        "/segmentation/upload",
        files={"file": ("invalid_script.py", io.BytesIO(b"print('hello')"), "text/plain")}
    )
    assert response.status_code == 400
    assert "Invalid file extension" in response.json()["detail"]


def test_inference_on_custom_3band_raster(client, valid_3band_sar_raster):
    """4. Test Small U-Net inference on a custom 3-band GeoTIFF."""
    payload = {
        "sar_raster_path": valid_3band_sar_raster,
        "probability_threshold": 0.50
    }
    response = client.post("/segmentation/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["input_filename"] == "valid_custom_sar_3band.tif"
    assert data["probability_threshold"] == 0.50
    assert os.path.exists(data["binary_mask_path"])
    assert os.path.exists(data["probability_map_path"])
    assert os.path.exists(data["binary_overlay_png_path"])
    assert os.path.exists(data["probability_overlay_png_path"])
    assert data["summary"]["total_pixels"] == 128 * 128
    assert data["summary"]["nodata_pixels"] >= 16 * 16


def test_nodata_preservation_in_custom_inference(valid_3band_sar_raster):
    """5. Test nodata preservation (255) in custom inference output."""
    service = UNetSegmentationService()
    req = SegmentationRequest(sar_raster_path=valid_3band_sar_raster, probability_threshold=0.50)
    res = service.predict(req)

    with rasterio.open(res.binary_mask_path) as src:
        mask = src.read(1)
        # Verify the 16x16 nodata block is 255
        assert (mask[:16, :16] == 255).all()


def test_threshold_sensitivity(valid_3band_sar_raster):
    """6. Test that higher decision threshold produces less or equal predicted water pixels."""
    service = UNetSegmentationService()
    
    res_low = service.predict(SegmentationRequest(sar_raster_path=valid_3band_sar_raster, probability_threshold=0.20))
    res_med = service.predict(SegmentationRequest(sar_raster_path=valid_3band_sar_raster, probability_threshold=0.50))
    res_high = service.predict(SegmentationRequest(sar_raster_path=valid_3band_sar_raster, probability_threshold=0.85))

    p_low = res_low.summary.predicted_water_pixels
    p_med = res_med.summary.predicted_water_pixels
    p_high = res_high.summary.predicted_water_pixels

    assert p_low >= p_med >= p_high, f"Threshold monotonicity failed: low={p_low}, med={p_med}, high={p_high}"


def test_auto_preparation_of_2band_with_metadata(client, valid_2band_sar_raster):
    """7. Test auto-preparation of 2-band SAR raster with explicit polarization metadata."""
    payload = {
        "sar_raster_path": valid_2band_sar_raster,
        "probability_threshold": 0.50,
        "auto_prepare_2band": True
    }
    response = client.post("/segmentation/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert os.path.exists(data["binary_mask_path"])


def test_ambiguous_band_order_strictly_rejected(client, ambiguous_2band_sar_raster):
    """8. Test that ambiguous band metadata is strictly rejected and never guessed."""
    payload = {
        "sar_raster_path": ambiguous_2band_sar_raster,
        "probability_threshold": 0.50,
        "auto_prepare_2band": True
    }
    response = client.post("/segmentation/predict", json=payload)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Polarization band identities" in detail or "Ambiguous" in detail or "prohibited" in detail


def test_incompatible_3band_order_rejected(client, reversed_3band_sar_raster):
    """9. Test that inverted 3-band order (Band 1=VH, Band 2=VV) is strictly rejected."""
    payload = {
        "sar_raster_path": reversed_3band_sar_raster,
        "probability_threshold": 0.50
    }
    response = client.post("/segmentation/predict", json=payload)
    assert response.status_code == 400
    assert "Incompatible band order detected" in response.json()["detail"]


def test_separate_output_directory_creation(valid_3band_sar_raster):
    """10. Test that multiple runs produce separate output directories and do not overwrite baseline."""
    service = UNetSegmentationService()
    
    res1 = service.predict(SegmentationRequest(sar_raster_path=valid_3band_sar_raster, run_id="custom_run_alpha"))
    res2 = service.predict(SegmentationRequest(sar_raster_path=valid_3band_sar_raster, run_id="custom_run_beta"))

    assert res1.output_directory != res2.output_directory
    assert os.path.exists(res1.binary_mask_path)
    assert os.path.exists(res2.binary_mask_path)
    assert "custom_run_alpha" in res1.output_directory
    assert "custom_run_beta" in res2.output_directory
