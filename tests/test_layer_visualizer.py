"""Unit and integration tests for LayerVisualizerService and Dashboard Integration."""

import os
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from backend.services.layer_visualizer import (
    LayerVisualizerService,
    AGREEMENT_PALETTE,
    SLOPE_PALETTE
)


@pytest.fixture
def test_client():
    return TestClient(app)


@pytest.fixture
def synthetic_agreement_raster(tmp_path):
    """Create a synthetic 64x64 agreement GeoTIFF containing classes 0, 1, 2, 3, 4, and 255."""
    raster_path = str(tmp_path / "synthetic_agreement.tif")
    height, width = 64, 64
    crs = "EPSG:4326"
    transform = from_origin(84.0, 29.0, 0.01, 0.01)

    data = np.zeros((height, width), dtype=np.uint8)
    # Class 0: Land (top-left quadrant)
    data[0:32, 0:32] = 0
    # Class 1: Persistent Water (top-right quadrant)
    data[0:32, 32:64] = 1
    # Class 2: Change Only (bottom-left quadrant top half)
    data[32:48, 0:32] = 2
    # Class 3: Agreed Inundation (bottom-left quadrant bottom half)
    data[48:64, 0:32] = 3
    # Class 4: Suspected Shadow (bottom-right quadrant top half)
    data[32:48, 32:64] = 4
    # Nodata 255: bottom-right corner
    data[48:64, 32:64] = 255

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": rasterio.uint8,
        "crs": crs,
        "transform": transform,
        "nodata": 255
    }

    with rasterio.open(raster_path, "w", **profile) as dst:
        dst.write(data, 1)

    return raster_path


@pytest.fixture
def synthetic_slope_raster(tmp_path):
    """Create a synthetic 64x64 slope GeoTIFF with flat, moderate, steep, and very steep values."""
    raster_path = str(tmp_path / "synthetic_slope.tif")
    height, width = 64, 64
    crs = "EPSG:4326"
    transform = from_origin(84.0, 29.0, 0.01, 0.01)

    data = np.full((height, width), fill_value=-9999.0, dtype=np.float32)
    # Flat (<5 deg)
    data[0:32, 0:32] = 2.5
    # Moderate (5-15 deg)
    data[0:32, 32:64] = 10.0
    # Steep (15-25 deg)
    data[32:48, 0:64] = 20.0
    # Very Steep (>25 deg)
    data[48:60, 0:64] = 45.0
    # Nodata (-9999.0): remaining rows 60-64

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": rasterio.float32,
        "crs": crs,
        "transform": transform,
        "nodata": -9999.0
    }

    with rasterio.open(raster_path, "w", **profile) as dst:
        dst.write(data, 1)

    return raster_path


def test_agreement_raster_reading_and_color_mapping(synthetic_agreement_raster, tmp_path):
    """Verify agreement raster reading, class-to-color mapping, and alpha transparency."""
    service = LayerVisualizerService()
    out_png = str(tmp_path / "test_agreement_overlay.png")

    png_path, bounds, meta = service.render_agreement_overlay(
        synthetic_agreement_raster,
        output_png_path=out_png,
        downsample_factor=1
    )

    assert os.path.exists(png_path)
    assert len(bounds) == 2
    assert bounds[0][0] < bounds[1][0]  # min_lat < max_lat

    img = Image.open(png_path)
    rgba = np.array(img)
    assert rgba.shape == (64, 64, 4)

    # Verify Class 0 is transparent
    assert (rgba[0:32, 0:32, 3] == 0).all()

    # Verify Class 1: Blue [29, 78, 216] with alpha > 0
    assert (rgba[0:32, 32:64, 0] == 29).all()
    assert (rgba[0:32, 32:64, 1] == 78).all()
    assert (rgba[0:32, 32:64, 2] == 216).all()
    assert (rgba[0:32, 32:64, 3] > 0).all()

    # Verify Class 2: Yellow [234, 179, 8] with alpha > 0
    assert (rgba[32:48, 0:32, 0] == 234).all()
    assert (rgba[32:48, 0:32, 1] == 179).all()
    assert (rgba[32:48, 0:32, 2] == 8).all()
    assert (rgba[32:48, 0:32, 3] > 0).all()

    # Verify Class 3: Cyan [6, 182, 212] with alpha > 0
    assert (rgba[48:64, 0:32, 0] == 6).all()
    assert (rgba[48:64, 0:32, 1] == 182).all()
    assert (rgba[48:64, 0:32, 2] == 212).all()
    assert (rgba[48:64, 0:32, 3] > 0).all()

    # Verify Class 4: Orange [234, 88, 12] with alpha > 0
    assert (rgba[32:48, 32:64, 0] == 234).all()
    assert (rgba[32:48, 32:64, 1] == 88).all()
    assert (rgba[32:48, 32:64, 2] == 12).all()
    assert (rgba[32:48, 32:64, 3] > 0).all()

    # Verify Nodata 255: Strictly alpha = 0
    assert (rgba[48:64, 32:64, 3] == 0).all()


def test_nodata_is_never_rendered_as_class_0(tmp_path):
    """Verify that a nodata-only raster produces 100% transparent output."""
    raster_path = str(tmp_path / "all_nodata.tif")
    profile = {
        "driver": "GTiff",
        "height": 32,
        "width": 32,
        "count": 1,
        "dtype": rasterio.uint8,
        "crs": "EPSG:4326",
        "transform": from_origin(84.0, 29.0, 0.01, 0.01),
        "nodata": 255
    }
    with rasterio.open(raster_path, "w", **profile) as dst:
        dst.write(np.full((32, 32), 255, dtype=np.uint8), 1)

    service = LayerVisualizerService()
    out_png = str(tmp_path / "all_nodata.png")
    service.render_agreement_overlay(raster_path, out_png, downsample_factor=1)

    img = Image.open(out_png)
    rgba = np.array(img)
    # Entire alpha channel must be 0
    assert (rgba[..., 3] == 0).all()


def test_missing_raster_handling():
    """Verify clear FileNotFoundError when input raster is missing."""
    service = LayerVisualizerService()
    with pytest.raises(FileNotFoundError, match="not found"):
        service.render_agreement_overlay("non_existent_file.tif")


def test_cache_behavior(synthetic_agreement_raster, tmp_path):
    """Verify caching behavior on repeated overlay requests."""
    service = LayerVisualizerService()
    out_png = str(tmp_path / "cached_overlay.png")

    # 1. Initial render
    _, _, meta1 = service.render_agreement_overlay(
        synthetic_agreement_raster,
        out_png,
        downsample_factor=1,
        force_regenerate=False
    )
    assert meta1["cached"] is False
    assert os.path.exists(out_png)

    # 2. Subsequent call returns cached version
    _, _, meta2 = service.render_agreement_overlay(
        synthetic_agreement_raster,
        out_png,
        downsample_factor=1,
        force_regenerate=False
    )
    assert meta2["cached"] is True

    # 3. Force regenerate
    _, _, meta3 = service.render_agreement_overlay(
        synthetic_agreement_raster,
        out_png,
        downsample_factor=1,
        force_regenerate=True
    )
    assert meta3["cached"] is False


def test_slope_overlay_rendering(synthetic_slope_raster, tmp_path):
    """Verify slope overlay rendering, geomorphological palette mapping, and nodata transparency."""
    service = LayerVisualizerService()
    out_png = str(tmp_path / "test_slope_overlay.png")

    png_path, bounds, meta = service.render_slope_overlay(
        synthetic_slope_raster,
        output_png_path=out_png,
        downsample_factor=1
    )

    assert os.path.exists(png_path)
    img = Image.open(png_path)
    rgba = np.array(img)

    # Flat (<5 deg): green [34, 197, 94]
    assert (rgba[0:32, 0:32, 0] == 34).all()
    assert (rgba[0:32, 0:32, 1] == 197).all()

    # Moderate (5-15 deg): yellow [234, 179, 8]
    assert (rgba[0:32, 32:64, 0] == 234).all()
    assert (rgba[0:32, 32:64, 1] == 179).all()

    # Steep (15-25 deg): orange [249, 115, 22]
    assert (rgba[32:48, :, 0] == 249).all()
    assert (rgba[32:48, :, 1] == 115).all()

    # Very steep (>25 deg): crimson [239, 68, 68]
    assert (rgba[48:60, :, 0] == 239).all()

    # Nodata rows (60-64): strictly alpha = 0
    assert (rgba[60:64, :, 3] == 0).all()


def test_nepal_summary_endpoint(test_client):
    """Test GET /dashboard/nepal-summary endpoint response schema and content."""
    res = test_client.get("/dashboard/nepal-summary")
    assert res.status_code == 200, res.text
    data = res.json()

    assert data["status"] == "ok"
    assert "sar_wgs84_bounds" in data
    assert "slope_wgs84_bounds" in data
    assert len(data["sar_wgs84_bounds"]) == 2
    assert "layers" in data
    assert "agreement_overlay" in data["layers"]
    assert "slope_overlay" in data["layers"]
    assert "disclaimers" in data
    assert "Exploratory SAR change-analysis candidates" in data["disclaimers"]["body"]


def test_dashboard_layer_serving_endpoints(test_client):
    """Test GET /dashboard/layers/{run_id}/{layer_name} endpoints for PNG overlays."""
    # 1. Agreement overlay
    res_agr = test_client.get("/dashboard/layers/run_nepal_multitemporal_change_20261003/agreement_overlay.png")
    assert res_agr.status_code == 200
    assert res_agr.headers["content-type"] == "image/png"

    # 2. Slope overlay
    res_slp = test_client.get("/dashboard/layers/run_nepal_multitemporal_change_20261003/slope_overlay.png")
    assert res_slp.status_code == 200
    assert res_slp.headers["content-type"] == "image/png"

    # 3. Water prediction overlay
    res_wat = test_client.get("/dashboard/layers/run_nepal_unet_eval_20261003/water_prediction_overlay.png")
    assert res_wat.status_code == 200
    assert res_wat.headers["content-type"] == "image/png"

    # 4. Path traversal attempt blocked with 403
    res_traversal = test_client.get("/dashboard/layers/run_nepal_multitemporal_change_20261003/..%2F..%2FWindows%2Fwin.ini")
    assert res_traversal.status_code in (400, 403, 404)


def test_dashboard_html_contains_required_controls_and_notices(test_client):
    """Verify that the Leaflet dashboard HTML includes all layer toggles, legend entries, and scientific disclaimers."""
    res = test_client.get("/dashboard")
    assert res.status_code == 200
    html = res.text

    # Layer checkboxes
    assert 'id="layerAgreement"' in html
    assert 'id="layerWaterBin"' in html
    assert 'id="layerWaterProb"' in html
    assert 'id="layerSlope"' in html
    assert 'id="layerAOI"' in html

    # Opacity sliders
    assert 'id="opacityAgreement"' in html
    assert 'id="opacityWaterBin"' in html
    assert 'id="opacityWaterProb"' in html
    assert 'id="opacitySlope"' in html

    # 5-Class Categorical descriptions
    assert "Persistent Water Candidate" in html
    assert "Backscatter Change Only" in html
    assert "Agreed Inundation Candidate" in html
    assert "Suspected High-Slope Shadow Candidate" in html

    # Required scientific disclaimers & threshold qualifiers
    assert "Exploratory SAR change-analysis candidates, not a verified flood boundary" in html
    assert "Independent ground-truth validation is not yet available" in html
    assert "-3 dB" in html

    # Attribution notices
    assert "Copernicus Sentinel Data" in html
    assert "Copernicus WorldDEM-30" in html
    assert "OpenStreetMap contributors" in html
