import os
import tempfile
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin

from geospatial.change_detection import (
    SARChangeDetector,
    ChangeDetectionConfig,
    ChangeDetectionResult
)

def create_synthetic_geotiff(
    path: str,
    data: np.ndarray,
    crs: str = "EPSG:4326",
    transform=None,
    nodata: float = -9999.0
):
    """Create a tiny synthetic GeoTIFF raster fixture for unit testing (Explicitly MOCK DATA)."""
    h, w = data.shape
    if transform is None:
        transform = from_origin(76.15, 10.15, 0.001, 0.001)

    meta = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": data.dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata
    }

    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data, 1)

def test_change_detection_decibel_success(tmp_path):
    pre_path = str(tmp_path / "pre_synthetic_db.tif")
    post_path = str(tmp_path / "post_synthetic_db.tif")
    out_path = str(tmp_path / "preliminary_change_mask.tif")

    # Create 32x32 synthetic decibel backscatter rasters
    # Pre-event: uniform -10.0 dB
    pre_data = np.full((32, 32), -10.0, dtype=np.float32)
    # Post-event: left half -10.0 dB (no change), right half -16.0 dB (-6 dB drop -> flood change)
    post_data = np.full((32, 32), -10.0, dtype=np.float32)
    post_data[:, 16:] = -16.0

    create_synthetic_geotiff(pre_path, pre_data)
    create_synthetic_geotiff(post_path, post_data)

    detector = SARChangeDetector(ChangeDetectionConfig(threshold_db=-3.0, input_units="dB", block_size=16))
    res = detector.detect_changes(pre_path, post_path, out_path)

    assert os.path.exists(out_path)
    assert res.total_valid_pixels == 32 * 32
    assert res.changed_pixels == 32 * 16  # Exactly half the raster changed
    assert res.change_percentage == 50.0

    # Read output mask
    with rasterio.open(out_path) as src:
        mask = src.read(1)
        tags = src.tags()
        assert mask.shape == (32, 32)
        assert np.all(mask[:, :16] == 0)
        assert np.all(mask[:, 16:] == 1)
        assert tags.get("THRESHOLD_DB") == "-3.0"
        assert "Preliminary SAR Change Detection" in tags.get("DESCRIPTION", "")

def test_change_detection_linear_units(tmp_path):
    pre_path = str(tmp_path / "pre_synthetic_lin.tif")
    post_path = str(tmp_path / "post_synthetic_lin.tif")
    out_path = str(tmp_path / "mask_lin.tif")

    # Linear power values: pre = 0.1, post = 0.01 -> log ratio = 10 * log10(0.01/0.1) = -10 dB (< -3.0 dB threshold)
    pre_data = np.full((16, 16), 0.1, dtype=np.float32)
    post_data = np.full((16, 16), 0.01, dtype=np.float32)

    create_synthetic_geotiff(pre_path, pre_data)
    create_synthetic_geotiff(post_path, post_data)

    detector = SARChangeDetector(ChangeDetectionConfig(threshold_db=-3.0, input_units="linear"))
    res = detector.detect_changes(pre_path, post_path, out_path)

    assert res.changed_pixels == 16 * 16
    assert res.change_percentage == 100.0

def test_crs_mismatch_error(tmp_path):
    pre_path = str(tmp_path / "pre.tif")
    post_path = str(tmp_path / "post.tif")

    data = np.ones((16, 16), dtype=np.float32)
    create_synthetic_geotiff(pre_path, data, crs="EPSG:4326")
    create_synthetic_geotiff(post_path, data, crs="EPSG:3857")

    detector = SARChangeDetector()
    with pytest.raises(ValueError, match="Raster CRS mismatch"):
        detector.detect_changes(pre_path, post_path)

def test_dimension_mismatch_error(tmp_path):
    pre_path = str(tmp_path / "pre_32.tif")
    post_path = str(tmp_path / "post_16.tif")

    create_synthetic_geotiff(pre_path, np.ones((32, 32), dtype=np.float32))
    create_synthetic_geotiff(post_path, np.ones((16, 16), dtype=np.float32))

    detector = SARChangeDetector()
    with pytest.raises(ValueError, match="Raster dimension mismatch"):
        detector.detect_changes(pre_path, post_path)

def test_transform_mismatch_error(tmp_path):
    pre_path = str(tmp_path / "pre_t1.tif")
    post_path = str(tmp_path / "post_t2.tif")

    data = np.ones((16, 16), dtype=np.float32)
    t1 = from_origin(76.15, 10.15, 0.001, 0.001)
    t2 = from_origin(70.00, 10.15, 0.001, 0.001)

    create_synthetic_geotiff(pre_path, data, transform=t1)
    create_synthetic_geotiff(post_path, data, transform=t2)

    detector = SARChangeDetector()
    with pytest.raises(ValueError, match="Raster Affine Transform mismatch"):
        detector.detect_changes(pre_path, post_path)

def test_nodata_handling(tmp_path):
    pre_path = str(tmp_path / "pre_nodata.tif")
    post_path = str(tmp_path / "post_nodata.tif")
    out_path = str(tmp_path / "mask_nodata.tif")

    pre_data = np.full((16, 16), -10.0, dtype=np.float32)
    post_data = np.full((16, 16), -16.0, dtype=np.float32)

    # Set top row to nodata -9999.0
    pre_data[0, :] = -9999.0

    create_synthetic_geotiff(pre_path, pre_data, nodata=-9999.0)
    create_synthetic_geotiff(post_path, post_data, nodata=-9999.0)

    detector = SARChangeDetector(config=ChangeDetectionConfig(input_units="dB"))
    res = detector.detect_changes(pre_path, post_path, out_path)

    # Top row (16 pixels) should be masked out as nodata
    assert res.total_valid_pixels == (16 * 16) - 16
    assert res.changed_pixels == (16 * 16) - 16

    with rasterio.open(out_path) as src:
        mask = src.read(1)
        # Nodata pixels in output mask should be fill value 255
        assert np.all(mask[0, :] == 255)

def test_all_nodata_empty_region(tmp_path):
    pre_path = str(tmp_path / "pre_empty.tif")
    post_path = str(tmp_path / "post_empty.tif")

    data = np.full((16, 16), -9999.0, dtype=np.float32)
    create_synthetic_geotiff(pre_path, data, nodata=-9999.0)
    create_synthetic_geotiff(post_path, data, nodata=-9999.0)

    detector = SARChangeDetector()
    res = detector.detect_changes(pre_path, post_path)

    assert res.total_valid_pixels == 0
    assert res.changed_pixels == 0
    assert res.change_percentage == 0.0
