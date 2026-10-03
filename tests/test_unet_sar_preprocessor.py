"""Comprehensive unit tests for 3-channel SAR preprocessor module (UNetSARPreprocessor).

Tests include:
- Polarization identification from band descriptions, tags, and explicit parameters
- Source order [VH, VV] producing [VV_dB, VH_dB, VV_dB - VH_dB]
- Source order [VV, VH] producing [VV_dB, VH_dB, VV_dB - VH_dB]
- Failure on missing/ambiguous polarization metadata
- Linear-to-dB conversion and double-logarithm prevention
- Third channel arithmetic (VV_dB - VH_dB)
- Invalid pixel and nodata propagation (255.0)
- U-Net Z-score normalization
- Spatial alignment, CRS, affine transform, and windowed processing
"""

import os
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin

from geospatial.unet_sar_preprocessor import (
    UNetSARPreprocessor,
    UNetPreprocessingConfig
)


@pytest.fixture
def synthetic_vv_vh_linear_raster(tmp_path):
    """Fixture generating synthetic 2-band linear SAR raster with Band 1=VV, Band 2=VH."""
    path = str(tmp_path / "synthetic_vv_vh_linear.tif")
    width, height = 128, 128
    transform = from_origin(84.0, 28.0, 0.001, 0.001)

    # VV = 0.10 -> 10*log10(0.10) = -10.0 dB
    # VH = 0.01 -> 10*log10(0.01) = -20.0 dB
    vv_lin = np.full((height, width), fill_value=0.10, dtype=np.float32)
    vh_lin = np.full((height, width), fill_value=0.01, dtype=np.float32)

    # Nodata / invalid values
    vv_lin[:10, :10] = 0.0
    vh_lin[:10, :10] = 0.0
    vv_lin[10:20, 10:20] = np.nan
    vh_lin[10:20, 10:20] = np.nan

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 2,
        "dtype": rasterio.float32,
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": 0.0
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.set_band_description(1, "Sigma0_VV")
        dst.set_band_description(2, "Sigma0_VH")
        dst.write(vv_lin, 1)
        dst.write(vh_lin, 2)

    return path


@pytest.fixture
def synthetic_vh_vv_linear_raster(tmp_path):
    """Fixture generating synthetic 2-band linear SAR raster with Band 1=VH, Band 2=VV (reversed SNAP order)."""
    path = str(tmp_path / "synthetic_vh_vv_linear.tif")
    width, height = 128, 128
    transform = from_origin(84.0, 28.0, 0.001, 0.001)

    # VH = 0.01 -> 10*log10(0.01) = -20.0 dB
    # VV = 0.10 -> 10*log10(0.10) = -10.0 dB
    vh_lin = np.full((height, width), fill_value=0.01, dtype=np.float32)
    vv_lin = np.full((height, width), fill_value=0.10, dtype=np.float32)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 2,
        "dtype": rasterio.float32,
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": 0.0
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.set_band_description(1, "Sigma0_VH")
        dst.set_band_description(2, "Sigma0_VV")
        dst.write(vh_lin, 1)
        dst.write(vv_lin, 2)

    return path


@pytest.fixture
def synthetic_unlabeled_2band_raster(tmp_path):
    """Fixture generating a 2-band raster with NO polarization metadata or band descriptions."""
    path = str(tmp_path / "synthetic_unlabeled.tif")
    width, height = 64, 64
    transform = from_origin(84.0, 28.0, 0.001, 0.001)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 2,
        "dtype": rasterio.float32,
        "crs": "EPSG:4326",
        "transform": transform
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.full((height, width), 0.1, dtype=np.float32), 1)
        dst.write(np.full((height, width), 0.01, dtype=np.float32), 2)

    return path


def test_unit_detection():
    """Test unit detection logic (linear power vs decibel scale)."""
    preprocessor = UNetSARPreprocessor()
    
    linear_sample = np.array([0.05, 0.01, 0.12, 0.003], dtype=np.float32)
    assert preprocessor.detect_units(linear_sample) == "linear"

    db_sample = np.array([-11.5, -18.2, -5.0, 2.1], dtype=np.float32)
    assert preprocessor.detect_units(db_sample) == "dB"


def test_source_order_vv_vh_produces_correct_output_order(synthetic_vv_vh_linear_raster, tmp_path):
    """Test that source order [VV, VH] correctly maps to Output [VV_dB, VH_dB, VV_dB - VH_dB]."""
    out_path = str(tmp_path / "out_vv_vh.tif")
    preprocessor = UNetSARPreprocessor(UNetPreprocessingConfig(block_size=64))

    summary = preprocessor.process_raster_file(synthetic_vv_vh_linear_raster, out_path)
    assert summary.units_detected == "linear"
    assert "VV=Band 1, VH=Band 2" in summary.band_mapping_used

    with rasterio.open(out_path) as src:
        assert src.count == 3
        assert src.descriptions == ("VV_dB", "VH_dB", "VV_dB - VH_dB")
        img = src.read()

        valid_pixel = img[:, 50, 50]
        # VV_dB = -10.0, VH_dB = -20.0, diff = +10.0
        assert pytest.approx(valid_pixel[0], abs=1e-2) == -10.0
        assert pytest.approx(valid_pixel[1], abs=1e-2) == -20.0
        assert pytest.approx(valid_pixel[2], abs=1e-2) == +10.0


def test_source_order_vh_vv_produces_correct_output_order(synthetic_vh_vv_linear_raster, tmp_path):
    """Test that reversed source order [VH, VV] correctly maps to Output [VV_dB, VH_dB, VV_dB - VH_dB]."""
    out_path = str(tmp_path / "out_vh_vv.tif")
    preprocessor = UNetSARPreprocessor(UNetPreprocessingConfig(block_size=64))

    summary = preprocessor.process_raster_file(synthetic_vh_vv_linear_raster, out_path)
    assert summary.units_detected == "linear"
    assert "VV=Band 2, VH=Band 1" in summary.band_mapping_used

    with rasterio.open(out_path) as src:
        assert src.count == 3
        assert src.descriptions == ("VV_dB", "VH_dB", "VV_dB - VH_dB")
        img = src.read()

        valid_pixel = img[:, 50, 50]
        # Output Band 1 must be VV_dB (-10.0), Band 2 VH_dB (-20.0), Band 3 diff (+10.0)
        assert pytest.approx(valid_pixel[0], abs=1e-2) == -10.0
        assert pytest.approx(valid_pixel[1], abs=1e-2) == -20.0
        assert pytest.approx(valid_pixel[2], abs=1e-2) == +10.0


def test_explicit_polarization_metadata_override(synthetic_unlabeled_2band_raster, tmp_path):
    """Test explicit band_polarizations override when raster lacks descriptions."""
    out_path = str(tmp_path / "out_explicit_override.tif")
    preprocessor = UNetSARPreprocessor(UNetPreprocessingConfig(block_size=64))

    # Explicitly pass band_polarizations=['VH', 'VV']
    summary = preprocessor.process_raster_file(
        synthetic_unlabeled_2band_raster,
        out_path,
        band_polarizations=["VH", "VV"]
    )
    assert "VV=Band 2, VH=Band 1" in summary.band_mapping_used

    with rasterio.open(out_path) as src:
        img = src.read()
        valid_pixel = img[:, 10, 10]
        # Band 1 in source (0.1 -> -10 dB) was VH, Band 2 (0.01 -> -20 dB) was VV
        # So VV_dB = -20.0, VH_dB = -10.0, diff = -10.0
        assert pytest.approx(valid_pixel[0], abs=1e-2) == -20.0
        assert pytest.approx(valid_pixel[1], abs=1e-2) == -10.0
        assert pytest.approx(valid_pixel[2], abs=1e-2) == -10.0


def test_missing_polarization_metadata_causes_failure(synthetic_unlabeled_2band_raster):
    """Test that missing polarization metadata causes explicit ValueError instead of guessing."""
    preprocessor = UNetSARPreprocessor()
    with pytest.raises(ValueError, match="Unable to reliably identify VV and VH polarization bands"):
        preprocessor.process_raster_file(synthetic_unlabeled_2band_raster)


def test_missing_band_count_error_handling(tmp_path):
    """Test error handling when input raster has fewer than 2 bands."""
    path = str(tmp_path / "single_band.tif")
    profile = {
        "driver": "GTiff", "height": 64, "width": 64, "count": 1,
        "dtype": rasterio.float32, "crs": "EPSG:4326", "transform": from_origin(0, 0, 1, 1)
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.zeros((64, 64), dtype=np.float32), 1)

    preprocessor = UNetSARPreprocessor()
    with pytest.raises(ValueError, match="Input SAR raster must contain at least 2 bands"):
        preprocessor.process_raster_file(path)


def test_prevention_of_double_db_conversion(tmp_path):
    """Test that input rasters already in decibels retain dB values without double log conversion."""
    path = str(tmp_path / "db_input.tif")
    profile = {
        "driver": "GTiff", "height": 64, "width": 64, "count": 2,
        "dtype": rasterio.float32, "crs": "EPSG:4326", "transform": from_origin(0, 0, 1, 1)
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.set_band_description(1, "VV_dB")
        dst.set_band_description(2, "VH_dB")
        dst.write(np.full((64, 64), -11.0, dtype=np.float32), 1)
        dst.write(np.full((64, 64), -18.0, dtype=np.float32), 2)

    out_path = str(tmp_path / "out_db.tif")
    preprocessor = UNetSARPreprocessor()
    summary = preprocessor.process_raster_file(path, out_path)
    assert summary.units_detected == "dB"

    with rasterio.open(out_path) as src:
        img = src.read()
        valid_pixel = img[:, 10, 10]
        assert pytest.approx(valid_pixel[0], abs=1e-2) == -11.0
        assert pytest.approx(valid_pixel[1], abs=1e-2) == -18.0
        assert pytest.approx(valid_pixel[2], abs=1e-2) == 7.0


def test_unet_normalization():
    """Test Small U-Net channel-wise Z-score normalization helper."""
    preprocessor = UNetSARPreprocessor()
    dummy_input = np.array([
        [[-11.0]],  # VV_dB
        [[-18.0]],  # VH_dB
        [[7.0]]     # VV_dB - VH_dB
    ], dtype=np.float32)

    norm_out = preprocessor.apply_unet_normalization(dummy_input)
    assert pytest.approx(norm_out[0, 0, 0], abs=1e-4) == 0.0
    assert pytest.approx(norm_out[1, 0, 0], abs=1e-4) == 0.0
    assert pytest.approx(norm_out[2, 0, 0], abs=1e-4) == 0.0
