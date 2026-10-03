"""Focused automated verification test suite for the Three Core Engines of SARA.

Engine 1: Multi-Temporal SAR Change Detection & 5-Class Agreement Mask Engine
Engine 2: Copernicus 30m Topographic Slope & Shadow Disambiguation Engine
Engine 3: Independent Validation & Situation Reporting Engine
"""

import os
import tempfile
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin

from geospatial.change_detection import (
    SARChangeDetector,
    ChangeDetectionConfig,
    ChangeDetectionResult,
    generate_agreement_mask,
    AgreementMaskResult
)
from geospatial.terrain_analysis import TerrainAnalyzer, TerrainAnalysisResult
from validation.raster_validator import (
    compute_binary_metrics_from_counts,
    RasterValidator,
    ValidationMetrics
)
from backend.services.situation_report import SituationReportGenerator, ReportRequest


# ==========================================
# HELPER FIXTURES
# ==========================================

def make_test_geotiff(path, data, crs="EPSG:4326", transform=None, nodata=255.0):
    h, w = data.shape
    if transform is None:
        transform = from_origin(85.0, 28.0, 0.0001, 0.0001)
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


# ==========================================
# PART 1: ENGINE 1 (SAR CHANGE DETECTION)
# ==========================================

def test_engine1_identical_rasters(tmp_path):
    """Engine 1: Identical pre/post rasters must produce 0% change."""
    pre_p = str(tmp_path / "pre.tif")
    post_p = str(tmp_path / "post.tif")
    out_p = str(tmp_path / "chg.tif")

    data = np.full((16, 16), -12.0, dtype=np.float32)
    make_test_geotiff(pre_p, data)
    make_test_geotiff(post_p, data)

    detector = SARChangeDetector(ChangeDetectionConfig(threshold_db=-3.0, input_units="dB"))
    res = detector.detect_changes(pre_p, post_p, out_p)
    assert res.changed_pixels == 0
    assert res.change_percentage == 0.0


def test_engine1_known_decrease_and_increase(tmp_path):
    """Engine 1: -6 dB decrease triggers change; +6 dB increase does not."""
    pre_p = str(tmp_path / "pre.tif")
    post_p = str(tmp_path / "post.tif")
    out_p = str(tmp_path / "chg.tif")

    pre = np.full((16, 16), -10.0, dtype=np.float32)
    post = np.full((16, 16), -10.0, dtype=np.float32)
    # Left half: decrease to -16 dB (diff = -6 dB <= -3 dB -> change)
    post[:, :8] = -16.0
    # Right half: increase to -4 dB (diff = +6 dB -> no change)
    post[:, 8:] = -4.0

    make_test_geotiff(pre_p, pre)
    make_test_geotiff(post_p, post)

    detector = SARChangeDetector(ChangeDetectionConfig(threshold_db=-3.0, input_units="dB"))
    res = detector.detect_changes(pre_p, post_p, out_p)
    assert res.changed_pixels == 16 * 8
    assert res.change_percentage == 50.0


def test_engine1_exact_threshold_boundary(tmp_path):
    """Engine 1: Delta_dB == -3.0000 exactly must trigger change (Delta_dB <= -3.0)."""
    pre_p = str(tmp_path / "pre.tif")
    post_p = str(tmp_path / "post.tif")
    out_p = str(tmp_path / "chg.tif")

    # Pixel (0,0): diff = -3.000 dB (exact threshold -> CHANGE)
    # Pixel (0,1): diff = -2.999 dB (above threshold -> NO CHANGE)
    # Pixel (0,2): diff = -3.001 dB (below threshold -> CHANGE)
    pre = np.array([[-10.0, -10.0, -10.0]], dtype=np.float32)
    post = np.array([[-13.0, -12.999, -13.001]], dtype=np.float32)

    make_test_geotiff(pre_p, pre)
    make_test_geotiff(post_p, post)

    detector = SARChangeDetector(ChangeDetectionConfig(threshold_db=-3.0, input_units="dB"))
    res = detector.detect_changes(pre_p, post_p, out_p)
    assert res.changed_pixels == 2
    with rasterio.open(out_p) as src:
        mask = src.read(1)
        assert mask[0, 0] == 1  # exact -3.0 dB triggers change
        assert mask[0, 1] == 0  # -2.999 dB does not
        assert mask[0, 2] == 1  # -3.001 dB triggers change


def test_engine1_nan_inf_nodata_exclusion(tmp_path):
    """Engine 1: Nodata, NaN, +Inf, -Inf must be excluded from valid count."""
    pre_p = str(tmp_path / "pre.tif")
    post_p = str(tmp_path / "post.tif")
    out_p = str(tmp_path / "chg.tif")

    pre = np.array([[-9999.0, np.nan], [-10.0, np.inf]], dtype=np.float32)
    post = np.array([[-16.0, -16.0], [-16.0, -16.0]], dtype=np.float32)

    make_test_geotiff(pre_p, pre, nodata=-9999.0)
    make_test_geotiff(post_p, post, nodata=-9999.0)

    detector = SARChangeDetector(ChangeDetectionConfig(input_units="dB"))
    res = detector.detect_changes(pre_p, post_p, out_p)
    assert res.total_valid_pixels == 1  # only (1,0) is valid
    assert res.changed_pixels == 1


def test_engine1_5class_agreement_logic(tmp_path):
    """Engine 1: Verify 5-class categorical agreement assignments on hand-constructed data."""
    w_p = str(tmp_path / "water.tif")
    c_p = str(tmp_path / "change.tif")
    s_p = str(tmp_path / "slope15.tif")
    out_p = str(tmp_path / "agreement.tif")

    # 1x6 synthetic array:
    # 0: Water=0, Change=0, Slope<=15 -> Class 0 (Dry Land)
    # 1: Water=1, Change=0, Slope<=15 -> Class 1 (Persistent Water)
    # 2: Water=0, Change=1, Slope<=15 -> Class 2 (Change Only)
    # 3: Water=1, Change=1, Slope<=15 -> Class 3 (Agreed Inundation)
    # 4: Water=1, Change=0, Slope>15  -> Class 4 (Suspected Shadow)
    # 5: Water=255 (Nodata)           -> Class 255
    water_arr = np.array([[0, 1, 0, 1, 1, 255]], dtype=np.uint8)
    chg_arr   = np.array([[0, 0, 1, 1, 0, 0]], dtype=np.uint8)
    s15_arr   = np.array([[1, 1, 1, 1, 0, 1]], dtype=np.uint8)  # 1 = slope <= 15 deg

    make_test_geotiff(w_p, water_arr, nodata=255)
    make_test_geotiff(c_p, chg_arr, nodata=255)
    make_test_geotiff(s_p, s15_arr, nodata=255)

    res = generate_agreement_mask(w_p, c_p, s_p, out_p)
    assert res.total_valid_pixels == 5
    with rasterio.open(out_p) as src:
        out_data = src.read(1)
        expected = np.array([[0, 1, 2, 3, 4, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(out_data, expected)


# ==========================================
# PART 2: ENGINE 2 (COPERNICUS DEM & SLOPE)
# ==========================================

def test_engine2_flat_and_inclined_slope(tmp_path):
    """Engine 2: Flat DEM yields ~0 deg slope; 45 deg incline yields 45.0 deg slope."""
    flat_p = str(tmp_path / "flat_dem.tif")
    inc_p = str(tmp_path / "inc_dem.tif")

    # Flat DEM
    make_test_geotiff(flat_p, np.full((16, 16), 100.0, dtype=np.float32), nodata=-9999.0)
    analyzer = TerrainAnalyzer(pixel_size_meters=30.0)
    res_flat = analyzer.analyze_terrain(flat_p, output_dir=str(tmp_path / "flat_out"))

    with rasterio.open(res_flat.slope_path) as src:
        slope_flat = src.read(1)
        assert np.max(slope_flat) < 1e-4

    # Inclined DEM (Rise equal to dx_meters per pixel = 45 deg)
    dx_meters = 0.0001 * 111320.0 * np.cos(np.radians(28.0))
    inc_data = np.zeros((16, 16), dtype=np.float32)
    for col in range(16):
        inc_data[:, col] = col * dx_meters
    make_test_geotiff(inc_p, inc_data, nodata=-9999.0)
    res_inc = analyzer.analyze_terrain(inc_p, output_dir=str(tmp_path / "inc_out"))

    with rasterio.open(res_inc.slope_path) as src:
        slope_inc = src.read(1)
        # Check interior pixels away from boundary edges
        central = slope_inc[4:12, 4:12]
        assert abs(np.mean(central) - 45.0) < 0.1


def test_engine2_nodata_boundary_integrity(tmp_path):
    """Engine 2: Nodata pixels must remain -9999.0 and not contaminate neighboring valid slopes."""
    nd_p = str(tmp_path / "nd_dem.tif")
    data = np.full((16, 16), 200.0, dtype=np.float32)
    data[:8, :] = -9999.0  # Top half nodata

    make_test_geotiff(nd_p, data, nodata=-9999.0)
    analyzer = TerrainAnalyzer(pixel_size_meters=30.0)
    res = analyzer.analyze_terrain(nd_p, output_dir=str(tmp_path / "nd_out"))

    with rasterio.open(res.slope_path) as src:
        slope = src.read(1)
        assert np.all(slope[:8, :] == -9999.0)
        # Bottom region should be flat
        assert np.all(slope[10:14, :] < 1e-4)


# ==========================================
# PART 3: ENGINE 3 (VALIDATION & SITREP)
# ==========================================

def test_engine3_confusion_matrix_and_metrics():
    """Engine 3: Verify precision, recall, F1, IoU, and accuracy calculations."""
    m = compute_binary_metrics_from_counts(tp=40, fp=10, fn=20, tn=130, nodata=15)
    assert m.precision == 0.80
    assert abs(m.recall - (40/60)) < 1e-4
    assert abs(m.f1_score - (80/110)) < 1e-4
    assert abs(m.iou - (40/70)) < 1e-4
    assert m.accuracy == 0.85


def test_engine3_division_by_zero_handling():
    """Engine 3: Zero denominators must yield explicit undefined flags without crashing."""
    # 1. Empty prediction (TP=0, FP=0)
    m1 = compute_binary_metrics_from_counts(tp=0, fp=0, fn=10, tn=90)
    assert m1.precision is None
    assert m1.precision_undefined is True
    assert m1.recall == 0.0

    # 2. Empty reference (TP=0, FN=0)
    m2 = compute_binary_metrics_from_counts(tp=0, fp=10, fn=0, tn=90)
    assert m2.recall is None
    assert m2.recall_undefined is True
    assert m2.precision == 0.0

    # 3. All nodata / Zero valid pixels
    m3 = compute_binary_metrics_from_counts(tp=0, fp=0, fn=0, tn=0, nodata=100)
    assert m3.status == "UNDEFINED_ZERO_VALID_PIXELS"
    assert m3.precision is None
    assert m3.accuracy is None


def test_engine3_quarantine_and_pending_status(tmp_path):
    """Engine 3: EMSR927 data must be quarantined; missing reference returns PENDING status."""
    pred_p = str(tmp_path / "pred.tif")
    make_test_geotiff(pred_p, np.ones((8, 8), dtype=np.float32))

    validator = RasterValidator()

    # Missing reference
    res_pend = validator.evaluate(pred_p, None)
    assert "INDEPENDENT VALIDATION: PENDING" in res_pend.status

    # Quarantined reference
    res_quar = validator.evaluate(pred_p, "data/raw/EMSR927_AOI01_test.tif")
    assert "QUARANTINED" in res_quar.status


def test_engine3_situation_report_provenance_and_limitations():
    """Engine 3: Situation report generator preserves mandatory caveats and legal attributions."""
    gen = SituationReportGenerator()
    req = ReportRequest(
        run_id="run_verify_test_001",
        aoi_name="Test Basin",
        event_date="2026-10-03"
    )
    rep = gen.generate_report(req)
    assert "run_verify_test_001" in rep
    assert "System Limitation Statement" in rep
    assert "Contains modified Copernicus Sentinel data 2026." in rep
    assert "Produced using Copernicus WorldDEM-30" in rep
