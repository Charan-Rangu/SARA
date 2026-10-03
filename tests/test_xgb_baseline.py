"""Unit tests for SARA Experiment 2: XGBoost Baseline module."""

import os
import json
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
import xgboost as xgb

from ml.xgb_config import XGBoostConfig
from ml.xgb_baseline import (
    train_xgboost,
    evaluate_xgboost
)
from ml.rf_baseline import (
    extract_tile_features_and_labels,
    sample_tile_pixels,
    calculate_rf_metrics
)


@pytest.fixture
def mock_geotiff_pair(tmp_path):
    """Creates mock SAR and label GeoTIFF files for testing."""
    sar_path = tmp_path / "mock_sar.tif"
    label_path = tmp_path / "mock_label.tif"

    h, w = 64, 64
    transform = from_origin(85.0, 28.0, 0.0001, 0.0001)
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 3,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform
    }

    # Generate SAR data with spatial variation: Band 1 = VV, Band 2 = VH, Band 3 = VV - VH
    sar_data = np.zeros((3, h, w), dtype=np.float32)
    sar_data[0] = -10.0 + np.arange(h * w, dtype=np.float32).reshape(h, w) * 0.001
    sar_data[1] = -18.0 + np.arange(h * w, dtype=np.float32).reshape(h, w) * 0.0005
    sar_data[2] = sar_data[0] - sar_data[1]

    with rasterio.open(sar_path, "w", **profile) as dst:
        dst.write(sar_data)

    # Generate label data: 0 = nonwater, 1 = water, 255 = nodata
    label_data = np.zeros((h, w), dtype=np.uint8)
    label_data[0:10, :] = 1     # 640 water pixels
    label_data[10:50, :] = 0    # 2560 non-water pixels
    label_data[50:, :] = 255    # 896 nodata pixels

    profile_lbl = profile.copy()
    profile_lbl["count"] = 1
    profile_lbl["dtype"] = "uint8"
    profile_lbl["nodata"] = 255

    with rasterio.open(label_path, "w", **profile_lbl) as dst:
        dst.write(label_data, 1)

    return str(sar_path), str(label_path)


def test_xgb_feature_order_and_extraction(mock_geotiff_pair):
    sar_path, label_path = mock_geotiff_pair
    features, labels, valid_mask, profile = extract_tile_features_and_labels(sar_path, label_path)

    assert features.shape == (64 * 64, 3)
    assert labels.shape == (64 * 64,)
    assert valid_mask.shape == (64 * 64,)

    # Feature 0 = VV (-10.0 at pixel 0), Feature 1 = VH (-18.0 at pixel 0), Feature 2 = VV - VH
    assert pytest.approx(features[0, 0], 0.001) == -10.0
    assert pytest.approx(features[0, 1], 0.001) == -18.0
    assert np.allclose(features[:, 2], features[:, 0] - features[:, 1])

    # Valid mask should exclude label 255
    valid_count = np.sum(valid_mask)
    expected_valid = 640 + 2560  # 3200
    assert valid_count == expected_valid
    assert np.sum(labels[valid_mask] == 255) == 0


def test_xgb_training_and_feature_importances(mock_geotiff_pair, tmp_path):
    sar_path, label_path = mock_geotiff_pair
    X_sample, y_sample = sample_tile_pixels(sar_path, label_path, samples_water=200, samples_nonwater=400, seed=42)

    cfg = XGBoostConfig(n_estimators=10, max_depth=3, experiment_dir=str(tmp_path / "exp_xgb"))

    clf, fit_stats = train_xgboost(X_sample, y_sample, cfg)

    assert isinstance(clf, xgb.XGBClassifier)
    assert fit_stats["train_duration_seconds"] >= 0.0
    assert "feature_importances_default_gain" in fit_stats
    assert "VV_dB" in fit_stats["feature_importances_default_gain"]


def test_xgb_prediction_export_and_metadata(mock_geotiff_pair, tmp_path):
    sar_path, label_path = mock_geotiff_pair
    X_sample, y_sample = sample_tile_pixels(sar_path, label_path, samples_water=200, samples_nonwater=400, seed=42)

    cfg = XGBoostConfig(n_estimators=10, max_depth=3, experiment_dir=str(tmp_path / "exp_xgb"))
    clf, _ = train_xgboost(X_sample, y_sample, cfg)

    val_manifest_file = tmp_path / "val_manifest.json"
    val_data = {
        "event_ids": ["Somalia"],
        "tiles": [{
            "tile_id": "Somalia_mock_xgb",
            "event_id": "Somalia",
            "sar_path": sar_path,
            "label_path": label_path,
            "width": 64,
            "height": 64
        }]
    }
    with open(val_manifest_file, "w") as f:
        json.dump(val_data, f)

    overall_metrics, per_tile = evaluate_xgboost(clf, str(val_manifest_file), cfg, export_predictions=True)

    pred_tif = tmp_path / "exp_xgb" / "predictions" / "Somalia_mock_xgb_pred.tif"
    assert pred_tif.exists()

    with rasterio.open(pred_tif) as src:
        assert src.shape == (64, 64)
        assert src.count == 1
        assert src.dtypes[0] == "uint8"
        assert src.nodata == 255
        assert src.crs.to_string() == "EPSG:4326"

        pred_data = src.read(1)
        # Verify nodata pixels remain 255
        assert np.all(pred_data[50:, :] == 255)
        # Valid predictions must contain only 0 or 1
        assert set(np.unique(pred_data[:50, :])).issubset({0, 1})
