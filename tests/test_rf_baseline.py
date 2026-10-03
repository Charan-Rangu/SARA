"""Unit tests for SARA Experiment 1: Random Forest Baseline module."""

import os
import json
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from sklearn.ensemble import RandomForestClassifier

from ml.rf_config import RandomForestConfig
from ml.rf_baseline import (
    extract_tile_features_and_labels,
    sample_tile_pixels,
    calculate_rf_metrics,
    train_random_forest,
    evaluate_random_forest
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
    np.random.seed(42)
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


def test_feature_order_and_extraction(mock_geotiff_pair):
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


def test_deterministic_sampling(mock_geotiff_pair):
    sar_path, label_path = mock_geotiff_pair

    # Run sampling twice with identical seed
    x1, y1 = sample_tile_pixels(sar_path, label_path, samples_water=100, samples_nonwater=200, seed=42)
    x2, y2 = sample_tile_pixels(sar_path, label_path, samples_water=100, samples_nonwater=200, seed=42)

    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)
    assert len(x1) == 300
    assert np.sum(y1 == 1) == 100
    assert np.sum(y1 == 0) == 200

    # Different seed should yield different sample order/indices
    x3, y3 = sample_tile_pixels(sar_path, label_path, samples_water=100, samples_nonwater=200, seed=99)
    assert not np.array_equal(x1, x3)


def test_synthetic_metrics_calculation():
    # Construct synthetic ground truth and predictions
    # 0 = nonwater, 1 = water, 255 = nodata
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 255, 255], dtype=np.uint8)
    y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1,   0  ], dtype=np.uint8)
    # Valid targets (excluding 255): 4 water (indices 0..3), 6 non-water (indices 4..9)
    # TP: 3 (idx 0,1,2), FN: 1 (idx 3), FP: 1 (idx 4), TN: 5 (idx 5..9)

    metrics = calculate_rf_metrics(y_true, y_pred, nodata_value=255)

    assert metrics["total_valid_pixels"] == 10
    cm = metrics["confusion_matrix"]
    assert cm["TP"] == 3
    assert cm["FP"] == 1
    assert cm["FN"] == 1
    assert cm["TN"] == 5

    # Precision = 3 / (3 + 1) = 0.75
    # Recall = 3 / (3 + 1) = 0.75
    # F1 = 2 * 3 / (2*3 + 1 + 1) = 6/8 = 0.75
    # IoU = 3 / (3 + 1 + 1) = 3/5 = 0.60
    assert pytest.approx(metrics["water_metrics"]["precision"], 0.001) == 0.75
    assert pytest.approx(metrics["water_metrics"]["recall"], 0.001) == 0.75
    assert pytest.approx(metrics["water_metrics"]["f1_dice"], 0.001) == 0.75
    assert pytest.approx(metrics["water_metrics"]["iou"], 0.001) == 0.60


def test_prediction_raster_export_and_metadata(mock_geotiff_pair, tmp_path):
    sar_path, label_path = mock_geotiff_pair

    # Train a simple model on mock data
    x_sample, y_sample = sample_tile_pixels(sar_path, label_path, samples_water=200, samples_nonwater=400, seed=42)
    cfg = RandomForestConfig(n_estimators=10, max_depth=5, experiment_dir=str(tmp_path / "exp"))

    clf, _ = train_random_forest(x_sample, y_sample, cfg)

    # Mock val manifest pointing to mock_geotiff_pair
    val_manifest_file = tmp_path / "val_manifest.json"
    val_data = {
        "event_ids": ["Somalia"],
        "tiles": [{
            "tile_id": "Somalia_mock",
            "event_id": "Somalia",
            "sar_path": sar_path,
            "label_path": label_path,
            "width": 64,
            "height": 64
        }]
    }
    with open(val_manifest_file, "w") as f:
        json.dump(val_data, f)

    overall_metrics, per_tile = evaluate_random_forest(clf, str(val_manifest_file), cfg, export_predictions=True)

    pred_tif = tmp_path / "exp" / "predictions" / "Somalia_mock_pred.tif"
    assert pred_tif.exists()

    with rasterio.open(pred_tif) as src:
        assert src.shape == (64, 64)
        assert src.count == 1
        assert src.dtypes[0] == "uint8"
        assert src.nodata == 255
        assert src.crs.to_string() == "EPSG:4326"

        pred_data = src.read(1)
        # Verify nodata pixels (where label was 255) remain 255
        assert np.all(pred_data[50:, :] == 255)
