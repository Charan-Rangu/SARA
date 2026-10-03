"""Unit tests for SARA Experiment 3: Leakage-Safe Stacking Ensemble module."""

import os
import json
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from sklearn.linear_model import LogisticRegression

from ml.stacking_config import StackingConfig
from ml.stacking_baseline import (
    StackingPipeline,
    generate_event_grouped_oof_predictions,
    train_stacking_meta_model,
    train_full_stacking_pipeline,
    evaluate_stacking_pipeline
)


@pytest.fixture
def mock_multi_event_dataset(tmp_path):
    """Creates mock dataset with 3 geographic events (2 tiles each) for testing grouped cross-validation."""
    events = ["EventA", "EventB", "EventC"]
    tiles_info = []

    h, w = 32, 32
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

    profile_lbl = profile.copy()
    profile_lbl["count"] = 1
    profile_lbl["dtype"] = "uint8"
    profile_lbl["nodata"] = 255

    for ev in events:
        for t_idx in range(2):
            tile_id = f"{ev}_{t_idx}"
            sar_path = tmp_path / f"{tile_id}_sar.tif"
            label_path = tmp_path / f"{tile_id}_label.tif"

            # Synthetic SAR data
            sar_data = np.zeros((3, h, w), dtype=np.float32)
            sar_data[0] = -10.0 + (t_idx * 0.1)
            sar_data[1] = -18.0 + (t_idx * 0.05)
            sar_data[2] = sar_data[0] - sar_data[1]

            with rasterio.open(sar_path, "w", **profile) as dst:
                dst.write(sar_data)

            # Synthetic Labels: 0 = nonwater, 1 = water, 255 = nodata
            label_data = np.zeros((h, w), dtype=np.uint8)
            label_data[0:5, :] = 1
            label_data[5:25, :] = 0
            label_data[25:, :] = 255

            with rasterio.open(label_path, "w", **profile_lbl) as dst:
                dst.write(label_data, 1)

            tiles_info.append({
                "tile_id": tile_id,
                "event_id": ev,
                "sar_path": str(sar_path),
                "label_path": str(label_path),
                "width": w,
                "height": h
            })

    train_manifest_path = tmp_path / "mock_train_manifest.json"
    manifest_data = {
        "event_ids": events,
        "tiles": tiles_info
    }
    with open(train_manifest_path, "w") as f:
        json.dump(manifest_data, f)

    return str(train_manifest_path), tiles_info


def test_event_grouped_oof_prediction_generation(mock_multi_event_dataset):
    train_manifest_path, tiles_info = mock_multi_event_dataset

    cfg = StackingConfig(
        train_manifest_path=train_manifest_path,
        samples_per_tile_water=50,
        samples_per_tile_nonwater=100,
        random_state=42
    )
    # Configure base models for fast test run
    cfg.rf_config.n_estimators = 5
    cfg.rf_config.max_depth = 3
    cfg.xgb_config.n_estimators = 5
    cfg.xgb_config.max_depth = 3

    X_oof, y_oof, oof_stats = generate_event_grouped_oof_predictions(train_manifest_path, cfg)

    assert oof_stats["num_events"] == 3
    assert X_oof.shape[1] == 2
    assert len(X_oof) == len(y_oof)

    # Probabilities should be bounded between 0.0 and 1.0
    assert np.all(X_oof >= 0.0) and np.all(X_oof <= 1.0)
    # Ensure no label 255 is present in y_oof
    assert 255 not in set(np.unique(y_oof))


def test_stacking_meta_model_training(mock_multi_event_dataset):
    train_manifest_path, _ = mock_multi_event_dataset

    cfg = StackingConfig(
        train_manifest_path=train_manifest_path,
        samples_per_tile_water=50,
        samples_per_tile_nonwater=100,
        random_state=42
    )
    cfg.rf_config.n_estimators = 5
    cfg.xgb_config.n_estimators = 5

    X_oof, y_oof, _ = generate_event_grouped_oof_predictions(train_manifest_path, cfg)
    meta_model, fit_stats = train_stacking_meta_model(X_oof, y_oof, cfg)

    assert isinstance(meta_model, LogisticRegression)
    assert "rf_probability_weight" in fit_stats
    assert "xgb_probability_weight" in fit_stats


def test_full_stacking_pipeline_eval_and_export(mock_multi_event_dataset, tmp_path):
    train_manifest_path, tiles_info = mock_multi_event_dataset

    cfg = StackingConfig(
        train_manifest_path=train_manifest_path,
        experiment_dir=str(tmp_path / "exp_stacking"),
        samples_per_tile_water=50,
        samples_per_tile_nonwater=100,
        random_state=42
    )
    cfg.rf_config.n_estimators = 5
    cfg.xgb_config.n_estimators = 5

    pipeline, pipeline_summary = train_full_stacking_pipeline(train_manifest_path, cfg)
    assert isinstance(pipeline, StackingPipeline)

    # Mock val manifest using first tile of EventA
    val_manifest_file = tmp_path / "mock_val_manifest.json"
    val_data = {
        "event_ids": ["Somalia"],
        "tiles": [{
            "tile_id": "Somalia_mock_stacking",
            "event_id": "Somalia",
            "sar_path": tiles_info[0]["sar_path"],
            "label_path": tiles_info[0]["label_path"],
            "width": 32,
            "height": 32
        }]
    }
    with open(val_manifest_file, "w") as f:
        json.dump(val_data, f)

    overall_metrics, per_tile = evaluate_stacking_pipeline(
        pipeline, str(val_manifest_file), cfg, export_predictions=True
    )

    pred_tif = tmp_path / "exp_stacking" / "predictions" / "Somalia_mock_stacking_pred.tif"
    assert pred_tif.exists()

    with rasterio.open(pred_tif) as src:
        assert src.shape == (32, 32)
        assert src.count == 1
        assert src.dtypes[0] == "uint8"
        assert src.nodata == 255
        assert src.crs.to_string() == "EPSG:4326"

        pred_data = src.read(1)
        # Nodata pixels (index 25:) must be 255
        assert np.all(pred_data[25:, :] == 255)
        # Valid prediction values must be 0 or 1
        assert set(np.unique(pred_data[:25, :])).issubset({0, 1})
