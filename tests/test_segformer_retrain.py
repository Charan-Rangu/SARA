"""Unit tests for SARA Controlled SegFormer-B0 Retraining module."""

import os
import json
import pytest
import rasterio
import numpy as np


def test_segformer_retrain_artifacts_exist():
    """Verify all required retraining outputs exist in experiments/segformer_retrain/."""
    base_dir = "experiments/segformer_retrain"
    assert os.path.exists(os.path.join(base_dir, "retraining_report.md"))
    assert os.path.exists(os.path.join(base_dir, "sara_segformer_retrain_independent_audit.md"))
    assert os.path.exists(os.path.join(base_dir, "retraining_metrics.json"))
    assert os.path.exists(os.path.join(base_dir, "config.json"))
    assert os.path.exists(os.path.join(base_dir, "best_model.pt"))
    assert os.path.exists(os.path.join(base_dir, "training_history.csv"))
    assert os.path.exists(os.path.join(base_dir, "predictions"))


def test_segformer_retrain_metrics_structure():
    """Verify retraining_metrics.json contains baseline reproduction, train summary, and fair comparison."""
    metrics_path = "experiments/segformer_retrain/retraining_metrics.json"
    assert os.path.exists(metrics_path)

    with open(metrics_path, "r") as f:
        data = json.load(f)

    assert "baseline_reproduced" in data
    assert "train_summary" in data
    assert "overall_metrics" in data
    assert "fair_comparison" in data
    assert "original_segformer" in data["fair_comparison"]
    assert "retrained_segformer" in data["fair_comparison"]
    assert "small_unet" in data["fair_comparison"]


def test_segformer_retrain_geotiff_metadata():
    """Verify prediction raster metadata for retrained SegFormer-B0."""
    pred_path = "experiments/segformer_retrain/predictions/Somalia_028_pred.tif"
    assert os.path.exists(pred_path)

    with rasterio.open(pred_path) as src:
        assert src.width == 256
        assert src.height == 256
        assert src.count == 1
        assert src.dtypes[0] == "uint8"
        assert src.nodata == 255.0
        data = src.read(1)
        assert set(np.unique(data)).issubset({0, 1, 255})
