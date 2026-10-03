"""Unit tests for SARA Experiment 5: Final Four-Model Benchmark module."""

import os
import json
import pytest
import rasterio
import numpy as np


def test_final_benchmark_files_exist():
    """Verify all required benchmark outputs exist in experiments/final_benchmark/."""
    base_dir = "experiments/final_benchmark"
    assert os.path.exists(os.path.join(base_dir, "sara_final_benchmark_report.md"))
    assert os.path.exists(os.path.join(base_dir, "final_metrics.json"))
    assert os.path.exists(os.path.join(base_dir, "artifact_inventory.json"))
    assert os.path.exists(os.path.join(base_dir, "isolation_audit.md"))


def test_final_benchmark_metrics_structure():
    """Verify final_metrics.json contains Somalia and Mekong results for all 4 models."""
    metrics_path = "experiments/final_benchmark/final_metrics.json"
    assert os.path.exists(metrics_path)
    
    with open(metrics_path, "r") as f:
        data = json.load(f)
        
    assert "benchmark_summary" in data
    assert "somalia_validation" in data
    assert "mekong_test" in data
    
    models = ["rf", "xgb", "stacking", "unet"]
    for m in models:
        assert m in data["somalia_validation"]
        assert m in data["mekong_test"]
        
        # Verify metric values are within valid probability/ratio ranges [0.0, 1.0]
        som_m = data["somalia_validation"][m]["overall"]["water_metrics"]
        mek_m = data["mekong_test"][m]["overall"]["water_metrics"]
        
        for metric_key in ["precision", "recall", "f1_dice", "iou"]:
            assert 0.0 <= som_m[metric_key] <= 1.0
            assert 0.0 <= mek_m[metric_key] <= 1.0


def test_final_benchmark_isolation_verification():
    """Verify zero manifest overlap across train, val, and test partitions."""
    with open("data/ml/train_manifest.json", "r") as f:
        train_m = json.load(f)
    with open("data/ml/val_manifest.json", "r") as f:
        val_m = json.load(f)
    with open("data/ml/test_manifest.json", "r") as f:
        test_m = json.load(f)
        
    train_events = set(train_m["event_ids"])
    val_events = set(val_m["event_ids"])
    test_events = set(test_m["event_ids"])
    
    assert len(train_events.intersection(val_events)) == 0
    assert len(train_events.intersection(test_events)) == 0
    assert len(val_events.intersection(test_events)) == 0


def test_final_benchmark_prediction_geotiffs():
    """Verify exported Mekong prediction GeoTIFF spatial properties."""
    pred_dir = "experiments/final_benchmark/predictions/rf"
    assert os.path.exists(pred_dir)
    
    pred_path = os.path.join(pred_dir, "Mekong_012_pred.tif")
    assert os.path.exists(pred_path)
    
    with rasterio.open(pred_path) as src:
        assert src.width == 256
        assert src.height == 256
        assert src.count == 1
        assert src.dtypes[0] == "uint8"
        assert src.nodata == 255.0
        data = src.read(1)
        assert set(np.unique(data)).issubset({0, 1, 255})
