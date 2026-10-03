"""Tests for SARA ML Dataset Preparation, Provenance, and Leakage-Resistant Splits."""

import pytest
from ml.dataset import DatasetMetadata, SplitManifest, generate_leakage_resistant_splits


def test_dataset_metadata_defaults():
    meta = DatasetMetadata()
    assert meta.dataset_name == "Sen1Floods11 (SAR Hand-Labeled Subset)"
    assert meta.license == "CC BY 4.0"
    assert meta.patch_size == 256
    assert meta.label_encoding["water"] == 1
    assert meta.label_encoding["invalid_nodata"] == 255


def test_leakage_resistant_splits_no_overlap():
    mock_tiles = [
        {"tile_id": f"ghana_{i}", "event_id": "Ghana", "path": f"tiles/ghana_{i}.tif"} for i in range(10)
    ] + [
        {"tile_id": f"india_{i}", "event_id": "India", "path": f"tiles/india_{i}.tif"} for i in range(10)
    ] + [
        {"tile_id": f"spain_{i}", "event_id": "Spain", "path": f"tiles/spain_{i}.tif"} for i in range(10)
    ] + [
        {"tile_id": f"bolivia_{i}", "event_id": "Bolivia", "path": f"tiles/bolivia_{i}.tif"} for i in range(10)
    ] + [
        {"tile_id": f"mekong_{i}", "event_id": "Mekong", "path": f"tiles/mekong_{i}.tif"} for i in range(10)
    ]

    manifest = generate_leakage_resistant_splits(mock_tiles, val_ratio=0.2, test_ratio=0.2, seed=42)

    assert manifest.total_tiles == 50
    assert manifest.train_count > 0
    assert manifest.val_count > 0
    assert manifest.test_count > 0

    # Ensure ZERO overlap between train, val, and test events
    train_events_set = set(manifest.train_events)
    val_events_set = set(manifest.val_events)
    test_events_set = set(manifest.test_events)

    assert len(train_events_set.intersection(val_events_set)) == 0
    assert len(train_events_set.intersection(test_events_set)) == 0
    assert len(val_events_set.intersection(test_events_set)) == 0

    # Ensure total tiles sum up
    assert manifest.train_count + manifest.val_count + manifest.test_count == 50
