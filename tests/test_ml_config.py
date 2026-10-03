"""Tests for SARA ML Model Configuration, Hardware Specs, and Training Paths."""

import pytest
import os
import json
from ml.config import ModelConfig, TrainingPaths


def test_model_config_defaults():
    cfg = ModelConfig()
    assert cfg.model_type == "SmallUNet"
    assert cfg.in_channels == 3
    assert cfg.input_size == 256
    assert cfg.batch_size == 4
    assert cfg.use_mixed_precision is True
    assert cfg.fixed_seed == 42
    assert cfg.nodata_value == 255
    assert cfg.checkpoint_metric == "val_iou"


def test_training_paths_and_manifest_existence():
    paths = TrainingPaths()
    assert os.path.exists(paths.train_manifest)
    assert os.path.exists(paths.val_manifest)
    assert os.path.exists(paths.test_manifest)

    # Verify JSON manifests can be parsed
    with open(paths.train_manifest, "r", encoding="utf-8") as f:
        train_data = json.load(f)
        assert "tiles" in train_data
        assert len(train_data["tiles"]) > 0

    with open(paths.val_manifest, "r", encoding="utf-8") as f:
        val_data = json.load(f)
        assert "tiles" in val_data
        assert len(val_data["tiles"]) > 0

    with open(paths.test_manifest, "r", encoding="utf-8") as f:
        test_data = json.load(f)
        assert "tiles" in test_data
        assert len(test_data["tiles"]) > 0
