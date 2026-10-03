"""Unit tests for Small U-Net primary model adoption and configuration."""

import os
import json
import torch
import pytest
from ml.model import SmallUNet
from ml.config import load_primary_model_config, TrainingPaths


def test_primary_model_config_file_exists():
    """Verify that config/primary_segmentation_model.json exists and loads correctly."""
    config_path = "config/primary_segmentation_model.json"
    assert os.path.exists(config_path), f"Config file missing: {config_path}"
    
    config_data = load_primary_model_config(config_path)
    assert config_data["model_name"] == "Small U-Net"
    assert config_data["architecture"]["class_name"] == "SmallUNet"
    assert config_data["architecture"]["checkpoint_path"] == "experiments/unet/best_model.pt"
    assert config_data["input_specification"]["num_channels"] == 3
    assert config_data["input_specification"]["channels"] == ["VV_dB", "VH_dB", "VV_dB - VH_dB"]
    assert config_data["inference_parameters"]["probability_threshold"] == 0.50
    assert config_data["inference_parameters"]["nodata_value"] == 255


def test_primary_model_checkpoint_loading_and_inference():
    """Verify loading state_dict from best_model.pt into SmallUNet and running forward pass."""
    checkpoint_path = "experiments/unet/best_model.pt"
    assert os.path.exists(checkpoint_path), f"Checkpoint missing: {checkpoint_path}"
    
    model = SmallUNet(in_channels=3, out_channels=1, init_features=32)
    
    # Check total trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_params == 1927009, f"Parameter count mismatch: {num_params} vs expected 1927009"
    
    # Load state dict (checkpoint contains dictionary with model_state_dict)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    
    # Dummy input forward pass (1, 3, 256, 256)
    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        logits = model(dummy_input)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.50).float()
        
    assert logits.shape == (1, 1, 256, 256)
    assert probs.shape == (1, 1, 256, 256)
    assert preds.shape == (1, 1, 256, 256)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_primary_model_training_paths():
    """Verify TrainingPaths default best_model_path points to experiments/unet/best_model.pt."""
    paths = TrainingPaths()
    assert paths.best_model_path == "experiments/unet/best_model.pt"
    assert os.path.exists(paths.best_model_path)


def test_primary_model_somalia_validation_metrics_recorded():
    """Verify reported Somalia validation metrics in configuration."""
    config_data = load_primary_model_config()
    somalia_metrics = config_data["somalia_validation_results"]["water_metrics"]
    
    assert round(somalia_metrics["precision"], 4) == 0.9161
    assert round(somalia_metrics["recall"], 4) == 0.9457
    assert round(somalia_metrics["f1_dice"], 4) == 0.9306
    assert round(somalia_metrics["iou"], 4) == 0.8703
