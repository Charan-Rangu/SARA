"""Unit tests for SARA Experiment 6: SegFormer-B0 Flood Segmentation module."""

import os
import json
import pytest
import torch
import torch.nn as nn
import numpy as np
import rasterio
from rasterio.transform import from_origin

from ml.model import MaskedBCEDiceLoss, calculate_segmentation_metrics
from ml.segformer_model import SegFormerB0
from ml.segformer_config import SegFormerExperimentConfig
from ml.segformer_train import save_atomic_checkpoint, set_seed


def test_segformer_input_compatibility_and_output_shape():
    """Test 3-channel input compatibility and 1-channel logit output shape."""
    model = SegFormerB0(in_channels=3, out_channels=1, embedding_dim=256)
    x = torch.randn(2, 3, 256, 256)
    logits = model(x)

    assert logits.shape == (2, 1, 256, 256)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_segformer_finite_loss_and_gradients():
    """Test forward pass, masked loss calculation, and backward gradient computation."""
    model = SegFormerB0(in_channels=3, out_channels=1, embedding_dim=256)
    criterion = MaskedBCEDiceLoss(nodata_value=255)

    x = torch.randn(2, 3, 64, 64)
    y = torch.zeros((2, 1, 64, 64), dtype=torch.long)
    y[:, :, 0:10, :] = 1
    y[:, :, 50:, :] = 255

    logits = model(x)
    loss = criterion(logits, y)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all()


def test_segformer_nodata_exclusion():
    """Test that all-nodata targets (label 255) return zero loss."""
    criterion = MaskedBCEDiceLoss(nodata_value=255)
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)

    nodata_targets = torch.full((2, 1, 32, 32), 255, dtype=torch.long)
    loss_nodata = criterion(logits, nodata_targets)

    assert loss_nodata.item() == 0.0


def test_segformer_atomic_checkpoint(tmp_path):
    """Test atomic checkpoint saving and resumption."""
    filepath = tmp_path / "segformer_ckpt.pt"
    dummy_state = {"epoch": 4, "state_dict": {"weight": torch.tensor([3.0, 4.0])}}

    save_atomic_checkpoint(dummy_state, str(filepath))
    assert filepath.exists()

    loaded = torch.load(filepath, weights_only=False)
    assert loaded["epoch"] == 4
    assert torch.equal(loaded["state_dict"]["weight"], torch.tensor([3.0, 4.0]))


def test_segformer_prediction_raster_metadata():
    """Test metadata compliance of exported SegFormer prediction GeoTIFFs."""
    pred_path = "experiments/segformer_b0/predictions/Somalia_028_pred.tif"
    assert os.path.exists(pred_path)

    with rasterio.open(pred_path) as src:
        assert src.width == 256
        assert src.height == 256
        assert src.count == 1
        assert src.dtypes[0] == "uint8"
        assert src.nodata == 255.0
        data = src.read(1)
        assert set(np.unique(data)).issubset({0, 1, 255})


def test_segformer_cpu_inference():
    """Test SegFormer-B0 inference on CPU."""
    model = SegFormerB0(in_channels=3, out_channels=1, embedding_dim=256)
    model.eval()

    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (1, 1, 256, 256)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_segformer_cuda_inference():
    """Test SegFormer-B0 inference on CUDA GPU when available."""
    device = torch.device("cuda")
    model = SegFormerB0(in_channels=3, out_channels=1, embedding_dim=256).to(device)
    model.eval()

    x = torch.randn(1, 3, 256, 256, device=device)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (1, 1, 256, 256)
    assert logits.is_cuda
