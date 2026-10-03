"""Tests for SmallUNet PyTorch architecture, MaskedBCEDiceLoss, and segmentation metrics."""

import pytest
import torch
from ml.model import SmallUNet, MaskedBCEDiceLoss, calculate_segmentation_metrics


def test_small_unet_forward_pass_dimensions():
    model = SmallUNet(in_channels=3, out_channels=1, init_features=32)
    dummy_input = torch.randn(4, 3, 256, 256)
    output = model(dummy_input)

    assert output.shape == (4, 1, 256, 256)
    assert not torch.isnan(output).any()


def test_masked_bce_dice_loss_nodata_handling():
    criterion = MaskedBCEDiceLoss(nodata_value=255)
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)

    targets = torch.zeros(2, 1, 64, 64, dtype=torch.uint8)
    targets[:, :, :32, :] = 1
    targets[:, :, 32:48, :] = 255  # Masked nodata

    loss = criterion(logits, targets)
    assert loss > 0.0
    assert torch.isfinite(loss)

    # Gradient backward test
    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_segmentation_metrics_calculation():
    logits = torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]]]])  # High confidence preds: [[1, 0], [0, 1]]
    targets = torch.tensor([[[[1, 0], [0, 255]]]], dtype=torch.uint8)  # Last pixel is nodata

    metrics = calculate_segmentation_metrics(logits, targets, nodata_value=255)
    assert metrics["precision"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["recall"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["iou"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["f1_dice"] == pytest.approx(1.0, abs=1e-4)
