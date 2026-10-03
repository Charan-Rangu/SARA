"""Unit tests for SARA Experiment 4: Small U-Net Flood Segmentation module."""

import os
import json
import pytest
import torch
import torch.nn as nn
import numpy as np
import rasterio
from rasterio.transform import from_origin

from ml.model import SmallUNet, MaskedBCEDiceLoss, calculate_segmentation_metrics
from ml.unet_config import UNetExperimentConfig
from ml.unet_dataset import SARAFloodDataset, create_dataloader
from ml.unet_train import train_unet_epoch, evaluate_unet_val_epoch, save_atomic_checkpoint, set_seed


@pytest.fixture
def mock_unet_dataset(tmp_path):
    """Creates mock dataset manifest and GeoTIFF tiles for U-Net testing."""
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

    profile_lbl = profile.copy()
    profile_lbl["count"] = 1
    profile_lbl["dtype"] = "uint8"
    profile_lbl["nodata"] = 255

    sar_path = tmp_path / "mock_unet_sar.tif"
    lbl_path = tmp_path / "mock_unet_lbl.tif"

    sar_data = np.zeros((3, h, w), dtype=np.float32)
    sar_data[0] = -10.0
    sar_data[1] = -18.0
    sar_data[2] = 8.0

    with rasterio.open(sar_path, "w", **profile) as dst:
        dst.write(sar_data)

    lbl_data = np.zeros((h, w), dtype=np.uint8)
    lbl_data[0:10, :] = 1
    lbl_data[10:50, :] = 0
    lbl_data[50:, :] = 255

    with rasterio.open(lbl_path, "w", **profile_lbl) as dst:
        dst.write(lbl_data, 1)

    manifest_path = tmp_path / "mock_unet_manifest.json"
    manifest_data = {
        "event_ids": ["MockEvent"],
        "tiles": [{
            "tile_id": "mock_unet_001",
            "event_id": "MockEvent",
            "sar_path": str(sar_path),
            "label_path": str(lbl_path),
            "width": w,
            "height": h
        }]
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f)

    return str(manifest_path)


def test_unet_dataset_loader_shapes(mock_unet_dataset):
    manifest_path = mock_unet_dataset
    dataset = SARAFloodDataset(manifest_path)

    assert len(dataset) == 1
    image, label, tile_id = dataset[0]

    assert image.shape == (3, 64, 64)
    assert label.shape == (1, 64, 64)
    assert image.dtype == torch.float32
    assert label.dtype == torch.int64
    assert tile_id == "mock_unet_001"


def test_small_unet_architecture_output_shape():
    model = SmallUNet(in_channels=3, out_channels=1, init_features=16)
    x = torch.randn(2, 3, 64, 64)
    logits = model(x)

    assert logits.shape == (2, 1, 64, 64)
    assert not torch.isnan(logits).any()


def test_masked_bce_dice_loss_nodata_exclusion():
    criterion = MaskedBCEDiceLoss(nodata_value=255)
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)

    # All targets are nodata (255)
    nodata_targets = torch.full((2, 1, 32, 32), 255, dtype=torch.long)
    loss_nodata = criterion(logits, nodata_targets)
    assert loss_nodata.item() == 0.0

    # Valid targets (0 and 1) with some 255 nodata
    valid_targets = torch.zeros((2, 1, 32, 32), dtype=torch.long)
    valid_targets[:, :, 0:5, :] = 1
    valid_targets[:, :, 25:, :] = 255

    loss_valid = criterion(logits, valid_targets)
    assert loss_valid.item() > 0.0
    loss_valid.backward()
    assert logits.grad is not None


def test_atomic_checkpoint_saving_and_resumption(tmp_path):
    filepath = tmp_path / "model_ckpt.pt"
    dummy_state = {"epoch": 5, "state_dict": {"weight": torch.tensor([1.0, 2.0])}}

    save_atomic_checkpoint(dummy_state, str(filepath))
    assert filepath.exists()

    loaded = torch.load(filepath, weights_only=False)
    assert loaded["epoch"] == 5
    assert torch.equal(loaded["state_dict"]["weight"], torch.tensor([1.0, 2.0]))


def test_unet_seed_reproducibility():
    set_seed(42)
    m1 = SmallUNet(3, 1, 16)
    w1 = m1.inc.double_conv[0].weight.clone()

    set_seed(42)
    m2 = SmallUNet(3, 1, 16)
    w2 = m2.inc.double_conv[0].weight.clone()

    assert torch.equal(w1, w2)


def test_unet_artifact_preservation():
    assert os.path.exists("experiments/rf_baseline/rf_baseline_model.joblib")
    assert os.path.exists("experiments/rf_baseline/val_metrics.json")
    assert os.path.exists("experiments/xgboost_baseline/xgb_baseline_model.json")
    assert os.path.exists("experiments/xgboost_baseline/val_metrics.json")
    assert os.path.exists("experiments/stacking/stacking_model.joblib")
    assert os.path.exists("experiments/stacking/stacking_metrics.json")
    assert os.path.exists("experiments/unet/best_model.pt")
    assert os.path.exists("experiments/unet/unet_metrics.json")


def test_unet_prediction_geotiff_properties():
    pred_path = "experiments/unet/predictions/Somalia_028_pred.tif"
    assert os.path.exists(pred_path)

    with rasterio.open(pred_path) as src:
        assert src.width == 256
        assert src.height == 256
        assert src.count == 1
        assert src.dtypes[0] == "uint8"
        assert src.nodata == 255.0
        data = src.read(1)
        assert set(np.unique(data)).issubset({0, 1, 255})
