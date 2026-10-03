"""SARA Machine Learning Model Training Configuration.

Defines reproducible hyperparameters, architecture specs, normalization parameters,
and loss/metric criteria optimized for an NVIDIA RTX 2050 GPU (4 GB VRAM) and 16 GB RAM.

Important:
This configuration specifies model hyperparameters for pre-training on open SAR datasets (Sen1Floods11).
EMSR927 validation data MUST NOT be loaded or used for training/tuning.
"""

from typing import Optional, Any
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Architecture and hyperparameter configuration for SARA SAR flood segmentation."""

    # Model Architecture
    model_type: str = "SmallUNet"
    in_channels: int = 3  # [VV_dB, VH_dB, VV_VH_ratio]
    out_channels: int = 1 # Binary water probability mask
    init_features: int = 32 # Base feature channels (32 -> 64 -> 128 -> 256)

    # Input Data Specifications
    input_size: int = 256  # 256x256 image tiles
    nodata_value: int = 255
    target_class: int = 1  # Water class

    # Per-channel Normalization Parameters (Mean and Std in dB scale)
    channel_means: list[float] = Field(default_factory=lambda: [-11.0, -18.0, 7.0])
    channel_stds: list[float] = Field(default_factory=lambda: [3.0, 3.5, 2.5])

    # Hardware & Optimization (RTX 2050 4GB VRAM)
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    use_mixed_precision: bool = True  # FP16 Automatic Mixed Precision (AMP)
    fixed_seed: int = 42

    # Training Control
    max_epochs: int = 50
    early_stopping_patience: int = 10
    checkpoint_metric: str = "val_iou"  # Save checkpoint on highest validation IoU

    # Metrics
    tracked_metrics: list[str] = Field(
        default_factory=lambda: ["precision", "recall", "f1_dice", "iou"]
    )

    # Disclaimers
    disclaimer: str = (
        "SARA segmentation model configuration. Model predictions represent probabilistic SAR water extent. "
        "Outputs require contextual validation and must NOT be used directly as operational life-safety decisions."
    )


class TrainingPaths(BaseModel):
    """File system paths for training inputs, manifests, and checkpoints."""

    train_manifest: str = "data/ml/train_manifest.json"
    val_manifest: str = "data/ml/val_manifest.json"
    test_manifest: str = "data/ml/test_manifest.json"
    checkpoint_dir: str = "experiments/unet"
    log_dir: str = "experiments/unet"
    best_model_path: str = "experiments/unet/best_model.pt"


def load_primary_model_config(config_path: str = "config/primary_segmentation_model.json") -> dict[str, Any]:
    """Load the primary segmentation model configuration file."""
    import os
    import json
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Primary model configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

