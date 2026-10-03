"""SARA Small U-Net Model & Training Configuration for Experiment 4."""

from typing import List, Optional
from pydantic import BaseModel, Field


class UNetExperimentConfig(BaseModel):
    """Hyperparameters and configuration for PyTorch Small U-Net flood segmentation."""

    # Model Architecture Specs
    in_channels: int = 3
    out_channels: int = 1
    init_features: int = 32  # 32 -> 64 -> 128 -> 256 feature channels

    # Data Specifications
    input_size: int = 256
    nodata_value: int = 255
    channel_means: List[float] = Field(default_factory=lambda: [-11.0, -18.0, 7.0])
    channel_stds: List[float] = Field(default_factory=lambda: [3.0, 3.5, 2.5])

    # Training Control Parameters
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 30
    early_stopping_patience: int = 8
    checkpoint_metric: str = "val_iou"  # Save checkpoint on highest validation IoU
    use_mixed_precision: bool = True    # PyTorch AMP FP16
    fixed_seed: int = 42

    # Paths
    train_manifest_path: str = "data/ml/train_manifest.json"
    val_manifest_path: str = "data/ml/val_manifest.json"
    test_manifest_path: str = "data/ml/test_manifest.json"
    experiment_dir: str = "experiments/unet"
    best_model_name: str = "best_model.pt"
    last_model_name: str = "last_checkpoint.pt"
    metrics_name: str = "unet_metrics.json"
    history_csv_name: str = "training_history.csv"
    config_json_name: str = "unet_config.json"

    # Safeguard Notice
    disclaimer: str = (
        "SARA Small U-Net model. Trained strictly on train_manifest.json using PyTorch CUDA AMP. "
        "Somalia validation partition used for early stopping and checkpoint selection. "
        "Mekong test set and EMSR927 validation assets are strictly isolated."
    )
