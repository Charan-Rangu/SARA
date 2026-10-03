"""Configuration dataclass for SARA Experiment 6: SegFormer-B0 Flood Segmentation."""

import json
from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class SegFormerExperimentConfig:
    """Experiment configuration for SegFormer-B0 baseline."""
    in_channels: int = 3
    out_channels: int = 1
    embedding_dim: int = 256
    input_size: int = 256
    nodata_value: int = 255
    channel_means: List[float] = field(default_factory=lambda: [-11.0, -18.0, 7.0])
    channel_stds: List[float] = field(default_factory=lambda: [3.0, 3.5, 2.5])
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    max_epochs: int = 30
    early_stopping_patience: int = 8
    checkpoint_metric: str = "val_iou"
    use_mixed_precision: bool = True
    fixed_seed: int = 42
    train_manifest_path: str = "data/ml/train_manifest.json"
    val_manifest_path: str = "data/ml/val_manifest.json"
    test_manifest_path: str = "data/ml/test_manifest.json"
    experiment_dir: str = "experiments/segformer_b0"
    best_model_name: str = "best_model.pt"
    metrics_name: str = "segformer_metrics.json"
    history_csv_name: str = "training_history.csv"
    config_json_name: str = "segformer_config.json"
    disclaimer: str = (
        "SARA SegFormer-B0 model. Trained strictly on train_manifest.json using PyTorch CUDA AMP FP16. "
        "Somalia validation partition used for early stopping and checkpoint selection. "
        "Mekong test set and EMSR927 validation assets remain strictly isolated."
    )

    def model_dump(self) -> dict:
        return asdict(self)

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.model_dump(), f, indent=2)
