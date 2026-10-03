"""Configuration dataclass for SARA Experiment 6: Controlled SegFormer-B0 Retraining."""

import json
from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class SegFormerRetrainConfig:
    """Experiment configuration for controlled SegFormer-B0 retraining."""
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
    warmup_epochs: int = 3
    weight_decay: float = 0.0001
    pos_weight: float = 3.0
    bce_weight: float = 0.5
    max_epochs: int = 30
    early_stopping_patience: int = 8
    checkpoint_metric: str = "val_iou"
    use_mixed_precision: bool = True
    fixed_seed: int = 42
    train_manifest_path: str = "data/ml/train_manifest.json"
    val_manifest_path: str = "data/ml/val_manifest.json"
    test_manifest_path: str = "data/ml/test_manifest.json"
    experiment_dir: str = "experiments/segformer_retrain"
    best_model_name: str = "best_model.pt"
    metrics_name: str = "retraining_metrics.json"
    history_csv_name: str = "training_history.csv"
    config_json_name: str = "config.json"
    disclaimer: str = (
        "SARA Controlled SegFormer-B0 Retraining. Fitted on train_manifest.json with Linear Warmup + Cosine Decay "
        "and pos_weight=3.0. Validated on Somalia partition. Mekong test set and EMSR927 remain strictly isolated."
    )

    def model_dump(self) -> dict:
        return asdict(self)

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.model_dump(), f, indent=2)
