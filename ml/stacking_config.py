"""SARA Stacking Ensemble Model Configuration.

Defines hyperparameters, event-grouped cross-validation settings, meta-model parameters,
and output experiment paths for the leakage-safe Stacking Ensemble.
"""

from typing import Optional, List
from pydantic import BaseModel, Field
from ml.rf_config import RandomForestConfig
from ml.xgb_config import XGBoostConfig


class StackingConfig(BaseModel):
    """Configuration container for Stacking Ensemble pipeline."""

    # Base Model Configurations
    rf_config: RandomForestConfig = Field(default_factory=RandomForestConfig)
    xgb_config: XGBoostConfig = Field(default_factory=XGBoostConfig)

    # Meta-Model Hyperparameters (scikit-learn LogisticRegression)
    meta_model_type: str = "LogisticRegression"
    C: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 1000
    random_state: int = 42
    class_weight: str = "balanced"

    # Cross-Validation & Sampling Parameters
    num_event_folds: int = 9
    samples_per_tile_water: int = 1000
    samples_per_tile_nonwater: int = 2000
    max_total_train_samples: int = 150000

    # Path Settings
    train_manifest_path: str = "data/ml/train_manifest.json"
    val_manifest_path: str = "data/ml/val_manifest.json"
    test_manifest_path: str = "data/ml/test_manifest.json"
    experiment_dir: str = "experiments/stacking"
    model_save_name: str = "stacking_model.joblib"
    metrics_save_name: str = "stacking_metrics.json"

    # Nodata Encoding
    nodata_value: int = 255

    # Safety & Isolation Notice
    disclaimer: str = (
        "SARA Stacking Ensemble model. Base models trained on 9 event-grouped folds of train_manifest.json. "
        "Meta-model trained strictly on Out-of-Fold (OOF) base model probability predictions. "
        "Somalia validation set used solely for post-training evaluation. "
        "Mekong test set and EMSR927 validation assets are strictly isolated and not touched."
    )
