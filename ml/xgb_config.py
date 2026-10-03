"""SARA XGBoost Baseline Model Configuration.

Defines hyperparameters, tile-stratified sampling parameters, feature definitions,
and output experiment paths for pixel-level Sentinel-1 flood segmentation.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class XGBoostConfig(BaseModel):
    """Hyperparameters and dataset sampling config for XGBoost flood segmentation."""

    # Model Hyperparameters (xgboost.XGBClassifier)
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    scale_pos_weight: float = 1.0
    random_state: int = 42
    tree_method: str = "hist"
    device: str = "cpu"
    n_jobs: int = -1
    eval_metric: str = "logloss"

    # Controlled Tile-Stratified Sampling (aligned with Random Forest baseline)
    samples_per_tile_water: int = 1000
    samples_per_tile_nonwater: int = 2000
    max_total_train_samples: int = 150000

    # Feature Definitions & Nodata Encoding
    feature_names: List[str] = Field(
        default_factory=lambda: ["VV_dB", "VH_dB", "VV_VH_ratio"]
    )
    nodata_value: int = 255

    # Path Settings
    train_manifest_path: str = "data/ml/train_manifest.json"
    val_manifest_path: str = "data/ml/val_manifest.json"
    test_manifest_path: str = "data/ml/test_manifest.json"
    experiment_dir: str = "experiments/xgboost_baseline"
    model_save_name: str = "xgb_baseline_model.json"
    metrics_save_name: str = "val_metrics.json"

    # Safety & Isolation Notice
    disclaimer: str = (
        "SARA XGBoost Baseline model. Trained solely on train_manifest.json. "
        "Evaluated on Somalia validation tiles only. Mekong test set and EMSR927 validation "
        "assets are strictly isolated and not touched."
    )
