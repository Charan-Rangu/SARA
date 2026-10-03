"""SARA Random Forest Baseline Model Configuration.

Defines hyperparameters, controlled tile-based sampling parameters, feature definitions,
and output experiment paths for pixel-level Sentinel-1 flood segmentation.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class RandomForestConfig(BaseModel):
    """Hyperparameters and dataset sampling config for Random Forest flood segmentation."""

    # Model Hyperparameters (scikit-learn RandomForestClassifier)
    n_estimators: int = 100
    max_depth: Optional[int] = 15
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    max_features: str = "sqrt"
    random_state: int = 42
    n_jobs: int = -1
    class_weight: str = "balanced"

    # Controlled Tile-Stratified Sampling
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
    experiment_dir: str = "experiments/rf_baseline"
    model_save_name: str = "rf_baseline_model.joblib"
    metrics_save_name: str = "val_metrics.json"

    # Safety & Isolation Notice
    disclaimer: str = (
        "SARA Random Forest Baseline model. Trained solely on train_manifest.json. "
        "Evaluated on Somalia validation tiles only. Mekong test set and EMSR927 validation "
        "assets are strictly isolated and not touched."
    )
