"""SARA Experiment 2: XGBoost Baseline for Sentinel-1 Flood Segmentation.

Implements:
1. Pixel-level gradient boosted decision tree classifier using XGBoost (XGBClassifier).
2. Reuse of tile-stratified sampling and feature extraction from ml/rf_baseline.py.
3. 3-band input features: VV_dB, VH_dB, VV_VH_ratio (VV_dB - VH_dB).
4. Strict evaluation on Somalia validation manifest (4 tiles) ignoring nodata label 255.
5. Export of prediction rasters preserving geospatial CRS, transform, and nodata metadata.
6. Resource-bounded CPU execution for 16 GB RAM systems.

Important:
Mekong test set (test_manifest.json) and EMSR927 validation assets are strictly isolated
and MUST NOT be loaded or used during fitting, tuning, or threshold selection.
"""

import os
import json
import time
import numpy as np
import rasterio
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import xgboost as xgb

try:
    import psutil
except ImportError:
    psutil = None

from ml.xgb_config import XGBoostConfig
from ml.rf_baseline import (
    extract_tile_features_and_labels,
    sample_tile_pixels,
    load_training_dataset,
    calculate_rf_metrics
)


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: XGBoostConfig
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """Train XGBClassifier on pixel features with resource timing and feature importance logging."""
    start_time = time.time()
    if psutil is not None:
        process = psutil.Process(os.getpid())
        mem_before_mb = process.memory_info().rss / (1024 * 1024)
    else:
        mem_before_mb = 0.0

    clf = xgb.XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        scale_pos_weight=config.scale_pos_weight,
        random_state=config.random_state,
        tree_method=config.tree_method,
        device=config.device,
        n_jobs=config.n_jobs,
        eval_metric=config.eval_metric
    )

    clf.fit(X_train, y_train)

    duration_sec = time.time() - start_time
    if psutil is not None:
        mem_after_mb = process.memory_info().rss / (1024 * 1024)
    else:
        mem_after_mb = 0.0

    # Extract feature importances (gain, weight, cover)
    booster = clf.get_booster()
    # Feature names map to f0, f1, f2
    feature_map = {f"f{i}": name for i, name in enumerate(config.feature_names)}

    gain_raw = booster.get_score(importance_type="gain")
    weight_raw = booster.get_score(importance_type="weight")
    cover_raw = booster.get_score(importance_type="cover")

    gain_importances = {feature_map.get(k, k): float(v) for k, v in gain_raw.items()}
    weight_importances = {feature_map.get(k, k): float(v) for k, v in weight_raw.items()}
    cover_importances = {feature_map.get(k, k): float(v) for k, v in cover_raw.items()}

    default_importances = {
        name: float(imp) for name, imp in zip(config.feature_names, clf.feature_importances_)
    }

    fit_stats = {
        "train_duration_seconds": round(duration_sec, 3),
        "memory_before_mb": round(mem_before_mb, 2),
        "memory_after_mb": round(mem_after_mb, 2),
        "memory_delta_mb": round(mem_after_mb - mem_before_mb, 2),
        "feature_importances_default_gain": default_importances,
        "feature_importances_gain": gain_importances,
        "feature_importances_weight": weight_importances,
        "feature_importances_cover": cover_importances
    }

    return clf, fit_stats


def evaluate_xgboost(
    model: xgb.XGBClassifier,
    val_manifest_path: str,
    config: XGBoostConfig,
    export_predictions: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Evaluate fitted XGBoost model on Somalia validation tiles only.

    Generates prediction rasters maintaining geospatial metadata (CRS, transform, nodata 255).

    Returns:
        Tuple of (overall_metrics, per_tile_metrics)
    """
    if not os.path.exists(val_manifest_path):
        raise FileNotFoundError(f"Validation manifest not found: {val_manifest_path}")

    with open(val_manifest_path, "r") as f:
        val_data = json.load(f)

    val_tiles = val_data.get("tiles", [])
    if len(val_tiles) == 0:
        raise ValueError(f"No tiles found in validation manifest {val_manifest_path}")

    predictions_dir = os.path.join(config.experiment_dir, "predictions")
    if export_predictions:
        os.makedirs(predictions_dir, exist_ok=True)

    all_y_true = []
    all_y_pred = []
    per_tile_metrics = {}

    for tile in val_tiles:
        tile_id = tile["tile_id"]
        sar_p = tile["sar_path"]
        lbl_p = tile["label_path"]

        features, labels, valid_mask, profile = extract_tile_features_and_labels(
            sar_p, lbl_p, nodata_value=config.nodata_value
        )

        pred_raster = np.full(labels.shape, fill_value=config.nodata_value, dtype=np.uint8)

        if np.any(valid_mask):
            X_valid = features[valid_mask]
            preds_valid = model.predict(X_valid).astype(np.uint8)
            pred_raster[valid_mask] = preds_valid

            all_y_true.append(labels[valid_mask])
            all_y_pred.append(preds_valid)

            tile_m = calculate_rf_metrics(labels, pred_raster, nodata_value=config.nodata_value)
            per_tile_metrics[tile_id] = tile_m

        if export_predictions:
            out_pred_path = os.path.join(predictions_dir, f"{tile_id}_pred.tif")
            profile.update({
                "count": 1,
                "dtype": "uint8",
                "nodata": config.nodata_value
            })
            h, w = tile["height"], tile["width"]
            with rasterio.open(out_pred_path, "w", **profile) as dst:
                dst.write(pred_raster.reshape(h, w), 1)

    if len(all_y_true) > 0:
        concat_y_true = np.concatenate(all_y_true)
        concat_y_pred = np.concatenate(all_y_pred)
        overall_metrics = calculate_rf_metrics(
            concat_y_true, concat_y_pred, nodata_value=config.nodata_value
        )
    else:
        overall_metrics = calculate_rf_metrics(
            np.array([], dtype=np.uint8), np.array([], dtype=np.uint8), nodata_value=config.nodata_value
        )

    return overall_metrics, per_tile_metrics


def run_xgb_experiment(
    config: Optional[XGBoostConfig] = None
) -> Dict[str, Any]:
    """End-to-end execution of Experiment 2: XGBoost Baseline."""
    if config is None:
        config = XGBoostConfig()

    os.makedirs(config.experiment_dir, exist_ok=True)

    print("=== SARA Experiment 2: XGBoost Baseline ===")
    print(f"Train Manifest: {config.train_manifest_path}")
    print(f"Val Manifest:   {config.val_manifest_path}")
    print(f"Output Dir:     {config.experiment_dir}")

    # 1. Load Training Dataset (reusing exact sampling strategy as RF baseline)
    print("\n[1/3] Sampling training tiles...")
    # Convert XGBoostConfig sampling parameters to RandomForestConfig format expected by load_training_dataset
    from ml.rf_config import RandomForestConfig
    rf_cfg_adapter = RandomForestConfig(
        samples_per_tile_water=config.samples_per_tile_water,
        samples_per_tile_nonwater=config.samples_per_tile_nonwater,
        max_total_train_samples=config.max_total_train_samples,
        random_state=config.random_state,
        nodata_value=config.nodata_value
    )
    X_train, y_train, data_summary = load_training_dataset(config.train_manifest_path, rf_cfg_adapter)
    print(f"  Processed {data_summary['num_tiles_processed']} tiles.")
    print(f"  Total samples: {data_summary['total_train_samples']}")
    print(f"  Water samples: {data_summary['water_samples']} ({data_summary['water_sample_pct']:.2f}%)")
    print(f"  Non-water samples: {data_summary['nonwater_samples']}")

    # 2. Train XGBoost Classifier
    print("\n[2/3] Fitting XGBClassifier on CPU...")
    model, fit_stats = train_xgboost(X_train, y_train, config)
    print(f"  Training completed in {fit_stats['train_duration_seconds']}s.")
    print(f"  Feature Importances (Gain): {fit_stats['feature_importances_default_gain']}")

    # 3. Evaluate on Validation Split (Somalia tiles)
    print("\n[3/3] Evaluating on Somalia validation tiles...")
    overall_metrics, per_tile_metrics = evaluate_xgboost(
        model, config.val_manifest_path, config, export_predictions=True
    )
    print("  Overall Validation Metrics:")
    print(f"    Valid Pixels:  {overall_metrics['total_valid_pixels']}")
    print(f"    Confusion Matrix: {overall_metrics['confusion_matrix']}")
    print(f"    Water Precision:  {overall_metrics['water_metrics']['precision']:.4f}")
    print(f"    Water Recall:     {overall_metrics['water_metrics']['recall']:.4f}")
    print(f"    Water F1 / Dice:  {overall_metrics['water_metrics']['f1_dice']:.4f}")
    print(f"    Water IoU:        {overall_metrics['water_metrics']['iou']:.4f}")
    print(f"    Predicted Water Pixels: {overall_metrics['predicted_water_pixels']}")

    # 4. Save Experiment Artifacts
    model_path = os.path.join(config.experiment_dir, config.model_save_name)
    model.save_model(model_path)

    experiment_results = {
        "config": config.model_dump(),
        "data_summary": data_summary,
        "fit_stats": fit_stats,
        "overall_metrics": overall_metrics,
        "per_tile_metrics": per_tile_metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    metrics_path = os.path.join(config.experiment_dir, config.metrics_save_name)
    with open(metrics_path, "w") as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\nSaved XGBoost model checkpoint to: {model_path}")
    print(f"Saved metrics & config to: {metrics_path}")

    return experiment_results


if __name__ == "__main__":
    run_xgb_experiment()
