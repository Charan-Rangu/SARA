"""SARA Experiment 3: Leakage-Safe Stacking Ensemble.

Implements:
1. Event-Grouped 9-Fold Cross-Validation on train_manifest.json to generate Out-of-Fold (OOF) base model predictions.
2. Base Models: RandomForestClassifier and XGBClassifier trained exclusively on fold-training partitions.
3. Meta-Model: LogisticRegression fitted strictly on the 2D OOF probability matrix [P_RF_OOF, P_XGB_OOF].
4. Full Base Model Assembly: Base models fitted on the complete 36-tile training partition for final pipeline assembly.
5. Somalia Validation Evaluation: Strict evaluation on the 4 Somalia validation tiles ignoring nodata label 255.
6. GeoTIFF Prediction Export: Export of prediction rasters maintaining geospatial metadata (CRS, transform, nodata 255).
"""

import os
import json
import time
import joblib
import numpy as np
import rasterio
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
from sklearn.linear_model import LogisticRegression

try:
    import psutil
except ImportError:
    psutil = None

from ml.stacking_config import StackingConfig
from ml.rf_baseline import (
    extract_tile_features_and_labels,
    sample_tile_pixels,
    train_random_forest,
    calculate_rf_metrics
)
from ml.xgb_baseline import train_xgboost


class StackingPipeline:
    """Wrapper container holding full base models and fitted meta-model."""

    def __init__(self, full_rf: Any, full_xgb: Any, meta_model: LogisticRegression):
        self.full_rf = full_rf
        self.full_xgb = full_xgb
        self.meta_model = meta_model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict water probabilities through base models and meta-model."""
        p_rf = self.full_rf.predict_proba(X)[:, 1]
        p_xgb = self.full_xgb.predict_proba(X)[:, 1]
        X_meta = np.column_stack([p_rf, p_xgb])
        return self.meta_model.predict_proba(X_meta)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary water mask (0 or 1) through meta-model."""
        p_rf = self.full_rf.predict_proba(X)[:, 1]
        p_xgb = self.full_xgb.predict_proba(X)[:, 1]
        X_meta = np.column_stack([p_rf, p_xgb])
        return self.meta_model.predict(X_meta)


def generate_event_grouped_oof_predictions(
    manifest_path: str,
    config: StackingConfig
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate Out-of-Fold (OOF) probability predictions using 9-fold event-grouped CV.

    Returns:
        Tuple of (X_oof matrix shape (N, 2), y_oof array shape (N,), oof_stats dict)
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Training manifest not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest_data = json.load(f)

    tiles = manifest_data.get("tiles", [])
    if len(tiles) == 0:
        raise ValueError(f"No tiles found in manifest {manifest_path}")

    events = sorted(list(set(t["event_id"] for t in tiles)))

    # First, sample all tiles into per-tile dataset list
    tile_samples = []
    for idx, tile in enumerate(tiles):
        tile_seed = config.random_state + idx
        x_tile, y_tile = sample_tile_pixels(
            tile["sar_path"],
            tile["label_path"],
            samples_water=config.samples_per_tile_water,
            samples_nonwater=config.samples_per_tile_nonwater,
            seed=tile_seed,
            nodata_value=config.nodata_value
        )
        if len(x_tile) > 0:
            tile_samples.append({
                "tile_id": tile["tile_id"],
                "event_id": tile["event_id"],
                "X": x_tile,
                "y": y_tile,
                "count": len(x_tile)
            })

    total_samples = sum(ts["count"] for ts in tile_samples)
    X_oof = np.zeros((total_samples, 2), dtype=np.float32)
    y_oof = np.zeros((total_samples,), dtype=np.uint8)

    fold_details = {}
    current_idx = 0

    print(f"Generating OOF predictions across {len(events)} event folds...")

    for fold_idx, held_out_event in enumerate(events):
        start_fold_time = time.time()

        # Split tile_samples into train_fold and val_fold
        train_fold_samples = [ts for ts in tile_samples if ts["event_id"] != held_out_event]
        val_fold_samples = [ts for ts in tile_samples if ts["event_id"] == held_out_event]

        # Stack training data for this fold
        X_tr_fold = np.vstack([ts["X"] for ts in train_fold_samples])
        y_tr_fold = np.concatenate([ts["y"] for ts in train_fold_samples])

        # Fit fold-level base models
        rf_fold_cfg = config.rf_config.model_copy()
        rf_fold_cfg.random_state = config.random_state + fold_idx
        rf_fold, _ = train_random_forest(X_tr_fold, y_tr_fold, rf_fold_cfg)

        xgb_fold_cfg = config.xgb_config.model_copy()
        xgb_fold_cfg.random_state = config.random_state + fold_idx
        xgb_fold, _ = train_xgboost(X_tr_fold, y_tr_fold, xgb_fold_cfg)

        # Predict OOF probabilities on held-out event tiles
        fold_val_count = 0
        for ts in val_fold_samples:
            n_samples = ts["count"]
            p_rf_val = rf_fold.predict_proba(ts["X"])[:, 1]
            p_xgb_val = xgb_fold.predict_proba(ts["X"])[:, 1]

            end_idx = current_idx + n_samples
            X_oof[current_idx:end_idx, 0] = p_rf_val
            X_oof[current_idx:end_idx, 1] = p_xgb_val
            y_oof[current_idx:end_idx] = ts["y"]

            current_idx += n_samples
            fold_val_count += n_samples

        duration = time.time() - start_fold_time
        fold_details[held_out_event] = {
            "val_tiles": [ts["tile_id"] for ts in val_fold_samples],
            "train_samples": len(X_tr_fold),
            "val_samples": fold_val_count,
            "duration_seconds": round(duration, 3)
        }
        print(f"  Fold {fold_idx + 1}/{len(events)} ({held_out_event}): {fold_val_count} OOF samples in {duration:.2f}s.")

    oof_stats = {
        "num_events": len(events),
        "total_oof_samples": total_samples,
        "water_samples": int(np.sum(y_oof == 1)),
        "nonwater_samples": int(np.sum(y_oof == 0)),
        "fold_details": fold_details
    }

    return X_oof, y_oof, oof_stats


def train_stacking_meta_model(
    X_oof: np.ndarray,
    y_oof: np.ndarray,
    config: StackingConfig
) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """Train LogisticRegression meta-model strictly on OOF base model predictions."""
    start_time = time.time()
    if psutil is not None:
        process = psutil.Process(os.getpid())
        mem_before_mb = process.memory_info().rss / (1024 * 1024)
    else:
        mem_before_mb = 0.0

    meta_model = LogisticRegression(
        C=config.C,
        solver=config.solver,
        max_iter=config.max_iter,
        random_state=config.random_state,
        class_weight=config.class_weight
    )

    meta_model.fit(X_oof, y_oof)

    duration_sec = time.time() - start_time
    if psutil is not None:
        mem_after_mb = process.memory_info().rss / (1024 * 1024)
    else:
        mem_after_mb = 0.0

    coefs = meta_model.coef_[0]
    fit_stats = {
        "train_duration_seconds": round(duration_sec, 4),
        "memory_before_mb": round(mem_before_mb, 2),
        "memory_after_mb": round(mem_after_mb, 2),
        "memory_delta_mb": round(mem_after_mb - mem_before_mb, 2),
        "rf_probability_weight": float(coefs[0]),
        "xgb_probability_weight": float(coefs[1]),
        "intercept": float(meta_model.intercept_[0])
    }

    return meta_model, fit_stats


def train_full_stacking_pipeline(
    train_manifest_path: str,
    config: StackingConfig
) -> Tuple[StackingPipeline, Dict[str, Any]]:
    """End-to-end training of full Stacking Ensemble pipeline."""
    # 1. Generate OOF predictions
    print("\n[Step 1/3] Generating Event-Grouped Out-of-Fold Predictions...")
    X_oof, y_oof, oof_stats = generate_event_grouped_oof_predictions(train_manifest_path, config)

    # 2. Train Meta-Model on OOF data
    print("\n[Step 2/3] Fitting LogisticRegression Meta-Model on OOF predictions...")
    meta_model, meta_fit_stats = train_stacking_meta_model(X_oof, y_oof, config)
    print(f"  Meta-Model Weights -> RF_Prob: {meta_fit_stats['rf_probability_weight']:.4f}, XGB_Prob: {meta_fit_stats['xgb_probability_weight']:.4f}, Intercept: {meta_fit_stats['intercept']:.4f}")

    # 3. Fit Full Base Models on full training set
    print("\n[Step 3/3] Training Full Base Models (RF & XGBoost) on complete training partition...")
    from ml.rf_baseline import load_training_dataset as load_rf_dataset
    X_train_full, y_train_full, full_data_summary = load_rf_dataset(train_manifest_path, config.rf_config)

    full_rf, _ = train_random_forest(X_train_full, y_train_full, config.rf_config)
    full_xgb, _ = train_xgboost(X_train_full, y_train_full, config.xgb_config)

    pipeline = StackingPipeline(full_rf, full_xgb, meta_model)

    pipeline_summary = {
        "oof_stats": oof_stats,
        "meta_fit_stats": meta_fit_stats,
        "full_data_summary": full_data_summary
    }

    return pipeline, pipeline_summary


def evaluate_stacking_pipeline(
    pipeline: StackingPipeline,
    val_manifest_path: str,
    config: StackingConfig,
    export_predictions: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Evaluate full Stacking Ensemble on Somalia validation partition only.

    Exports prediction rasters maintaining geospatial metadata (CRS, transform, nodata 255).
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

    start_eval_time = time.time()

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
            preds_valid = pipeline.predict(X_valid).astype(np.uint8)
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

    eval_duration_sec = time.time() - start_eval_time

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

    overall_metrics["eval_duration_seconds"] = round(eval_duration_sec, 3)

    return overall_metrics, per_tile_metrics


def run_stacking_experiment(
    config: Optional[StackingConfig] = None
) -> Dict[str, Any]:
    """End-to-end execution of Experiment 3: Stacking Ensemble."""
    if config is None:
        config = StackingConfig()

    os.makedirs(config.experiment_dir, exist_ok=True)

    print("=== SARA Experiment 3: Leakage-Safe Stacking Ensemble ===")
    print(f"Train Manifest: {config.train_manifest_path}")
    print(f"Val Manifest:   {config.val_manifest_path}")
    print(f"Output Dir:     {config.experiment_dir}")

    # 1. Train Full Stacking Pipeline
    pipeline, pipeline_summary = train_full_stacking_pipeline(config.train_manifest_path, config)

    # 2. Evaluate on Validation Partition (Somalia tiles)
    print("\n[Evaluating Stacking Pipeline on Somalia Validation Partition...]")
    overall_metrics, per_tile_metrics = evaluate_stacking_pipeline(
        pipeline, config.val_manifest_path, config, export_predictions=True
    )

    print("  Overall Stacking Validation Metrics:")
    print(f"    Valid Pixels:  {overall_metrics['total_valid_pixels']}")
    print(f"    Confusion Matrix: {overall_metrics['confusion_matrix']}")
    print(f"    Water Precision:  {overall_metrics['water_metrics']['precision']:.4f}")
    print(f"    Water Recall:     {overall_metrics['water_metrics']['recall']:.4f}")
    print(f"    Water F1 / Dice:  {overall_metrics['water_metrics']['f1_dice']:.4f}")
    print(f"    Water IoU:        {overall_metrics['water_metrics']['iou']:.4f}")
    print(f"    Predicted Water Pixels: {overall_metrics['predicted_water_pixels']}")

    # 3. Save Experiment Artifacts
    model_path = os.path.join(config.experiment_dir, config.model_save_name)
    joblib.dump(pipeline, model_path)

    experiment_results = {
        "config": config.model_dump(),
        "pipeline_summary": pipeline_summary,
        "overall_metrics": overall_metrics,
        "per_tile_metrics": per_tile_metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    metrics_path = os.path.join(config.experiment_dir, config.metrics_save_name)
    with open(metrics_path, "w") as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\nSaved Stacking Ensemble pipeline checkpoint to: {model_path}")
    print(f"Saved metrics & config to: {metrics_path}")

    return experiment_results


if __name__ == "__main__":
    run_stacking_experiment()
