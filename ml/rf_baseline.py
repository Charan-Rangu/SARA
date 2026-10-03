"""SARA Experiment 1: Random Forest Baseline for Sentinel-1 Flood Segmentation.

Implements:
1. Pixel-level Random Forest classifier using scikit-learn.
2. Controlled tile-stratified pixel sampling from training manifest (36 tiles).
3. Leakage-safe feature extraction: VV_dB, VH_dB, VV_dB - VH_dB (log-ratio).
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
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

try:
    import psutil
except ImportError:
    psutil = None

import joblib
import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier

from ml.rf_config import RandomForestConfig


def extract_tile_features_and_labels(
    sar_path: str,
    label_path: str,
    nodata_value: int = 255
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Extract 3-band SAR features and 1-band label mask from GeoTIFF rasters.

    Features:
    1. VV_dB (Band 1)
    2. VH_dB (Band 2)
    3. VV_VH_ratio (Band 1 - Band 2 = VV_dB - VH_dB)

    Returns:
        Tuple of (features array (N, 3), labels array (N,), valid_mask (N,), raster_profile dict)
    """
    if not os.path.exists(sar_path):
        raise FileNotFoundError(f"SAR raster not found: {sar_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label raster not found: {label_path}")

    with rasterio.open(sar_path) as src_sar:
        sar_img = src_sar.read()  # Shape: (3, H, W)
        profile = src_sar.profile.copy()

    with rasterio.open(label_path) as src_lbl:
        label_img = src_lbl.read(1)  # Shape: (H, W)

    height, width = label_img.shape
    num_pixels = height * width

    # Reshape features to (N, 3)
    # Band 1: VV_dB, Band 2: VH_dB, Band 3: VV_VH_ratio (VV_dB - VH_dB)
    features = np.zeros((num_pixels, 3), dtype=np.float32)
    features[:, 0] = sar_img[0].reshape(-1)
    features[:, 1] = sar_img[1].reshape(-1)
    features[:, 2] = sar_img[2].reshape(-1)

    labels = label_img.reshape(-1).astype(np.uint8)
    valid_mask = (labels != nodata_value)

    return features, labels, valid_mask, profile


def sample_tile_pixels(
    sar_path: str,
    label_path: str,
    samples_water: int = 1000,
    samples_nonwater: int = 2000,
    seed: int = 42,
    nodata_value: int = 255
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministically sample valid water and non-water pixels from a single tile.

    Returns:
        Tuple of (X_sampled, y_sampled)
    """
    features, labels, valid_mask, _ = extract_tile_features_and_labels(
        sar_path, label_path, nodata_value=nodata_value
    )

    X_valid = features[valid_mask]
    y_valid = labels[valid_mask]

    rng = np.random.default_rng(seed)

    water_indices = np.where(y_valid == 1)[0]
    nonwater_indices = np.where(y_valid == 0)[0]

    # Sample water pixels (if fewer available, take all available)
    if len(water_indices) > 0:
        n_water = min(samples_water, len(water_indices))
        sampled_water_idx = rng.choice(water_indices, size=n_water, replace=False)
    else:
        sampled_water_idx = np.array([], dtype=int)

    # Sample non-water pixels
    if len(nonwater_indices) > 0:
        n_nonwater = min(samples_nonwater, len(nonwater_indices))
        sampled_nonwater_idx = rng.choice(nonwater_indices, size=n_nonwater, replace=False)
    else:
        sampled_nonwater_idx = np.array([], dtype=int)

    if len(sampled_water_idx) == 0 and len(sampled_nonwater_idx) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.uint8)

    all_sampled_idx = np.concatenate([sampled_water_idx, sampled_nonwater_idx])
    rng.shuffle(all_sampled_idx)

    return X_valid[all_sampled_idx], y_valid[all_sampled_idx]


def load_training_dataset(
    manifest_path: str,
    config: RandomForestConfig
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Stream valid pixels from training manifest tiles into a bounded feature matrix.

    Returns:
        Tuple of (X_train, y_train, dataset_summary)
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Training manifest not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest_data = json.load(f)

    tiles = manifest_data.get("tiles", [])
    if len(tiles) == 0:
        raise ValueError(f"No tiles found in training manifest {manifest_path}")

    x_list = []
    y_list = []
    total_water_sampled = 0
    total_nonwater_sampled = 0

    for idx, tile in enumerate(tiles):
        sar_p = tile["sar_path"]
        lbl_p = tile["label_path"]
        tile_seed = config.random_state + idx

        x_tile, y_tile = sample_tile_pixels(
            sar_p,
            lbl_p,
            samples_water=config.samples_per_tile_water,
            samples_nonwater=config.samples_per_tile_nonwater,
            seed=tile_seed,
            nodata_value=config.nodata_value
        )

        if len(x_tile) > 0:
            x_list.append(x_tile)
            y_list.append(y_tile)
            total_water_sampled += int(np.sum(y_tile == 1))
            total_nonwater_sampled += int(np.sum(y_tile == 0))

    if len(x_list) == 0:
        raise ValueError("No valid pixels extracted from training manifest tiles.")

    X_train = np.vstack(x_list)
    y_train = np.concatenate(y_list)

    # Apply cap if total samples exceed max_total_train_samples
    if len(X_train) > config.max_total_train_samples:
        rng = np.random.default_rng(config.random_state)
        keep_idx = rng.choice(len(X_train), size=config.max_total_train_samples, replace=False)
        X_train = X_train[keep_idx]
        y_train = y_train[keep_idx]
        total_water_sampled = int(np.sum(y_train == 1))
        total_nonwater_sampled = int(np.sum(y_train == 0))

    summary = {
        "num_tiles_processed": len(tiles),
        "total_train_samples": int(len(X_train)),
        "water_samples": total_water_sampled,
        "nonwater_samples": total_nonwater_sampled,
        "water_sample_pct": float((total_water_sampled / max(1, len(X_train))) * 100.0)
    }

    return X_train, y_train, summary


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: RandomForestConfig
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """Train RandomForestClassifier on pixel features with resource timing."""
    start_time = time.time()
    if psutil is not None:
        process = psutil.Process(os.getpid())
        mem_before_mb = process.memory_info().rss / (1024 * 1024)
    else:
        mem_before_mb = 0.0

    clf = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        class_weight=config.class_weight
    )

    clf.fit(X_train, y_train)

    duration_sec = time.time() - start_time
    if psutil is not None:
        mem_after_mb = process.memory_info().rss / (1024 * 1024)
    else:
        mem_after_mb = 0.0

    fit_stats = {
        "train_duration_seconds": round(duration_sec, 3),
        "memory_before_mb": round(mem_before_mb, 2),
        "memory_after_mb": round(mem_after_mb, 2),
        "memory_delta_mb": round(mem_after_mb - mem_before_mb, 2),
        "feature_importances": {
            name: float(imp) for name, imp in zip(config.feature_names, clf.feature_importances_)
        }
    }

    return clf, fit_stats


def calculate_rf_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nodata_value: int = 255
) -> Dict[str, Any]:
    """Calculate confusion matrix, precision, recall, F1/Dice, and IoU ignoring nodata (255)."""
    valid_mask = (y_true != nodata_value)
    if not np.any(valid_mask):
        return {
            "total_valid_pixels": 0,
            "confusion_matrix": {"TN": 0, "FP": 0, "FN": 0, "TP": 0},
            "water_metrics": {"precision": 0.0, "recall": 0.0, "f1_dice": 0.0, "iou": 0.0, "support": 0},
            "nonwater_metrics": {"precision": 0.0, "recall": 0.0, "f1_dice": 0.0, "iou": 0.0, "support": 0},
            "predicted_water_pixels": 0
        }

    yt = y_true[valid_mask]
    yp = y_pred[valid_mask]

    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tp = int(np.sum((yt == 1) & (yp == 1)))

    # Water metrics
    water_prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    water_rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    water_f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    water_iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0

    # Non-water metrics
    nonwater_prec = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    nonwater_rec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    nonwater_f1 = float(2 * tn / (2 * tn + fn + fp)) if (2 * tn + fn + fp) > 0 else 0.0
    nonwater_iou = float(tn / (tn + fn + fp)) if (tn + fn + fp) > 0 else 0.0

    return {
        "total_valid_pixels": int(len(yt)),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "water_metrics": {
            "precision": round(water_prec, 6),
            "recall": round(water_rec, 6),
            "f1_dice": round(water_f1, 6),
            "iou": round(water_iou, 6),
            "support": int(tp + fn)
        },
        "nonwater_metrics": {
            "precision": round(nonwater_prec, 6),
            "recall": round(nonwater_rec, 6),
            "f1_dice": round(nonwater_f1, 6),
            "iou": round(nonwater_iou, 6),
            "support": int(tn + fp)
        },
        "predicted_water_pixels": int(tp + fp)
    }


def evaluate_random_forest(
    model: RandomForestClassifier,
    val_manifest_path: str,
    config: RandomForestConfig,
    export_predictions: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Evaluate fitted model on Somalia validation tiles only.

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


def run_rf_experiment(
    config: Optional[RandomForestConfig] = None
) -> Dict[str, Any]:
    """End-to-end execution of Experiment 1: Random Forest Baseline."""
    if config is None:
        config = RandomForestConfig()

    os.makedirs(config.experiment_dir, exist_ok=True)

    print("=== SARA Experiment 1: Random Forest Baseline ===")
    print(f"Train Manifest: {config.train_manifest_path}")
    print(f"Val Manifest:   {config.val_manifest_path}")
    print(f"Output Dir:     {config.experiment_dir}")

    # 1. Load Training Dataset
    print("\n[1/3] Sampling training tiles...")
    X_train, y_train, data_summary = load_training_dataset(config.train_manifest_path, config)
    print(f"  Processed {data_summary['num_tiles_processed']} tiles.")
    print(f"  Total samples: {data_summary['total_train_samples']}")
    print(f"  Water samples: {data_summary['water_samples']} ({data_summary['water_sample_pct']:.2f}%)")
    print(f"  Non-water samples: {data_summary['nonwater_samples']}")

    # 2. Train Random Forest Classifier
    print("\n[2/3] Fitting RandomForestClassifier on CPU...")
    model, fit_stats = train_random_forest(X_train, y_train, config)
    print(f"  Training completed in {fit_stats['train_duration_seconds']}s.")
    print(f"  Feature Importances: {fit_stats['feature_importances']}")

    # 3. Evaluate on Validation Split (Somalia tiles)
    print("\n[3/3] Evaluating on Somalia validation tiles...")
    overall_metrics, per_tile_metrics = evaluate_random_forest(
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
    joblib.dump(model, model_path)

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

    print(f"\nSaved model checkpoint to: {model_path}")
    print(f"Saved metrics & config to: {metrics_path}")

    return experiment_results


if __name__ == "__main__":
    run_rf_experiment()
