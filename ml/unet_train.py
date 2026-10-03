"""SARA Experiment 4: Small U-Net Flood Segmentation Training & Evaluation Pipeline.

Implements:
1. PyTorch SmallUNet deep learning architecture (3-channel input -> binary water probability head).
2. Masked BCE + Dice loss ignoring label 255 (nodata / cloud invalid).
3. PyTorch Automatic Mixed Precision (AMP FP16) and atomic checkpointing.
4. Early stopping based on validation IoU on Somalia validation partition.
5. GeoTIFF prediction export preserving spatial CRS, affine transform, dimensions (256x256), and nodata=255.
6. Resource-bounded CUDA GPU execution for NVIDIA RTX 2050 (4 GB VRAM).
"""

import os
import json
import time
import csv
import random
import torch
import torch.nn as nn
import numpy as np
import rasterio
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

try:
    import psutil
except ImportError:
    psutil = None

from ml.model import SmallUNet, MaskedBCEDiceLoss, calculate_segmentation_metrics
from ml.unet_config import UNetExperimentConfig
from ml.unet_dataset import create_dataloader, SARAFloodDataset
from ml.rf_baseline import extract_tile_features_and_labels


def set_seed(seed: int = 42):
    """Set deterministic random seed across Python, NumPy, PyTorch, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_atomic_checkpoint(state: dict, filepath: str):
    """Atomically save PyTorch checkpoint to prevent file corruption if interrupted."""
    tmp_filepath = f"{filepath}.tmp"
    torch.save(state, tmp_filepath)
    if os.path.exists(filepath):
        os.remove(filepath)
    os.rename(tmp_filepath, filepath)


def train_unet_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    use_amp: bool = True
) -> float:
    """Train model for one epoch over the training partition."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if use_amp and device.type == "cuda" and scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError("Non-finite loss encountered during training!")

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


def evaluate_unet_val_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    nodata_value: int = 255
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model on validation dataloader in torch.no_grad() mode."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                    loss = criterion(logits, labels)
            else:
                logits = model(images)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            num_batches += 1

            all_logits.append(logits.cpu())
            all_targets.append(labels.cpu())

    avg_loss = total_loss / max(1, num_batches)

    concat_logits = torch.cat(all_logits, dim=0)
    concat_targets = torch.cat(all_targets, dim=0)

    metrics = calculate_segmentation_metrics(
        concat_logits, concat_targets, nodata_value=nodata_value, threshold=0.5
    )

    return avg_loss, metrics


def train_small_unet(
    config: Optional[UNetExperimentConfig] = None
) -> Tuple[SmallUNet, Dict[str, Any]]:
    """Execute complete PyTorch Small U-Net training loop with early stopping and checkpointing."""
    if config is None:
        config = UNetExperimentConfig()

    set_seed(config.fixed_seed)
    os.makedirs(config.experiment_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Small U-Net training on device: {device}")
    if device.type == "cuda":
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.cuda.reset_peak_memory_stats()

    # Create DataLoaders
    train_loader = create_dataloader(
        manifest_path=config.train_manifest_path,
        batch_size=config.batch_size,
        shuffle=True,
        channel_means=config.channel_means,
        channel_stds=config.channel_stds,
        nodata_value=config.nodata_value
    )

    val_loader = create_dataloader(
        manifest_path=config.val_manifest_path,
        batch_size=config.batch_size,
        shuffle=False,
        channel_means=config.channel_means,
        channel_stds=config.channel_stds,
        nodata_value=config.nodata_value
    )

    model = SmallUNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        init_features=config.init_features
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model Parameter Count: {num_params:,}")

    criterion = MaskedBCEDiceLoss(nodata_value=config.nodata_value)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    use_amp = config.use_mixed_precision and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_model_path = os.path.join(config.experiment_dir, config.best_model_name)
    history_csv_path = os.path.join(config.experiment_dir, config.history_csv_name)

    best_val_iou = -1.0
    patience_counter = 0
    history = []

    start_train_time = time.time()

    for epoch in range(1, config.max_epochs + 1):
        epoch_start = time.time()

        train_loss = train_unet_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp=use_amp
        )

        val_loss, val_metrics = evaluate_unet_val_epoch(
            model, val_loader, criterion, device, nodata_value=config.nodata_value
        )

        epoch_duration = time.time() - epoch_start
        val_iou = val_metrics["iou"]
        val_f1 = val_metrics["f1_dice"]
        val_prec = val_metrics["precision"]
        val_rec = val_metrics["recall"]

        peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_precision": round(val_prec, 6),
            "val_recall": round(val_rec, 6),
            "val_f1_dice": round(val_f1, 6),
            "val_iou": round(val_iou, 6),
            "duration_sec": round(epoch_duration, 2),
            "peak_vram_gb": round(peak_vram_gb, 3)
        }
        history.append(epoch_record)

        print(
            f"Epoch [{epoch:02d}/{config.max_epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val IoU: {val_iou:.4f} | Val F1: {val_f1:.4f} | Time: {epoch_duration:.1f}s"
        )

        # Checkpoint if best validation IoU
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": config.model_dump()
            }
            save_atomic_checkpoint(checkpoint_state, best_model_path)
            print(f"  -> Saved new best model checkpoint to {best_model_path} (Val IoU: {val_iou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch} (no improvement for {config.early_stopping_patience} epochs).")
                break

    total_train_duration = time.time() - start_train_time

    # Save training history CSV
    with open(history_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    # Load best checkpoint back into model for return
    best_checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    summary = {
        "num_parameters": num_params,
        "completed_epochs": len(history),
        "best_epoch": best_checkpoint["epoch"],
        "best_val_iou": round(best_val_iou, 6),
        "total_train_duration_sec": round(total_train_duration, 2),
        "peak_vram_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0, 3)
    }

    return model, summary


def evaluate_unet_predictions(
    model: nn.Module,
    val_manifest_path: str,
    config: UNetExperimentConfig,
    export_predictions: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Evaluate trained Small U-Net model on Somalia validation partition and export prediction GeoTIFFs."""
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    channel_means = np.array(config.channel_means, dtype=np.float32).reshape(3, 1, 1)
    channel_stds = np.array(config.channel_stds, dtype=np.float32).reshape(3, 1, 1)

    all_y_true = []
    all_y_pred = []
    per_tile_metrics = {}

    start_eval_time = time.time()

    for tile in val_tiles:
        tile_id = tile["tile_id"]
        sar_p = tile["sar_path"]
        lbl_p = tile["label_path"]

        with rasterio.open(sar_p) as src_sar:
            sar_img = src_sar.read()  # (3, H, W)
            profile = src_sar.profile.copy()

        with rasterio.open(lbl_p) as src_lbl:
            label_img = src_lbl.read(1)  # (H, W)

        norm_sar = (sar_img - channel_means) / (channel_stds + 1e-7)
        input_tensor = torch.from_numpy(norm_sar).unsqueeze(0).float().to(device)

        valid_mask = (label_img != config.nodata_value)

        with torch.no_grad():
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(input_tensor)
            else:
                logits = model(input_tensor)

            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        pred_mask = (probs >= 0.5).astype(np.uint8)
        pred_raster = np.full(label_img.shape, fill_value=config.nodata_value, dtype=np.uint8)
        pred_raster[valid_mask] = pred_mask[valid_mask]

        if np.any(valid_mask):
            all_y_true.append(label_img[valid_mask])
            all_y_pred.append(pred_mask[valid_mask])

            tile_m = calculate_segmentation_metrics(
                logits.cpu(),
                torch.from_numpy(label_img).unsqueeze(0).unsqueeze(0),
                nodata_value=config.nodata_value,
                threshold=0.5
            )
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

        tn = int(np.sum((concat_y_true == 0) & (concat_y_pred == 0)))
        fp = int(np.sum((concat_y_true == 0) & (concat_y_pred == 1)))
        fn = int(np.sum((concat_y_true == 1) & (concat_y_pred == 0)))
        tp = int(np.sum((concat_y_true == 1) & (concat_y_pred == 1)))

        water_prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        water_rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        water_f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        water_iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0

        overall_metrics = {
            "total_valid_pixels": int(len(concat_y_true)),
            "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
            "water_metrics": {
                "precision": round(water_prec, 6),
                "recall": round(water_rec, 6),
                "f1_dice": round(water_f1, 6),
                "iou": round(water_iou, 6),
                "support": int(tp + fn)
            },
            "predicted_water_pixels": int(tp + fp),
            "eval_duration_seconds": round(eval_duration_sec, 3)
        }
    else:
        overall_metrics = {}

    return overall_metrics, per_tile_metrics


def run_unet_experiment(
    config: Optional[UNetExperimentConfig] = None
) -> Dict[str, Any]:
    """End-to-end execution of Experiment 4: Small U-Net Flood Segmentation."""
    if config is None:
        config = UNetExperimentConfig()

    os.makedirs(config.experiment_dir, exist_ok=True)

    print("=== SARA Experiment 4: Small U-Net Flood Segmentation ===")
    print(f"Train Manifest: {config.train_manifest_path}")
    print(f"Val Manifest:   {config.val_manifest_path}")
    print(f"Output Dir:     {config.experiment_dir}")

    # 1. Train Small U-Net Model
    model, train_summary = train_small_unet(config)

    # 2. Evaluate Best Checkpoint on Somalia Validation Partition
    print("\n[Evaluating Best Small U-Net Checkpoint on Somalia Validation Partition...]")
    overall_metrics, per_tile_metrics = evaluate_unet_predictions(
        model, config.val_manifest_path, config, export_predictions=True
    )

    print("  Overall U-Net Validation Metrics:")
    print(f"    Valid Pixels:  {overall_metrics['total_valid_pixels']}")
    print(f"    Confusion Matrix: {overall_metrics['confusion_matrix']}")
    print(f"    Water Precision:  {overall_metrics['water_metrics']['precision']:.4f}")
    print(f"    Water Recall:     {overall_metrics['water_metrics']['recall']:.4f}")
    print(f"    Water F1 / Dice:  {overall_metrics['water_metrics']['f1_dice']:.4f}")
    print(f"    Water IoU:        {overall_metrics['water_metrics']['iou']:.4f}")
    print(f"    Predicted Water Pixels: {overall_metrics['predicted_water_pixels']}")

    # 3. Save Experiment Config and Metrics JSON
    config_path = os.path.join(config.experiment_dir, config.config_json_name)
    with open(config_path, "w") as f:
        json.dump(config.model_dump(), f, indent=2)

    experiment_results = {
        "config": config.model_dump(),
        "train_summary": train_summary,
        "overall_metrics": overall_metrics,
        "per_tile_metrics": per_tile_metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    metrics_path = os.path.join(config.experiment_dir, config.metrics_name)
    with open(metrics_path, "w") as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\nSaved U-Net configuration to: {config_path}")
    print(f"Saved metrics & summary to: {metrics_path}")

    return experiment_results


if __name__ == "__main__":
    run_unet_experiment()
