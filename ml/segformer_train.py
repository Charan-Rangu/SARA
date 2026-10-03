"""SARA Experiment 6: SegFormer-B0 Flood Segmentation Training & Evaluation Pipeline.

Implements:
1. PyTorch SegFormer-B0 architecture (3-channel input -> 1-channel binary water probability logit).
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

from ml.model import MaskedBCEDiceLoss, calculate_segmentation_metrics
from ml.segformer_model import SegFormerB0
from ml.segformer_config import SegFormerExperimentConfig
from ml.unet_dataset import create_dataloader, SARAFloodDataset


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


def train_segformer_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    use_amp: bool = True
) -> float:
    """Train SegFormer model for one epoch over training partition."""
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


def evaluate_segformer_val_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    nodata_value: int = 255
) -> Tuple[float, Dict[str, float]]:
    """Evaluate SegFormer model on validation partition."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)

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
        concat_logits, concat_targets, nodata_value=nodata_value
    )

    return avg_loss, metrics


def run_segformer_training(
    config: Optional[SegFormerExperimentConfig] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Execute complete SegFormer-B0 training loop on train_manifest.json."""
    if config is None:
        config = SegFormerExperimentConfig()

    set_seed(config.fixed_seed)
    os.makedirs(config.experiment_dir, exist_ok=True)

    config_path = os.path.join(config.experiment_dir, config.config_json_name)
    config.save(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"SegFormer-B0 Training Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    train_loader = create_dataloader(
        config.train_manifest_path,
        batch_size=config.batch_size,
        shuffle=True,
        channel_means=config.channel_means,
        channel_stds=config.channel_stds
    )

    val_loader = create_dataloader(
        config.val_manifest_path,
        batch_size=config.batch_size,
        shuffle=False,
        channel_means=config.channel_means,
        channel_stds=config.channel_stds
    )

    model = SegFormerB0(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        embedding_dim=config.embedding_dim
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Loaded SegFormerB0 model with {num_params:,} trainable parameters.")

    criterion = MaskedBCEDiceLoss(nodata_value=config.nodata_value, bce_weight=0.5, pos_weight=3.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    use_amp = config.use_mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_model_path = os.path.join(config.experiment_dir, config.best_model_name)
    history_csv_path = os.path.join(config.experiment_dir, config.history_csv_name)

    best_val_iou = -1.0
    patience_counter = 0
    history = []

    start_train_time = time.time()

    for epoch in range(1, config.max_epochs + 1):
        epoch_start = time.time()

        train_loss = train_segformer_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp=use_amp
        )

        val_loss, val_metrics = evaluate_segformer_val_epoch(
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
            print(f"  -> Saved new best SegFormer checkpoint to {best_model_path} (Val IoU: {val_iou:.4f})")
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


def evaluate_segformer_predictions(
    model: nn.Module,
    val_manifest_path: str,
    config: SegFormerExperimentConfig,
    export_predictions: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Evaluate trained SegFormer model on Somalia validation partition and export prediction GeoTIFFs."""
    if not os.path.exists(val_manifest_path):
        raise FileNotFoundError(f"Validation manifest not found: {val_manifest_path}")

    with open(val_manifest_path, "r") as f:
        val_data = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions_dir = os.path.join(config.experiment_dir, "predictions")
    if export_predictions:
        os.makedirs(predictions_dir, exist_ok=True)

    dataset = SARAFloodDataset(
        val_manifest_path,
        channel_means=config.channel_means,
        channel_stds=config.channel_stds
    )

    tile_map = {t["tile_id"]: t for t in val_data["tiles"]}

    all_concat_logits = []
    all_concat_targets = []
    per_tile_metrics = {}

    eval_start = time.time()

    with torch.no_grad():
        for idx in range(len(dataset)):
            image_tensor, label_tensor, tile_id = dataset[idx]
            image_batch = image_tensor.unsqueeze(0).to(device)
            label_batch = label_tensor.unsqueeze(0).to(device)

            logits_batch = model(image_batch)

            all_concat_logits.append(logits_batch.cpu())
            all_concat_targets.append(label_batch.cpu())

            tile_metrics = calculate_segmentation_metrics(
                logits_batch.cpu(), label_batch.cpu(), nodata_value=config.nodata_value
            )

            # Recompute exact per-tile confusion matrix counts
            probs = torch.sigmoid(logits_batch).squeeze().cpu().numpy()
            pred_mask = (probs >= 0.5).astype(np.uint8)

            lbl_path = tile_map[tile_id]["label_path"]
            sar_path = tile_map[tile_id]["sar_path"]

            with rasterio.open(lbl_path) as src_lbl:
                lbl_arr = src_lbl.read(1)

            valid_mask = (lbl_arr != config.nodata_value)
            t_valid = lbl_arr[valid_mask]
            p_valid = pred_mask[valid_mask]

            tp = int(np.sum((p_valid == 1) & (t_valid == 1)))
            fp = int(np.sum((p_valid == 1) & (t_valid == 0)))
            fn = int(np.sum((p_valid == 0) & (t_valid == 1)))
            tn = int(np.sum((p_valid == 0) & (t_valid == 0)))

            tile_metrics["TP"] = tp
            tile_metrics["FP"] = fp
            tile_metrics["FN"] = fn
            tile_metrics["TN"] = tn
            tile_metrics["valid_pixels"] = int(np.sum(valid_mask))
            per_tile_metrics[tile_id] = tile_metrics

            if export_predictions:
                out_arr = np.full(lbl_arr.shape, config.nodata_value, dtype=np.uint8)
                out_arr[valid_mask] = p_valid

                with rasterio.open(sar_path) as src_sar:
                    profile = src_sar.profile.copy()

                profile.update({
                    "count": 1,
                    "dtype": "uint8",
                    "nodata": float(config.nodata_value)
                })

                out_tif_path = os.path.join(predictions_dir, f"{tile_id}_pred.tif")
                with rasterio.open(out_tif_path, "w", **profile) as dst:
                    dst.write(out_arr, 1)

    eval_duration = time.time() - eval_start

    concat_logits = torch.cat(all_concat_logits, dim=0)
    concat_targets = torch.cat(all_concat_targets, dim=0)

    overall_metrics = calculate_segmentation_metrics(
        concat_logits, concat_targets, nodata_value=config.nodata_value
    )

    total_valid = sum(m["valid_pixels"] for m in per_tile_metrics.values())
    total_tp = sum(m["TP"] for m in per_tile_metrics.values())
    total_fp = sum(m["FP"] for m in per_tile_metrics.values())
    total_fn = sum(m["FN"] for m in per_tile_metrics.values())
    total_tn = sum(m["TN"] for m in per_tile_metrics.values())

    overall_results = {
        "total_valid_pixels": total_valid,
        "confusion_matrix": {
            "TN": total_tn,
            "FP": total_fp,
            "FN": total_fn,
            "TP": total_tp
        },
        "water_metrics": overall_metrics,
        "predicted_water_pixels": total_tp + total_fp,
        "eval_duration_seconds": round(eval_duration, 4)
    }

    return overall_results, per_tile_metrics


def main():
    print("=== Starting SARA Experiment 6: SegFormer-B0 Flood Segmentation Pipeline ===")
    config = SegFormerExperimentConfig()

    model, train_summary = run_segformer_training(config)

    print("\n=== Evaluating SegFormer-B0 on Somalia Validation Partition ===")
    overall_results, per_tile_metrics = evaluate_segformer_predictions(
        model, config.val_manifest_path, config, export_predictions=True
    )

    metrics_payload = {
        "config": config.model_dump(),
        "train_summary": train_summary,
        "overall_metrics": overall_results,
        "per_tile_metrics": per_tile_metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    metrics_json_path = os.path.join(config.experiment_dir, config.metrics_name)
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\nSaved metrics payload to {metrics_json_path}")
    print(f"SegFormer-B0 Validation IoU: {overall_results['water_metrics']['iou'] * 100:.2f}%")
    print(f"SegFormer-B0 Validation F1:  {overall_results['water_metrics']['f1_dice'] * 100:.2f}%")


if __name__ == "__main__":
    main()
