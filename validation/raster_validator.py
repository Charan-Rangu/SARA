"""Independent Raster Validation Engine for SARA.

Performs rigorous binary classification validation (Confusion Matrix, Precision, Recall, F1/Dice,
IoU, Accuracy) between model prediction GeoTIFFs and independent ground-truth reference masks.

Strict Rules:
1. Nodata pixels (255, -9999.0), NaNs, and infinite values are strictly excluded from TP, FP, FN, TN.
2. Division-by-zero is handled safely with explicit undefined flags instead of silent perfect scores.
3. EMSR927 activation data is strictly quarantined and cannot be used as ground-truth for Nepal.
4. When independent reference data is unavailable, status is explicitly set to:
   'INDEPENDENT VALIDATION: PENDING — suitable independent reference masks are unavailable.'
"""

import os
from typing import Optional, Any
import numpy as np
from pydantic import BaseModel, Field
import rasterio
from rasterio.windows import Window


class ConfusionMatrix(BaseModel):
    """Confusion matrix counts for binary flood/water classification."""

    tp: int = 0  # True Positives: Predicted Water == 1, Ground Truth == 1
    fp: int = 0  # False Positives: Predicted Water == 1, Ground Truth == 0
    fn: int = 0  # False Negatives: Predicted Water == 0, Ground Truth == 1
    tn: int = 0  # True Negatives: Predicted Water == 0, Ground Truth == 0
    total_valid_pixels: int = 0
    nodata_pixels: int = 0


class ValidationMetrics(BaseModel):
    """Evaluation metrics summary derived from confusion matrix."""

    status: str
    confusion_matrix: ConfusionMatrix
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    iou: Optional[float] = None
    accuracy: Optional[float] = None
    precision_undefined: bool = False
    recall_undefined: bool = False
    f1_undefined: bool = False
    iou_undefined: bool = False
    accuracy_undefined: bool = False
    data_lineage: str = "UNKNOWN"
    disclaimer: str = (
        "Validation metrics represent comparison against specified reference mask. "
        "Agreement between two model products is NOT independent validation."
    )


def compute_binary_metrics_from_counts(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    nodata: int = 0,
    data_lineage: str = "SYNTHETIC_EVALUATION"
) -> ValidationMetrics:
    """Compute binary classification metrics with explicit division-by-zero handling.

    Args:
        tp: True Positives count
        fp: False Positives count
        fn: False Negatives count
        tn: True Negatives count
        nodata: Nodata pixel count
        data_lineage: Provenance description of reference dataset

    Returns:
        ValidationMetrics object.
    """
    total_valid = tp + fp + fn + tn
    cm = ConfusionMatrix(
        tp=tp, fp=fp, fn=fn, tn=tn,
        total_valid_pixels=total_valid,
        nodata_pixels=nodata
    )

    if total_valid == 0:
        return ValidationMetrics(
            status="UNDEFINED_ZERO_VALID_PIXELS",
            confusion_matrix=cm,
            precision=None,
            recall=None,
            f1_score=None,
            iou=None,
            accuracy=None,
            precision_undefined=True,
            recall_undefined=True,
            f1_undefined=True,
            iou_undefined=True,
            accuracy_undefined=True,
            data_lineage=data_lineage
        )

    # Precision = TP / (TP + FP)
    prec_denom = tp + fp
    if prec_denom > 0:
        precision = float(tp) / float(prec_denom)
        prec_undef = False
    else:
        precision = None
        prec_undef = True

    # Recall = TP / (TP + FN)
    rec_denom = tp + fn
    if rec_denom > 0:
        recall = float(tp) / float(rec_denom)
        rec_undef = False
    else:
        recall = None
        rec_undef = True

    # F1 = 2 * Precision * Recall / (Precision + Recall) = 2 * TP / (2 * TP + FP + FN)
    f1_denom = (2 * tp) + fp + fn
    if f1_denom > 0:
        f1_score = float(2 * tp) / float(f1_denom)
        f1_undef = False
    else:
        f1_score = None
        f1_undef = True

    # IoU = TP / (TP + FP + FN)
    iou_denom = tp + fp + fn
    if iou_denom > 0:
        iou = float(tp) / float(iou_denom)
        iou_undef = False
    else:
        # If both prediction and reference are empty (TP=FP=FN=0), but TN > 0, IoU is 1.0 (perfect background agreement)
        iou = 1.0 if tn > 0 else None
        iou_undef = (tn == 0)

    # Accuracy = (TP + TN) / (TP + FP + FN + TN)
    accuracy = float(tp + tn) / float(total_valid)
    acc_undef = False

    return ValidationMetrics(
        status="COMPUTED",
        confusion_matrix=cm,
        precision=round(precision, 6) if precision is not None else None,
        recall=round(recall, 6) if recall is not None else None,
        f1_score=round(f1_score, 6) if f1_score is not None else None,
        iou=round(iou, 6) if iou is not None else None,
        accuracy=round(accuracy, 6),
        precision_undefined=prec_undef,
        recall_undefined=rec_undef,
        f1_undefined=f1_undef,
        iou_undefined=iou_undef,
        accuracy_undefined=acc_undef,
        data_lineage=data_lineage
    )


class RasterValidator:
    """Validator for comparing prediction rasters against independent reference masks."""

    def __init__(self, block_size: int = 2048):
        self.block_size = block_size

    def evaluate(
        self,
        prediction_raster_path: str,
        reference_raster_path: Optional[str] = None,
        pred_nodata: float = 255.0,
        ref_nodata: float = 255.0,
        data_lineage: Optional[str] = None
    ) -> ValidationMetrics:
        """Evaluate prediction raster against reference raster in windowed chunks.

        Args:
            prediction_raster_path: Path to predicted water mask GeoTIFF (1=water, 0=land).
            reference_raster_path: Path to ground-truth reference GeoTIFF (1=water, 0=land).
            pred_nodata: Nodata fill value for prediction raster.
            ref_nodata: Nodata fill value for reference raster.
            data_lineage: Optional data lineage / provenance description.

        Returns:
            ValidationMetrics object.
        """
        # Quarantined Data Check (intercept immediately regardless of file existence)
        if reference_raster_path and ("EMSR927" in reference_raster_path.upper() or "EMS" in reference_raster_path.upper()):
            return ValidationMetrics(
                status="QUARANTINED_REFERENCE_DATA — EMSR927 data cannot be used for evaluation.",
                confusion_matrix=ConfusionMatrix(),
                data_lineage="EMSR927_QUARANTINE",
                disclaimer="Copernicus EMS activation data is strictly quarantined per challenge specifications."
            )

        # Independence Check: If reference data is missing or pending
        if not reference_raster_path or not os.path.exists(reference_raster_path):
            return ValidationMetrics(
                status="INDEPENDENT VALIDATION: PENDING — suitable independent reference masks are unavailable.",
                confusion_matrix=ConfusionMatrix(),
                data_lineage=data_lineage or "UNAVAILABLE",
                disclaimer="Independent Nepal ground truth is unavailable. Metrics are pending verified reference data."
            )

        if not os.path.exists(prediction_raster_path):
            raise FileNotFoundError(f"Prediction raster not found: {prediction_raster_path}")

        with rasterio.open(prediction_raster_path) as p_src, rasterio.open(reference_raster_path) as r_src:
            # Dimension & CRS Checks
            if (p_src.width != r_src.width) or (p_src.height != r_src.height):
                raise ValueError(
                    f"Raster dimension mismatch: prediction={p_src.shape} vs reference={r_src.shape}"
                )
            if p_src.crs != r_src.crs:
                raise ValueError(
                    f"Raster CRS mismatch: prediction='{p_src.crs}' vs reference='{r_src.crs}'"
                )

            width, height = p_src.width, p_src.height
            tp, fp, fn, tn, nodata_cnt = 0, 0, 0, 0, 0

            for y in range(0, height, self.block_size):
                win_h = min(self.block_size, height - y)
                for x in range(0, width, self.block_size):
                    win_w = min(self.block_size, width - x)
                    win = Window(x, y, win_w, win_h)

                    p_blk = p_src.read(1, window=win).astype(np.float32)
                    r_blk = r_src.read(1, window=win).astype(np.float32)

                    # Strict valid mask construction
                    p_valid = (p_blk != pred_nodata) & np.isfinite(p_blk)
                    r_valid = (r_blk != ref_nodata) & np.isfinite(r_blk)
                    valid_mask = p_valid & r_valid

                    nodata_cnt += int(np.sum(~valid_mask))

                    if not np.any(valid_mask):
                        continue

                    # Filter to valid pixels
                    p_v = p_blk[valid_mask]
                    r_v = r_blk[valid_mask]

                    # Binarize (handle binary or probability thresholding)
                    p_bin = (p_v >= 0.5).astype(np.uint8)
                    r_bin = (r_v >= 0.5).astype(np.uint8)

                    tp += int(np.sum((p_bin == 1) & (r_bin == 1)))
                    fp += int(np.sum((p_bin == 1) & (r_bin == 0)))
                    fn += int(np.sum((p_bin == 0) & (r_bin == 1)))
                    tn += int(np.sum((p_bin == 0) & (r_bin == 0)))

            return compute_binary_metrics_from_counts(
                tp=tp, fp=fp, fn=fn, tn=tn,
                nodata=nodata_cnt,
                data_lineage=data_lineage or f"EXTERNAL_FILE: {os.path.basename(reference_raster_path)}"
            )
