"""SARA SmallUNet Segmentation Model for SAR Flood Mapping.

Optimized for 256x256 3-band SAR inputs (VV_dB, VH_dB, VV_VH_ratio)
and hardware execution on NVIDIA RTX 2050 (4 GB VRAM) / 16 GB system RAM.

Features:
1. Lightweight 4-stage encoder/decoder architecture with skip connections.
2. Masked BCE + Dice Loss ignoring label 255 (nodata / cloud invalid).
3. Precision, Recall, Dice/F1, and IoU metric tracking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm2d -> ReLU) * 2 block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class SmallUNet(nn.Module):
    """Lightweight U-Net architecture for binary SAR flood segmentation."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, init_features: int = 32):
        super().__init__()
        features = init_features

        # Encoder
        self.inc = DoubleConv(in_channels, features)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features, features * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features * 2, features * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features * 4, features * 8))

        # Decoder
        self.up1 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(features * 8, features * 4)

        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(features * 4, features * 2)

        self.up3 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(features * 2, features)

        # Output head
        self.outc = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)

        logits = self.outc(x)
        return logits


class MaskedBCEDiceLoss(nn.Module):
    """Combined Binary Cross Entropy and Dice Loss with explicit nodata masking (ignoring label 255)."""

    def __init__(self, nodata_value: int = 255, bce_weight: float = 0.5, pos_weight: Optional[float] = None, smooth: float = 1e-6):
        super().__init__()
        self.nodata_value = nodata_value
        self.bce_weight = bce_weight
        self.pos_weight = pos_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args:

        logits: Tensor of shape (B, 1, H, W)
        targets: Tensor of shape (B, 1, H, W) or (B, H, W)
        """
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        valid_mask = (targets != self.nodata_value)
        if not torch.any(valid_mask):
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits_valid = logits[valid_mask]
        targets_valid = targets[valid_mask].float()

        # BCE Loss with optional pos_weight
        if self.pos_weight is not None:
            pw_tensor = torch.tensor([self.pos_weight], device=logits.device)
            bce_loss = F.binary_cross_entropy_with_logits(logits_valid, targets_valid, pos_weight=pw_tensor)
        else:
            bce_loss = F.binary_cross_entropy_with_logits(logits_valid, targets_valid)

        # Dice Loss on probabilities
        probs = torch.sigmoid(logits_valid)
        intersection = torch.sum(probs * targets_valid)
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            torch.sum(probs) + torch.sum(targets_valid) + self.smooth
        )

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


def calculate_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    nodata_value: int = 255,
    threshold: float = 0.5
) -> dict[str, float]:
    """Calculate Precision, Recall, F1/Dice, and IoU metrics ignoring nodata pixels."""
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)

    valid_mask = (targets != nodata_value)
    if not torch.any(valid_mask):
        return {"precision": 0.0, "recall": 0.0, "f1_dice": 0.0, "iou": 0.0}

    probs = torch.sigmoid(logits[valid_mask])
    preds = (probs >= threshold).float()
    targets_valid = targets[valid_mask].float()

    tp = torch.sum(preds * targets_valid).item()
    fp = torch.sum(preds * (1.0 - targets_valid)).item()
    fn = torch.sum((1.0 - preds) * targets_valid).item()

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1_dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)

    return {
        "precision": precision,
        "recall": recall,
        "f1_dice": f1_dice,
        "iou": iou
    }
