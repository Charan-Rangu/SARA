"""PyTorch Dataset & DataLoader for SARA Small U-Net SAR Flood Segmentation."""

import os
import json
import torch
import numpy as np
import rasterio
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any, List, Optional


class SARAFloodDataset(Dataset):
    """PyTorch Dataset reading Sentinel-1 3-band SAR chips and 1-band label masks."""

    def __init__(
        self,
        manifest_path: str,
        channel_means: List[float] = [-11.0, -18.0, 7.0],
        channel_stds: List[float] = [3.0, 3.5, 2.5],
        nodata_value: int = 255
    ):
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest path not found: {manifest_path}")

        with open(manifest_path, "r") as f:
            manifest_data = json.load(f)

        self.tiles = manifest_data.get("tiles", [])
        if len(self.tiles) == 0:
            raise ValueError(f"No tiles found in manifest {manifest_path}")

        self.channel_means = np.array(channel_means, dtype=np.float32).reshape(3, 1, 1)
        self.channel_stds = np.array(channel_stds, dtype=np.float32).reshape(3, 1, 1)
        self.nodata_value = nodata_value

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        tile = self.tiles[idx]
        sar_path = tile["sar_path"]
        label_path = tile["label_path"]
        tile_id = tile["tile_id"]

        with rasterio.open(sar_path) as src_sar:
            sar_img = src_sar.read()  # Shape: (3, H, W), float32

        with rasterio.open(label_path) as src_lbl:
            label_img = src_lbl.read(1)  # Shape: (H, W), uint8

        # Channel-wise Z-score normalization
        norm_sar = (sar_img - self.channel_means) / (self.channel_stds + 1e-7)

        # Convert to PyTorch Tensors
        image_tensor = torch.from_numpy(norm_sar).float()
        label_tensor = torch.from_numpy(label_img).unsqueeze(0).long()

        return image_tensor, label_tensor, tile_id


def create_dataloader(
    manifest_path: str,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0,
    channel_means: List[float] = [-11.0, -18.0, 7.0],
    channel_stds: List[float] = [3.0, 3.5, 2.5],
    nodata_value: int = 255
) -> DataLoader:
    """Create PyTorch DataLoader for SARA flood segmentation."""
    dataset = SARAFloodDataset(
        manifest_path=manifest_path,
        channel_means=channel_means,
        channel_stds=channel_stds,
        nodata_value=nodata_value
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
