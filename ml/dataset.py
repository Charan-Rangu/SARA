"""SARA Machine Learning Dataset & Leakage-Resistant Split Generator.

Supports:
1. Sen1Floods11 SAR flood dataset structure and metadata parsing.
2. Leakage-resistant train/validation/test split generation grouped by geographic event region.
3. Windowed tile/patch extraction (256x256) compatible with 4 GB VRAM GPUs (NVIDIA RTX 2050).
4. Mask encoding standards (0=non-water, 1=water, 255=invalid/cloud/nodata).

Important:
This module handles training dataset structure and manifests.
EMSR927 validation data MUST NOT be loaded or included in training splits.
"""

import os
import json
import random
from pathlib import Path
from typing import Optional, Any
from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    """Metadata detailing training dataset provenance and configuration."""

    dataset_name: str = "Sen1Floods11 (SAR Hand-Labeled Subset)"
    license: str = "CC BY 4.0"
    provenance: str = "Cloud to Street / Bonafilia et al. (2020)"
    sensor: str = "Sentinel-1 IW GRD (Dual-pol VV/VH)"
    patch_size: int = 256
    num_channels: int = 3
    channel_names: list[str] = Field(default_factory=lambda: ["VV_dB", "VH_dB", "VV_VH_ratio"])
    label_encoding: dict[str, int] = Field(default_factory=lambda: {"non_water": 0, "water": 1, "invalid_nodata": 255})
    hardware_target: str = "NVIDIA RTX 2050 (4GB VRAM) / 16GB System RAM"


class SplitManifest(BaseModel):
    """Manifest container for leakage-resistant train/val/test splits."""

    dataset_name: str
    total_tiles: int
    train_count: int
    val_count: int
    test_count: int
    train_events: list[str]
    val_events: list[str]
    test_events: list[str]
    train_tiles: list[str]
    val_tiles: list[str]
    test_tiles: list[str]


def generate_leakage_resistant_splits(
    tile_list: list[dict[str, Any]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> SplitManifest:
    """Group tiles by geographic event region to ensure zero spatial leakage between splits.

    Args:
        tile_list: List of dicts containing 'tile_id', 'event_id', and 'path'.
        val_ratio: Target proportion of events for validation.
        test_ratio: Target proportion of events for testing.
        seed: Random seed for reproducible splitting.

    Returns:
        SplitManifest object.
    """
    events = sorted(list(set(t["event_id"] for t in tile_list)))
    rng = random.Random(seed)
    rng.shuffle(events)

    num_events = len(events)
    num_val = max(1, int(num_events * val_ratio))
    num_test = max(1, int(num_events * test_ratio))

    val_events = set(events[:num_val])
    test_events = set(events[num_val : num_val + num_test])
    train_events = set(events[num_val + num_test :])

    train_tiles = [t["tile_id"] for t in tile_list if t["event_id"] in train_events]
    val_tiles = [t["tile_id"] for t in tile_list if t["event_id"] in val_events]
    test_tiles = [t["tile_id"] for t in tile_list if t["event_id"] in test_events]

    return SplitManifest(
        dataset_name="Sen1Floods11 (SAR Hand-Labeled)",
        total_tiles=len(tile_list),
        train_count=len(train_tiles),
        val_count=len(val_tiles),
        test_count=len(test_tiles),
        train_events=sorted(list(train_events)),
        val_events=sorted(list(val_events)),
        test_events=sorted(list(test_events)),
        train_tiles=train_tiles,
        val_tiles=val_tiles,
        test_tiles=test_tiles
    )
