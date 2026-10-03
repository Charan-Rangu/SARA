"""Sen1Floods11 Dataset Tile Generator & Split Manifest Builder for SARA.

Generates georeferenced 256x256 SAR chips (VV, VH, Ratio) and paired water masks
across 11 global flood event regions, and exports physical JSON split manifests.

Events:
1. Bolivia
2. Ghana
3. India
4. Mekong
5. Nigeria
6. Pakistan
7. Paraguay
8. Somalia
9. Spain
10. SriLanka
11. USA

Important:
This dataset is designated for model training and validation splits.
EMSR927 validation data MUST NOT be loaded or included in training splits.
"""

import os
import json
import numpy as np
import rasterio
from rasterio.transform import from_origin

from ml.dataset import generate_leakage_resistant_splits, DatasetMetadata

EVENTS = [
    "Bolivia", "Ghana", "India", "Mekong", "Nigeria",
    "Pakistan", "Paraguay", "Somalia", "Spain", "SriLanka", "USA"
]

OUTPUT_DIR = os.path.join("data", "ml", "sen1floods11")
MANIFEST_DIR = os.path.join("data", "ml")


def create_sample_sar_chip(chip_path: str, mask_path: str, event_id: str, index: int):
    """Generate a valid 256x256 3-band SAR GeoTIFF chip and paired binary label mask."""
    os.makedirs(os.path.dirname(chip_path), exist_ok=True)
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)

    transform = from_origin(85.0 + index * 0.05, 28.0 + index * 0.05, 0.0001, 0.0001)

    # 3-channel SAR array (VV dB, VH dB, ratio dB)
    rng = np.random.RandomState(seed=42 + index)
    vv = rng.normal(loc=-11.0, scale=3.0, size=(256, 256)).astype(np.float32)
    vh = rng.normal(loc=-18.0, scale=3.5, size=(256, 256)).astype(np.float32)
    ratio = vv - vh

    sar_data = np.stack([vv, vh, ratio], axis=0)

    # Binary water mask (0 = non-water, 1 = water, 255 = nodata)
    water_mask = (vv < -15.0) & (vh < -22.0)
    label_data = np.zeros((256, 256), dtype=np.uint8)
    label_data[water_mask] = 1
    # Add nodata border
    label_data[0, :] = 255
    label_data[-1, :] = 255

    sar_meta = {
        "driver": "GTiff",
        "height": 256,
        "width": 256,
        "count": 3,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": None
    }
    with rasterio.open(chip_path, "w", **sar_meta) as dst:
        dst.write(sar_data)

    mask_meta = {
        "driver": "GTiff",
        "height": 256,
        "width": 256,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": 255
    }
    with rasterio.open(mask_path, "w", **mask_meta) as dst:
        dst.write(label_data, 1)

    # Calculate pixel statistics
    valid_px = int(np.sum(label_data != 255))
    water_px = int(np.sum(label_data == 1))
    non_water_px = int(np.sum(label_data == 0))

    return {
        "tile_id": f"{event_id}_{index:03d}",
        "event_id": event_id,
        "sar_path": chip_path.replace("\\", "/"),
        "label_path": mask_path.replace("\\", "/"),
        "width": 256,
        "height": 256,
        "num_channels": 3,
        "valid_pixels": valid_px,
        "water_pixels": water_px,
        "non_water_pixels": non_water_px
    }


def build_sen1floods11_dataset():
    """Build physical sample tiles and leakage-resistant split manifests."""
    print("Building Sen1Floods11 sample dataset chips and leakage-resistant manifests...")
    tiles = []

    global_idx = 0
    for event in EVENTS:
        # Generate 4 chips per event region -> 44 total chips
        for i in range(4):
            chip_p = os.path.join(OUTPUT_DIR, "images", f"{event}_{i:03d}_sar.tif")
            mask_p = os.path.join(OUTPUT_DIR, "labels", f"{event}_{i:03d}_label.tif")
            meta_tile = create_sample_sar_chip(chip_p, mask_p, event, global_idx)
            tiles.append(meta_tile)
            global_idx += 1

    # Generate Leakage-Resistant Event-Disjoint Splits
    split_manifest = generate_leakage_resistant_splits(tiles, val_ratio=0.18, test_ratio=0.18, seed=42)

    os.makedirs(MANIFEST_DIR, exist_ok=True)
    manifest_dict = split_manifest.model_dump()
    manifest_dict["tiles_metadata"] = tiles

    # Export Full Dataset Inventory Manifest
    inv_path = os.path.join(MANIFEST_DIR, "dataset_inventory.json")
    with open(inv_path, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2)

    # Export Separate Split Manifests
    train_manifest = [t for t in tiles if t["tile_id"] in set(split_manifest.train_tiles)]
    val_manifest = [t for t in tiles if t["tile_id"] in set(split_manifest.val_tiles)]
    test_manifest = [t for t in tiles if t["tile_id"] in set(split_manifest.test_tiles)]

    with open(os.path.join(MANIFEST_DIR, "train_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"event_ids": split_manifest.train_events, "tiles": train_manifest}, f, indent=2)

    with open(os.path.join(MANIFEST_DIR, "val_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"event_ids": split_manifest.val_events, "tiles": val_manifest}, f, indent=2)

    with open(os.path.join(MANIFEST_DIR, "test_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"event_ids": split_manifest.test_events, "tiles": test_manifest}, f, indent=2)

    print("Sen1Floods11 Dataset Preparation Complete!")
    print(f"Total Tiles: {split_manifest.total_tiles}")
    print(f"Train Tiles ({len(split_manifest.train_events)} events: {split_manifest.train_events}): {split_manifest.train_count}")
    print(f"Val Tiles ({len(split_manifest.val_events)} events: {split_manifest.val_events}): {split_manifest.val_count}")
    print(f"Test Tiles ({len(split_manifest.test_events)} events: {split_manifest.test_events}): {split_manifest.test_count}")
    print(f"Manifests saved to: {MANIFEST_DIR}")


if __name__ == "__main__":
    build_sen1floods11_dataset()
