"""Sentinel-1 SAR Windowed Change Detection Engine for SARA.

Performs memory-efficient, windowed logarithmic backscatter change detection and preliminary mask generation
for two calibrated, terrain-corrected, co-registered Sentinel-1 SAR rasters.

Important:
Output masks are titled 'preliminary_change_mask.tif'.
This module identifies changes in radar backscatter. It does NOT, by itself, prove that a location is flooded,
that a building is damaged, or that a road is impassable.
"""

import os
from typing import Optional, Any
import numpy as np
from pydantic import BaseModel, Field
import rasterio
from rasterio.windows import Window


class ChangeDetectionConfig(BaseModel):
    """Configuration options for SAR change detection processing."""

    threshold_db: float = -3.0  # Decibel threshold for backscatter reduction (e.g. -3.0 dB)
    input_units: str = "linear"  # "linear" (for calibrated sigma0 power) or "dB"
    block_size: int = 256        # Window size for tile processing
    epsilon: float = 1e-7        # Numerical stability constant for log calculations


class ChangeDetectionResult(BaseModel):
    """Result summary of SAR change detection execution."""

    output_raster_path: str
    threshold_db: float
    input_units: str
    total_valid_pixels: int
    changed_pixels: int
    change_percentage: float
    pre_product_id: Optional[str] = None
    post_product_id: Optional[str] = None
    processing_metadata: dict[str, Any] = Field(default_factory=dict)
    disclaimer: str = (
        "Preliminary SAR change detection. Identifies changes in radar backscatter. "
        "Does not prove confirmed flooding, building damage, or road blockage without contextual analysis."
    )


class SARChangeDetector:
    """Windowed raster change detection engine for Sentinel-1 SAR backscatter images."""

    def __init__(self, config: Optional[ChangeDetectionConfig] = None):
        self.config = config or ChangeDetectionConfig()

    def detect_changes(
        self,
        pre_raster_path: str,
        post_raster_path: str,
        output_mask_path: Optional[str] = None,
        pre_product_id: Optional[str] = None,
        post_product_id: Optional[str] = None
    ) -> ChangeDetectionResult:
        """Process pre/post calibrated SAR rasters in memory-efficient windows.

        Args:
            pre_raster_path: Path to calibrated pre-event GeoTIFF.
            post_raster_path: Path to calibrated post-event GeoTIFF.
            output_mask_path: Target path for preliminary change mask raster.
            pre_product_id: Optional identifier/title of pre-event satellite scene.
            post_product_id: Optional identifier/title of post-event satellite scene.

        Returns:
            ChangeDetectionResult object with output raster path, statistics, and metadata.
        """
        if not os.path.exists(pre_raster_path):
            raise FileNotFoundError(f"Pre-event raster file not found: {pre_raster_path}")
        if not os.path.exists(post_raster_path):
            raise FileNotFoundError(f"Post-event raster file not found: {post_raster_path}")

        if output_mask_path is None:
            out_dir = os.path.dirname(post_raster_path) or os.path.join("data", "processed")
            os.makedirs(out_dir, exist_ok=True)
            output_mask_path = os.path.join(out_dir, "preliminary_change_mask.tif")
        else:
            out_dir = os.path.dirname(os.path.abspath(output_mask_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

        # Open both rasters and verify compatibility
        with rasterio.open(pre_raster_path) as src_pre, rasterio.open(post_raster_path) as src_post:
            # 1. Dimension Verification
            if (src_pre.width != src_post.width) or (src_pre.height != src_post.height):
                raise ValueError(
                    f"Raster dimension mismatch: pre=({src_pre.width}x{src_pre.height}) vs "
                    f"post=({src_post.width}x{src_post.height})"
                )

            # 2. CRS Verification
            if src_pre.crs != src_post.crs:
                raise ValueError(f"Raster CRS mismatch: pre='{src_pre.crs}' vs post='{src_post.crs}'")

            # 3. Affine Transform Verification
            if not np.allclose(src_pre.transform, src_post.transform, atol=1e-5):
                raise ValueError(
                    f"Raster Affine Transform mismatch: pre={src_pre.transform} vs post={src_post.transform}"
                )

            width = src_pre.width
            height = src_pre.height
            crs = src_pre.crs
            transform = src_pre.transform

            pre_nodata = src_pre.nodata
            post_nodata = src_post.nodata

            # Prepare output GeoTIFF metadata
            out_meta = src_pre.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": 1,
                "dtype": rasterio.uint8,
                "nodata": 255,
                "compress": "lzw"
            })

            total_valid_pixels = 0
            total_changed_pixels = 0

            # Windowed Processing loop to keep RAM footprint low (< 50MB)
            block_sz = self.config.block_size
            with rasterio.open(output_mask_path, "w", **out_meta) as dst_out:
                for y in range(0, height, block_sz):
                    win_height = min(block_sz, height - y)
                    for x in range(0, width, block_sz):
                        win_width = min(block_sz, width - x)
                        window = Window(x, y, win_width, win_height)

                        pre_block = src_pre.read(1, window=window).astype(np.float32)
                        post_block = src_post.read(1, window=window).astype(np.float32)

                        # Create Nodata Mask (strictly exclude nodata, NaN, and +/- Inf)
                        valid_mask = np.ones_like(pre_block, dtype=bool)
                        if pre_nodata is not None:
                            valid_mask &= (pre_block != pre_nodata) & np.isfinite(pre_block)
                        else:
                            valid_mask &= np.isfinite(pre_block)

                        if post_nodata is not None:
                            valid_mask &= (post_block != post_nodata) & np.isfinite(post_block)
                        else:
                            valid_mask &= np.isfinite(post_block)

                        out_mask_block = np.full((win_height, win_width), fill_value=255, dtype=np.uint8)

                        if np.any(valid_mask):
                            # Backscatter change calculation
                            if self.config.input_units.lower() == "db":
                                # Inputs in decibels: difference Delta_dB = post_dB - pre_dB
                                diff_db = post_block - pre_block
                            else:
                                # Inputs in linear power: 10 * log10((post + eps) / (pre + eps))
                                safe_pre = np.maximum(pre_block, self.config.epsilon)
                                safe_post = np.maximum(post_block, self.config.epsilon)
                                diff_db = 10.0 * np.log10(safe_post / safe_pre)

                            # Potential flood water detection: significant reduction in backscatter (Delta_dB <= threshold_db)
                            change_condition = (diff_db <= self.config.threshold_db) & valid_mask

                            out_mask_block[valid_mask] = 0
                            out_mask_block[change_condition] = 1

                            num_valid = int(np.sum(valid_mask))
                            num_changed = int(np.sum(change_condition))

                            total_valid_pixels += num_valid
                            total_changed_pixels += num_changed

                        dst_out.write(out_mask_block, 1, window=window)

                # Set custom metadata tags
                dst_out.update_tags(
                    DESCRIPTION="Preliminary SAR Change Detection Mask",
                    THRESHOLD_DB=str(self.config.threshold_db),
                    INPUT_UNITS=self.config.input_units,
                    PRE_PRODUCT_ID=str(pre_product_id or "UNKNOWN"),
                    POST_PRODUCT_ID=str(post_product_id or "UNKNOWN"),
                    DISCLAIMER="Preliminary SAR change detection. Identifies changes in radar backscatter."
                )

        pct = (total_changed_pixels / total_valid_pixels * 100.0) if total_valid_pixels > 0 else 0.0

        return ChangeDetectionResult(
            output_raster_path=output_mask_path,
            threshold_db=self.config.threshold_db,
            input_units=self.config.input_units,
            total_valid_pixels=total_valid_pixels,
            changed_pixels=total_changed_pixels,
            change_percentage=round(pct, 2),
            pre_product_id=pre_product_id,
            post_product_id=post_product_id,
            processing_metadata={
                "raster_width": width,
                "raster_height": height,
                "crs": str(crs),
                "transform": [float(t) for t in list(transform)[:6]],
                "pre_product_id": pre_product_id,
                "post_product_id": post_product_id,
                "threshold_db": self.config.threshold_db,
                "input_units": self.config.input_units
            }
        )


class AgreementMaskResult(BaseModel):
    """Result summary of 5-class multi-temporal agreement mask generation."""

    output_raster_path: str
    total_valid_pixels: int
    category_counts: dict[str, int]
    category_percentages: dict[str, float]
    disclaimer: str = (
        "Exploratory 5-class agreement analysis. Class 3 represents an agreed inundation candidate, "
        "not confirmed flood extent. Class 4 represents a suspected high-slope shadow candidate, "
        "not confirmed radar shadow."
    )


def generate_agreement_mask(
    water_mask_path: str,
    change_mask_path: str,
    slope_candidate_mask_path: str,
    output_path: str,
    block_size: int = 2048
) -> AgreementMaskResult:
    """Generate memory-efficient 5-class categorical agreement mask raster.

    Classes:
        0: Dry Land / No Change
        1: Persistent Water Candidate (Water == 1, Slope <= 15°, Change == 0)
        2: Backscatter Change Only (Water == 0, Change == 1)
        3: Agreed Inundation Candidate (Water == 1, Slope <= 15°, Change == 1)
        4: Suspected High-Slope Shadow Candidate (Water == 1, Slope > 15°)
        255: Nodata / Invalid Observation

    Args:
        water_mask_path: Path to binary model prediction raster (1=water, 0=land, 255=nodata).
        change_mask_path: Path to SAR change mask raster (1=change, 0=no change, 255=nodata).
        slope_candidate_mask_path: Path to slope <= 15° retained mask (1=slope<=15°, 0=slope>15°, 255=nodata).
        output_path: Destination path for 5-class GeoTIFF.
        block_size: Window size for block-by-block processing.

    Returns:
        AgreementMaskResult with pixel counts and percentages.
    """
    for p in [water_mask_path, change_mask_path, slope_candidate_mask_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Input raster not found: {p}")

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(water_mask_path) as w_src, \
         rasterio.open(change_mask_path) as c_src, \
         rasterio.open(slope_candidate_mask_path) as s_src:

        if (w_src.width != c_src.width) or (w_src.height != c_src.height):
            raise ValueError(f"Raster dimension mismatch: water={w_src.shape} vs change={c_src.shape}")
        if w_src.crs != c_src.crs:
            raise ValueError(f"Raster CRS mismatch: water={w_src.crs} vs change={c_src.crs}")

        width, height = w_src.width, w_src.height
        profile = w_src.profile.copy()
        profile.update(
            dtype=rasterio.uint8,
            nodata=255,
            compress="lzw",
            BIGTIFF="IF_SAFER" if (width * height > 100_000_000) else "NO"
        )

        cat_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 255: 0}

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.update_tags(
                DESCRIPTION="SARA Multi-Temporal Agreement Mask",
                VALUES="0=Dry Land, 1=Persistent Water Candidate, 2=Change Only, 3=Agreed Inundation Candidate, 4=Suspected High-Slope Shadow Candidate, 255=Nodata",
                DISCLAIMER="Exploratory multi-temporal analysis. Not confirmed flood extent."
            )

            for y in range(0, height, block_size):
                win_h = min(block_size, height - y)
                for x in range(0, width, block_size):
                    win_w = min(block_size, width - x)
                    win = Window(x, y, win_w, win_h)

                    w_blk = w_src.read(1, window=win)
                    c_blk = c_src.read(1, window=win)
                    s_blk = s_src.read(1, window=win)

                    valid_mask = (w_blk != 255) & (c_blk != 255) & (s_blk != 255)

                    agr_blk = np.full((win_h, win_w), 255, dtype=np.uint8)
                    agr_blk[valid_mask] = 0

                    is_w = (w_blk == 1) & valid_mask
                    is_c = (c_blk == 1) & valid_mask
                    is_s15_w = (s_blk == 1) & is_w
                    is_shadow = is_w & (s_blk == 0)

                    # Class assignments
                    agr_blk[is_s15_w & ~is_c] = 1   # Persistent Water Candidate
                    agr_blk[is_c & ~is_w] = 2       # Change Only
                    agr_blk[is_s15_w & is_c] = 3    # Agreed Inundation Candidate
                    agr_blk[is_shadow] = 4          # Suspected High-Slope Shadow Candidate

                    dst.write(agr_blk, 1, window=win)

                    for k in cat_counts.keys():
                        cat_counts[k] += int(np.sum(agr_blk == k))

    total_valid = sum(cat_counts[k] for k in [0, 1, 2, 3, 4])
    percentages = {
        f"class_{k}": round((cat_counts[k] / total_valid * 100.0), 2) if total_valid > 0 else 0.0
        for k in [0, 1, 2, 3, 4]
    }

    named_counts = {
        "0_dry_land": cat_counts[0],
        "1_persistent_water_candidate": cat_counts[1],
        "2_change_only": cat_counts[2],
        "3_agreed_inundation_candidate": cat_counts[3],
        "4_suspected_shadow_candidate": cat_counts[4],
        "255_nodata": cat_counts[255]
    }

    return AgreementMaskResult(
        output_raster_path=output_path,
        total_valid_pixels=total_valid,
        category_counts=named_counts,
        category_percentages=percentages
    )

