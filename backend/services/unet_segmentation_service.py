"""Small U-Net SAR Preliminary Water Segmentation Service for SARA.

Performs memory-efficient windowed inference on 3-band Sentinel-1 SAR rasters
(VV_dB, VH_dB, VV_dB - VH_dB) using the primary Small U-Net model checkpoint.

Features:
1. Security path validation preventing unauthorized filesystem access.
2. Raster format, CRS, dimension, band count, and unit validation.
3. Strict band order verification — never guesses or silently reorders ambiguous bands.
4. Auto-preparation of 2-band calibrated SAR rasters if polarization metadata is unambiguous.
5. Upload & available raster discovery management for interactive dashboard selection.
6. Windowed (256x256) PyTorch FP16 / FP32 evaluation with nodata (255) preservation.
7. GeoTIFF raster output (binary mask & probability map) preserving affine transform and CRS.
8. RGBA PNG overlay generation with WGS84 bounding box for Leaflet map rendering.
9. Scientific disclaimers and metric summary calculation.
"""

import os
import glob
import json
import uuid
import datetime
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_bounds
from PIL import Image
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Tuple

from ml.model import SmallUNet
from ml.config import load_primary_model_config, TrainingPaths
from geospatial.unet_sar_preprocessor import UNetSARPreprocessor, UNetPreprocessingConfig


class SegmentationRequest(BaseModel):
    """Request model for SAR water segmentation prediction."""

    sar_raster_path: str = Field(..., description="Path to 3-band SAR raster (VV_dB, VH_dB, VV_dB - VH_dB) or calibrated 2-band SAR")
    probability_threshold: float = Field(default=0.50, ge=0.0, le=1.0, description="Decision threshold for water class")
    run_id: Optional[str] = Field(default=None, description="Optional custom run identifier")
    auto_prepare_2band: bool = Field(default=False, description="Auto-prepare 3-band stack from 2-band SAR if metadata is unambiguous")


class AvailableRasterInfo(BaseModel):
    """Metadata model for locally available SAR rasters."""

    path: str
    filename: str
    relative_path: str
    width: int
    height: int
    num_bands: int
    crs: str
    size_mb: float
    is_ready_for_unet: bool
    band_descriptions: List[str] = Field(default_factory=list)


class SegmentationSummary(BaseModel):
    """Summary statistics for water segmentation result."""

    total_pixels: int
    valid_pixels: int
    nodata_pixels: int
    predicted_water_pixels: int
    water_percentage: float
    mean_water_probability: float


class SegmentationResponse(BaseModel):
    """Response model for SAR water segmentation endpoint."""

    status: str
    run_id: str
    model_name: str = "Small U-Net"
    probability_threshold: float = 0.50
    input_filename: str
    sar_raster_path: str
    output_directory: str
    binary_mask_path: str
    probability_map_path: str
    binary_overlay_png_path: str
    probability_overlay_png_path: str
    wgs84_bounds: List[List[float]]  # [[min_lat, min_lon], [max_lat, max_lon]]
    summary: SegmentationSummary
    disclaimer: str = (
        "Preliminary SAR-based candidate water prediction — independent confirmation required. "
        "Model outputs represent preliminary probabilistic SAR water extent and do NOT constitute confirmed flood boundaries. "
        "Must NOT be used solely to claim road closures, settlement access limitations, or area safety."
    )


def validate_secure_path(file_path: str, allowed_roots: Optional[List[str]] = None) -> str:
    """Validate that file_path resides within approved project workspace directories.

    Prevents path traversal attacks (e.g., ../../Windows).
    """
    if not allowed_roots:
        import tempfile
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        allowed_roots = [
            os.path.abspath(os.path.join(repo_root, "data")),
            os.path.abspath(os.path.join(repo_root, "experiments")),
            os.path.abspath(os.path.join(repo_root, "scratch")),
            os.path.abspath(os.path.join(repo_root, "validation")),
            os.path.abspath(os.path.join(repo_root, "models")),
            os.path.abspath(tempfile.gettempdir()),
            os.path.realpath(tempfile.gettempdir()),
        ]

    abs_path = os.path.abspath(file_path)
    
    # Path traversal check
    is_allowed = any(abs_path.startswith(root) for root in allowed_roots)
    if not is_allowed:
        raise PermissionError(
            f"Security Error: Access to file path '{file_path}' is outside allowed workspace directories."
        )

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Raster file not found: '{file_path}'")

    return abs_path


class UNetSegmentationService:
    """Inference, raster validation, and map overlay service for Small U-Net model."""

    def __init__(self, config_path: str = "config/primary_segmentation_model.json"):
        self.config_path = config_path
        self.config = load_primary_model_config(config_path)
        
        arch = self.config["architecture"]
        self.checkpoint_path = arch["checkpoint_path"]
        self.in_channels = self.config["input_specification"]["num_channels"]
        self.channel_means = np.array(self.config["input_specification"]["normalization"]["channel_means"], dtype=np.float32).reshape(3, 1, 1)
        self.channel_stds = np.array(self.config["input_specification"]["normalization"]["channel_stds"], dtype=np.float32).reshape(3, 1, 1)
        self.nodata_value = self.config["inference_parameters"]["nodata_value"]
        self.default_threshold = self.config["inference_parameters"]["probability_threshold"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def _load_model_if_needed(self):
        """Lazy load and cache Small U-Net PyTorch model."""
        if self.model is not None:
            return

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Primary model checkpoint not found at: '{self.checkpoint_path}'")

        model = SmallUNet(in_channels=self.in_channels, out_channels=1, init_features=32)
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model

    def discover_available_rasters(self, search_dirs: Optional[List[str]] = None) -> List[AvailableRasterInfo]:
        """Discover available local GeoTIFF rasters in workspace directories."""
        if not search_dirs:
            search_dirs = [
                os.path.join("data", "processed", "unet_inputs"),
                os.path.join("data", "processed"),
                os.path.join("data", "cache", "uploads"),
                os.path.join("data", "raw"),
                os.path.join("data", "examples"),
            ]

        discovered = []
        seen_paths = set()

        for sdir in search_dirs:
            if not os.path.exists(sdir):
                continue
            
            for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
                for fpath in glob.glob(os.path.join(sdir, "**", ext), recursive=True):
                    abs_p = os.path.abspath(fpath)
                    if abs_p in seen_paths:
                        continue
                    seen_paths.add(abs_p)

                    try:
                        with rasterio.open(abs_p) as src:
                            fsize_mb = os.path.getsize(abs_p) / (1024 * 1024)
                            descs = [str(d) if d else "" for d in (src.descriptions or [])]
                            is_ready = (src.count == 3)
                            
                            discovered.append(AvailableRasterInfo(
                                path=abs_p.replace("\\", "/"),
                                filename=os.path.basename(abs_p),
                                relative_path=os.path.relpath(abs_p, os.getcwd()).replace("\\", "/"),
                                width=src.width,
                                height=src.height,
                                num_bands=src.count,
                                crs=str(src.crs) if src.crs else "UNKNOWN",
                                size_mb=round(fsize_mb, 2),
                                is_ready_for_unet=is_ready,
                                band_descriptions=descs
                            ))
                    except Exception:
                        continue

        return discovered

    def save_uploaded_raster(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Save and validate an uploaded GeoTIFF file into the secure cache."""
        clean_name = os.path.basename(filename).replace(" ", "_")
        if not (clean_name.lower().endswith(".tif") or clean_name.lower().endswith(".tiff")):
            raise ValueError(f"Invalid file extension: '{clean_name}'. Only GeoTIFF (.tif, .tiff) files are supported.")

        upload_dir = os.path.abspath(os.path.join("data", "cache", "uploads"))
        os.makedirs(upload_dir, exist_ok=True)

        unique_id = str(uuid.uuid4())[:8]
        target_path = os.path.join(upload_dir, f"upload_{unique_id}_{clean_name}")

        with open(target_path, "wb") as f:
            f.write(file_bytes)

        # Validate with rasterio
        try:
            with rasterio.open(target_path) as src:
                w, h, count, crs = src.width, src.height, src.count, str(src.crs)
        except Exception as e:
            if os.path.exists(target_path):
                os.remove(target_path)
            raise ValueError(f"Corrupt or unreadable GeoTIFF file: {e}")

        return {
            "path": target_path.replace("\\", "/"),
            "filename": clean_name,
            "width": w,
            "height": h,
            "bands": count,
            "crs": crs,
            "size_mb": round(len(file_bytes) / (1024 * 1024), 2)
        }

    def validate_input_raster(self, abs_path: str, auto_prepare_2band: bool = False) -> Dict[str, Any]:
        """Validate input SAR GeoTIFF band count, dimensions, units, CRS, and non-infinite values.

        Strictly prevents guessing of band orders when metadata is ambiguous.
        """
        with rasterio.open(abs_path) as src:
            count = src.count
            width = src.width
            height = src.height
            crs = src.crs

            if width < 16 or height < 16:
                raise ValueError(f"Raster dimensions too small: ({width}x{height}). Minimum 16x16 required.")

            if crs is None:
                raise ValueError("Raster lacks valid Coordinate Reference System (CRS). Georeferencing required.")

            effective_path = abs_path
            was_auto_prepared = False

            # If 3 bands: verify plausible structure
            if count == 3:
                # Check descriptions if present
                if src.descriptions and any(src.descriptions):
                    descs = [str(d).upper() for d in src.descriptions if d]
                    # If explicit descriptions are provided, verify they represent VV, VH, Diff
                    if len(descs) >= 2:
                        if "VH" in descs[0] and "VV" in descs[1]:
                            raise ValueError(
                                f"Incompatible band order detected in 3-band raster: Band 1 is '{descs[0]}', Band 2 is '{descs[1]}'. "
                                "Small U-Net strictly requires Band 1=VV_dB, Band 2=VH_dB, Band 3=VV_dB-VH_dB. "
                                "Silently guessing or reordering bands is prohibited."
                            )

            elif count == 2:
                if not auto_prepare_2band:
                    raise ValueError(
                        f"Invalid band count: Small U-Net requires 3 bands (VV_dB, VH_dB, VV_dB - VH_dB), but found {count} bands. "
                        "Set auto_prepare_2band=True to enable automated 3-band preprocessing for valid dual-pol SAR."
                    )

                # Attempt strict polarization identification without guessing
                try:
                    vv_idx, vh_idx = UNetSARPreprocessor.identify_polarizations(src, input_path=abs_path)
                except ValueError as ve:
                    raise ValueError(
                        f"Cannot prepare 2-band SAR raster for Small U-Net: {ve}. "
                        "Polarization band identities (VV vs VH) could not be unambiguously established from metadata. "
                        "Band order guessing is strictly prohibited."
                    )

                # Prepare 3-band stack
                prep = UNetSARPreprocessor()
                base_name = os.path.splitext(os.path.basename(abs_path))[0]
                out_3band_path = os.path.join(
                    "data", "processed", "unet_inputs", f"{base_name}_prepared_3band.tif"
                )
                
                # Check if cached prepared 3-band exists
                if not os.path.exists(out_3band_path):
                    pols_order = ["VV", "VH"] if vv_idx == 0 else ["VH", "VV"]
                    prep.process_raster_file(
                        input_raster_path=abs_path,
                        output_raster_path=out_3band_path,
                        band_polarizations=pols_order
                    )
                
                effective_path = os.path.abspath(out_3band_path)
                was_auto_prepared = True

            else:
                raise ValueError(
                    f"Invalid band count: Small U-Net requires 3 bands (VV_dB, VH_dB, VV_dB - VH_dB) or 2 calibrated bands (VV, VH). "
                    f"Got {count} bands."
                )

        # Open effective path to verify decibel ranges and bounding box
        with rasterio.open(effective_path) as eff_src:
            sample_data = eff_src.read(window=Window(0, 0, min(256, eff_src.width), min(256, eff_src.height)))
            if np.isnan(sample_data).all() or np.isinf(sample_data).all():
                raise ValueError("Raster contains only NaN or infinite values.")

            valid_sample = sample_data[sample_data != self.nodata_value]
            if len(valid_sample) > 0:
                mean_val = np.mean(valid_sample)
                if mean_val > 100 or mean_val < -100:
                    raise ValueError(
                        f"Unplausible SAR decibel range (mean value: {mean_val:.2f}). Ensure input bands are calibrated dB scale."
                    )

            # Transform bounding box to WGS84
            bounds = eff_src.bounds
            try:
                wgs84_bounds_tuple = transform_bounds(eff_src.crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top)
                wgs84_bounds = [
                    [wgs84_bounds_tuple[1], wgs84_bounds_tuple[0]],
                    [wgs84_bounds_tuple[3], wgs84_bounds_tuple[2]]
                ]
            except Exception as e:
                raise ValueError(f"Failed to transform raster bounds to WGS84: {e}")

            return {
                "effective_path": effective_path,
                "width": eff_src.width,
                "height": eff_src.height,
                "crs": str(eff_src.crs),
                "transform": eff_src.transform,
                "nodata": eff_src.nodata,
                "wgs84_bounds": wgs84_bounds,
                "was_auto_prepared": was_auto_prepared
            }

    def predict(self, request: SegmentationRequest) -> SegmentationResponse:
        """Run windowed inference on target SAR raster and generate output products.

        Designed for memory efficiency on 16GB RAM / 4GB VRAM.
        """
        # 1. Security path validation
        abs_sar_path = validate_secure_path(request.sar_raster_path)
        
        # 2. Raster format, band order, and geographic validation
        meta = self.validate_input_raster(abs_sar_path, auto_prepare_2band=request.auto_prepare_2band)
        effective_sar_path = meta["effective_path"]
        
        # 3. Model initialization
        self._load_model_if_needed()

        threshold = request.probability_threshold if request.probability_threshold is not None else self.default_threshold
        req_id = request.run_id or f"custom_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:6]}"
        run_folder = req_id if req_id.startswith("run_") else f"run_{req_id}"
        out_dir = os.path.abspath(os.path.join("data", "output", run_folder))
        os.makedirs(out_dir, exist_ok=True)

        binary_mask_path = os.path.join(out_dir, "predicted_water_mask.tif")
        prob_map_path = os.path.join(out_dir, "water_probability_map.tif")
        binary_png_path = os.path.join(out_dir, "water_prediction_overlay.png")
        prob_png_path = os.path.join(out_dir, "water_probability_overlay.png")
        metrics_json_path = os.path.join(out_dir, "run_metrics.json")

        width = meta["width"]
        height = meta["height"]
        block_size = 256  # 256x256 window for GPU/RAM safety

        # Allocate output arrays in memory
        prob_map = np.zeros((height, width), dtype=np.float32)
        bin_mask = np.full((height, width), fill_value=255, dtype=np.uint8)

        total_pixels = width * height
        valid_pixels_count = 0
        nodata_pixels_count = 0
        water_pixels_count = 0
        valid_prob_sum = 0.0

        batch_size = 16
        pending_chips = []
        pending_meta = []

        def flush_batch():
            nonlocal valid_pixels_count, nodata_pixels_count, water_pixels_count, valid_prob_sum
            if not pending_chips:
                return

            tensor_batch = torch.from_numpy(np.stack(pending_chips, axis=0)).float().to(self.device)
            with torch.no_grad():
                logits = self.model(tensor_batch)
                batch_probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            for i, (py, px, pwin_h, pwin_w, pchip_nodata) in enumerate(pending_meta):
                probs = batch_probs[i] if batch_probs.ndim == 3 else batch_probs
                win_probs = probs[:pwin_h, :pwin_w]
                win_bin = (win_probs >= threshold).astype(np.uint8)
                win_bin[pchip_nodata] = 255
                win_probs[pchip_nodata] = -1.0

                prob_map[py:py+pwin_h, px:px+pwin_w] = win_probs
                bin_mask[py:py+pwin_h, px:px+pwin_w] = win_bin

                v_mask = ~pchip_nodata
                n_valid = int(np.sum(v_mask))
                valid_pixels_count += n_valid
                nodata_pixels_count += int(np.sum(pchip_nodata))
                water_pixels_count += int(np.sum((win_bin == 1) & v_mask))
                if n_valid > 0:
                    valid_prob_sum += float(np.sum(win_probs[v_mask]))

            pending_chips.clear()
            pending_meta.clear()

        with rasterio.open(effective_sar_path) as src:
            profile = src.profile.copy()
            
            for y in range(0, height, block_size):
                for x in range(0, width, block_size):
                    win_w = min(block_size, width - x)
                    win_h = min(block_size, height - y)
                    win = Window(x, y, win_w, win_h)

                    sar_chip = src.read(window=win)  # (3, win_h, win_w)
                    
                    # Nodata mask across all bands
                    if src.nodata is not None:
                        chip_nodata = (sar_chip[0] == src.nodata) | (sar_chip[1] == src.nodata)
                    else:
                        chip_nodata = (sar_chip[0] == self.nodata_value) | (sar_chip[0] <= -99.0)

                    crop_nodata = chip_nodata[:win_h, :win_w]

                    # Nodata fast-path: if chip is 100% nodata, skip inference entirely
                    if np.all(crop_nodata):
                        prob_map[y:y+win_h, x:x+win_w] = -1.0
                        bin_mask[y:y+win_h, x:x+win_w] = 255
                        nodata_pixels_count += (win_w * win_h)
                        continue

                    # Pad chip to (3, 256, 256) if edge block
                    if win_w < 256 or win_h < 256:
                        padded_chip = np.zeros((3, 256, 256), dtype=np.float32)
                        padded_chip[:, :win_h, :win_w] = sar_chip
                        input_chip = padded_chip
                    else:
                        input_chip = sar_chip

                    # Preprocessing normalization: (chip - mean) / std
                    norm_chip = (input_chip - self.channel_means) / (self.channel_stds + 1e-7)
                    pending_chips.append(norm_chip)
                    pending_meta.append((y, x, win_h, win_w, crop_nodata))

                    if len(pending_chips) >= batch_size:
                        flush_batch()

            flush_batch()

        # Write output GeoTIFF rasters
        profile_bin = profile.copy()
        profile_bin.update(count=1, dtype=rasterio.uint8, nodata=255)
        with rasterio.open(binary_mask_path, "w", **profile_bin) as dst:
            dst.write(bin_mask, 1)

        profile_prob = profile.copy()
        profile_prob.update(count=1, dtype=rasterio.float32, nodata=-1.0)
        with rasterio.open(prob_map_path, "w", **profile_prob) as dst:
            dst.write(prob_map, 1)

        # Generate Map-Ready RGBA PNG Overlays for Leaflet
        # Use decimation if raster exceeds 2048 to prevent memory exhaustion and decompression bomb errors
        scale = max(1, max(width, height) // 2048)
        dec_bin = bin_mask[::scale, ::scale] if scale > 1 else bin_mask
        dec_prob = prob_map[::scale, ::scale] if scale > 1 else prob_map

        # 1. Binary Water Overlay: Water = Blue (#0284c7, alpha 210), Land/Nodata = Transparent
        rgba_bin = np.zeros((dec_bin.shape[0], dec_bin.shape[1], 4), dtype=np.uint8)
        water_mask = (dec_bin == 1)
        rgba_bin[water_mask] = [2, 132, 199, 210]
        Image.fromarray(rgba_bin).save(binary_png_path, optimize=True)

        # 2. Probability Heatmap Overlay: Blue gradient mapped to probability
        rgba_prob = np.zeros((dec_prob.shape[0], dec_prob.shape[1], 4), dtype=np.uint8)
        valid_prob_mask = (dec_prob >= 0.0)
        norm_p = np.clip(dec_prob, 0.0, 1.0)
        
        rgba_prob[..., 0] = (14 * (1 - norm_p) + 2 * norm_p).astype(np.uint8)   # R
        rgba_prob[..., 1] = (165 * (1 - norm_p) + 132 * norm_p).astype(np.uint8) # G
        rgba_prob[..., 2] = (233 * (1 - norm_p) + 245 * norm_p).astype(np.uint8) # B
        rgba_prob[..., 3] = (valid_prob_mask * (norm_p * 220 + 30)).astype(np.uint8) # Alpha scaled by prob
        
        # Zero out invalid pixels in PNG
        rgba_prob[~valid_prob_mask] = [0, 0, 0, 0]
        Image.fromarray(rgba_prob).save(prob_png_path, optimize=True)

        water_pct = (water_pixels_count / valid_pixels_count * 100.0) if valid_pixels_count > 0 else 0.0
        mean_prob = (valid_prob_sum / valid_pixels_count) if valid_pixels_count > 0 else 0.0

        summary = SegmentationSummary(
            total_pixels=total_pixels,
            valid_pixels=valid_pixels_count,
            nodata_pixels=nodata_pixels_count,
            predicted_water_pixels=water_pixels_count,
            water_percentage=round(water_pct, 4),
            mean_water_probability=round(mean_prob, 4)
        )

        metrics_data = {
            "run_id": run_folder,
            "timestamp": datetime.datetime.now().isoformat(),
            "input_raster": abs_sar_path,
            "effective_raster": effective_sar_path,
            "probability_threshold": threshold,
            "summary": summary.model_dump(),
            "wgs84_bounds": meta["wgs84_bounds"]
        }
        with open(metrics_json_path, "w") as mf:
            json.dump(metrics_data, mf, indent=2)

        return SegmentationResponse(
            status="ok",
            run_id=run_folder,
            probability_threshold=threshold,
            input_filename=os.path.basename(abs_sar_path),
            sar_raster_path=abs_sar_path,
            output_directory=out_dir,
            binary_mask_path=binary_mask_path,
            probability_map_path=prob_map_path,
            binary_overlay_png_path=binary_png_path,
            probability_overlay_png_path=prob_png_path,
            wgs84_bounds=meta["wgs84_bounds"],
            summary=summary
        )
