"""Layer Visualization & Tile Overlay Service for SARA Dashboard.

Provides memory-bounded raster visualization, color palette management,
nodata transparency enforcement, and cached RGBA PNG overlay generation
for multi-temporal SAR agreement masks, terrain slope layers, and water predictions.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling
from PIL import Image


# Centralized color palettes for dashboard layers
AGREEMENT_PALETTE: Dict[int, Dict[str, Any]] = {
    0: {
        "name": "Dry Land",
        "color": (0, 0, 0, 0),
        "hex": "#00000000",
        "description": "Dry land — no predicted water and no significant backscatter decrease."
    },
    1: {
        "name": "Persistent Water Candidate",
        "color": (29, 78, 216, 215),  # Blue
        "hex": "#1d4ed8",
        "description": "Persistent water candidate — predicted water on terrain with slope <= 15 degrees and stable backscatter."
    },
    2: {
        "name": "Backscatter Change Only",
        "color": (234, 179, 8, 215),  # Yellow
        "hex": "#eab308",
        "description": "Backscatter change only — significant backscatter decrease without a corresponding baseline U-Net water prediction."
    },
    3: {
        "name": "Agreed Inundation Candidate",
        "color": (6, 182, 212, 230),  # Cyan
        "hex": "#06b6d4",
        "description": "Agreed inundation candidate — predicted water, backscatter decrease <= -3 dB, and slope <= 15 degrees."
    },
    4: {
        "name": "Suspected High-Slope Shadow Candidate",
        "color": (234, 88, 12, 215),  # Orange
        "hex": "#ea580c",
        "description": "Suspected high-slope shadow candidate — predicted water on terrain with slope > 15 degrees."
    },
    255: {
        "name": "Nodata",
        "color": (0, 0, 0, 0),
        "hex": "#00000000",
        "description": "Nodata / invalid observation"
    }
}


SLOPE_PALETTE: Dict[str, Dict[str, Any]] = {
    "flat": {
        "name": "Flat (0° - 5°)",
        "color": (34, 197, 94, 60),  # Subtle green
        "hex": "#22c55e",
        "range": (0.0, 5.0),
        "description": "Flat alluvial floodplains and valley floors."
    },
    "moderate": {
        "name": "Moderate (5° - 15°)",
        "color": (234, 179, 8, 120),  # Yellow
        "hex": "#eab308",
        "range": (5.0, 15.0),
        "description": "Moderate rolling terrain and foot-slopes."
    },
    "steep": {
        "name": "Steep (15° - 25°)",
        "color": (249, 115, 22, 170),  # Orange
        "hex": "#f97316",
        "range": (15.0, 25.0),
        "description": "Steep mountain slopes (suspected radar shadow zone)."
    },
    "very_steep": {
        "name": "Very Steep (> 25°)",
        "color": (239, 68, 68, 200),  # Crimson
        "hex": "#ef4444",
        "range": (25.0, 90.0),
        "description": "Very steep relief with severe radar geometric distortion."
    }
}


class LayerVisualizerService:
    """Service for rendering and caching map overlays from geospatial rasters."""

    def __init__(self, cache_dir: str = "data/output"):
        self.cache_dir = cache_dir

    def extract_wgs84_bounds(self, raster_path: str) -> List[List[float]]:
        """Extract [[min_lat, min_lon], [max_lat, max_lon]] in EPSG:4326 for Leaflet."""
        if not os.path.exists(raster_path):
            raise FileNotFoundError(f"Raster file not found: {raster_path}")

        with rasterio.open(raster_path) as src:
            bounds = src.bounds
            crs = src.crs or "EPSG:4326"
            wgs84_tuple = transform_bounds(crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top)
            return [
                [float(wgs84_tuple[1]), float(wgs84_tuple[0])],  # min_lat, min_lon
                [float(wgs84_tuple[3]), float(wgs84_tuple[2])]   # max_lat, max_lon
            ]

    def render_agreement_overlay(
        self,
        agreement_raster_path: str,
        output_png_path: Optional[str] = None,
        downsample_factor: int = 4,
        force_regenerate: bool = False
    ) -> Tuple[str, List[List[float]], Dict[str, Any]]:
        """Render a memory-bounded transparent RGBA PNG overlay for the agreement raster.

        Preserves nodata (255) as transparent, maps classes 0-4 to centralized palette.
        """
        if not os.path.exists(agreement_raster_path):
            raise FileNotFoundError(f"Agreement raster not found at: {agreement_raster_path}")

        if output_png_path is None:
            parent_dir = os.path.dirname(agreement_raster_path)
            output_png_path = os.path.join(parent_dir, "agreement_overlay.png")

        # Return cached PNG if it exists and is newer than source raster
        if not force_regenerate and os.path.exists(output_png_path):
            if os.path.getmtime(output_png_path) >= os.path.getmtime(agreement_raster_path):
                wgs84_bounds = self.extract_wgs84_bounds(agreement_raster_path)
                return output_png_path, wgs84_bounds, {"cached": True, "path": output_png_path}

        os.makedirs(os.path.dirname(os.path.abspath(output_png_path)), exist_ok=True)

        with rasterio.open(agreement_raster_path) as src:
            w = src.width
            h = src.height
            nodata_val = src.nodata if src.nodata is not None else 255
            wgs84_bounds = self.extract_wgs84_bounds(agreement_raster_path)

            scale = max(1, int(downsample_factor))
            out_w = max(16, w // scale)
            out_h = max(16, h // scale)

            data = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest)

        rgba = np.zeros((out_h, out_w, 4), dtype=np.uint8)

        # Apply centralized color palette
        for class_id, style in AGREEMENT_PALETTE.items():
            color = style["color"]
            if color[3] > 0:
                mask = (data == class_id)
                rgba[mask] = color

        # Guarantee nodata is strictly transparent
        rgba[data == nodata_val] = [0, 0, 0, 0]

        img = Image.fromarray(rgba)
        img.save(output_png_path, optimize=True)

        meta = {
            "cached": False,
            "width": out_w,
            "height": out_h,
            "source_width": w,
            "source_height": h,
            "downsample_factor": scale,
            "file_size_bytes": os.path.getsize(output_png_path),
            "output_path": output_png_path
        }
        return output_png_path, wgs84_bounds, meta

    def render_slope_overlay(
        self,
        slope_raster_path: str,
        output_png_path: Optional[str] = None,
        downsample_factor: int = 2,
        force_regenerate: bool = False
    ) -> Tuple[str, List[List[float]], Dict[str, Any]]:
        """Render a memory-bounded transparent RGBA PNG overlay for the slope raster.

        Maps terrain slope ranges to geomorphological color gradients.
        """
        if not os.path.exists(slope_raster_path):
            raise FileNotFoundError(f"Slope raster not found at: {slope_raster_path}")

        if output_png_path is None:
            parent_dir = os.path.dirname(slope_raster_path)
            output_png_path = os.path.join(parent_dir, "slope_overlay.png")

        if not force_regenerate and os.path.exists(output_png_path):
            if os.path.getmtime(output_png_path) >= os.path.getmtime(slope_raster_path):
                wgs84_bounds = self.extract_wgs84_bounds(slope_raster_path)
                return output_png_path, wgs84_bounds, {"cached": True, "path": output_png_path}

        os.makedirs(os.path.dirname(os.path.abspath(output_png_path)), exist_ok=True)

        with rasterio.open(slope_raster_path) as src:
            w = src.width
            h = src.height
            nodata_val = src.nodata if src.nodata is not None else -9999.0
            wgs84_bounds = self.extract_wgs84_bounds(slope_raster_path)

            scale = max(1, int(downsample_factor))
            out_w = max(16, w // scale)
            out_h = max(16, h // scale)

            data = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear)

        rgba = np.zeros((out_h, out_w, 4), dtype=np.uint8)
        valid = (data != nodata_val) & (data >= 0.0) & (~np.isnan(data))

        m_flat = valid & (data < 5.0)
        m_mod = valid & (data >= 5.0) & (data < 15.0)
        m_steep = valid & (data >= 15.0) & (data < 25.0)
        m_vsteep = valid & (data >= 25.0)

        rgba[m_flat] = SLOPE_PALETTE["flat"]["color"]
        rgba[m_mod] = SLOPE_PALETTE["moderate"]["color"]
        rgba[m_steep] = SLOPE_PALETTE["steep"]["color"]
        rgba[m_vsteep] = SLOPE_PALETTE["very_steep"]["color"]

        rgba[~valid] = [0, 0, 0, 0]

        img = Image.fromarray(rgba)
        img.save(output_png_path, optimize=True)

        meta = {
            "cached": False,
            "width": out_w,
            "height": out_h,
            "source_width": w,
            "source_height": h,
            "downsample_factor": scale,
            "file_size_bytes": os.path.getsize(output_png_path),
            "output_path": output_png_path
        }
        return output_png_path, wgs84_bounds, meta

    def get_nepal_summary(self) -> Dict[str, Any]:
        """Generate structured metadata and layer configuration for the Nepal analysis run."""
        run_id = "run_nepal_multitemporal_change_20261003"
        agr_raster = "data/output/run_nepal_multitemporal_change_20261003/nepal_water_and_change_agreement_mask.tif"
        slope_raster = "data/processed/nepal_copernicus_slope_30m.tif"
        eval_run_id = "run_nepal_unet_eval_20261003"

        # Check and ensure overlays are generated
        agr_overlay = os.path.join("data", "output", run_id, "agreement_overlay.png")
        slope_overlay = os.path.join("data", "output", run_id, "slope_overlay.png")

        if os.path.exists(agr_raster) and not os.path.exists(agr_overlay):
            self.render_agreement_overlay(agr_raster, agr_overlay)

        if os.path.exists(slope_raster) and not os.path.exists(slope_overlay):
            self.render_slope_overlay(slope_raster, slope_overlay)

        sar_wgs84 = [
            [27.591277113250623, 83.92583581969234],
            [29.524361773147422, 86.80691259892046]
        ]
        slope_wgs84 = [
            [27.000138888888888, 82.99986111111112],
            [30.000138888888888, 86.99986111111112]
        ]

        metrics_file = "experiments/unet/nepal_multitemporal_change_metrics.json"
        metrics = {}
        if os.path.exists(metrics_file):
            try:
                metrics = json.load(open(metrics_file))
            except Exception:
                pass

        return {
            "status": "ok",
            "run_id": run_id,
            "eval_run_id": eval_run_id,
            "region": "Nepal Himalayan Flood & Slope AOI",
            "center": [28.5578, 85.3664],
            "sar_wgs84_bounds": sar_wgs84,
            "slope_wgs84_bounds": slope_wgs84,
            "layers": {
                "agreement_overlay": f"/dashboard/layers/{run_id}/agreement_overlay.png",
                "water_prediction_overlay": f"/dashboard/layers/{eval_run_id}/water_prediction_overlay.png",
                "water_probability_overlay": f"/dashboard/layers/{eval_run_id}/water_probability_overlay.png",
                "slope_overlay": f"/dashboard/layers/{run_id}/slope_overlay.png"
            },
            "metrics": metrics,
            "palettes": {
                "agreement": AGREEMENT_PALETTE,
                "slope": SLOPE_PALETTE
            },
            "disclaimers": {
                "title": "Exploratory SAR Change-Analysis Candidates",
                "body": (
                    "Exploratory SAR change-analysis candidates, not a verified flood boundary. "
                    "Independent ground-truth validation is not yet available. "
                    "Class 3 represents candidate inundation based on exploratory -3 dB backscatter change and slope <= 15°. "
                    "Class 4 represents suspected high-slope shadow candidates (slope > 15°). "
                    "Model outputs must NOT be used as sole proof of flooding, damage, road closures, or safety status."
                ),
                "attributions": "© Copernicus Sentinel Data (2024–2026) | © Copernicus WorldDEM-30 | © OpenStreetMap contributors"
            }
        }
