"""Copernicus DEM Terrain Analysis Module for SARA.

Calculates slope, aspect, and elevation statistics from an AOI-clipped Copernicus DEM elevation raster.
Uses memory-efficient gradient algorithms with low RAM footprint.
"""

import os
from typing import Optional, Any
import numpy as np
from pydantic import BaseModel, Field
import rasterio

from geospatial.aoi import AOI


class TerrainAnalysisResult(BaseModel):
    """Container holding elevation, slope, and aspect summary metrics."""

    dem_path: str
    slope_path: str
    aspect_path: str
    min_elevation_m: float
    max_elevation_m: float
    mean_elevation_m: float
    mean_slope_deg: float
    crs: str
    raster_width: int
    raster_height: int
    processing_metadata: dict[str, Any] = Field(default_factory=dict)


class TerrainAnalyzer:
    """Analyzer for deriving slope, aspect, and surface metrics from elevation rasters."""

    def __init__(self, pixel_size_meters: float = 30.0):
        self.pixel_size_meters = pixel_size_meters

    def analyze_terrain(
        self,
        dem_path: str,
        output_dir: Optional[str] = None
    ) -> TerrainAnalysisResult:
        """Derive slope and aspect rasters from input DEM.

        Args:
            dem_path: Path to input elevation GeoTIFF raster.
            output_dir: Target directory for saving derived slope.tif and aspect.tif.

        Returns:
            TerrainAnalysisResult containing paths and summary statistics.
        """
        if not os.path.exists(dem_path):
            raise FileNotFoundError(f"Input DEM raster not found: '{dem_path}'")

        if output_dir is None:
            output_dir = os.path.dirname(dem_path) or os.path.join("data", "processed")
        os.makedirs(output_dir, exist_ok=True)

        slope_path = os.path.join(output_dir, "slope.tif")
        aspect_path = os.path.join(output_dir, "aspect.tif")

        with rasterio.open(dem_path) as src:
            crs_str = str(src.crs or "EPSG:4326")
            nodata_val = src.nodata
            meta = src.meta.copy()

            elev_array = src.read(1).astype(np.float32)
            height, width = elev_array.shape

            # Handle nodata mask (strictly exclude nodata, NaN, and +/- Inf)
            if nodata_val is not None:
                valid_mask = (elev_array != nodata_val) & np.isfinite(elev_array)
            else:
                valid_mask = np.isfinite(elev_array)

            # Compute elevation stats
            if np.any(valid_mask):
                min_elev = float(np.min(elev_array[valid_mask]))
                max_elev = float(np.max(elev_array[valid_mask]))
                mean_elev = float(np.mean(elev_array[valid_mask]))
            else:
                min_elev = max_elev = mean_elev = 0.0

            # Prevent nodata edge contamination by filling invalid pixels with nearest valid values
            clean_elev = elev_array.copy()
            if not np.all(valid_mask):
                if np.any(valid_mask):
                    import scipy.ndimage
                    indices = scipy.ndimage.distance_transform_edt(
                        ~valid_mask, return_distances=False, return_indices=True
                    )
                    clean_elev = clean_elev[tuple(indices)]
                else:
                    clean_elev[:] = 0.0

            # Calculate gradients dz/dx and dz/dy with exact metric cell spacing
            # Check if CRS is geographic (degrees) vs projected (meters)
            if "4326" in crs_str.upper() or "WGS" in crs_str.upper() or "GEOGCS" in crs_str.upper():
                # Extract pixel resolution from transform if available
                res_x_deg = abs(src.transform.a) if src.transform else (self.pixel_size_meters / 111320.0)
                res_y_deg = abs(src.transform.e) if src.transform else (self.pixel_size_meters / 111320.0)
                center_lat = (src.bounds.bottom + src.bounds.top) / 2.0 if src.bounds else 0.0
                
                # Metric cell dimensions
                dy_meters = res_y_deg * 111320.0
                dx_meters = res_x_deg * 111320.0 * np.cos(np.radians(center_lat))
                # Fallback to nominal pixel size if transform degenerate
                if dy_meters <= 0.0 or dx_meters <= 0.0:
                    dy_meters = dx_meters = self.pixel_size_meters
            else:
                # Projected coordinates in meters
                dy_meters = abs(src.transform.e) if src.transform else self.pixel_size_meters
                dx_meters = abs(src.transform.a) if src.transform else self.pixel_size_meters

            # Calculate gradients (dy along axis 0, dx along axis 1)
            dy, dx = np.gradient(clean_elev, dy_meters, dx_meters)

            # Slope in degrees: atan(sqrt(dx^2 + dy^2)) * 180 / pi
            slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
            slope_deg = np.degrees(slope_rad)
            slope_deg[~valid_mask] = -9999.0

            # Aspect in degrees: 270 - atan2(dy, -dx) * 180 / pi (normalized 0-360 deg)
            aspect_deg = np.degrees(np.arctan2(dy, -dx))
            aspect_deg = (270.0 - aspect_deg) % 360.0
            aspect_deg[~valid_mask] = -9999.0

            mean_slope = float(np.mean(slope_deg[valid_mask])) if np.any(valid_mask) else 0.0

            # Write Slope GeoTIFF
            meta_slope = meta.copy()
            meta_slope.update({"dtype": rasterio.float32, "nodata": -9999.0})
            with rasterio.open(slope_path, "w", **meta_slope) as dst_s:
                dst_s.write(slope_deg.astype(np.float32), 1)

            # Write Aspect GeoTIFF
            with rasterio.open(aspect_path, "w", **meta_slope) as dst_a:
                dst_a.write(aspect_deg.astype(np.float32), 1)

            return TerrainAnalysisResult(
                dem_path=dem_path,
                slope_path=slope_path,
                aspect_path=aspect_path,
                min_elevation_m=round(min_elev, 1),
                max_elevation_m=round(max_elev, 1),
                mean_elevation_m=round(mean_elev, 1),
                mean_slope_deg=round(mean_slope, 1),
                crs=crs_str,
                raster_width=width,
                raster_height=height,
                processing_metadata={
                    "pixel_size_meters_dx": round(dx_meters, 2),
                    "pixel_size_meters_dy": round(dy_meters, 2),
                    "slope_unit": "degrees",
                    "aspect_unit": "degrees"
                }
            )
