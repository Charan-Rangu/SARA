"""Copernicus Digital Elevation Model (DEM) Service for SARA.

Manages, validates, and clips Copernicus DEM (WorldDEM-30) elevation rasters for a user-specified AOI.
Supports local GeoTIFF elevation files with spatial bounds validation, CRS reprojection, and attribution tracking.

Attribution Statement:
"Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018
provided under COPERNICUS by the European Union and ESA; all rights reserved."
"""

import os
from typing import Optional, Any
import numpy as np
from pydantic import BaseModel, Field
import rasterio
from rasterio.mask import mask
import shapely.geometry

from geospatial.aoi import AOI

DEM_ATTRIBUTION = (
    "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and "
    "© Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved."
)


class DEMMetadata(BaseModel):
    """Metadata container documenting DEM raster provenance and spatial properties."""

    source_name: str = "Copernicus WorldDEM-30"
    is_copernicus_dem: bool = True
    crs: str
    width: int
    height: int
    pixel_size_x: float
    pixel_size_y: float
    min_elevation_m: float
    max_elevation_m: float
    nodata_value: Optional[float] = None
    attribution: str = DEM_ATTRIBUTION
    clipped_to_aoi: bool = False


class DEMService:
    """Service for loading, validating, and clipping Copernicus DEM elevation rasters."""

    def __init__(self, default_dem_dir: str = os.path.join("data", "raw")):
        self.default_dem_dir = default_dem_dir

    def load_and_clip_dem(
        self,
        aoi: AOI,
        dem_file_path: str,
        output_clipped_path: Optional[str] = None
    ) -> tuple[str, DEMMetadata]:
        """Load a DEM GeoTIFF, validate spatial metadata, and clip to AOI boundary.

        Args:
            aoi: Target Area of Interest.
            dem_file_path: Path to local DEM GeoTIFF raster file.
            output_clipped_path: Optional path for clipped output DEM GeoTIFF.

        Returns:
            Tuple of (path to clipped DEM GeoTIFF, DEMMetadata).
        """
        if not os.path.exists(dem_file_path):
            raise FileNotFoundError(f"DEM raster file not found at path: '{dem_file_path}'")

        if output_clipped_path is None:
            out_dir = os.path.join("data", "processed")
            os.makedirs(out_dir, exist_ok=True)
            output_clipped_path = os.path.join(out_dir, "clipped_dem.tif")

        with rasterio.open(dem_file_path) as src:
            crs_str = str(src.crs or "EPSG:4326")
            nodata_val = src.nodata

            # Reproject AOI polygon to DEM CRS if needed
            aoi_geom = aoi.to_shapely()
            if crs_str.upper() != "EPSG:4326":
                import geopandas as gpd
                aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_geom], crs="EPSG:4326").to_crs(crs_str)
                aoi_geom = aoi_gdf.geometry.iloc[0]

            # Clip raster using AOI geometry
            try:
                clipped_image, clipped_transform = mask(src, [aoi_geom], crop=True)
            except ValueError as ve:
                raise ValueError(f"AOI boundary does not overlap DEM raster extent: {ve}")

            clipped_array = clipped_image[0]
            out_height, out_width = clipped_array.shape

            # Handle Nodata values to compute valid elevation min/max
            if nodata_val is not None:
                valid_mask = (clipped_array != nodata_val) & (~np.isnan(clipped_array))
            else:
                valid_mask = ~np.isnan(clipped_array)

            if np.any(valid_mask):
                min_elev = float(np.min(clipped_array[valid_mask]))
                max_elev = float(np.max(clipped_array[valid_mask]))
            else:
                min_elev = 0.0
                max_elev = 0.0

            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_height,
                "width": out_width,
                "transform": clipped_transform,
                "nodata": nodata_val or -9999.0
            })

            # Save clipped DEM GeoTIFF
            os.makedirs(os.path.dirname(output_clipped_path), exist_ok=True)
            with rasterio.open(output_clipped_path, "w", **out_meta) as dst:
                dst.write(clipped_image[0], 1)
                dst.update_tags(
                    SOURCE="Copernicus WorldDEM-30",
                    ATTRIBUTION=DEM_ATTRIBUTION
                )

            px_x = abs(clipped_transform[0])
            px_y = abs(clipped_transform[4])

            metadata = DEMMetadata(
                source_name="Copernicus WorldDEM-30",
                is_copernicus_dem=True,
                crs=crs_str,
                width=out_width,
                height=out_height,
                pixel_size_x=px_x,
                pixel_size_y=px_y,
                min_elevation_m=min_elev,
                max_elevation_m=max_elev,
                nodata_value=nodata_val,
                attribution=DEM_ATTRIBUTION,
                clipped_to_aoi=True
            )

            return output_clipped_path, metadata
