import os
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin

from geospatial.aoi import AOI
from backend.services.dem_service import DEMService, DEM_ATTRIBUTION

def create_synthetic_dem(path: str, data: np.ndarray, nodata: float = -9999.0):
    """Create synthetic DEM GeoTIFF raster fixture (Explicitly MOCK DATA)."""
    h, w = data.shape
    transform = from_origin(76.15, 10.15, 0.01, 0.01)
    meta = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": rasterio.float32,
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": nodata
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data.astype(np.float32), 1)

def test_dem_service_load_and_clip(tmp_path):
    dem_path = str(tmp_path / "synthetic_dem.tif")
    out_clip = str(tmp_path / "clipped_dem.tif")

    # Create 10x10 elevation array (10m to 100m slope)
    data = np.linspace(10, 100, 100).reshape((10, 10)).astype(np.float32)
    create_synthetic_dem(dem_path, data)

    aoi = AOI(bbox=(76.16, 9.86, 76.24, 10.14))
    service = DEMService()

    clipped_path, meta = service.load_and_clip_dem(aoi, dem_path, out_clip)

    assert os.path.exists(clipped_path)
    assert meta.is_copernicus_dem is True
    assert meta.min_elevation_m >= 10.0
    assert meta.max_elevation_m <= 100.0
    assert meta.attribution == DEM_ATTRIBUTION

def test_dem_service_missing_file():
    service = DEMService()
    aoi = AOI(bbox=(76.15, 9.85, 76.45, 10.15))
    with pytest.raises(FileNotFoundError, match="DEM raster file not found"):
        service.load_and_clip_dem(aoi, "nonexistent/dem.tif")
