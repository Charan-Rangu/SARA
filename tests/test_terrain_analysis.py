import os
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin

from geospatial.terrain_analysis import TerrainAnalyzer
from tests.test_dem_service import create_synthetic_dem

def test_terrain_analysis_slope_aspect(tmp_path):
    dem_path = str(tmp_path / "dem_slope_test.tif")

    # 10x10 synthetic inclined plane sloping downwards to the east (right)
    # Row by row, values decrease from left (100m) to right (10m)
    x = np.linspace(100, 10, 10)
    data = np.tile(x, (10, 1))

    create_synthetic_dem(dem_path, data)

    analyzer = TerrainAnalyzer(pixel_size_meters=30.0)
    res = analyzer.analyze_terrain(dem_path, output_dir=str(tmp_path))

    assert os.path.exists(res.slope_path)
    assert os.path.exists(res.aspect_path)
    assert res.min_elevation_m == 10.0
    assert res.max_elevation_m == 100.0
    assert res.mean_slope_deg > 0.0

    # Read slope raster
    with rasterio.open(res.slope_path) as src:
        slope_arr = src.read(1)
        assert slope_arr.shape == (10, 10)
        assert np.all(slope_arr > 0)
