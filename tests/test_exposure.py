import pytest
import os
import numpy as np
import geopandas as gpd
import shapely.geometry
import rasterio
from rasterio.transform import from_origin

from geospatial.aoi import AOI
from geospatial.infrastructure import InfrastructureContainer
from geospatial.infrastructure_exposure import InfrastructureExposureAnalyzer

def create_synthetic_change_mask(path: str, mask_data: np.ndarray):
    """Create a tiny synthetic change mask GeoTIFF raster fixture (1=change, 0=no change, 255=nodata)."""
    h, w = mask_data.shape
    transform = from_origin(76.15, 10.15, 0.01, 0.01)  # 0.01 deg spacing
    meta = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": rasterio.uint8,
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": 255
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(mask_data, 1)

def test_exposure_assessment_overlap(tmp_path):
    mask_path = str(tmp_path / "preliminary_change_mask.tif")

    # 10x10 raster grid
    # Right half (x >= 5) has change pixels (value 1)
    mask_grid = np.zeros((10, 10), dtype=np.uint8)
    mask_grid[:, 5:] = 1
    create_synthetic_change_mask(mask_path, mask_grid)

    # Line 1: In left half (76.16 to 76.18, Lat 10.10) -> no overlap
    line_no_overlap = shapely.geometry.LineString([[76.16, 10.10], [76.18, 10.10]])
    # Line 2: Crosses into right half (76.18 to 76.22, Lat 10.10) -> overlap
    line_overlap = shapely.geometry.LineString([[76.18, 10.10], [76.22, 10.10]])

    roads_gdf = gpd.GeoDataFrame([
        {"id": "R_CLEAR", "geometry": line_no_overlap},
        {"id": "R_AFFECTED", "geometry": line_overlap}
    ], crs="EPSG:4326")

    container = InfrastructureContainer(roads=roads_gdf, crs="EPSG:4326")
    aoi = AOI(bbox=(76.15, 9.85, 76.45, 10.15))

    analyzer = InfrastructureExposureAnalyzer()
    out_gdfs, summary = analyzer.analyze_exposure(aoi, container, mask_path)

    roads_out = out_gdfs["potentially_affected_roads"]
    assert len(roads_out) == 2
    assert roads_out.iloc[0]["exposure_label"] == "no_detected_overlap"
    assert roads_out.iloc[1]["exposure_label"] == "potential_change_overlap"
    assert summary.roads_potential_overlap_count == 1

def test_exposure_unassessed_when_no_mask():
    line1 = shapely.geometry.LineString([[76.16, 10.10], [76.18, 10.10]])
    roads_gdf = gpd.GeoDataFrame([{"id": "R1", "geometry": line1}], crs="EPSG:4326")
    container = InfrastructureContainer(roads=roads_gdf, crs="EPSG:4326")
    aoi = AOI(bbox=(76.15, 9.85, 76.45, 10.15))

    analyzer = InfrastructureExposureAnalyzer()
    out_gdfs, summary = analyzer.analyze_exposure(aoi, container, change_mask_path=None)

    roads_out = out_gdfs["potentially_affected_roads"]
    assert roads_out.iloc[0]["exposure_label"] == "not_assessed"
