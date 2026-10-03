import os
import tempfile
import pytest
import numpy as np
import geopandas as gpd
import shapely.geometry
import rasterio
from rasterio.transform import from_origin

from geospatial.aoi import AOI
from geospatial.infrastructure import validate_vector_layer, InfrastructureContainer
from geospatial.infrastructure_exposure import InfrastructureExposureAnalyzer
from geospatial.road_network import RoadNetworkBuilder
from geospatial.accessibility import AccessibilityAnalyzer

def test_end_to_end_phase4_pipeline(tmp_path):
    # 1. Create synthetic change mask raster (10x10 grid, right half has change value 1)
    mask_path = str(tmp_path / "preliminary_change_mask.tif")
    mask_data = np.zeros((10, 10), dtype=np.uint8)
    mask_data[:, 5:] = 1  # Right half has change

    transform = from_origin(76.15, 10.15, 0.01, 0.01)
    meta = {
        "driver": "GTiff",
        "height": 10,
        "width": 10,
        "count": 1,
        "dtype": rasterio.uint8,
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": 255
    }
    with rasterio.open(mask_path, "w", **meta) as dst:
        dst.write(mask_data, 1)

    # 2. Create synthetic vector infrastructure
    pt_settlement = shapely.geometry.Point(76.18, 10.10)
    pt_hospital = shapely.geometry.Point(76.22, 10.10)

    # Road 1 (left side, no change): 76.16, 10.10 -> 76.19, 10.10
    line_r1 = shapely.geometry.LineString([[76.16, 10.10], [76.19, 10.10]])
    # Road 2 (crosses into change mask): 76.19, 10.10 -> 76.22, 10.10
    line_r2 = shapely.geometry.LineString([[76.19, 10.10], [76.22, 10.10]])

    roads_gdf = gpd.GeoDataFrame([
        {"id": "R1", "name": "Road 1", "geometry": line_r1},
        {"id": "R2", "name": "Road 2", "geometry": line_r2}
    ], crs="EPSG:4326")

    settlements_gdf = gpd.GeoDataFrame([{"id": "S1", "name": "Village A", "geometry": pt_settlement}], crs="EPSG:4326")
    hospitals_gdf = gpd.GeoDataFrame([{"id": "H1", "name": "City Hospital", "geometry": pt_hospital}], crs="EPSG:4326")

    aoi = AOI(bbox=(76.15, 9.85, 76.45, 10.15))

    # 3. Vector Validation
    val_roads, _ = validate_vector_layer(roads_gdf, "roads", ["LineString"], aoi=aoi)
    container = InfrastructureContainer(roads=val_roads, settlements=settlements_gdf, facilities=hospitals_gdf)

    # 4. Exposure Analysis
    exposure_analyzer = InfrastructureExposureAnalyzer()
    out_gdfs, summary = exposure_analyzer.analyze_exposure(aoi, container, mask_path)

    affected_roads = out_gdfs["potentially_affected_roads"]
    assert len(affected_roads) == 2
    r2_exposure = affected_roads[affected_roads["id"] == "R2"].iloc[0]["exposure_label"]
    assert r2_exposure == "potential_change_overlap"

    # 5. Road Network Graph Construction
    graph_builder = RoadNetworkBuilder()
    graph, g_summary = graph_builder.build_graph(val_roads)
    assert g_summary.total_edges >= 2

    # 6. Accessibility Analysis (Baseline & Scenario)
    access_analyzer = AccessibilityAnalyzer()
    # Mode A: Baseline Reachability
    res_base, _ = access_analyzer.analyze_accessibility(graph, settlements_gdf, hospitals_gdf)
    assert bool(res_base.iloc[0]["baseline_reachable"]) is True

    # Mode B: Scenario Reachability (hypothetical closure of R2)
    res_scen, sum_scen = access_analyzer.analyze_accessibility(graph, settlements_gdf, hospitals_gdf, closed_road_ids={"R2"})
    assert bool(res_scen.iloc[0]["scenario_reachable"]) is False
    assert res_scen.iloc[0]["status_label"] == "HYPOTHETICALLY_ISOLATED"
    assert sum_scen.scenario_isolated_count == 1
