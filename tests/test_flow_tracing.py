import os
import pytest
import numpy as np
from geospatial.flow_tracing import TerrainFlowTracer
from tests.test_dem_service import create_synthetic_dem

def test_flow_tracing_d8_descent(tmp_path):
    dem_path = str(tmp_path / "flow_dem.tif")

    # 10x10 synthetic DEM sloping downwards from top-left (100m) to bottom-right (10m)
    r = np.linspace(100, 10, 10)
    data = np.add.outer(r, r)  # Highest at (0,0), lowest at (9,9)

    create_synthetic_dem(dem_path, data)

    tracer = TerrainFlowTracer(max_steps=50)
    # Start near top-left (76.15, 10.15)
    res = tracer.trace_flow_paths(dem_path, start_points_wgs84=[(76.15, 10.15)])

    assert res.status == "SUCCESS"
    assert res.traced_paths_count == 1
    assert len(res.flow_path_records) == 1
    rec = res.flow_path_records[0]
    assert rec.status == "FLOW_PATH_TRACED"
    assert rec.path_length_steps > 1
    assert rec.elevation_drop_m > 0

    # Verify GeoJSON output format
    fc = res.flow_paths_geojson
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 1
    feat = fc["features"][0]
    assert feat["properties"]["layer"] == "MODELED_TERRAIN_FLOW_PATH"

def test_flow_tracing_out_of_bounds(tmp_path):
    dem_path = str(tmp_path / "flow_dem_bounds.tif")
    create_synthetic_dem(dem_path, np.ones((10, 10), dtype=np.float32))

    tracer = TerrainFlowTracer()
    # Point far outside raster (0.0, 0.0)
    res = tracer.trace_flow_paths(dem_path, start_points_wgs84=[(0.0, 0.0)])

    assert res.traced_paths_count == 0
    assert res.flow_path_records[0].status == "OUT_OF_BOUNDS"
