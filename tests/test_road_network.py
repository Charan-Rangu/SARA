import pytest
import geopandas as gpd
import shapely.geometry
import networkx as nx
from geospatial.road_network import RoadNetworkBuilder

def test_build_road_graph_single_line():
    line1 = shapely.geometry.LineString([[76.15, 9.85], [76.45, 10.15]])
    roads_gdf = gpd.GeoDataFrame([{"id": "R1", "name": "Main St", "geometry": line1}], crs="EPSG:4326")

    builder = RoadNetworkBuilder(metric_crs="EPSG:3857")
    graph, summary = builder.build_graph(roads_gdf)

    assert summary.total_nodes == 2
    assert summary.total_edges == 1
    assert summary.is_fully_connected is True
    assert summary.connected_components_count == 1

def test_build_road_graph_disconnected_lines():
    line1 = shapely.geometry.LineString([[76.15, 9.85], [76.20, 9.90]])
    line2 = shapely.geometry.LineString([[76.40, 10.10], [76.45, 10.15]])
    roads_gdf = gpd.GeoDataFrame([
        {"id": "R1", "geometry": line1},
        {"id": "R2", "geometry": line2}
    ], crs="EPSG:4326")

    builder = RoadNetworkBuilder(metric_crs="EPSG:3857")
    graph, summary = builder.build_graph(roads_gdf)

    assert summary.total_nodes == 4
    assert summary.total_edges == 2
    assert summary.connected_components_count == 2
    assert summary.is_fully_connected is False

def test_build_road_graph_multilinestring():
    line1 = shapely.geometry.LineString([[76.15, 9.85], [76.20, 9.90]])
    line2 = shapely.geometry.LineString([[76.20, 9.90], [76.25, 9.95]])
    multi = shapely.geometry.MultiLineString([line1, line2])
    roads_gdf = gpd.GeoDataFrame([{"id": "MULTI_1", "geometry": multi}], crs="EPSG:4326")

    builder = RoadNetworkBuilder()
    graph, summary = builder.build_graph(roads_gdf)

    assert summary.total_edges == 2
    assert summary.connected_components_count == 1
