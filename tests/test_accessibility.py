import pytest
import geopandas as gpd
import shapely.geometry
from geospatial.road_network import RoadNetworkBuilder
from geospatial.accessibility import AccessibilityAnalyzer

def test_accessibility_baseline_and_scenario():
    # Construct a 2-segment road network: Settlement -> Road 1 -> Junction -> Road 2 -> Hospital
    pt_settlement = shapely.geometry.Point(76.15, 9.85)
    pt_junction = shapely.geometry.Point(76.25, 9.95)
    pt_hospital = shapely.geometry.Point(76.35, 10.05)

    line_r1 = shapely.geometry.LineString([[76.15, 9.85], [76.25, 9.95]])
    line_r2 = shapely.geometry.LineString([[76.25, 9.95], [76.35, 10.05]])

    roads_gdf = gpd.GeoDataFrame([
        {"id": "R1", "name": "Road 1", "geometry": line_r1},
        {"id": "R2", "name": "Road 2", "geometry": line_r2}
    ], crs="EPSG:4326")

    settlements_gdf = gpd.GeoDataFrame([{"id": "S1", "name": "Village A", "geometry": pt_settlement}], crs="EPSG:4326")
    hospitals_gdf = gpd.GeoDataFrame([{"id": "H1", "name": "City Hospital", "geometry": pt_hospital}], crs="EPSG:4326")

    builder = RoadNetworkBuilder()
    graph, summary = builder.build_graph(roads_gdf)

    analyzer = AccessibilityAnalyzer()

    # Mode A: Baseline Accessibility
    res_base, sum_base = analyzer.analyze_accessibility(graph, settlements_gdf, hospitals_gdf)
    assert len(res_base) == 1
    assert bool(res_base.iloc[0]["baseline_reachable"]) is True
    assert res_base.iloc[0]["status_label"] == "CONNECTED"
    assert res_base.iloc[0]["baseline_distance_km"] > 0

    # Mode B: Scenario Accessibility (Road 2 hypothetical closure)
    res_scen, sum_scen = analyzer.analyze_accessibility(graph, settlements_gdf, hospitals_gdf, closed_road_ids={"R2"})
    assert len(res_scen) == 1
    assert bool(res_scen.iloc[0]["baseline_reachable"]) is True
    assert bool(res_scen.iloc[0]["scenario_reachable"]) is False
    assert res_scen.iloc[0]["status_label"] == "HYPOTHETICALLY_ISOLATED"
    assert sum_scen.scenario_isolated_count == 1

def test_unreachable_destination_in_base_network():
    pt_settlement = shapely.geometry.Point(76.15, 9.85)
    pt_hospital = shapely.geometry.Point(76.85, 10.85)

    # Line 1 connected to settlement (76.15, 9.85 to 76.20, 9.90)
    line1 = shapely.geometry.LineString([[76.15, 9.85], [76.20, 9.90]])
    # Line 2 connected to hospital (76.85, 10.85 to 76.90, 10.90) - disconnected component
    line2 = shapely.geometry.LineString([[76.85, 10.85], [76.90, 10.90]])

    roads_gdf = gpd.GeoDataFrame([
        {"id": "R1", "geometry": line1},
        {"id": "R2", "geometry": line2}
    ], crs="EPSG:4326")

    settlements_gdf = gpd.GeoDataFrame([{"id": "S1", "name": "Isolated Village", "geometry": pt_settlement}], crs="EPSG:4326")
    hospitals_gdf = gpd.GeoDataFrame([{"id": "H1", "name": "Far Hospital", "geometry": pt_hospital}], crs="EPSG:4326")

    builder = RoadNetworkBuilder()
    graph, summary = builder.build_graph(roads_gdf)

    analyzer = AccessibilityAnalyzer()
    res, summary = analyzer.analyze_accessibility(graph, settlements_gdf, hospitals_gdf)

    assert bool(res.iloc[0]["baseline_reachable"]) is False
    assert res.iloc[0]["status_label"] == "UNREACHABLE_IN_BASE_NETWORK"
