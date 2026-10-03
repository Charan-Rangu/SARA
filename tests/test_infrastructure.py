import pytest
import geopandas as gpd
import shapely.geometry
from geospatial.aoi import AOI
from geospatial.infrastructure import validate_vector_layer

def test_validate_vector_layer_clean():
    line1 = shapely.geometry.LineString([[76.15, 9.85], [76.45, 10.15]])
    gdf = gpd.GeoDataFrame([{"id": "R1", "highway": "primary", "geometry": line1}], crs="EPSG:4326")

    aoi = AOI(bbox=(76.10, 9.80, 76.50, 10.20))
    cleaned, summary = validate_vector_layer(gdf, "roads", ["LineString"], "EPSG:4326", aoi=aoi)

    assert len(cleaned) == 1
    assert summary.valid_features_count == 1
    assert summary.excluded_invalid_count == 0

def test_validate_vector_reprojects_crs():
    # Geometry in EPSG:3857
    pt_3857 = shapely.geometry.Point(8477000, 1102000)
    gdf = gpd.GeoDataFrame([{"id": "H1", "geometry": pt_3857}], crs="EPSG:3857")

    cleaned, summary = validate_vector_layer(gdf, "facilities", ["Point"], target_crs="EPSG:4326")

    assert cleaned.crs.to_string() == "EPSG:4326"
    assert summary.target_crs == "EPSG:4326"
    assert any("Reprojected layer" in note for note in summary.notes)

def test_validate_vector_filters_outside_aoi():
    line_inside = shapely.geometry.LineString([[76.15, 9.85], [76.45, 10.15]])
    line_outside = shapely.geometry.LineString([[0.0, 0.0], [1.0, 1.0]])

    gdf = gpd.GeoDataFrame([
        {"id": "IN", "geometry": line_inside},
        {"id": "OUT", "geometry": line_outside}
    ], crs="EPSG:4326")

    aoi = AOI(bbox=(76.10, 9.80, 76.50, 10.20))
    cleaned, summary = validate_vector_layer(gdf, "roads", ["LineString"], "EPSG:4326", aoi=aoi)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["id"] == "IN"
    assert summary.excluded_outside_aoi_count == 1

def test_validate_vector_deduplication():
    line1 = shapely.geometry.LineString([[76.15, 9.85], [76.45, 10.15]])
    gdf = gpd.GeoDataFrame([
        {"id": "R1", "geometry": line1},
        {"id": "R1", "geometry": line1}
    ], crs="EPSG:4326")

    cleaned, summary = validate_vector_layer(gdf, "roads", ["LineString"], id_column="id")

    assert len(cleaned) == 1
    assert summary.duplicate_features_removed == 1
