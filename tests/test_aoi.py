import pytest
import shapely.geometry
from geospatial.aoi import AOI, KERALA_FLOOD_EXAMPLE_AOI, NEPAL_BHOTE_KOSHI_AOI, MAX_AOI_AREA_SQ_DEG

def test_valid_bbox_aoi():
    aoi = AOI(bbox=(76.15, 9.85, 76.45, 10.15), name="Test BBox")
    assert aoi.to_bbox() == (76.15, 9.85, 76.45, 10.15)
    assert aoi.area_sq_deg() == pytest.approx(0.09, abs=1e-3)
    assert aoi.to_geojson()["type"] == "Polygon"

def test_invalid_bbox_coords():
    with pytest.raises(ValueError, match="Longitude coordinates out of WGS84 bounds"):
        AOI(bbox=(190.0, 9.85, 76.45, 10.15))

    with pytest.raises(ValueError, match="Latitude coordinates out of WGS84 bounds"):
        AOI(bbox=(76.15, -95.0, 76.45, 10.15))

    with pytest.raises(ValueError, match="Invalid bounding box extent"):
        AOI(bbox=(76.45, 9.85, 76.15, 10.15))

def test_area_exceeds_threshold():
    # Construct massive 6x6 degree bounding box (~36 sq deg > 25.0 sq deg limit)
    with pytest.raises(ValueError, match="exceeds maximum allowed threshold"):
        AOI(bbox=(70.0, 10.0, 76.5, 16.5))

def test_valid_geojson_aoi():
    poly_dict = {
        "type": "Polygon",
        "coordinates": [
            [[76.15, 9.85], [76.45, 9.85], [76.45, 10.15], [76.15, 10.15], [76.15, 9.85]]
        ]
    }
    aoi = AOI(geojson=poly_dict, name="Test GeoJSON")
    assert aoi.to_bbox() == (76.15, 9.85, 76.45, 10.15)

def test_invalid_geojson_structure():
    with pytest.raises(ValueError, match="Invalid GeoJSON geometry structure"):
        AOI(geojson={"type": "Polygon", "coordinates": "invalid"})

def test_aoi_intersection_pct():
    aoi = AOI(bbox=(0.0, 0.0, 2.0, 2.0))
    # Target geometry covers bottom-left quadrant (0.0, 0.0 to 1.0, 1.0) -> 25% overlap
    target_box = shapely.geometry.box(0.0, 0.0, 1.0, 1.0)
    pct = aoi.intersection_percentage(target_box)
    assert pct == 25.0

def test_example_aoi():
    assert KERALA_FLOOD_EXAMPLE_AOI.to_bbox() == (76.15, 9.85, 76.45, 10.15)

def test_nepal_bhote_koshi_aoi():
    assert NEPAL_BHOTE_KOSHI_AOI.to_bbox() == (85.25, 28.10, 85.42, 28.30)
    assert "Nepal" in NEPAL_BHOTE_KOSHI_AOI.name
    assert NEPAL_BHOTE_KOSHI_AOI.area_sq_deg() == pytest.approx(0.034, abs=1e-3)

