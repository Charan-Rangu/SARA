import pytest
import os
import json
from geospatial.aoi import AOI
from backend.services.sentinel1_catalog import (
    Sentinel1CatalogService,
    normalize_stac_item,
    parse_datetime
)

# Synthetic Mock STAC Response for Offline Unit Testing (Explicitly labeled as MOCK TEST DATA)
MOCK_STAC_RESPONSE = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "id": "S1A_IW_GRDH_1SDV_20260810T003412_MOCK_PRE",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[76.0, 9.5], [76.5, 9.5], [76.5, 10.2], [76.0, 10.2], [76.0, 9.5]]]
            },
            "properties": {
                "title": "S1A_IW_GRDH_1SDV_20260810T003412_MOCK_PRE",
                "datetime": "2026-08-10T00:34:12Z",
                "platform": "sentinel-1a",
                "sar:product_type": "GRD",
                "sar:instrument_mode": "IW",
                "sar:polarizations": ["VV", "VH"],
                "sat:relative_orbit": 114,
                "sat:orbit_state": "ASCENDING"
            },
            "assets": {
                "download": {
                    "href": "https://mock.dataspace.copernicus.eu/s1a_pre.zip",
                    "file:size": 1850000000
                }
            }
        },
        {
            "type": "Feature",
            "id": "S1A_IW_GRDH_1SDV_20260817T003412_MOCK_POST",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[76.0, 9.5], [76.5, 9.5], [76.5, 10.2], [76.0, 10.2], [76.0, 9.5]]]
            },
            "properties": {
                "title": "S1A_IW_GRDH_1SDV_20260817T003412_MOCK_POST",
                "datetime": "2026-08-17T00:34:12Z",
                "platform": "sentinel-1a",
                "sar:product_type": "GRD",
                "sar:instrument_mode": "IW",
                "sar:polarizations": ["VV", "VH"],
                "sat:relative_orbit": 114,
                "sat:orbit_state": "ASCENDING"
            },
            "assets": {
                "download": {
                    "href": "https://mock.dataspace.copernicus.eu/s1a_post.zip",
                    "file:size": 1920000000
                }
            }
        },
        # Duplicate feature entry to test deduplication
        {
            "type": "Feature",
            "id": "S1A_IW_GRDH_1SDV_20260810T003412_MOCK_PRE",
            "geometry": None,
            "properties": {"datetime": "2026-08-10T00:34:12Z"}
        }
    ]
}

def test_parse_datetime():
    dt = parse_datetime("2026-08-15")
    assert dt.year == 2026 and dt.month == 8 and dt.day == 15
    dt_iso = parse_datetime("2026-08-15T12:30:00Z")
    assert dt_iso.hour == 12 and dt_iso.minute == 30

def test_normalize_stac_item():
    item = MOCK_STAC_RESPONSE["features"][0]
    scene = normalize_stac_item(item)
    assert scene.id == "S1A_IW_GRDH_1SDV_20260810T003412_MOCK_PRE"
    assert scene.platform == "SENTINEL-1A"
    assert scene.product_type == "GRD"
    assert scene.sensor_mode == "IW"
    assert scene.polarizations == ["VV", "VH"]
    assert scene.relative_orbit_number == 114
    assert scene.orbit_direction == "ASCENDING"
    assert scene.size_bytes == 1850000000

def test_stac_normalization_missing_fields():
    sparse_item = {
        "id": "SPARSE_SCENE_001",
        "properties": {}
    }
    scene = normalize_stac_item(sparse_item)
    assert scene.id == "SPARSE_SCENE_001"
    assert scene.platform == "SENTINEL-1"
    assert scene.relative_orbit_number is None
    assert scene.orbit_direction is None
    assert scene.size_bytes is None

def test_offline_catalog_search_mock():
    aoi = AOI(bbox=(76.15, 9.85, 76.45, 10.15))
    service = Sentinel1CatalogService(cache_file="data/cache/test_cache.json")
    
    result = service.search_scenes(
        aoi=aoi,
        event_date_str="2026-08-15",
        mock_response=MOCK_STAC_RESPONSE
    )

    assert result.event_date == "2026-08-15"
    assert len(result.pre_event_scenes) == 1
    assert len(result.post_event_scenes) == 1
    assert result.total_scenes_found == 2
    # Verify deduplication
    assert result.pre_event_scenes[0].id == "S1A_IW_GRDH_1SDV_20260810T003412_MOCK_PRE"
    assert result.post_event_scenes[0].id == "S1A_IW_GRDH_1SDV_20260817T003412_MOCK_POST"

def test_empty_catalog_response():
    aoi = AOI(bbox=(76.15, 9.85, 76.45, 10.15))
    service = Sentinel1CatalogService(cache_file="data/cache/test_cache.json")
    
    result = service.search_scenes(
        aoi=aoi,
        event_date_str="2026-08-15",
        mock_response={"type": "FeatureCollection", "features": []}
    )

    assert result.total_scenes_found == 0
    assert len(result.pre_event_scenes) == 0
    assert len(result.post_event_scenes) == 0
