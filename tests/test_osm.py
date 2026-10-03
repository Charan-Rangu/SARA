import pytest
import os
from geospatial.aoi import AOI
from backend.services.osm_infrastructure import OSMInfrastructureService

MOCK_OVERPASS_JSON = {
    "elements": [
        {"type": "node", "id": 101, "lon": 76.20, "lat": 9.90},
        {"type": "node", "id": 102, "lon": 76.30, "lat": 10.00},
        {"type": "node", "id": 103, "lon": 76.25, "lat": 9.95, "tags": {"amenity": "hospital", "name": "General Hospital"}},
        {"type": "node", "id": 104, "lon": 76.35, "lat": 10.05, "tags": {"place": "village", "name": "Kochi Village"}},
        {
            "type": "way",
            "id": 201,
            "nodes": [101, 102],
            "tags": {"highway": "primary", "name": "Main Highway"}
        }
    ]
}

def test_parse_overpass_json_mock():
    service = OSMInfrastructureService()
    aoi = AOI(bbox=(76.15, 9.85, 76.45, 10.15))
    
    container = service._parse_overpass_json(MOCK_OVERPASS_JSON, aoi)

    assert len(container.roads) == 1
    assert container.roads.iloc[0]["name"] == "Main Highway"
    assert len(container.facilities) == 1
    assert container.facilities.iloc[0]["name"] == "General Hospital"
    assert len(container.settlements) == 1
    assert container.settlements.iloc[0]["name"] == "Kochi Village"

def test_load_local_snapshot_empty(tmp_path):
    service = OSMInfrastructureService()
    aoi = AOI(bbox=(76.15, 9.85, 76.45, 10.15))

    container, meta = service.load_local_snapshot(aoi=aoi, snapshot_date="2024-08-01")

    assert meta.source_type == "LOCAL_PRE_EVENT_SNAPSHOT"
    assert meta.is_pre_event_snapshot is True
    assert len(container.roads) == 0
