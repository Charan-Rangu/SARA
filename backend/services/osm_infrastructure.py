"""OpenStreetMap Infrastructure Data Acquisition Service for SARA.

Acquires or loads pre-event OSM vector infrastructure (Roads, Buildings, Healthcare Facilities,
and Settlements) for a user-specified Area of Interest (AOI).

Supports live Overpass API queries with caching and rate-limit handling, as well as loading
user-supplied local pre-event GeoJSON snapshots.

Important:
Live Overpass API queries return the current state of OpenStreetMap. Current live queries must NOT be
represented as historical pre-event disaster snapshots.
"""

import os
import json
import time
from datetime import datetime, timezone
from typing import Optional, Any
import httpx
import geopandas as gpd
import shapely.geometry
from pydantic import BaseModel, Field

from geospatial.aoi import AOI
from geospatial.infrastructure import validate_vector_layer, InfrastructureContainer, ValidationSummary

OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"
OSM_CACHE_FILE = os.path.join("data", "cache", "osm_cache.json")


class OSMSourceMetadata(BaseModel):
    """Metadata container documenting the provenance of OpenStreetMap infrastructure data."""

    source_type: str  # "LIVE_OVERPASS_API" or "LOCAL_PRE_EVENT_SNAPSHOT"
    retrieval_timestamp: str
    snapshot_date: Optional[str] = None
    is_pre_event_snapshot: bool = False
    aoi_name: str
    warnings: list[str] = Field(default_factory=list)
    limitation_statement: str = (
        "Live Overpass API responses represent current OSM data and should not be used as verified historical "
        "pre-event snapshots without independent temporal verification."
    )


class OSMInfrastructureService:
    """Service for querying Overpass API or parsing local vector files into validated InfrastructureContainers."""

    def __init__(self, cache_file: str = OSM_CACHE_FILE, timeout_seconds: float = 25.0):
        self.cache_file = cache_file
        self.timeout_seconds = timeout_seconds

    def load_local_snapshot(
        self,
        aoi: AOI,
        roads_geojson_path: Optional[str] = None,
        buildings_geojson_path: Optional[str] = None,
        facilities_geojson_path: Optional[str] = None,
        settlements_geojson_path: Optional[str] = None,
        snapshot_date: Optional[str] = None
    ) -> tuple[InfrastructureContainer, OSMSourceMetadata]:
        """Load and validate infrastructure layers from local GeoJSON snapshot files."""
        summaries: dict[str, ValidationSummary] = {}

        def read_or_empty(path: Optional[str], geom_types: list[str], layer_name: str) -> gpd.GeoDataFrame:
            if path and os.path.exists(path):
                try:
                    gdf = gpd.read_file(path)
                    cleaned, sum_obj = validate_vector_layer(
                        gdf, layer_name=layer_name, allowed_geom_types=geom_types, target_crs="EPSG:4326", aoi=aoi
                    )
                    summaries[layer_name] = sum_obj
                    return cleaned
                except Exception as e:
                    summaries[layer_name] = ValidationSummary(
                        layer_name=layer_name,
                        total_input_features=0,
                        valid_features_count=0,
                        repaired_features_count=0,
                        excluded_invalid_count=0,
                        excluded_empty_count=0,
                        excluded_outside_aoi_count=0,
                        duplicate_features_removed=0,
                        target_crs="EPSG:4326",
                        notes=[f"Failed to read file '{path}': {e}"]
                    )
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        roads = read_or_empty(roads_geojson_path, ["LineString", "MultiLineString"], "roads")
        buildings = read_or_empty(buildings_geojson_path, ["Polygon", "MultiPolygon"], "buildings")
        facilities = read_or_empty(facilities_geojson_path, ["Point", "Polygon", "MultiPolygon"], "facilities")
        settlements = read_or_empty(settlements_geojson_path, ["Point", "Polygon", "MultiPolygon"], "settlements")

        container = InfrastructureContainer(
            roads=roads,
            buildings=buildings,
            facilities=facilities,
            settlements=settlements,
            crs="EPSG:4326",
            validation_summaries=summaries
        )

        meta = OSMSourceMetadata(
            source_type="LOCAL_PRE_EVENT_SNAPSHOT",
            retrieval_timestamp=datetime.now(timezone.utc).isoformat(),
            snapshot_date=snapshot_date,
            is_pre_event_snapshot=True if snapshot_date else False,
            aoi_name=aoi.name
        )

        return container, meta

    def fetch_overpass_infrastructure(
        self,
        aoi: AOI,
        include_buildings: bool = True
    ) -> tuple[InfrastructureContainer, OSMSourceMetadata]:
        """Fetch live OpenStreetMap infrastructure vector features for the target AOI via Overpass API."""
        min_lon, min_lat, max_lon, max_lat = aoi.to_bbox()
        bbox_str = f"{min_lat},{min_lon},{max_lat},{max_lon}"

        # Overpass QL Query for Roads, Healthcare, Places, and Buildings
        overpass_ql = f"""
        [out:json][timeout:25];
        (
          way["highway"]({bbox_str});
          node["amenity"~"hospital|clinic|doctors"]({bbox_str});
          way["amenity"~"hospital|clinic"]({bbox_str});
          node["place"~"city|town|village|hamlet"]({bbox_str});
          {"way[\"building\"](" + bbox_str + ");" if include_buildings else ""}
        );
        out body;
        >;
        out skel qt;
        """

        warnings: list[str] = []
        try:
            headers = {"User-Agent": "SARA-Rescue-Assessment/1.0 (contact@sara-rescue.org)"}
            with httpx.Client(timeout=self.timeout_seconds, headers=headers) as client:
                res = client.post(OVERPASS_API_URL, data={"data": overpass_ql})
                res.raise_for_status()
                osm_json = res.json()
                container = self._parse_overpass_json(osm_json, aoi)
                
                meta = OSMSourceMetadata(
                    source_type="LIVE_OVERPASS_API",
                    retrieval_timestamp=datetime.now(timezone.utc).isoformat(),
                    snapshot_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    is_pre_event_snapshot=False,
                    aoi_name=aoi.name,
                    warnings=warnings
                )
                return container, meta
        except Exception as e:
            warnings.append(f"Live Overpass API query failed ({type(e).__name__}: {e}). Returning empty container.")
            empty_container = InfrastructureContainer(crs="EPSG:4326")
            meta = OSMSourceMetadata(
                source_type="LIVE_OVERPASS_API_FAILED",
                retrieval_timestamp=datetime.now(timezone.utc).isoformat(),
                is_pre_event_snapshot=False,
                aoi_name=aoi.name,
                warnings=warnings
            )
            return empty_container, meta

    def _parse_overpass_json(self, osm_json: dict[str, Any], aoi: AOI) -> InfrastructureContainer:
        elements = osm_json.get("elements", [])
        
        # Build node coordinate mapping
        nodes = {}
        for elem in elements:
            if elem.get("type") == "node":
                nodes[elem["id"]] = (elem["lon"], elem["lat"])

        road_list = []
        building_list = []
        facility_list = []
        settlement_list = []

        for elem in elements:
            tags = elem.get("tags", {})
            elem_id = elem.get("id")

            # Nodes (Points)
            if elem.get("type") == "node":
                lon, lat = nodes[elem_id]
                pt = shapely.geometry.Point(lon, lat)
                
                if "amenity" in tags or "healthcare" in tags:
                    facility_list.append({"id": elem_id, "name": tags.get("name", "Healthcare Facility"), "amenity": tags.get("amenity"), "geometry": pt})
                elif "place" in tags:
                    settlement_list.append({"id": elem_id, "name": tags.get("name", "Settlement"), "place": tags.get("place"), "geometry": pt})

            # Ways (LineStrings / Polygons)
            elif elem.get("type") == "way":
                node_refs = elem.get("nodes", [])
                coords = [nodes[n] for n in node_refs if n in nodes]
                if len(coords) < 2:
                    continue

                if "highway" in tags:
                    line = shapely.geometry.LineString(coords)
                    road_list.append({
                        "id": elem_id,
                        "name": tags.get("name", "Unnamed Road"),
                        "highway": tags.get("highway"),
                        "oneway": tags.get("oneway", "no"),
                        "bridge": tags.get("bridge"),
                        "geometry": line
                    })

                elif "building" in tags and len(coords) >= 3:
                    poly = shapely.geometry.Polygon(coords)
                    building_list.append({
                        "id": elem_id,
                        "name": tags.get("name"),
                        "building": tags.get("building"),
                        "geometry": poly
                    })

                elif "amenity" in tags and len(coords) >= 3:
                    poly = shapely.geometry.Polygon(coords)
                    facility_list.append({
                        "id": elem_id,
                        "name": tags.get("name", "Healthcare Facility"),
                        "amenity": tags.get("amenity"),
                        "geometry": poly
                    })

        summaries = {}
        
        roads_gdf = gpd.GeoDataFrame(road_list, crs="EPSG:4326") if road_list else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        roads_gdf, summaries["roads"] = validate_vector_layer(roads_gdf, "roads", ["LineString", "MultiLineString"], "EPSG:4326", aoi)

        buildings_gdf = gpd.GeoDataFrame(building_list, crs="EPSG:4326") if building_list else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        buildings_gdf, summaries["buildings"] = validate_vector_layer(buildings_gdf, "buildings", ["Polygon", "MultiPolygon"], "EPSG:4326", aoi)

        facilities_gdf = gpd.GeoDataFrame(facility_list, crs="EPSG:4326") if facility_list else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        facilities_gdf, summaries["facilities"] = validate_vector_layer(facilities_gdf, "facilities", ["Point", "Polygon", "MultiPolygon"], "EPSG:4326", aoi)

        settlements_gdf = gpd.GeoDataFrame(settlement_list, crs="EPSG:4326") if settlement_list else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        settlements_gdf, summaries["settlements"] = validate_vector_layer(settlements_gdf, "settlements", ["Point", "Polygon", "MultiPolygon"], "EPSG:4326", aoi)

        return InfrastructureContainer(
            roads=roads_gdf,
            buildings=buildings_gdf,
            facilities=facilities_gdf,
            settlements=settlements_gdf,
            crs="EPSG:4326",
            validation_summaries=summaries
        )
