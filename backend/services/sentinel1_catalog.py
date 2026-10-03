"""Sentinel-1 Catalogue Discovery Service for SARA.

Discovers Sentinel-1 SAR satellite imagery using the official Copernicus Data Space Ecosystem (CDSE)
OData API endpoint (https://catalogue.dataspace.copernicus.eu/odata/v1/Products).
Metadata search is free, open, and anonymous without requiring credentials.
"""

import os
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from pydantic import BaseModel, Field
import httpx

from geospatial.aoi import AOI

CDSE_ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CACHE_FILE_PATH = os.path.join("data", "cache", "catalog_cache.json")


class Sentinel1Scene(BaseModel):
    """Metadata model representing a single Sentinel-1 SAR acquisition scene."""

    id: str
    title: str
    acquisition_time: str
    platform: str = "UNKNOWN"
    product_type: str = "UNKNOWN"
    sensor_mode: str = "UNKNOWN"
    polarizations: list[str] = Field(default_factory=list)
    relative_orbit_number: Optional[int] = None
    orbit_direction: Optional[str] = None  # "ASCENDING" or "DESCENDING"
    footprint_geojson: Optional[dict[str, Any]] = None
    download_url: Optional[str] = None
    size_bytes: Optional[int] = None


class CatalogSearchResult(BaseModel):
    """Container for pre-event and post-event satellite scene search results."""

    event_date: str
    pre_event_window: tuple[str, str]
    post_event_window: tuple[str, str]
    pre_event_scenes: list[Sentinel1Scene] = Field(default_factory=list)
    post_event_scenes: list[Sentinel1Scene] = Field(default_factory=list)
    total_scenes_found: int = 0
    from_cache: bool = False
    warning_messages: list[str] = Field(default_factory=list)


def parse_datetime(dt_str: str) -> datetime:
    """Parse ISO date or YYYY-MM-DD string into UTC datetime object."""
    if "T" in dt_str:
        dt_clean = dt_str.replace("Z", "+00:00")
        return datetime.fromisoformat(dt_clean).astimezone(timezone.utc)
    return datetime.strptime(dt_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def normalize_odata_product(item: dict[str, Any]) -> Sentinel1Scene:
    """Normalize CDSE OData product JSON into a structured Sentinel1Scene model."""
    prod_id = item.get("Id", "UNKNOWN_ID")
    name = item.get("Name", prod_id)
    
    # Dates
    content_date = item.get("ContentDate", {})
    acq_time = content_date.get("Start") or item.get("PublicationDate") or "UNKNOWN_TIME"

    # Platform (SENTINEL-1A, SENTINEL-1B)
    platform = "SENTINEL-1"
    if name.startswith("S1A"):
        platform = "SENTINEL-1A"
    elif name.startswith("S1B"):
        platform = "SENTINEL-1B"
    elif name.startswith("S1C"):
        platform = "SENTINEL-1C"

    # Extract attributes array if present
    attributes = item.get("Attributes", [])
    attr_map = {}
    for attr in attributes:
        aname = attr.get("Name")
        avalue = attr.get("Value")
        if aname and avalue is not None:
            attr_map[aname] = avalue

    # Product Type (GRD, SLC, etc.)
    prod_type = attr_map.get("productType") or ("GRD" if "GRD" in name else "UNKNOWN")

    # Sensor Mode (IW, EW, SM)
    sensor_mode = attr_map.get("operationalMode") or attr_map.get("sensorMode") or ("IW" if "_IW_" in name else "UNKNOWN")

    # Polarizations (e.g. VV, VH)
    pols = []
    if "polarizationChannels" in attr_map:
        p_val = attr_map["polarizationChannels"]
        pols = [p_val] if isinstance(p_val, str) else list(p_val)
    elif "_1SDV_" in name:
        pols = ["VV", "VH"]
    elif "_1SDH_" in name:
        pols = ["HH", "HV"]
    elif "_1SSV_" in name:
        pols = ["VV"]

    # Relative Orbit Number
    rel_orbit = attr_map.get("relativeOrbitNumber")
    if rel_orbit is not None:
        try:
            rel_orbit = int(rel_orbit)
        except (ValueError, TypeError):
            rel_orbit = None

    # Orbit Direction (ASCENDING / DESCENDING)
    orbit_dir = attr_map.get("orbitDirection")
    if orbit_dir:
        orbit_dir = str(orbit_dir).upper()

    # Geometry GeoJSON
    geo_dict = item.get("GeoFootprint")

    # Size & Download URL
    size_bytes = item.get("ContentLength")
    if size_bytes is not None:
        try:
            size_bytes = int(size_bytes)
        except (ValueError, TypeError):
            size_bytes = None

    download_url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({prod_id})/$value"

    return Sentinel1Scene(
        id=prod_id,
        title=name,
        acquisition_time=acq_time,
        platform=platform.upper(),
        product_type=prod_type.upper(),
        sensor_mode=sensor_mode.upper(),
        polarizations=[p.upper() for p in pols],
        relative_orbit_number=rel_orbit,
        orbit_direction=orbit_dir,
        footprint_geojson=geo_dict,
        download_url=download_url,
        size_bytes=size_bytes
    )


def normalize_stac_item(item: dict[str, Any]) -> Sentinel1Scene:
    """Normalize raw STAC API JSON item into a structured Sentinel1Scene model (STAC compatibility wrapper)."""
    scene_id = item.get("id", "UNKNOWN_ID")
    props = item.get("properties", {})
    geometry = item.get("geometry")

    title = props.get("title") or props.get("productIdentifier") or scene_id
    acq_time = props.get("datetime") or props.get("start_datetime") or "UNKNOWN_TIME"

    platform = props.get("platform") or props.get("sat:platform") or "SENTINEL-1"
    prod_type = props.get("sar:product_type") or props.get("productType") or props.get("s1:product_type") or "GRD"
    sensor_mode = props.get("sar:instrument_mode") or props.get("sensorMode") or props.get("s1:instrument_mode") or "IW"

    polarizations = props.get("sar:polarizations") or props.get("polarizationChannels") or []
    if isinstance(polarizations, str):
        polarizations = [polarizations]

    rel_orbit = props.get("sat:relative_orbit") or props.get("relativeOrbitNumber") or props.get("s1:relative_orbit")
    if rel_orbit is not None:
        try:
            rel_orbit = int(rel_orbit)
        except (ValueError, TypeError):
            rel_orbit = None

    orbit_dir = props.get("sat:orbit_state") or props.get("orbitDirection") or props.get("s1:orbit_direction")
    if orbit_dir:
        orbit_dir = str(orbit_dir).upper()

    assets = item.get("assets", {})
    download_url = None
    size_bytes = None

    if "download" in assets:
        download_url = assets["download"].get("href")
        size_bytes = assets["download"].get("file:size")
    elif "data" in assets:
        download_url = assets["data"].get("href")
        size_bytes = assets["data"].get("file:size")

    if size_bytes is not None:
        try:
            size_bytes = int(size_bytes)
        except (ValueError, TypeError):
            size_bytes = None

    return Sentinel1Scene(
        id=scene_id,
        title=title,
        acquisition_time=acq_time,
        platform=platform.upper(),
        product_type=prod_type.upper(),
        sensor_mode=sensor_mode.upper(),
        polarizations=[p.upper() for p in polarizations],
        relative_orbit_number=rel_orbit,
        orbit_direction=orbit_dir,
        footprint_geojson=geometry,
        download_url=download_url,
        size_bytes=size_bytes
    )


class Sentinel1CatalogService:
    """Service for querying Sentinel-1 metadata via CDSE OData API and managing cache."""

    def __init__(self, cache_file: str = CACHE_FILE_PATH, timeout_seconds: float = 15.0):
        self.cache_file = cache_file
        self.timeout_seconds = timeout_seconds

    def search_scenes(
        self,
        aoi: AOI,
        event_date_str: str,
        pre_event_days: int = 14,
        post_event_days: int = 14,
        max_results: int = 50,
        mock_response: Optional[dict[str, Any]] = None
    ) -> CatalogSearchResult:
        """Search Sentinel-1 scenes around an event date for the given AOI."""
        event_dt = parse_datetime(event_date_str)
        pre_start = event_dt - timedelta(days=pre_event_days)
        post_end = event_dt + timedelta(days=post_event_days)

        pre_win = (pre_start.strftime("%Y-%m-%dT%H:%M:%SZ"), event_dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
        post_win = (event_dt.strftime("%Y-%m-%dT%H:%M:%SZ"), post_end.strftime("%Y-%m-%dT%H:%M:%SZ"))

        if mock_response is not None:
            return self._parse_mock_response(mock_response, event_dt, pre_win, post_win)

        min_lon, min_lat, max_lon, max_lat = aoi.to_bbox()
        poly_wkt = f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, {max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
        
        start_iso = pre_start.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        end_iso = post_end.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        filter_str = (
            f"Collection/Name eq 'SENTINEL-1' and "
            f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/Value eq 'GRD') and "
            f"OData.CSC.Intersects(area=geography'SRID=4326;{poly_wkt}') and "
            f"ContentDate/Start ge {start_iso} and ContentDate/Start le {end_iso}"
        )

        params = {
            "$filter": filter_str,
            "$top": max_results,
            "$expand": "Attributes"
        }

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.get(CDSE_ODATA_URL, params=params)
                response.raise_for_status()
                odata_data = response.json()
                result = self._parse_odata_response(odata_data, event_dt, pre_win, post_win)
                self._save_cache(result)
                return result
        except Exception as e:
            cached_result = self._load_cache()
            if cached_result:
                cached_result.warning_messages.append(
                    f"Live catalogue search failed ({type(e).__name__}: {e}). Returned cached metadata."
                )
                return cached_result
            
            return CatalogSearchResult(
                event_date=event_date_str,
                pre_event_window=pre_win,
                post_event_window=post_win,
                warning_messages=[f"Catalogue search failed and no local cache available: {e}"]
            )

    def _parse_odata_response(
        self,
        odata_json: dict[str, Any],
        event_dt: datetime,
        pre_win: tuple[str, str],
        post_win: tuple[str, str]
    ) -> CatalogSearchResult:
        items = odata_json.get("value", [])
        seen_ids = set()
        pre_scenes: list[Sentinel1Scene] = []
        post_scenes: list[Sentinel1Scene] = []

        for item in items:
            scene = normalize_odata_product(item)
            if scene.id in seen_ids:
                continue
            seen_ids.add(scene.id)

            try:
                acq_dt = parse_datetime(scene.acquisition_time)
            except Exception:
                continue

            if acq_dt <= event_dt:
                pre_scenes.append(scene)
            else:
                post_scenes.append(scene)

        pre_scenes.sort(key=lambda s: s.acquisition_time, reverse=True)
        post_scenes.sort(key=lambda s: s.acquisition_time)

        return CatalogSearchResult(
            event_date=event_dt.strftime("%Y-%m-%d"),
            pre_event_window=pre_win,
            post_event_window=post_win,
            pre_event_scenes=pre_scenes,
            post_event_scenes=post_scenes,
            total_scenes_found=len(pre_scenes) + len(post_scenes)
        )

    def _parse_mock_response(
        self,
        mock_data: dict[str, Any],
        event_dt: datetime,
        pre_win: tuple[str, str],
        post_win: tuple[str, str]
    ) -> CatalogSearchResult:
        features = mock_data.get("features", [])
        if not features and "value" in mock_data:
            return self._parse_odata_response(mock_data, event_dt, pre_win, post_win)

        seen_ids = set()
        pre_scenes: list[Sentinel1Scene] = []
        post_scenes: list[Sentinel1Scene] = []

        for feat in features:
            scene = normalize_stac_item(feat)
            if scene.id in seen_ids:
                continue
            seen_ids.add(scene.id)

            try:
                acq_dt = parse_datetime(scene.acquisition_time)
            except Exception:
                continue

            if acq_dt <= event_dt:
                pre_scenes.append(scene)
            else:
                post_scenes.append(scene)

        pre_scenes.sort(key=lambda s: s.acquisition_time, reverse=True)
        post_scenes.sort(key=lambda s: s.acquisition_time)

        return CatalogSearchResult(
            event_date=event_dt.strftime("%Y-%m-%d"),
            pre_event_window=pre_win,
            post_event_window=post_win,
            pre_event_scenes=pre_scenes,
            post_event_scenes=post_scenes,
            total_scenes_found=len(pre_scenes) + len(post_scenes)
        )

    def _save_cache(self, result: CatalogSearchResult) -> None:
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(result.model_dump(), f, indent=2)
        except Exception:
            pass

    def _load_cache(self) -> Optional[CatalogSearchResult]:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    res = CatalogSearchResult(**data)
                    res.from_cache = True
                    return res
            except Exception:
                return None
        return None
