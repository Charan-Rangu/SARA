"""Area of Interest (AOI) representation and spatial validation module for SARA.

Supports bounding boxes and GeoJSON polygons with strict WGS84 coordinate bounds,
validity checks, and spatial intersection calculations.
"""

from typing import Any
import shapely.geometry
from shapely.geometry.base import BaseGeometry

# Maximum allowed AOI bounding box area in square degrees (approx 5 deg x 5 deg ~ 300,000 km2)
MAX_AOI_AREA_SQ_DEG = 25.0

class AOI:
    """Represents a spatial Area of Interest (AOI) in WGS84 (EPSG:4326) coordinates."""

    def __init__(
        self,
        bbox: tuple[float, float, float, float] | list[float] | None = None,
        geojson: dict[str, Any] | None = None,
        name: str = "Custom AOI"
    ):
        """Initialize and validate an AOI.

        Args:
            bbox: Tuple or list of (min_lon, min_lat, max_lon, max_lat) in WGS84.
            geojson: GeoJSON geometry dict or Feature dict.
            name: Human-readable name for the AOI.
        """
        self.name = name
        self._shapely_geom: BaseGeometry | None = None
        self._bbox: tuple[float, float, float, float] | None = None

        if bbox is None and geojson is None:
            raise ValueError("Must provide either 'bbox' or 'geojson' to construct an AOI.")

        if geojson is not None:
            self._parse_geojson(geojson)
        elif bbox is not None:
            self._parse_bbox(bbox)

    def _parse_bbox(self, bbox: tuple[float, float, float, float] | list[float]) -> None:
        if len(bbox) != 4:
            raise ValueError(f"Bounding box must contain exactly 4 coordinates, got {len(bbox)}.")

        min_lon, min_lat, max_lon, max_lat = (float(c) for c in bbox)

        # Coordinate bound checks
        if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0):
            raise ValueError(f"Longitude coordinates out of WGS84 bounds [-180, 180]: min_lon={min_lon}, max_lon={max_lon}")
        if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
            raise ValueError(f"Latitude coordinates out of WGS84 bounds [-90, 90]: min_lat={min_lat}, max_lat={max_lat}")

        if min_lon >= max_lon:
            raise ValueError(f"Invalid bounding box extent: min_lon ({min_lon}) >= max_lon ({max_lon})")
        if min_lat >= max_lat:
            raise ValueError(f"Invalid bounding box extent: min_lat ({min_lat}) >= max_lat ({max_lat})")

        self._bbox = (min_lon, min_lat, max_lon, max_lat)
        self._shapely_geom = shapely.geometry.box(min_lon, min_lat, max_lon, max_lat)
        self._validate_geometry()

    def _parse_geojson(self, geojson: dict[str, Any]) -> None:
        # If passed a GeoJSON Feature or FeatureCollection, extract geometry
        geom_dict = geojson
        if geojson.get("type") == "Feature":
            geom_dict = geojson.get("geometry", {})
        elif geojson.get("type") == "FeatureCollection":
            features = geojson.get("features", [])
            if not features:
                raise ValueError("GeoJSON FeatureCollection is empty.")
            geom_dict = features[0].get("geometry", {})

        try:
            geom = shapely.geometry.shape(geom_dict)
        except Exception as e:
            raise ValueError(f"Invalid GeoJSON geometry structure: {e}")

        if geom.is_empty:
            raise ValueError("GeoJSON geometry is empty.")

        if not geom.is_valid:
            raise ValueError("GeoJSON geometry is invalid according to OGC standards.")

        min_lon, min_lat, max_lon, max_lat = geom.bounds
        if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0):
            raise ValueError(f"GeoJSON Longitude out of bounds [-180, 180]: [{min_lon}, {max_lon}]")
        if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
            raise ValueError(f"GeoJSON Latitude out of bounds [-90, 90]: [{min_lat}, {max_lat}]")

        self._shapely_geom = geom
        self._bbox = (min_lon, min_lat, max_lon, max_lat)
        self._validate_geometry()

    def _validate_geometry(self) -> None:
        area = self.area_sq_deg()
        if area > MAX_AOI_AREA_SQ_DEG:
            raise ValueError(
                f"AOI area ({area:.2f} sq deg) exceeds maximum allowed threshold ({MAX_AOI_AREA_SQ_DEG} sq deg)."
            )

    def area_sq_deg(self) -> float:
        """Calculate AOI area in square degrees."""
        return float(self._shapely_geom.area) if self._shapely_geom else 0.0

    def to_bbox(self) -> tuple[float, float, float, float]:
        """Return bounding box as (min_lon, min_lat, max_lon, max_lat)."""
        assert self._bbox is not None
        return self._bbox

    def to_geojson(self) -> dict[str, Any]:
        """Return GeoJSON Geometry dict."""
        assert self._shapely_geom is not None
        return shapely.geometry.mapping(self._shapely_geom)

    def to_shapely(self) -> BaseGeometry:
        """Return underlying Shapely geometry object."""
        assert self._shapely_geom is not None
        return self._shapely_geom

    def to_wkt(self) -> str:
        """Return Well-Known Text (WKT) string."""
        assert self._shapely_geom is not None
        return self._shapely_geom.wkt

    def intersection_percentage(self, other_geom: dict[str, Any] | BaseGeometry) -> float:
        """Calculate what percentage of this AOI is covered by another geometry."""
        if self._shapely_geom is None or self._shapely_geom.area == 0:
            return 0.0

        if isinstance(other_geom, dict):
            try:
                target_geom = shapely.geometry.shape(other_geom)
            except Exception:
                return 0.0
        else:
            target_geom = other_geom

        if not target_geom.is_valid or target_geom.is_empty:
            return 0.0

        intersection = self._shapely_geom.intersection(target_geom)
        overlap_pct = (intersection.area / self._shapely_geom.area) * 100.0
        return round(float(overlap_pct), 2)


# EXAMPLE CONFIGURATION AOI (Configuration Example — Not a verified disaster boundary)
KERALA_FLOOD_EXAMPLE_AOI = AOI(
    bbox=(76.15, 9.85, 76.45, 10.15),
    name="Configuration Example: Ernakulam / Kochi Region, Kerala"
)

# TRACK B CHALLENGE AOI — Northern Nepal (Bhote Koshi–Trishuli Corridor, Rasuwa District)
NEPAL_BHOTE_KOSHI_AOI = AOI(
    bbox=(85.25, 28.10, 85.42, 28.30),
    name="Bhote Koshi–Trishuli Corridor, Rasuwa, Nepal"
)

