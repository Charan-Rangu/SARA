"""Vector Infrastructure Data Contract & Validation Module for SARA.

Manages, validates, and standardizes OpenStreetMap vector infrastructure layers (Roads, Buildings,
Healthcare Facilities, and Settlements) using GeoPandas, Shapely, and PyProj.
"""

from typing import Optional, Any
import pandas as pd
import geopandas as gpd
import shapely.geometry
from shapely.geometry.base import BaseGeometry
from pydantic import BaseModel, Field

from geospatial.aoi import AOI


class ValidationSummary(BaseModel):
    """Summary statistics for vector layer validation and cleaning."""

    layer_name: str
    total_input_features: int
    valid_features_count: int
    repaired_features_count: int
    excluded_invalid_count: int
    excluded_empty_count: int
    excluded_outside_aoi_count: int
    duplicate_features_removed: int
    target_crs: str
    notes: list[str] = Field(default_factory=list)


def validate_vector_layer(
    gdf: gpd.GeoDataFrame,
    layer_name: str,
    allowed_geom_types: list[str],
    target_crs: str = "EPSG:4326",
    aoi: Optional[AOI] = None,
    id_column: Optional[str] = None
) -> tuple[gpd.GeoDataFrame, ValidationSummary]:
    """Validate, repair, reproject, and clean a vector GeoDataFrame layer.

    Args:
        gdf: Input GeoPandas GeoDataFrame.
        layer_name: Name of the infrastructure layer (e.g. "roads", "hospitals").
        allowed_geom_types: List of acceptable geometry types (e.g. ["LineString", "MultiLineString"]).
        target_crs: Target coordinate reference system (default EPSG:4326).
        aoi: Optional AOI object to filter features spatially.
        id_column: Optional column name for deduplication.

    Returns:
        Tuple of (cleaned GeoDataFrame, ValidationSummary).
    """
    total_input = len(gdf)
    notes: list[str] = []

    if total_input == 0:
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs=target_crs)
        summary = ValidationSummary(
            layer_name=layer_name,
            total_input_features=0,
            valid_features_count=0,
            repaired_features_count=0,
            excluded_invalid_count=0,
            excluded_empty_count=0,
            excluded_outside_aoi_count=0,
            duplicate_features_removed=0,
            target_crs=target_crs,
            notes=["Input GeoDataFrame was empty."]
        )
        return empty_gdf, summary

    # 1. Reproject if CRS differs
    cleaned_gdf = gdf.copy()
    if cleaned_gdf.crs is None:
        notes.append("Input CRS was missing; assuming EPSG:4326.")
        cleaned_gdf.set_crs("EPSG:4326", inplace=True)
    
    if str(cleaned_gdf.crs).upper() != target_crs.upper():
        notes.append(f"Reprojected layer from '{cleaned_gdf.crs}' to '{target_crs}'.")
        cleaned_gdf = cleaned_gdf.to_crs(target_crs)

    # 2. Filter empty geometries
    empty_mask = cleaned_gdf.geometry.is_empty | cleaned_gdf.geometry.isna()
    excluded_empty = int(empty_mask.sum())
    cleaned_gdf = cleaned_gdf[~empty_mask].copy()

    # 3. Check & Repair invalid geometries
    invalid_mask = ~cleaned_gdf.geometry.is_valid
    excluded_invalid = 0
    repaired_count = 0

    if invalid_mask.any():
        repaired_geoms = cleaned_gdf.loc[invalid_mask, 'geometry'].apply(lambda g: shapely.make_valid(g))
        # Keep features that became valid and match allowed geometry types
        valid_repaired_mask = repaired_geoms.is_valid & ~repaired_geoms.is_empty
        
        cleaned_gdf.loc[invalid_mask & valid_repaired_mask, 'geometry'] = repaired_geoms[valid_repaired_mask]
        repaired_count = int(valid_repaired_mask.sum())
        
        still_invalid = invalid_mask & ~valid_repaired_mask
        excluded_invalid = int(still_invalid.sum())
        cleaned_gdf = cleaned_gdf[~still_invalid].copy()

    # 4. Geometry Type Filter
    geom_type_mask = cleaned_gdf.geometry.geom_type.isin(allowed_geom_types)
    if not geom_type_mask.all():
        num_rejected_types = int((~geom_type_mask).sum())
        notes.append(f"Excluded {num_rejected_types} feature(s) with disallowed geometry type(s).")
        cleaned_gdf = cleaned_gdf[geom_type_mask].copy()

    # 5. Spatial AOI Filtering
    excluded_outside_aoi = 0
    if aoi is not None:
        aoi_geom = aoi.to_shapely()
        if target_crs.upper() != "EPSG:4326":
            # Reproject AOI geometry if target CRS is metric
            aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_geom], crs="EPSG:4326").to_crs(target_crs)
            aoi_geom = aoi_gdf.geometry.iloc[0]

        intersects_mask = cleaned_gdf.geometry.intersects(aoi_geom)
        excluded_outside_aoi = int((~intersects_mask).sum())
        cleaned_gdf = cleaned_gdf[intersects_mask].copy()

    # 6. Deduplication
    duplicates_removed = 0
    if id_column and id_column in cleaned_gdf.columns:
        num_before_dedup = len(cleaned_gdf)
        cleaned_gdf = cleaned_gdf.drop_duplicates(subset=[id_column], keep="first").copy()
        duplicates_removed = num_before_dedup - len(cleaned_gdf)
        if duplicates_removed > 0:
            notes.append(f"Removed {duplicates_removed} duplicate feature(s) based on column '{id_column}'.")

    summary = ValidationSummary(
        layer_name=layer_name,
        total_input_features=total_input,
        valid_features_count=len(cleaned_gdf),
        repaired_features_count=repaired_count,
        excluded_invalid_count=excluded_invalid,
        excluded_empty_count=excluded_empty,
        excluded_outside_aoi_count=excluded_outside_aoi,
        duplicate_features_removed=duplicates_removed,
        target_crs=target_crs,
        notes=notes
    )

    return cleaned_gdf, summary


class InfrastructureContainer:
    """Container holding validated GeoDataFrames for all infrastructure layers."""

    def __init__(
        self,
        roads: Optional[gpd.GeoDataFrame] = None,
        buildings: Optional[gpd.GeoDataFrame] = None,
        facilities: Optional[gpd.GeoDataFrame] = None,
        settlements: Optional[gpd.GeoDataFrame] = None,
        crs: str = "EPSG:4326",
        validation_summaries: Optional[dict[str, ValidationSummary]] = None
    ):
        self.crs = crs
        self.roads = roads if roads is not None else gpd.GeoDataFrame(geometry=[], crs=crs)
        self.buildings = buildings if buildings is not None else gpd.GeoDataFrame(geometry=[], crs=crs)
        self.facilities = facilities if facilities is not None else gpd.GeoDataFrame(geometry=[], crs=crs)
        self.settlements = settlements if settlements is not None else gpd.GeoDataFrame(geometry=[], crs=crs)
        self.validation_summaries = validation_summaries or {}

    def summary_dict(self) -> dict[str, Any]:
        """Return counts of features per infrastructure layer."""
        return {
            "crs": self.crs,
            "roads_count": len(self.roads),
            "buildings_count": len(self.buildings),
            "facilities_count": len(self.facilities),
            "settlements_count": len(self.settlements),
            "validation_summaries": {k: v.model_dump() for k, v in self.validation_summaries.items()}
        }
