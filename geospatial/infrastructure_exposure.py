"""Flood-Change Exposure Overlay & Infrastructure Assessment Module for SARA.

Intersects OpenStreetMap vector infrastructure (Roads, Buildings, Healthcare Facilities, and Settlements)
with preliminary SAR change-detection masks to identify potentially exposed features.

Important:
Exposure labels use explicit cautious terms:
- 'potential_change_overlap'
- 'near_preliminary_change'
- 'no_detected_overlap'
- 'not_assessed'

Features MUST NEVER be labeled 'destroyed' or 'confirmed_flooded' based solely on SAR backscatter change overlap.
"""

import os
from typing import Optional, Any
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from pydantic import BaseModel, Field

from geospatial.aoi import AOI
from geospatial.infrastructure import InfrastructureContainer


class ExposureSummary(BaseModel):
    """Summary statistics for infrastructure exposure analysis."""

    total_roads: int = 0
    roads_potential_overlap_count: int = 0
    roads_near_change_count: int = 0
    total_buildings: int = 0
    buildings_potential_overlap_count: int = 0
    total_facilities: int = 0
    facilities_potential_overlap_count: int = 0
    total_settlements: int = 0
    settlements_potential_overlap_count: int = 0
    disclaimer: str = (
        "Exposure analysis is preliminary and based on SAR backscatter change detection. "
        "Features labeled 'potential_change_overlap' require field/optical verification and must NOT be classified as confirmed destroyed."
    )


class InfrastructureExposureAnalyzer:
    """Analyzer for evaluating raster-vector intersections between preliminary SAR change masks and OSM features."""

    def __init__(self, proximity_distance_meters: float = 30.0):
        self.proximity_distance_meters = proximity_distance_meters

    def analyze_exposure(
        self,
        aoi: AOI,
        infrastructure: InfrastructureContainer,
        change_mask_path: Optional[str] = None
    ) -> tuple[dict[str, gpd.GeoDataFrame], ExposureSummary]:
        """Intersect infrastructure vector layers with a preliminary SAR change mask.

        Args:
            aoi: Target Area of Interest.
            infrastructure: InfrastructureContainer with validated GeoDataFrames.
            change_mask_path: Path to preliminary_change_mask.tif (1=change, 0=no change, 255=nodata).

        Returns:
            Tuple of (dict of assessed GeoDataFrames, ExposureSummary).
        """
        output_gdfs: dict[str, gpd.GeoDataFrame] = {}

        if change_mask_path is None or not os.path.exists(change_mask_path):
            # No change mask available — tag all features as 'not_assessed'
            return self._tag_unassessed(infrastructure)

        with rasterio.open(change_mask_path) as mask_src:
            raster_crs = mask_src.crs or "EPSG:4326"
            mask_transform = mask_src.transform
            mask_shape = (mask_src.height, mask_src.width)

            # Read full mask raster (or window) into array
            mask_array = mask_src.read(1)
            change_pixels_mask = (mask_array == 1)

            # 1. Assess Roads
            roads_assessed, r_overlap_count, r_near_count = self._assess_roads(
                infrastructure.roads, mask_src, change_pixels_mask, raster_crs
            )
            output_gdfs["potentially_affected_roads"] = roads_assessed

            # 2. Assess Buildings
            bldg_assessed, b_overlap_count = self._assess_polygons_or_points(
                infrastructure.buildings, mask_src, change_pixels_mask, raster_crs, layer_type="building"
            )
            output_gdfs["potentially_exposed_buildings"] = bldg_assessed

            # 3. Assess Healthcare Facilities
            fac_assessed, f_overlap_count = self._assess_polygons_or_points(
                infrastructure.facilities, mask_src, change_pixels_mask, raster_crs, layer_type="facility"
            )
            output_gdfs["potentially_exposed_facilities"] = fac_assessed

            # 4. Assess Settlements
            settle_assessed, s_overlap_count = self._assess_polygons_or_points(
                infrastructure.settlements, mask_src, change_pixels_mask, raster_crs, layer_type="settlement"
            )
            output_gdfs["settlements_exposure"] = settle_assessed

        summary = ExposureSummary(
            total_roads=len(infrastructure.roads),
            roads_potential_overlap_count=r_overlap_count,
            roads_near_change_count=r_near_count,
            total_buildings=len(infrastructure.buildings),
            buildings_potential_overlap_count=b_overlap_count,
            total_facilities=len(infrastructure.facilities),
            facilities_potential_overlap_count=f_overlap_count,
            total_settlements=len(infrastructure.settlements),
            settlements_potential_overlap_count=s_overlap_count
        )

        return output_gdfs, summary

    def _tag_unassessed(self, infrastructure: InfrastructureContainer) -> tuple[dict[str, gpd.GeoDataFrame], ExposureSummary]:
        out_dict = {}
        for name, gdf in [
            ("potentially_affected_roads", infrastructure.roads),
            ("potentially_exposed_buildings", infrastructure.buildings),
            ("potentially_exposed_facilities", infrastructure.facilities),
            ("settlements_exposure", infrastructure.settlements)
        ]:
            df = gdf.copy()
            df["exposure_label"] = "not_assessed"
            df["overlap_pixel_count"] = 0
            df["intersecting_fraction"] = 0.0
            out_dict[name] = df

        summary = ExposureSummary(
            total_roads=len(infrastructure.roads),
            total_buildings=len(infrastructure.buildings),
            total_facilities=len(infrastructure.facilities),
            total_settlements=len(infrastructure.settlements)
        )
        return out_dict, summary

    def _assess_roads(
        self,
        roads_gdf: gpd.GeoDataFrame,
        mask_src: rasterio.DatasetReader,
        change_pixels_mask: np.ndarray,
        raster_crs: Any
    ) -> tuple[gpd.GeoDataFrame, int, int]:
        if len(roads_gdf) == 0:
            empty_df = roads_gdf.copy()
            empty_df["exposure_label"] = []
            empty_df["intersecting_length_meters"] = []
            empty_df["intersecting_fraction"] = []
            return empty_df, 0, 0

        # Reproject roads to raster CRS & metric projection for length calculations
        roads_proj = roads_gdf.to_crs(raster_crs).copy()
        roads_metric = roads_gdf.to_crs("EPSG:3857").copy()

        labels = []
        intersect_lengths = []
        intersect_fractions = []

        overlap_count = 0
        near_count = 0

        transform = mask_src.transform
        inv_transform = ~transform

        for idx, row in roads_proj.iterrows():
            geom = row.geometry
            metric_geom = roads_metric.loc[idx, 'geometry']
            total_length_m = float(metric_geom.length)

            if geom is None or geom.is_empty:
                labels.append("no_detected_overlap")
                intersect_lengths.append(0.0)
                intersect_fractions.append(0.0)
                continue

            # Rasterize road line to find intersected pixel coordinates
            try:
                line_mask = rasterize(
                    [(geom, 1)],
                    out_shape=(mask_src.height, mask_src.width),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8
                ) == 1
            except Exception:
                line_mask = np.zeros((mask_src.height, mask_src.width), dtype=bool)

            sampled_change = change_pixels_mask & line_mask
            intersected_pixel_count = int(np.sum(sampled_change))
            total_road_pixels = int(np.sum(line_mask))

            fraction = (intersected_pixel_count / total_road_pixels) if total_road_pixels > 0 else 0.0
            approx_intersect_len_m = round(total_length_m * fraction, 1)

            if intersected_pixel_count > 0:
                label = "potential_change_overlap"
                overlap_count += 1
            else:
                label = "no_detected_overlap"

            labels.append(label)
            intersect_lengths.append(approx_intersect_len_m)
            intersect_fractions.append(round(float(fraction), 3))

        result_gdf = roads_gdf.copy()
        result_gdf["exposure_label"] = labels
        result_gdf["intersecting_length_meters"] = intersect_lengths
        result_gdf["intersecting_fraction"] = intersect_fractions

        return result_gdf, overlap_count, near_count

    def _assess_polygons_or_points(
        self,
        gdf: gpd.GeoDataFrame,
        mask_src: rasterio.DatasetReader,
        change_pixels_mask: np.ndarray,
        raster_crs: Any,
        layer_type: str
    ) -> tuple[gpd.GeoDataFrame, int]:
        if len(gdf) == 0:
            empty_df = gdf.copy()
            empty_df["exposure_label"] = []
            empty_df["overlap_pixel_count"] = []
            return empty_df, 0

        gdf_proj = gdf.to_crs(raster_crs).copy()
        labels = []
        pixel_counts = []
        overlap_count = 0

        transform = mask_src.transform

        for idx, row in gdf_proj.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                labels.append("no_detected_overlap")
                pixel_counts.append(0)
                continue

            try:
                if geom.geom_type == "Point":
                    # Buffer point slightly to sample underlying pixel
                    geom_sample = geom.buffer(0.0001)
                else:
                    geom_sample = geom

                feat_mask = rasterize(
                    [(geom_sample, 1)],
                    out_shape=(mask_src.height, mask_src.width),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8
                ) == 1
            except Exception:
                feat_mask = np.zeros((mask_src.height, mask_src.width), dtype=bool)

            sampled_change = change_pixels_mask & feat_mask
            count = int(np.sum(sampled_change))

            if count > 0:
                label = "potential_change_overlap"
                overlap_count += 1
            else:
                label = "no_detected_overlap"

            labels.append(label)
            pixel_counts.append(count)

        res_df = gdf.copy()
        res_df["exposure_label"] = labels
        res_df["overlap_pixel_count"] = pixel_counts
        return res_df, overlap_count
