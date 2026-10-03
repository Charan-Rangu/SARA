"""Preliminary Downstream Flow-Path Tracing Module for SARA.

Traces preliminary D8 steepest-descent terrain flow paths from user-specified starting points on a
Copernicus DEM elevation raster.

Important:
Generated flow paths are explicitly labeled 'MODELED_TERRAIN_FLOW_PATH'.
They represent modeled Indications of possible surface water movement based on elevation topology;
they do NOT predict downstream water depth, flood timing, or confirmed flood extent.
"""

from typing import Optional, Any
import numpy as np
import rasterio
from pydantic import BaseModel, Field


class FlowPathRecord(BaseModel):
    """Container for an individual modeled D8 flow path."""

    start_point_wgs84: tuple[float, float]
    path_points_wgs84: list[tuple[float, float]]
    path_length_steps: int
    elevation_drop_m: float
    status: str  # "FLOW_PATH_TRACED", "LOCAL_MINIMUM_REACHED", "OUT_OF_BOUNDS", "INVALID_START_POINT"


class FlowTracingResult(BaseModel):
    """Result summary containing modeled flow paths and GeoJSON FeatureCollection."""

    status: str
    total_start_points: int
    traced_paths_count: int
    flow_paths_geojson: dict[str, Any]
    flow_path_records: list[FlowPathRecord] = Field(default_factory=list)
    algorithm_info: str = "D8 Steepest Descent Flow Direction Algorithm"
    disclaimer: str = (
        "Flow paths are labeled 'MODELED_TERRAIN_FLOW_PATH'. "
        "They represent terrain-based downward flow directions and must NOT be interpreted as confirmed flood forecasts or inundation boundaries."
    )


class TerrainFlowTracer:
    """Tracer for computing D8 steepest-descent flow trajectories on DEM elevation rasters."""

    def __init__(self, max_steps: int = 500):
        self.max_steps = max_steps

    def trace_flow_paths(
        self,
        dem_path: str,
        start_points_wgs84: list[tuple[float, float]]
    ) -> FlowTracingResult:
        """Trace D8 downstream flow paths from starting points.

        Args:
            dem_path: Path to elevation GeoTIFF raster file.
            start_points_wgs84: List of starting (longitude, latitude) tuples.

        Returns:
            FlowTracingResult object containing GeoJSON LineStrings and path records.
        """
        if not start_points_wgs84:
            return FlowTracingResult(
                status="NO_START_POINTS",
                total_start_points=0,
                traced_paths_count=0,
                flow_paths_geojson={"type": "FeatureCollection", "features": []}
            )

        with rasterio.open(dem_path) as src:
            elev_array = src.read(1).astype(np.float32)
            height, width = elev_array.shape
            nodata_val = src.nodata
            transform = src.transform
            inv_transform = ~transform

            features = []
            records: list[FlowPathRecord] = []

            # 8-neighbor offsets: (drow, dcol)
            d8_neighbors = [
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)
            ]

            for start_pt in start_points_wgs84:
                start_lon, start_lat = start_pt
                # Convert WGS84 (lon, lat) to pixel (row, col)
                row, col = rasterio.transform.rowcol(transform, start_lon, start_lat)

                if not (0 <= row < height and 0 <= col < width):
                    records.append(
                        FlowPathRecord(
                            start_point_wgs84=start_pt,
                            path_points_wgs84=[start_pt],
                            path_length_steps=0,
                            elevation_drop_m=0.0,
                            status="OUT_OF_BOUNDS"
                        )
                    )
                    continue

                curr_row, curr_col = row, col
                start_elev = float(elev_array[curr_row, curr_col])

                if nodata_val is not None and start_elev == nodata_val:
                    records.append(
                        FlowPathRecord(
                            start_point_wgs84=start_pt,
                            path_points_wgs84=[start_pt],
                            path_length_steps=0,
                            elevation_drop_m=0.0,
                            status="INVALID_START_POINT"
                        )
                    )
                    continue

                path_pixels = [(curr_row, curr_col)]
                visited = {(curr_row, curr_col)}

                for _ in range(self.max_steps):
                    curr_val = elev_array[curr_row, curr_col]
                    best_drop = 0.0
                    best_next = None

                    # Find D8 neighbor with maximum elevation drop
                    for dr, dc in d8_neighbors:
                        nr, nc = curr_row + dr, curr_col + dc
                        if 0 <= nr < height and 0 <= nc < width:
                            n_val = elev_array[nr, nc]
                            if nodata_val is not None and n_val == nodata_val:
                                continue
                            drop = curr_val - n_val
                            if drop > best_drop and (nr, nc) not in visited:
                                best_drop = drop
                                best_next = (nr, nc)

                    if best_next is None or best_drop <= 0:
                        # Local minimum reached or no lower valid neighbor
                        break

                    curr_row, curr_col = best_next
                    visited.add((curr_row, curr_col))
                    path_pixels.append((curr_row, curr_col))

                # Convert pixel path coordinates back to WGS84 (lon, lat)
                path_wgs84 = []
                for pr, pc in path_pixels:
                    wx, wy = rasterio.transform.xy(transform, pr, pc)
                    path_wgs84.append((round(wx, 5), round(wy, 5)))

                end_elev = float(elev_array[path_pixels[-1][0], path_pixels[-1][1]])
                elev_drop = round(start_elev - end_elev, 1)

                rec = FlowPathRecord(
                    start_point_wgs84=start_pt,
                    path_points_wgs84=path_wgs84,
                    path_length_steps=len(path_wgs84),
                    elevation_drop_m=elev_drop,
                    status="FLOW_PATH_TRACED" if len(path_wgs84) > 1 else "LOCAL_MINIMUM_REACHED"
                )
                records.append(rec)

                if len(path_wgs84) > 1:
                    feat = {
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[p[0], p[1]] for p in path_wgs84]
                        },
                        "properties": {
                            "layer": "MODELED_TERRAIN_FLOW_PATH",
                            "start_lon": start_lon,
                            "start_lat": start_lat,
                            "elevation_drop_m": elev_drop,
                            "steps_count": len(path_wgs84)
                        }
                    }
                    features.append(feat)

            geojson_fc = {
                "type": "FeatureCollection",
                "features": features
            }

            return FlowTracingResult(
                status="SUCCESS",
                total_start_points=len(start_points_wgs84),
                traced_paths_count=len(features),
                flow_paths_geojson=geojson_fc,
                flow_path_records=records
            )
