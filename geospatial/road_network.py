"""Road Network Graph Construction Module for SARA.

Converts GeoPandas road LineString and MultiLineString vector geometries into a NetworkX graph
supporting topological distance metrics and route analysis.
"""

from typing import Optional, Any
import math
import geopandas as gpd
import shapely.geometry
import networkx as nx
from pydantic import BaseModel, Field


class GraphSummary(BaseModel):
    """Summary metrics describing road network topology and component connectivity."""

    total_nodes: int
    total_edges: int
    connected_components_count: int
    largest_component_nodes: int
    is_fully_connected: bool
    metric_crs: str
    node_snapping_tolerance_meters: float
    notes: list[str] = Field(default_factory=list)


class RoadNetworkBuilder:
    """Builder for constructing NetworkX topological graphs from spatial GeoDataFrames."""

    def __init__(self, metric_crs: str = "EPSG:3857", node_snap_tolerance_meters: float = 2.0):
        self.metric_crs = metric_crs
        self.node_snap_tolerance_meters = node_snap_tolerance_meters

    def build_graph(self, roads_gdf: gpd.GeoDataFrame) -> tuple[nx.Graph, GraphSummary]:
        """Convert roads GeoDataFrame into a weighted NetworkX Graph.

        Args:
            roads_gdf: GeoDataFrame containing road LineString/MultiLineString geometries.

        Returns:
            Tuple of (nx.Graph, GraphSummary).
        """
        graph = nx.Graph()

        if len(roads_gdf) == 0:
            summary = GraphSummary(
                total_nodes=0,
                total_edges=0,
                connected_components_count=0,
                largest_component_nodes=0,
                is_fully_connected=True,
                metric_crs=self.metric_crs,
                node_snapping_tolerance_meters=self.node_snap_tolerance_meters,
                notes=["Roads GeoDataFrame was empty."]
            )
            return graph, summary

        # Reproject to metric CRS for accurate distance calculations
        roads_proj = roads_gdf.to_crs(self.metric_crs).copy()

        def snap_coord(pt: tuple[float, float]) -> tuple[float, float]:
            """Snap node coordinates based on tolerance distance in meters."""
            tol = self.node_snap_tolerance_meters
            if tol <= 0:
                return (round(pt[0], 2), round(pt[1], 2))
            return (round(pt[0] / tol) * tol, round(pt[1] / tol) * tol)

        notes: list[str] = []

        for idx, row in roads_proj.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            road_id = row.get("id") or str(idx)
            name = row.get("name", "Unnamed Road")
            highway = row.get("highway", "road")

            # Extract individual LineStrings if MultiLineString
            lines = [geom] if geom.geom_type == "LineString" else list(geom.geoms)

            for line in lines:
                coords = list(line.coords)
                if len(coords) < 2:
                    continue

                # Add edges for line segments
                for i in range(len(coords) - 1):
                    p1 = snap_coord(coords[i])
                    p2 = snap_coord(coords[i + 1])

                    if p1 == p2:
                        continue  # Skip zero-length segment resulting from snapping

                    segment_len = math.sqrt((coords[i+1][0] - coords[i][0])**2 + (coords[i+1][1] - coords[i][1])**2)

                    # Store edge with metadata
                    graph.add_edge(
                        p1, p2,
                        road_id=road_id,
                        name=name,
                        highway=highway,
                        weight=segment_len,  # Length in meters
                        original_idx=idx
                    )

                    # Store node spatial coordinates
                    graph.nodes[p1]["x"] = p1[0]
                    graph.nodes[p1]["y"] = p1[1]
                    graph.nodes[p2]["x"] = p2[0]
                    graph.nodes[p2]["y"] = p2[1]

        total_nodes = graph.number_of_nodes()
        total_edges = graph.number_of_edges()

        if total_nodes > 0:
            components = list(nx.connected_components(graph))
            comp_count = len(components)
            largest_comp = max(len(c) for c in components)
            is_connected = (comp_count == 1)
        else:
            comp_count = 0
            largest_comp = 0
            is_connected = True

        summary = GraphSummary(
            total_nodes=total_nodes,
            total_edges=total_edges,
            connected_components_count=comp_count,
            largest_component_nodes=largest_comp,
            is_fully_connected=is_connected,
            metric_crs=self.metric_crs,
            node_snapping_tolerance_meters=self.node_snap_tolerance_meters,
            notes=notes
        )

        return graph, summary
