"""Road Accessibility & Disconnected Settlement Assessment Module for SARA.

Evaluates baseline and scenario connectivity between populated settlements and healthcare destinations
using NetworkX topological graph analysis.

Important:
Mode B (Scenario Connectivity) evaluates hypothetical road closures. Excluding a road edge from the graph
models a hypothetical scenario based on preliminary SAR change; it does NOT prove that an unmapped ground
route or alternative access path does not exist.
"""

from typing import Optional, Any
import math
import geopandas as gpd
import shapely.geometry
import networkx as nx
from pydantic import BaseModel, Field


class SettlementAccessibilityRecord(BaseModel):
    """Accessibility record for an individual settlement."""

    settlement_id: str
    settlement_name: str
    nearest_destination_id: Optional[str] = None
    nearest_destination_name: Optional[str] = None
    baseline_reachable: bool
    baseline_distance_km: Optional[float] = None
    scenario_reachable: Optional[bool] = None
    scenario_distance_km: Optional[float] = None
    status_label: str  # "CONNECTED", "HYPOTHETICALLY_ISOLATED", "UNREACHABLE_IN_BASE_NETWORK"
    notes: list[str] = Field(default_factory=list)


class AccessibilitySummary(BaseModel):
    """Summary metrics for baseline and scenario accessibility analysis."""

    total_settlements: int
    baseline_connected_count: int
    baseline_unconnected_count: int
    scenario_enabled: bool = False
    scenario_connected_count: Optional[int] = None
    scenario_isolated_count: Optional[int] = None
    hypothetical_closed_roads_count: int = 0
    disclaimer: str = (
        "Accessibility results evaluate network reachability along mapped roads. "
        "Hypothetical road closure scenarios model preliminary change overlap and do NOT confirm physical ground blockage."
    )


class AccessibilityAnalyzer:
    """Analyzer for performing Dijkstra shortest-path reachability between settlements and hospitals."""

    def __init__(self, metric_crs: str = "EPSG:3857"):
        self.metric_crs = metric_crs

    def _find_nearest_node(self, graph: nx.Graph, pt_geom: shapely.geometry.Point) -> Optional[tuple[float, float]]:
        if graph.number_of_nodes() == 0:
            return None

        px, py = pt_geom.x, pt_geom.y
        min_dist = float("inf")
        nearest_node = None

        for node in graph.nodes():
            nx_val = graph.nodes[node].get("x", node[0])
            ny_val = graph.nodes[node].get("y", node[1])
            dist = math.sqrt((px - nx_val)**2 + (py - ny_val)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        return nearest_node

    def analyze_accessibility(
        self,
        road_graph: nx.Graph,
        settlements_gdf: gpd.GeoDataFrame,
        destinations_gdf: gpd.GeoDataFrame,
        closed_road_ids: Optional[set[str]] = None
    ) -> tuple[gpd.GeoDataFrame, AccessibilitySummary]:
        """Perform baseline and optional scenario accessibility analysis.

        Args:
            road_graph: Base NetworkX road graph.
            settlements_gdf: GeoDataFrame of settlements (points/polygons).
            destinations_gdf: GeoDataFrame of healthcare facilities/hospitals.
            closed_road_ids: Optional set of road IDs to exclude in Scenario Mode B.

        Returns:
            Tuple of (assessed settlements GeoDataFrame, AccessibilitySummary).
        """
        if len(settlements_gdf) == 0 or len(destinations_gdf) == 0 or road_graph.number_of_nodes() == 0:
            return self._empty_result(settlements_gdf)

        # Reproject inputs to metric CRS
        settlements_proj = settlements_gdf.to_crs(self.metric_crs).copy()
        destinations_proj = destinations_gdf.to_crs(self.metric_crs).copy()

        # Find nearest graph nodes for all destinations
        dest_nodes: list[tuple[str, str, tuple[float, float]]] = []
        for idx, row in destinations_proj.iterrows():
            geom = row.geometry.centroid if row.geometry.geom_type != "Point" else row.geometry
            dest_id = str(row.get("id", idx))
            dest_name = str(row.get("name", "Healthcare Destination"))
            node = self._find_nearest_node(road_graph, geom)
            if node:
                dest_nodes.append((dest_id, dest_name, node))

        # Create Scenario Graph if closed_road_ids supplied
        scenario_enabled = closed_road_ids is not None and len(closed_road_ids) > 0
        scenario_graph = road_graph.copy() if scenario_enabled else None

        if scenario_graph and closed_road_ids:
            edges_to_remove = [
                (u, v) for u, v, d in scenario_graph.edges(data=True)
                if str(d.get("road_id")) in closed_road_ids or str(d.get("original_idx")) in closed_road_ids
            ]
            scenario_graph.remove_edges_from(edges_to_remove)

        records: list[SettlementAccessibilityRecord] = []
        output_rows = []

        baseline_conn_count = 0
        scenario_conn_count = 0
        scenario_iso_count = 0

        for idx, row in settlements_proj.iterrows():
            geom = row.geometry.centroid if row.geometry.geom_type != "Point" else row.geometry
            s_id = str(row.get("id", idx))
            s_name = str(row.get("name", f"Settlement_{s_id}"))

            s_node = self._find_nearest_node(road_graph, geom)

            base_reachable = False
            base_dist_km: Optional[float] = None
            best_dest_id = None
            best_dest_name = None

            if s_node and dest_nodes:
                min_base_dist = float("inf")
                for d_id, d_name, d_node in dest_nodes:
                    try:
                        dist_m = nx.dijkstra_path_length(road_graph, s_node, d_node, weight="weight")
                        if dist_m < min_base_dist:
                            min_base_dist = dist_m
                            best_dest_id = d_id
                            best_dest_name = d_name
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass

                if min_base_dist < float("inf"):
                    base_reachable = True
                    base_dist_km = round(min_base_dist / 1000.0, 2)
                    baseline_conn_count += 1

            # Scenario Reachability
            scen_reachable: Optional[bool] = None
            scen_dist_km: Optional[float] = None

            if scenario_enabled and scenario_graph:
                scen_reachable = False
                if s_node and dest_nodes:
                    min_scen_dist = float("inf")
                    for d_id, d_name, d_node in dest_nodes:
                        try:
                            dist_m = nx.dijkstra_path_length(scenario_graph, s_node, d_node, weight="weight")
                            if dist_m < min_scen_dist:
                                min_scen_dist = dist_m
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            pass

                    if min_scen_dist < float("inf"):
                        scen_reachable = True
                        scen_dist_km = round(min_scen_dist / 1000.0, 2)
                        scenario_conn_count += 1
                    else:
                        scenario_iso_count += 1

            # Determine Status Label
            if not base_reachable:
                status_label = "UNREACHABLE_IN_BASE_NETWORK"
            elif scenario_enabled and not scen_reachable:
                status_label = "HYPOTHETICALLY_ISOLATED"
            else:
                status_label = "CONNECTED"

            record = SettlementAccessibilityRecord(
                settlement_id=s_id,
                settlement_name=s_name,
                nearest_destination_id=best_dest_id,
                nearest_destination_name=best_dest_name,
                baseline_reachable=base_reachable,
                baseline_distance_km=base_dist_km,
                scenario_reachable=scen_reachable,
                scenario_distance_km=scen_dist_km,
                status_label=status_label
            )
            records.append(record)

            orig_row = settlements_gdf.loc[idx].to_dict()
            orig_row["baseline_reachable"] = base_reachable
            orig_row["baseline_distance_km"] = base_dist_km
            orig_row["scenario_reachable"] = scen_reachable
            orig_row["scenario_distance_km"] = scen_dist_km
            orig_row["status_label"] = status_label
            output_rows.append(orig_row)

        res_gdf = gpd.GeoDataFrame(output_rows, crs=settlements_gdf.crs)

        summary = AccessibilitySummary(
            total_settlements=len(settlements_gdf),
            baseline_connected_count=baseline_conn_count,
            baseline_unconnected_count=len(settlements_gdf) - baseline_conn_count,
            scenario_enabled=scenario_enabled,
            scenario_connected_count=scenario_conn_count if scenario_enabled else None,
            scenario_isolated_count=scenario_iso_count if scenario_enabled else None,
            hypothetical_closed_roads_count=len(closed_road_ids) if closed_road_ids else 0
        )

        return res_gdf, summary

    def _empty_result(self, settlements_gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, AccessibilitySummary]:
        res_df = settlements_gdf.copy()
        res_df["baseline_reachable"] = False
        res_df["baseline_distance_km"] = None
        res_df["status_label"] = "UNREACHABLE_IN_BASE_NETWORK"

        summary = AccessibilitySummary(
            total_settlements=len(settlements_gdf),
            baseline_connected_count=0,
            baseline_unconnected_count=len(settlements_gdf)
        )
        return res_df, summary
