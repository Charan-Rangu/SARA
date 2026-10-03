"""Evidence-Grounded Situation Report Generator for SARA.

Generates structured, evidence-grounded Markdown situation reports from deterministic analysis outputs.
Ensures strict data compliance by omitting unavailable statistics and including mandatory system limitation statements.
"""

import os
from datetime import datetime, timezone
from typing import Optional, Any
from pydantic import BaseModel, Field

LIMITATION_STATEMENT = (
    "SARA produces preliminary geospatial assessments to support further investigation. "
    "Satellite backscatter changes and terrain-derived flow paths are not, by themselves, proof of flooding, "
    "infrastructure damage, road closure, or safe accessibility."
)


class ReportRequest(BaseModel):
    """Request parameters for situation report generation."""

    run_id: str
    aoi_name: str = "Target Area of Interest"
    event_date: str = "2024-08-15"
    satellite_search_summary: Optional[dict[str, Any]] = None
    exposure_summary: Optional[dict[str, Any]] = None
    accessibility_summary: Optional[dict[str, Any]] = None
    terrain_summary: Optional[dict[str, Any]] = None
    flow_tracing_summary: Optional[dict[str, Any]] = None
    analyst_notes: Optional[str] = None


class SituationReportGenerator:
    """Generator for creating structured Markdown situational reports."""

    def generate_report(self, request: ReportRequest, output_dir: Optional[str] = None) -> str:
        """Generate a structured Markdown situation report.

        Args:
            request: ReportRequest object containing run parameters and summary metrics.
            output_dir: Target directory to save the output markdown report.

        Returns:
            String containing complete Markdown report content.
        """
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        lines = [
            f"# SARA — Situation Report (Run ID: {request.run_id})",
            "",
            f"**Generated:** `{now_utc}`  ",
            f"**Target AOI:** `{request.aoi_name}`  ",
            f"**Target Flood/Event Date:** `{request.event_date}`  ",
            "",
            "> [!IMPORTANT]",
            f"> **System Limitation Statement:**  \n> {LIMITATION_STATEMENT}",
            "",
            "---",
            "",
            "## 1. Executive Summary & Operational Context",
            "",
            f"This preliminary situation report synthesizes satellite discovery metadata, vector infrastructure "
            f"exposure assessment, NetworkX road accessibility metrics, and Copernicus DEM elevation analysis for `{request.aoi_name}`.",
            ""
        ]

        if request.analyst_notes:
            lines.extend([
                "### Analyst Operational Notes:",
                f"> {request.analyst_notes}",
                ""
            ])

        # Section 2: Satellite Discovery
        lines.extend([
            "---",
            "",
            "## 2. Satellite Data Acquisition & Catalogue Discovery"
        ])
        sat = request.satellite_search_summary
        if sat:
            lines.extend([
                f"- **Total Satellite Scenes Retrieved:** `{sat.get('total_scenes_found', 0)}`",
                f"- **Pre-Event Window Scenes:** `{sat.get('pre_scenes_count', 0)}`",
                f"- **Post-Event Window Scenes:** `{sat.get('post_scenes_count', 0)}`",
                f"- **Compatible Scene Pairs Identified:** `{sat.get('suitable_pairs_count', 0)}`",
                f"- **Catalogue Source:** Copernicus Data Space Ecosystem OData API"
            ])
        else:
            lines.append("- *Satellite catalogue search results: NOT_AVAILABLE / NOT_RUN*")
        lines.append("")

        # Section 3: Preliminary SAR Change
        lines.extend([
            "---",
            "",
            "## 3. Preliminary SAR Backscatter Change Analysis"
        ])
        if sat and sat.get("top_pair"):
            top = sat["top_pair"]
            lines.extend([
                f"- **Selected Pre-Event Scene:** `{top.get('pre_title', 'UNKNOWN')}` ({top.get('pre_date', '')})",
                f"- **Selected Post-Event Scene:** `{top.get('post_title', 'UNKNOWN')}` ({top.get('post_date', '')})",
                r"- **Processing Algorithm:** Windowed Logarithmic Backscatter Difference ($\Delta \sigma^0_{\text{dB}} < -3.0\text{ dB}$)",
                "- **Status:** Preliminary backscatter reduction mask generated."
            ])
        else:
            lines.append("- *SAR backscatter change analysis status: NOT_AVAILABLE / PRELIMINARY*")
        lines.append("")

        # Section 4: Infrastructure Exposure
        lines.extend([
            "---",
            "",
            "## 4. Vector Infrastructure Exposure Assessment"
        ])
        exp = request.exposure_summary
        if exp:
            lines.extend([
                f"- **Total Roads Assessed:** `{exp.get('total_roads', 0)}` | **Potential Overlap:** `{exp.get('roads_potential_overlap_count', 0)}`",
                f"- **Total Buildings Assessed:** `{exp.get('total_buildings', 0)}` | **Potential Overlap:** `{exp.get('buildings_potential_overlap_count', 0)}`",
                f"- **Healthcare Facilities Assessed:** `{exp.get('total_facilities', 0)}` | **Potential Overlap:** `{exp.get('facilities_potential_overlap_count', 0)}`",
                "",
                "> [!CAUTION]",
                "> Exposure classification uses cautious labels (`potential_change_overlap`, `near_preliminary_change`). "
                "Features are **NOT** classified as confirmed destroyed or flooded."
            ])
        else:
            lines.append("- *Infrastructure exposure summary: NOT_AVAILABLE*")
        lines.append("")

        # Section 5: Road Accessibility
        lines.extend([
            "---",
            "",
            "## 5. Road Network Accessibility & Settlement Reachability"
        ])
        acc = request.accessibility_summary
        if acc:
            lines.extend([
                f"- **Total Populated Settlements:** `{acc.get('total_settlements', 0)}`",
                f"- **Baseline Network Connectivity (Mode A):** `{acc.get('baseline_connected_count', 0)}` connected | `{acc.get('baseline_unconnected_count', 0)}` unconnected",
                f"- **Scenario Connectivity (Mode B):** Enabled = `{acc.get('scenario_enabled', False)}`",
            ])
            if acc.get('scenario_enabled'):
                lines.extend([
                    f"  - **Hypothetically Isolated Settlements:** `{acc.get('scenario_isolated_count', 0)}`",
                    f"  - **Hypothetical Road Closures Applied:** `{acc.get('hypothetical_closed_roads_count', 0)}`"
                ])
        else:
            lines.append("- *Accessibility analysis summary: NOT_AVAILABLE*")
        lines.append("")

        # Section 6: Terrain & Flow Paths
        lines.extend([
            "---",
            "",
            "## 6. Copernicus DEM Terrain Analysis & Modeled Flow Paths"
        ])
        terr = request.terrain_summary
        flow = request.flow_tracing_summary
        if terr:
            lines.extend([
                f"- **Elevation Range:** `{terr.get('min_elevation_m', 0.0)}m` to `{terr.get('max_elevation_m', 0.0)}m` (Mean: `{terr.get('mean_elevation_m', 0.0)}m`)",
                f"- **Mean Slope:** `{terr.get('mean_slope_deg', 0.0)}°`",
                f"- **DEM Source:** Copernicus WorldDEM-30"
            ])
        if flow:
            lines.extend([
                f"- **Modeled D8 Terrain Flow Paths:** `{flow.get('traced_paths_count', 0)}` paths generated.",
                "- **Layer Label:** `MODELED_TERRAIN_FLOW_PATH` (Indicates topological gradient flow, not flood forecast)."
            ])
        if not terr and not flow:
            lines.append("- *Terrain elevation and flow tracing: NOT_AVAILABLE*")
        lines.append("")

        # Section 7: Attributions & Copyright
        lines.extend([
            "---",
            "",
            "## 7. Legal Copyright & Dataset Attributions",
            "",
            "- *\"Contains modified Copernicus Sentinel data 2026.\"*",
            "- *\"Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.\"*",
            "- *\"© OpenStreetMap contributors.\"*",
            ""
        ])

        report_content = "\n".join(lines)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            report_file_path = os.path.join(output_dir, f"sara_situation_report_{request.run_id}.md")
            with open(report_file_path, "w", encoding="utf-8") as f:
                f.write(report_content)

        return report_content
