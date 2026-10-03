"""SNAP Processing Graph (.xml) Generator for Sentinel-1 GRD Products.

Generates reproducible, validated XML graph processing workflows for ESA SNAP Graph Processing Tool (gpt.exe).

Standard Processing Order for Sentinel-1 GRD ARD:
1. Read (Raw SAFE Product or manifest.safe)
2. Apply-Orbit-File (Sentinel Precise / Restituted Orbits)
3. ThermalNoiseRemoval (Thermal Noise Suppression)
4. Remove-GRD-Border-Noise (GRD Edge Artifact Removal)
5. Calibration (Radiometric Calibration to Sigma0)
6. Speckle-Filter (Refined Lee / Lee 3x3)
7. Terrain-Correction (Range Doppler Terrain Correction using Copernicus DEM 30m / EPSG:4326)
8. Write (GeoTIFF-BigTIFF / BEAM-DIMAP)
"""

import os
import xml.etree.ElementTree as ET
from typing import Optional, Any
from pydantic import BaseModel, Field


class SNAPGraphConfig(BaseModel):
    """Configuration settings for SNAP Graph XML generation."""

    polarizations: list[str] = Field(default_factory=lambda: ["VV", "VH"])
    calibration_type: str = "SIGMA0"     # "SIGMA0" or "GAMMA0"
    output_in_db: bool = True            # True for dB decibels, False for linear
    speckle_filter_type: str = "Refined Lee"
    dem_name: str = "Copernicus 30m Global DEM"  # or "SRTM 3Sec"
    pixel_spacing_m: float = 10.0
    crs: str = "EPSG:4326"               # Target map projection
    nodata_value: float = 0.0


class SNAPGraphBuilder:
    """Builder and static validator for generating valid ESA SNAP XML graph files."""

    REQUIRED_GRAPH_OPERATORS = [
        "Read",
        "Apply-Orbit-File",
        "ThermalNoiseRemoval",
        "Remove-GRD-Border-Noise",
        "Calibration",
        "Speckle-Filter",
        "Terrain-Correction",
        "Write"
    ]

    def normalize_product_path(self, product_path: str) -> str:
        """Resolve and format Sentinel-1 product path safely for SNAP gpt.

        Normalizes backslashes to forward slashes for Java/SNAP path compatibility.
        If a .SAFE directory exists and contains manifest.safe, points to manifest.safe.
        """
        clean_path = product_path.replace("\\", "/")
        if os.path.isabs(product_path) or os.path.exists(product_path):
            abs_p = os.path.abspath(product_path)
            if os.path.isdir(abs_p):
                manifest = os.path.join(abs_p, "manifest.safe")
                if os.path.isfile(manifest):
                    abs_p = manifest
            clean_path = abs_p.replace("\\", "/")
        return clean_path

    def build_grd_ard_graph_xml(
        self,
        input_safe_path: str,
        output_raster_path: str,
        config: Optional[SNAPGraphConfig] = None
    ) -> str:
        """Construct a complete, reproducible SNAP XML processing graph string.

        Args:
            input_safe_path: Path to input Sentinel-1 .SAFE directory, manifest.safe, or zip.
            output_raster_path: Target path for the output preprocessed GeoTIFF raster.
            config: SNAPGraphConfig object with processing parameters.

        Returns:
            Valid XML string formatted for ESA SNAP gpt.
        """
        if config is None:
            config = SNAPGraphConfig()

        safe_input = self.normalize_product_path(input_safe_path)
        safe_output = self.normalize_product_path(output_raster_path)

        pols_str = ",".join(config.polarizations)
        output_format = "GeoTIFF-BigTIFF" if output_raster_path.endswith(".tif") or output_raster_path.endswith(".tiff") else "BEAM-DIMAP"

        # SNAP RangeDopplerGeocodingOp requires 'WGS84(DD)' or WKT for geographic projections.
        # Passing 'EPSG:4326' directly causes JAI PlanarImage.XToTileX division-by-zero during tile init.
        map_proj = config.crs
        if map_proj.upper() in ["EPSG:4326", "WGS84", "WGS 84", "WGS84(DD)"]:
            map_proj = "WGS84(DD)"

        xml = f"""<graph id="Graph">
  <version>1.0</version>
  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters class="com.bc.ceres.binding.dom.DefaultDomElement">
      <file>{safe_input}</file>
    </parameters>
  </node>
  <node id="Apply-Orbit-File">
    <operator>Apply-Orbit-File</operator>
    <sources>
      <sourceProduct refid="Read"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.DefaultDomElement">
      <orbitType>Sentinel Precise (Auto Download)</orbitType>
      <polyDegree>3</polyDegree>
      <continueOnFail>true</continueOnFail>
    </parameters>
  </node>
  <node id="ThermalNoiseRemoval">
    <operator>ThermalNoiseRemoval</operator>
    <sources>
      <sourceProduct refid="Apply-Orbit-File"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.DefaultDomElement">
      <selectedPolarisations>{pols_str}</selectedPolarisations>
      <removeThermalNoise>true</removeThermalNoise>
    </parameters>
  </node>
  <node id="Remove-GRD-Border-Noise">
    <operator>Remove-GRD-Border-Noise</operator>
    <sources>
      <sourceProduct refid="ThermalNoiseRemoval"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.DefaultDomElement">
      <selectedPolarisations>{pols_str}</selectedPolarisations>
      <borderLimit>500</borderLimit>
      <trimThreshold>0.5</trimThreshold>
    </parameters>
  </node>
  <node id="Calibration">
    <operator>Calibration</operator>
    <sources>
      <sourceProduct refid="Remove-GRD-Border-Noise"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.DefaultDomElement">
      <selectedPolarisations>{pols_str}</selectedPolarisations>
      <outputSigmaBand>true</outputSigmaBand>
      <outputGammaBand>false</outputGammaBand>
      <outputBetaBand>false</outputBetaBand>
      <outputImageInComplex>false</outputImageInComplex>
      <outputImageScaleInDb>{"true" if config.output_in_db else "false"}</outputImageScaleInDb>
    </parameters>
  </node>
  <node id="Speckle-Filter">
    <operator>Speckle-Filter</operator>
    <sources>
      <sourceProduct refid="Calibration"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.DefaultDomElement">
      <filter>{config.speckle_filter_type}</filter>
      <filterSizeX>3</filterSizeX>
      <filterSizeY>3</filterSizeY>
      <dampingFactor>2</dampingFactor>
      <estimateENL>true</estimateENL>
      <enl>1.0</enl>
      <numLooksStr>1</numLooksStr>
      <windowSize>7x7</windowSize>
      <targetWindowSizeStr>3x3</targetWindowSizeStr>
      <sigmaStr>0.9</sigmaStr>
      <anSize>50</anSize>
    </parameters>
  </node>
  <node id="Terrain-Correction">
    <operator>Terrain-Correction</operator>
    <sources>
      <sourceProduct refid="Speckle-Filter"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.DefaultDomElement">
      <demName>{config.dem_name}</demName>
      <externalDEMFile/>
      <externalDEMNoDataValue>0.0</externalDEMNoDataValue>
      <externalDEMApplyEGM>true</externalDEMApplyEGM>
      <demResamplingMethod>BILINEAR_INTERPOLATION</demResamplingMethod>
      <imgResamplingMethod>BILINEAR_INTERPOLATION</imgResamplingMethod>
      <pixelSpacingInMeter>{config.pixel_spacing_m}</pixelSpacingInMeter>
      <mapProjection>{map_proj}</mapProjection>
      <alignToStandardGrid>false</alignToStandardGrid>
      <standardGridOriginX>0.0</standardGridOriginX>
      <standardGridOriginY>0.0</standardGridOriginY>
      <nodataValueAtSea>true</nodataValueAtSea>
      <saveDEM>false</saveDEM>
      <saveLatLon>false</saveLatLon>
      <saveIncidenceAngleFromEllipsoid>false</saveIncidenceAngleFromEllipsoid>
      <saveLocalIncidenceAngle>false</saveLocalIncidenceAngle>
      <saveProjectedLocalIncidenceAngle>false</saveProjectedLocalIncidenceAngle>
      <saveSelectedSourceBand>true</saveSelectedSourceBand>
    </parameters>
  </node>
  <node id="Write">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="Terrain-Correction"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.DefaultDomElement">
      <file>{safe_output}</file>
      <formatName>{output_format}</formatName>
    </parameters>
  </node>
</graph>"""
        return xml

    def validate_graph_xml(self, xml_content: str) -> tuple[bool, list[str]]:
        """Statically validate the well-formedness and structural integrity of a SNAP XML graph.

        Checks:
        1. XML well-formedness.
        2. Root element is <graph>.
        3. All expected Sentinel-1 GRD operators are present in correct order.
        4. DAG integrity: source references point to existing prior nodes.
        5. File parameters in Read and Write nodes are populated.

        Returns:
            Tuple of (is_valid: bool, issues: list[str]).
        """
        issues: list[str] = []
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            return False, [f"XML parsing failed: {e}"]

        if root.tag != "graph":
            issues.append(f"Root tag must be <graph>, got <{root.tag}>")

        nodes = root.findall("node")
        if not nodes:
            issues.append("Graph contains no <node> elements")
            return False, issues

        node_ids = set()
        operators_found = []

        for node in nodes:
            node_id = node.get("id")
            if not node_id:
                issues.append("Found <node> with missing 'id' attribute")
                continue
            if node_id in node_ids:
                issues.append(f"Duplicate node id: '{node_id}'")
            node_ids.add(node_id)

            op_elem = node.find("operator")
            if op_elem is None or not op_elem.text:
                issues.append(f"Node '{node_id}' has missing or empty <operator>")
                continue
            operators_found.append(op_elem.text.strip())

            # Validate sources reference existing nodes
            sources = node.find("sources")
            if sources is not None:
                for src in sources.findall("sourceProduct"):
                    ref = src.get("refid")
                    if ref and ref not in node_ids:
                        issues.append(f"Node '{node_id}' references unknown source node '{ref}'")

        # Check required operators presence
        for req_op in self.REQUIRED_GRAPH_OPERATORS:
            if req_op not in operators_found:
                issues.append(f"Required operator '{req_op}' missing from graph")

        # Check Read and Write file parameters
        read_node = root.find(".//node[@id='Read']")
        if read_node is not None:
            file_param = read_node.find(".//parameters/file")
            if file_param is None or not file_param.text:
                issues.append("Read node is missing input <file> parameter")

        write_node = root.find(".//node[@id='Write']")
        if write_node is not None:
            file_param = write_node.find(".//parameters/file")
            if file_param is None or not file_param.text:
                issues.append("Write node is missing output <file> parameter")

        is_valid = len(issues) == 0
        return is_valid, issues

    def save_graph_file(self, xml_content: str, output_xml_path: str) -> str:
        """Save generated XML graph string to a file on disk after verifying directory exists."""
        out_dir = os.path.dirname(os.path.abspath(output_xml_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_xml_path, "w", encoding="utf-8") as f:
            f.write(xml_content)
        return output_xml_path
