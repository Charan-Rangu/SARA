"""Sentinel-1 SAR Preprocessing Interface, SNAP Execution Adapter, and Raster Validation.

Defines the engineering contract for converting raw Sentinel-1 GRD SAFE products into
Analysis-Ready Data (ARD) rasters via radiometric calibration, terrain correction,
speckle filtering, and spatial co-registration using ESA SNAP Graph Processing Tool (gpt.exe).

Includes:
1. Multi-path & environment-variable based SNAP gpt.exe discovery.
2. Operator verification and execution diagnostics.
3. Execution timeout, stdout/stderr capture, and exit code verification.
4. Windowed Rasterio-based output raster validation (CRS, transform, dimensions, pixel bounds).
5. Pre- and Post-raster grid alignment and affine transform validation.
"""

import os
import shutil
import subprocess
import time
from typing import Optional, Any
from pydantic import BaseModel, Field
import rasterio
from rasterio.windows import Window
import numpy as np

from geospatial.aoi import AOI
from geospatial.snap_graph import SNAPGraphBuilder, SNAPGraphConfig


class SARPreprocessingConfig(BaseModel):
    """Configuration options for Sentinel-1 GRD preprocessing pipeline."""

    output_dir: str = os.path.join("data", "processed")
    target_crs: str = "EPSG:4326"
    pixel_spacing_m: float = 10.0
    calibration_type: str = "SIGMA0"  # "SIGMA0" or "GAMMA0"
    output_units: str = "dB"           # "dB" or "linear"
    apply_speckle_filter: bool = True
    speckle_filter_type: str = "Refined Lee"
    apply_terrain_correction: bool = True
    dem_name: str = "Copernicus 30m Global DEM"
    apply_coregistration: bool = True
    timeout_seconds: int = 3600        # 60-minute timeout for heavy SAR processing
    max_memory_mb: int = 8192          # JVM max heap for SNAP gpt


class SARPreprocessingResult(BaseModel):
    """Output container for preprocessed analysis-ready SAR rasters."""

    status: str  # "SUCCESS", "SNAP_NOT_INSTALLED", "SNAP_VERIFICATION_FAILED", "MISSING_OPERATORS", "EXECUTION_FAILED", "TIMEOUT", "VALIDATION_FAILED"
    calibrated_pre_raster: Optional[str] = None
    calibrated_post_raster: Optional[str] = None
    is_coregistered: bool = False
    units: str = "dB"
    execution_time_seconds: Optional[float] = None
    processing_metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    snap_available: bool = False


def validate_processed_raster(
    raster_path: str,
    expected_crs: Optional[str] = "EPSG:4326"
) -> tuple[bool, dict[str, Any], list[str]]:
    """Validate a processed GeoTIFF raster using Rasterio without loading full array into RAM.

    Checks:
    - File exists and is a valid readable raster.
    - CRS matches expected target.
    - Dimensions are non-zero.
    - Affine transform has non-zero pixel dimensions.
    - Band count is at least 1.
    - Windowed inspection checks for finite, valid pixels (not all NaN or nodata).

    Returns:
        Tuple of (is_valid: bool, metadata: dict, issues: list[str]).
    """
    issues: list[str] = []
    metadata: dict[str, Any] = {}

    if not os.path.isfile(raster_path):
        return False, {}, [f"Raster file not found: {raster_path}"]

    if os.path.getsize(raster_path) == 0:
        return False, {}, [f"Raster file is empty (0 bytes): {raster_path}"]

    try:
        with rasterio.open(raster_path) as src:
            metadata = {
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "crs": str(src.crs),
                "transform": [float(v) for v in list(src.transform)[:6]],
                "nodata": src.nodata,
                "dtypes": [str(d) for d in src.dtypes]
            }

            if src.width <= 0 or src.height <= 0:
                issues.append(f"Invalid dimensions: ({src.width}x{src.height})")

            if expected_crs and src.crs:
                if str(src.crs).upper() != expected_crs.upper():
                    issues.append(f"CRS mismatch: expected '{expected_crs}', got '{src.crs}'")
            elif expected_crs and not src.crs:
                issues.append(f"Raster missing CRS (expected '{expected_crs}')")

            # Check affine transform validity (non-zero scale)
            t = src.transform
            if t[0] == 0 or t[4] == 0:
                issues.append(f"Invalid affine transform scale factors: xres={t[0]}, yres={t[4]}")

            # Sanity check file size: require at least 10 KB to ensure it's not a dummy or truncated header
            actual_bytes = os.path.getsize(raster_path)
            if actual_bytes < 10240:
                issues.append(
                    f"Raster file size ({actual_bytes:,} bytes) is suspiciously small (<10 KB)"
                )

            # Windowed pixel validity inspection (sample 3 windows: top-left, center, bottom-right)
            sample_windows = [
                Window(0, 0, min(128, src.width), min(128, src.height)),
                Window(max(0, src.width // 2 - 64), max(0, src.height // 2 - 64), min(128, src.width), min(128, src.height)),
                Window(max(0, src.width - 128), max(0, src.height - 128), min(128, src.width), min(128, src.height))
            ]

            found_finite_pixels = False
            for win in sample_windows:
                if win.width <= 0 or win.height <= 0:
                    continue
                block = src.read(1, window=win)
                finite_mask = np.isfinite(block)
                if src.nodata is not None:
                    finite_mask &= (block != src.nodata)
                if np.any(finite_mask):
                    found_finite_pixels = True
                    break

            if not found_finite_pixels:
                issues.append("Sampled raster blocks contain only NaN or NoData values")

    except Exception as e:
        return False, metadata, [f"Rasterio read error: {e}"]

    is_valid = len(issues) == 0
    return is_valid, metadata, issues


def validate_raster_grid_compatibility(
    pre_raster_path: str,
    post_raster_path: str,
    atol_transform: float = 1e-5
) -> tuple[bool, list[str]]:
    """Validate that pre and post rasters share an identical spatial grid for change detection.

    Validates:
    - Matching CRS.
    - Matching raster width and height.
    - Matching affine transform (origin and pixel size).
    """
    issues: list[str] = []

    if not os.path.exists(pre_raster_path):
        return False, [f"Pre-event raster not found: {pre_raster_path}"]
    if not os.path.exists(post_raster_path):
        return False, [f"Post-event raster not found: {post_raster_path}"]

    try:
        with rasterio.open(pre_raster_path) as pre_src, rasterio.open(post_raster_path) as post_src:
            if pre_src.crs != post_src.crs:
                issues.append(f"Grid CRS mismatch: pre='{pre_src.crs}' vs post='{post_src.crs}'")

            if (pre_src.width != post_src.width) or (pre_src.height != post_src.height):
                issues.append(
                    f"Grid dimension mismatch: pre=({pre_src.width}x{pre_src.height}) vs "
                    f"post=({post_src.width}x{post_src.height})"
                )

            if not np.allclose(pre_src.transform, post_src.transform, atol=atol_transform):
                issues.append(
                    f"Grid affine transform mismatch: pre={pre_src.transform} vs post={post_src.transform}"
                )
    except Exception as e:
        return False, [f"Error verifying grid compatibility: {e}"]

    return len(issues) == 0, issues


class SNAPProcessorAdapter:
    """Adapter for executing ESA SNAP Graph Processing Tool (gpt.exe) workflows."""

    REQUIRED_OPERATORS = [
        "Read",
        "Apply-Orbit-File",
        "ThermalNoiseRemoval",
        "Remove-GRD-Border-Noise",
        "Calibration",
        "Speckle-Filter",
        "Terrain-Correction",
        "Write"
    ]

    def __init__(self, gpt_path: Optional[str] = None):
        self.gpt_path = gpt_path or self._discover_snap_gpt()
        self.graph_builder = SNAPGraphBuilder()

    def _discover_snap_gpt(self) -> Optional[str]:
        """Discover ESA SNAP gpt.exe through environment variables, PATH, and standard Windows paths."""
        # 1. Check explicit environment variables
        env_vars = ["SNAP_GPT_PATH", "ESA_SNAP_HOME", "SNAP_HOME"]
        for env in env_vars:
            val = os.getenv(env)
            if val:
                if os.path.isfile(val) and val.lower().endswith("gpt.exe"):
                    return val
                # If directory provided, check bin/gpt.exe
                candidate = os.path.join(val, "bin", "gpt.exe")
                if os.path.isfile(candidate):
                    return candidate
                candidate_direct = os.path.join(val, "gpt.exe")
                if os.path.isfile(candidate_direct):
                    return candidate_direct

        # 2. Check PATH
        path_bin = shutil.which("gpt")
        if path_bin and os.path.isfile(path_bin):
            return path_bin

        # 3. Check standard Windows installation directories
        username = os.getenv("USERNAME", "")
        possible_paths = [
            r"C:\Program Files\snap\bin\gpt.exe",
            r"C:\Program Files\esa-snap\bin\gpt.exe",
            r"C:\Program Files\SNAP 14.0.0\bin\gpt.exe",
            r"C:\Program Files\SNAP\bin\gpt.exe",
            r"C:\Program Files (x86)\snap\bin\gpt.exe",
            r"C:\Program Files (x86)\esa-snap\bin\gpt.exe",
            rf"C:\Users\{username}\AppData\Local\Programs\snap\bin\gpt.exe",
            rf"C:\Users\{username}\AppData\Local\Programs\esa-snap\bin\gpt.exe",
            rf"C:\Users\{username}\AppData\Local\snap\bin\gpt.exe",
            r"C:\snap\bin\gpt.exe"
        ]
        for p in possible_paths:
            if os.path.isfile(p):
                return p

        return None

    def is_snap_available(self) -> bool:
        """Check if ESA SNAP gpt executable is available on system PATH or configured path."""
        if not self.gpt_path:
            return False
        try:
            res = subprocess.run([self.gpt_path, "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            return res.returncode == 0
        except Exception:
            return False

    def get_discovery_status(self) -> dict[str, Any]:
        """Return comprehensive discovery status: status code, path, version, and operator report."""
        if not self.gpt_path:
            return {
                "status": "SNAP_NOT_INSTALLED",
                "gpt_path": None,
                "version": None,
                "operators": {op: False for op in self.REQUIRED_OPERATORS},
                "instructions": self.get_installation_instructions()
            }

        if not os.path.isfile(self.gpt_path):
            return {
                "status": "SNAP_NOT_INSTALLED",
                "gpt_path": self.gpt_path,
                "version": None,
                "operators": {op: False for op in self.REQUIRED_OPERATORS},
                "instructions": self.get_installation_instructions()
            }

        # Executable exists, run verification diagnostic
        try:
            res = subprocess.run([self.gpt_path, "-h"], capture_output=True, text=True, timeout=15)
            if res.returncode != 0:
                return {
                    "status": "SNAP_VERIFICATION_FAILED",
                    "gpt_path": self.gpt_path,
                    "version": None,
                    "diagnostics": f"Executable failed with returncode {res.returncode}: {res.stderr.strip()}",
                    "operators": {op: False for op in self.REQUIRED_OPERATORS},
                    "instructions": self.get_installation_instructions()
                }
        except subprocess.TimeoutExpired:
            # Proceed with version check if executable exists
            pass
        except Exception as e:
            return {
                "status": "SNAP_VERIFICATION_FAILED",
                "gpt_path": self.gpt_path,
                "version": None,
                "diagnostics": f"Failed to execute diagnostic check: {e}",
                "operators": {op: False for op in self.REQUIRED_OPERATORS},
                "instructions": self.get_installation_instructions()
            }

        version_str = self.get_snap_version()
        op_status = self.verify_operators()
        all_ops = all(op_status.values())

        return {
            "status": "SNAP_AVAILABLE" if all_ops else "MISSING_OPERATORS",
            "gpt_path": self.gpt_path,
            "version": version_str or "ESA SNAP 14.0.0",
            "operators": op_status,
            "instructions": None if all_ops else "Some required operators are missing."
        }

    def get_snap_version(self) -> Optional[str]:
        """Detect and return installed ESA SNAP version string if available."""
        if not self.is_snap_available():
            return None

        # Check for VERSION.txt in installation root directory
        if self.gpt_path:
            version_txt = os.path.join(os.path.dirname(os.path.dirname(self.gpt_path)), "VERSION.txt")
            if os.path.isfile(version_txt):
                try:
                    with open(version_txt, "r", encoding="utf-8") as f:
                        ver = f.read().strip()
                        if ver:
                            return f"ESA SNAP {ver}"
                except Exception:
                    pass

        try:
            res = subprocess.run([self.gpt_path, "-h"], capture_output=True, text=True, timeout=5)
            for line in res.stdout.splitlines():
                if "version" in line.lower() or "snap" in line.lower() or "gpt" in line.lower():
                    return line.strip()
            return "ESA SNAP GPT"
        except Exception:
            return "ESA SNAP GPT (Installed)"

    def verify_operators(self) -> dict[str, bool]:
        """Verify availability of essential Sentinel-1 GRD preprocessing operators in SNAP."""
        if not self.is_snap_available():
            return {op: False for op in self.REQUIRED_OPERATORS}

        results: dict[str, bool] = {}
        for op in self.REQUIRED_OPERATORS:
            try:
                res = subprocess.run([self.gpt_path, "-h", op], capture_output=True, timeout=3)
                results[op] = (res.returncode == 0)
            except Exception:
                results[op] = False
        return results

    def get_installation_instructions(self) -> str:
        """Return clear instructions for installing ESA SNAP for Sentinel-1 ARD preprocessing."""
        return (
            "ESA SNAP (Sentinel Application Platform) with gpt.exe is required for full GRD ARD preprocessing.\n"
            "Installation Steps:\n"
            "1. Download ESA SNAP (v9.0 / v10.0 / v14.0) for Windows from: https://step.esa.int/main/download/snap-download/\n"
            "2. Ensure the 'Sentinel-1 Toolbox' is selected during installation.\n"
            "3. Add the SNAP bin directory (e.g. C:\\Program Files\\snap\\bin) to your system PATH or set SNAP_GPT_PATH.\n"
            "4. Verify in PowerShell by running: gpt -h"
        )

    def generate_processing_graph(
        self,
        input_safe_path: str,
        output_raster_path: str,
        config: Optional[SARPreprocessingConfig] = None
    ) -> str:
        """Generate a reproducible SNAP XML graph for Sentinel-1 GRD ARD processing."""
        snap_config = SNAPGraphConfig(
            polarizations=["VV", "VH"],
            calibration_type=config.calibration_type if config else "SIGMA0",
            output_in_db=(config.output_units == "dB") if config else True,
            speckle_filter_type=config.speckle_filter_type if config else "Refined Lee",
            dem_name=config.dem_name if config else "Copernicus 30m Global DEM",
            pixel_spacing_m=config.pixel_spacing_m if config else 10.0,
            crs=config.target_crs if config else "EPSG:4326"
        )
        return self.graph_builder.build_grd_ard_graph_xml(input_safe_path, output_raster_path, snap_config)

    def execute_snap_graph(
        self,
        xml_graph_path: str,
        timeout_seconds: int = 1800,
        max_memory_mb: int = 8192
    ) -> tuple[bool, str, float]:
        """Execute a generated SNAP XML graph using gpt.exe.

        Captures stdout, stderr, execution time, and enforces timeout.

        Returns:
            Tuple of (success: bool, log_message: str, elapsed_time_seconds: float).
        """
        if not self.is_snap_available():
            return False, "SNAP gpt.exe is not available for execution.", 0.0

        cmd = [
            self.gpt_path,
            xml_graph_path,
            f"-J-Xmx{max_memory_mb}m",
            "-q", "8"  # Quiet mode with parallel threads
        ]

        start_time = time.time()
        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds
            )
            elapsed = time.time() - start_time
            if res.returncode == 0:
                return True, f"SNAP execution succeeded in {elapsed:.1f}s.", elapsed
            else:
                return False, f"SNAP execution failed (code {res.returncode}): {res.stderr.strip()}", elapsed
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            return False, f"SNAP execution timed out after {timeout_seconds} seconds.", elapsed
        except Exception as e:
            elapsed = time.time() - start_time
            return False, f"SNAP execution error: {e}", elapsed

    def preprocess_scene_pair(
        self,
        pre_safe_path: str,
        post_safe_path: str,
        aoi: AOI,
        config: Optional[SARPreprocessingConfig] = None
    ) -> SARPreprocessingResult:
        """Execute SNAP GPT graph processing pipeline on a pre/post SAFE scene pair.

        Args:
            pre_safe_path: Local path to pre-event Sentinel-1 SAFE directory or manifest.safe.
            post_safe_path: Local path to post-event Sentinel-1 SAFE directory or manifest.safe.
            aoi: Target spatial Area of Interest.
            config: SARPreprocessingConfig parameters.

        Returns:
            SARPreprocessingResult object with processed ARD raster paths or clear status warnings.
        """
        cfg = config or SARPreprocessingConfig()
        discovery = self.get_discovery_status()

        if discovery["status"] != "SNAP_AVAILABLE":
            return SARPreprocessingResult(
                status=discovery["status"],
                snap_available=False,
                warnings=[
                    f"ESA SNAP gpt.exe is not installed or not found on PATH. SNAP status: {discovery['status']}. "
                    f"Full GRD calibration, terrain correction, and co-registration require ESA SNAP.\n"
                    f"{self.get_installation_instructions()}"
                ],
                processing_metadata={
                    "discovery": discovery,
                    "required_steps": [
                        "1. Apply Orbit File (Precise Orbits)",
                        "2. Thermal Noise Removal",
                        "3. Radiometric Calibration (Sigma0)",
                        "4. Speckle Filter (Refined Lee 7x7)",
                        "5. Range-Doppler Terrain Correction (Copernicus DEM 30m)",
                        "6. Co-registration / Spatial Alignment"
                    ]
                }
            )

        # Check input product paths
        if not os.path.exists(pre_safe_path):
            return SARPreprocessingResult(
                status="INPUT_NOT_FOUND",
                snap_available=True,
                warnings=[f"Pre-event SAFE product not found at: {pre_safe_path}"]
            )
        if not os.path.exists(post_safe_path):
            return SARPreprocessingResult(
                status="INPUT_NOT_FOUND",
                snap_available=True,
                warnings=[f"Post-event SAFE product not found at: {post_safe_path}"]
            )

        os.makedirs(cfg.output_dir, exist_ok=True)
        pre_out = os.path.join(cfg.output_dir, "pre_calibrated_sigma0.tif")
        post_out = os.path.join(cfg.output_dir, "post_calibrated_sigma0.tif")

        pre_xml = os.path.join(cfg.output_dir, "pre_process_graph.xml")
        post_xml = os.path.join(cfg.output_dir, "post_process_graph.xml")

        self.graph_builder.save_graph_file(
            self.generate_processing_graph(pre_safe_path, pre_out, cfg),
            pre_xml
        )
        self.graph_builder.save_graph_file(
            self.generate_processing_graph(post_safe_path, post_out, cfg),
            post_xml
        )

        # 1. Execute Pre-event graph (skip if valid raster already exists)
        val_pre_exists, _, _ = validate_processed_raster(pre_out, expected_crs=cfg.target_crs)
        if val_pre_exists:
            print(f"  [INFO] Pre-event raster already processed and verified at: {pre_out}")
            ok_pre, log_pre, time_pre = True, "Pre-event raster already exists", 0.0
        else:
            ok_pre, log_pre, time_pre = self.execute_snap_graph(pre_xml, timeout_seconds=cfg.timeout_seconds, max_memory_mb=cfg.max_memory_mb)

        if not ok_pre:
            return SARPreprocessingResult(
                status="EXECUTION_FAILED",
                snap_available=True,
                execution_time_seconds=time_pre,
                warnings=[f"Pre-event scene processing failed: {log_pre}"]
            )

        # 2. Execute Post-event graph (skip if valid raster already exists)
        val_post_exists, _, _ = validate_processed_raster(post_out, expected_crs=cfg.target_crs)
        if val_post_exists:
            print(f"  [INFO] Post-event raster already processed and verified at: {post_out}")
            ok_post, log_post, time_post = True, "Post-event raster already exists", 0.0
        else:
            ok_post, log_post, time_post = self.execute_snap_graph(post_xml, timeout_seconds=cfg.timeout_seconds, max_memory_mb=cfg.max_memory_mb)

        if not ok_post:
            return SARPreprocessingResult(
                status="EXECUTION_FAILED",
                snap_available=True,
                execution_time_seconds=time_pre + time_post,
                warnings=[f"Post-event scene processing failed: {log_post}"]
            )

        # 3. Post-execution Rasterio validation
        val_pre, meta_pre, issues_pre = validate_processed_raster(pre_out, expected_crs=cfg.target_crs)
        val_post, meta_post, issues_post = validate_processed_raster(post_out, expected_crs=cfg.target_crs)

        if not val_pre or not val_post:
            return SARPreprocessingResult(
                status="VALIDATION_FAILED",
                snap_available=True,
                execution_time_seconds=time_pre + time_post,
                warnings=issues_pre + issues_post
            )

        # 4. Grid compatibility check
        compat_ok, grid_issues = validate_raster_grid_compatibility(pre_out, post_out)
        if not compat_ok:
            return SARPreprocessingResult(
                status="VALIDATION_FAILED",
                snap_available=True,
                calibrated_pre_raster=pre_out,
                calibrated_post_raster=post_out,
                is_coregistered=False,
                execution_time_seconds=time_pre + time_post,
                warnings=grid_issues
            )

        return SARPreprocessingResult(
            status="SUCCESS",
            snap_available=True,
            calibrated_pre_raster=pre_out,
            calibrated_post_raster=post_out,
            is_coregistered=True,
            units=cfg.output_units,
            execution_time_seconds=round(time_pre + time_post, 1),
            processing_metadata={
                "config": cfg.model_dump(),
                "pre_raster_meta": meta_pre,
                "post_raster_meta": meta_post,
                "snap_version": discovery.get("version"),
                "gpt_path": self.gpt_path
            }
        )
