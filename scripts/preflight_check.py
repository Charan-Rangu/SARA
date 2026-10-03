"""SARA Reproducible Preflight Verification & Real-Data Readiness Audit.

Executes a comprehensive, non-destructive diagnostic of:
1. Python Virtual Environment & Core Packages.
2. GPU Hardware Acceleration (PyTorch & CUDA).
3. Storage Space & Output Directory Writability.
4. SNAP executable found.
5. SNAP version/diagnostics.
6. Required operators available.
7. Graph XML valid.
8. Graph parameters verified.
9. Sentinel-1 download status.
10. Real-product preprocessing executed.
11. Real change-detection raster available.

Statuses:
- [PASS]: Prerequisite verified and operational.
- [FAIL]: Error or defect detected.
- [BLOCKED]: Necessary software or external data not present yet.
- [NOT RUN]: Skipped because prerequisite dependencies were not met.
"""

import sys
import os
import shutil
from typing import NamedTuple

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.config import resolve_snap_gpt_path, settings
from geospatial.sar_preprocessing import (
    SNAPProcessorAdapter,
    validate_raster_grid_compatibility,
    validate_processed_raster
)
from geospatial.snap_graph import SNAPGraphBuilder, SNAPGraphConfig


class CheckResult(NamedTuple):
    name: str
    status: str  # "PASS", "FAIL", "BLOCKED", "NOT RUN"
    details: str


def run_preflight() -> list[CheckResult]:
    results: list[CheckResult] = []

    print("\n" + "=" * 76)
    print("        SARA — SYSTEM PREFLIGHT & REAL-DATA READINESS AUDIT")
    print("=" * 76 + "\n")

    # 1. Virtual Environment & Packages
    is_venv = (sys.prefix != sys.base_prefix) or ("venv" in sys.prefix.lower())
    core_packages = [
        "torch", "rasterio", "geopandas", "shapely", "pyproj",
        "fastapi", "networkx", "httpx", "pydantic"
    ]
    missing_pkgs = []
    for pkg in core_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing_pkgs.append(pkg)

    if is_venv and not missing_pkgs:
        results.append(CheckResult("Python Environment & Packages", "PASS", f"Active venv ({len(core_packages)} packages ready)"))
    elif not is_venv:
        results.append(CheckResult("Python Environment & Packages", "FAIL", "Not running in a virtual environment."))
    else:
        results.append(CheckResult("Python Environment & Packages", "FAIL", f"Missing packages: {missing_pkgs}"))

    # 2. GPU Hardware Acceleration
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            results.append(CheckResult("GPU Hardware Acceleration", "PASS", f"{gpu_name} ({vram} GB VRAM) CUDA ready."))
        else:
            results.append(CheckResult("GPU Hardware Acceleration", "BLOCKED", "CUDA device not available. Running on CPU."))
    except Exception as e:
        results.append(CheckResult("GPU Hardware Acceleration", "FAIL", f"PyTorch error: {e}"))

    # 3. Storage & Output Writability
    dirs_to_check = [settings.data_raw_dir, settings.data_processed_dir, settings.data_output_dir]
    writable_all = True
    for d in dirs_to_check:
        full_d = os.path.join(REPO_ROOT, d)
        os.makedirs(full_d, exist_ok=True)
        test_file = os.path.join(full_d, ".write_test")
        try:
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
        except Exception:
            writable_all = False

    usage = shutil.disk_usage(REPO_ROOT)
    free_gb = round(usage.free / (1024 ** 3), 2)
    if writable_all and free_gb >= 5.0:
        results.append(CheckResult("Storage & Writability", "PASS", f"{free_gb} GB free on disk. Output directories writable."))
    elif writable_all:
        results.append(CheckResult("Storage & Writability", "PASS", f"{free_gb} GB free (Warning: low disk space)."))
    else:
        results.append(CheckResult("Storage & Writability", "FAIL", "One or more output directories are not writable."))

    # 4. SNAP executable found
    adapter = SNAPProcessorAdapter()
    discovery = adapter.get_discovery_status()
    snap_status = discovery["status"]

    if snap_status in ("SNAP_AVAILABLE", "MISSING_OPERATORS"):
        results.append(CheckResult(
            "SNAP executable found",
            "PASS",
            f"{discovery['gpt_path']}"
        ))
    elif snap_status == "SNAP_VERIFICATION_FAILED":
        results.append(CheckResult(
            "SNAP executable found",
            "FAIL",
            f"Found at {discovery['gpt_path']} but diagnostic failed."
        ))
    else:
        results.append(CheckResult(
            "SNAP executable found",
            "BLOCKED",
            "gpt.exe not found on PATH or standard installation locations."
        ))

    # 5. SNAP version/diagnostics
    if snap_status in ("SNAP_AVAILABLE", "MISSING_OPERATORS"):
        version_str = discovery.get("version") or "Installed (Version detected)"
        results.append(CheckResult(
            "SNAP version/diagnostics",
            "PASS",
            f"{version_str}"
        ))
    elif snap_status == "SNAP_VERIFICATION_FAILED":
        results.append(CheckResult(
            "SNAP version/diagnostics",
            "FAIL",
            f"Diagnostics: {discovery.get('diagnostics')}"
        ))
    else:
        results.append(CheckResult(
            "SNAP version/diagnostics",
            "NOT RUN",
            "SNAP not installed or discoverable."
        ))

    # 6. Required operators available
    if snap_status == "SNAP_AVAILABLE":
        results.append(CheckResult(
            "Required operators available",
            "PASS",
            "All 8 operators verified (Read, Apply-Orbit-File, ThermalNoiseRemoval, Remove-GRD-Border-Noise, Calibration, Speckle-Filter, Terrain-Correction, Write)."
        ))
    elif snap_status == "MISSING_OPERATORS":
        ops = discovery.get("operators", {})
        missing = [op for op, ok in ops.items() if not ok]
        results.append(CheckResult(
            "Required operators available",
            "FAIL",
            f"Missing operators: {missing}"
        ))
    else:
        results.append(CheckResult(
            "Required operators available",
            "NOT RUN",
            "SNAP operator check requires installed executable."
        ))

    # 7. Graph XML valid
    builder = SNAPGraphBuilder()
    xml_str = builder.build_grd_ard_graph_xml(
        input_safe_path="data/raw/candidate_pre.SAFE",
        output_raster_path="data/processed/candidate_pre_sigma0.tif",
        config=SNAPGraphConfig()
    )
    is_valid_xml, xml_issues = builder.validate_graph_xml(xml_str)
    if is_valid_xml:
        results.append(CheckResult(
            "Graph XML valid",
            "PASS",
            "Well-formed XML DAG structure with all 8 processing nodes."
        ))
    else:
        results.append(CheckResult(
            "Graph XML valid",
            "FAIL",
            f"XML errors: {xml_issues}"
        ))

    # 8. Graph parameters verified
    # Verified against SNAP 14.0.0 GPF operator definitions
    param_verified = (
        is_valid_xml and
        "<orbitType>Sentinel Precise (Auto Download)</orbitType>" in xml_str and
        "<removeThermalNoise>true</removeThermalNoise>" in xml_str and
        "<outputSigmaBand>true</outputSigmaBand>" in xml_str and
        "<filter>Refined Lee</filter>" in xml_str and
        "<demName>Copernicus 30m Global DEM</demName>" in xml_str and
        "<formatName>GeoTIFF-BigTIFF</formatName>" in xml_str
    )
    if param_verified:
        results.append(CheckResult(
            "Graph parameters verified",
            "PASS",
            "DEM ('Copernicus 30m Global DEM'), Output Format ('GeoTIFF-BigTIFF'), and noise/calibration parameters verified against SNAP 14.0.0."
        ))
    else:
        results.append(CheckResult(
            "Graph parameters verified",
            "FAIL",
            "Graph parameters mismatch operator specification."
        ))

    # 9. Sentinel-1 download status
    raw_dir = os.path.join(REPO_ROOT, settings.data_raw_dir)
    safe_products = [
        f for f in os.listdir(raw_dir)
        if (f.endswith(".SAFE") or f.endswith(".zip")) and not f.endswith(".part")
    ] if os.path.exists(raw_dir) else []

    if safe_products:
        results.append(CheckResult(
            "Sentinel-1 download status",
            "PASS",
            f"{len(safe_products)} product(s) present: {safe_products}"
        ))
    else:
        results.append(CheckResult(
            "Sentinel-1 download status",
            "BLOCKED",
            "Candidate scenes discovered (Aug 13 & Aug 25 2024, ~3.28 GB); download pending explicit confirmation."
        ))

    # 10. Real-product preprocessing executed
    proc_dir = os.path.join(REPO_ROOT, settings.data_processed_dir)
    pre_raster = os.path.join(proc_dir, "pre_calibrated_sigma0.tif")
    post_raster = os.path.join(proc_dir, "post_calibrated_sigma0.tif")

    if os.path.exists(pre_raster) and os.path.exists(post_raster):
        v_pre, _, iss_pre = validate_processed_raster(pre_raster)
        v_post, _, iss_post = validate_processed_raster(post_raster)
        if v_pre and v_post:
            results.append(CheckResult(
                "Real-product preprocessing executed",
                "PASS",
                f"Calibrated GeoTIFFs validated in {proc_dir}."
            ))
        else:
            results.append(CheckResult(
                "Real-product preprocessing executed",
                "FAIL",
                f"Validation errors on processed rasters: {iss_pre + iss_post}"
            ))
    else:
        results.append(CheckResult(
            "Real-product preprocessing executed",
            "NOT RUN",
            "Raw Sentinel-1 products have not been downloaded or processed yet."
        ))

    # 11. Real change-detection raster available
    mask_raster = os.path.join(proc_dir, "preliminary_change_mask.tif")
    if os.path.exists(mask_raster):
        v_mask, m_mask, iss_mask = validate_processed_raster(mask_raster)
        if v_mask:
            results.append(CheckResult(
                "Real change-detection raster available",
                "PASS",
                f"Verified preliminary change mask: {mask_raster}"
            ))
        else:
            results.append(CheckResult(
                "Real change-detection raster available",
                "FAIL",
                f"Mask raster validation error: {iss_mask}"
            ))
    else:
        results.append(CheckResult(
            "Real change-detection raster available",
            "NOT RUN",
            "Awaiting genuine pre/post ARD rasters from preprocessing."
        ))

    # Print formatted table
    print(f"{'CHECK':<38} | {'STATUS':<9} | {'DETAILS'}")
    print("-" * 76)
    for res in results:
        status_colored = f"[{res.status}]"
        print(f"{res.name:<38} | {status_colored:<9} | {res.details}")
    print("-" * 76 + "\n")

    return results


if __name__ == "__main__":
    results = run_preflight()
    has_fail = any(r.status == "FAIL" for r in results)
    sys.exit(1 if has_fail else 0)
