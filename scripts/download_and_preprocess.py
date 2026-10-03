"""SARA Sentinel-1 Download and Real-Data Preprocessing Workflow Manager.

Provides a safe, verified workflow for downloading candidate Sentinel-1 scenes
from CDSE and executing ESA SNAP GPF preprocessing.

Safeguards:
- Dry-run by default.
- Strict confirmation required (`--confirm-download`) for multi-GB acquisitions.
- Resumable downloads with .part temporary files.
- ZIP integrity verification before unpacking or processing.
- No silent overwrites.
- No exposed credentials.
"""

import os
import sys
import argparse
import json
import zipfile
import shutil
from typing import Optional

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.config import settings, resolve_snap_gpt_path
from backend.services.sentinel1_catalog import Sentinel1Scene
from backend.services.scene_compatibility import SceneCompatibilityAnalyzer
from backend.services.download_service import (
    DataDownloadService,
    ProductManifestEntry,
    check_cdse_auth_status,
    get_cdse_auth_token
)
from geospatial.aoi import AOI, NEPAL_BHOTE_KOSHI_AOI, KERALA_FLOOD_EXAMPLE_AOI
from geospatial.snap_graph import SNAPGraphBuilder, SNAPGraphConfig
from geospatial.sar_preprocessing import (
    SNAPProcessorAdapter,
    SARPreprocessingConfig,
    validate_processed_raster
)


def load_candidate_scenes(scenario: str = "nepal") -> tuple[AOI, Sentinel1Scene, Sentinel1Scene]:
    """Load and return the verified candidate scene pair from catalog cache.
    
    Supports 'nepal' (Track B challenge scenario, default) and 'kerala' (test scenario).
    """
    scenario_clean = (scenario or "nepal").lower().strip()
    if scenario_clean == "nepal":
        cache_path = os.path.join(REPO_ROOT, "data", "cache", "catalog_cache_nepal.json")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Nepal catalog cache not found at: {cache_path}. Run catalog discovery first.")
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        aoi = NEPAL_BHOTE_KOSHI_AOI
        pre_scenes = [Sentinel1Scene(**s) for s in cache.get("pre_event_scenes", [])]
        post_scenes = [Sentinel1Scene(**s) for s in cache.get("post_event_scenes", [])]
        if not pre_scenes or not post_scenes:
            raise ValueError(f"Nepal catalog cache at {cache_path} missing pre/post scenes.")
        return aoi, pre_scenes[0], post_scenes[0]

    elif scenario_clean == "kerala":
        cache_path = os.path.join(REPO_ROOT, "data", "cache", "catalog_cache.json")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Kerala catalog cache not found at: {cache_path}. Run catalog discovery first.")
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        aoi = KERALA_FLOOD_EXAMPLE_AOI
        pre_scenes = [Sentinel1Scene(**s) for s in cache.get("pre_event_scenes", [])]
        post_scenes = [Sentinel1Scene(**s) for s in cache.get("post_event_scenes", [])]
        selected_pre = next((s for s in pre_scenes if "20240813" in s.title), pre_scenes[0])
        selected_post = next((s for s in post_scenes if "20240825" in s.title), post_scenes[0])
        return aoi, selected_pre, selected_post

    else:
        raise ValueError(f"Unknown scenario '{scenario}'. Supported scenarios: 'nepal', 'kerala'.")


def verify_compatibility(aoi: AOI, pre_scene: Sentinel1Scene, post_scene: Sentinel1Scene) -> bool:
    """Verify geometric and spectral compatibility between selected scenes."""
    analyzer = SceneCompatibilityAnalyzer()
    eval_res = analyzer.evaluate_compatibility(aoi, [pre_scene], [post_scene])
    if eval_res.suitable_pairs_count == 0:
        print("[ERROR] Selected scenes are not compatible:")
        for r in eval_res.rejected_pairs:
            print(f"  - Reasons: {r.rejection_reasons}")
        return False
    return True


def ensure_unpacked_safe(zip_path: str, extract_dir: str) -> Optional[str]:
    """Unpack a downloaded Sentinel-1 ZIP archive into its .SAFE directory if not already unpacked."""
    if not os.path.exists(zip_path):
        return None

    # Determine SAFE directory name inside archive
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = zf.namelist()
            safe_root = namelist[0].split("/")[0] if namelist else None
            if not safe_root or not safe_root.endswith(".SAFE"):
                safe_root = os.path.basename(zip_path).replace(".zip", "")
                if not safe_root.endswith(".SAFE"):
                    safe_root += ".SAFE"

            target_safe_path = os.path.join(extract_dir, safe_root)
            if os.path.isdir(target_safe_path) and os.path.exists(os.path.join(target_safe_path, "manifest.safe")):
                print(f"[INFO] Product already extracted at: {target_safe_path}")
                return target_safe_path

            print(f"[INFO] Extracting archive '{zip_path}' to '{target_safe_path}'...")
            zf.extractall(extract_dir)
            return target_safe_path
    except Exception as e:
        print(f"[ERROR] Failed to unpack archive '{zip_path}': {e}")
        return None


def run_workflow(
    confirm_download: bool = False,
    skip_download: bool = False,
    download_only: bool = False,
    dry_run: bool = False,
    scenario: str = "nepal"
) -> int:
    """Execute download and preprocessing workflow with strict safety verification."""
    print("=" * 76)
    print("        SARA — SENTINEL-1 DOWNLOAD & PREPROCESSING WORKFLOW")
    print("=" * 76)

    # 1. Load candidate scenes & AOI
    aoi, pre_scene, post_scene = load_candidate_scenes(scenario=scenario)
    print(f"\n[1/6] CANDIDATE SCENES & AOI VERIFICATION:")
    print(f"  AOI:                {aoi.name} (BBOX: {aoi.to_bbox()})")
    print(f"  Pre-Event Scene:    {pre_scene.title}")
    print(f"    - Acquisition:    {pre_scene.acquisition_time}")
    print(f"    - Relative Orbit: {pre_scene.relative_orbit_number} ({pre_scene.orbit_direction})")
    print(f"    - Mode/Polariz:   {pre_scene.sensor_mode} / {pre_scene.polarizations}")
    print(f"    - Size:           {pre_scene.size_bytes / (1024**3):.2f} GB ({pre_scene.size_bytes} bytes)")
    print(f"  Post-Event Scene:   {post_scene.title}")
    print(f"    - Acquisition:    {post_scene.acquisition_time}")
    print(f"    - Relative Orbit: {post_scene.relative_orbit_number} ({post_scene.orbit_direction})")
    print(f"    - Mode/Polariz:   {post_scene.sensor_mode} / {post_scene.polarizations}")
    print(f"    - Size:           {post_scene.size_bytes / (1024**3):.2f} GB ({post_scene.size_bytes} bytes)")

    # 2. Geometric Compatibility Check
    print(f"\n[2/6] GEOMETRIC COMPATIBILITY ASSESSMENT:")
    is_compat = verify_compatibility(aoi, pre_scene, post_scene)
    if not is_compat:
        return 1
    print(
        f"  [PASS] Scenes share relative orbit {pre_scene.relative_orbit_number}, "
        f"{pre_scene.orbit_direction} direction, {pre_scene.sensor_mode} mode, and full AOI coverage."
    )

    # 3. Disk Space & Storage Budget Assessment
    dl_service = DataDownloadService()
    free_disk_gb = dl_service.get_free_disk_space_gb()
    total_size_gb = ((pre_scene.size_bytes or 0) + (post_scene.size_bytes or 0)) / (1024 ** 3)
    batch_id = f"batch_{scenario}_{pre_scene.acquisition_time[:4]}"
    manifest = dl_service.prepare_download_manifest(batch_id, [pre_scene, post_scene])

    print(f"\n[3/6] STORAGE & BUDGET AUDIT:")
    print(f"  Available Free Disk Space:  {free_disk_gb:.2f} GB")
    print(f"  Expected Download Size:     {total_size_gb:.2f} GB")
    print(f"  Storage Budget:             {manifest.storage_budget_gb:.2f} GB")
    print(f"  Target Download Directory:  {os.path.abspath(dl_service.download_dir)}")

    if total_size_gb > free_disk_gb:
        print(f"[ERROR] Insufficient disk space ({free_disk_gb:.2f} GB free < {total_size_gb:.2f} GB needed).")
        return 1
    print(f"  [PASS] Storage capacity verified ({total_size_gb:.2f} GB is {total_size_gb/free_disk_gb*100:.1f}% of free space).")

    # 4. CDSE Authentication Audit
    print(f"\n[4/6] CDSE AUTHENTICATION STATUS:")
    auth_status = check_cdse_auth_status()
    print(f"  Status:             {'CONFIGURED' if auth_status['configured'] else 'NOT CONFIGURED'}")
    print(f"  Method:             {auth_status['method']}")
    print(f"  Details:            {auth_status['status_message']}")

    # 5. Preprocessing Graph Readiness Check
    print(f"\n[5/6] SNAP GPF GRAPH & DESTINATION PATHS:")
    builder = SNAPGraphBuilder()
    cfg = SNAPGraphConfig(
        polarizations=["VV", "VH"],
        calibration_type="SIGMA0",
        output_in_db=True,
        speckle_filter_type="Refined Lee",
        dem_name="Copernicus 30m Global DEM",
        pixel_spacing_m=10.0,
        crs="EPSG:4326"
    )

    proc_dir = os.path.abspath(settings.data_processed_dir)
    os.makedirs(proc_dir, exist_ok=True)

    pre_safe_path = os.path.join(REPO_ROOT, settings.data_raw_dir, pre_scene.title)
    post_safe_path = os.path.join(REPO_ROOT, settings.data_raw_dir, post_scene.title)
    pre_out_path = os.path.join(proc_dir, "pre_calibrated_sigma0.tif")
    post_out_path = os.path.join(proc_dir, "post_calibrated_sigma0.tif")

    pre_xml = builder.build_grd_ard_graph_xml(pre_safe_path, pre_out_path, cfg)
    post_xml = builder.build_grd_ard_graph_xml(post_safe_path, post_out_path, cfg)

    ok_pre, iss_pre = builder.validate_graph_xml(pre_xml)
    ok_post, iss_post = builder.validate_graph_xml(post_xml)

    print(f"  Pre-event Graph XML:  {'VALID [PASS]' if ok_pre else 'INVALID [FAIL]'}")
    print(f"  Post-event Graph XML: {'VALID [PASS]' if ok_post else 'INVALID [FAIL]'}")
    print(f"  Output Pre-Raster:    {pre_out_path}")
    print(f"  Output Post-Raster:   {post_out_path}")

    # 6. Action Execution / Safeguard Halt
    print(f"\n[6/6] WORKFLOW ACTION EXECUTION:")
    if dry_run or (not confirm_download and not skip_download):
        print("  [SAFEGUARD ACTIVE] Download execution paused.")
        print("  Reason: Explicit user confirmation required (`--confirm-download`) to acquire multi-GB data.")
        if not auth_status["configured"]:
            print("  Note: CDSE credentials must be set in .env (CDSE_USERNAME and CDSE_PASSWORD) before running.")
        print("\nTo execute verified download and preprocessing, run:")
        print("  python scripts/download_and_preprocess.py --confirm-download")
        return 0

    if not skip_download:
        if not auth_status["configured"]:
            print("[ERROR] Cannot proceed with download: CDSE authentication is not configured.")
            print("Please configure CDSE_USERNAME and CDSE_PASSWORD in .env or provide CDSE_ACCESS_TOKEN.")
            return 1

        print("  [CONFIRMED] User confirmation provided. Initiating download...")
        for entry in manifest.manifest_entries:
            res = dl_service.download_product_with_safeguards(entry, user_confirmed=True)
            print(f"  Product '{entry.title}': Status = {res.status}")
            print(f"    - Target Path:       {os.path.abspath(res.target_filepath)}")
            print(f"    - Verified Size:     {res.downloaded_size_bytes / (1024**3):.3f} GB ({res.downloaded_size_bytes:,} bytes)")
            print(f"    - Integrity Checked: {'YES [PASS]' if res.integrity_verified else 'NO [FAIL]'}")
            if res.checksum_sha256:
                print(f"    - SHA256:            {res.checksum_sha256}")
            if res.status != "COMPLETED":
                print(f"  [ERROR] Download failed: {res.error_message}")
                return 1

        print("\n[SUCCESS] All satellite products successfully downloaded and verified.")

    if download_only:
        print("\n[INFO] --download-only flag specified. Skipping SNAP preprocessing and change detection.")
        print(f"Downloaded archives verified in: {os.path.abspath(settings.data_raw_dir)}")
        print("\nNext step when ready to preprocess:")
        print("  python scripts/download_and_preprocess.py --skip-download")
        return 0

    # 7. Archive Extraction to .SAFE
    print("\n[7/8] ARCHIVE EXTRACTION TO .SAFE DIRECTORIES:")
    raw_dir = os.path.abspath(settings.data_raw_dir)
    pre_zip = os.path.join(raw_dir, f"{pre_scene.title}.zip")
    post_zip = os.path.join(raw_dir, f"{post_scene.title}.zip")

    pre_safe_path = ensure_unpacked_safe(pre_zip, raw_dir) if os.path.exists(pre_zip) else os.path.join(raw_dir, pre_scene.title)
    post_safe_path = ensure_unpacked_safe(post_zip, raw_dir) if os.path.exists(post_zip) else os.path.join(raw_dir, post_scene.title)

    if not os.path.isdir(pre_safe_path) or not os.path.isdir(post_safe_path):
        print(f"[ERROR] SAFE directories not available: pre={pre_safe_path}, post={post_safe_path}")
        return 1

    # 8. SNAP GPF Preprocessing & Change Detection
    print("\n[8/8] EXECUTING ESA SNAP GPF PREPROCESSING:")
    adapter = SNAPProcessorAdapter()
    preproc_res = adapter.preprocess_scene_pair(
        pre_safe_path=pre_safe_path,
        post_safe_path=post_safe_path,
        aoi=aoi,
        config=SARPreprocessingConfig(
            polarizations=["VV", "VH"],
            calibration_type="SIGMA0",
            output_units="dB",
            speckle_filter_type="Refined Lee",
            dem_name="Copernicus 30m Global DEM",
            output_dir=proc_dir
        )
    )

    if preproc_res.status != "SUCCESS":
        print(f"[ERROR] SNAP preprocessing failed: {preproc_res.warnings}")
        return 1

    print("  [PASS] SNAP ARD Preprocessing completed successfully.")
    print(f"  Calibrated Pre:  {preproc_res.calibrated_pre_raster}")
    print(f"  Calibrated Post: {preproc_res.calibrated_post_raster}")

    # Change Detection
    from geospatial.change_detection import SARChangeDetector, ChangeDetectionConfig
    detector = SARChangeDetector(ChangeDetectionConfig(threshold_db=-3.0, input_units="dB"))
    mask_path = os.path.join(proc_dir, "preliminary_change_mask.tif")
    change_res = detector.detect_changes(
        pre_raster_path=preproc_res.calibrated_pre_raster,
        post_raster_path=preproc_res.calibrated_post_raster,
        output_mask_path=mask_path,
        pre_product_id=pre_scene.title,
        post_product_id=post_scene.title
    )
    print(f"  [PASS] Change detection mask generated at: {mask_path}")
    print(f"  Detected changed pixels: {change_res.changed_pixels} ({change_res.change_percentage:.2f}%)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SARA Sentinel-1 Download & Preprocessing Workflow")
    parser.add_argument("--confirm-download", action="store_true", help="Explicit confirmation to download multi-GB scenes")
    parser.add_argument("--download-only", action="store_true", help="Download and verify satellite scenes without running SNAP preprocessing")
    parser.add_argument("--skip-download", action="store_true", help="Skip download and proceed with existing files")
    parser.add_argument("--dry-run", action="store_true", help="Simulate workflow without network transfers")
    parser.add_argument("--scenario", choices=["nepal", "kerala"], default="nepal", help="Disaster scenario AOI and candidate pair (default: nepal)")
    args = parser.parse_args()

    sys.exit(run_workflow(
        confirm_download=args.confirm_download,
        download_only=args.download_only,
        skip_download=args.skip_download,
        dry_run=args.dry_run,
        scenario=args.scenario
    ))

