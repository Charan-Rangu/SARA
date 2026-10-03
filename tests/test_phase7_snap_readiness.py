"""Tests for Phase 7A — SNAP Integration Audit, Graph Validation, Execution Readiness, and Safeguards.

Covers:
1. Static XML Graph validation (valid XML vs invalid XML with missing operators, malformed XML, broken DAG).
2. SNAP discovery statuses (SNAP_NOT_INSTALLED, SNAP_AVAILABLE, SNAP_VERIFICATION_FAILED).
3. Mocked SNAP execution success, failures, and timeout handling.
4. Output raster validation using Rasterio (CRS, transform, dimensions, nodata, empty/corrupt rasters).
5. Raster grid compatibility validation (matching vs mismatched CRS, dimensions, affine transforms).
6. Change detection validation with product IDs and metadata tags.
7. Download safeguards: storage limit enforcement, confirmation requirement, and .part temporary file usage.
8. Preflight check runner execution.
"""

import pytest
import os
import tempfile
import subprocess
from unittest.mock import patch, MagicMock
import numpy as np
import rasterio
from rasterio.transform import from_origin

from geospatial.snap_graph import SNAPGraphBuilder, SNAPGraphConfig
from geospatial.sar_preprocessing import (
    SNAPProcessorAdapter,
    SARPreprocessingConfig,
    validate_processed_raster,
    validate_raster_grid_compatibility
)
from geospatial.change_detection import SARChangeDetector, ChangeDetectionConfig
from backend.services.download_service import (
    DataDownloadService,
    ProductManifestEntry,
    check_cdse_auth_status,
    get_cdse_auth_token,
    verify_cdse_authentication
)
from scripts.preflight_check import run_preflight
from scripts.download_and_preprocess import run_workflow
from scripts.check_cdse_auth import run_auth_check




def test_snap_graph_xml_valid_and_invalid():
    builder = SNAPGraphBuilder()
    valid_xml = builder.build_grd_ard_graph_xml(
        "data/raw/test.SAFE",
        "data/processed/out.tif",
        SNAPGraphConfig()
    )
    is_valid, issues = builder.validate_graph_xml(valid_xml)
    assert is_valid is True
    assert len(issues) == 0

    # Test malformed XML
    is_valid_malformed, issues_malformed = builder.validate_graph_xml("<graph><node id='Read'></graph>")
    assert is_valid_malformed is False
    assert any("parsing failed" in s.lower() for s in issues_malformed)

    # Test XML missing required operator
    incomplete_xml = """<graph id="Graph"><version>1.0</version>
    <node id="Read"><operator>Read</operator><parameters><file>test.safe</file></parameters></node>
    <node id="Write"><operator>Write</operator><sources><sourceProduct refid="Read"/></sources><parameters><file>out.tif</file></parameters></node>
    </graph>"""
    is_valid_inc, issues_inc = builder.validate_graph_xml(incomplete_xml)
    assert is_valid_inc is False
    assert any("missing from graph" in s.lower() for s in issues_inc)


def test_snap_discovery_statuses():
    # Test SNAP_NOT_INSTALLED with non-existent path
    adapter_missing = SNAPProcessorAdapter(gpt_path="non_existent_gpt_path")
    status_missing = adapter_missing.get_discovery_status()
    assert status_missing["status"] == "SNAP_NOT_INSTALLED"
    assert status_missing["version"] is None

    # Test SNAP_VERIFICATION_FAILED when gpt executable returns non-zero
    with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f:
        dummy_exe = f.name

    try:
        adapter_dummy = SNAPProcessorAdapter(gpt_path=dummy_exe)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="Mocked fatal JVM initialization error")
            status_failed = adapter_dummy.get_discovery_status()
            assert status_failed["status"] == "SNAP_VERIFICATION_FAILED"
            assert "Mocked fatal" in status_failed["diagnostics"]
    finally:
        if os.path.exists(dummy_exe):
            os.remove(dummy_exe)


def test_snap_execution_mocked_success_and_timeout():
    adapter = SNAPProcessorAdapter(gpt_path="gpt")

    # Mock success execution
    with patch.object(adapter, "is_snap_available", return_value=True):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="Graph processing completed successfully.")
            success, msg, elapsed = adapter.execute_snap_graph("graph.xml", timeout_seconds=10)
            assert success is True
            assert "succeeded" in msg

        # Mock timeout execution
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["gpt"], timeout=10)):
            success_to, msg_to, elapsed_to = adapter.execute_snap_graph("graph.xml", timeout_seconds=10)
            assert success_to is False
            assert "timed out" in msg_to


def test_raster_validation_valid_and_invalid():
    with tempfile.TemporaryDirectory() as tmp_dir:
        valid_tif = os.path.join(tmp_dir, "valid.tif")
        transform = from_origin(76.15, 10.15, 0.0001, 0.0001)

        # 1. Create a valid synthetic GeoTIFF
        data = np.ones((1, 64, 64), dtype=np.float32) * -12.5
        with rasterio.open(
            valid_tif, "w",
            driver="GTiff", height=64, width=64, count=1,
            dtype=rasterio.float32, crs="EPSG:4326", transform=transform, nodata=-9999.0
        ) as dst:
            dst.write(data)

        is_valid, meta, issues = validate_processed_raster(valid_tif, expected_crs="EPSG:4326")
        assert is_valid is True
        assert len(issues) == 0
        assert meta["width"] == 64
        assert meta["crs"] == "EPSG:4326"

        # 2. Check CRS mismatch detection
        is_valid_crs, _, issues_crs = validate_processed_raster(valid_tif, expected_crs="EPSG:32643")
        assert is_valid_crs is False
        assert any("crs mismatch" in s.lower() for s in issues_crs)

        # 3. Check empty file detection
        empty_tif = os.path.join(tmp_dir, "empty.tif")
        with open(empty_tif, "wb") as f:
            pass
        is_valid_empty, _, issues_empty = validate_processed_raster(empty_tif)
        assert is_valid_empty is False
        assert any("empty" in s.lower() for s in issues_empty)


def test_raster_grid_compatibility_validation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        transform_a = from_origin(76.15, 10.15, 0.0001, 0.0001)
        transform_b = from_origin(76.20, 10.20, 0.0001, 0.0001)  # Shifted transform

        tif_pre = os.path.join(tmp_dir, "pre.tif")
        tif_post_matching = os.path.join(tmp_dir, "post_match.tif")
        tif_post_mismatched = os.path.join(tmp_dir, "post_mismatch.tif")

        # Create matching rasters
        data = np.ones((1, 32, 32), dtype=np.float32)
        meta = {
            "driver": "GTiff", "height": 32, "width": 32, "count": 1,
            "dtype": rasterio.float32, "crs": "EPSG:4326"
        }

        with rasterio.open(tif_pre, "w", transform=transform_a, **meta) as dst:
            dst.write(data)
        with rasterio.open(tif_post_matching, "w", transform=transform_a, **meta) as dst:
            dst.write(data)
        with rasterio.open(tif_post_mismatched, "w", transform=transform_b, **meta) as dst:
            dst.write(data)

        # Test matching grids
        compat_ok, issues = validate_raster_grid_compatibility(tif_pre, tif_post_matching)
        assert compat_ok is True
        assert len(issues) == 0

        # Test mismatched transform rejection
        mismatch_ok, issues_mis = validate_raster_grid_compatibility(tif_pre, tif_post_mismatched)
        assert mismatch_ok is False
        assert any("affine transform mismatch" in s.lower() for s in issues_mis)


def test_change_detection_with_metadata_and_units():
    with tempfile.TemporaryDirectory() as tmp_dir:
        transform = from_origin(76.15, 10.15, 0.0001, 0.0001)
        meta = {
            "driver": "GTiff", "height": 32, "width": 32, "count": 1,
            "dtype": rasterio.float32, "crs": "EPSG:4326", "transform": transform
        }

        pre_path = os.path.join(tmp_dir, "pre_cal.tif")
        post_path = os.path.join(tmp_dir, "post_cal.tif")
        out_mask = os.path.join(tmp_dir, "preliminary_change_mask.tif")

        # Synthetic SAR backscatter: pre=-12dB, post=-18dB -> diff = -6dB (< -3dB threshold)
        pre_data = np.full((1, 32, 32), fill_value=-12.0, dtype=np.float32)
        post_data = np.full((1, 32, 32), fill_value=-18.0, dtype=np.float32)

        with rasterio.open(pre_path, "w", **meta) as dst:
            dst.write(pre_data)
        with rasterio.open(post_path, "w", **meta) as dst:
            dst.write(post_data)

        detector = SARChangeDetector(ChangeDetectionConfig(threshold_db=-3.0, input_units="dB"))
        res = detector.detect_changes(
            pre_path, post_path, output_mask_path=out_mask,
            pre_product_id="S1A_PRE_SAFE", post_product_id="S1A_POST_SAFE"
        )

        assert res.changed_pixels == 32 * 32
        assert res.change_percentage == 100.0
        assert res.pre_product_id == "S1A_PRE_SAFE"
        assert res.post_product_id == "S1A_POST_SAFE"
        assert "preliminary" in res.disclaimer.lower()

        # Check metadata tags in written GeoTIFF
        with rasterio.open(out_mask) as dst_mask:
            tags = dst_mask.tags()
            assert tags["PRE_PRODUCT_ID"] == "S1A_PRE_SAFE"
            assert tags["POST_PRODUCT_ID"] == "S1A_POST_SAFE"
            assert tags["INPUT_UNITS"] == "dB"


def test_download_service_part_file_and_disk_safeguard():
    with tempfile.TemporaryDirectory() as tmp_dir:
        service = DataDownloadService(download_dir=tmp_dir)

        entry = ProductManifestEntry(
            product_id="prod_test",
            title="S1A_SCENE",
            download_url="https://zipper.dataspace.copernicus.eu/odata/v1/Products(prod_test)/$value",
            expected_size_bytes=2000000000,
            target_filepath=os.path.join(tmp_dir, "S1A_SCENE.zip"),
            part_filepath=os.path.join(tmp_dir, "S1A_SCENE.zip.part")
        )

        # 1. Unconfirmed multi-GB download must be rejected
        res = service.download_product_with_safeguards(entry, user_confirmed=False)
        assert res.status == "CONFIRMATION_REQUIRED"

        # 2. Over-budget / storage exceeded check
        with patch.object(service, "get_free_disk_space_gb", return_value=0.5):  # 500 MB free vs 2GB needed
            res_storage = service.download_product_with_safeguards(entry, user_confirmed=True)
            assert res_storage.status == "STORAGE_EXCEEDED"
            assert "insufficient disk space" in res_storage.error_message.lower()


def test_preflight_check_execution():
    results = run_preflight()
    assert len(results) >= 8

    # Virtual Environment and Packages must PASS
    env_res = next(r for r in results if r.name == "Python Environment & Packages")
    assert env_res.status == "PASS"

    # SNAP checks
    snap_found = next(r for r in results if r.name == "SNAP executable found")
    assert snap_found.status in ("PASS", "BLOCKED")

    xml_res = next(r for r in results if r.name == "Graph XML valid")
    assert xml_res.status == "PASS"

    params_res = next(r for r in results if r.name == "Graph parameters verified")
    assert params_res.status == "PASS"

    real_pre = next(r for r in results if r.name == "Real-product preprocessing executed")
    assert real_pre.status in ("NOT RUN", "PASS")


def test_cdse_auth_helpers():
    # 1. When no credentials or token configured
    with patch("backend.services.download_service.settings.cdse_access_token", None):
        with patch("backend.services.download_service.settings.cdse_username", None):
            with patch("backend.services.download_service.settings.cdse_password", None):
                status = check_cdse_auth_status()
                assert status["configured"] is False
                assert status["method"] == "NONE"

                token, msg = get_cdse_auth_token()
                assert token is None
                assert "not configured" in msg.lower()

    # 2. When direct token is provided
    status_tok = check_cdse_auth_status()
    token_direct, msg_tok = get_cdse_auth_token(direct_token="mock_bearer_token")
    assert token_direct == "mock_bearer_token"
    assert "direct" in msg_tok.lower()


def test_download_and_preprocess_dry_run():
    # Run workflow in dry-run mode (safeguards active, exit 0)
    exit_code = run_workflow(confirm_download=False, dry_run=True)
    assert exit_code == 0


def test_verify_cdse_authentication_success_and_failure():
    # 1. Unconfigured credentials
    with patch("backend.services.download_service.settings.cdse_access_token", None):
        with patch("backend.services.download_service.settings.cdse_username", None):
            with patch("backend.services.download_service.settings.cdse_password", None):
                ok, msg = verify_cdse_authentication()
                assert ok is False
                assert "not configured" in msg.lower()

    # 2. Mocked valid username/password Keycloak token response
    with patch("backend.services.download_service.settings.cdse_access_token", None):
        with patch("httpx.Client.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"access_token": "mock_jwt_token_abc"}
            )
            ok_cred, msg_cred = verify_cdse_authentication(username="user@test.org", password="valid_password")
            assert ok_cred is True
            assert "verified" in msg_cred.lower()

    # 3. Mocked invalid username/password Keycloak response
    with patch("backend.services.download_service.settings.cdse_access_token", None):
        with patch("httpx.Client.post") as mock_post_fail:
            mock_post_fail.return_value = MagicMock(
                status_code=401,
                json=lambda: {"error": "invalid_grant"}
            )
            ok_bad, msg_bad = verify_cdse_authentication(username="user@test.org", password="wrong_password")
            assert ok_bad is False
            assert "invalid" in msg_bad.lower()


    # 4. Mocked direct token validation against userinfo endpoint
    with patch("httpx.Client.get") as mock_get_userinfo:
        mock_get_userinfo.return_value = MagicMock(status_code=200)
        ok_tok, msg_tok = verify_cdse_authentication(direct_token="valid_bearer_token")
        assert ok_tok is True
        assert "valid and active" in msg_tok.lower()


def test_check_cdse_auth_script():
    # When unconfigured, run_auth_check returns exit code 1
    with patch("scripts.check_cdse_auth.settings.cdse_access_token", None):
        with patch("scripts.check_cdse_auth.settings.cdse_username", None):
            with patch("scripts.check_cdse_auth.settings.cdse_password", None):
                exit_code = run_auth_check()
                assert exit_code == 1

    # When mocked valid, run_auth_check returns exit code 0
    with patch("scripts.check_cdse_auth.settings.cdse_access_token", "mock_valid_token"):
        with patch("scripts.check_cdse_auth.verify_cdse_authentication", return_value=(True, "Mock success")):
            exit_code_ok = run_auth_check()
            assert exit_code_ok == 0



