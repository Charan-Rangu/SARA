"""Tests for Phase 6 SNAP XML Graph Generation, Operator Verification, Download Safeguards, and Manifest Management.

Verifies:
1. SNAP XML graph builder node hierarchy, operators, and parameters.
2. SNAP Processor Adapter version check, operator verification, and missing SNAP handling.
3. Data Download Service storage checks, resumable manifest creation, archive integrity verification.
4. User confirmation requirement for multi-gigabyte downloads.
5. FastAPI /preprocessing endpoints.
"""

import pytest
import os
import tempfile
import zipfile
from fastapi.testclient import TestClient

from geospatial.snap_graph import SNAPGraphBuilder, SNAPGraphConfig
from geospatial.sar_preprocessing import SNAPProcessorAdapter, SARPreprocessingConfig
from backend.services.download_service import DataDownloadService, ProductManifestEntry
from backend.services.sentinel1_catalog import Sentinel1Scene
from backend.main import app

client = TestClient(app)


def test_snap_graph_builder_structure():
    builder = SNAPGraphBuilder()
    config = SNAPGraphConfig(
        polarizations=["VV", "VH"],
        calibration_type="SIGMA0",
        output_in_db=True,
        speckle_filter_type="Refined Lee",
        crs="EPSG:4326"
    )
    xml = builder.build_grd_ard_graph_xml("input/S1A_PRE.SAFE", "output/pre_sigma0.tif", config)

    assert "<graph id=\"Graph\">" in xml
    assert "<node id=\"Read\">" in xml
    assert "<node id=\"Apply-Orbit-File\">" in xml
    assert "<node id=\"ThermalNoiseRemoval\">" in xml
    assert "<node id=\"Remove-GRD-Border-Noise\">" in xml
    assert "<node id=\"Calibration\">" in xml
    assert "<node id=\"Speckle-Filter\">" in xml
    assert "<node id=\"Terrain-Correction\">" in xml
    assert "<node id=\"Write\">" in xml
    assert "<file>input/S1A_PRE.SAFE</file>" in xml
    assert "<file>output/pre_sigma0.tif</file>" in xml
    assert ("<mapProjection>WGS84(DD)</mapProjection>" in xml or "<mapProjection>EPSG:4326</mapProjection>" in xml)
    assert "<outputImageScaleInDb>true</outputImageScaleInDb>" in xml


def test_snap_adapter_operator_verification_missing_gpt():
    # Force missing gpt executable
    adapter = SNAPProcessorAdapter(gpt_path="non_existent_gpt_path")
    assert adapter.is_snap_available() is False
    assert adapter.get_snap_version() is None
    
    op_status = adapter.verify_operators()
    assert len(op_status) == 8
    assert all(val is False for val in op_status.values())


def test_download_service_free_space_and_manifest():
    with tempfile.TemporaryDirectory() as tmp_dir:
        service = DataDownloadService(download_dir=tmp_dir, default_budget_gb=5.0)
        free_space = service.get_free_disk_space_gb()
        assert free_space > 0.0

        scenes = [
            Sentinel1Scene(
                id="s1_test",
                title="S1A_IW_GRDH_TEST.SAFE",
                acquisition_time="2024-08-15T00:00:00Z",
                size_bytes=1500000000,
                download_url="https://zipper.dataspace.copernicus.eu/odata/v1/Products(s1_test)/$value"
            )
        ]
        manifest = service.prepare_download_manifest("batch_001", scenes)
        assert manifest.total_products_count == 1
        assert manifest.total_estimated_size_mb > 1400
        assert len(manifest.manifest_entries) == 1
        assert manifest.manifest_entries[0].status == "CONFIRMATION_REQUIRED"


def test_download_safeguard_confirmation_required():
    with tempfile.TemporaryDirectory() as tmp_dir:
        service = DataDownloadService(download_dir=tmp_dir)
        entry = ProductManifestEntry(
            product_id="prod_123",
            title="S1A_LARGE.SAFE",
            download_url="https://zipper.dataspace.copernicus.eu/odata/v1/Products(prod_123)/$value",
            expected_size_bytes=1600000000,
            target_filepath=os.path.join(tmp_dir, "S1A_LARGE.SAFE.zip")
        )

        # Attempt download without user confirmation -> Must be rejected cleanly
        res = service.download_product_with_safeguards(entry, user_confirmed=False)
        assert res.status == "CONFIRMATION_REQUIRED"
        assert "requires explicit user confirmation" in res.error_message


def test_archive_integrity_verification():
    with tempfile.TemporaryDirectory() as tmp_dir:
        service = DataDownloadService(download_dir=tmp_dir)
        valid_zip = os.path.join(tmp_dir, "valid.zip")

        # Create a valid test ZIP file
        with zipfile.ZipFile(valid_zip, "w") as zf:
            zf.writestr("manifest.safe", "<manifest></manifest>")

        actual_size = os.path.getsize(valid_zip)
        is_valid, msg = service.verify_archive_integrity(valid_zip, expected_size_bytes=actual_size)
        assert is_valid is True
        assert "passed integrity" in msg.lower()

        # Create a corrupted zip
        corrupt_zip = os.path.join(tmp_dir, "corrupt.zip")
        with open(corrupt_zip, "wb") as f:
            f.write(b"NOT_A_VALID_ZIP_HEADER_DATA")

        is_valid_c, msg_c = service.verify_archive_integrity(corrupt_zip)
        assert is_valid_c is False
        assert "not a valid zip" in msg_c.lower() or "corrupted" in msg_c.lower()


def test_preprocessing_snap_status_api():
    res = client.get("/preprocessing/snap-status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "snap_installed" in data
    assert "operator_status" in data


def test_preprocessing_generate_graph_api():
    payload = {
        "input_safe_path": "data/raw/S1A_PRE.SAFE",
        "output_raster_path": "data/processed/pre_sigma0.tif",
        "calibration_type": "SIGMA0",
        "output_in_db": True,
        "speckle_filter_type": "Refined Lee",
        "crs": "EPSG:4326"
    }
    res = client.post("/preprocessing/generate-graph", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "<graph id=\"Graph\">" in data["xml_graph_content"]
    assert "<operator>Terrain-Correction</operator>" in data["xml_graph_content"]
