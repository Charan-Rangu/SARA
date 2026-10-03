"""Unit tests for SARA Experiment 5 Independent Integrity Audit module."""

import os
import json
import pytest


def test_integrity_audit_outputs_exist():
    """Verify independent_integrity_audit.md and .json exist in experiments/final_benchmark/."""
    base_dir = "experiments/final_benchmark"
    assert os.path.exists(os.path.join(base_dir, "independent_integrity_audit.md"))
    assert os.path.exists(os.path.join(base_dir, "independent_integrity_audit.json"))


def test_integrity_audit_json_structure():
    """Verify structure and status of independent_integrity_audit.json."""
    json_path = "experiments/final_benchmark/independent_integrity_audit.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    assert data["audit_meta"]["overall_status"] == "PASS"
    assert data["reference_labels"]["status"] == "PASS"
    assert data["raster_alignment"]["status"] == "PASS"
    assert data["metric_recomputation"]["status"] == "PASS"
    assert data["data_leakage"]["status"] == "PASS"
    assert data["class_imbalance"]["status"] == "PASS"
    assert data["model_implementation"]["status"] == "PASS"


def test_integrity_audit_pixel_totals():
    """Verify total valid pixel count (260,096) across Somalia and Mekong audits."""
    json_path = "experiments/final_benchmark/independent_integrity_audit.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    somalia_valid = data["class_imbalance"]["somalia_validation"]["total_valid_pixels"]
    mekong_valid = data["class_imbalance"]["mekong_test"]["total_valid_pixels"]

    assert somalia_valid == 260096
    assert mekong_valid == 260096
