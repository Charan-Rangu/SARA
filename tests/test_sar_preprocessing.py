import pytest
from geospatial.aoi import AOI
from geospatial.sar_preprocessing import (
    SARPreprocessingConfig,
    SNAPProcessorAdapter
)

def test_preprocessing_config_defaults():
    cfg = SARPreprocessingConfig()
    assert cfg.target_crs == "EPSG:4326"
    assert cfg.pixel_spacing_m == 10.0
    assert cfg.calibration_type == "SIGMA0"
    assert cfg.output_units == "dB"

def test_snap_adapter_available_check():
    adapter = SNAPProcessorAdapter()
    available = adapter.is_snap_available()
    assert isinstance(available, bool)

def test_snap_adapter_missing_warning():
    adapter = SNAPProcessorAdapter(gpt_path="/nonexistent/path/to/gpt")
    assert adapter.is_snap_available() is False

    aoi = AOI(bbox=(76.15, 9.85, 76.45, 10.15))
    res = adapter.preprocess_scene_pair(
        pre_safe_path="data/raw/pre.SAFE",
        post_safe_path="data/raw/post.SAFE",
        aoi=aoi,
        config=SARPreprocessingConfig()
    )

    assert res.status == "SNAP_NOT_INSTALLED"
    assert res.snap_available is False
    assert len(res.warnings) > 0
    assert "ESA SNAP gpt.exe is not installed" in res.warnings[0]

def test_installation_instructions_string():
    adapter = SNAPProcessorAdapter()
    instructions = adapter.get_installation_instructions()
    assert "ESA SNAP" in instructions
    assert "https://step.esa.int/main/download/snap-download/" in instructions
