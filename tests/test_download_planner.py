import pytest
from backend.services.sentinel1_catalog import Sentinel1Scene
from backend.services.download_planner import DownloadPlanner

def test_download_plan_within_budget():
    planner = DownloadPlanner(default_storage_budget_gb=10.0)

    scene1 = Sentinel1Scene(
        id="SCENE_01",
        title="SCENE_01",
        acquisition_time="2026-08-10T00:00:00Z",
        size_bytes=1850000000  # ~1.72 GB
    )
    scene2 = Sentinel1Scene(
        id="SCENE_02",
        title="SCENE_02",
        acquisition_time="2026-08-17T00:00:00Z",
        size_bytes=1950000000  # ~1.82 GB
    )

    plan = planner.create_download_plan([scene1, scene2], storage_budget_gb=5.0)

    assert plan.scenes_count == 2
    assert plan.within_budget is True
    assert plan.total_estimated_size_gb == pytest.approx(3.54, abs=0.1)
    assert len(plan.warnings) == 0

def test_download_plan_exceeds_budget_warning():
    planner = DownloadPlanner()

    scene1 = Sentinel1Scene(
        id="SCENE_LARGE_01",
        title="SCENE_LARGE_01",
        acquisition_time="2026-08-10T00:00:00Z",
        size_bytes=2500000000  # ~2.33 GB
    )
    scene2 = Sentinel1Scene(
        id="SCENE_LARGE_02",
        title="SCENE_LARGE_02",
        acquisition_time="2026-08-17T00:00:00Z",
        size_bytes=2500000000  # ~2.33 GB
    )

    # Set tight budget of 3.0 GB
    plan = planner.create_download_plan([scene1, scene2], storage_budget_gb=3.0)

    assert plan.within_budget is False
    assert any("exceeds configured storage budget" in w for w in plan.warnings)

def test_missing_size_metadata_warning():
    planner = DownloadPlanner()

    scene_no_size = Sentinel1Scene(
        id="SCENE_NO_SIZE",
        title="SCENE_NO_SIZE",
        acquisition_time="2026-08-10T00:00:00Z",
        size_bytes=None
    )

    plan = planner.create_download_plan([scene_no_size], storage_budget_gb=10.0)

    assert plan.unknown_size_count == 1
    assert any("lack explicit size metadata" in w for w in plan.warnings)
