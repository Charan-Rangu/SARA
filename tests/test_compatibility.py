import pytest
from geospatial.aoi import AOI
from backend.services.sentinel1_catalog import Sentinel1Scene
from backend.services.scene_compatibility import SceneCompatibilityAnalyzer

def create_scene(scene_id: str, rel_orbit: int, orbit_dir: str, mode: str = "IW", pols: list = ["VV", "VH"]):
    return Sentinel1Scene(
        id=scene_id,
        title=scene_id,
        acquisition_time="2026-08-10T00:00:00Z",
        platform="SENTINEL-1A",
        product_type="GRD",
        sensor_mode=mode,
        polarizations=pols,
        relative_orbit_number=rel_orbit,
        orbit_direction=orbit_dir,
        footprint_geojson={
            "type": "Polygon",
            "coordinates": [[[76.0, 9.5], [76.5, 9.5], [76.5, 10.2], [76.0, 10.2], [76.0, 9.5]]]
        }
    )

def test_matching_compatible_pair():
    aoi = AOI(bbox=(76.15, 9.85, 76.45, 10.15))
    analyzer = SceneCompatibilityAnalyzer()

    pre = create_scene("PRE_001", rel_orbit=114, orbit_dir="ASCENDING")
    post = create_scene("POST_001", rel_orbit=114, orbit_dir="ASCENDING")

    eval_res = analyzer.evaluate_compatibility(aoi, [pre], [post])

    assert eval_res.status == "SUITABLE_PAIRS_FOUND"
    assert len(eval_res.ranked_candidate_pairs) == 1
    top_pair = eval_res.ranked_candidate_pairs[0]
    assert top_pair.compatibility_score >= 90.0
    assert top_pair.relative_orbit_match is True
    assert top_pair.orbit_direction_match is True

def test_orbit_direction_mismatch_rejection():
    aoi = AOI(bbox=(76.15, 9.85, 76.45, 10.15))
    analyzer = SceneCompatibilityAnalyzer()

    pre = create_scene("PRE_001", rel_orbit=114, orbit_dir="ASCENDING")
    post = create_scene("POST_002", rel_orbit=114, orbit_dir="DESCENDING")

    eval_res = analyzer.evaluate_compatibility(aoi, [pre], [post])

    assert eval_res.status == "NO_SUITABLE_PAIR_EXISTS"
    assert len(eval_res.rejected_pairs) == 1
    rejection = eval_res.rejected_pairs[0]
    assert any("Orbit direction mismatch" in r for r in rejection.rejection_reasons)

def test_sensor_mode_mismatch_rejection():
    aoi = AOI(bbox=(76.15, 9.85, 76.45, 10.15))
    analyzer = SceneCompatibilityAnalyzer()

    pre = create_scene("PRE_001", rel_orbit=114, orbit_dir="ASCENDING", mode="IW")
    post = create_scene("POST_003", rel_orbit=114, orbit_dir="ASCENDING", mode="EW")

    eval_res = analyzer.evaluate_compatibility(aoi, [pre], [post])

    assert eval_res.status == "NO_SUITABLE_PAIR_EXISTS"
    assert len(eval_res.rejected_pairs) == 1
    rejection = eval_res.rejected_pairs[0]
    assert any("Sensor mode mismatch" in r for r in rejection.rejection_reasons)

def test_different_relative_orbits_ranking():
    aoi = AOI(bbox=(76.15, 9.85, 76.45, 10.15))
    analyzer = SceneCompatibilityAnalyzer()

    pre = create_scene("PRE_001", rel_orbit=114, orbit_dir="ASCENDING")
    post_same_orbit = create_scene("POST_MATCH", rel_orbit=114, orbit_dir="ASCENDING")
    post_diff_orbit = create_scene("POST_DIFF", rel_orbit=42, orbit_dir="ASCENDING")

    eval_res = analyzer.evaluate_compatibility(aoi, [pre], [post_same_orbit, post_diff_orbit])

    assert eval_res.status == "SUITABLE_PAIRS_FOUND"
    assert len(eval_res.ranked_candidate_pairs) == 2
    # Same relative orbit pair should rank higher score than different relative orbit pair
    assert eval_res.ranked_candidate_pairs[0].post_scene.id == "POST_MATCH"
    assert eval_res.ranked_candidate_pairs[0].compatibility_score > eval_res.ranked_candidate_pairs[1].compatibility_score
