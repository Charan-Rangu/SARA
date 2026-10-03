"""Manual Live Smoke Test for Copernicus Catalogue API.

Run manually when internet connectivity is available:
    python tests/smoke_test_live_catalog.py

Queries the live CDSE OData API endpoint (https://catalogue.dataspace.copernicus.eu/odata/v1/Products)
for Sentinel-1 GRD scenes over the Kerala example region during August 2024 without downloading rasters.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geospatial.aoi import KERALA_FLOOD_EXAMPLE_AOI
from backend.services.sentinel1_catalog import Sentinel1CatalogService
from backend.services.scene_compatibility import SceneCompatibilityAnalyzer
from backend.services.download_planner import DownloadPlanner

def run_live_smoke_test():
    print("=" * 70)
    print("SARA LIVE CATALOGUE SMOKE TEST (COPERNICUS DATA SPACE ECOSYSTEM)")
    print("=" * 70)
    print(f"Target AOI:  {KERALA_FLOOD_EXAMPLE_AOI.name}")
    print(f"BBox:        {KERALA_FLOOD_EXAMPLE_AOI.to_bbox()}")
    print("Event Date:  2024-08-15 (Pre-event: -14d, Post-event: +14d)")
    print("-" * 70)

    catalog_service = Sentinel1CatalogService(timeout_seconds=20.0)
    analyzer = SceneCompatibilityAnalyzer()
    planner = DownloadPlanner()

    print("Querying live CDSE endpoint: https://catalogue.dataspace.copernicus.eu/odata/v1/Products ...")
    
    result = catalog_service.search_scenes(
        aoi=KERALA_FLOOD_EXAMPLE_AOI,
        event_date_str="2024-08-15",
        pre_event_days=14,
        post_event_days=14,
        max_results=20
    )

    print(f"Pre-event scenes found:  {len(result.pre_event_scenes)}")
    print(f"Post-event scenes found: {len(result.post_event_scenes)}")
    print(f"Total scenes retrieved:  {result.total_scenes_found}")
    print(f"Loaded from cache:       {result.from_cache}")

    if result.pre_event_scenes:
        print("\nSAMPLE PRE-EVENT SCENES:")
        for s in result.pre_event_scenes[:3]:
            print(f"  - [{s.id[:8]}...] {s.title}")
            print(f"    Acquisition: {s.acquisition_time} | Orbit Dir: {s.orbit_direction} | Rel Orbit: {s.relative_orbit_number}")

    if result.post_event_scenes:
        print("\nSAMPLE POST-EVENT SCENES:")
        for s in result.post_event_scenes[:3]:
            print(f"  - [{s.id[:8]}...] {s.title}")
            print(f"    Acquisition: {s.acquisition_time} | Orbit Dir: {s.orbit_direction} | Rel Orbit: {s.relative_orbit_number}")

    if result.warning_messages:
        print("\nWarnings:")
        for w in result.warning_messages:
            print(f"  [WARN] {w}")

    print("-" * 70)
    print("Evaluating Scene Compatibility...")
    compat = analyzer.evaluate_compatibility(
        aoi=KERALA_FLOOD_EXAMPLE_AOI,
        pre_scenes=result.pre_event_scenes,
        post_scenes=result.post_event_scenes
    )

    print(f"Compatibility Status:    {compat.status}")
    print(f"Suitable pairs count:    {compat.suitable_pairs_count}")
    print(f"Rejected pairs count:    {len(compat.rejected_pairs)}")

    if compat.ranked_candidate_pairs:
        top_pair = compat.ranked_candidate_pairs[0]
        print("\nTOP COMPATIBLE SCENE PAIR:")
        print(f"  Pre-event ID:   {top_pair.pre_scene.id}")
        print(f"  Pre Title:      {top_pair.pre_scene.title}")
        print(f"  Pre Date:       {top_pair.pre_scene.acquisition_time}")
        print(f"  Post-event ID:  {top_pair.post_scene.id}")
        print(f"  Post Title:     {top_pair.post_scene.title}")
        print(f"  Post Date:      {top_pair.post_scene.acquisition_time}")
        print(f"  Score:          {top_pair.compatibility_score}/100")

        plan = planner.create_download_plan(
            scenes=[top_pair.pre_scene, top_pair.post_scene],
            storage_budget_gb=10.0
        )
        print("\nDOWNLOAD PLAN ESTIMATE (NO DOWNLOADS EXECUTED):")
        print(f"  Estimated size: {plan.total_estimated_size_gb:.2f} GB")
        print(f"  Within budget:  {plan.within_budget}")

    print("=" * 70)
    print("LIVE SMOKE TEST FINISHED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    try:
        run_live_smoke_test()
    except Exception as e:
        print(f"Smoke test failed: {e}", file=sys.stderr)
        sys.exit(1)
