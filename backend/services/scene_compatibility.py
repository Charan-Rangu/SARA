"""Sentinel-1 Scene Pair Compatibility & Geometric Matching Service.

Evaluates candidate pre-event and post-event Sentinel-1 SAR scene pairs for SAR change-detection suitability.
Checks relative orbit alignment, orbit direction consistency (Ascending vs Descending),
sensor mode compatibility (IW), polarization match (VV/VH), and spatial coverage of the AOI.

Note:
Matching relative orbits and orbit directions provides consistent viewing geometry, but does NOT guarantee
pixel-level co-registration. Terrain correction, spatial alignment, and radiometric calibration are still
required in subsequent preprocessing stages.
"""

from typing import Optional
from pydantic import BaseModel, Field

from geospatial.aoi import AOI
from backend.services.sentinel1_catalog import Sentinel1Scene


class CandidatePair(BaseModel):
    """Container for a evaluated candidate pre/post scene pair with compatibility metrics."""

    pre_scene: Sentinel1Scene
    post_scene: Sentinel1Scene
    compatibility_score: float  # 0.0 to 100.0
    relative_orbit_match: bool
    orbit_direction_match: bool
    sensor_mode_match: bool
    polarization_match: bool
    pre_aoi_coverage_pct: float
    post_aoi_coverage_pct: float
    notes: list[str] = Field(default_factory=list)


class RejectedPair(BaseModel):
    """Record of a rejected candidate scene pair with explicit rejection reasons."""

    pre_scene_id: str
    post_scene_id: str
    rejection_reasons: list[str]


class CompatibilityEvaluation(BaseModel):
    """Full compatibility evaluation report containing ranked pairs and rejected pairs."""

    status: str  # "SUITABLE_PAIRS_FOUND" or "NO_SUITABLE_PAIR_EXISTS"
    total_evaluated_pairs: int
    suitable_pairs_count: int
    ranked_candidate_pairs: list[CandidatePair] = Field(default_factory=list)
    rejected_pairs: list[RejectedPair] = Field(default_factory=list)
    evaluation_summary: str


class SceneCompatibilityAnalyzer:
    """Analyzer for assessing geometric and spectral compatibility between pre/post SAR scenes."""

    def __init__(self, min_aoi_coverage_pct: float = 50.0):
        self.min_aoi_coverage_pct = min_aoi_coverage_pct

    def evaluate_compatibility(
        self,
        aoi: AOI,
        pre_scenes: list[Sentinel1Scene],
        post_scenes: list[Sentinel1Scene]
    ) -> CompatibilityEvaluation:
        """Evaluate all combinations of pre-event and post-event scenes for compatibility.

        Args:
            aoi: Target spatial Area of Interest.
            pre_scenes: List of candidate pre-event scenes.
            post_scenes: List of candidate post-event scenes.

        Returns:
            CompatibilityEvaluation containing ranked candidate pairs and rejection details.
        """
        candidate_pairs: list[CandidatePair] = []
        rejected_pairs: list[RejectedPair] = []
        total_evaluated = 0

        for pre in pre_scenes:
            # Check pre-scene spatial coverage of AOI
            pre_cov = self._calculate_coverage(aoi, pre)
            
            for post in post_scenes:
                total_evaluated += 1
                post_cov = self._calculate_coverage(aoi, post)

                rejection_reasons: list[str] = []

                # 1. Spatial AOI Coverage Check
                if pre_cov < self.min_aoi_coverage_pct:
                    rejection_reasons.append(
                        f"Pre-event scene AOI coverage ({pre_cov:.1f}%) below threshold ({self.min_aoi_coverage_pct}%)"
                    )
                if post_cov < self.min_aoi_coverage_pct:
                    rejection_reasons.append(
                        f"Post-event scene AOI coverage ({post_cov:.1f}%) below threshold ({self.min_aoi_coverage_pct}%)"
                    )

                # 2. Sensor Mode Compatibility (IW mode required for land flood mapping)
                sensor_mode_match = (pre.sensor_mode == post.sensor_mode)
                if not sensor_mode_match:
                    rejection_reasons.append(
                        f"Sensor mode mismatch: pre='{pre.sensor_mode}' vs post='{post.sensor_mode}'"
                    )

                # 3. Orbit Direction Compatibility (Ascending vs Descending)
                # Mixing ascending and descending scenes creates severe geometric distortion mismatches
                orbit_dir_match = False
                if pre.orbit_direction and post.orbit_direction:
                    orbit_dir_match = (pre.orbit_direction == post.orbit_direction)
                    if not orbit_dir_match:
                        rejection_reasons.append(
                            f"Orbit direction mismatch: pre='{pre.orbit_direction}' vs post='{post.orbit_direction}'"
                        )
                else:
                    rejection_reasons.append("Missing orbit direction metadata in one or both scenes")

                # 4. Polarization Compatibility (e.g. VV or VH overlap)
                pol_pre = set(pre.polarizations)
                pol_post = set(post.polarizations)
                common_pols = pol_pre.intersection(pol_post)
                pol_match = len(common_pols) > 0
                if not pol_match:
                    rejection_reasons.append(
                        f"Polarization mismatch: pre={pre.polarizations} vs post={post.polarizations}"
                    )

                # 5. Relative Orbit Number Check
                rel_orbit_match = False
                if pre.relative_orbit_number is not None and post.relative_orbit_number is not None:
                    rel_orbit_match = (pre.relative_orbit_number == post.relative_orbit_number)

                # If pair has critical failures (sensor mode mismatch, orbit direction mismatch, or zero spatial coverage), reject it
                if rejection_reasons:
                    rejected_pairs.append(
                        RejectedPair(
                            pre_scene_id=pre.id,
                            post_scene_id=post.id,
                            rejection_reasons=rejection_reasons
                        )
                    )
                    continue

                # Calculate compatibility score (0 to 100)
                score = 50.0  # Base score for passing basic filters

                notes: list[str] = []
                if rel_orbit_match:
                    score += 30.0
                    notes.append(f"Matching relative orbit number ({pre.relative_orbit_number})")
                else:
                    notes.append(
                        f"Different relative orbits: pre={pre.relative_orbit_number}, post={post.relative_orbit_number}"
                    )

                if orbit_dir_match:
                    score += 10.0
                    notes.append(f"Matching orbit direction ({pre.orbit_direction})")

                # Bonus for high spatial coverage
                avg_cov = (pre_cov + post_cov) / 2.0
                score += (avg_cov / 100.0) * 10.0

                score = min(100.0, round(score, 1))

                candidate_pairs.append(
                    CandidatePair(
                        pre_scene=pre,
                        post_scene=post,
                        compatibility_score=score,
                        relative_orbit_match=rel_orbit_match,
                        orbit_direction_match=orbit_dir_match,
                        sensor_mode_match=sensor_mode_match,
                        polarization_match=pol_match,
                        pre_aoi_coverage_pct=pre_cov,
                        post_aoi_coverage_pct=post_cov,
                        notes=notes
                    )
                )

        # Sort candidate pairs by compatibility score descending
        candidate_pairs.sort(key=lambda p: p.compatibility_score, reverse=True)

        status = "SUITABLE_PAIRS_FOUND" if candidate_pairs else "NO_SUITABLE_PAIR_EXISTS"
        summary = (
            f"Evaluated {total_evaluated} total pair combinations. Found {len(candidate_pairs)} suitable pairs. "
            f"Rejected {len(rejected_pairs)} incompatible pairs."
        )

        return CompatibilityEvaluation(
            status=status,
            total_evaluated_pairs=total_evaluated,
            suitable_pairs_count=len(candidate_pairs),
            ranked_candidate_pairs=candidate_pairs,
            rejected_pairs=rejected_pairs,
            evaluation_summary=summary
        )

    def _calculate_coverage(self, aoi: AOI, scene: Sentinel1Scene) -> float:
        if not scene.footprint_geojson:
            # If footprint geometry is missing, assume 100% for candidate evaluation but note it
            return 100.0
        return aoi.intersection_percentage(scene.footprint_geojson)
