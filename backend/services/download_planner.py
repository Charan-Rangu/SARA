"""Download Planning and Storage Budget Estimation Module for SARA.

Estimates storage requirements for selected satellite scenes prior to acquisition.
Enforces user-configured storage budgets, identifies missing size metadata, and emits clear storage warnings.

Important:
This module is for PLANNING ONLY and does NOT trigger automated image downloads.
Compressed Sentinel-1 SAFE product download sizes (~1.5 to 2.5 GB per scene) do NOT equal final uncompressed
processed raster sizes. Processed GeoTIFFs may require additional disk space after unzipping and calibration.
"""

from typing import Optional
from pydantic import BaseModel, Field

from backend.services.sentinel1_catalog import Sentinel1Scene


class ProductDownloadEstimate(BaseModel):
    """Storage estimate for an individual scene product."""

    id: str
    title: str
    download_url: Optional[str] = None
    size_bytes: Optional[int] = None
    size_mb: Optional[float] = None
    size_unknown: bool = False


class DownloadPlan(BaseModel):
    """Full download plan summary containing storage budget verification and warnings."""

    scenes_count: int
    products: list[ProductDownloadEstimate] = Field(default_factory=list)
    total_estimated_size_mb: float = 0.0
    total_estimated_size_gb: float = 0.0
    storage_budget_gb: float
    within_budget: bool
    unknown_size_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    plan_summary: str


class DownloadPlanner:
    """Planner for evaluating satellite product download sizes against disk storage budgets."""

    def __init__(self, default_storage_budget_gb: float = 10.0):
        self.default_storage_budget_gb = default_storage_budget_gb

    def create_download_plan(
        self,
        scenes: list[Sentinel1Scene],
        storage_budget_gb: Optional[float] = None
    ) -> DownloadPlan:
        """Calculate total estimated download sizes and evaluate against storage budget.

        Args:
            scenes: List of Sentinel1Scene products to be downloaded.
            storage_budget_gb: Configured disk storage limit in GB.

        Returns:
            DownloadPlan object containing estimated size, budget check, and warnings.
        """
        budget = storage_budget_gb if storage_budget_gb is not None else self.default_storage_budget_gb

        product_estimates: list[ProductDownloadEstimate] = []
        total_bytes = 0
        unknown_count = 0
        warnings: list[str] = []

        for scene in scenes:
            size_b = scene.size_bytes
            size_mb = None
            size_unknown = False

            if size_b is not None and size_b > 0:
                size_mb = round(size_b / (1024 * 1024), 2)
                total_bytes += size_b
            else:
                size_unknown = True
                unknown_count += 1
                # Standard Sentinel-1 GRD SAFE product estimate ~1.8 GB if metadata missing
                estimated_fallback_bytes = int(1.8 * 1024 * 1024 * 1024)
                total_bytes += estimated_fallback_bytes

            product_estimates.append(
                ProductDownloadEstimate(
                    id=scene.id,
                    title=scene.title,
                    download_url=scene.download_url,
                    size_bytes=size_b,
                    size_mb=size_mb,
                    size_unknown=size_unknown
                )
            )

        total_mb = round(total_bytes / (1024 * 1024), 2)
        total_gb = round(total_bytes / (1024 * 1024 * 1024), 2)

        within_budget = (total_gb <= budget)

        if unknown_count > 0:
            warnings.append(
                f"{unknown_count} scene(s) lack explicit size metadata from catalogue. "
                f"Estimated size uses default estimate (~1.8 GB per scene)."
            )

        if not within_budget:
            warnings.append(
                f"Total estimated download size ({total_gb:.2f} GB) exceeds configured storage budget ({budget:.2f} GB)."
            )

        summary = (
            f"Download plan generated for {len(scenes)} product(s). "
            f"Estimated download size: {total_gb:.2f} GB ({total_mb:.2f} MB). "
            f"Storage budget: {budget:.2f} GB. Within budget: {within_budget}."
        )

        return DownloadPlan(
            scenes_count=len(scenes),
            products=product_estimates,
            total_estimated_size_mb=total_mb,
            total_estimated_size_gb=total_gb,
            storage_budget_gb=budget,
            within_budget=within_budget,
            unknown_size_count=unknown_count,
            warnings=warnings,
            plan_summary=summary
        )
