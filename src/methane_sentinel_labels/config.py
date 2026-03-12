"""Pipeline configuration."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the methane sentinel labels pipeline.

    Secrets come from environment variables. Everything else from CLI flags.
    """

    # Carbon Mapper credentials (from env)
    cm_email: str = field(default_factory=lambda: os.environ.get("CM_EMAIL", ""))
    cm_password: str = field(
        default_factory=lambda: os.environ.get("CM_PASSWORD", "")
    )

    # Carbon Mapper API
    cm_api_base: str = "https://api.carbonmapper.org/api/v1"

    # Sentinel-2 STAC
    stac_url: str = "https://earth-search.aws.element84.com/v1"
    stac_collection: str = "sentinel-2-l2a"

    # Matching parameters
    max_time_delta_hours: float = 120.0
    max_cloud_cover_pct: float = 30.0
    max_matches_per_detection: int = 3

    # Patch extraction
    patch_half_size_m: float = 2560.0
    target_resolution_m: float = 20.0
    min_cloud_free_fraction: float = 0.5
    bands: tuple[str, ...] = ("B02", "B03", "B04", "B8A", "B11", "B12", "SCL")

    # Output
    output_dir: Path = Path("output")

    # Limits
    limit: int | None = None
