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

    # MethaneSAT
    msat_gcs_bucket: str = "msat-prod-data-public-methanesat-level3"
    msat_gcs_project: str = "methanesat-public"
    msat_local_cache: Path = field(default_factory=lambda: Path("output/msat_cache"))
    msat_anomaly_method: str = "median_subtract"
    msat_plume_threshold_ppb: float = 50.0
    msat_min_plume_pixels: int = 10
    msat_morpho_kernel_size: int = 3

    # Cross-sensor matching
    msat_max_time_delta_hours: float = 72.0
    msat_min_spatial_overlap: float = 0.1

    # Enhancement products
    varon_reference_method: str = "spatial"

    # Training
    training_input_bands: tuple[str, ...] = ("varon", "B11", "B12", "B8A")
    training_patch_size_px: int = 256

    # Output
    output_dir: Path = Path("output")

    # Limits
    limit: int | None = None
