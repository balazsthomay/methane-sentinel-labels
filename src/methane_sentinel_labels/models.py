"""Domain models for the methane sentinel labels pipeline."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Detection:
    """A confirmed methane point-source detection from a label provider."""

    source_id: str
    latitude: float
    longitude: float
    detection_time: datetime
    emission_rate_kg_hr: float | None
    emission_uncertainty_kg_hr: float | None
    sensor: str
    provider: str


@dataclass(frozen=True)
class SceneMatch:
    """A Sentinel-2 scene matched to a detection by temporal proximity."""

    detection_source_id: str
    scene_id: str
    acquisition_time: datetime
    time_delta_hours: float
    cloud_cover_pct: float
    mgrs_tile: str
    band_hrefs: dict[str, str]


@dataclass(frozen=True)
class PatchRecord:
    """Metadata for an extracted GeoTIFF patch."""

    detection_source_id: str
    scene_id: str
    patch_path: str
    latitude: float
    longitude: float
    emission_rate_kg_hr: float | None
    time_delta_hours: float
    cloud_free_fraction: float
    crs: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class MethaneSATScene:
    """Metadata for a MethaneSAT L3 concentration scene."""

    scene_id: str
    gcs_path: str
    local_path: str
    acquisition_time: datetime
    bbox: tuple[float, float, float, float]  # (lon_min, lat_min, lon_max, lat_max)
    crs: str
    resolution_m: float
    xch4_median_ppb: float
    target_id: str


@dataclass(frozen=True)
class PlumeMask:
    """A binary plume mask derived from MethaneSAT L3 anomaly thresholding."""

    scene_id: str
    mask_path: str
    threshold_ppb: float
    anomaly_method: str  # "median_subtract"
    plume_pixel_count: int
    total_valid_pixels: int
    plume_fraction: float
    bbox: tuple[float, float, float, float]
    crs: str


@dataclass(frozen=True)
class MatchedPair:
    """A MethaneSAT scene matched to a Sentinel-2 scene."""

    msat_scene_id: str
    s2_scene_id: str
    msat_acquisition_time: datetime
    s2_acquisition_time: datetime
    time_delta_hours: float
    msat_mask_path: str
    s2_band_hrefs: dict[str, str]
    s2_mgrs_tile: str
    bbox: tuple[float, float, float, float]  # intersection bbox
    s2_cloud_cover_pct: float


@dataclass(frozen=True)
class TrainingPatch:
    """Metadata for an extracted training patch with MethaneSAT-derived label."""

    msat_scene_id: str
    s2_scene_id: str
    patch_path: str
    bbox: tuple[float, float, float, float]
    crs: str
    time_delta_hours: float
    cloud_free_fraction: float
    plume_pixel_count: int
    plume_fraction: float
    band_names: tuple[str, ...]
