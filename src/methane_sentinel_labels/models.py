"""Domain models for the methane sentinel labels pipeline."""

from dataclasses import dataclass
from datetime import datetime


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
