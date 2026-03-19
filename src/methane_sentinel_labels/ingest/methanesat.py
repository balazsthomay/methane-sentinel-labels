"""MethaneSAT L3 ingestion and plume mask generation."""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from scipy import ndimage

from methane_sentinel_labels.config import PipelineConfig
from methane_sentinel_labels.models import MethaneSATScene, PlumeMask

logger = logging.getLogger(__name__)

# Regex to parse scene filenames
_FILENAME_RE = re.compile(
    r"MSAT_L3_45m_COG_GEE_"
    r"(?P<collection_id>[^_]+)_"
    r"(?P<processing_id>[^_]+)_"
    r"(?P<version>[^_]+)_"
    r"(?P<start>\d{8}T\d{6})Z_"
    r"(?P<end>\d{6})Z"
    r"\.tif$"
)


def list_l3_scenes(cfg: PipelineConfig) -> list[str]:
    """List all L3 COG paths in the GCS bucket."""
    from google.cloud import storage

    client = storage.Client(project=cfg.msat_gcs_project)
    bucket = client.bucket(cfg.msat_gcs_bucket)
    blobs = client.list_blobs(bucket, prefix="cog_gee/")
    paths = [b.name for b in blobs if b.name.endswith(".tif")]
    logger.info("Found %d L3 scenes in gs://%s", len(paths), cfg.msat_gcs_bucket)
    return paths


def download_scene(gcs_path: str, cfg: PipelineConfig) -> Path:
    """Download a scene from GCS to local cache. Skips if already present."""
    local_path = cfg.msat_local_cache / gcs_path
    if local_path.exists():
        logger.debug("Cache hit: %s", local_path)
        return local_path

    from google.cloud import storage

    local_path.parent.mkdir(parents=True, exist_ok=True)
    client = storage.Client(project=cfg.msat_gcs_project)
    bucket = client.bucket(cfg.msat_gcs_bucket)
    blob = bucket.blob(gcs_path)
    blob.download_to_filename(str(local_path))
    logger.info("Downloaded %s (%.1f MB)", gcs_path, local_path.stat().st_size / 1e6)
    return local_path


def parse_scene(local_path: Path) -> MethaneSATScene:
    """Extract metadata and XCH4 stats from a downloaded L3 COG."""
    with rasterio.open(local_path) as ds:
        tags = ds.tags()
        bounds = ds.bounds
        xch4 = ds.read(1)

    valid = xch4[~np.isnan(xch4)]
    if valid.size == 0:
        raise ValueError(f"No valid XCH4 pixels in {local_path}")

    # Parse acquisition time from tags
    time_str = tags.get("time_coverage_start", "")
    if time_str:
        acq_time = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    else:
        acq_time = _parse_time_from_filename(local_path.name)

    scene_id = tags.get("collection_id", local_path.stem)
    target_id = tags.get("target_id", "")

    # Normalize bbox: ensure (lon_min, lat_min, lon_max, lat_max) order
    lon_min = min(bounds.left, bounds.right)
    lon_max = max(bounds.left, bounds.right)
    lat_min = min(bounds.bottom, bounds.top)
    lat_max = max(bounds.bottom, bounds.top)

    return MethaneSATScene(
        scene_id=scene_id,
        gcs_path=str(local_path),
        local_path=str(local_path),
        acquisition_time=acq_time,
        bbox=(lon_min, lat_min, lon_max, lat_max),
        crs=str(ds.crs) if hasattr(ds, "crs") else "EPSG:4326",
        resolution_m=46.4,
        xch4_median_ppb=float(np.median(valid)),
        target_id=target_id,
    )


def generate_plume_mask(
    scene: MethaneSATScene,
    cfg: PipelineConfig,
) -> PlumeMask | None:
    """Generate a binary plume mask from XCH4 anomaly thresholding.

    Returns None if the mask has fewer plume pixels than the minimum.
    """
    with rasterio.open(scene.local_path) as ds:
        xch4 = ds.read(1)
        transform = ds.transform
        crs = ds.crs

    valid_mask = ~np.isnan(xch4)
    total_valid = int(valid_mask.sum())
    if total_valid == 0:
        return None

    # Compute anomaly: subtract scene median
    median_val = np.nanmedian(xch4)
    anomaly = np.where(valid_mask, xch4 - median_val, 0.0)

    # Threshold
    binary = (anomaly > cfg.msat_plume_threshold_ppb).astype(np.uint8)

    # Morphological opening to remove isolated noise pixels
    if cfg.msat_morpho_kernel_size > 1:
        kernel = np.ones(
            (cfg.msat_morpho_kernel_size, cfg.msat_morpho_kernel_size),
            dtype=np.uint8,
        )
        binary = ndimage.binary_opening(binary, structure=kernel).astype(np.uint8)

    plume_count = int(binary.sum())
    if plume_count < cfg.msat_min_plume_pixels:
        logger.debug(
            "Scene %s: only %d plume pixels (< %d threshold), skipping",
            scene.scene_id,
            plume_count,
            cfg.msat_min_plume_pixels,
        )
        return None

    # Save mask as 1-band uint8 GeoTIFF
    mask_dir = Path(scene.local_path).parent / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_filename = f"{scene.scene_id}_mask_t{cfg.msat_plume_threshold_ppb:.0f}ppb.tif"
    mask_path = mask_dir / mask_filename

    height, width = binary.shape
    with rasterio.open(
        mask_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
        nodata=255,
    ) as dst:
        dst.write(binary, 1)
        dst.set_band_description(1, "plume_mask")
        dst.update_tags(
            scene_id=scene.scene_id,
            threshold_ppb=str(cfg.msat_plume_threshold_ppb),
            anomaly_method=cfg.msat_anomaly_method,
            plume_pixel_count=str(plume_count),
        )

    plume_fraction = plume_count / total_valid

    logger.info(
        "Scene %s: %d plume pixels (%.3f%% of valid), mask saved to %s",
        scene.scene_id,
        plume_count,
        plume_fraction * 100,
        mask_path,
    )

    return PlumeMask(
        scene_id=scene.scene_id,
        mask_path=str(mask_path),
        threshold_ppb=cfg.msat_plume_threshold_ppb,
        anomaly_method=cfg.msat_anomaly_method,
        plume_pixel_count=plume_count,
        total_valid_pixels=total_valid,
        plume_fraction=plume_fraction,
        bbox=scene.bbox,
        crs=scene.crs,
    )


def ingest_methanesat(
    cfg: PipelineConfig,
) -> tuple[list[MethaneSATScene], list[PlumeMask]]:
    """Full MethaneSAT ingestion pipeline: list → download → parse → mask."""
    gcs_paths = list_l3_scenes(cfg)
    if cfg.limit is not None:
        gcs_paths = gcs_paths[: cfg.limit]

    scenes: list[MethaneSATScene] = []
    masks: list[PlumeMask] = []

    for gcs_path in gcs_paths:
        try:
            local_path = download_scene(gcs_path, cfg)
            scene = parse_scene(local_path)
            scenes.append(scene)

            mask = generate_plume_mask(scene, cfg)
            if mask is not None:
                masks.append(mask)
        except Exception:
            logger.exception("Failed to process %s", gcs_path)

    logger.info(
        "MethaneSAT ingestion: %d scenes, %d usable plume masks",
        len(scenes),
        len(masks),
    )
    return scenes, masks


def _parse_time_from_filename(filename: str) -> datetime:
    """Parse acquisition time from MSAT filename as fallback."""
    match = _FILENAME_RE.search(filename)
    if match:
        start_str = match.group("start")
        return datetime.strptime(start_str, "%Y%m%dT%H%M%S").replace(
            tzinfo=timezone.utc
        )
    raise ValueError(f"Cannot parse time from filename: {filename}")
