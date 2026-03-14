"""Component C: Patch extraction from Sentinel-2 COGs."""

import logging
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds

from methane_sentinel_labels.config import PipelineConfig
from methane_sentinel_labels.models import PatchRecord, SceneMatch

logger = logging.getLogger(__name__)

# SCL values 4-7 are considered clear (vegetation, bare soil, water, unclassified)
_SCL_CLEAR_VALUES = {4, 5, 6, 7}


def extract_patches(
    matches: list[SceneMatch],
    cfg: PipelineConfig,
    *,
    latitude: float,
    longitude: float,
) -> list[PatchRecord]:
    """Extract GeoTIFF patches for each scene match."""
    patches_dir = cfg.output_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    records: list[PatchRecord] = []
    with rasterio.Env(GDAL_HTTP_TIMEOUT=120, GDAL_HTTP_CONNECTTIMEOUT=30):
        for match in matches:
            record = _extract_single_patch(match, cfg, latitude, longitude, patches_dir)
            if record is not None:
                records.append(record)

    logger.info("Extracted %d patches from %d matches", len(records), len(matches))
    return records


def _extract_single_patch(
    match: SceneMatch,
    cfg: PipelineConfig,
    latitude: float,
    longitude: float,
    patches_dir: Path,
) -> PatchRecord | None:
    """Extract a single patch, returning None if it fails filters or errors."""
    try:
        crs, bounds = _compute_utm_bounds(
            longitude, latitude, half_size_m=cfg.patch_half_size_m
        )

        filename = f"{match.detection_source_id}_{match.scene_id}.tif"
        patch_path = patches_dir / filename

        # Resume: reuse existing patch instead of re-downloading
        if patch_path.exists():
            logger.debug("Reusing existing patch: %s", filename)
            with rasterio.open(patch_path) as ds:
                tags = ds.tags()
            cloud_free = float(
                tags.get("cloud_free_fraction", cfg.min_cloud_free_fraction)
            )
            return PatchRecord(
                detection_source_id=match.detection_source_id,
                scene_id=match.scene_id,
                patch_path=str(patch_path.relative_to(cfg.output_dir)),
                latitude=latitude,
                longitude=longitude,
                emission_rate_kg_hr=None,
                time_delta_hours=match.time_delta_hours,
                cloud_free_fraction=cloud_free,
                crs=crs,
                bbox=bounds,
            )

        # Read all bands
        band_data: dict[str, np.ndarray] = {}
        for band_name in cfg.bands:
            href = match.band_hrefs.get(band_name)
            if href is None:
                logger.warning("Band %s not available for %s", band_name, match.scene_id)
                continue
            data = _read_band_window(
                href=href, crs=crs, bounds=bounds, target_res=cfg.target_resolution_m
            )
            if data is not None:
                band_data[band_name] = data

        if not band_data:
            logger.warning("No bands read for %s", match.scene_id)
            return None

        # Compute cloud-free fraction from SCL band
        cloud_free = 1.0
        if "SCL" in band_data:
            scl = band_data["SCL"].astype(np.uint8)
            cloud_free = _compute_cloud_free_fraction(scl)

        if cloud_free < cfg.min_cloud_free_fraction:
            logger.info(
                "Skipping %s: cloud-free %.2f < threshold %.2f",
                match.scene_id,
                cloud_free,
                cfg.min_cloud_free_fraction,
            )
            return None

        # Write the non-SCL bands to a GeoTIFF
        output_bands = {
            k: v for k, v in band_data.items() if k != "SCL"
        }
        if not output_bands:
            return None

        _write_patch_geotiff(
            output_bands, crs, bounds, patch_path,
            cloud_free_fraction=round(cloud_free, 4),
        )

        return PatchRecord(
            detection_source_id=match.detection_source_id,
            scene_id=match.scene_id,
            patch_path=str(patch_path.relative_to(cfg.output_dir)),
            latitude=latitude,
            longitude=longitude,
            emission_rate_kg_hr=None,  # Filled in by caller if needed
            time_delta_hours=match.time_delta_hours,
            cloud_free_fraction=round(cloud_free, 4),
            crs=crs,
            bbox=bounds,
        )
    except Exception:
        logger.exception("Error extracting patch for %s", match.scene_id)
        return None


def _compute_utm_bounds(
    longitude: float,
    latitude: float,
    half_size_m: float,
) -> tuple[str, tuple[float, float, float, float]]:
    """Compute UTM CRS and bounding box centered on lon/lat."""
    # Determine UTM zone
    zone = int((longitude + 180) / 6) + 1
    hemisphere = "north" if latitude >= 0 else "south"
    epsg = 32600 + zone if hemisphere == "north" else 32700 + zone
    crs = f"EPSG:{epsg}"

    # Transform lon/lat to UTM
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(longitude, latitude)

    bounds = (
        x - half_size_m,
        y - half_size_m,
        x + half_size_m,
        y + half_size_m,
    )
    return crs, bounds


def _read_band_window(
    *,
    href: str,
    crs: str,
    bounds: tuple[float, float, float, float],
    target_res: float,
) -> np.ndarray | None:
    """Read a windowed region from a COG, resampling to target resolution."""
    left, bottom, right, top = bounds
    width = int((right - left) / target_res)
    height = int((top - bottom) / target_res)

    with rasterio.open(href) as ds:
        # Compute the window in the dataset's pixel coordinates
        window = window_from_bounds(left, bottom, right, top, ds.transform)
        # boundless=True handles detections near MGRS tile edges where the
        # window extends beyond raster bounds — fills out-of-bounds pixels with 0
        data = ds.read(
            1,
            window=window,
            out_shape=(height, width),
            resampling=rasterio.enums.Resampling.bilinear,
            boundless=True,
            fill_value=0,
        )
    return data


def _compute_cloud_free_fraction(scl: np.ndarray) -> float:
    """Compute fraction of valid pixels that are cloud-free using SCL band."""
    # Exclude nodata (0) from consideration
    valid_mask = scl > 0
    valid_count = valid_mask.sum()
    if valid_count == 0:
        return 0.0

    clear_mask = np.isin(scl, list(_SCL_CLEAR_VALUES)) & valid_mask
    return float(clear_mask.sum()) / float(valid_count)


def _write_patch_geotiff(
    bands: dict[str, np.ndarray],
    crs: str,
    bounds: tuple[float, float, float, float],
    path: Path,
    *,
    cloud_free_fraction: float | None = None,
) -> None:
    """Write a multi-band GeoTIFF patch."""
    band_names = sorted(bands.keys())
    first = bands[band_names[0]]
    height, width = first.shape
    left, bottom, right, top = bounds
    transform = from_bounds(left, bottom, right, top, width, height)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=len(band_names),
        dtype=first.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        for i, name in enumerate(band_names, 1):
            dst.write(bands[name], i)
            dst.set_band_description(i, name)
        if cloud_free_fraction is not None:
            dst.update_tags(cloud_free_fraction=str(cloud_free_fraction))
