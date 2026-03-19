"""Component C: Patch extraction from Sentinel-2 COGs."""

import logging
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds as window_from_bounds

from methane_sentinel_labels.config import PipelineConfig
from methane_sentinel_labels.extraction.enhancement import compute_varon_ratio
from methane_sentinel_labels.models import MatchedPair, PatchRecord, SceneMatch, TrainingPatch

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


def extract_training_patch(
    pair: MatchedPair,
    cfg: PipelineConfig,
) -> TrainingPatch | None:
    """Extract a multi-band training patch with Varon ratio and reprojected mask.

    Output bands: [B02, B03, B04, B8A, B11, B12, varon, mask]
    """
    patches_dir = cfg.output_dir / "training_patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{pair.msat_scene_id}_{pair.s2_scene_id}.tif"
    patch_path = patches_dir / filename

    if patch_path.exists():
        logger.debug("Reusing existing training patch: %s", filename)
        with rasterio.open(patch_path) as ds:
            tags = ds.tags()
        return TrainingPatch(
            msat_scene_id=pair.msat_scene_id,
            s2_scene_id=pair.s2_scene_id,
            patch_path=str(patch_path.relative_to(cfg.output_dir)),
            bbox=pair.bbox,
            crs=tags.get("crs", ""),
            time_delta_hours=pair.time_delta_hours,
            cloud_free_fraction=float(tags.get("cloud_free_fraction", 0)),
            plume_pixel_count=int(tags.get("plume_pixel_count", 0)),
            plume_fraction=float(tags.get("plume_fraction", 0)),
            band_names=tuple(tags.get("band_names", "").split(",")),
        )

    try:
        # Determine UTM CRS from center of intersection bbox
        lon_center = (pair.bbox[0] + pair.bbox[2]) / 2
        lat_center = (pair.bbox[1] + pair.bbox[3]) / 2
        crs, bounds = _compute_utm_bounds(
            lon_center, lat_center, half_size_m=cfg.patch_half_size_m
        )

        target_size = int(2 * cfg.patch_half_size_m / cfg.target_resolution_m)

        # Read S2 bands
        with rasterio.Env(
            GDAL_HTTP_TIMEOUT=120,
            GDAL_HTTP_CONNECTTIMEOUT=30,
        ):
            band_data: dict[str, np.ndarray] = {}
            for band_name in ("B02", "B03", "B04", "B8A", "B11", "B12", "SCL"):
                href = pair.s2_band_hrefs.get(band_name)
                if href is None:
                    continue
                data = _read_band_window(
                    href=href, crs=crs, bounds=bounds, target_res=cfg.target_resolution_m
                )
                if data is not None:
                    band_data[band_name] = data

        if "B11" not in band_data or "B12" not in band_data:
            logger.warning("Missing SWIR bands for %s", pair.s2_scene_id)
            return None

        # Check cloud cover
        cloud_free = 1.0
        if "SCL" in band_data:
            scl = band_data["SCL"].astype(np.uint8)
            cloud_free = _compute_cloud_free_fraction(scl)
        if cloud_free < cfg.min_cloud_free_fraction:
            logger.info("Skipping %s: cloud-free %.2f", pair.s2_scene_id, cloud_free)
            return None

        # Compute Varon ratio
        varon = compute_varon_ratio(
            band_data["B11"], band_data["B12"],
            reference_method=cfg.varon_reference_method,
        )
        band_data["varon"] = varon

        # Reproject MethaneSAT mask to S2 grid
        mask = _reproject_mask_to_s2_grid(
            pair.msat_mask_path, crs, bounds, (target_size, target_size)
        )
        band_data["mask"] = mask

        # Write output (exclude SCL)
        output_band_names = ["B02", "B03", "B04", "B8A", "B11", "B12", "varon", "mask"]
        output_bands = {}
        for name in output_band_names:
            if name in band_data:
                output_bands[name] = band_data[name]

        plume_count = int((mask > 0).sum())
        total_pixels = mask.size
        plume_fraction = plume_count / total_pixels if total_pixels > 0 else 0.0

        _write_patch_geotiff(
            output_bands, crs, bounds, patch_path,
            cloud_free_fraction=round(cloud_free, 4),
        )

        # Write extra metadata tags
        with rasterio.open(patch_path, "r+") as ds:
            ds.update_tags(
                crs=crs,
                plume_pixel_count=str(plume_count),
                plume_fraction=str(round(plume_fraction, 6)),
                band_names=",".join(output_bands.keys()),
                msat_scene_id=pair.msat_scene_id,
                s2_scene_id=pair.s2_scene_id,
            )

        return TrainingPatch(
            msat_scene_id=pair.msat_scene_id,
            s2_scene_id=pair.s2_scene_id,
            patch_path=str(patch_path.relative_to(cfg.output_dir)),
            bbox=pair.bbox,
            crs=crs,
            time_delta_hours=pair.time_delta_hours,
            cloud_free_fraction=round(cloud_free, 4),
            plume_pixel_count=plume_count,
            plume_fraction=round(plume_fraction, 6),
            band_names=tuple(output_bands.keys()),
        )
    except Exception:
        logger.exception("Error extracting training patch for %s", pair.s2_scene_id)
        return None


def _reproject_mask_to_s2_grid(
    mask_path: str,
    target_crs: str,
    target_bounds: tuple[float, float, float, float],
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Reproject a MethaneSAT binary mask to the Sentinel-2 UTM grid.

    Uses nearest-neighbor resampling to preserve binary values.
    """
    left, bottom, right, top = target_bounds
    height, width = target_shape
    dst_transform = from_bounds(left, bottom, right, top, width, height)

    dst_data = np.zeros((height, width), dtype=np.uint8)

    with rasterio.open(mask_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest,
        )

    return dst_data


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
