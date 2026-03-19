"""Cross-sensor temporal matching: MethaneSAT → Sentinel-2."""

import logging
from datetime import datetime, timezone

from methane_sentinel_labels.config import PipelineConfig
from methane_sentinel_labels.matching.sentinel2 import (
    _BAND_TO_ASSET,
    query_stac_bbox,
)
from methane_sentinel_labels.models import MatchedPair, MethaneSATScene, PlumeMask

logger = logging.getLogger(__name__)


def find_sentinel2_matches(
    masks: list[PlumeMask],
    scenes: list[MethaneSATScene],
    cfg: PipelineConfig,
) -> list[MatchedPair]:
    """Find the best Sentinel-2 match for each MethaneSAT plume mask."""
    scene_by_id = {s.scene_id: s for s in scenes}
    pairs: list[MatchedPair] = []

    for mask in masks:
        scene = scene_by_id.get(mask.scene_id)
        if scene is None:
            logger.warning("No scene found for mask %s", mask.scene_id)
            continue

        items = query_stac_bbox(
            bbox=mask.bbox,
            time_center=scene.acquisition_time,
            cfg=cfg,
        )

        candidates: list[MatchedPair] = []
        for item in items:
            pair = _item_to_matched_pair(item, scene, mask, cfg)
            if pair is not None:
                candidates.append(pair)

        # Sort by time delta, take best
        candidates.sort(key=lambda p: p.time_delta_hours)
        if candidates:
            best = candidates[0]
            pairs.append(best)
            logger.info(
                "Scene %s → S2 %s (Δt=%.1fh, cloud=%.1f%%)",
                mask.scene_id,
                best.s2_scene_id,
                best.time_delta_hours,
                best.s2_cloud_cover_pct,
            )
        else:
            logger.info("Scene %s: no S2 match found", mask.scene_id)

    logger.info(
        "Cross-sensor matching: %d masks → %d matched pairs",
        len(masks),
        len(pairs),
    )
    return pairs


def _item_to_matched_pair(
    item,
    scene: MethaneSATScene,
    mask: PlumeMask,
    cfg: PipelineConfig,
) -> MatchedPair | None:
    """Convert a STAC item to a MatchedPair, or None if it fails filters."""
    cloud_cover = item.properties.get("eo:cloud_cover", 100.0)
    if cloud_cover > cfg.max_cloud_cover_pct:
        return None

    acq_time = item.datetime
    if acq_time.tzinfo is None:
        acq_time = acq_time.replace(tzinfo=timezone.utc)

    time_delta = abs((acq_time - scene.acquisition_time).total_seconds()) / 3600.0
    if time_delta > cfg.msat_max_time_delta_hours:
        return None

    # Compute spatial overlap
    item_bbox = item.bbox  # [lon_min, lat_min, lon_max, lat_max]
    overlap = _compute_bbox_overlap(mask.bbox, tuple(item_bbox))
    if overlap < cfg.msat_min_spatial_overlap:
        return None

    # Intersection bbox
    intersection = _intersect_bboxes(mask.bbox, tuple(item_bbox))

    # MGRS tile
    grid_code = item.properties.get("grid:code", "")
    mgrs_tile = grid_code.replace("MGRS-", "") if grid_code else ""

    # Band HREFs
    band_hrefs: dict[str, str] = {}
    for band in cfg.bands:
        asset_key = _BAND_TO_ASSET.get(band, band)
        if asset_key in item.assets:
            band_hrefs[band] = item.assets[asset_key].href

    return MatchedPair(
        msat_scene_id=scene.scene_id,
        s2_scene_id=item.id,
        msat_acquisition_time=scene.acquisition_time,
        s2_acquisition_time=acq_time,
        time_delta_hours=round(time_delta, 2),
        msat_mask_path=mask.mask_path,
        s2_band_hrefs=band_hrefs,
        s2_mgrs_tile=mgrs_tile,
        bbox=intersection,
        s2_cloud_cover_pct=cloud_cover,
    )


def _compute_bbox_overlap(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> float:
    """Compute the fraction of bbox_a covered by bbox_b (0 to 1)."""
    a_left, a_bottom, a_right, a_top = bbox_a
    b_left, b_bottom, b_right, b_top = bbox_b

    # Intersection
    i_left = max(a_left, b_left)
    i_bottom = max(a_bottom, b_bottom)
    i_right = min(a_right, b_right)
    i_top = min(a_top, b_top)

    if i_left >= i_right or i_bottom >= i_top:
        return 0.0

    intersection_area = (i_right - i_left) * (i_top - i_bottom)
    a_area = (a_right - a_left) * (a_top - a_bottom)

    if a_area == 0:
        return 0.0

    return intersection_area / a_area


def _intersect_bboxes(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Compute the intersection of two bounding boxes."""
    return (
        max(bbox_a[0], bbox_b[0]),
        max(bbox_a[1], bbox_b[1]),
        min(bbox_a[2], bbox_b[2]),
        min(bbox_a[3], bbox_b[3]),
    )
