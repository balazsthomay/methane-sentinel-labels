"""Component B: Sentinel-2 temporal matching via STAC."""

import logging
from datetime import datetime, timedelta, timezone

from pystac_client import Client as STACClient

from methane_sentinel_labels.config import PipelineConfig
from methane_sentinel_labels.models import Detection, SceneMatch

logger = logging.getLogger(__name__)


def find_matches(
    detections: list[Detection],
    cfg: PipelineConfig,
) -> list[SceneMatch]:
    """Find temporally close Sentinel-2 scenes for each detection."""
    all_matches: list[SceneMatch] = []

    for detection in detections:
        items = _query_stac(detection, cfg)
        matches: list[SceneMatch] = []

        for item in items:
            match = _item_to_scene_match(item, detection, cfg)
            if match is not None:
                matches.append(match)

        # Sort by time delta ascending, take top N
        matches.sort(key=lambda m: m.time_delta_hours)
        matches = matches[: cfg.max_matches_per_detection]
        all_matches.extend(matches)

        logger.info(
            "Detection %s: %d matches found", detection.source_id, len(matches)
        )

    logger.info("Total matches: %d for %d detections", len(all_matches), len(detections))
    return all_matches


def _query_stac(detection: Detection, cfg: PipelineConfig) -> list:
    """Query the STAC catalog for Sentinel-2 scenes near a detection."""
    catalog = STACClient.open(cfg.stac_url)

    delta = timedelta(hours=cfg.max_time_delta_hours)
    start = detection.detection_time - delta
    end = detection.detection_time + delta

    search = catalog.search(
        collections=[cfg.stac_collection],
        intersects={
            "type": "Point",
            "coordinates": [detection.longitude, detection.latitude],
        },
        datetime=f"{start.isoformat()}/{end.isoformat()}",
        query={"eo:cloud_cover": {"lte": cfg.max_cloud_cover_pct}},
    )

    return list(search.items())


def _item_to_scene_match(
    item,
    detection: Detection,
    cfg: PipelineConfig,
) -> SceneMatch | None:
    """Convert a STAC item to a SceneMatch, or None if it doesn't pass filters."""
    cloud_cover = item.properties.get("eo:cloud_cover", 100.0)
    if cloud_cover > cfg.max_cloud_cover_pct:
        return None

    acq_time = item.datetime
    if acq_time.tzinfo is None:
        acq_time = acq_time.replace(tzinfo=timezone.utc)

    time_delta = abs((acq_time - detection.detection_time).total_seconds()) / 3600.0
    if time_delta > cfg.max_time_delta_hours:
        return None

    # Extract MGRS tile from grid:code property (format: "MGRS-13SDA")
    grid_code = item.properties.get("grid:code", "")
    mgrs_tile = grid_code.replace("MGRS-", "") if grid_code else ""

    # Extract band HREFs
    band_hrefs: dict[str, str] = {}
    for band in cfg.bands:
        if band in item.assets:
            band_hrefs[band] = item.assets[band].href

    return SceneMatch(
        detection_source_id=detection.source_id,
        scene_id=item.id,
        acquisition_time=acq_time,
        time_delta_hours=round(time_delta, 2),
        cloud_cover_pct=cloud_cover,
        mgrs_tile=mgrs_tile,
        band_hrefs=band_hrefs,
    )
