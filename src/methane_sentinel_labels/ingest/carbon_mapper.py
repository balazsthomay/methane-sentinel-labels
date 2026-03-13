"""Component A: Carbon Mapper label ingestion."""

import logging
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import httpx
import pandas as pd

from methane_sentinel_labels.config import PipelineConfig
from methane_sentinel_labels.models import Detection

logger = logging.getLogger(__name__)

_PAGE_SIZE = 1000


def fetch_detections(
    cfg: PipelineConfig,
    *,
    client: httpx.Client | None = None,
) -> list[Detection]:
    """Fetch CH4 plume detections from the Carbon Mapper API."""
    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=30.0)

    try:
        token: str | None = None
        if cfg.cm_email and cfg.cm_password:
            token = _authenticate(client, cfg)
        detections = _paginate(client, cfg, token)
    finally:
        if own_client:
            client.close()

    # Deduplicate by source_id
    seen: set[str] = set()
    unique: list[Detection] = []
    for d in detections:
        if d.source_id not in seen:
            seen.add(d.source_id)
            unique.append(d)

    # Apply limit
    if cfg.limit is not None:
        unique = unique[: cfg.limit]

    logger.info("Fetched %d detections (%d unique)", len(detections), len(unique))
    return unique


def _authenticate(client: httpx.Client, cfg: PipelineConfig) -> str:
    """Obtain a JWT access token from the Carbon Mapper API."""
    url = f"{cfg.cm_api_base}/token/pair"
    resp = client.post(url, json={"email": cfg.cm_email, "password": cfg.cm_password})
    resp.raise_for_status()
    data = resp.json()
    return data["access"]


def _paginate(
    client: httpx.Client,
    cfg: PipelineConfig,
    token: str | None,
) -> list[Detection]:
    """Paginate through the annotated plumes endpoint."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    detections: list[Detection] = []
    offset = 0

    while True:
        params: dict[str, int | str] = {
            "limit": _PAGE_SIZE,
            "offset": offset,
            "plume_gas": "CH4",
        }
        url = f"{cfg.cm_api_base}/catalog/plumes/annotated"
        resp = client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        items = data.get("items", [])
        if not items:
            break

        for item in items:
            try:
                detections.append(_parse_plume(item))
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping malformed plume: %s", exc)

        offset += len(items)

        # If we got fewer than PAGE_SIZE, we're done
        if len(items) < _PAGE_SIZE:
            break

    return detections


def _parse_plume(item: dict) -> Detection:
    """Parse a single plume item from the Carbon Mapper API into a Detection."""
    coords = item["geometry_json"]["coordinates"]

    dt_str = item["scene_timestamp"]
    detection_time = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))

    emission_rate = item.get("emission_auto")
    emission_unc = item.get("emission_uncertainty_auto")

    return Detection(
        source_id=item["plume_id"],
        longitude=coords[0],
        latitude=coords[1],
        detection_time=detection_time,
        emission_rate_kg_hr=float(emission_rate) if emission_rate is not None else None,
        emission_uncertainty_kg_hr=(
            float(emission_unc) if emission_unc is not None else None
        ),
        sensor=item.get("instrument", "unknown"),
        provider="carbon_mapper",
    )


def save_detections(detections: list[Detection], path: Path) -> None:
    """Save detections to GeoParquet."""
    records = [
        {
            "source_id": d.source_id,
            "latitude": d.latitude,
            "longitude": d.longitude,
            "detection_time": d.detection_time.isoformat(),
            "emission_rate_kg_hr": d.emission_rate_kg_hr,
            "emission_uncertainty_kg_hr": d.emission_uncertainty_kg_hr,
            "sensor": d.sensor,
            "provider": d.provider,
        }
        for d in detections
    ]
    df = pd.DataFrame(records)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    gdf.to_parquet(path)
    logger.info("Saved %d detections to %s", len(detections), path)


def load_detections(path: Path) -> list[Detection]:
    """Load detections from GeoParquet."""
    gdf = gpd.read_parquet(path)
    detections: list[Detection] = []
    for _, row in gdf.iterrows():
        detections.append(
            Detection(
                source_id=row["source_id"],
                latitude=row["latitude"],
                longitude=row["longitude"],
                detection_time=datetime.fromisoformat(row["detection_time"]),
                emission_rate_kg_hr=(
                    float(row["emission_rate_kg_hr"])
                    if pd.notna(row["emission_rate_kg_hr"])
                    else None
                ),
                emission_uncertainty_kg_hr=(
                    float(row["emission_uncertainty_kg_hr"])
                    if pd.notna(row["emission_uncertainty_kg_hr"])
                    else None
                ),
                sensor=row["sensor"],
                provider=row["provider"],
            )
        )
    return detections
