"""Component D: Dataset assembly and quality metrics."""

import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from methane_sentinel_labels.models import PatchRecord, TrainingPatch

logger = logging.getLogger(__name__)


def assemble_dataset(records: list[PatchRecord] | list[TrainingPatch], output_dir: Path) -> None:
    """Assemble the final dataset: manifest + summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.parquet"
    _write_manifest(records, manifest_path)

    summary = _compute_summary(records)
    summary_path = output_dir / "summary.json"
    _write_summary(summary, summary_path)

    logger.info(
        "Dataset assembled: %d patches, manifest at %s", len(records), manifest_path
    )


def _write_manifest(records: list[PatchRecord] | list[TrainingPatch], path: Path) -> None:
    """Write patch records to a Parquet manifest."""
    rows = []
    for r in records:
        d = asdict(r)
        # Convert bbox tuple to string for Parquet compatibility
        d["bbox"] = str(d["bbox"])
        rows.append(d)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    logger.info("Manifest written: %d records to %s", len(records), path)


def _compute_summary(records: list[PatchRecord] | list[TrainingPatch]) -> dict:
    """Compute aggregate summary statistics."""
    if not records:
        return {"total_patches": 0}

    time_deltas = [r.time_delta_hours for r in records]
    cloud_fracs = [r.cloud_free_fraction for r in records]
    emission_rates = [
        r.emission_rate_kg_hr for r in records if r.emission_rate_kg_hr is not None
    ]

    summary: dict = {
        "total_patches": len(records),
        "median_time_delta_hours": float(pd.Series(time_deltas).median()),
        "mean_time_delta_hours": float(pd.Series(time_deltas).mean()),
        "min_time_delta_hours": min(time_deltas),
        "max_time_delta_hours": max(time_deltas),
        "mean_cloud_free_fraction": float(pd.Series(cloud_fracs).mean()),
        "patches_with_emission_rate": len(emission_rates),
    }

    if emission_rates:
        summary["min_emission_rate_kg_hr"] = min(emission_rates)
        summary["max_emission_rate_kg_hr"] = max(emission_rates)
        summary["median_emission_rate_kg_hr"] = float(
            pd.Series(emission_rates).median()
        )

    # Unique detections and scenes
    summary["unique_detections"] = len({r.detection_source_id for r in records})
    summary["unique_scenes"] = len({r.scene_id for r in records})

    return summary


def _write_summary(summary: dict, path: Path) -> None:
    """Write summary stats to JSON."""
    path.write_text(json.dumps(summary, indent=2))
    logger.info("Summary written to %s", path)
