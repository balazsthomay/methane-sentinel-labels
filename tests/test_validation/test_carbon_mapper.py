"""Tests for Carbon Mapper validation."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_bounds

from methane_sentinel_labels.models import PatchRecord
from methane_sentinel_labels.validation.carbon_mapper import (
    compute_validation_metrics,
    run_inference_on_cm_patches,
)


def _create_cm_patch(path: Path, *, height: int = 64, width: int = 64) -> None:
    """Create a synthetic CM-style 6-band patch."""
    rng = np.random.default_rng(42)
    band_names = ["B02", "B03", "B04", "B8A", "B11", "B12"]
    data = (rng.random((6, height, width)) * 1000 + 500).astype(np.float32)

    transform = from_bounds(540000, 3570000, 541280, 3571280, width, height)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=height, width=width, count=6, dtype="float32",
        crs="EPSG:32613", transform=transform,
    ) as ds:
        for i in range(6):
            ds.write(data[i], i + 1)
            ds.set_band_description(i + 1, band_names[i])


@pytest.fixture
def cm_records_and_dir(tmp_path: Path) -> tuple[list[PatchRecord], Path]:
    records = []
    for i in range(3):
        rel_path = f"patches/cm_{i}.tif"
        full_path = tmp_path / rel_path
        _create_cm_patch(full_path)
        records.append(
            PatchRecord(
                detection_source_id=f"CM-{i:03d}",
                scene_id=f"S2A_{i:03d}",
                patch_path=rel_path,
                latitude=31.5 + i * 0.1,
                longitude=-103.0 + i * 0.1,
                emission_rate_kg_hr=100.0 + i * 50.0,
                time_delta_hours=20.0,
                cloud_free_fraction=0.9,
                crs="EPSG:32613",
                bbox=(540000, 3570000, 541280, 3571280),
            )
        )
    return records, tmp_path


class TestRunInferenceOnCmPatches:
    def test_produces_results(self, cm_records_and_dir):
        records, base_dir = cm_records_and_dir

        # Create a mock model that returns a constant prediction
        model = MagicMock()
        model.to.return_value = model
        model.eval.return_value = None
        model.return_value = torch.ones(1, 1, 64, 64) * 2.0  # high logit → high prob

        results = run_inference_on_cm_patches(
            model, records, base_dir, device="cpu"
        )
        assert len(results) == 3
        assert all("detection_source_id" in r for r in results)
        assert all("detection_score" in r for r in results)
        assert all("plume_fraction" in r for r in results)
        assert all("is_detected" in r for r in results)

    def test_handles_missing_patch(self, tmp_path):
        records = [
            PatchRecord(
                detection_source_id="CM-MISSING",
                scene_id="S2A_MISSING",
                patch_path="nonexistent.tif",
                latitude=31.5,
                longitude=-103.0,
                emission_rate_kg_hr=100.0,
                time_delta_hours=20.0,
                cloud_free_fraction=0.9,
                crs="EPSG:32613",
                bbox=(0, 0, 1, 1),
            )
        ]
        model = MagicMock()
        model.to.return_value = model
        model.eval.return_value = None

        results = run_inference_on_cm_patches(model, records, tmp_path, device="cpu")
        assert len(results) == 0

    def test_detection_score_range(self, cm_records_and_dir):
        records, base_dir = cm_records_and_dir
        model = MagicMock()
        model.to.return_value = model
        model.eval.return_value = None
        # Moderate logit → moderate probability
        model.return_value = torch.zeros(1, 1, 64, 64)

        results = run_inference_on_cm_patches(model, records, base_dir, device="cpu")
        for r in results:
            assert 0 <= r["detection_score"] <= 1


class TestComputeValidationMetrics:
    def test_basic_metrics(self):
        results = [
            {
                "detection_source_id": "CM-001",
                "scene_id": "S2A_001",
                "emission_rate_kg_hr": 100.0,
                "detection_score": 0.8,
                "plume_fraction": 0.05,
                "plume_pixel_count": 200,
                "is_detected": True,
            },
            {
                "detection_source_id": "CM-002",
                "scene_id": "S2A_002",
                "emission_rate_kg_hr": 50.0,
                "detection_score": 0.3,
                "plume_fraction": 0.01,
                "plume_pixel_count": 40,
                "is_detected": False,
            },
            {
                "detection_source_id": "CM-003",
                "scene_id": "S2A_003",
                "emission_rate_kg_hr": 200.0,
                "detection_score": 0.9,
                "plume_fraction": 0.08,
                "plume_pixel_count": 320,
                "is_detected": True,
            },
        ]
        metrics = compute_validation_metrics(results, [])
        assert metrics["n_patches"] == 3
        assert metrics["detection_rate"] == pytest.approx(2 / 3)
        assert 0 < metrics["mean_detection_score"] < 1
        assert "emission_plume_correlation" in metrics

    def test_empty_results(self):
        metrics = compute_validation_metrics([], [])
        assert metrics["n_patches"] == 0
        assert metrics["detection_rate"] == 0.0

    def test_no_emission_rates(self):
        results = [
            {
                "detection_source_id": "CM-001",
                "scene_id": "S2A_001",
                "emission_rate_kg_hr": None,
                "detection_score": 0.5,
                "plume_fraction": 0.02,
                "plume_pixel_count": 80,
                "is_detected": True,
            }
        ]
        metrics = compute_validation_metrics(results, [])
        assert metrics["n_with_emission_rate"] == 0
        assert "emission_plume_correlation" not in metrics
