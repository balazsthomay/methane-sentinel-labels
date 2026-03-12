"""Tests for dataset assembly."""

import json
from pathlib import Path

import pandas as pd
import pytest

from methane_sentinel_labels.assembly.dataset import (
    _compute_summary,
    _write_manifest,
    _write_summary,
    assemble_dataset,
)
from methane_sentinel_labels.models import PatchRecord


@pytest.fixture
def patch_records() -> list[PatchRecord]:
    return [
        PatchRecord(
            detection_source_id="CM-001",
            scene_id="S2A_20240616",
            patch_path="patches/CM-001_S2A_20240616.tif",
            latitude=31.872,
            longitude=-103.245,
            emission_rate_kg_hr=150.0,
            time_delta_hours=20.0,
            cloud_free_fraction=0.95,
            crs="EPSG:32613",
            bbox=(540000, 3520000, 545120, 3525120),
        ),
        PatchRecord(
            detection_source_id="CM-002",
            scene_id="S2A_20240617",
            patch_path="patches/CM-002_S2A_20240617.tif",
            latitude=32.015,
            longitude=-101.534,
            emission_rate_kg_hr=85.0,
            time_delta_hours=45.5,
            cloud_free_fraction=0.80,
            crs="EPSG:32613",
            bbox=(560000, 3540000, 565120, 3545120),
        ),
        PatchRecord(
            detection_source_id="CM-003",
            scene_id="S2A_20240618",
            patch_path="patches/CM-003_S2A_20240618.tif",
            latitude=31.5,
            longitude=-102.1,
            emission_rate_kg_hr=None,
            time_delta_hours=72.0,
            cloud_free_fraction=0.60,
            crs="EPSG:32613",
            bbox=(550000, 3530000, 555120, 3535120),
        ),
    ]


class TestWriteManifest:
    def test_correct_schema(self, tmp_output: Path, patch_records: list[PatchRecord]):
        path = tmp_output / "manifest.parquet"
        _write_manifest(patch_records, path)
        df = pd.read_parquet(path)
        expected_cols = {
            "detection_source_id",
            "scene_id",
            "patch_path",
            "latitude",
            "longitude",
            "emission_rate_kg_hr",
            "time_delta_hours",
            "cloud_free_fraction",
            "crs",
            "bbox",
        }
        assert expected_cols == set(df.columns)
        assert len(df) == 3

    def test_roundtrip(self, tmp_output: Path, patch_records: list[PatchRecord]):
        path = tmp_output / "manifest.parquet"
        _write_manifest(patch_records, path)
        df = pd.read_parquet(path)
        assert df.iloc[0]["detection_source_id"] == "CM-001"
        assert df.iloc[0]["emission_rate_kg_hr"] == 150.0
        assert pd.isna(df.iloc[2]["emission_rate_kg_hr"])


class TestComputeSummary:
    def test_summary_stats(self, patch_records: list[PatchRecord]):
        summary = _compute_summary(patch_records)
        assert summary["total_patches"] == 3
        assert summary["median_time_delta_hours"] == 45.5
        assert summary["mean_cloud_free_fraction"] == pytest.approx(
            (0.95 + 0.80 + 0.60) / 3, abs=0.01
        )
        assert summary["patches_with_emission_rate"] == 2

    def test_emission_rate_stats(self, patch_records: list[PatchRecord]):
        summary = _compute_summary(patch_records)
        assert summary["min_emission_rate_kg_hr"] == 85.0
        assert summary["max_emission_rate_kg_hr"] == 150.0

    def test_empty_records(self):
        summary = _compute_summary([])
        assert summary["total_patches"] == 0


class TestWriteSummary:
    def test_writes_json(self, tmp_output: Path, patch_records: list[PatchRecord]):
        summary = _compute_summary(patch_records)
        path = tmp_output / "summary.json"
        _write_summary(summary, path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["total_patches"] == 3


class TestAssembleDataset:
    def test_creates_output_structure(
        self, tmp_output: Path, patch_records: list[PatchRecord]
    ):
        assemble_dataset(patch_records, tmp_output)
        assert (tmp_output / "manifest.parquet").exists()
        assert (tmp_output / "summary.json").exists()
