"""Tests for domain models."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from methane_sentinel_labels.models import Detection, PatchRecord, SceneMatch


class TestDetection:
    def test_create(self):
        d = Detection(
            source_id="CM-123",
            latitude=31.5,
            longitude=-103.2,
            detection_time=datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc),
            emission_rate_kg_hr=150.0,
            emission_uncertainty_kg_hr=30.0,
            sensor="Tanager-1",
            provider="carbon_mapper",
        )
        assert d.source_id == "CM-123"
        assert d.latitude == 31.5
        assert d.emission_rate_kg_hr == 150.0

    def test_frozen(self):
        d = Detection(
            source_id="CM-123",
            latitude=31.5,
            longitude=-103.2,
            detection_time=datetime(2024, 6, 15, tzinfo=timezone.utc),
            emission_rate_kg_hr=None,
            emission_uncertainty_kg_hr=None,
            sensor="EMIT",
            provider="carbon_mapper",
        )
        with pytest.raises(FrozenInstanceError):
            d.latitude = 0.0  # type: ignore[misc]

    def test_nullable_emission_fields(self):
        d = Detection(
            source_id="CM-456",
            latitude=0.0,
            longitude=0.0,
            detection_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            emission_rate_kg_hr=None,
            emission_uncertainty_kg_hr=None,
            sensor="AVIRIS-NG",
            provider="carbon_mapper",
        )
        assert d.emission_rate_kg_hr is None
        assert d.emission_uncertainty_kg_hr is None


class TestSceneMatch:
    def test_create(self):
        sm = SceneMatch(
            detection_source_id="CM-123",
            scene_id="S2A_MSIL2A_20240617T170901",
            acquisition_time=datetime(2024, 6, 17, 17, 9, tzinfo=timezone.utc),
            time_delta_hours=51.15,
            cloud_cover_pct=12.3,
            mgrs_tile="13SDA",
            band_hrefs={"B11": "https://example.com/B11.tif"},
        )
        assert sm.time_delta_hours == 51.15
        assert sm.band_hrefs["B11"] == "https://example.com/B11.tif"

    def test_frozen(self):
        sm = SceneMatch(
            detection_source_id="CM-123",
            scene_id="S2A_MSIL2A_20240617T170901",
            acquisition_time=datetime(2024, 6, 17, tzinfo=timezone.utc),
            time_delta_hours=10.0,
            cloud_cover_pct=5.0,
            mgrs_tile="13SDA",
            band_hrefs={},
        )
        with pytest.raises(FrozenInstanceError):
            sm.cloud_cover_pct = 0.0  # type: ignore[misc]


class TestPatchRecord:
    def test_create(self):
        pr = PatchRecord(
            detection_source_id="CM-123",
            scene_id="S2A_MSIL2A_20240617T170901",
            patch_path="patches/CM-123_S2A.tif",
            latitude=31.5,
            longitude=-103.2,
            emission_rate_kg_hr=150.0,
            time_delta_hours=51.15,
            cloud_free_fraction=0.95,
            crs="EPSG:32613",
            bbox=(-103.25, 31.45, -103.15, 31.55),
        )
        assert pr.cloud_free_fraction == 0.95
        assert pr.crs == "EPSG:32613"
        assert len(pr.bbox) == 4

    def test_frozen(self):
        pr = PatchRecord(
            detection_source_id="CM-123",
            scene_id="scene1",
            patch_path="p.tif",
            latitude=0.0,
            longitude=0.0,
            emission_rate_kg_hr=None,
            time_delta_hours=1.0,
            cloud_free_fraction=1.0,
            crs="EPSG:4326",
            bbox=(0, 0, 1, 1),
        )
        with pytest.raises(FrozenInstanceError):
            pr.patch_path = "other.tif"  # type: ignore[misc]
