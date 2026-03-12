"""Tests for Sentinel-2 temporal matching."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from methane_sentinel_labels.config import PipelineConfig
from methane_sentinel_labels.matching.sentinel2 import (
    find_matches,
    _item_to_scene_match,
    _query_stac,
)
from methane_sentinel_labels.models import Detection, SceneMatch


@pytest.fixture
def detection() -> Detection:
    return Detection(
        source_id="CM-001",
        latitude=31.872,
        longitude=-103.245,
        detection_time=datetime(2024, 6, 15, 14, 30, tzinfo=timezone.utc),
        emission_rate_kg_hr=150.0,
        emission_uncertainty_kg_hr=30.0,
        sensor="tanager",
        provider="carbon_mapper",
    )


def _make_stac_item(
    item_id: str,
    dt: datetime,
    cloud_cover: float,
    mgrs_tile: str = "13SDA",
) -> MagicMock:
    """Create a mock STAC item."""
    item = MagicMock()
    item.id = item_id
    item.datetime = dt
    item.properties = {
        "eo:cloud_cover": cloud_cover,
        "grid:code": f"MGRS-{mgrs_tile}",
    }
    item.assets = {
        "B02": MagicMock(href="https://example.com/B02.tif"),
        "B03": MagicMock(href="https://example.com/B03.tif"),
        "B04": MagicMock(href="https://example.com/B04.tif"),
        "B8A": MagicMock(href="https://example.com/B8A.tif"),
        "B11": MagicMock(href="https://example.com/B11.tif"),
        "B12": MagicMock(href="https://example.com/B12.tif"),
        "SCL": MagicMock(href="https://example.com/SCL.tif"),
    }
    return item


class TestItemToSceneMatch:
    def test_converts_item(self, detection: Detection):
        item = _make_stac_item(
            "S2A_20240616",
            datetime(2024, 6, 16, 10, 30, tzinfo=timezone.utc),
            cloud_cover=12.5,
        )
        cfg = PipelineConfig()
        match = _item_to_scene_match(item, detection, cfg)
        assert match is not None
        assert match.scene_id == "S2A_20240616"
        assert match.detection_source_id == "CM-001"
        assert match.cloud_cover_pct == 12.5
        assert match.mgrs_tile == "13SDA"
        assert "B11" in match.band_hrefs
        assert "B12" in match.band_hrefs
        assert "SCL" in match.band_hrefs

    def test_computes_time_delta(self, detection: Detection):
        # Detection at 2024-06-15 14:30 UTC
        # Scene at 2024-06-16 10:30 UTC → 20 hours later
        item = _make_stac_item(
            "S2A_20240616",
            datetime(2024, 6, 16, 10, 30, tzinfo=timezone.utc),
            cloud_cover=5.0,
        )
        cfg = PipelineConfig()
        match = _item_to_scene_match(item, detection, cfg)
        assert match is not None
        assert abs(match.time_delta_hours - 20.0) < 0.01

    def test_filters_high_cloud_cover(self, detection: Detection):
        item = _make_stac_item(
            "S2A_cloudy",
            datetime(2024, 6, 16, 10, 0, tzinfo=timezone.utc),
            cloud_cover=50.0,
        )
        cfg = PipelineConfig(max_cloud_cover_pct=30.0)
        match = _item_to_scene_match(item, detection, cfg)
        assert match is None

    def test_filters_large_time_delta(self, detection: Detection):
        # 10 days later — exceeds 120h default
        item = _make_stac_item(
            "S2A_late",
            datetime(2024, 6, 25, 14, 30, tzinfo=timezone.utc),
            cloud_cover=5.0,
        )
        cfg = PipelineConfig(max_time_delta_hours=120.0)
        match = _item_to_scene_match(item, detection, cfg)
        assert match is None

    def test_extracts_band_hrefs(self, detection: Detection):
        item = _make_stac_item(
            "S2A_bands",
            datetime(2024, 6, 16, 10, 0, tzinfo=timezone.utc),
            cloud_cover=5.0,
        )
        cfg = PipelineConfig()
        match = _item_to_scene_match(item, detection, cfg)
        assert match is not None
        for band in cfg.bands:
            assert band in match.band_hrefs

    def test_extracts_mgrs_tile(self, detection: Detection):
        item = _make_stac_item(
            "S2A_mgrs",
            datetime(2024, 6, 16, 10, 0, tzinfo=timezone.utc),
            cloud_cover=5.0,
            mgrs_tile="14TQL",
        )
        cfg = PipelineConfig()
        match = _item_to_scene_match(item, detection, cfg)
        assert match is not None
        assert match.mgrs_tile == "14TQL"


class TestFindMatches:
    @patch("methane_sentinel_labels.matching.sentinel2._query_stac")
    def test_returns_matches(self, mock_query, detection: Detection):
        items = [
            _make_stac_item(
                "S2A_close",
                datetime(2024, 6, 16, 10, 0, tzinfo=timezone.utc),
                cloud_cover=10.0,
            ),
            _make_stac_item(
                "S2A_further",
                datetime(2024, 6, 17, 10, 0, tzinfo=timezone.utc),
                cloud_cover=15.0,
            ),
        ]
        mock_query.return_value = items
        cfg = PipelineConfig()
        matches = find_matches([detection], cfg)
        assert len(matches) == 2
        assert all(isinstance(m, SceneMatch) for m in matches)

    @patch("methane_sentinel_labels.matching.sentinel2._query_stac")
    def test_no_matches(self, mock_query, detection: Detection):
        mock_query.return_value = []
        cfg = PipelineConfig()
        matches = find_matches([detection], cfg)
        assert matches == []

    @patch("methane_sentinel_labels.matching.sentinel2._query_stac")
    def test_sorted_by_time_delta(self, mock_query, detection: Detection):
        items = [
            _make_stac_item(
                "S2A_far",
                datetime(2024, 6, 18, 14, 0, tzinfo=timezone.utc),
                cloud_cover=5.0,
            ),
            _make_stac_item(
                "S2A_close",
                datetime(2024, 6, 15, 17, 0, tzinfo=timezone.utc),
                cloud_cover=5.0,
            ),
            _make_stac_item(
                "S2A_mid",
                datetime(2024, 6, 16, 14, 0, tzinfo=timezone.utc),
                cloud_cover=5.0,
            ),
        ]
        mock_query.return_value = items
        cfg = PipelineConfig()
        matches = find_matches([detection], cfg)
        deltas = [m.time_delta_hours for m in matches]
        assert deltas == sorted(deltas)

    @patch("methane_sentinel_labels.matching.sentinel2._query_stac")
    def test_max_matches_per_detection(self, mock_query, detection: Detection):
        items = [
            _make_stac_item(
                f"S2A_{i}",
                datetime(2024, 6, 15 + i, 10, 0, tzinfo=timezone.utc),
                cloud_cover=5.0,
            )
            for i in range(5)
        ]
        mock_query.return_value = items
        cfg = PipelineConfig(max_matches_per_detection=2)
        matches = find_matches([detection], cfg)
        assert len(matches) == 2

    @patch("methane_sentinel_labels.matching.sentinel2._query_stac")
    def test_filters_out_cloudy_and_late(self, mock_query, detection: Detection):
        items = [
            _make_stac_item(
                "S2A_good",
                datetime(2024, 6, 16, 10, 0, tzinfo=timezone.utc),
                cloud_cover=10.0,
            ),
            _make_stac_item(
                "S2A_cloudy",
                datetime(2024, 6, 16, 11, 0, tzinfo=timezone.utc),
                cloud_cover=50.0,
            ),
            _make_stac_item(
                "S2A_late",
                datetime(2024, 6, 25, 10, 0, tzinfo=timezone.utc),
                cloud_cover=5.0,
            ),
        ]
        mock_query.return_value = items
        cfg = PipelineConfig(max_cloud_cover_pct=30.0, max_time_delta_hours=120.0)
        matches = find_matches([detection], cfg)
        assert len(matches) == 1
        assert matches[0].scene_id == "S2A_good"


@pytest.mark.integration
def test_query_stac_real():
    """Integration test: query real STAC for a known location."""
    detection = Detection(
        source_id="CM-PERM-001",
        latitude=31.872,
        longitude=-103.245,
        detection_time=datetime(2024, 6, 15, 14, 30, tzinfo=timezone.utc),
        emission_rate_kg_hr=150.0,
        emission_uncertainty_kg_hr=30.0,
        sensor="tanager",
        provider="carbon_mapper",
    )
    cfg = PipelineConfig(max_matches_per_detection=2)
    matches = find_matches([detection], cfg)
    assert len(matches) > 0
    assert all(isinstance(m, SceneMatch) for m in matches)
