"""Tests for cross-sensor temporal matching (MethaneSAT → Sentinel-2)."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from methane_sentinel_labels.config import PipelineConfig
from methane_sentinel_labels.matching.cross_sensor import (
    _compute_bbox_overlap,
    _intersect_bboxes,
    _item_to_matched_pair,
    find_sentinel2_matches,
)
from methane_sentinel_labels.models import MatchedPair, MethaneSATScene, PlumeMask


@pytest.fixture
def msat_scene() -> MethaneSATScene:
    return MethaneSATScene(
        scene_id="MST001",
        gcs_path="cog_gee/t100/scene.tif",
        local_path="/tmp/scene.tif",
        acquisition_time=datetime(2024, 9, 11, 22, 0, tzinfo=timezone.utc),
        bbox=(-103.5, 31.0, -103.0, 31.5),
        crs="EPSG:4326",
        resolution_m=46.4,
        xch4_median_ppb=1930.0,
        target_id="100",
    )


@pytest.fixture
def plume_mask() -> PlumeMask:
    return PlumeMask(
        scene_id="MST001",
        mask_path="/tmp/mask.tif",
        threshold_ppb=50.0,
        anomaly_method="median_subtract",
        plume_pixel_count=400,
        total_valid_pixels=10000,
        plume_fraction=0.04,
        bbox=(-103.5, 31.0, -103.0, 31.5),
        crs="EPSG:4326",
    )


def _make_stac_item(
    item_id: str,
    dt: datetime,
    cloud_cover: float,
    bbox: tuple[float, float, float, float] = (-104.0, 30.5, -102.5, 32.0),
    mgrs_tile: str = "13SDA",
) -> MagicMock:
    """Create a mock STAC item with a bbox."""
    item = MagicMock()
    item.id = item_id
    item.datetime = dt
    item.bbox = list(bbox)
    item.properties = {
        "eo:cloud_cover": cloud_cover,
        "grid:code": f"MGRS-{mgrs_tile}",
    }
    item.assets = {
        "blue": MagicMock(href="https://example.com/blue.tif"),
        "green": MagicMock(href="https://example.com/green.tif"),
        "red": MagicMock(href="https://example.com/red.tif"),
        "nir08": MagicMock(href="https://example.com/nir08.tif"),
        "swir16": MagicMock(href="https://example.com/swir16.tif"),
        "swir22": MagicMock(href="https://example.com/swir22.tif"),
        "scl": MagicMock(href="https://example.com/scl.tif"),
    }
    return item


class TestComputeBboxOverlap:
    def test_full_overlap(self):
        bbox_a = (0.0, 0.0, 1.0, 1.0)
        bbox_b = (-0.5, -0.5, 1.5, 1.5)
        assert _compute_bbox_overlap(bbox_a, bbox_b) == pytest.approx(1.0)

    def test_no_overlap(self):
        bbox_a = (0.0, 0.0, 1.0, 1.0)
        bbox_b = (2.0, 2.0, 3.0, 3.0)
        assert _compute_bbox_overlap(bbox_a, bbox_b) == 0.0

    def test_partial_overlap(self):
        bbox_a = (0.0, 0.0, 2.0, 2.0)  # area = 4
        bbox_b = (1.0, 1.0, 3.0, 3.0)  # intersection = 1x1 = 1
        assert _compute_bbox_overlap(bbox_a, bbox_b) == pytest.approx(0.25)

    def test_zero_area_bbox(self):
        bbox_a = (0.0, 0.0, 0.0, 0.0)
        bbox_b = (0.0, 0.0, 1.0, 1.0)
        assert _compute_bbox_overlap(bbox_a, bbox_b) == 0.0


class TestIntersectBboxes:
    def test_intersection(self):
        a = (0.0, 0.0, 2.0, 2.0)
        b = (1.0, 1.0, 3.0, 3.0)
        assert _intersect_bboxes(a, b) == (1.0, 1.0, 2.0, 2.0)

    def test_contained(self):
        a = (1.0, 1.0, 2.0, 2.0)
        b = (0.0, 0.0, 3.0, 3.0)
        assert _intersect_bboxes(a, b) == (1.0, 1.0, 2.0, 2.0)


class TestItemToMatchedPair:
    def test_converts_item(self, msat_scene, plume_mask):
        item = _make_stac_item(
            "S2A_20240912",
            datetime(2024, 9, 12, 10, 30, tzinfo=timezone.utc),
            cloud_cover=12.5,
        )
        cfg = PipelineConfig(msat_max_time_delta_hours=72.0)
        pair = _item_to_matched_pair(item, msat_scene, plume_mask, cfg)
        assert pair is not None
        assert isinstance(pair, MatchedPair)
        assert pair.msat_scene_id == "MST001"
        assert pair.s2_scene_id == "S2A_20240912"
        assert "B11" in pair.s2_band_hrefs

    def test_computes_time_delta(self, msat_scene, plume_mask):
        # MSAT at 2024-09-11 22:00, S2 at 2024-09-12 10:30 → 12.5h
        item = _make_stac_item(
            "S2A_20240912",
            datetime(2024, 9, 12, 10, 30, tzinfo=timezone.utc),
            cloud_cover=5.0,
        )
        cfg = PipelineConfig(msat_max_time_delta_hours=72.0)
        pair = _item_to_matched_pair(item, msat_scene, plume_mask, cfg)
        assert pair is not None
        assert abs(pair.time_delta_hours - 12.5) < 0.01

    def test_filters_large_time_delta(self, msat_scene, plume_mask):
        item = _make_stac_item(
            "S2A_late",
            datetime(2024, 9, 20, 10, 0, tzinfo=timezone.utc),
            cloud_cover=5.0,
        )
        cfg = PipelineConfig(msat_max_time_delta_hours=72.0)
        pair = _item_to_matched_pair(item, msat_scene, plume_mask, cfg)
        assert pair is None

    def test_filters_high_cloud_cover(self, msat_scene, plume_mask):
        item = _make_stac_item(
            "S2A_cloudy",
            datetime(2024, 9, 12, 10, 0, tzinfo=timezone.utc),
            cloud_cover=50.0,
        )
        cfg = PipelineConfig(max_cloud_cover_pct=30.0, msat_max_time_delta_hours=72.0)
        pair = _item_to_matched_pair(item, msat_scene, plume_mask, cfg)
        assert pair is None

    def test_filters_low_spatial_overlap(self, msat_scene, plume_mask):
        # S2 bbox completely outside MSAT bbox
        item = _make_stac_item(
            "S2A_far",
            datetime(2024, 9, 12, 10, 0, tzinfo=timezone.utc),
            cloud_cover=5.0,
            bbox=(10.0, 50.0, 11.0, 51.0),  # Europe, not Texas
        )
        cfg = PipelineConfig(
            msat_max_time_delta_hours=72.0, msat_min_spatial_overlap=0.1
        )
        pair = _item_to_matched_pair(item, msat_scene, plume_mask, cfg)
        assert pair is None

    def test_intersection_bbox(self, msat_scene, plume_mask):
        item = _make_stac_item(
            "S2A_partial",
            datetime(2024, 9, 12, 10, 0, tzinfo=timezone.utc),
            cloud_cover=5.0,
            bbox=(-103.3, 31.2, -102.0, 32.0),  # partial overlap
        )
        cfg = PipelineConfig(
            msat_max_time_delta_hours=72.0, msat_min_spatial_overlap=0.1
        )
        pair = _item_to_matched_pair(item, msat_scene, plume_mask, cfg)
        assert pair is not None
        # Intersection should be (-103.3, 31.2, -103.0, 31.5)
        assert pair.bbox[0] == pytest.approx(-103.3)
        assert pair.bbox[1] == pytest.approx(31.2)


class TestFindSentinel2Matches:
    @patch("methane_sentinel_labels.matching.cross_sensor.query_stac_bbox")
    def test_returns_best_match(self, mock_query, msat_scene, plume_mask):
        items = [
            _make_stac_item(
                "S2A_far",
                datetime(2024, 9, 13, 10, 0, tzinfo=timezone.utc),
                cloud_cover=5.0,
            ),
            _make_stac_item(
                "S2A_close",
                datetime(2024, 9, 12, 8, 0, tzinfo=timezone.utc),
                cloud_cover=5.0,
            ),
        ]
        mock_query.return_value = items
        cfg = PipelineConfig(msat_max_time_delta_hours=72.0)
        pairs = find_sentinel2_matches([plume_mask], [msat_scene], cfg)
        assert len(pairs) == 1
        assert pairs[0].s2_scene_id == "S2A_close"  # Closest in time

    @patch("methane_sentinel_labels.matching.cross_sensor.query_stac_bbox")
    def test_no_matches(self, mock_query, msat_scene, plume_mask):
        mock_query.return_value = []
        cfg = PipelineConfig(msat_max_time_delta_hours=72.0)
        pairs = find_sentinel2_matches([plume_mask], [msat_scene], cfg)
        assert pairs == []

    @patch("methane_sentinel_labels.matching.cross_sensor.query_stac_bbox")
    def test_skips_mask_without_scene(self, mock_query, plume_mask):
        cfg = PipelineConfig(msat_max_time_delta_hours=72.0)
        pairs = find_sentinel2_matches([plume_mask], [], cfg)  # No scenes
        assert pairs == []
        mock_query.assert_not_called()

    @patch("methane_sentinel_labels.matching.cross_sensor.query_stac_bbox")
    def test_uses_bbox_not_point(self, mock_query, msat_scene, plume_mask):
        """Verify STAC query uses bounding box from the mask."""
        mock_query.return_value = []
        cfg = PipelineConfig(msat_max_time_delta_hours=72.0)
        find_sentinel2_matches([plume_mask], [msat_scene], cfg)
        mock_query.assert_called_once()
        call_kwargs = mock_query.call_args
        bbox_arg = call_kwargs.kwargs.get("bbox") or call_kwargs.args[0]
        assert bbox_arg == plume_mask.bbox
