"""Tests for patch extraction."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from methane_sentinel_labels.config import PipelineConfig
from methane_sentinel_labels.extraction.patches import (
    _compute_cloud_free_fraction,
    _compute_utm_bounds,
    _read_band_window,
    _write_patch_geotiff,
    extract_patches,
)
from methane_sentinel_labels.models import PatchRecord, SceneMatch


@pytest.fixture
def scene_match() -> SceneMatch:
    return SceneMatch(
        detection_source_id="CM-001",
        scene_id="S2A_20240616",
        acquisition_time=datetime(2024, 6, 16, 10, 30, tzinfo=timezone.utc),
        time_delta_hours=20.0,
        cloud_cover_pct=10.0,
        mgrs_tile="13SDA",
        band_hrefs={
            "B02": "https://example.com/B02.tif",
            "B03": "https://example.com/B03.tif",
            "B04": "https://example.com/B04.tif",
            "B8A": "https://example.com/B8A.tif",
            "B11": "https://example.com/B11.tif",
            "B12": "https://example.com/B12.tif",
            "SCL": "https://example.com/SCL.tif",
        },
    )


class TestComputeUtmBounds:
    def test_permian_basin(self):
        # Permian Basin, West Texas → UTM zone 13N (EPSG:32613)
        crs, bounds = _compute_utm_bounds(-103.245, 31.872, half_size_m=2560.0)
        assert "326" in crs  # UTM zone
        left, bottom, right, top = bounds
        assert right - left == pytest.approx(5120.0, abs=1.0)
        assert top - bottom == pytest.approx(5120.0, abs=1.0)

    def test_southern_hemisphere(self):
        # Buenos Aires area → UTM zone 21S (EPSG:32721)
        crs, bounds = _compute_utm_bounds(-58.4, -34.6, half_size_m=2560.0)
        assert "327" in crs  # Southern hemisphere UTM
        left, bottom, right, top = bounds
        assert right - left == pytest.approx(5120.0, abs=1.0)


class TestComputeCloudFreeFraction:
    def test_all_clear(self):
        # SCL values 4-7 are clear
        scl = np.full((256, 256), 4, dtype=np.uint8)
        assert _compute_cloud_free_fraction(scl) == pytest.approx(1.0)

    def test_all_cloudy(self):
        # SCL value 9 = cloud high probability
        scl = np.full((256, 256), 9, dtype=np.uint8)
        assert _compute_cloud_free_fraction(scl) == pytest.approx(0.0)

    def test_mixed(self):
        scl = np.zeros((100, 100), dtype=np.uint8)
        scl[:50, :] = 4  # 50% clear
        scl[50:, :] = 9  # 50% cloudy
        assert _compute_cloud_free_fraction(scl) == pytest.approx(0.5, abs=0.01)

    def test_nodata_excluded(self):
        scl = np.zeros((100, 100), dtype=np.uint8)
        scl[:50, :] = 4   # 50 rows clear
        scl[50:75, :] = 0  # 25 rows nodata (SCL=0)
        scl[75:, :] = 9   # 25 rows cloudy
        # valid = 50 + 25 = 75 rows, clear = 50 rows → 50/75 ≈ 0.667
        frac = _compute_cloud_free_fraction(scl)
        assert frac == pytest.approx(50 / 75, abs=0.01)


class TestReadBandWindow:
    def test_reads_correct_shape(self):
        """Mock rasterio.open to return a synthetic array."""
        mock_dataset = MagicMock()
        mock_dataset.crs = rasterio.crs.CRS.from_epsg(32613)
        mock_dataset.transform = from_bounds(
            500000, 3520000, 609760, 3629760, 10980, 10980
        )
        mock_dataset.res = (10.0, 10.0)
        mock_dataset.read.return_value = np.random.rand(256, 256).astype(np.float32)
        mock_dataset.__enter__ = MagicMock(return_value=mock_dataset)
        mock_dataset.__exit__ = MagicMock(return_value=False)

        with patch("rasterio.open", return_value=mock_dataset):
            data = _read_band_window(
                href="https://example.com/B11.tif",
                crs="EPSG:32613",
                bounds=(540000, 3570000, 545120, 3575120),
                target_res=20.0,
            )
        assert data is not None
        assert data.ndim == 2

    def test_tile_edge_boundless_read(self, tmp_output: Path):
        """Window extending beyond raster should not crash (boundless read)."""
        # Create a small real GeoTIFF (100x100 pixels, 10m res)
        raster_path = tmp_output / "tile_edge.tif"
        height, width = 100, 100
        raster_left, raster_bottom = 540000.0, 3570000.0
        raster_right = raster_left + width * 10.0  # 541000
        raster_top = raster_bottom + height * 10.0  # 3571000
        transform = from_bounds(
            raster_left, raster_bottom, raster_right, raster_top, width, height
        )
        data = np.ones((height, width), dtype=np.uint16) * 1000
        with rasterio.open(
            raster_path, "w", driver="GTiff",
            height=height, width=width, count=1, dtype="uint16",
            crs="EPSG:32613", transform=transform,
        ) as dst:
            dst.write(data, 1)

        # Request a window that's mostly OUTSIDE the raster
        # (negative row_off in pixel coords → RasterioIOError without boundless)
        result = _read_band_window(
            href=str(raster_path),
            crs="EPSG:32613",
            bounds=(540000, 3571000, 545120, 3576120),  # starts at top edge, extends 5120m beyond
            target_res=20.0,
        )
        assert result is not None
        assert result.shape == (256, 256)


class TestWritePatchGeotiff:
    def test_write_and_read_back(self, tmp_output: Path):
        bands = {
            "B11": np.random.rand(256, 256).astype(np.float32),
            "B12": np.random.rand(256, 256).astype(np.float32),
        }
        crs = "EPSG:32613"
        bounds = (540000.0, 3570000.0, 545120.0, 3575120.0)
        out_path = tmp_output / "test_patch.tif"

        _write_patch_geotiff(bands, crs, bounds, out_path)

        assert out_path.exists()
        with rasterio.open(out_path) as ds:
            assert ds.count == 2
            assert ds.width == 256
            assert ds.height == 256
            assert ds.crs.to_epsg() == 32613
            data = ds.read(1)
            assert data.shape == (256, 256)


class TestExtractPatches:
    @patch("methane_sentinel_labels.extraction.patches._read_band_window")
    def test_extract_produces_patch_record(
        self, mock_read, scene_match: SceneMatch, tmp_output: Path
    ):
        # Return synthetic data for each band
        mock_read.side_effect = lambda **kwargs: np.random.rand(256, 256).astype(
            np.float32
        )
        cfg = PipelineConfig(
            output_dir=tmp_output,
            bands=("B11", "B12", "SCL"),
            min_cloud_free_fraction=0.0,
        )
        # Override SCL to return all-clear
        original_side_effect = mock_read.side_effect

        def smart_read(**kwargs):
            if "SCL" in kwargs.get("href", ""):
                scl = np.full((256, 256), 4, dtype=np.uint8)
                return scl.astype(np.float32)
            return original_side_effect(**kwargs)

        mock_read.side_effect = smart_read

        records = extract_patches(
            [scene_match], cfg, latitude=31.872, longitude=-103.245
        )
        assert len(records) == 1
        assert isinstance(records[0], PatchRecord)
        assert records[0].detection_source_id == "CM-001"

    @patch("methane_sentinel_labels.extraction.patches._read_band_window")
    def test_skip_below_cloud_threshold(
        self, mock_read, scene_match: SceneMatch, tmp_output: Path
    ):
        def smart_read(**kwargs):
            if "SCL" in kwargs.get("href", ""):
                # All cloudy
                return np.full((256, 256), 9, dtype=np.float32)
            return np.random.rand(256, 256).astype(np.float32)

        mock_read.side_effect = smart_read
        cfg = PipelineConfig(
            output_dir=tmp_output,
            bands=("B11", "B12", "SCL"),
            min_cloud_free_fraction=0.5,
        )
        records = extract_patches(
            [scene_match], cfg, latitude=31.872, longitude=-103.245
        )
        assert len(records) == 0

    @patch("methane_sentinel_labels.extraction.patches._read_band_window")
    def test_handles_download_error(
        self, mock_read, scene_match: SceneMatch, tmp_output: Path
    ):
        mock_read.side_effect = Exception("Connection timeout")
        cfg = PipelineConfig(
            output_dir=tmp_output,
            bands=("B11",),
            min_cloud_free_fraction=0.0,
        )
        records = extract_patches(
            [scene_match], cfg, latitude=31.872, longitude=-103.245
        )
        assert len(records) == 0
