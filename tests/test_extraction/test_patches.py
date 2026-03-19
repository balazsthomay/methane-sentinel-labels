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
    _reproject_mask_to_s2_grid,
    _write_patch_geotiff,
    extract_patches,
    extract_training_patch,
    find_plume_patch_centers,
)
from methane_sentinel_labels.models import MatchedPair, PatchRecord, SceneMatch, TrainingPatch


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

    @patch("methane_sentinel_labels.extraction.patches._read_band_window")
    def test_skips_existing_patch(
        self, mock_read, scene_match: SceneMatch, tmp_output: Path
    ):
        """Existing patch file should be reused without S3 reads."""
        cfg = PipelineConfig(
            output_dir=tmp_output,
            bands=("B11", "B12"),
            min_cloud_free_fraction=0.5,
        )
        # Create the patch file that would already exist
        patches_dir = tmp_output / "patches"
        patches_dir.mkdir(parents=True)
        patch_path = patches_dir / f"{scene_match.detection_source_id}_{scene_match.scene_id}.tif"
        bands = {"B11": np.ones((256, 256), dtype=np.uint16) * 1000}
        bounds = (540000.0, 3570000.0, 545120.0, 3575120.0)
        _write_patch_geotiff(bands, "EPSG:32613", bounds, patch_path, cloud_free_fraction=0.95)

        records = extract_patches(
            [scene_match], cfg, latitude=31.872, longitude=-103.245
        )
        assert len(records) == 1
        assert records[0].cloud_free_fraction == pytest.approx(0.95)
        # No S3 reads should have happened
        mock_read.assert_not_called()

    def test_cloud_free_tag_roundtrip(self, tmp_output: Path):
        """cloud_free_fraction should survive write → read via GeoTIFF tags."""
        bands = {"B11": np.ones((256, 256), dtype=np.uint16)}
        bounds = (540000.0, 3570000.0, 545120.0, 3575120.0)
        out_path = tmp_output / "tagged.tif"
        _write_patch_geotiff(bands, "EPSG:32613", bounds, out_path, cloud_free_fraction=0.8765)

        with rasterio.open(out_path) as ds:
            tags = ds.tags()
        assert float(tags["cloud_free_fraction"]) == pytest.approx(0.8765)


class TestReprojectMaskToS2Grid:
    def test_reprojects_known_mask(self, tmp_output: Path):
        """A mask in EPSG:4326 should reproject to UTM preserving plume location."""
        # Create a mask in EPSG:4326 with a known plume at center
        mask_path = tmp_output / "mask_4326.tif"
        height, width = 100, 100
        mask_data = np.zeros((height, width), dtype=np.uint8)
        mask_data[40:60, 40:60] = 1  # plume at center

        transform_4326 = from_bounds(-103.5, 31.0, -103.0, 31.5, width, height)
        with rasterio.open(
            mask_path, "w", driver="GTiff",
            height=height, width=width, count=1, dtype="uint8",
            crs="EPSG:4326", transform=transform_4326,
        ) as ds:
            ds.write(mask_data, 1)

        # Reproject to UTM zone 13N
        crs, bounds = _compute_utm_bounds(-103.25, 31.25, half_size_m=2560.0)
        result = _reproject_mask_to_s2_grid(
            str(mask_path), crs, bounds, (256, 256)
        )
        assert result.shape == (256, 256)
        assert result.dtype == np.uint8
        # Should have some plume pixels
        assert result.sum() > 0
        # Should be binary
        assert set(np.unique(result)) <= {0, 1}

    def test_no_overlap_returns_zeros(self, tmp_output: Path):
        """Mask far from target bounds should produce all-zero output."""
        mask_path = tmp_output / "mask_far.tif"
        height, width = 50, 50
        mask_data = np.ones((height, width), dtype=np.uint8)

        # Mask in Europe
        transform_4326 = from_bounds(10.0, 50.0, 11.0, 51.0, width, height)
        with rasterio.open(
            mask_path, "w", driver="GTiff",
            height=height, width=width, count=1, dtype="uint8",
            crs="EPSG:4326", transform=transform_4326,
        ) as ds:
            ds.write(mask_data, 1)

        # Target in Texas
        crs, bounds = _compute_utm_bounds(-103.25, 31.25, half_size_m=2560.0)
        result = _reproject_mask_to_s2_grid(
            str(mask_path), crs, bounds, (256, 256)
        )
        assert result.sum() == 0


class TestFindPlumePatchCenters:
    def test_finds_plume_tiles(self, tmp_output: Path):
        """Tiles with plume pixels should produce centers."""
        mask_path = tmp_output / "multi_plume_mask.tif"
        height, width = 400, 400
        mask = np.zeros((height, width), dtype=np.uint8)
        # Two plume regions in different grid tiles
        mask[20:50, 20:50] = 1  # top-left region
        mask[300:330, 300:330] = 1  # bottom-right region

        transform_4326 = from_bounds(-103.5, 31.0, -102.5, 32.0, width, height)
        with rasterio.open(
            mask_path, "w", driver="GTiff",
            height=height, width=width, count=1, dtype="uint8",
            crs="EPSG:4326", transform=transform_4326,
        ) as ds:
            ds.write(mask, 1)

        centers = find_plume_patch_centers(str(mask_path), min_plume_pixels=10)
        assert len(centers) >= 2  # at least 2 positive + some negatives

    def test_filters_sparse_plume(self, tmp_output: Path):
        """Tiles with very few plume pixels should be filtered."""
        mask_path = tmp_output / "sparse_mask.tif"
        height, width = 200, 200
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[50, 50] = 1  # single pixel

        transform_4326 = from_bounds(-103.5, 31.0, -103.0, 31.5, width, height)
        with rasterio.open(
            mask_path, "w", driver="GTiff",
            height=height, width=width, count=1, dtype="uint8",
            crs="EPSG:4326", transform=transform_4326,
        ) as ds:
            ds.write(mask, 1)

        centers = find_plume_patch_centers(
            str(mask_path), min_plume_pixels=50, include_negatives=False
        )
        assert len(centers) == 0

    def test_empty_mask(self, tmp_output: Path):
        mask_path = tmp_output / "empty_mask.tif"
        height, width = 50, 50
        mask = np.zeros((height, width), dtype=np.uint8)
        transform_4326 = from_bounds(-103.5, 31.0, -103.0, 31.5, width, height)
        with rasterio.open(
            mask_path, "w", driver="GTiff",
            height=height, width=width, count=1, dtype="uint8",
            crs="EPSG:4326", transform=transform_4326,
        ) as ds:
            ds.write(mask, 1)

        centers = find_plume_patch_centers(str(mask_path), min_plume_pixels=10)
        assert len(centers) == 0

    def test_respects_max_patches(self, tmp_output: Path):
        mask_path = tmp_output / "many_plumes.tif"
        height, width = 1000, 1000
        mask = np.zeros((height, width), dtype=np.uint8)
        # Fill with scattered plume regions
        for i in range(0, 900, 100):
            for j in range(0, 900, 100):
                mask[i:i+20, j:j+20] = 1

        transform_4326 = from_bounds(-105.0, 30.0, -100.0, 35.0, width, height)
        with rasterio.open(
            mask_path, "w", driver="GTiff",
            height=height, width=width, count=1, dtype="uint8",
            crs="EPSG:4326", transform=transform_4326,
        ) as ds:
            ds.write(mask, 1)

        centers = find_plume_patch_centers(
            str(mask_path), min_plume_pixels=5, max_patches_per_scene=5,
            include_negatives=False,
        )
        assert len(centers) <= 5


class TestExtractTrainingPatch:
    @patch("methane_sentinel_labels.extraction.patches._read_band_window")
    def test_produces_training_patch(self, mock_read, tmp_output: Path):
        # Create a real mask file for reprojection
        mask_path = tmp_output / "mask.tif"
        mask_data = np.zeros((100, 100), dtype=np.uint8)
        mask_data[40:60, 40:60] = 1
        transform_4326 = from_bounds(-103.5, 31.0, -103.0, 31.5, 100, 100)
        with rasterio.open(
            mask_path, "w", driver="GTiff",
            height=100, width=100, count=1, dtype="uint8",
            crs="EPSG:4326", transform=transform_4326,
        ) as ds:
            ds.write(mask_data, 1)

        pair = MatchedPair(
            msat_scene_id="MST001",
            s2_scene_id="S2A_20240912",
            msat_acquisition_time=datetime(2024, 9, 11, 22, 0, tzinfo=timezone.utc),
            s2_acquisition_time=datetime(2024, 9, 12, 10, 30, tzinfo=timezone.utc),
            time_delta_hours=12.5,
            msat_mask_path=str(mask_path),
            s2_band_hrefs={
                "B02": "https://example.com/B02.tif",
                "B03": "https://example.com/B03.tif",
                "B04": "https://example.com/B04.tif",
                "B8A": "https://example.com/B8A.tif",
                "B11": "https://example.com/B11.tif",
                "B12": "https://example.com/B12.tif",
                "SCL": "https://example.com/SCL.tif",
            },
            s2_mgrs_tile="13SDA",
            bbox=(-103.5, 31.0, -103.0, 31.5),
            s2_cloud_cover_pct=10.0,
        )

        def smart_read(**kwargs):
            if "SCL" in kwargs.get("href", ""):
                return np.full((256, 256), 4, dtype=np.float32)
            return np.random.default_rng(42).random((256, 256)).astype(np.float32) * 1000 + 500

        mock_read.side_effect = smart_read

        cfg = PipelineConfig(
            output_dir=tmp_output,
            min_cloud_free_fraction=0.0,
        )
        result = extract_training_patch(pair, cfg)
        assert result is not None
        assert isinstance(result, TrainingPatch)
        assert result.msat_scene_id == "MST001"
        assert "varon" in result.band_names
        assert "mask" in result.band_names

        # Verify GeoTIFF output
        full_path = tmp_output / result.patch_path
        assert full_path.exists()
        with rasterio.open(full_path) as ds:
            assert ds.count == 8  # B02, B03, B04, B8A, B11, B12, varon, mask
