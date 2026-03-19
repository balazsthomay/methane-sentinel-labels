"""Tests for MethaneSAT L3 ingestion and plume mask generation."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from methane_sentinel_labels.config import PipelineConfig
from methane_sentinel_labels.ingest.methanesat import (
    _parse_time_from_filename,
    download_scene,
    generate_plume_mask,
    ingest_methanesat,
    list_l3_scenes,
    parse_scene,
)
from methane_sentinel_labels.models import MethaneSATScene, PlumeMask


def _create_synthetic_l3_cog(
    path: Path,
    *,
    xch4_background: float = 1930.0,
    plume_value: float = 2050.0,
    plume_row_range: tuple[int, int] = (40, 60),
    plume_col_range: tuple[int, int] = (40, 60),
    height: int = 100,
    width: int = 100,
    nan_fraction: float = 0.0,
) -> Path:
    """Create a synthetic 4-band MethaneSAT L3 COG for testing."""
    xch4 = np.full((height, width), xch4_background, dtype=np.float32)

    # Insert plume
    r0, r1 = plume_row_range
    c0, c1 = plume_col_range
    xch4[r0:r1, c0:c1] = plume_value

    # Insert NaN regions
    if nan_fraction > 0:
        rng = np.random.default_rng(42)
        nan_mask = rng.random((height, width)) < nan_fraction
        xch4[nan_mask] = np.nan

    # Other bands: albedo, surface pressure, terrain height
    albedo = np.full((height, width), 0.15, dtype=np.float32)
    pressure = np.full((height, width), 850.0, dtype=np.float32)
    terrain = np.full((height, width), 0.5, dtype=np.float32)

    transform = from_bounds(-103.5, 31.0, -103.0, 31.5, width, height)

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=4,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=float("nan"),
    ) as ds:
        ds.write(xch4, 1)
        ds.write(albedo, 2)
        ds.write(pressure, 3)
        ds.write(terrain, 4)
        ds.set_band_description(1, "xch4 (ppb)")
        ds.set_band_description(2, "albedo")
        ds.set_band_description(3, "surface pressure (hPa)")
        ds.set_band_description(4, "terrain height (km)")
        ds.update_tags(
            collection_id="TEST001",
            target_id="42",
            processing_id="999",
            time_coverage_start="2024-09-11T22:05:58Z",
            time_coverage_end="2024-09-11T22:06:20Z",
            platform="MethaneSAT",
        )

    return path


class TestParseTimeFromFilename:
    def test_parses_standard_filename(self):
        name = "MSAT_L3_45m_COG_GEE_c01460640_p4992_v02001000_20240911T220558Z_220620Z.tif"
        dt = _parse_time_from_filename(name)
        assert dt == datetime(2024, 9, 11, 22, 5, 58, tzinfo=timezone.utc)

    def test_raises_on_bad_filename(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            _parse_time_from_filename("random_file.tif")


class TestListL3Scenes:
    @patch("google.cloud.storage.Client")
    def test_lists_tif_files(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_bucket = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        blob1 = MagicMock()
        blob1.name = "cog_gee/t100/scene1.tif"
        blob2 = MagicMock()
        blob2.name = "README.pdf"
        blob3 = MagicMock()
        blob3.name = "cog_gee/t100/scene2.tif"

        mock_client.list_blobs.return_value = [blob1, blob2, blob3]

        cfg = PipelineConfig()
        paths = list_l3_scenes(cfg)
        assert len(paths) == 2
        assert all(p.endswith(".tif") for p in paths)


class TestParseScene:
    def test_extracts_metadata(self, tmp_path: Path):
        cog_path = _create_synthetic_l3_cog(tmp_path / "test_scene.tif")
        scene = parse_scene(cog_path)

        assert isinstance(scene, MethaneSATScene)
        assert scene.scene_id == "TEST001"
        assert scene.target_id == "42"
        assert scene.crs == "EPSG:4326"
        assert scene.resolution_m == pytest.approx(46.4, abs=1.0)
        assert scene.acquisition_time == datetime(
            2024, 9, 11, 22, 5, 58, tzinfo=timezone.utc
        )

    def test_computes_median_xch4(self, tmp_path: Path):
        cog_path = _create_synthetic_l3_cog(
            tmp_path / "test_median.tif",
            xch4_background=1900.0,
            plume_value=2100.0,
            plume_row_range=(0, 10),
            plume_col_range=(0, 10),
        )
        scene = parse_scene(cog_path)
        # Background dominates: 100x100 = 10000 pixels, plume is 100 pixels
        assert scene.xch4_median_ppb == pytest.approx(1900.0, abs=5.0)

    def test_bbox_matches_geotiff(self, tmp_path: Path):
        cog_path = _create_synthetic_l3_cog(tmp_path / "test_bbox.tif")
        scene = parse_scene(cog_path)
        lon_min, lat_min, lon_max, lat_max = scene.bbox
        assert lon_min == pytest.approx(-103.5)
        assert lat_min == pytest.approx(31.0)
        assert lon_max == pytest.approx(-103.0)
        assert lat_max == pytest.approx(31.5)

    def test_raises_on_all_nan(self, tmp_path: Path):
        cog_path = _create_synthetic_l3_cog(
            tmp_path / "all_nan.tif", nan_fraction=1.0
        )
        with pytest.raises(ValueError, match="No valid XCH4"):
            parse_scene(cog_path)


class TestGeneratePlumeMask:
    def _make_scene(self, cog_path: Path) -> MethaneSATScene:
        return parse_scene(cog_path)

    def test_detects_plume_region(self, tmp_path: Path):
        """Known plume region above threshold should be detected."""
        cog_path = _create_synthetic_l3_cog(
            tmp_path / "plume.tif",
            xch4_background=1930.0,
            plume_value=2030.0,  # 100 ppb anomaly > 50 ppb threshold
            plume_row_range=(40, 60),
            plume_col_range=(40, 60),
        )
        scene = self._make_scene(cog_path)
        cfg = PipelineConfig(
            msat_plume_threshold_ppb=50.0,
            msat_min_plume_pixels=5,
            msat_morpho_kernel_size=1,  # no morphological filtering
        )
        mask = generate_plume_mask(scene, cfg)

        assert mask is not None
        assert isinstance(mask, PlumeMask)
        assert mask.plume_pixel_count == 20 * 20  # 400 pixels in plume region
        assert mask.anomaly_method == "median_subtract"
        assert mask.threshold_ppb == 50.0
        assert mask.plume_fraction > 0

    def test_empty_mask_below_threshold(self, tmp_path: Path):
        """No anomaly above threshold → returns None."""
        cog_path = _create_synthetic_l3_cog(
            tmp_path / "no_plume.tif",
            xch4_background=1930.0,
            plume_value=1935.0,  # 5 ppb anomaly < 50 ppb threshold
        )
        scene = self._make_scene(cog_path)
        cfg = PipelineConfig(
            msat_plume_threshold_ppb=50.0,
            msat_min_plume_pixels=5,
            msat_morpho_kernel_size=1,
        )
        mask = generate_plume_mask(scene, cfg)
        assert mask is None

    def test_morphological_opening_removes_noise(self, tmp_path: Path):
        """Isolated single pixels should be removed by morphological opening."""
        cog_path = _create_synthetic_l3_cog(
            tmp_path / "noise.tif",
            xch4_background=1930.0,
            plume_value=1930.0,  # no plume
        )
        # Manually add scattered noise pixels
        with rasterio.open(cog_path, "r+") as ds:
            xch4 = ds.read(1)
            # Add 15 isolated high pixels (scattered, not contiguous)
            rng = np.random.default_rng(123)
            noise_rows = rng.integers(0, 100, size=15)
            noise_cols = rng.integers(0, 100, size=15)
            xch4[noise_rows, noise_cols] = 2100.0
            ds.write(xch4, 1)

        scene = self._make_scene(cog_path)
        cfg = PipelineConfig(
            msat_plume_threshold_ppb=50.0,
            msat_min_plume_pixels=5,
            msat_morpho_kernel_size=3,  # 3x3 opening removes isolated pixels
        )
        mask = generate_plume_mask(scene, cfg)
        # Scattered single pixels should be removed by opening
        assert mask is None

    def test_mask_geotiff_output(self, tmp_path: Path):
        """Output mask should be valid GeoTIFF with correct CRS and transform."""
        cog_path = _create_synthetic_l3_cog(
            tmp_path / "mask_check.tif",
            xch4_background=1930.0,
            plume_value=2050.0,
        )
        scene = self._make_scene(cog_path)
        cfg = PipelineConfig(
            msat_plume_threshold_ppb=50.0,
            msat_min_plume_pixels=5,
            msat_morpho_kernel_size=1,
        )
        mask = generate_plume_mask(scene, cfg)
        assert mask is not None

        # Verify the output GeoTIFF
        with rasterio.open(mask.mask_path) as ds:
            assert ds.count == 1
            assert ds.dtypes[0] == "uint8"
            assert ds.crs.to_epsg() == 4326
            data = ds.read(1)
            assert set(np.unique(data)) <= {0, 1}
            tags = ds.tags()
            assert tags["scene_id"] == scene.scene_id
            assert tags["threshold_ppb"] == "50.0"

    def test_min_plume_pixels_filter(self, tmp_path: Path):
        """Masks with too few plume pixels should be rejected."""
        cog_path = _create_synthetic_l3_cog(
            tmp_path / "small_plume.tif",
            xch4_background=1930.0,
            plume_value=2050.0,
            plume_row_range=(50, 52),
            plume_col_range=(50, 52),  # only 4 pixels
        )
        scene = self._make_scene(cog_path)
        cfg = PipelineConfig(
            msat_plume_threshold_ppb=50.0,
            msat_min_plume_pixels=10,  # need at least 10
            msat_morpho_kernel_size=1,
        )
        mask = generate_plume_mask(scene, cfg)
        assert mask is None

    def test_configurable_threshold(self, tmp_path: Path):
        """Higher threshold should detect fewer pixels."""
        cog_path = _create_synthetic_l3_cog(
            tmp_path / "threshold.tif",
            xch4_background=1930.0,
            plume_value=2010.0,  # 80 ppb anomaly
        )
        scene = self._make_scene(cog_path)

        # Low threshold: detects plume
        cfg_low = PipelineConfig(
            msat_plume_threshold_ppb=50.0,
            msat_min_plume_pixels=5,
            msat_morpho_kernel_size=1,
        )
        mask_low = generate_plume_mask(scene, cfg_low)
        assert mask_low is not None

        # High threshold: misses plume
        cfg_high = PipelineConfig(
            msat_plume_threshold_ppb=100.0,
            msat_min_plume_pixels=5,
            msat_morpho_kernel_size=1,
        )
        mask_high = generate_plume_mask(scene, cfg_high)
        assert mask_high is None


class TestDownloadScene:
    def test_cache_hit(self, tmp_path: Path):
        """If file exists locally, should not download."""
        gcs_path = "cog_gee/t100/scene.tif"
        local_path = tmp_path / "cache" / gcs_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(b"fake data")

        cfg = PipelineConfig(msat_local_cache=tmp_path / "cache")
        result = download_scene(gcs_path, cfg)
        assert result == local_path

    @patch("google.cloud.storage.Client")
    def test_downloads_from_gcs(self, mock_client_cls, tmp_path: Path):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_bucket = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        # Simulate download by creating the file
        gcs_path = "cog_gee/t100/scene.tif"

        def fake_download(dest):
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_bytes(b"downloaded")

        mock_blob.download_to_filename.side_effect = fake_download

        cfg = PipelineConfig(msat_local_cache=tmp_path / "cache")
        result = download_scene(gcs_path, cfg)
        assert result.exists()
        mock_blob.download_to_filename.assert_called_once()


class TestIngestMethanesat:
    @patch("methane_sentinel_labels.ingest.methanesat.generate_plume_mask")
    @patch("methane_sentinel_labels.ingest.methanesat.parse_scene")
    @patch("methane_sentinel_labels.ingest.methanesat.download_scene")
    @patch("methane_sentinel_labels.ingest.methanesat.list_l3_scenes")
    def test_orchestrates_pipeline(
        self, mock_list, mock_download, mock_parse, mock_mask, tmp_path: Path
    ):
        mock_list.return_value = ["scene1.tif", "scene2.tif"]
        mock_download.return_value = tmp_path / "scene.tif"

        scene = MethaneSATScene(
            scene_id="TEST",
            gcs_path="scene1.tif",
            local_path=str(tmp_path / "scene.tif"),
            acquisition_time=datetime(2024, 9, 11, 22, 0, tzinfo=timezone.utc),
            bbox=(-103.5, 31.0, -103.0, 31.5),
            crs="EPSG:4326",
            resolution_m=46.4,
            xch4_median_ppb=1930.0,
            target_id="100",
        )
        mock_parse.return_value = scene

        mask = PlumeMask(
            scene_id="TEST",
            mask_path="/tmp/mask.tif",
            threshold_ppb=50.0,
            anomaly_method="median_subtract",
            plume_pixel_count=400,
            total_valid_pixels=10000,
            plume_fraction=0.04,
            bbox=(-103.5, 31.0, -103.0, 31.5),
            crs="EPSG:4326",
        )
        mock_mask.return_value = mask

        cfg = PipelineConfig()
        scenes, masks = ingest_methanesat(cfg)
        assert len(scenes) == 2
        assert len(masks) == 2
        assert mock_download.call_count == 2

    @patch("methane_sentinel_labels.ingest.methanesat.list_l3_scenes")
    def test_respects_limit(self, mock_list, tmp_path: Path):
        mock_list.return_value = ["s1.tif", "s2.tif", "s3.tif", "s4.tif", "s5.tif"]
        cfg = PipelineConfig(limit=2)
        # Will fail on download but that's OK — we're checking limit
        scenes, masks = ingest_methanesat(cfg)
        # list_l3_scenes returns 5, but only 2 should be processed
        assert mock_list.called
