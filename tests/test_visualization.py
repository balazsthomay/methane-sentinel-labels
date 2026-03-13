"""Tests for visualization utilities."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from methane_sentinel_labels.models import PatchRecord
from methane_sentinel_labels.visualization import (
    _compose_rgb,
    _read_patch_bands,
    visualize_dataset,
    visualize_patch,
)


@pytest.fixture
def sample_geotiff(tmp_output: Path) -> Path:
    """Create a multi-band GeoTIFF with known band descriptions."""
    patches_dir = tmp_output / "patches"
    patches_dir.mkdir()
    tif_path = patches_dir / "CM-001_S2A_20240616.tif"

    h, w = 256, 256
    # 6 bands: B02, B03, B04, B8A, B11, B12
    band_names = ["B02", "B03", "B04", "B8A", "B11", "B12"]
    rng = np.random.default_rng(42)
    data = rng.integers(100, 5000, size=(len(band_names), h, w), dtype=np.uint16)

    transform = from_bounds(540000, 3570000, 545120, 3575120, w, h)
    with rasterio.open(
        tif_path, "w", driver="GTiff",
        height=h, width=w, count=len(band_names),
        dtype="uint16", crs="EPSG:32613", transform=transform,
    ) as dst:
        for i, name in enumerate(band_names):
            dst.write(data[i], i + 1)
            dst.set_band_description(i + 1, name)

    return tif_path


@pytest.fixture
def patch_record(sample_geotiff: Path, tmp_output: Path) -> PatchRecord:
    rel_path = str(sample_geotiff.relative_to(tmp_output))
    return PatchRecord(
        detection_source_id="CM-001",
        scene_id="S2A_20240616",
        patch_path=rel_path,
        latitude=31.872,
        longitude=-103.245,
        emission_rate_kg_hr=150.0,
        time_delta_hours=20.0,
        cloud_free_fraction=0.95,
        crs="EPSG:32613",
        bbox=(540000.0, 3570000.0, 545120.0, 3575120.0),
    )


class TestReadPatchBands:
    def test_reads_all_bands(self, sample_geotiff: Path):
        data, names = _read_patch_bands(sample_geotiff)
        assert data.shape[0] == 6
        assert data.shape[1] == 256
        assert data.shape[2] == 256
        assert names == ["B02", "B03", "B04", "B8A", "B11", "B12"]


class TestComposeRgb:
    def test_creates_rgb_from_bands(self):
        bands = np.random.rand(6, 100, 100).astype(np.float32) * 5000
        name_to_idx = {"B02": 0, "B03": 1, "B04": 2, "B8A": 3, "B11": 4, "B12": 5}
        rgb = _compose_rgb(bands, name_to_idx, ["B04", "B03", "B02"])
        assert rgb is not None
        assert rgb.shape == (100, 100, 3)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_returns_none_for_missing_band(self):
        bands = np.random.rand(3, 100, 100).astype(np.float32)
        name_to_idx = {"B02": 0, "B03": 1, "B04": 2}
        result = _compose_rgb(bands, name_to_idx, ["B12", "B11", "B8A"])
        assert result is None


class TestVisualizePatch:
    def test_creates_png(self, patch_record: PatchRecord, tmp_output: Path):
        path = visualize_patch(patch_record, tmp_output)
        assert path is not None
        assert path.exists()
        assert path.suffix == ".png"

    def test_missing_file_returns_none(self, tmp_output: Path):
        record = PatchRecord(
            detection_source_id="CM-MISSING",
            scene_id="S2A_MISSING",
            patch_path="patches/nonexistent.tif",
            latitude=0.0, longitude=0.0,
            emission_rate_kg_hr=None,
            time_delta_hours=10.0,
            cloud_free_fraction=1.0,
            crs="EPSG:32613",
            bbox=(0.0, 0.0, 1.0, 1.0),
        )
        path = visualize_patch(record, tmp_output)
        assert path is None

    def test_metadata_in_filename(self, patch_record: PatchRecord, tmp_output: Path):
        path = visualize_patch(patch_record, tmp_output)
        assert "CM-001" in path.name
        assert "S2A_20240616" in path.name


class TestVisualizeDataset:
    def test_generates_multiple(self, patch_record: PatchRecord, tmp_output: Path):
        records = [patch_record, patch_record]
        paths = visualize_dataset(records, tmp_output)
        # Same record twice → same file (overwritten), so 1 unique path
        assert len(paths) >= 1

    def test_max_plots(self, patch_record: PatchRecord, tmp_output: Path):
        records = [patch_record] * 5
        paths = visualize_dataset(records, tmp_output, max_plots=2)
        assert len(paths) <= 2
