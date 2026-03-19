"""Tests for the PyTorch Dataset and basin split."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_bounds

from methane_sentinel_labels.models import MethaneSATScene, TrainingPatch
from methane_sentinel_labels.training.dataset import (
    MethanePlumeDataset,
    create_basin_split,
)
from datetime import datetime, timezone


def _create_training_patch_tif(
    path: Path,
    *,
    height: int = 64,
    width: int = 64,
    plume_fraction: float = 0.05,
) -> None:
    """Create a synthetic 8-band training patch GeoTIFF."""
    rng = np.random.default_rng(42)
    band_names = ["B02", "B03", "B04", "B8A", "B11", "B12", "varon", "mask"]

    data = rng.random((8, height, width)).astype(np.float32) * 1000

    # Make varon band more realistic (centered at 1.0)
    data[6] = rng.normal(1.0, 0.02, (height, width)).astype(np.float32)

    # Make mask binary
    mask = np.zeros((height, width), dtype=np.float32)
    plume_size = int(np.sqrt(height * width * plume_fraction))
    start = height // 2 - plume_size // 2
    end = start + plume_size
    mask[start:end, start:end] = 1.0
    data[7] = mask

    transform = from_bounds(540000, 3570000, 541280, 3571280, width, height)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=height, width=width, count=8, dtype="float32",
        crs="EPSG:32613", transform=transform,
    ) as ds:
        for i in range(8):
            ds.write(data[i], i + 1)
            ds.set_band_description(i + 1, band_names[i])


@pytest.fixture
def training_patches(tmp_path: Path) -> tuple[list[TrainingPatch], Path]:
    """Create synthetic training patches and return records + base dir."""
    patches_dir = tmp_path / "training_patches"
    records = []
    for i in range(4):
        rel_path = f"training_patches/patch_{i}.tif"
        full_path = tmp_path / rel_path
        _create_training_patch_tif(full_path)
        records.append(
            TrainingPatch(
                msat_scene_id=f"MST{i:03d}",
                s2_scene_id=f"S2A_{i:03d}",
                patch_path=rel_path,
                bbox=(540000, 3570000, 541280, 3571280),
                crs="EPSG:32613",
                time_delta_hours=12.0 + i,
                cloud_free_fraction=0.9,
                plume_pixel_count=50,
                plume_fraction=0.05,
                band_names=("B02", "B03", "B04", "B8A", "B11", "B12", "varon", "mask"),
            )
        )
    return records, tmp_path


class TestMethanePlumeDataset:
    def test_length(self, training_patches):
        records, base_dir = training_patches
        ds = MethanePlumeDataset(records, base_dir)
        assert len(ds) == 4

    def test_getitem_shapes(self, training_patches):
        records, base_dir = training_patches
        ds = MethanePlumeDataset(
            records, base_dir, input_channels=("varon", "B11", "B12", "B8A")
        )
        inputs, mask = ds[0]
        assert inputs.shape == (4, 64, 64)  # 4 channels
        assert mask.shape == (1, 64, 64)
        assert inputs.dtype == torch.float32
        assert mask.dtype == torch.float32

    def test_mask_is_binary(self, training_patches):
        records, base_dir = training_patches
        ds = MethanePlumeDataset(records, base_dir)
        _, mask = ds[0]
        unique = torch.unique(mask)
        assert all(v in [0.0, 1.0] for v in unique.tolist())

    def test_channel_selection(self, training_patches):
        records, base_dir = training_patches
        ds = MethanePlumeDataset(
            records, base_dir, input_channels=("B11", "B12")
        )
        inputs, _ = ds[0]
        assert inputs.shape[0] == 2

    def test_augmentation_preserves_shapes(self, training_patches):
        records, base_dir = training_patches
        ds = MethanePlumeDataset(records, base_dir, augment=True)
        inputs, mask = ds[0]
        assert inputs.shape == (4, 64, 64)
        assert mask.shape == (1, 64, 64)

    def test_missing_channel_raises(self, training_patches):
        records, base_dir = training_patches
        ds = MethanePlumeDataset(
            records, base_dir, input_channels=("nonexistent",)
        )
        with pytest.raises(ValueError, match="not found"):
            ds[0]


class TestCreateBasinSplit:
    def test_no_leak_between_splits(self):
        """No basin (target_id) should appear in more than one split."""
        scenes = [
            MethaneSATScene(
                scene_id=f"MST{i:03d}", gcs_path="", local_path="",
                acquisition_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                bbox=(0, 0, 1, 1), crs="EPSG:4326", resolution_m=46.4,
                xch4_median_ppb=1930.0, target_id=f"t{i // 2}",
            )
            for i in range(10)
        ]
        records = [
            TrainingPatch(
                msat_scene_id=f"MST{i:03d}", s2_scene_id=f"S2_{i}",
                patch_path=f"p{i}.tif", bbox=(0, 0, 1, 1), crs="EPSG:32613",
                time_delta_hours=10.0, cloud_free_fraction=0.9,
                plume_pixel_count=50, plume_fraction=0.05,
                band_names=("varon", "B11", "B12", "B8A", "mask"),
            )
            for i in range(10)
        ]
        scene_to_target = {s.scene_id: s.target_id for s in scenes}

        train, val, test = create_basin_split(
            records, scenes, test_basins={"t0"}, val_fraction=0.2
        )

        train_targets = {scene_to_target[r.msat_scene_id] for r in train}
        val_targets = {scene_to_target[r.msat_scene_id] for r in val}
        test_targets = {scene_to_target[r.msat_scene_id] for r in test}

        assert test_targets == {"t0"}
        assert len(train_targets & test_targets) == 0
        assert len(val_targets & test_targets) == 0
        assert len(train_targets & val_targets) == 0

    def test_all_records_accounted_for(self):
        scenes = [
            MethaneSATScene(
                scene_id=f"MST{i:03d}", gcs_path="", local_path="",
                acquisition_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                bbox=(0, 0, 1, 1), crs="EPSG:4326", resolution_m=46.4,
                xch4_median_ppb=1930.0, target_id=f"t{i}",
            )
            for i in range(5)
        ]
        records = [
            TrainingPatch(
                msat_scene_id=f"MST{i:03d}", s2_scene_id=f"S2_{i}",
                patch_path=f"p{i}.tif", bbox=(0, 0, 1, 1), crs="EPSG:32613",
                time_delta_hours=10.0, cloud_free_fraction=0.9,
                plume_pixel_count=50, plume_fraction=0.05,
                band_names=("varon", "B11", "B12", "B8A", "mask"),
            )
            for i in range(5)
        ]
        train, val, test = create_basin_split(
            records, scenes, test_basins={"t0"}, val_fraction=0.2
        )
        assert len(train) + len(val) + len(test) == len(records)
