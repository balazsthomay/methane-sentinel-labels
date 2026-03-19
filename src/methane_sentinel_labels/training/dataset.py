"""PyTorch Dataset for methane plume segmentation training."""

import logging
from pathlib import Path

import albumentations as A
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from methane_sentinel_labels.models import TrainingPatch

logger = logging.getLogger(__name__)

# Default augmentations: geometric only (preserve spectral signal)
_DEFAULT_AUGMENT = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
])


class MethanePlumeDataset(Dataset):
    """Dataset that reads multi-band GeoTIFF training patches.

    Each patch contains [B02, B03, B04, B8A, B11, B12, varon, mask].
    The dataset selects the specified input channels and returns the mask
    as the target.
    """

    def __init__(
        self,
        records: list[TrainingPatch],
        base_dir: Path,
        *,
        input_channels: tuple[str, ...] = ("varon", "B11", "B12", "B8A"),
        augment: bool = False,
    ):
        self.records = records
        self.base_dir = base_dir
        self.input_channels = input_channels
        self.augment = augment
        self._transform = _DEFAULT_AUGMENT if augment else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[idx]
        patch_path = self.base_dir / record.patch_path

        with rasterio.open(patch_path) as ds:
            all_data = ds.read()  # (bands, H, W)
            band_names = [ds.descriptions[i] or f"band_{i+1}" for i in range(ds.count)]

        name_to_idx = {name: i for i, name in enumerate(band_names)}

        # Select input channels
        channel_indices = []
        for ch in self.input_channels:
            if ch in name_to_idx:
                channel_indices.append(name_to_idx[ch])
            else:
                raise ValueError(
                    f"Channel '{ch}' not found in {patch_path}. "
                    f"Available: {band_names}"
                )

        input_data = all_data[channel_indices]  # (C, H, W)

        # Mask is always the last band named "mask"
        if "mask" in name_to_idx:
            mask = all_data[name_to_idx["mask"]]
        else:
            raise ValueError(f"No 'mask' band found in {patch_path}")

        # Apply augmentations (need HWC format for albumentations)
        if self._transform is not None:
            input_hwc = input_data.transpose(1, 2, 0)  # (H, W, C)
            transformed = self._transform(image=input_hwc, mask=mask)
            input_data = transformed["image"].transpose(2, 0, 1)  # back to (C, H, W)
            mask = transformed["mask"]

        input_tensor = torch.from_numpy(input_data.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)  # (1, H, W)

        return input_tensor, mask_tensor


def create_basin_split(
    records: list[TrainingPatch],
    scenes: list,
    *,
    test_basins: set[str],
    val_fraction: float = 0.15,
) -> tuple[list[TrainingPatch], list[TrainingPatch], list[TrainingPatch]]:
    """Split records into train/val/test by geographic basin (target_id).

    Test set: all records from test_basins.
    Val set: val_fraction of remaining records (randomly sampled by basin).
    Train set: the rest.
    """
    # Build scene_id → target_id lookup
    scene_to_target = {s.scene_id: s.target_id for s in scenes}

    test_records = []
    remaining = []
    for r in records:
        target = scene_to_target.get(r.msat_scene_id, "")
        if target in test_basins:
            test_records.append(r)
        else:
            remaining.append(r)

    # Split remaining into val/train by basin
    remaining_targets = {
        scene_to_target.get(r.msat_scene_id, "") for r in remaining
    }
    remaining_targets = sorted(remaining_targets)

    rng = np.random.default_rng(42)
    n_val_basins = max(1, int(len(remaining_targets) * val_fraction))
    val_basins = set(
        rng.choice(remaining_targets, size=min(n_val_basins, len(remaining_targets)), replace=False)
    )

    val_records = []
    train_records = []
    for r in remaining:
        target = scene_to_target.get(r.msat_scene_id, "")
        if target in val_basins:
            val_records.append(r)
        else:
            train_records.append(r)

    logger.info(
        "Basin split: %d train, %d val, %d test (test_basins=%s)",
        len(train_records),
        len(val_records),
        len(test_records),
        test_basins,
    )

    return train_records, val_records, test_records
