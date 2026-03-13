"""Visualization utilities for methane plume patches."""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from methane_sentinel_labels.models import PatchRecord

logger = logging.getLogger(__name__)


def visualize_patch(
    record: PatchRecord,
    output_dir: Path,
    *,
    patches_base: Path | None = None,
) -> Path | None:
    """Create a side-by-side visualization of a patch.

    Left: RGB true color (B04, B03, B02).
    Right: SWIR false color (B12, B11, B8A) — highlights methane absorption.
    Detection point marked as a crosshair on both panels.

    Returns the path to the saved figure, or None on error.
    """
    base = patches_base or output_dir
    tif_path = base / record.patch_path
    if not tif_path.exists():
        logger.warning("Patch file not found: %s", tif_path)
        return None

    try:
        bands, band_names = _read_patch_bands(tif_path)
    except Exception:
        logger.exception("Failed to read patch %s", tif_path)
        return None

    name_to_idx = {name: i for i, name in enumerate(band_names)}

    rgb = _compose_rgb(bands, name_to_idx, ["B04", "B03", "B02"])
    swir = _compose_rgb(bands, name_to_idx, ["B12", "B11", "B8A"])

    if rgb is None and swir is None:
        logger.warning("No composites possible for %s", record.patch_path)
        return None

    # Detection is at the center of the patch (by construction)
    h, w = bands.shape[1], bands.shape[2]
    cx, cy = w / 2, h / 2

    n_panels = sum(x is not None for x in [rgb, swir])
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0
    if rgb is not None:
        axes[panel_idx].imshow(rgb)
        axes[panel_idx].plot(cx, cy, "+", color="red", markersize=14, markeredgewidth=2)
        axes[panel_idx].set_title("RGB True Color")
        axes[panel_idx].set_axis_off()
        panel_idx += 1

    if swir is not None:
        axes[panel_idx].imshow(swir)
        axes[panel_idx].plot(cx, cy, "+", color="red", markersize=14, markeredgewidth=2)
        axes[panel_idx].set_title("SWIR False Color (CH₄ sensitive)")
        axes[panel_idx].set_axis_off()
        panel_idx += 1

    # Metadata subtitle
    parts = [f"Δt = {record.time_delta_hours:.1f}h"]
    if record.emission_rate_kg_hr is not None:
        parts.append(f"Q = {record.emission_rate_kg_hr:.0f} kg/hr")
    parts.append(f"cloud-free = {record.cloud_free_fraction:.0%}")
    fig.suptitle(
        f"{record.detection_source_id}  |  {record.scene_id}\n"
        + "  ·  ".join(parts),
        fontsize=10,
    )
    fig.tight_layout()

    viz_dir = output_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    out_path = viz_dir / f"{record.detection_source_id}_{record.scene_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved visualization: %s", out_path)
    return out_path


def visualize_dataset(
    records: list[PatchRecord],
    output_dir: Path,
    *,
    max_plots: int | None = None,
) -> list[Path]:
    """Generate visualizations for multiple patches."""
    to_plot = records[:max_plots] if max_plots else records
    paths: list[Path] = []
    for record in to_plot:
        path = visualize_patch(record, output_dir)
        if path is not None:
            paths.append(path)
    logger.info("Generated %d visualizations", len(paths))
    return paths


def _read_patch_bands(tif_path: Path) -> tuple[np.ndarray, list[str]]:
    """Read all bands and their descriptions from a GeoTIFF."""
    with rasterio.open(tif_path) as ds:
        data = ds.read()  # (bands, H, W)
        names = [ds.descriptions[i] or f"band_{i+1}" for i in range(ds.count)]
    return data, names


def _compose_rgb(
    bands: np.ndarray,
    name_to_idx: dict[str, int],
    channel_names: list[str],
) -> np.ndarray | None:
    """Compose a 3-channel RGB image from named bands.

    Applies percentile stretching for display.
    """
    indices = []
    for name in channel_names:
        if name not in name_to_idx:
            return None
        indices.append(name_to_idx[name])

    rgb = np.stack([bands[i] for i in indices], axis=-1).astype(np.float32)

    # Percentile stretch per channel
    for c in range(3):
        ch = rgb[:, :, c]
        valid = ch[ch > 0]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, [2, 98])
        if hi > lo:
            rgb[:, :, c] = np.clip((ch - lo) / (hi - lo), 0, 1)
        else:
            rgb[:, :, c] = 0

    return rgb
