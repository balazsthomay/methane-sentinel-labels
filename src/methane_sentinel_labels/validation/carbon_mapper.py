"""Validation of MethaneSAT-trained model against Carbon Mapper detections."""

import logging
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn

from methane_sentinel_labels.extraction.enhancement import compute_varon_ratio
from methane_sentinel_labels.models import PatchRecord

logger = logging.getLogger(__name__)


def run_inference_on_cm_patches(
    model: nn.Module,
    cm_records: list[PatchRecord],
    base_dir: Path,
    *,
    input_channels: tuple[str, ...] = ("varon", "B11", "B12", "B8A"),
    device: str = "cpu",
    threshold: float = 0.5,
) -> list[dict]:
    """Run model inference on existing Carbon Mapper patches.

    For each CM patch: read bands, compute Varon ratio on-the-fly,
    run model, compute detection score.
    """
    model = model.to(device)
    model.eval()

    results: list[dict] = []
    for record in cm_records:
        patch_path = base_dir / record.patch_path
        if not patch_path.exists():
            logger.warning("Patch not found: %s", patch_path)
            continue

        try:
            result = _infer_single_patch(
                model, record, patch_path,
                input_channels=input_channels,
                device=device,
                threshold=threshold,
            )
            results.append(result)
        except Exception:
            logger.exception("Inference failed for %s", record.patch_path)

    logger.info("Inference complete: %d/%d patches", len(results), len(cm_records))
    return results


def _infer_single_patch(
    model: nn.Module,
    record: PatchRecord,
    patch_path: Path,
    *,
    input_channels: tuple[str, ...],
    device: str,
    threshold: float,
) -> dict:
    """Run inference on a single CM patch."""
    with rasterio.open(patch_path) as ds:
        all_data = ds.read()
        band_names = [ds.descriptions[i] or f"band_{i+1}" for i in range(ds.count)]

    name_to_idx = {name: i for i, name in enumerate(band_names)}

    # Compute Varon ratio if needed
    band_arrays: dict[str, np.ndarray] = {}
    for name in band_names:
        band_arrays[name] = all_data[name_to_idx[name]]

    if "varon" not in band_arrays and "B11" in band_arrays and "B12" in band_arrays:
        band_arrays["varon"] = compute_varon_ratio(
            band_arrays["B11"], band_arrays["B12"]
        )

    # Build input tensor
    channels = []
    for ch in input_channels:
        if ch in band_arrays:
            channels.append(band_arrays[ch].astype(np.float32))
        else:
            logger.warning("Channel %s not available, using zeros", ch)
            h, w = all_data.shape[1], all_data.shape[2]
            channels.append(np.zeros((h, w), dtype=np.float32))

    input_tensor = torch.from_numpy(np.stack(channels)).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output).cpu().numpy()[0, 0]

    pred_mask = (probs > threshold).astype(np.uint8)
    detection_score = float(probs.max())
    plume_fraction = float(pred_mask.sum()) / pred_mask.size if pred_mask.size > 0 else 0.0

    return {
        "detection_source_id": record.detection_source_id,
        "scene_id": record.scene_id,
        "emission_rate_kg_hr": record.emission_rate_kg_hr,
        "detection_score": detection_score,
        "plume_fraction": plume_fraction,
        "plume_pixel_count": int(pred_mask.sum()),
        "is_detected": detection_score > threshold,
    }


def compute_validation_metrics(
    results: list[dict],
    cm_records: list[PatchRecord],
) -> dict:
    """Compute validation metrics from inference results.

    Returns detection rate, correlation with emission rate, etc.
    """
    if not results:
        return {
            "n_patches": 0,
            "detection_rate": 0.0,
            "mean_detection_score": 0.0,
        }

    scores = [r["detection_score"] for r in results]
    detected = [r for r in results if r["is_detected"]]
    emission_rates = [
        r["emission_rate_kg_hr"]
        for r in results
        if r["emission_rate_kg_hr"] is not None
    ]
    plume_fractions = [
        r["plume_fraction"]
        for r in results
        if r["emission_rate_kg_hr"] is not None
    ]

    metrics: dict = {
        "n_patches": len(results),
        "detection_rate": len(detected) / len(results),
        "mean_detection_score": float(np.mean(scores)),
        "median_detection_score": float(np.median(scores)),
        "n_with_emission_rate": len(emission_rates),
    }

    # Correlation between plume fraction and emission rate
    if len(emission_rates) >= 3:
        corr = np.corrcoef(emission_rates, plume_fractions)[0, 1]
        metrics["emission_plume_correlation"] = (
            float(corr) if np.isfinite(corr) else 0.0
        )

    return metrics
