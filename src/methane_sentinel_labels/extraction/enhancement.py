"""Enhancement products for methane detection in Sentinel-2 SWIR bands."""

import numpy as np


def compute_varon_ratio(
    b11: np.ndarray,
    b12: np.ndarray,
    *,
    reference_method: str = "spatial",
) -> np.ndarray:
    """Compute Varon ratio: (B12/B11) normalized by spatial reference.

    Varon et al. (2021): ratio of B12/B11, normalized against a local
    reference. Values near 1.0 = background, >1.0 = methane enhancement.

    Parameters
    ----------
    b11 : np.ndarray
        Sentinel-2 Band 11 (SWIR-1, 1610nm), 2D array.
    b12 : np.ndarray
        Sentinel-2 Band 12 (SWIR-2, 2190nm), 2D array.
    reference_method : str
        "spatial" — normalize by median ratio across the patch.

    Returns
    -------
    np.ndarray
        Varon ratio, same shape as input. Background ~1.0, plumes >1.0.
    """
    # Avoid division by zero
    safe_b11 = np.where(b11 > 0, b11, np.nan)

    ratio = b12.astype(np.float32) / safe_b11.astype(np.float32)

    if reference_method == "spatial":
        # Normalize by median ratio (excluding NaN/zero regions)
        valid = ratio[np.isfinite(ratio)]
        if valid.size == 0:
            return np.ones_like(b11, dtype=np.float32)
        ref = np.median(valid)
        if ref == 0:
            return np.ones_like(b11, dtype=np.float32)
        varon = ratio / ref
    else:
        raise ValueError(f"Unknown reference method: {reference_method}")

    # Replace NaN with 1.0 (background)
    varon = np.where(np.isfinite(varon), varon, 1.0).astype(np.float32)
    return varon
