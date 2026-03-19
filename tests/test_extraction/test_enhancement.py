"""Tests for Varon ratio enhancement product."""

import numpy as np
import pytest

from methane_sentinel_labels.extraction.enhancement import compute_varon_ratio


class TestComputeVaronRatio:
    def test_uniform_input_gives_ones(self):
        """Uniform B11 and B12 → ratio ~1.0 everywhere."""
        b11 = np.full((100, 100), 1000.0, dtype=np.float32)
        b12 = np.full((100, 100), 800.0, dtype=np.float32)
        varon = compute_varon_ratio(b11, b12)
        assert varon.shape == (100, 100)
        np.testing.assert_allclose(varon, 1.0, atol=1e-6)

    def test_synthetic_plume_elevated(self):
        """Plume region with enhanced B12 should have ratio > 1.0."""
        b11 = np.full((100, 100), 1000.0, dtype=np.float32)
        b12 = np.full((100, 100), 800.0, dtype=np.float32)
        # Plume: B12 enhanced (methane absorbs B11 more than B12)
        b12[40:60, 40:60] = 1000.0
        varon = compute_varon_ratio(b11, b12)
        # Background should be ~1.0, plume should be > 1.0
        bg_val = np.median(varon[:20, :20])
        plume_val = np.median(varon[45:55, 45:55])
        assert plume_val > bg_val
        assert plume_val > 1.0

    def test_div_by_zero_handling(self):
        """Zero B11 should not crash — returns 1.0 for those pixels."""
        b11 = np.zeros((50, 50), dtype=np.float32)
        b12 = np.full((50, 50), 100.0, dtype=np.float32)
        varon = compute_varon_ratio(b11, b12)
        assert not np.any(np.isnan(varon))
        assert not np.any(np.isinf(varon))

    def test_output_dtype(self):
        b11 = np.ones((10, 10), dtype=np.uint16) * 1000
        b12 = np.ones((10, 10), dtype=np.uint16) * 800
        varon = compute_varon_ratio(b11, b12)
        assert varon.dtype == np.float32

    def test_mixed_zero_and_valid(self):
        """Mix of valid and zero pixels."""
        b11 = np.full((100, 100), 1000.0, dtype=np.float32)
        b12 = np.full((100, 100), 800.0, dtype=np.float32)
        b11[:10, :] = 0  # zero region
        varon = compute_varon_ratio(b11, b12)
        # Valid region should still be ~1.0
        valid_varon = varon[20:80, 20:80]
        np.testing.assert_allclose(valid_varon, 1.0, atol=0.01)

    def test_unknown_method_raises(self):
        b11 = np.ones((10, 10), dtype=np.float32)
        b12 = np.ones((10, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown reference method"):
            compute_varon_ratio(b11, b12, reference_method="temporal")
