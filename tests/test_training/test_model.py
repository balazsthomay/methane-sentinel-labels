"""Tests for segmentation model setup."""

import pytest
import torch

from methane_sentinel_labels.training.model import create_segmentation_model, get_loss_fn


class TestCreateSegmentationModel:
    def test_forward_pass_shape(self):
        """Random tensor through U-Net should produce (B, 1, H, W)."""
        model = create_segmentation_model(
            encoder="resnet34", in_channels=4, encoder_weights=None
        )
        x = torch.randn(2, 4, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1, 256, 256)

    def test_different_encoder(self):
        model = create_segmentation_model(
            encoder="resnet18", in_channels=4, encoder_weights=None
        )
        x = torch.randn(1, 4, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 128, 128)

    def test_custom_in_channels(self):
        model = create_segmentation_model(
            encoder="resnet34", in_channels=6, encoder_weights=None
        )
        x = torch.randn(1, 6, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 64, 64)


class TestGetLossFn:
    def test_focal_loss(self):
        loss_fn = get_loss_fn("focal")
        pred = torch.randn(2, 1, 32, 32)
        target = torch.zeros(2, 1, 32, 32)
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_bce_loss(self):
        loss_fn = get_loss_fn("bce")
        pred = torch.randn(2, 1, 32, 32)
        target = torch.zeros(2, 1, 32, 32)
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_unknown_loss_raises(self):
        with pytest.raises(ValueError, match="Unknown loss type"):
            get_loss_fn("unknown")
