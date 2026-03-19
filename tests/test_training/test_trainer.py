"""Tests for training loop and evaluation."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from methane_sentinel_labels.training.model import create_segmentation_model, get_loss_fn
from methane_sentinel_labels.training.trainer import (
    TrainMetrics,
    evaluate_model,
    train_model,
)


def _make_synthetic_loader(n_samples: int = 4, size: int = 64, in_channels: int = 4):
    """Create a DataLoader with synthetic data."""
    rng = np.random.default_rng(42)
    inputs = torch.from_numpy(
        rng.random((n_samples, in_channels, size, size)).astype(np.float32)
    )
    # Create masks with small plume regions
    masks = torch.zeros(n_samples, 1, size, size, dtype=torch.float32)
    for i in range(n_samples):
        masks[i, 0, 25:35, 25:35] = 1.0
    dataset = TensorDataset(inputs, masks)
    return DataLoader(dataset, batch_size=2, shuffle=True)


class TestTrainModel:
    def test_one_epoch(self):
        """1-epoch training on synthetic data should produce metrics."""
        model = create_segmentation_model(
            encoder="resnet18", in_channels=4, encoder_weights=None
        )
        loss_fn = get_loss_fn("focal")
        train_loader = _make_synthetic_loader(4)
        val_loader = _make_synthetic_loader(2)

        metrics = train_model(
            model, train_loader, val_loader,
            loss_fn=loss_fn, epochs=1, lr=1e-3, device="cpu",
        )
        assert len(metrics) == 1
        assert isinstance(metrics[0], TrainMetrics)
        assert metrics[0].epoch == 1
        assert metrics[0].train_loss >= 0
        assert metrics[0].val_loss >= 0
        assert 0 <= metrics[0].val_f1 <= 1
        assert 0 <= metrics[0].val_precision <= 1
        assert 0 <= metrics[0].val_recall <= 1

    def test_multi_epoch(self):
        """3-epoch training should return 3 metrics."""
        model = create_segmentation_model(
            encoder="resnet18", in_channels=4, encoder_weights=None
        )
        loss_fn = get_loss_fn("focal")
        train_loader = _make_synthetic_loader(4)
        val_loader = _make_synthetic_loader(2)

        metrics = train_model(
            model, train_loader, val_loader,
            loss_fn=loss_fn, epochs=3, lr=1e-3, device="cpu",
        )
        assert len(metrics) == 3
        assert all(m.epoch == i + 1 for i, m in enumerate(metrics))


class TestEvaluateModel:
    def test_returns_expected_keys(self):
        model = create_segmentation_model(
            encoder="resnet18", in_channels=4, encoder_weights=None
        )
        loss_fn = get_loss_fn("focal")
        loader = _make_synthetic_loader(2)

        result = evaluate_model(model, loader, loss_fn=loss_fn, device="cpu")
        assert "val_loss" in result
        assert "f1" in result
        assert "precision" in result
        assert "recall" in result
        assert "iou" in result

    def test_empty_loader(self):
        model = create_segmentation_model(
            encoder="resnet18", in_channels=4, encoder_weights=None
        )
        empty_dataset = TensorDataset(
            torch.zeros(0, 4, 64, 64), torch.zeros(0, 1, 64, 64)
        )
        loader = DataLoader(empty_dataset, batch_size=1)
        result = evaluate_model(model, loader, device="cpu")
        assert result["f1"] == 0.0
