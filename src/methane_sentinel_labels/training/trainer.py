"""Training loop and evaluation for methane plume segmentation."""

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainMetrics:
    """Metrics for a single training epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    val_f1: float
    val_precision: float
    val_recall: float


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    loss_fn: nn.Module,
    epochs: int = 50,
    lr: float = 1e-4,
    device: str = "cpu",
) -> list[TrainMetrics]:
    """Train the segmentation model.

    Returns metrics for each epoch.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    metrics_history: list[TrainMetrics] = []

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_losses: list[float] = []
        for inputs, masks in train_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses) if train_losses else 0.0

        # Validate
        val_metrics = evaluate_model(model, val_loader, loss_fn=loss_fn, device=device)
        scheduler.step(val_metrics["val_loss"])

        epoch_metrics = TrainMetrics(
            epoch=epoch,
            train_loss=float(avg_train_loss),
            val_loss=val_metrics["val_loss"],
            val_f1=val_metrics["f1"],
            val_precision=val_metrics["precision"],
            val_recall=val_metrics["recall"],
        )
        metrics_history.append(epoch_metrics)

        logger.info(
            "Epoch %d/%d: train_loss=%.4f, val_loss=%.4f, val_f1=%.4f",
            epoch,
            epochs,
            epoch_metrics.train_loss,
            epoch_metrics.val_loss,
            epoch_metrics.val_f1,
        )

    return metrics_history


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    loss_fn: nn.Module | None = None,
    device: str = "cpu",
    threshold: float = 0.5,
) -> dict:
    """Evaluate model on a DataLoader.

    Returns dict with keys: val_loss, f1, precision, recall, iou.
    """
    model = model.to(device)
    model.eval()

    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    losses: list[float] = []

    with torch.no_grad():
        for inputs, masks in loader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)

            if loss_fn is not None:
                loss = loss_fn(outputs, masks)
                losses.append(loss.item())

            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).cpu().numpy().flatten()
            targets = masks.cpu().numpy().flatten()
            all_preds.append(preds)
            all_targets.append(targets)

    all_preds_flat = np.concatenate(all_preds) if all_preds else np.array([])
    all_targets_flat = np.concatenate(all_targets) if all_targets else np.array([])

    # Handle edge cases
    if all_targets_flat.size == 0:
        return {
            "val_loss": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "iou": 0.0,
        }

    f1 = float(f1_score(all_targets_flat, all_preds_flat, zero_division=0))
    precision = float(
        precision_score(all_targets_flat, all_preds_flat, zero_division=0)
    )
    recall = float(
        recall_score(all_targets_flat, all_preds_flat, zero_division=0)
    )

    # IoU
    intersection = ((all_preds_flat == 1) & (all_targets_flat == 1)).sum()
    union = ((all_preds_flat == 1) | (all_targets_flat == 1)).sum()
    iou = float(intersection / union) if union > 0 else 0.0

    return {
        "val_loss": float(np.mean(losses)) if losses else 0.0,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "iou": iou,
    }
