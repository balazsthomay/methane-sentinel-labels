"""Segmentation model setup for methane plume detection."""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def create_segmentation_model(
    encoder: str = "resnet34",
    in_channels: int = 4,
    *,
    encoder_weights: str | None = "imagenet",
) -> nn.Module:
    """Create a U-Net segmentation model.

    Parameters
    ----------
    encoder : str
        Encoder backbone name (e.g., "resnet34", "resnet50").
    in_channels : int
        Number of input channels (default 4: varon + B11 + B12 + B8A).
    encoder_weights : str | None
        Pretrained weights. "imagenet" or None.

    Returns
    -------
    nn.Module
        U-Net model with sigmoid activation, outputting (B, 1, H, W).
    """
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=1,
        activation=None,  # We apply sigmoid in loss or post-processing
    )
    return model


def get_loss_fn(loss_type: str = "focal") -> nn.Module:
    """Get a segmentation loss function.

    Parameters
    ----------
    loss_type : str
        "focal" — focal loss (handles class imbalance).
        "bce" — weighted binary cross-entropy.

    Returns
    -------
    nn.Module
        Loss function that accepts (prediction, target).
    """
    if loss_type == "focal":
        return smp.losses.FocalLoss(mode="binary", alpha=0.25, gamma=2.0)
    elif loss_type == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
