from typing import Dict

import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

preprocess_input = get_preprocessing_fn("resnet18", pretrained="imagenet")


def to_tensor_resnet18(x):
    """Image to tensor

    normalize to (0, 1)
    apply imagenet preprocessing
    permute dims (H, W, C) to (C, H, W)
    convert to float
    """
    x = x / x.max()
    x = preprocess_input(x.unsqueeze(-1))
    x = x.permute(2, 0, 1)
    return x.type(torch.FloatTensor)


def prepare_atom_localization_batch(
    batch: Dict[str, torch.Tensor],
    device=None,
    non_blocking=False,
):
    """Extract image and mask from batch dictionary."""
    image, label, ids = batch["image"], batch["label"], batch["id"]
    batch = (
        image.to(device, non_blocking=non_blocking),
        label.unsqueeze(1).to(device, non_blocking=non_blocking),
    )

    return batch
