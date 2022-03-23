from typing import Dict
from pathlib import Path

import segmentation_models_pytorch as smp

import torch
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Accuracy, Loss
from jarvis.db.figshare import data
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler

from atomvision.data.stem import Jarvis2dSTEMDataset

print("fine-tune a ResNet18 UNet")
# fine-tune a ResNet18 starting from an imagenet encoder
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    encoder_depth=3,
    decoder_channels=(64, 32, 16),
    in_channels=3,
    classes=1,
)
preprocess_input = get_preprocessing_fn("resnet18", pretrained="imagenet")


def to_tensor(x):
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


def prepare_batch(
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


checkpoint_dir = Path("models/test")


j2d = Jarvis2dSTEMDataset(
    label_mode="radius",
    rotation_degrees=90,
    shift_angstrom=0.5,
    zoom_pct=5,
    to_tensor=to_tensor,
)

batch_size = 32

train_loader = DataLoader(
    j2d,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(j2d.train_ids),
    drop_last=True,
)
val_loader = DataLoader(
    j2d, batch_size=batch_size, sampler=SubsetRandomSampler(j2d.val_ids)
)


optimizer = torch.optim.AdamW(
    [
        {"params": model.encoder.parameters(), "lr": 3e-5},
        {"params": model.decoder.parameters()},
        {"params": model.segmentation_head.parameters()},
    ],
    lr=1e-3,
)

criterion = nn.BCEWithLogitsLoss()


def acc_transform(output):
    y_pred, y_true = output
    pred = torch.sigmoid(y_pred) > 0.5
    return pred.type(torch.float32), y_true


val_metrics = {
    "accuracy": Accuracy(output_transform=acc_transform),
    "nll": Loss(criterion),
}

trainer = create_supervised_trainer(
    model, optimizer, criterion, prepare_batch=prepare_batch
)
evaluator = create_supervised_evaluator(
    model, metrics=val_metrics, prepare_batch=prepare_batch
)

to_save = {
    "model": model,
    "optimizer": optimizer,
    # "lr_scheduler": scheduler,
    # "trainer": trainer,
}
handler = Checkpoint(
    to_save,
    DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
    n_saved=2,
    global_step_transform=lambda *_: trainer.state.epoch,
)
trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)


@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(trainer):
    print(
        f"Epoch[{trainer.state.epoch}.{trainer.state.iteration}] Loss: {trainer.state.output:.2f}"
    )


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    print("evaluating")
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    print(
        f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg nll loss: {metrics['nll']:.2f}",
    )
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print(
        f"Val Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg nll loss: {metrics['nll']:.2f}",
    )
    print()


# trainer.run(train_loader, max_epochs=20)
