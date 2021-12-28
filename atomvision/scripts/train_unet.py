import torch
from torch import nn
import skimage
from jarvis.db.figshare import data
from typing import Dict
from ignite.metrics import Accuracy, Loss
from ignite.engine import (
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

import matplotlib.pyplot as plt

# from atomvision.data.stemconv import STEMConv
from atomvision.data.stem import Jarvis2dSTEMDataset

# fine-tune an ResNet18 starting from an imagenet encoder
# model(preprocess_input(I).permute((2, 0, 1)).type(torch.FloatTensor).unsqueeze(0))
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
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


my_data = data("dft_2d")
test_perc = 10
n_train = int(len(my_data) * (100 - test_perc) / 100)
train_data = my_data[0:n_train]
val_data = my_data[n_train : len(my_data)]
train_set = Jarvis2dSTEMDataset(
    image_data=train_data, label_mode="radius", to_tensor=to_tensor
)
val_set = Jarvis2dSTEMDataset(
    image_data=val_data, label_mode="radius", to_tensor=to_tensor
)
batch_size = 64
checkpoint_dir = "."
print("n_train", len(train_data))
print("n_test", len(val_data))
# train_set = Jarvis2dSTEMDataset(image_data=train_data, label_mode="radius",to_tensor=to_tensor)

# j2d = Jarvis2dSTEMDataset(label_mode="radius", to_tensor=to_tensor)

# train_sampler = torch.utils.data.SubsetRandomSampler(list(range(16)))
# train_loader = torch.utils.data.DataLoader(j2d, batch_size=4, sampler=train_sampler)

# batch = next(iter(train_loader))
# x = model(batch["image"])
#

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)

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
    print(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output:.2f}")


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


trainer.run(train_loader, max_epochs=20)
