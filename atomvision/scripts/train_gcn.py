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

from alignn.models import alignn

from atomvision.data.stem import Jarvis2dSTEMDataset, build_prepare_graph_batch
from atomvision.models.segmentation_utils import (
    to_tensor_resnet18,
    prepare_atom_localization_batch,
)

# import warnings
# warnings.filterwarnings("error")

checkpoint_dir = Path("models/test")

# pre-trained UNet model
preprocess_input = get_preprocessing_fn("resnet18", pretrained="imagenet")
unet = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    encoder_depth=3,
    decoder_channels=(64, 32, 16),
    in_channels=3,
    classes=1,
)
state = torch.load(checkpoint_dir / "checkpoint_5.pt")
unet.load_state_dict(state["model"])
prepare_graph_batch = build_prepare_graph_batch(unet, prepare_atom_localization_batch)

j2d = Jarvis2dSTEMDataset(
    label_mode="radius",
    rotation_degrees=90,
    shift_angstrom=0.5,
    zoom_pct=5,
    to_tensor=to_tensor_resnet18,
)

cfg = alignn.ALIGNNConfig(
    name="alignn", alignn_layers=0, atom_input_features=2, output_features=j2d.n_classes
)
gcn_model = alignn.ALIGNN(cfg)

optimizer = torch.optim.AdamW(
    gcn_model.parameters(),
    lr=1e-3,
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


criterion = nn.CrossEntropyLoss()


def acc_transform(output):
    y_pred, y_true = output
    pred = torch.softmax(y_pred)
    return pred, y_true


val_metrics = {
    "accuracy": Accuracy(output_transform=acc_transform),
    "nll": Loss(criterion),
}

trainer = create_supervised_trainer(
    gcn_model, optimizer, criterion, prepare_batch=prepare_graph_batch
)
evaluator = create_supervised_evaluator(
    gcn_model, metrics=val_metrics, prepare_batch=prepare_graph_batch
)

to_save = {
    "model": gcn_model,
    "optimizer": optimizer,
    # "lr_scheduler": scheduler,
    # "trainer": trainer,
}
handler = Checkpoint(
    to_save,
    DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
    filename_prefix="gcn",
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


trainer.run(train_loader, max_epochs=20)
