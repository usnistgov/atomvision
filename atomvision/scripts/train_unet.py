import typer
from typing import Dict, Optional
from pathlib import Path

import segmentation_models_pytorch as smp

import torch
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Accuracy, Loss
from ignite.utils import setup_logger

from jarvis.db.figshare import data
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler

from atomvision.data.stem import Jarvis2dSTEMDataset
from atomvision.models.segmentation_utils import (
    to_tensor_resnet18,
    prepare_atom_localization_batch,
)


def get_train_val_loaders(config):
    """UNet dataloader specification."""
    batch_size = 32

    j2d = Jarvis2dSTEMDataset(
        label_mode="radius",
        rotation_degrees=90,
        shift_angstrom=0.5,
        zoom_pct=5,
        to_tensor=to_tensor_resnet18,
    )

    train_loader = DataLoader(
        j2d,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(j2d.train_ids),
        drop_last=True,
    )
    val_loader = DataLoader(
        j2d, batch_size=batch_size, sampler=SubsetRandomSampler(j2d.val_ids)
    )

    return train_loader, val_loader


def setup_unet_optimizer(model, train_loader, config):
    epochs = 100
    lr = 1e-3
    lr_finetune = 3e-5

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": lr_finetune},
            {"params": model.decoder.parameters()},
            {"params": model.segmentation_head.parameters()},
        ],
        lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[lr_finetune, lr, lr],
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )

    return optimizer, scheduler


# performance tracking
def accuracy_transform(output):
    y_pred, y_true = output
    pred = torch.sigmoid(y_pred) > 0.5
    return pred.type(torch.float32), y_true


def log_training_loss(engine):
    print(
        f"Epoch[{engine.state.epoch}.{engine.state.iteration}] Loss: {engine.state.output:.2f}"
    )


def setup_evaluation(evaluator, train_loader, val_loader):
    def log_train_val_results(engine):
        print("evaluating")
        epoch = engine.state.epoch
        for phase, loader in zip(("train", "val"), (train_loader, val_loader)):
            evaluator.run(loader)
            metrics = evaluator.state.metrics
            acc = metrics["accuracy"]
            nll = metrics["nll"]
            print(
                f"{phase} results - Epoch: {epoch}  Avg accuracy: {acc:.2f} Avg nll loss: {nll:.2f}",
            )

    return log_train_val_results


def main(
    config: Optional[Path] = typer.Argument(
        Path("models/test/config.json"), exists=True, file_okay=True, dir_okay=False
    )
):
    print(config)
    checkpoint_dir = config.parent

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    # model setup: fine-tune a ResNet18 starting from an imagenet encoder
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        encoder_depth=3,
        decoder_channels=(64, 32, 16),
        in_channels=3,
        classes=1,
    )
    # preprocess_input = get_preprocessing_fn("resnet18", pretrained="imagenet")

    # data and optimizer setup
    train_loader, val_loader = get_train_val_loaders(config)
    optimizer, scheduler = setup_unet_optimizer(model, train_loader, config)

    # task and evaluation setup
    criterion = nn.BCEWithLogitsLoss()
    val_metrics = {
        "accuracy": Accuracy(output_transform=accuracy_transform),
        "nll": Loss(criterion),
    }

    trainer = create_supervised_trainer(
        model,
        optimizer,
        criterion,
        prepare_batch=prepare_atom_localization_batch,
        device=device,
    )
    evaluator = create_supervised_evaluator(
        model,
        metrics=val_metrics,
        prepare_batch=prepare_atom_localization_batch,
        device=device,
    )

    trainer.logger = setup_logger("trainer")
    evaluator.logger = setup_logger("trainer")

    # apply learning rate scheduler
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
    )

    # configure evaluation and setup
    checkpoint_handler = Checkpoint(
        {"model": model, "optimizer": optimizer, "lr_scheduler": scheduler},
        DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
        n_saved=2,
        global_step_transform=lambda *_: trainer.state.epoch,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)

    trainer.add_event_handler(Events.ITERATION_COMPLETED, log_training_loss)
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, setup_evaluation(evaluator, train_loader, val_loader)
    )

    trainer.run(train_loader, max_epochs=20)


if __name__ == "__main__":
    typer.run(main)
