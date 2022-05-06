import json
import typer
import pydantic
from typing import Dict, List, Optional, Literal
from pathlib import Path
from functools import partial
import segmentation_models_pytorch as smp
import torch
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.handlers import Checkpoint, DiskSaver
from ignite.metrics import Accuracy, Loss
from ignite.utils import setup_logger

# from jarvis.db.figshare import data

# from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler

from alignn.models import alignn

from atomvision.data.stem import Jarvis2dSTEMDataset, build_prepare_graph_batch
from atomvision.models.segmentation_utils import (
    to_tensor_resnet18,
    prepare_atom_localization_batch,
)


class DatasetSettings(pydantic.BaseSettings):
    rotation_degrees: float = 90
    shift_angstrom: float = 0.5
    zoom_pct: float = 5


class TrainingSettings(pydantic.BaseSettings):
    batch_size: int = 32
    prefetch_workers: int = 4
    epochs: int = 100
    learning_rate: float = 1e-3
    learning_rate_finetune: float = 3e-5
    n_train: Optional[int] = None
    n_val: Optional[int] = None
    n_test: Optional[int] = None
    keep_data_order: bool = False
    val_frac: float = 0.1
    test_frac: float = 0.1
    model_name: str = "alignn"
    alignn_layers: int = 1
    atom_input_features: int = 2
    output_features: int = 5


class LocalizationSettings(pydantic.BaseSettings):
    encoder_weights: str = "imagenet"
    checkpoint: str = "checkpoint"


class Config(pydantic.BaseSettings):
    dataset: DatasetSettings = DatasetSettings()
    training: TrainingSettings = TrainingSettings()
    localization: LocalizationSettings = LocalizationSettings()
    gcn: alignn.ALIGNNConfig = alignn.ALIGNNConfig(
        name=training.model_name,
        alignn_layers=training.alignn_layers,
        atom_input_features=training.atom_input_features,
        output_features=training.output_features,
    )


def get_train_val_loaders(config: Config = Config()):
    """UNet dataloader specification."""
    batch_size = config.training.batch_size

    j2d = Jarvis2dSTEMDataset(
        label_mode="radius",
        rotation_degrees=config.dataset.rotation_degrees,
        shift_angstrom=config.dataset.shift_angstrom,
        zoom_pct=config.dataset.zoom_pct,
        to_tensor=to_tensor_resnet18,
        n_train=config.training.n_train,
        n_val=config.training.n_val,
        n_test=config.training.n_test,
        val_frac=config.training.val_frac,
        test_frac=config.training.test_frac,
        keep_data_order=config.training.keep_data_order,
    )

    train_loader = DataLoader(
        j2d,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(j2d.train_ids),
        num_workers=config.training.prefetch_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        j2d,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(j2d.val_ids),
        num_workers=config.training.prefetch_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def setup_unet_optimizer(
    model, train_loader, config: TrainingSettings = TrainingSettings()
):
    """Configure Unet optimizer and scheduler for fine-tuning."""
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": config.learning_rate_finetune,
            },
            {"params": model.decoder.parameters()},
            {"params": model.segmentation_head.parameters()},
        ],
        lr=config.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[
            config.learning_rate_finetune,
            config.learning_rate,
            config.learning_rate,
        ],
        epochs=config.epochs,
        steps_per_epoch=len(train_loader),
    )

    return optimizer, scheduler


def setup_gcn_optimizer(
    model, train_loader, config: TrainingSettings = TrainingSettings()
):
    """Configure ALIGNN optimizer and scheduler for fine-tuning."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.epochs,
        steps_per_epoch=len(train_loader),
    )

    return optimizer, scheduler


def setup_accuracy(mode: Literal["binary", "categorical"]):
    softmax = partial(torch.softmax, dim=1)
    link = {"binary": torch.sigmoid, "categorical": softmax}[mode]

    def accuracy_transform(output):
        y_pred, y_true = output
        pred = link(y_pred) > 0.5
        return pred.type(torch.float32), y_true

    return accuracy_transform


def log_training_loss(engine):
    epoch = engine.state.epoch
    iteration = engine.state.iteration
    print(f"Epoch[{epoch}.{iteration}] Loss: {engine.state.output:.2f}")


def setup_evaluation(
    evaluator, dataloaders: Dict[str, DataLoader], metrics: List[str]
):
    """Close over history dictionary history:Dict[str,Dict[str,List[float]]]"""

    history = {
        "train": {m: [] for m in metrics},
        "validation": {m: [] for m in metrics},
    }

    def log_train_val_results(engine):
        epoch = engine.state.epoch

        for tag, loader in dataloaders.items():
            evaluator.run(loader)
            metrics = evaluator.state.metrics

            for m in history[tag].keys():
                history[tag][m].append(metrics[m])

            acc = metrics["accuracy"]
            nll = metrics["nll"]
            print(
                f"{tag} - Epoch:{epoch} Avg acc:{acc:.2f} Avg nlll:{nll:.2f}",
            )

    return log_train_val_results, history


cli = typer.Typer()


@cli.command()
def gcn(
    config: Optional[Path] = typer.Argument(
        Path("config.json"),
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    train_loader=None,
    val_loader=None,
):

    checkpoint_dir = config.parent
    with open(config, "r") as f:
        config = Config(**json.load(f))
    if train_loader is None:
        print("No dataloader, using STEMDataset.")
        train_loader, val_loader = get_train_val_loaders(config)

    print("config gcn", config, type(config))
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    # model setup: fine-tune a ResNet18 starting from an imagenet encoder
    localization_model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        encoder_depth=3,
        decoder_channels=(64, 32, 16),
        in_channels=3,
        classes=1,
    )

    localization_checkpoint = (
        checkpoint_dir
        / f"{config.localization.checkpoint}_{config.training.epochs}.pt"
    )
    state = torch.load(
        localization_checkpoint, map_location=torch.device("cpu")
    )
    localization_model.load_state_dict(state["model"])
    localization_model.to(device)

    prepare_batch = build_prepare_graph_batch(
        localization_model, prepare_atom_localization_batch
    )

    model = alignn.ALIGNN(config.gcn)
    model.to(device)

    # data and optimizer setup
    # train_loader, val_loader = get_train_val_loaders(config)
    optimizer, scheduler = setup_gcn_optimizer(
        model, train_loader, config.training
    )

    # task and evaluation setup
    criterion = nn.CrossEntropyLoss()
    metrics = {
        "accuracy": Accuracy(
            output_transform=setup_accuracy(mode="categorical")
        ),
        "nll": Loss(criterion),
    }

    trainer = create_supervised_trainer(
        model,
        optimizer,
        criterion,
        prepare_batch=prepare_batch,
        device=device,
    )
    evaluator = create_supervised_evaluator(
        model,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
    )

    trainer.logger = setup_logger(
        "trainer", filepath=checkpoint_dir / "train_gcn.log"
    )
    evaluator.logger = setup_logger(
        "trainer", filepath=checkpoint_dir / "train_gcn.log"
    )

    # apply learning rate scheduler
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
    )

    # configure evaluation and setup
    checkpoint_handler = Checkpoint(
        {"model": model, "optimizer": optimizer, "lr_scheduler": scheduler},
        DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
        filename_prefix="gcn",
        n_saved=2,
        global_step_transform=lambda *_: trainer.state.epoch,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)

    trainer.add_event_handler(Events.ITERATION_COMPLETED, log_training_loss)

    # configure simple history tracking...

    dataloaders = {"train": train_loader, "validation": val_loader}
    evaluation_handler, history = setup_evaluation(
        evaluator, dataloaders, metrics.keys()
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, evaluation_handler)

    trainer.run(train_loader, max_epochs=config.training.epochs)

    torch.save(history, checkpoint_dir / "gcn_metrics.pt")
    return history


@cli.command()
def localization(
    config: Optional[Path] = typer.Argument(
        Path("config.json"),
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    train_loader=None,
    val_loader=None,
):

    # if train_loader is None:
    #    print ('No dataloader, using STEMDataset.')
    #    train_loader, val_loader = get_train_val_loaders(Config(config))
    prepare_batch = prepare_atom_localization_batch

    checkpoint_dir = config.parent
    with open(config, "r") as f:
        config = Config(**json.load(f))
    if train_loader is None:
        print("No dataloader, using STEMDataset.")
        train_loader, val_loader = get_train_val_loaders(config)
    print("config localization", config, type(config))
    # print(config)

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
    model.to(device)

    # data and optimizer setup
    # train_loader, val_loader = get_train_val_loaders(config)
    optimizer, scheduler = setup_unet_optimizer(
        model, train_loader, config.training
    )

    # task and evaluation setup
    criterion = nn.BCEWithLogitsLoss()
    metrics = {
        "accuracy": Accuracy(output_transform=setup_accuracy(mode="binary")),
        "nll": Loss(criterion),
    }

    trainer = create_supervised_trainer(
        model,
        optimizer,
        criterion,
        prepare_batch=prepare_batch,
        device=device,
    )
    evaluator = create_supervised_evaluator(
        model,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
    )

    trainer.logger = setup_logger(
        "trainer", filepath=checkpoint_dir / "train.log"
    )
    evaluator.logger = setup_logger(
        "trainer", filepath=checkpoint_dir / "train.log"
    )

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

    # configure simple history tracking...

    dataloaders = {"train": train_loader, "validation": val_loader}
    evaluation_handler, history = setup_evaluation(
        evaluator, dataloaders, metrics.keys()
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, evaluation_handler)

    trainer.run(train_loader, max_epochs=config.training.epochs)

    torch.save(history, checkpoint_dir / "localization_metrics.pt")
    return history


if __name__ == "__main__":
    cli()
