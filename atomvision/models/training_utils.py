"""Module for training utils."""
import pydantic
from typing import Dict, List, Literal
from functools import partial
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from atomvision.models.segmentation_utils import to_tensor_resnet18
from alignn.models import alignn
from atomvision.data.stem import Jarvis2dSTEMDataset


class DatasetSettings(pydantic.BaseSettings):
    """Module for DatasetSettings."""

    rotation_degrees: float = 90
    shift_angstrom: float = 0.5
    zoom_pct: float = 5
    label_mode: str = "radius"


class TrainingSettings(pydantic.BaseSettings):
    """Module for TrainingSettings."""

    batch_size: int = 32
    prefetch_workers: int = 4
    epochs: int = 100
    learning_rate: float = 1e-3
    learning_rate_finetune: float = 3e-5
    n_train: int = None
    n_val: int = None
    n_test: int = None
    keep_data_order: bool = False
    val_frac: float = 0.1
    test_frac: float = 0.1
    output_feats: int = 1
    atom_input_feats: int = 4
    nlayers_alignn: int = 4


class LocalizationSettings(pydantic.BaseSettings):
    """Module for LocalizationSettings."""

    encoder_weights: str = "imagenet"
    checkpoint: str = "checkpoint"


class Config(pydantic.BaseSettings):
    """Module for Config Settings."""

    dataset: DatasetSettings = DatasetSettings()
    training: TrainingSettings = TrainingSettings()
    localization: LocalizationSettings = LocalizationSettings()
    print("training.output_feats", training.output_feats)
    gcn: alignn.ALIGNNConfig = alignn.ALIGNNConfig(
        name="alignn",
        alignn_layers=training.nlayers_alignn,
        atom_input_features=training.atom_input_feats,
        output_features=training.output_feats,
    )


def get_train_val_loaders(config: Config = Config(), localization_model=None):
    """Get UNet dataloader specification."""
    batch_size = config.training.batch_size

    j2d = Jarvis2dSTEMDataset(
        label_mode=config.dataset.label_mode,
        rotation_degrees=config.dataset.rotation_degrees,
        shift_angstrom=config.dataset.shift_angstrom,
        zoom_pct=config.dataset.zoom_pct,
        to_tensor=to_tensor_resnet18,
        localization_model=localization_model,
        n_train=config.training.n_train,
        n_val=config.training.n_val,
        n_test=config.training.n_test,
        val_frac=config.training.val_frac,
        test_frac=config.training.test_frac,
        keep_data_order=config.training.keep_data_order,
    )
    # print("localization_model", localization_model)
    # print("j2d.model", j2d.model)
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


def setup_optimizer(
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
    """Set up accuracy."""
    softmax = partial(torch.softmax, dim=1)
    link = {"binary": torch.sigmoid, "categorical": softmax}[mode]

    def accuracy_transform(output):
        y_pred, y_true = output
        pred = link(y_pred) > 0.5
        return pred.type(torch.float32), y_true

    return accuracy_transform


def log_training_loss(engine):
    """Record traiing loss."""
    epoch = engine.state.epoch
    iteration = engine.state.iteration
    print(f"Epoch[{epoch}.{iteration}] Loss: {engine.state.output:.2f}")


def setup_evaluation(
    evaluator, dataloaders: Dict[str, DataLoader], metrics: List[str]
):
    """Close over history dictionary history."""
    # Dict[str,Dict[str,List[float]]]."""
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


# Output Operations for Neural Network:
def thresholded_output_transform(output):
    """Round off output."""
    y_pred, y = output
    y_pred = torch.round(torch.exp(y_pred))
    # print ('output',y_pred)
    return y_pred, y


def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def activated_output_transform(output):
    """Exponentiate output."""
    y_pred, y = output
    y_pred = torch.exp(y_pred)
    y_pred = y_pred[:, 1]
    return y_pred, y
