#!/usr/bin/env python
"""Module to perform segmentation on atomistic dataset."""
import torch
import os
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from jarvis.core.lattice import get_2d_lattice
import numpy as np
import pandas as pd
from skimage import draw
from jarvis.core.atoms import Atoms
from jarvis.core.specie import chem_data
from jarvis.db.figshare import data
from atomvision.data.stemconv import STEMConv
from collections.abc import Callable
from typing import Optional, List, Dict, Literal
import json
import pydantic
from functools import partial
import segmentation_models_pytorch as smp
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.handlers import Checkpoint, DiskSaver
from ignite.metrics import Accuracy, Loss
from ignite.utils import setup_logger
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import random
import ignite

random_seed = 123
ignite.utils.manual_seed(random_seed)
torch.manual_seed(random_seed)
random.seed(0)
np.random.seed(random_seed)
torch.backends.cudnn.deterministic = True


class DatasetSettings(pydantic.BaseSettings):
    rotation_degrees: float = 90
    shift_angstrom: float = 0.5
    zoom_pct: float = 5


class TrainingSettings(pydantic.BaseSettings):
    parent_dir: str = "."
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
    output_feats: int = 5
    atom_input_feats: int = 2
    nlayers_alignn: int = 0


class LocalizationSettings(pydantic.BaseSettings):
    encoder_weights: str = "imagenet"
    checkpoint: str = "checkpoint"


class Config(pydantic.BaseSettings):
    dataset: DatasetSettings = DatasetSettings()
    training: TrainingSettings = TrainingSettings()
    localization: LocalizationSettings = LocalizationSettings()
    # print("training.output_feats", training.output_feats)
    # gcn: alignn.ALIGNNConfig = alignn.ALIGNNConfig(
    #    name="alignn",
    #    alignn_layers=training.nlayers_alignn,
    #    atom_input_features=training.atom_input_feats,
    #    output_features=training.output_feats,
    # )


preprocess_input = get_preprocessing_fn("resnet18", pretrained="imagenet")
LABEL_MODES = {"delta", "radius"}

# atomic radii
pt = pd.DataFrame(chem_data).T
pt = pt.sort_values(by="Z")
RADII = {int(row.Z): row.atom_rad for id, row in pt.iterrows()}


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
    if len(ids) < 1:
        raise ValueError("ids length", len(ids))
    batch = (
        image.to(device, non_blocking=non_blocking),
        label.unsqueeze(1).to(device, non_blocking=non_blocking),
    )

    return batch


def atomic_radius_mask(shape, X, N, px_scale=0.1):
    """Atom localization masks, with footprints scaled to atomic radii.
    Atoms occluding each other along the Z (transmission) dimension are
    not guaranteed to be masked nicely; these are not multilabel masks
    """
    labels = np.zeros(shape, dtype=int)
    for x, n in zip(X, N):

        rr, cc = draw.disk(
            tuple(x), 0.5 * RADII[n] / px_scale, shape=labels.shape
        )
        labels[rr, cc] = n

    return labels


class Jarvis2dSTEMDataset:
    """Simulated STEM dataset (jarvis dft_2d)"""

    def __init__(
        self,
        px_scale: float = 0.1,
        label_mode: str = "delta",
        image_data=[],
        rotation_degrees: Optional[float] = None,
        shift_angstrom: Optional[float] = None,
        zoom_pct: Optional[float] = None,
        to_tensor: Optional[Callable] = None,
        n_train=None,
        n_val=None,
        n_test=None,
        val_frac=0.1,
        test_frac=0.1,
        keep_data_order=False,
    ):
        """Simulated STEM dataset, jarvis-2d data
        px_scale: pixel size in angstroms
        label_mode: `delta` or `radius`, controls atom localization mask style
        ## augmentation settings
        rotation_degrees: if specified, sample from
        Unif(-rotation_degrees, rotation_degrees)
        shift_angstrom: if specified, sample from
        Unif(-shift_angstrom, shift_angstrom)
        zoom_pct: optional image scale factor: s *= 1 + (zoom_pct/100)
        """
        print("n_train", n_train)
        # import sys
        # sys.exit()
        if label_mode not in LABEL_MODES:
            raise NotImplementedError(f"label mode {label_mode} not supported")

        self.px_scale = px_scale
        self.label_mode = label_mode
        self.to_tensor = to_tensor

        self.rotation_degrees = rotation_degrees
        self.shift_angstrom = shift_angstrom
        self.zoom_pct = zoom_pct
        self.keep_data_order = keep_data_order

        # image_data as array of dicts, with atoms,crys and jid info
        """
        tmp_data = []
        dft_2d = data('dft_2d')
        for i in dft_2d:
            info={}
            info['jid']=i['jid']
            info['atoms']=i['atoms']
            info['crys']=get_2d_lattice(i['atoms'])[1]
            tmp_data.append(info)
        image_data = tmp_data
        """
        self.data = pd.DataFrame(image_data)
        self.df = pd.DataFrame(image_data)
        # if image_data is not None:
        #    self.df = pd.DataFrame(image_data)
        # else:
        #    # dft_2d = data("dft_2d")
        #    # Overriding the crys with 2D lattice type
        #    self.df = pd.DataFrame(data("dft_2d"))
        #    self.df["crys"] = self.df["atoms"].apply(
        #        lambda x: get_2d_lattice(x)[0]
        #    )
        print(self.df)
        self.stem = STEMConv(output_size=[256, 256])

        train_ids, val_ids, test_ids = self.split_dataset(
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            val_frac=val_frac,
            test_frac=test_frac,
        )
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids

        # label encoding dictionary: Dict[str, int]
        self.class_labels = {
            key: id for id, key in enumerate(self.df.crys.unique())
        }
        self.n_classes = len(self.class_labels)
        print("Data n_classes", len(self.class_labels), self.n_classes)

    def split_dataset(
        self,
        n_train=None,
        n_val=None,
        n_test=None,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
    ):

        N = len(self.df)
        if n_train is None:
            n_val = int(N * val_frac)
            n_test = int(N * test_frac)
            n_train = N - (n_val + n_test)
        if not self.keep_data_order:
            # set a consistent train/val/test split
            torch.manual_seed(0)
            shuf = torch.randperm(N)
            torch.random.seed()
            train_ids = shuf[:n_train].tolist()
            val_ids = shuf[n_train : n_train + n_val].tolist()
            test_ids = shuf[
                n_train + n_val : n_train + n_val + n_test
            ].tolist()
        else:
            ids = list(np.arange(N))
            train_ids = ids[:n_train]
            val_ids = ids[-(n_val + n_test) : -n_test]
            test_ids = ids[-n_test:]
        print("train_ids", len(train_ids), n_train)
        print("val_ids", len(val_ids), n_val)
        print("test_ids", len(test_ids), n_test)

        return train_ids, val_ids, test_ids

    def __len__(self):
        """Datset size: len(jarvis_2d)"""
        return self.df.shape[0]

    def __getitem__(self, idx):
        """Sample: image, label mask, atomic coords, numbers, structure ids."""
        row = self.df.iloc[idx]
        # print (row.jid)
        a = Atoms.from_dict(row.atoms)

        # defaults:
        rot = 0
        shift_x = 0
        shift_y = 0
        px_scale = self.px_scale

        # apply pre-rendering structure augmentation
        if self.rotation_degrees is not None:
            rot = np.random.uniform(
                -self.rotation_degrees, self.rotation_degrees
            )

        if self.shift_angstrom is not None:
            shift_x, shift_y = np.random.uniform(
                -self.shift_angstrom, self.shift_angstrom, size=2
            )

        if self.zoom_pct is not None:
            frac = self.zoom_pct / 100
            px_scale *= 1 + np.random.uniform(-frac, frac)

        image, label, pos, nb = self.stem.simulate_surface(
            a, px_scale=px_scale, eps=0.6, rot=rot, shift=[shift_x, shift_y]
        )

        if self.label_mode == "radius":
            label = atomic_radius_mask(image.shape, pos, nb, px_scale)

        if self.to_tensor is not None:
            image = self.to_tensor(torch.tensor(image))

        sample = {
            "image": image,
            "label": torch.FloatTensor(label > 0),
            "id": row.jid,
            "px_scale": px_scale,
            "crys": self.class_labels[row.crys],
        }
        return sample

    def get_rotation_series(self, idx, angles=np.linspace(0, 90, 32)):
        """helper for evaluating models through a series of augmentations.
        ```python
        samples = dataset.get_rotation_series(0)
        angle_batch = dataloader.collate_fn(samples)
        graphs, targets = prepare_batch(angle_batch)
        ps = gnn(a_graphs)
        ```
        """
        row = self.df.iloc[idx]
        # print (row.jid)
        a = Atoms.from_dict(row.atoms)

        # defaults:
        # rot = 0
        shift_x = 0
        shift_y = 0
        px_scale = self.px_scale

        # apply pre-rendering structure augmentation
        if self.shift_angstrom is not None:
            shift_x, shift_y = np.random.uniform(
                -self.shift_angstrom, self.shift_angstrom, size=2
            )

        samples = []
        for angle in angles:
            image, label, pos, nb = self.stem.simulate_surface(
                a,
                px_scale=px_scale,
                eps=0.6,
                rot=angle,
                shift=[shift_x, shift_y],
            )

            if self.label_mode == "radius":
                label = atomic_radius_mask(image.shape, pos, nb, px_scale)

            if self.to_tensor is not None:
                image = self.to_tensor(torch.tensor(image))

            sample = {
                "image": image,
                "label": torch.FloatTensor(label > 0),
                "id": row.jid,
                "px_scale": px_scale,
                "crys": self.class_labels[row.crys],
            }
            samples.append(sample)

        return samples


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


def setup_accuracy(mode: Literal["binary", "categorical"]):
    softmax = partial(torch.softmax, dim=1)
    link = {"binary": torch.sigmoid, "categorical": softmax}[mode]

    def accuracy_transform(output):
        y_pred, y_true = output
        pred = link(y_pred) > 0.5
        return pred.type(torch.float32), y_true

    return accuracy_transform


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


def get_train_val_loaders(config: Config = Config(), image_data=[]):
    """UNet dataloader specification."""
    batch_size = config.training.batch_size

    j2d = Jarvis2dSTEMDataset(
        image_data=image_data,
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


def localization(
    config="config.json",
    prepare_batch=prepare_atom_localization_batch,
    image_data=[],
):

    # checkpoint_dir = config.parent
    with open(config, "r") as f:
        config = Config(**json.load(f))
    checkpoint_dir = config.training.parent_dir
    train_loader, val_loader = get_train_val_loaders(
        config, image_data=image_data
    )
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
        "trainer", filepath=os.path.join(checkpoint_dir, "train.log")
    )
    evaluator.logger = setup_logger(
        "trainer", filepath=os.path.join(checkpoint_dir, "train.log")
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

    tmp_data = []
    dft_2d = data("dft_2d")
    for i in dft_2d:
        info = {}
        info["jid"] = i["jid"]
        info["atoms"] = i["atoms"]
        info["crys"] = get_2d_lattice(i["atoms"])[1]
        tmp_data.append(info)
    image_data = tmp_data
    localization(image_data=image_data)
