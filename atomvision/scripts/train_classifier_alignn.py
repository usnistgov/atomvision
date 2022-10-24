"""Module for ALIGNN Image Classifier."""
import numpy as np
from jarvis.db.figshare import data
from atomvision.data.stem import write_image_directory
import os
from alignn.models import alignn
from atomvision.data.graph import GraphDataset
from atomvision.scripts.image_to_graph import (
    crop_image,
    get_blob_positions,
    blob_list_to_graph,
)
from jarvis.db.jsonutils import dumpjson
from atomvision.models.training_utils import (
    setup_evaluation,
    log_training_loss,
    TrainingSettings,
    DatasetSettings,
    thresholded_output_transform,
    group_decay,
    #    get_train_val_loaders,
    #    setup_accuracy,
    #    activated_output_transform,
    #    setup_optimizer,
)

from atomvision.models.training_metrics import (
    log_confusion_matrix,
    performance_traces,
)

import torch
from torch import nn

# import typer
from pathlib import Path

# from typing import Optional
import pydantic
import json
from functools import partial

from jarvis.core.lattice import get_2d_lattice
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# import ignite
# from ignite.handlers import ModelCheckpoint
from ignite.handlers import Checkpoint, DiskSaver
from ignite.metrics import (
    Accuracy,
    Precision,
    Recall,
    Loss,
    ConfusionMatrix,
    #    RunningAverage,
)
from ignite.utils import setup_logger
from ignite.engine import (
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
)
import matplotlib.pyplot as plt
import random
import ignite

random_seed = 123
ignite.utils.manual_seed(random_seed)
torch.manual_seed(random_seed)
random.seed(0)
np.random.seed(random_seed)
torch.backends.cudnn.deterministic = True


plt.switch_backend("agg")


class Config(pydantic.BaseSettings):
    """Module for config."""

    dataset: DatasetSettings = DatasetSettings()
    training: TrainingSettings = TrainingSettings()
    print("training.output_feats", training.output_feats)
    gcn: alignn.ALIGNNConfig = alignn.ALIGNNConfig(
        name="alignn",
        alignn_layers=training.nlayers_alignn,
        atom_input_features=training.atom_input_feats,
        output_features=training.output_feats,
        classification=True,
        num_classes=5,
    )


def labelled_images_to_graphs(images, labels, border_pxl=0, saveto=""):
    """
    Get labelled image to graph.

    Args:
        images: list of image arrays.

        labels: list of corresponding class labels.
    """
    graphs = []
    line_graphs = []
    n = 0
    for img, lbl in zip(images, labels):
        if border_pxl != 0:
            img = crop_image(img, border_pxl)
        blob_list = get_blob_positions(img)
        g, lg = blob_list_to_graph(img, blob_list)
        graphs.append(g)
        line_graphs.append(lg)
        n = n + 1
    return graphs, line_graphs


def linegraph_dataloader(
    graphs=[],
    line_graphs=[],
    labels=[],
    batch_size=5,
    workers=0,
    pin_memory=True,
):
    """Return a line graph loader."""
    graph_data = GraphDataset(
        graphs=graphs, line_graphs=line_graphs, labels=labels
    )
    col = graph_data.collate_line_graph
    data_loader = DataLoader(
        graph_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=col,
        drop_last=True,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    return data_loader


def train_classifier(
    # config: Optional[Path] = typer.Argument(
    #    Path("config.json"),
    #    exists=True,
    #    file_okay=True,
    #    dir_okay=False,
    # ),
    config="config.json",
    num_classes=5,
    train_loader=None,
    val_loader=None,
    output_dir=None,
):
    """Run classifier."""
    checkpoint_dir = output_dir
    with open(config, "r") as f:
        config = Config(**json.load(f))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = alignn.ALIGNN(config.gcn)
    model.to(device)
    # print("model is cuda", next(model.parameters()).is_cuda)
    params = group_decay(model)
    prepare_batch = train_loader.dataset.prepare_batch
    prepare_batch = partial(prepare_batch, device=device)
    optimizer = torch.optim.AdamW(
        params,
        lr=config.training.learning_rate,
        # weight_decay=config.weight_decay,
    )
    steps_per_epoch = len(train_loader)
    # pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.training.learning_rate,
        epochs=config.training.epochs,
        steps_per_epoch=steps_per_epoch,
        # pct_start=pct_start,
        pct_start=0.3,
    )
    # optimizer, scheduler = setup_optimizer(
    #     model, train_loader, config.training
    # )
    criterion = nn.NLLLoss()

    metrics = {
        "accuracy": Accuracy(output_transform=thresholded_output_transform),
        "precision": Precision(output_transform=thresholded_output_transform),
        "recall": Recall(output_transform=thresholded_output_transform),
        "nll": Loss(criterion),
        # "rocauc": ROC_AUC(output_transform=activated_output_transform),
        # "roccurve": RocCurve(output_transform=activated_output_transform),
        "cm": ConfusionMatrix(
            output_transform=thresholded_output_transform,
            num_classes=num_classes,
        ),
    }
    trainer = create_supervised_trainer(
        model,
        optimizer,
        criterion,
        prepare_batch=prepare_batch,
        device=device,
        deterministic=True,
    )
    evaluator = create_supervised_evaluator(
        model,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
    )

    trainer.logger = setup_logger(
        "trainer", filepath=checkpoint_dir / "train_alignn.log"
    )
    evaluator.logger = setup_logger(
        "trainer", filepath=checkpoint_dir / "train_alignn.log"
    )

    # apply learning rate scheduler
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
    )

    # configure evaluation and setup
    checkpoint_handler = Checkpoint(
        {
            "model": model,
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "trainer": trainer,
        },
        DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
        filename_prefix="alignn",
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

    torch.save(history, checkpoint_dir / "alignn_metrics.pt")
    log_confusion_matrix(evaluator, val_loader)
    performance_traces(history)
    return history


def main(
    info=[],
    test_size=0.25,
    outdir="Images",
    config_file="config_alignn.json",
    id_tag="jid",
):
    """Generate image directory."""
    # ONLY NEED TO RUN ONCE TO GENERATE IMAGE DIRECTORY
    ts_lbls_file = os.path.join(outdir, "test_set_labels.txt")
    tr_lbls_file = os.path.join(outdir, "training_set_labels.txt")
    if not info:
        info = data("dft_2d")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

        mem = []
        labels = []
        for i in info:
            mem.append(i[id_tag])
            labels.append(get_2d_lattice(i["atoms"])[1])
        tr, ts = train_test_split(mem, stratify=labels, test_size=test_size)

        # Training Set Images
        tr_lbls, ts_lbls = write_image_directory(info, tr, ts, outdir=outdir)

    ts_lbls = np.loadtxt(ts_lbls_file, dtype=str)
    tr_lbls = np.loadtxt(tr_lbls_file, dtype=str)

    # Get image lists
    # Test Set
    ts_images = []
    ts_labels = [int(lbl) for lbl in ts_lbls[:, 1]]
    for ts in ts_lbls:
        ts_images.append(
            np.genfromtxt(os.path.join(outdir, ts[0] + ".txt"), delimiter=",")
        )

    ts_g, ts_lg = labelled_images_to_graphs(ts_images, ts_labels, border_pxl=0)

    val_loader = linegraph_dataloader(ts_g, ts_lg, ts_labels, batch_size=32)

    # Training Set
    tr_images = []
    tr_labels = [int(lbl) for lbl in tr_lbls[:, 1]]
    for tr in tr_lbls:
        tr_images.append(
            np.genfromtxt(os.path.join(outdir, tr[0] + ".txt"), delimiter=",")
        )

    tr_g, tr_lg = labelled_images_to_graphs(tr_images, tr_labels, border_pxl=0)

    train_loader = linegraph_dataloader(tr_g, tr_lg, tr_labels, batch_size=32)

    history = train_classifier(
        config_file,
        num_classes=5,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=Path(outdir),
    )
    filename = os.path.join(outdir, "history.json")
    dumpjson(data=history, filename=filename)


if __name__ == "__main__":
    info = data("dft_2d")
    main(info=info, config_file="config_alignn.json", test_size=0.25)
    # Get Images from the JARVIS 2D Dataset

    # default test_size is 0.25
    # mem = []
    # labels = []
    # for i in dft_2d:
    #     mem.append(i["jid"])
    #     labels.append(get_2d_lattice(i["atoms"])[1])
    # tr, ts = train_test_split(mem, stratify=labels)

    # #Training Set Images
    # from atomvision.data.stem import write_image_directory
    # tr_lbls, ts_lbls = write_image_directory(dft_2d, tr,\
    #  ts, outdir = "../data/2D_STEM_Images")

    """
    READ DATA FROM IMAGE DIRECTORY
    output_dir = "../data/2D_STEM_Images"
    ts_lbls = np.loadtxt(
        "../data/2D_STEM_Images/test_set_labels.txt", dtype=str
    )
    tr_lbls = np.loadtxt(
        "../data/2D_STEM_Images/training_set_labels.txt", dtype=str
    )

    # Get image lists
    # Test Set
    ts_images = []
    ts_labels = [int(lbl) for lbl in ts_lbls[:, 1]]
    for ts in ts_lbls:
        ts_images.append(
            np.genfromtxt(
                os.path.join(output_dir, ts[0] + ".txt"), delimiter=","
            )
        )

    ts_g, ts_lg = labelled_images_to_graphs(ts_images, ts_labels, border_pxl=0)

    val_loader = linegraph_dataloader(ts_g, ts_lg, ts_labels, batch_size=32)

    # Training Set
    tr_images = []
    tr_labels = [int(lbl) for lbl in tr_lbls[:, 1]]
    for tr in tr_lbls:
        tr_images.append(
            np.genfromtxt(
                os.path.join(output_dir, tr[0] + ".txt"), delimiter=","
            )
        )

    tr_g, tr_lg = labelled_images_to_graphs(tr_images, tr_labels, border_pxl=0)

    train_loader = linegraph_dataloader(tr_g, tr_lg, tr_labels, batch_size=32)

    history = train_classifier(
        config_file,
        num_classes=5,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=Path(output_dir),
    )
    """
