import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import random
import argparse
from torch import nn, optim
from torch.utils.data import DataLoader
import ignite
from ignite.engine import (
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix
from ignite.handlers import ModelCheckpoint, EarlyStopping
from atomvision.models.cnn_classifiers import (
    densenet,
    googlenet,
    vgg,
    mobilenet,
    resnet,
    squeezenet,
)
from atomvision.scripts.focal_loss import FocalLoss

# from atomvision.models.alignn_classifier import ALIGNN
from alignn.models.alignn import ALIGNNConfig

from skimage.future import graph
from skimage.future.graph import rag_mean_color
from skimage import graph, data, io, segmentation, color
import dgl
import torch
import cv2
import numpy as np
from skimage.measure import regionprops
from skimage import draw
from matplotlib import pyplot as plt
from typing import Tuple, Union
from alignn.models.alignn import (
    ALIGNNConfig,
    MLPLayer,
    ALIGNNConv,
    EdgeGatedGraphConv,
)
import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.nn import AvgPooling

# from dgl.nn.functional import edge_softmax
from pydantic.typing import Literal
from torch import nn
from torch.nn import functional as F

from alignn.models.utils import RBFExpansion
from alignn.utils import BaseSettings
import torchvision
import random

random_seed = 123
ignite.utils.manual_seed(random_seed)
torch.manual_seed(random_seed)
random.seed(0)
np.random.seed(random_seed)
torch.backends.cudnn.deterministic = True
plt.switch_backend("agg")


def show_img(img, filename=None):
    width = 10.0
    height = img.shape[0] * width / img.shape[1]
    f = plt.figure(figsize=(width, height))
    plt.imshow(img)
    if filename is not None:
        plt.savefig(filename)
        plt.close()


def compute_edge_props(edges):
    r1 = -edges.src["r"]
    r2 = edges.dst["r"]
    diffs = r2 - r1
    return {"h": diffs}


def display_edges(image, g, threshold):
    image = image.copy()
    for edge in g.edges():
        n1, n2 = edge
        r1, c1 = map(int, g.nodes.data()[n1]["label"])
        r2, c2 = map(int, g.nodes.data()[n2]["label"])
        line = draw.line(r1, c1, r2, c2)
        circle = draw.circle_perimeter(r1, c1, 2)
        if g[n1][n2]["weight"] < threshold:
            image[line] = 0, 1, 0
        image[circle] = 1, 1, 0

    return image


def image_to_dgl_graph(
    img=[],
    n_segments=300,
    compactness=30,
    resize=None,
    plot=True,
    filename=None,
):
    if resize is not None:
        img = cv2.resize(
            img, resize, interpolation=cv2.INTER_AREA
        )  # [1000,1000]

    labels = segmentation.slic(
        img, compactness=compactness, n_segments=n_segments
    )
    labels = (
        labels + 1
    )  # So that no labelled region is 0 and ignored by regionprops
    regions = regionprops(labels)
    label_rgb = color.label2rgb(labels, img, kind="avg")
    label_rgb = segmentation.mark_boundaries(label_rgb, labels, (0, 0, 0))
    rag = rag_mean_color(img, labels)
    node_features = []
    for region in regions:
        intensity = img[:, :, 0][
            int(region.centroid[0]), int(region.centroid[1])
        ]
        vals = [
            intensity,
            region["feret_diameter_max"],
            region["orientation"],
            region["eccentricity"],
            region["perimeter"],
            int(region.centroid[0]),
            int(region.centroid[1]),
            region["area"],
            region["extent"],
        ]
        node_features.append(vals)
        rag.add_node(region["label"], {"label": region["centroid"]})
    g = dgl.from_networkx(rag)
    if plot:
        edges_drawn_all = display_edges(label_rgb, rag, np.inf)
        show_img(edges_drawn_all, filename=filename)

    g.ndata["atom_features"] = torch.tensor(np.array(node_features)).type(
        torch.get_default_dtype()
    )
    edge_data = torch.tensor(
        np.array([i[2]["weight"] for i in ((list(rag.edges.data())) * 2)])
    )  # NEED TO CHECK SRC/DST connections
    g.edata["r"] = edge_data.type(torch.get_default_dtype())
    lg = g.line_graph(shared=True)
    lg.apply_edges(compute_edge_props)
    return g, lg  # ,rag


class ALIGNN(nn.Module):
    """Atomistic Line graph network.
    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self, config: ALIGNNConfig = ALIGNNConfig(name="alignn")):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        # print(config)
        self.classification = config.classification

        self.atom_embedding = MLPLayer(
            config.atom_input_features, config.hidden_features
        )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_input_features,
            ),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1,
                vmax=1.0,
                bins=config.triplet_input_features,
            ),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(
                    config.hidden_features,
                    config.hidden_features,
                )
                for idx in range(config.alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(
                    config.hidden_features, config.hidden_features
                )
                for idx in range(config.gcn_layers)
            ]
        )

        self.readout = AvgPooling()

        if self.classification:
            self.fc = nn.Linear(config.hidden_features, 5)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc = nn.Linear(config.hidden_features, config.output_features)
        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            self.fc.bias.data = torch.tensor(
                np.log(avg_gap), dtype=torch.float
            )
        elif config.link == "logit":
            self.link = torch.sigmoid

    def forward(
        # self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]
        self,
        image,
    ):
        """ALIGNN : start with `atom_features`.
        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        g, lg = image_to_dgl_graph(image, resize=[256, 256])

        if len(self.alignn_layers) > 0:
            # g, lg = g
            lg = lg.local_var()

            # angle features (fixed)
            z = self.angle_embedding(lg.edata.pop("h"))

        g = g.local_var()

        # initial node features: atom feature network...
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)

        # initial bond features
        bondlength = g.edata.pop("r")  # torch.norm(g.edata.pop("r"), dim=1)
        y = self.edge_embedding(bondlength)

        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)

        # norm-activation-pool-classify
        h = self.readout(g, x)
        out = self.fc(h)

        if self.link:
            out = self.link(out)

        if self.classification:
            # out = torch.round(torch.sigmoid(out))
            out = self.softmax(out)
        return torch.squeeze(out)


parser = argparse.ArgumentParser(description="AtomVison package.")
parser.add_argument(
    "--train_folder",
    default="train_folder",
    help="Folder with training images. Each class should have its own folder.",
)

parser.add_argument(
    "--test_folder",
    default="test_folder",
    help="Folder with test images. Each class should have its own folder.",
)

parser.add_argument("--batch_size", default=32, help="Batch size.")

parser.add_argument("--epochs", default=10, help="Number of epochs.")

parser.add_argument("--num_classes", default=5, help="Number of classes.")

parser.add_argument(
    "--model_name",
    default="densenet",
    help="Name of the pretrained torchvision model.",
)

parser.add_argument("--criterion", default="nll_loss", help="Loss function.")

parser.add_argument(
    "--output_dir",
    default="./",
    help="Folder to save outputs",
)


ignite.utils.manual_seed(123)
torch.manual_seed(123)
random.seed(123)
np.random.seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
# Load in each dataset and apply transformations using
# the torchvision.datasets as datasets library


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)

    train_dataset = datasets.ImageFolder(
        args.train_folder, transform=transform
    )
    val_dataset = datasets.ImageFolder(args.test_folder, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    model_name = args.model_name
    output_dir = args.output_dir
    criterion = args.criterion
    if criterion == "nll_loss":
        criterion = nn.NLLLoss()
    elif criterion == "focal_loss":
        criterion = FocalLoss(alpha=0.5, gamma=2.0, reduction="mean")
    else:
        raise ValueError("Loss function not available.")

    # moving model to gpu if available
    if model_name == "densenet":
        model = densenet(num_labels=int(args.num_classes))
    elif model_name == "resnet":
        model = resnet(num_labels=int(args.num_classes))
    elif model_name == "alignn_cf":
        model = ALIGNN(
            ALIGNNConfig(
                name="alignn", atom_input_features=9, classification=True
            )
        )
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # defining the number of epochs
    # creating trainer,evaluator
    trainer = create_supervised_trainer(
        model, optimizer, criterion, device=device
    )
    metrics = {
        "accuracy": Accuracy(),
        "nll": Loss(criterion),
        "cm": ConfusionMatrix(num_classes=5),
    }
    train_evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device
    )
    val_evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device
    )
    training_history = {"accuracy": [], "loss": []}
    validation_history = {"accuracy": [], "loss": []}
    last_epoch = []

    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

    def score_function(engine):
        val_loss = engine.state.metrics["nll"]
        return -val_loss

    handler = EarlyStopping(
        patience=10, score_function=score_function, trainer=trainer
    )
    val_evaluator.add_event_handler(Events.COMPLETED, handler)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        accuracy = metrics["accuracy"] * 100
        loss = metrics["nll"]
        last_epoch.append(0)
        training_history["accuracy"].append(accuracy)
        training_history["loss"].append(loss)
        print(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                trainer.state.epoch, accuracy, loss
            )
        )

    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        accuracy = metrics["accuracy"] * 100
        loss = metrics["nll"]
        validation_history["accuracy"].append(accuracy)
        validation_history["loss"].append(loss)
        print(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                trainer.state.epoch, accuracy, loss
            )
        )

    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)

    @trainer.on(Events.COMPLETED)
    def log_confusion_matrix(trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        cm = metrics["cm"]
        cm = cm.numpy()
        cm = cm.astype(int)
        classes = ["0", "1", "2", "3", "4"]
        fig, ax = plt.subplots(figsize=(10, 10))
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax=ax, fmt="d")
        # labels, title and ticks
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title("Confusion Matrix")
        ax.xaxis.set_ticklabels(classes, rotation=90)
        ax.yaxis.set_ticklabels(classes, rotation=0)
        plt.savefig("CM.png")
        plt.close()

    # Cheange n_save if we need to store more models
    checkpointer = ModelCheckpoint(
        output_dir, "output", n_saved=2, create_dir=True, require_empty=False
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, checkpointer, {"output": model}
    )
    trainer.run(train_loader, max_epochs=epochs)
    plt.plot(training_history["accuracy"], label="Training Accuracy")
    plt.plot(validation_history["accuracy"], label="Validation Accuracy")
    plt.xlabel("No. of Epochs")
    plt.ylabel("Accuracy")
    plt.legend(frameon=False)
    plt.savefig("Acc.png")
    plt.close()

    plt.plot(training_history["loss"], label="Training Loss")
    plt.plot(validation_history["loss"], label="Validation Loss")
    plt.xlabel("No. of Epochs")
    plt.ylabel("Loss")
    plt.legend(frameon=False)
    plt.savefig("Loss.png")
    plt.close()
