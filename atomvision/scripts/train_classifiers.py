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
from atomvision.models.classifiers import (
    densenet,
    googlenet,
    vgg,
    mobilenet,
    resnet,
    squeezenet,
)
from atomvision.scripts.focal_loss import FocalLoss

plt.switch_backend("agg")


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

    train_dataset = datasets.ImageFolder(
        args.train_folder, transform=transform
    )
    val_dataset = datasets.ImageFolder(args.test_folder, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True
    )

    epochs = int(args.epochs)
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
