import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import random
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import ignite
from ignite.engine import (
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix
from ignite.handlers import ModelCheckpoint, EarlyStopping

plt.switch_backend("agg")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
"""
transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
"""

ignite.utils.manual_seed(123)
torch.manual_seed(123)
random.seed(123)
np.random.seed(123)
# Load in each dataset and apply transformations using
# the torchvision.datasets as datasets library
train_path = (
    "/home/knc6/Software/atomvision/atomvision/data/STM_JV/train_folder"
)
test_path = "/home/knc6/Software/atomvision/atomvision/data/STM_JV/test_folder"
train_dataset = datasets.ImageFolder(train_path, transform=transform)
val_dataset = datasets.ImageFolder(test_path, transform=transform)
# val_set = train_set #datasets.ImageFolder("root/label/valid", transform = transformations)

# test_ratio=0.2
# n_train=int((1-test_ratio)*len(dataset))
# n_test=len(dataset)-n_train
# print (len(dataset),n_train,n_test)
# train_set, val_set = torch.utils.data.random_split(dataset, [n_train,n_test])
# Put into a Dataloader using torch library
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=32, shuffle=True
)


model = models.densenet161(pretrained=True)
print(model)
classifier_input = model.classifier.in_features
num_labels = 5  # PUT IN THE NUMBER OF LABELS IN YOUR DATA
classifier = nn.Sequential(
    nn.Linear(classifier_input, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, num_labels),
    nn.LogSoftmax(dim=1),
)
# Replace default classifier with new classifier
model.classifier = classifier

# moving model to gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()

from focal_loss import FocalLoss

# from atomvision.models.focal_loss import FocalLoss
criterion = FocalLoss(alpha=0.5, gamma=2.0, reduction="mean")

# defining the number of epochs
epochs = 200
# creating trainer,evaluator
trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
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


checkpointer = ModelCheckpoint(
    "./saved_models", "STEM5", n_saved=2, create_dir=True, require_empty=False
)
trainer.add_event_handler(
    Events.EPOCH_COMPLETED, checkpointer, {"STEM5": model}
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
