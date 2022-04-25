import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np

# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# import numpy as np
import seaborn as sns

# import torch
import random

# from torch import nn, optim
import torch.nn.functional as F

# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
import ignite
from ignite.engine import (
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix
from ignite.handlers import ModelCheckpoint, EarlyStopping


plt.switch_backend("agg")


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions
        # to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
        )
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(224),
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
train_path = (
    "/home/knc6/Software/atomvision/atomvision/data/STEM_JV/train_folder"
)
test_path = (
    "/home/knc6/Software/atomvision/atomvision/data/STEM_JV/test_folder"
)
train_dataset = datasets.ImageFolder(train_path, transform=transform)
val_dataset = datasets.ImageFolder(test_path, transform=transform)

# test_ratio=0.2
# n_train=int((1-test_ratio)*len(dataset))
# n_test=len(dataset)-n_train
# print (len(dataset),n_train,n_test)
# train_set, val_set = torch.utils.data.random_split(dataset, [n_train,n_test])
# Put into a Dataloader using torch library
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=True
)


model = models.densenet161(pretrained=True)
model = UNet(3, 5)
print(model)


# moving model to gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()


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
        "Training Results-Epoch:{}Avg accuracy:{:.2f}Avg loss:{:.2f}".format(
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
        "Valid. Results-Epoch:{} Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
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
    "./saved_models",
    "STM_densenet",
    n_saved=2,
    create_dir=True,
    require_empty=False,
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
