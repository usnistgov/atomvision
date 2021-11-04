"""Module to get classifer models."""

import torchvision.models as models
import torch.nn as nn


def vgg(num_labels=5, in_features=4096):
    """Get VGG model."""
    model = models.vgg16(pretrained=True)
    num_labels = 5
    model.classifier[6] = nn.Linear(in_features, num_labels)
    return model


def resnet(num_labels=5, in_features=2048):
    """Get Resnet model."""
    model = models.resnet101(pretrained=True)
    model.fc = nn.Linear(in_features, num_labels)
    return model


def googlenet(num_labels=5, in_features=1024):
    """Get Googlenet model."""
    model = models.googlenet(pretrained=True)
    model.fc = nn.Linear(in_features, num_labels, bias=True)
    return model


def densenet(num_labels=5, in_features1=1024, in_features2=512):
    """Get densenet model."""
    model = models.densenet161(pretrained=True)
    classifier_input = model.classifier.in_features
    classifier = nn.Sequential(
        nn.Linear(classifier_input, in_features1),
        nn.ReLU(),
        nn.Linear(in_features1, in_features2),
        nn.ReLU(),
        nn.Linear(in_features2, num_labels),
        nn.LogSoftmax(dim=1),
    )
    model.classifier = classifier
    return model


def mobilenet(num_labels=5, in_features=1024):
    """Get mobilenet model."""
    model = models.mobilenet_v3_small(pretrained=True)
    model.classifier[3] = nn.Linear(in_features, num_labels, bias=True)
    return model


def squeezenet(num_labels=5, in_features=512, kernel_size=(1, 1)):
    """Get squeezenet model."""
    model = models.squeezenet1_1(pretrained=True)
    model.classifier[1] = nn.Conv2d(in_features, num_labels, kernel_size=kernel_size)
