"""Module for PyTorch autoencoder training."""
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import argparse
# from jarvis.db.jsonutils import dumpjson

random_seed = 123
torch.manual_seed(random_seed)
random.seed(0)
np.random.seed(random_seed)
torch.backends.cudnn.deterministic = True
# https://gist.github.com
# /AFAgarap/4f8a8d8edf352271fa06d85ba0361f26


class AE(nn.Module):
    """Module for auto-encoder."""

    def __init__(self, input_shape=50176, feats=1120):
        # def __init__(self, input_shape=50176,feats=448):
        """Initialize class."""
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=input_shape, out_features=feats
        )
        self.encoder_output_layer = nn.Linear(
            in_features=feats, out_features=feats
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=feats, out_features=feats
        )
        self.decoder_output_layer = nn.Linear(
            in_features=feats, out_features=input_shape
        )

    def forward(self, features):
        """Make forward prediction."""
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed


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

parser.add_argument(
    "--input_size",
    default=50176,
    help="Input size e.g 224x224."
    # "--input_size", default=784, help="Input size e.g 224x224."
)
parser.add_argument("--epochs", default=200, help="Number of epochs.")


parser.add_argument(
    "--output_dir",
    default="./",
    help="Folder to save outputs",
)


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    epochs = int(args.epochs)
    input_size = int(args.input_size)
    batch_size = int(args.batch_size)

    output_dir = args.output_dir
    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = AE(input_shape=input_size).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # mean-squared error loss
    criterion = nn.MSELoss()

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda x: x[0, :, :]),
            # transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_dataset = datasets.ImageFolder(
        args.train_folder, transform=transform
    )
    test_dataset = datasets.ImageFolder(args.test_folder, transform=transform)
    # print("train_dataset", train_dataset[0])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    for epoch in range(epochs):
        loss = 0
        for batch_features, _ in train_loader:
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            batch_features = batch_features.view(-1, input_size).to(device)
            # batch_features = batch_features.view(-1, 784).to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs = model(batch_features)

            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(train_loader)

        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

    test_examples = None
    with torch.no_grad():
        for batch_features in test_loader:
            batch_features = batch_features[0]
            test_examples = batch_features.view(-1, input_size).to(device)
            reconstruction = model(test_examples)
            break

    with torch.no_grad():
        number = 10
        plt.figure(figsize=(20, 4))
        for index in range(number):
            # display original
            ax = plt.subplot(2, number, index + 1)
            plt.imshow(test_examples[index].cpu().numpy().reshape(224, 224))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, number, index + 1 + number)
            plt.imshow(reconstruction[index].cpu().numpy().reshape(224, 224))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig("test.png")
    plt.close()
