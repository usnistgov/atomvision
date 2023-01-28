"""Module to train variational autoencoder."""
# Still WIP
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import ignite
import random
import numpy as np
import time
import torch as T
import torchvision as tv
# import torch.nn as nn
# from torchvision import datasets, transforms
# from torchvision.utils import save_image
# import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
learning_rate = 1e-3
num_epochs = 10

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
        transforms.Lambda(lambda x: x[0, :, :]),
    ]
)

train_path = "/wrk/knc6/AtomVision/Combined/J2D_C2D_2DMatP/test_folder"
test_path = "/wrk/knc6/AtomVision/Combined/J2D_C2D_2DMatP/test_folder"
train_dataset = datasets.ImageFolder(train_path, transform=transform)
val_dataset = datasets.ImageFolder(test_path, transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True
)


class VAE(T.nn.Module):
    """Module for VAE."""

    def __init__(self):
        """Initialize the model."""
        super(VAE, self).__init__()
        self.fc1 = T.nn.Linear(224 * 224, 400)
        self.fc2a = T.nn.Linear(400, 20)
        self.fc2b = T.nn.Linear(400, 20)
        self.fc3 = T.nn.Linear(20, 400)
        self.fc4 = T.nn.Linear(400, 224 * 224)

        T.nn.init.xavier_uniform_(self.fc1.weight)
        T.nn.init.zeros_(self.fc1.bias)
        T.nn.init.xavier_uniform_(self.fc2a.weight)
        T.nn.init.zeros_(self.fc2a.bias)
        T.nn.init.xavier_uniform_(self.fc2b.weight)
        T.nn.init.zeros_(self.fc2b.bias)
        T.nn.init.xavier_uniform_(self.fc3.weight)
        T.nn.init.zeros_(self.fc3.bias)
        T.nn.init.xavier_uniform_(self.fc4.weight)
        T.nn.init.zeros_(self.fc4.bias)

    def encode(self, x):  # 784-400-[20,20]
        """Encode pixels."""
        z = T.relu(self.fc1(x))
        z1 = self.fc2a(z)  # u
        z2 = self.fc2b(z)  # logvar
        return (z1, z2)

    def decode(self, z):  # 20-400-784
        """Decode latent space."""
        z = T.relu(self.fc3(z))
        z = T.sigmoid(self.fc4(z))
        return z

    def forward(self, x):  # 784-400-[20,20]-20-400-784
        """Make a forward pass."""
        x = x.view(-1, 224 * 224)
        (u, logvar) = self.encode(x)
        stdev = T.exp(0.5 * logvar)
        noise = T.randn_like(stdev)
        z = u + (noise * stdev)  # 20
        z = self.decode(z)  # 20-400-784
        return (z, u, logvar)


# -----------------------------------------------


def cus_loss_func(recon_x, x, u, logvar):
    """Apply a custom loss function."""
    # https://arxiv.org/abs/1312.6114
    # KLD = 0.5 * sum(1 + log(sigma^2) - u^2 - sigma^2)
    bce = T.nn.functional.binary_cross_entropy(
        recon_x, x.view(-1, 224 * 224), reduction="sum"
    )
    kld = -0.5 * T.sum(1 + logvar - u.pow(2) - logvar.exp())
    return bce + kld


# def main():
#   # 0. preparation
#   print("\nBegin VAE / MNIST demo ")
#   T.manual_seed(1)
#   np.random.seed(1)

#   # 1. load data
#   bat_size = 128
#   trfm = tv.transforms.ToTensor()
#   train_ds = tv.datasets.MNIST('.\\Data', train=True,
#     download=True, transform=trfm)
#   train_ldr = T.utils.data.DataLoader(train_ds,
#     batch_size=bat_size, shuffle=True)

#   #2. create model
vae = VAE().to(device)

# 3. train model
max_epochs = 20
log_interval = 2
max_lines = 60000  # early-exit hack
lrn_rate = 0.001
optimizer = T.optim.Adam(vae.parameters(), lr=lrn_rate)

print("\nbat_size = %3d " % batch_size)
print("loss = custom BCE plus KLD ")
print("optimizer = Adam")
print("max_epochs = %3d " % max_epochs)
print("lrn_rate = %0.3f " % lrn_rate)

print("\nStarting training with saved checkpoints")
for epoch in range(0, max_epochs):
    vae = vae.train()
    T.manual_seed(1 + epoch)  # recovery reproducibility
    train_loss = 0.0  # accumulated custom loss
    num_lines_read = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        num_lines_read += batch_size
        if num_lines_read > max_lines:
            break  # early-exit hack
        data = data.to(device)  # (bat_sz, 1, 28, 28)

        optimizer.zero_grad()
        recon_x, u, logvar = vae(data)
        loss_val = cus_loss_func(recon_x, data, u, logvar)
        loss_val.backward()
        train_loss += loss_val.item()
        optimizer.step()

    if epoch % log_interval == 0 or epoch == max_epochs - 1:
        print(" epoch: %3d loss: %0.4f " % (epoch, train_loss))

        vae = vae.eval()
        num_images = 64
        rinpt = T.randn(num_images, 20).to(device)
        with T.no_grad():
            fakes = vae.decode(rinpt)
        fakes = fakes.view(num_images, 1, 224, 224)
        tv.utils.save_image(
            fakes,
            "fakes_" + str(epoch) + ".jpg",
            padding=4,
            pad_value=1.0,
        )  # no overwrite

        # save checkpoint
        dt = time.strftime("%Y_%m_%d-%H_%M_%S")
        fn = "Log_" + str(dt) + str("-") + str(epoch) + "_checkpoint.pt"

        info_dict = {
            "epoch": epoch,
            "vae_state": vae.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        T.save(info_dict, fn)

print("Training complete ")

# 6. save final model
print("\nSaving trained model state")
fn = "vae_mnist_model.pth"
T.save(vae.state_dict(), fn)

print("\nEnd demo \n")
