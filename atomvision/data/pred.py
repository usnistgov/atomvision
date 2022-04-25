from atomvision.data.train import ResNetUNet
import torch
from jarvis.db.figshare import data
from torch.utils.data import DataLoader
from atomvision.data.stem import Jarvis2dSTEMDataset
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.switch_backend("agg")

num_class = 2
batch_size = 1
model = ResNetUNet(num_class)
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

model.load_state_dict(torch.load("checkpoint.pth", map_location=device))
model.eval()
model.to(device)
for bl in model.base_layers:
    for param in bl.parameters():
        param.requires_grad = False

my_data = data("dft_2d")[0:6]

test_set = Jarvis2dSTEMDataset(image_data=my_data)
test_loader = DataLoader(
    test_set, batch_size=batch_size, shuffle=True, num_workers=0
)

the_grid = GridSpec(len(my_data), 4)
plt.figure(figsize=(16, 14))
for ii, sample in enumerate(test_loader):
    inputs = sample["image"].to(
        device
    )  # .unsqueeze(1).repeat((1, 3, 1, 1), 1)
    inputs = inputs.unsqueeze(1).repeat(1, 3, 1, 1)
    labels = sample["label"].unsqueeze(1)
    labels = (
        (
            torch.cat((labels == 0, labels > 0), 1)
            .type(torch.float32)
            .to(device)
        )
        .cpu()
        .numpy()
    )
    pred = model(inputs)
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu().numpy()
    print()
    print(
        pred[0][0].shape,
        pred[0][1].shape,
        labels[0][0].shape,
        labels[0][1].shape,
    )
    print("pred.shape", pred.shape)
    print("labels.shape", labels.shape)

    plt.subplot(the_grid[ii, 0])
    plt.imshow(labels[0][0])
    plt.axis("off")
    # plt.tight_layout()

    plt.subplot(the_grid[ii, 1])
    plt.imshow(pred[0][0])
    plt.axis("off")
    # plt.tight_layout()

    plt.subplot(the_grid[ii, 2])
    plt.imshow(labels[0][1])
    plt.axis("off")
    # plt.tight_layout()

    plt.subplot(the_grid[ii, 3])
    plt.imshow(pred[0][1])
    plt.axis("off")
    # plt.tight_layout()
plt.tight_layout()
plt.savefig("pred.png")
plt.close()
