import torch
from jarvis.db.figshare import data
from torch.utils.data import DataLoader
from atomvision.data.stem import Jarvis2dSTEMDataset
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

plt.switch_backend("agg")
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)
preprocess_input = get_preprocessing_fn("resnet18", pretrained="imagenet")
# model = ResNetUNet(num_class)
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

batch_size = 1


def to_tensor(x):
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


model.load_state_dict(
    torch.load("checkpoint_20.pt", map_location=device)["model"]
)
model.eval()
model.to(device)

my_data = data("dft_2d")[-5:-1]

test_set = Jarvis2dSTEMDataset(
    image_data=my_data, label_mode="radius", to_tensor=to_tensor
)
test_loader = DataLoader(
    test_set, batch_size=batch_size, shuffle=True, num_workers=0
)

the_grid = GridSpec(len(my_data), 4)
plt.figure(figsize=(16, 14))
for ii, sample in enumerate(test_loader):
    inputs = sample["image"].to(
        device
    )  # .unsqueeze(1).repeat((1, 3, 1, 1), 1)
    # inputs = inputs.unsqueeze(1).repeat(1, 3, 1, 1)
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
    print("preds,labels", pred.shape, labels.shape)
    print(
        pred[0][0].shape,
        pred[0][0].shape,
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
    plt.imshow(pred[0][0])
    plt.axis("off")
    # plt.tight_layout()
plt.tight_layout()
plt.savefig("pred_rad.png")
plt.close()
