from atomvision.data.train import ResNetUNet
import torch
from jarvis.db.figshare import data
from torch.utils.data import DataLoader
from atomvision.data.stem import Jarvis2dSTEMDataset

num_class = 2
batch_size = 3
model = ResNetUNet(num_class)
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

model.load_state_dict(torch.load("checkpoint.pth"))
model.eval()
model.to(device)
for l in model.base_layers:
    for param in l.parameters():
        param.requires_grad = False

my_data = data("dft_2d")[0:6]

test_set = Jarvis2dSTEMDataset(image_data=my_data)
test_loader = DataLoader(
    test_set, batch_size=batch_size, shuffle=True, num_workers=0
)


for sample in test_loader:
    inputs = sample["image"].to(
        device
    )  # .unsqueeze(1).repeat((1, 3, 1, 1), 1)
    inputs = inputs.unsqueeze(1).repeat(1, 3, 1, 1)
    labels = sample["label"].unsqueeze(1)
    labels = (
        torch.cat((labels == 0, labels > 0), 1).type(torch.float32).to(device)
    )
    pred = model(inputs)
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu().numpy()
    print("pred.shape", pred.shape)
    print("labels.shape", labels.shape)
