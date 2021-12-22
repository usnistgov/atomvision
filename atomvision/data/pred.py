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


my_data = data("dft_2d")[0:6]

test_set = Jarvis2dSTEMDataset(image_data=my_data)
#test_loader = DataLoader(
#    test_set, batch_size=batch_size, shuffle=True, num_workers=0
#)


# Get the first batch
inputs = test_set[0]["image"]
labels = test_set[0]["label"]
# inputs, labels = next(iter(test_loader))
inputs = inputs.to(device)
labels = labels  # .to(device)
print("inputs.shape", inputs.shape)
print("labels.shape", labels.shape)

# Predict
pred = model(inputs)
# The loss functions include the sigmoid function.
pred = torch.sigmoid(pred)
pred = pred.data.cpu().numpy()
print("pred.shape", pred.shape)
