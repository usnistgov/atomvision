#%%
import matplotlib
import skimage

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import segmentation_models_pytorch as smp



from atomvision import data
from atomvision.plotting import plot_edges
from atomvision.data.stemconv import STEMConv
from atomvision.models.segmentation_utils import to_tensor_resnet18, prepare_atom_localization_batch
from atomvision.data.stem import (
    Jarvis2dSTEMDataset,
    atom_mask_to_graph,
)

%matplotlib inline

# %%
dataset = Jarvis2dSTEMDataset(label_mode="radius", rotation_degrees=90, shift_angstrom=0.5, zoom_pct=10)
j2d = Jarvis2dSTEMDataset(label_mode="radius", rotation_degrees=90, shift_angstrom=0.5, zoom_pct=10, to_tensor=to_tensor_resnet18)
val_loader = DataLoader(
    j2d, batch_size=32, sampler=SubsetRandomSampler(j2d.val_ids)
)


# %%
fig, axes = plt.subplots(figsize=(16, 16), ncols=2, nrows=2)
for ax, idx in zip(axes.flat, range(4)):

    sample = dataset[0]

    im = sample["image"]
    label = sample["label"].numpy()
    I = skimage.color.label2rgb(label, image=im / im.max(), alpha=0.3, bg_label=0)
    ax.imshow(I, origin="lower")
    g, props = atom_mask_to_graph(label, im, sample["px_scale"])
    plot_edges(g, ax=ax)
    ax.set_title(sample["id"])


plt.tight_layout()


# %%
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    encoder_depth=3,
    decoder_channels=(64, 32, 16),
    in_channels=3,
    classes=1,
)
state = torch.load("../models/baseline/checkpoint_100.pt", map_location=torch.device('cpu'))
model.load_state_dict(state["model"])


# %%
model.eval()
batch = next(iter(val_loader))
x, y = prepare_atom_localization_batch(batch)
with torch.no_grad():
    yhat = model(x).detach()

# %%
fig, m_axes = plt.subplots(nrows=4, ncols=4, figsize=(16,16))

for idx, axes in enumerate(m_axes):
    img = batch["image"][idx,0,:,:].numpy()
    lbl = torch.sigmoid(yhat.squeeze()[idx,:,:]).numpy() > 0.5
    g, props = atom_mask_to_graph(lbl, img, batch["px_scale"][idx].item())

    g_true, props_true = atom_mask_to_graph(batch["label"][idx,:,:].numpy(), img, batch["px_scale"][idx].item())

    axes[0].imshow(img)
    axes[1].imshow(torch.sigmoid(yhat.squeeze()[idx,:,:]).numpy())
    axes[2].imshow(torch.sigmoid(yhat.squeeze()[idx,:,:]).numpy() > 0.5)
    axes[3].imshow(batch["label"][idx,:,:].numpy())
    plot_edges(g, ax=axes[2])
    plot_edges(g_true, ax=axes[3])

plt.tight_layout()


# %%
from atomvision.data.stem import build_prepare_graph_batch
prepare_g = build_prepare_graph_batch(model, prepare_atom_localization_batch)
prepare_g(batch)

# %%
%time batch = next(iter(val_loader))

# %%
from alignn.models import alignn
gcn_state = torch.load("../models/baseline/gcn_checkpoint_100.pt", map_location=torch.device('cpu'))
cfg = alignn.ALIGNNConfig(name="alignn", alignn_layers=0, atom_input_features=2, output_features=6)
gnn = alignn.ALIGNN(cfg)
# %%
graphs, targets = prepare_g(batch)

# %%
gnn(graphs)


# %%
samples = j2d.get_rotation_series(0)
angle_batch = val_loader.collate_fn(samples)
a_graphs, a_targets = prepare_g(angle_batch)
ps = gnn(a_graphs)

# %%
plt.plot(np.linspace(0,90,32), ps.detach().numpy())
plt.xlabel("image rotation (degrees")
plt.ylabel("class logits")
# %%
a_targets

# %%
fig, axes = plt.subplots(nrows=2, ncols=4)
for idx, ax in enumerate(axes.flat):
    im = samples[idx]["image"]
    ax.imshow(im[0].numpy())
    ax.axis("off")
    ax.axis("equal")

plt.tight_layout()

# %%
