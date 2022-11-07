# import matplotlib
from jarvis.core.lattice import get_2d_lattice
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import blob_log
import networkx as nx
from scipy.spatial import KDTree
from typing import Tuple, Union, List
from alignn.models.alignn import (
    ALIGNNConfig,
    MLPLayer,
    ALIGNNConv,
    EdgeGatedGraphConv,
)
from functools import partial
import dgl
from dgl.nn import AvgPooling
from alignn.models.utils import RBFExpansion
import torchvision
from jarvis.db.figshare import data
from atomvision.data.stemconv import STEMConv
from jarvis.core.atoms import Atoms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from ignite.contrib.handlers.stores import EpochOutputStore
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.metrics import (
    Accuracy,
    Precision,
    Recall,
    ConfusionMatrix,
)
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from torch import nn
from jarvis.db.jsonutils import dumpjson
import os

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


def image_reshape(img):
    imgr = torchvision.transforms.functional.to_tensor(img)[np.newaxis, ...]
    return imgr


def compute_edge_props(edges):
    r1 = -edges.src["r"]
    r2 = edges.dst["r"]
    # diffs =r2-r1
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    return {"h": bond_cosine}


def get_segmented_image(
    image=None,
    n_labels=2,
    threshold=0.5,
    image_path="JARVIS-2D-STM-JPG/JARVIS-2D-STM-JPG/JVASP-723_pos.jpg",
    model_path="checkpoint_100.pt",
    device=device,
):
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        encoder_depth=3,
        decoder_channels=(64, 32, 16),
        in_channels=3,
        classes=1,
    )
    state = torch.load(model_path, map_location=torch.device(device))
    # state = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state["model"])
    model.eval()
    if image is None:
        image = cv2.imread(image_path)  # [:,:,0]
        # print('xxx',len(image.shape))
    if len(image.shape) >= 3:
        image = image[:, :, 0]
    npx = cv2.resize(image, [256, 256], interpolation=cv2.INTER_AREA)
    scale = np.max(npx)
    npx = torch.tensor(np.tile(npx / scale, (3, 1, 1))[np.newaxis, ...])
    # print (npx.shape)
    # xx=(np.tile(npx, (3,1, 1)))
    # xx=torch.tensor(xx[np.newaxis,...])
    with torch.no_grad():
        yhat = model(npx.float()).detach()

    imx = torch.sigmoid(yhat.squeeze())
    lbl = np.digitize(imx.numpy(), bins=np.array(range(n_labels)) / n_labels)
    # lbl=imx>=threshold
    return imx, lbl


# To get rid of boundary effects during graph generation
def crop_image(int_image, lbl_image, px=20):
    crp_int = int_image[px:-px, px:-px]
    crp_lbl = lbl_image[px:-px, px:-px]
    return crp_int, crp_lbl


def get_blob_positions(lbl_image, method=blob_log, plot=False, min_sigma=8):
    blobs_log = blob_log(
        np.array(lbl_image, dtype="int"), threshold=0, min_sigma=min_sigma
    )
    # print('blobs_log',blobs_log)
    if plot:
        for i, j in blobs_log[:, 0:2]:
            plt.plot(j, i, "o", c="red")
        plt.imshow(np.array(lbl_image, dtype="int"))
        plt.colorbar()
    return blobs_log


def bond_vector(edges):
    """Compute bond vectors from node pairs."""
    u = edges.src["pos"]
    v = edges.dst["pos"]
    return {"r": v - u}


def blob_list_to_graph(blobs_log, px_angstrom=0.1, cutoff_angstrom=6):
    g = nx.Graph()
    sigmas = []
    for indx, row in enumerate(blobs_log):

        pos = (row[1], row[0], 0)
        sigma = row  # [2]
        sigmas.append(sigma)
        g.add_node(indx, pos=pos)

    points = np.array(blobs_log[:, 0:2]) * px_angstrom
    nbrs = KDTree(points)
    g.add_edges_from(nbrs.query_pairs(cutoff_angstrom))
    # print(g)
    return g, sigmas


def convert_to_dgl(g, node_attrs=["pos"], sigmas=[]):
    """
    Converts from networkx to dgl graph.
    Adds bond vectors as edge attributes.

    Need to add other node attributes.
    """
    g = dgl.from_networkx(g, node_attrs)
    g.apply_edges(bond_vector)
    g.edata["r"] = g.edata["r"].type(torch.float32)
    g.ndata["atom_features"] = torch.tensor(np.array(sigmas)).type(
        torch.float32
    )
    return g


def image_to_dgl_graph_blob(
    image_data=None,
    image_path="JARVIS-2D-STM-JPG/JARVIS-2D-STM-JPG/JVASP-723_pos.jpg",
    crop_thresh=30,
    min_sigma=8,
    model_path="checkpoint_100.pt",
    device="cpu",
):
    # image_path='JARVIS-2D-STM-JPG/JARVIS-2D-STM-JPG/JVASP-723_pos.jpg'
    # image_path='JARVIS-2D-STM-JPG/JARVIS-2D-STM-JPG/JVASP-667_pos.jpg'

    if image_data is not None:
        img = image_data
    else:
        img = cv2.imread(image_path)

    imx, lbl = get_segmented_image(
        image=image_data, n_labels=2, model_path=model_path, device=device
    )
    # Plot thresholded image
    lbl_img = np.invert(lbl)
    crp_img, crp_lbl = crop_image(img, lbl_img, crop_thresh)

    blobs_log = get_blob_positions(crp_lbl, min_sigma=min_sigma)
    g, sigmas = blob_list_to_graph(blobs_log)
    g_dgl = convert_to_dgl(g, sigmas=sigmas)
    lg = g_dgl.line_graph(shared=True)
    lg.apply_edges(compute_edge_props)
    # g.to(device)
    # lg.to(device)
    return g_dgl, lg


num_classes = 5


class ALIGNN(nn.Module):
    """Atomistic Line graph network.
    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self, config: ALIGNNConfig = ALIGNNConfig(name="alignn")):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        # print(config)
        self.classification = config.classification

        self.atom_embedding = MLPLayer(
            config.atom_input_features, config.hidden_features
        )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=6.0,
                bins=config.edge_input_features,
            ),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1,
                vmax=1.0,
                bins=config.triplet_input_features,
            ),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(
                    config.hidden_features,
                    config.hidden_features,
                )
                for idx in range(config.alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(
                    config.hidden_features, config.hidden_features
                )
                for idx in range(config.gcn_layers)
            ]
        )

        self.readout = AvgPooling()
        num_classes = 5
        if self.classification:
            self.fc = nn.Linear(config.hidden_features, num_classes)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc = nn.Linear(config.hidden_features, config.output_features)
        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            self.fc.bias.data = torch.tensor(
                np.log(avg_gap), dtype=torch.float
            )
        elif config.link == "logit":
            self.link = torch.sigmoid

    def forward(
        self,
        g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]
        # self, image
    ):
        """ALIGNN : start with `atom_features`.
        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        # g, lg = image_to_dgl_graph(image,resize=[256, 256])

        if len(self.alignn_layers) > 0:
            g, lg = g
            lg = lg.local_var()

            # angle features (fixed)
            z = self.angle_embedding(lg.edata.pop("h"))

        g = g.local_var()

        # initial node features: atom feature network...
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)

        # initial bond features
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        y = self.edge_embedding(bondlength)

        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)

        # norm-activation-pool-classify
        h = self.readout(g, x)
        out = self.fc(h)

        if self.link:
            out = self.link(out)

        if self.classification:
            # out = torch.round(torch.sigmoid(out))
            out = self.softmax(out)
        return torch.squeeze(out)


def prepare_line_graph_batch(
    batch: Tuple[Tuple[dgl.DGLGraph, dgl.DGLGraph], torch.Tensor],
    device=None,
    non_blocking=False,
):
    """Send line graph batch to device.
    Note: the batch is a nested tuple, with the graph and line graph together
    """
    g, lg, t = batch
    batch = (
        (
            g.to(device, non_blocking=non_blocking),
            lg.to(device, non_blocking=non_blocking),
        ),
        t.to(device, non_blocking=non_blocking),
    )

    return batch


class ImageDataset(torch.utils.data.Dataset):
    """Dataset of crystal DGLGraphs."""

    def __init__(
        self,
        ids=[],
        graphs=[],
        line_graphs=[],
        labels=[],  # 1,2,3,4,5 etc
        transform=None,
        line_graph=True,
        classification=True,
        id_tag="jid",
    ):
        """Pytorch Dataset for atomistic graphs.
        `df`: pandas dataframe from e.g. jarvis.db.figshare.data
        `graphs`: DGLGraph representations corresponding to rows in `df`
        `target`: key for label column in `df`
        """
        self.ids = ids
        self.images = images
        if not self.ids:
            self.ids = [str(j) for j in np.arange(len(self.images))]
        self.graphs = graphs
        self.line_graphs = line_graphs
        # self.targets = target
        self.line_graph = line_graph
        self.labels = labels
        # (OneHotEncoder(sparse=False).fit_transform(np.array(labels).reshape(-1,1)))
        # self.labels = labels
        self.ids = ids
        self.labels = torch.tensor(self.labels).type(torch.get_default_dtype())
        self.transform = transform

        #         for i in images:
        #             g,lg = image_to_dgl_graph(i,resize=[256, 256])
        #             self.graphs.append(g)
        #             self.line_graphs.append(lg)

        # print ('self.graphs',len(self.graphs))
        self.prepare_batch = prepare_line_graph_batch

        if classification:
            self.labels = self.labels.view(-1).long()
            print("Classification dataset.", self.labels)

    def __len__(self):
        """Get length."""
        return self.labels.shape[0]

    def __getitem__(self, idx):
        """Get StructureDataset sample."""
        # print ('GRAPHS',len(self.graphs),idx)
        g = self.graphs[idx]
        label = self.labels[idx]

        if self.transform:
            g = self.transform(g)

        if self.line_graph:
            return g, self.line_graphs[idx], label

        return g, label

    @staticmethod
    def collate(samples: List[Tuple[dgl.DGLGraph, torch.Tensor]]):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(labels)

    @staticmethod
    def collate_line_graph(
        samples: List[Tuple[dgl.DGLGraph, dgl.DGLGraph, torch.Tensor]]
    ):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, line_graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_line_graph = dgl.batch(line_graphs)
        if len(labels[0].size()) > 0:
            return batched_graph, batched_line_graph, torch.stack(labels)
        else:
            return batched_graph, batched_line_graph, torch.tensor(labels)


def get_dataloader(
    graphs=[],
    line_graphs=[],
    labels=[],
    batch_size=5,
    workers=0,
    pin_memory=True,
):

    # d = pd.DataFrame(d[:100])

    img_data = ImageDataset(
        graphs=graphs, line_graphs=line_graphs, labels=labels
    )

    col = img_data.collate_line_graph
    data_loader = DataLoader(
        img_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=col,
        drop_last=True,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    return data_loader


images = []
labels = []
graphs = []
line_graphs = []
batch_size = 32
test_size = 0.2
dft_2d = data("dft_2d")  # [0:100]
mem = []
labels = []
for i in dft_2d:
    mem.append(i["jid"])
    labels.append(get_2d_lattice(i["atoms"])[1])
tr, ts = train_test_split(mem, stratify=labels)


labels = []
graphs = []
line_graphs = []
for i in tqdm(dft_2d):
    if i["jid"] in tr:
        structure = Atoms.from_dict(i["atoms"])
        img = STEMConv(output_size=[256, 256]).simulate_surface(structure)[0]
        label = get_2d_lattice(structure.to_dict())[1]
        labels.append(label)
        g, lg = image_to_dgl_graph_blob(image_data=img, device=device)
        graphs.append(g)
        line_graphs.append(lg)
train_loader = get_dataloader(
    graphs=graphs,
    line_graphs=line_graphs,
    labels=labels,
    batch_size=batch_size,
)

labels = []
graphs = []
line_graphs = []
for i in tqdm(dft_2d):
    if i["jid"] in ts:
        structure = Atoms.from_dict(i["atoms"])
        img = STEMConv(output_size=[256, 256]).simulate_surface(structure)[0]
        label = get_2d_lattice(structure.to_dict())[1]
        labels.append(label)
        g, lg = image_to_dgl_graph_blob(image_data=img, device=device)
        graphs.append(g)
        line_graphs.append(lg)
val_loader = get_dataloader(
    graphs=graphs,
    line_graphs=line_graphs,
    labels=labels,
    batch_size=batch_size,
)
test_loader = val_loader


def thresholded_output_transform(output):
    """Round off output."""
    y_pred, y = output
    y_pred = torch.round(torch.exp(y_pred))
    # print ('output',y_pred)
    return y_pred, y


def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def activated_output_transform(output):
    """Exponentiate output."""
    y_pred, y = output
    y_pred = torch.exp(y_pred)
    y_pred = y_pred[:, 1]
    return y_pred, y


num_classes = 5
net = ALIGNN(
    ALIGNNConfig(name="alignn", atom_input_features=3, classification=True)
)
net.to(device)
store_outputs = True
learning_rate = 0.001
epochs = 200
prepare_batch = train_loader.dataset.prepare_batch
prepare_batch = partial(prepare_batch, device=device)
deterministic = True
# device = "cpu"
params = group_decay(net)
checkpoint_dir = "."
output_dir = "."

optimizer = torch.optim.AdamW(
    params,
    lr=learning_rate,
    # weight_decay=config.weight_decay,
)
steps_per_epoch = len(train_loader)
# pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=learning_rate,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    # pct_start=pct_start,
    pct_start=0.3,
)
# scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer,
# )
criterion = nn.NLLLoss()
# criterion = nn.CrossEntropyLoss()
metrics = {
    "accuracy": Accuracy(output_transform=thresholded_output_transform),
    "precision": Precision(output_transform=thresholded_output_transform),
    "recall": Recall(output_transform=thresholded_output_transform),
    # "rocauc": ROC_AUC(output_transform=activated_output_transform),
    # "roccurve": RocCurve(output_transform=activated_output_transform),
    "confmat": ConfusionMatrix(
        output_transform=thresholded_output_transform, num_classes=num_classes
    ),
}


trainer = create_supervised_trainer(
    net,
    optimizer,
    criterion,
    prepare_batch=prepare_batch,
    device=device,
    deterministic=deterministic,
    # output_transform=make_standard_scalar_and_pca,
)

evaluator = create_supervised_evaluator(
    net,
    metrics=metrics,
    prepare_batch=prepare_batch,
    device=device,
    # output_transform=make_standard_scalar_and_pca,
)

train_evaluator = create_supervised_evaluator(
    net,
    metrics=metrics,
    prepare_batch=prepare_batch,
    device=device,
    # output_transform=make_standard_scalar_and_pca,
)

# ignite event handlers:
trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())

# apply learning rate scheduler
trainer.add_event_handler(
    Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
)
to_save = {
    "model": net,
    "optimizer": optimizer,
    "lr_scheduler": scheduler,
    "trainer": trainer,
}
handler = Checkpoint(
    to_save,
    DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
    n_saved=2,
    global_step_transform=lambda *_: trainer.state.epoch,
)
trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)

pbar = ProgressBar()
pbar.attach(trainer, output_transform=lambda x: {"loss": x})
# pbar.attach(evaluator,output_transform=lambda x: {"mae": x})

history = {
    "train": {m: [] for m in metrics.keys()},
    "validation": {m: [] for m in metrics.keys()},
}
eos = EpochOutputStore()
eos.attach(evaluator)
train_eos = EpochOutputStore()
train_eos.attach(train_evaluator)
# collect evaluation performance


@trainer.on(Events.EPOCH_COMPLETED)
def log_results(engine):
    """Print training and validation metrics to console."""
    train_evaluator.run(train_loader)
    evaluator.run(val_loader)

    tmetrics = train_evaluator.state.metrics
    vmetrics = evaluator.state.metrics
    for metric in metrics.keys():
        tm = tmetrics[metric]
        vm = vmetrics[metric]
        if metric == "roccurve":
            tm = [k.tolist() for k in tm]
            vm = [k.tolist() for k in vm]
        if isinstance(tm, torch.Tensor):
            tm = tm.cpu().numpy().tolist()
            vm = vm.cpu().numpy().tolist()

        history["train"][metric].append(tm)
        history["validation"][metric].append(vm)

    # for metric in metrics.keys():
    #    history["train"][metric].append(tmetrics[metric])
    #    history["validation"][metric].append(vmetrics[metric])

    if store_outputs:
        history["EOS"] = eos.data
        history["trainEOS"] = train_eos.data
        dumpjson(
            filename=os.path.join(output_dir, "history_val.json"),
            data=history["validation"],
        )
        dumpjson(
            filename=os.path.join(output_dir, "history_train.json"),
            data=history["train"],
        )


# train the model!
trainer.run(train_loader, max_epochs=epochs)
print(history["validation"])
