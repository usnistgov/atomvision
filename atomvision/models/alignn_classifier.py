from skimage.future import graph
from skimage.future.graph import rag_mean_color
from skimage import graph, data, io, segmentation, color
import dgl
import torch
import cv2
import numpy as np
from skimage.measure import regionprops
from skimage import draw
from matplotlib import pyplot as plt
from typing import Tuple, Union
from alignn.models.alignn import ALIGNNConfig,MLPLayer,ALIGNNConv,EdgeGatedGraphConv
import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.nn import AvgPooling

# from dgl.nn.functional import edge_softmax
from pydantic.typing import Literal
from torch import nn
from torch.nn import functional as F

from alignn.models.utils import RBFExpansion
from alignn.utils import BaseSettings
import torchvision

#%matplotlib inline


def show_img(img,filename=None):
    width = 10.0
    height = img.shape[0] * width / img.shape[1]
    f = plt.figure(figsize=(width, height))
    plt.imshow(img)
    if filename is not None:
        plt.savefig(filename)
        plt.close()

def compute_edge_props(edges):
    r1 = -edges.src["r"]
    r2 = edges.dst["r"]
    diffs =r2-r1
    return {"h": diffs}

def display_edges(image, g, threshold):
    image = image.copy()
    for edge in g.edges():
        n1, n2 = edge
        r1, c1 = map(int, g.nodes.data()[n1]["label"])
        r2, c2 = map(int, g.nodes.data()[n2]["label"])
        line = draw.line(r1, c1, r2, c2)
        circle = draw.circle_perimeter(r1, c1, 2)
        if g[n1][n2]["weight"] < threshold:
            image[line] = 0, 1, 0
        image[circle] = 1, 1, 0

    return image

def image_to_dgl_graph(
    img=[], n_segments=300, compactness=30, resize=None, plot=True,filename=None
):
    if resize is not None:
        img = cv2.resize(
            img, resize, interpolation=cv2.INTER_AREA
        )  # [1000,1000]

    labels = segmentation.slic(
        img, compactness=compactness, n_segments=n_segments
    )
    labels = (
        labels + 1
    )  # So that no labelled region is 0 and ignored by regionprops
    regions = regionprops(labels)
    label_rgb = color.label2rgb(labels, img, kind="avg")
    label_rgb = segmentation.mark_boundaries(label_rgb, labels, (0, 0, 0))
    rag = rag_mean_color(img, labels)
    node_features = []
    for region in regions:
        intensity = img[:, :, 0][
            int(region.centroid[0]), int(region.centroid[1])
        ]
        vals = [
            intensity,
            region["feret_diameter_max"],
            region["orientation"],
            region["eccentricity"],
            region["perimeter"],
            int(region.centroid[0]),
            int(region.centroid[1]),
            region["area"],
            region["extent"],
        ]
        node_features.append(vals)
        rag.add_node(region["label"], {"label": region["centroid"]})
    g = dgl.from_networkx(rag)
    if plot:
        edges_drawn_all = display_edges(label_rgb, rag, np.inf)
        show_img(edges_drawn_all,filename=filename)

    g.ndata["atom_features"] = torch.tensor(np.array(node_features)).type(torch.get_default_dtype())
    edge_data = torch.tensor(
        np.array([i[2]["weight"] for i in ((list(rag.edges.data())) * 2)])
    )  # NEED TO CHECK SRC/DST connections
    g.edata["r"] = edge_data.type(torch.get_default_dtype())
    lg=g.line_graph(shared=True)
    lg.apply_edges(compute_edge_props)
    return g,lg #,rag


def image_reshape(img):
    imgr = torchvision.transforms.functional.to_tensor(img)[np.newaxis,...]
    return imgr

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
            RBFExpansion(vmin=0, vmax=8.0, bins=config.edge_input_features,),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1, vmax=1.0, bins=config.triplet_input_features,
            ),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(config.hidden_features, config.hidden_features,)
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

        if self.classification:
            self.fc = nn.Linear(config.hidden_features, 5)
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
        #self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]
        self, image
    ):
        """ALIGNN : start with `atom_features`.
        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        g, lg = image_to_dgl_graph(image,resize=[256, 256])
        
        if len(self.alignn_layers) > 0:
            #g, lg = g
            lg = lg.local_var()

            # angle features (fixed)
            z = self.angle_embedding(lg.edata.pop("h"))

        g = g.local_var()

        # initial node features: atom feature network...
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)

        # initial bond features
        bondlength = g.edata.pop("r") #torch.norm(g.edata.pop("r"), dim=1)
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

#img = cv2.imread("JARVIS-2D-STM-JPG/JARVIS-2D-STM-JPG/JVASP-723_pos.jpg")
#g, lg = image_to_dgl_graph(img,resize=[256, 256],filename='x.png')
#model = ALIGNN(ALIGNNConfig(name='alignn',atom_input_features=9,classification=True))
#model(img)
