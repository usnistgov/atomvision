"""General methods to convert image to DGL graph."""
import torch
import segmentation_models_pytorch as smp
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from skimage import draw
from skimage.feature import blob_log
import networkx as nx
import dgl
from scipy.spatial import KDTree
import pandas as pd

plt.switch_backend("agg")

# Image Modification


def crop_image(image, px=20):
    """Use to remove border pixels."""
    # image : 2D intensity image
    # border_pxl: int of list of ints. Number of
    # border pixels to remove. If scalar integer,
    # same number of pixels are removed from each side.
    # If array-type of length 4, give border pixel count in format
    if np.isscalar(px):
        crp_int = image[px:-px, px:-px]
    elif len(px) == 4:
        crp_int = image[px[0] : -px[1], px[2] : -px[3]]
    else:
        raise Exception("Incorrect format for px.")
    return crp_int


def image_unet_segmentation(
    image=None,
    threshold=0.5,
    image_path="example_image.png",
    model_path="checkpoint_100.pt",
):
    """Get UNet Segmentation and Graph Generation."""
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        encoder_depth=3,
        decoder_channels=(64, 32, 16),
        in_channels=3,
        classes=1,
    )
    state = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state["model"])

    model.eval()
    if image is None:
        image = cv2.imread(image_path)  # [:,:,0]
        print("xxx", len(image.shape))
    if len(image.shape) >= 3:
        image = image[:, :, 0]
    npx = cv2.resize(image, [256, 256], interpolation=cv2.INTER_AREA)
    scale = np.max(npx)
    npx = torch.tensor(np.tile(npx / scale, (3, 1, 1))[np.newaxis, ...])
    with torch.no_grad():
        yhat = model(npx.float()).detach()

    imx = torch.sigmoid(yhat.squeeze())
    lbl = imx >= threshold
    return imx, lbl


def get_blob_positions(image, method=blob_log, plot=False, saveto=None):
    """Get Blob Detection and Graph Generation."""
    blobs_list = method(np.array(image), threshold=0, min_sigma=5, max_sigma=8)
    if plot:
        fig, ax = plt.subplots()
        plt.imshow(np.array(image))
        if saveto is not None:
            plt.savefig(saveto + "_img.png", bbox_inches="tight")
            plt.close()
        fig, ax = plt.subplots()
        for y, x, r in blobs_list:
            plt.scatter(x, y, s=5, color="white")
            c = plt.Circle((x, y), r, color="red", linewidth=2, fill=False)
            ax.add_patch(c)
        plt.imshow(np.array(image))
        if saveto is not None:
            plt.savefig(saveto + "_blobs.png", bbox_inches="tight")
            plt.close()
    return blobs_list


def bond_vector(edges):
    """Compute bond vectors from node pairs."""
    u = edges.src["pos"]
    v = edges.dst["pos"]
    return {"r": v - u}


def compute_edge_props(edges):
    """Compute edge properties."""
    r1 = -edges.src["r"]
    r2 = edges.dst["r"]
    # diffs =r2-r1
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    return {"h": bond_cosine}


def convert_to_dgl(gx, node_attrs, props):
    """Convert from networkx to dgl graph."""
    # Adds bond vectors as edge attributes.

    # Need to add other node attributes to better distinguish chemical species:
    # 1. Average intensity
    # 2. Min/Max intensity
    g = dgl.from_networkx(gx, node_attrs)
    g.ndata["atom_features"] = torch.tensor(
        np.array(
            props[
                ["mean_intensity", "max_intensity", "min_intensity", "radius"]
            ]
        )
    ).type(torch.float32)
    # print("atom_features", g.ndata["atom_features"])
    # print("atom_features shape", g.ndata["atom_features"].shape)
    g.apply_edges(bond_vector)
    g.edata["r"] = g.edata["r"].type(torch.float32)
    lg = g.line_graph(shared=True)
    lg.apply_edges(compute_edge_props)
    return g, lg


def blob_list_to_graph(
    im, blobs_log, px_angstrom=0.1, cutoff_angstrom=4, plot=False, saveto=None
):
    """Convert blob list to graph."""
    g = nx.Graph()
    # Add position and radius as attributes
    # Generate an atom mask from blobs
    mask = np.zeros(im.shape[0:2], dtype=int)
    # rows = []
    for indx, row in enumerate(blobs_log):
        pos = (row[1], row[0])
        radius = row[2]
        # g.add_node(indx, pos = pos, radius = radius)
        rr, cc = draw.disk(pos, radius, shape=mask.shape)
        mask[rr, cc] = 1

    # connected component analysis
    rlab = measure.label(mask)

    # per-atom-detection properties for node attributes
    props = pd.DataFrame(
        measure.regionprops_table(
            rlab,
            intensity_image=im / im.max(),
            properties=[
                "label",
                "centroid",
                "equivalent_diameter",
                "min_intensity",
                "mean_intensity",
                "max_intensity",
            ],
        )
    )
    # add nodes with attributes to graph
    radius_col = []
    for idx, row in props.iterrows():
        # px * angstrom/px -> angstrom
        pos = np.array([row["centroid-0"], row["centroid-1"], 0]) * px_angstrom
        eq_radius = 0.5 * row.equivalent_diameter * px_angstrom
        radius_col.append(eq_radius)
        g.add_node(
            idx,
            pos=pos,
            mean_intensity=row["mean_intensity"],
            max_intensity=row["max_intensity"],
            min_intensity=row["min_intensity"],
            r=eq_radius,
        )
    props["radius"] = radius_col
    points = props.loc[:, ("centroid-0", "centroid-1")].values * px_angstrom
    nbrs = KDTree(points)
    g.add_edges_from(nbrs.query_pairs(cutoff_angstrom))
    if plot:
        nx.draw_networkx(
            g,
            props.loc[:, ("centroid-0", "centroid-1")].values,
            edge_color="r",
            node_color="xkcd:white",
            node_size=100,
            font_size=8,
        )
        plt.imshow(im)
        if saveto:
            plt.savefig(saveto + "_graphs.png", bbox_inches="tight")
            plt.close()
    node_attrs = [
        "pos",
        "mean_intensity",
        "max_intensity",
        "min_intensity",
        "r",
    ]
    g_dgl, lg_dgl = convert_to_dgl(g, node_attrs, props)
    return g_dgl, lg_dgl


if __name__ == "__main__":
    # img = cv2.imread("../../notebooks/JVASP-723_pos.jpg")
    img = np.genfromtxt(
        "../data/2D_STEM_Images/JVASP-76516.txt", delimiter=","
    )
    blobs_list = get_blob_positions(img, plot=True)
    g, lg = blob_list_to_graph(img, blobs_list)
