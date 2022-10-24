"""STEM torch dataloader for atom localization and crystal classification."""
import dgl
import torch
from jarvis.core.lattice import get_2d_lattice
import numpy as np
import pandas as pd
from skimage import draw
from jarvis.core.atoms import Atoms
from jarvis.core.specie import chem_data
from jarvis.db.figshare import data
from atomvision.data.stemconv import STEMConv
from collections.abc import Callable
from typing import Optional, List, Dict, Any
from skimage import measure
from scipy.spatial import KDTree
import networkx as nx
import os
from tqdm import tqdm


LABEL_MODES = {"delta", "radius", "predicted"}

# atomic radii
pt = pd.DataFrame(chem_data).T
pt = pt.sort_values(by="Z")
RADII = {int(row.Z): row.atom_rad for id, row in pt.iterrows()}


def atomic_radius_mask(shape, X, N, px_scale=0.1):
    """Atom localization masks, with footprints scaled to atomic radii."""
    # Atoms occluding each other along the Z (transmission) dimension are
    # not guaranteed to be masked nicely; these are not multilabel masks
    labels = np.zeros(shape, dtype=int)
    for x, n in zip(X, N):

        rr, cc = draw.disk(
            tuple(x), 0.5 * RADII[n] / px_scale, shape=labels.shape
        )
        labels[rr, cc] = n

    return labels


class Jarvis2dSTEMDataset:
    """Module for Simulated STEM dataset."""

    def __init__(
        self,
        px_scale: float = 0.1,
        label_mode: str = "delta",
        image_data: Optional[List[Dict[str, Any]]] = None,
        rotation_degrees: Optional[float] = None,
        shift_angstrom: Optional[float] = None,
        zoom_pct: Optional[float] = None,
        to_tensor: Optional[Callable] = None,
        localization_model=None,
        n_train=None,
        n_val=None,
        n_test=None,
        val_frac=0.1,
        test_frac=0.1,
        keep_data_order=False,
    ):
        """Intialize Simulated STEM dataset."""
        """
        px_scale: pixel size in angstroms
        label_mode: `delta` or `radius`, controls atom localization mask style
        adding label mode "predicted" to generate mask using loaded
        localization model

        For "predicted" labelling, localization_model cannot be None
        ## augmentation settings
        rotation_degrees: if specified, sample from
        Unif(-rotation_degrees, rotation_degrees)
        shift_angstrom: if specified, sample from
        Unif(-shift_angstrom, shift_angstrom)
        zoom_pct: optional image scale factor: s *= 1 + (zoom_pct/100)

        """
        print("n_train", n_train)
        # import sys
        # sys.exit()
        if label_mode not in LABEL_MODES:
            raise NotImplementedError(f"label mode {label_mode} not supported")

        self.px_scale = px_scale
        self.label_mode = label_mode
        self.to_tensor = to_tensor
        self.model = localization_model
        self.rotation_degrees = rotation_degrees
        self.shift_angstrom = shift_angstrom
        self.zoom_pct = zoom_pct
        self.keep_data_order = keep_data_order
        if image_data is not None:
            self.df = pd.DataFrame(image_data)
        else:
            # dft_2d = data("dft_2d")
            # Overriding the crys with 2D lattice type
            self.df = pd.DataFrame(data("dft_2d"))
            self.df["crys"] = self.df["atoms"].apply(
                lambda x: get_2d_lattice(x)[0]
            )
        print(self.df)
        self.stem = STEMConv(output_size=[256, 256])

        train_ids, val_ids, test_ids = self.split_dataset(
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            val_frac=val_frac,
            test_frac=test_frac,
        )
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids

        # label encoding dictionary: Dict[str, int]
        self.class_labels = {
            key: id for id, key in enumerate(self.df.crys.unique())
        }
        self.n_classes = len(self.class_labels)
        print("Data n_classes", len(self.class_labels), self.n_classes)

    def split_dataset(
        self,
        n_train=None,
        n_val=None,
        n_test=None,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
    ):
        """Split dataset."""
        N = len(self.df)
        if n_train is None:
            n_val = int(N * val_frac)
            n_test = int(N * test_frac)
            n_train = N - (n_val + n_test)
        if not self.keep_data_order:
            # set a consistent train/val/test split
            torch.manual_seed(0)
            shuf = torch.randperm(N)
            torch.random.seed()
            train_ids = shuf[:n_train].tolist()
            val_ids = shuf[n_train : n_train + n_val].tolist()
            test_ids = shuf[
                n_train + n_val : n_train + n_val + n_test
            ].tolist()
        else:
            ids = list(np.arange(N))
            train_ids = ids[:n_train]
            val_ids = ids[-(n_val + n_test) : -n_test]
            test_ids = ids[-n_test:]
        print("train_ids", len(train_ids), n_train)
        print("val_ids", len(val_ids), n_val)
        print("test_ids", len(test_ids), n_test)

        return train_ids, val_ids, test_ids

    def __len__(self):
        """Get Datset size: len(jarvis_2d)."""
        return self.df.shape[0]

    def __getitem__(self, idx):
        """Get Sample:image,label mask,atomic coords,numbers,structure ids."""
        row = self.df.iloc[idx]
        # print (row.jid)
        a = Atoms.from_dict(row.atoms)

        # defaults:
        rot = 0
        shift_x = 0
        shift_y = 0
        px_scale = self.px_scale

        # apply pre-rendering structure augmentation
        if self.rotation_degrees is not None:
            rot = np.random.uniform(
                -self.rotation_degrees, self.rotation_degrees
            )

        if self.shift_angstrom is not None:
            shift_x, shift_y = np.random.uniform(
                -self.shift_angstrom, self.shift_angstrom, size=2
            )

        if self.zoom_pct is not None:
            frac = self.zoom_pct / 100
            px_scale *= 1 + np.random.uniform(-frac, frac)

        image, label, pos, nb = self.stem.simulate_surface(
            a, px_scale=px_scale, eps=0.6, rot=rot, shift=[shift_x, shift_y]
        )

        if self.label_mode == "radius":
            label = atomic_radius_mask(image.shape, pos, nb, px_scale)
        # elif self.label_mode == "predicted":
        #    from atomvision.models.segmentation_utils
        #    import get_segmented_image
        #    label = get_segmented_image(image, self.model)

        else:
            raise ValueError("Currently unsupported label mode")

        if self.to_tensor is not None:
            image = self.to_tensor(torch.tensor(image))

        sample = {
            "image": image,
            "label": torch.FloatTensor(label > 0),
            "id": row.jid,
            "px_scale": px_scale,
            "crys": self.class_labels[row.crys],
        }
        return sample

    def get_rotation_series(self, idx, angles=np.linspace(0, 90, 32)):
        """Get a series of augmentations."""
        """
        ```python
        samples = dataset.get_rotation_series(0)
        angle_batch = dataloader.collate_fn(samples)
        graphs, targets = prepare_batch(angle_batch)
        ps = gnn(a_graphs)

        ```
        """
        row = self.df.iloc[idx]
        # print (row.jid)
        a = Atoms.from_dict(row.atoms)

        # defaults:
        # rot = 0
        shift_x = 0
        shift_y = 0
        px_scale = self.px_scale

        # apply pre-rendering structure augmentation
        if self.shift_angstrom is not None:
            shift_x, shift_y = np.random.uniform(
                -self.shift_angstrom, self.shift_angstrom, size=2
            )

        samples = []
        for angle in angles:
            image, label, pos, nb = self.stem.simulate_surface(
                a,
                px_scale=px_scale,
                eps=0.6,
                rot=angle,
                shift=[shift_x, shift_y],
            )

            if self.label_mode == "radius":
                label = atomic_radius_mask(image.shape, pos, nb, px_scale)

            if self.to_tensor is not None:
                image = self.to_tensor(torch.tensor(image))

            sample = {
                "image": image,
                "label": torch.FloatTensor(label > 0),
                "id": row.jid,
                "px_scale": px_scale,
                "crys": self.class_labels[row.crys],
            }
            samples.append(sample)

        return samples


def write_image_directory(dft_2d, train_ids, test_ids, outdir="stem"):
    """Write images to directory."""
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    train_list = []
    test_list = []
    for i in tqdm(dft_2d):
        structure = Atoms.from_dict(i["atoms"])
        img = STEMConv(output_size=[256, 256]).simulate_surface(
            structure, px_scale=0.1, eps=0.6
        )[0]
        img = np.array(img) / np.amax(img)
        label = get_2d_lattice(structure.to_dict())[1]
        np.savetxt(os.path.join(outdir, i["jid"] + ".txt"), img, delimiter=",")
        if i["jid"] in train_ids:
            train_list.append([i["jid"], label])
        elif i["jid"] in test_ids:
            test_list.append([i["jid"], label])
    np.savetxt(
        os.path.join(outdir, "training_set_labels.txt"), train_list, fmt="%s"
    )
    np.savetxt(
        os.path.join(outdir, "test_set_labels.txt"), test_list, fmt="%s"
    )
    return np.array(train_list), np.array(test_list)


def atom_mask_to_graph(label, image, px_angstrom=0.1, cutoff_angstrom=4):
    """Construct attributed atomistic graph from foreground mask."""
    """
    px_angstrom: pixel size in angstrom
    Performs connected component analysis on label image
    Computes region properties (centroids, radius, mean intensity)
    Constructs a radius graph of atoms within `cutoff` (px)
    """
    g = nx.Graph()
    g.graph["px_angstrom"] = px_angstrom

    # connected component analysis
    rlab = measure.label(label)

    # per-atom-detection properties for node attributes
    props = pd.DataFrame(
        measure.regionprops_table(
            rlab,
            intensity_image=image / image.max(),
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
    for id, row in props.iterrows():
        # px * angstrom/px -> angstrom
        pos = np.array([row["centroid-1"], row["centroid-0"], 0]) * px_angstrom
        eq_radius = 0.5 * row.equivalent_diameter * px_angstrom
        g.add_node(id, pos=pos, intensity=row.mean_intensity, r=eq_radius)

    # construct radius graph edges via kd-tree
    points = props.loc[:, ("centroid-1", "centroid-0")].values * px_angstrom
    nbrs = KDTree(points)
    g.add_edges_from(nbrs.query_pairs(cutoff_angstrom))

    return g, props


def bond_vectors(edges):
    """Compute bond displacement vectors from pairwise atom coordinates."""
    u = edges.src["pos"]
    v = edges.dst["pos"]
    return {"r": v - u}


def to_dgl(g):
    """Construct atom detection DGLGraph from networkx graph."""
    g = dgl.from_networkx(g, node_attrs=["pos", "intensity", "r"])

    # compute bond vectors from atomic coordinates
    # store results in g.edata["r"]
    g.apply_edges(bond_vectors)
    g.edata["r"] = g.edata["r"].type(torch.float32)

    # coalesce atom features
    h = torch.stack((g.ndata["intensity"], g.ndata["r"]), dim=1)
    g.ndata["atom_features"] = h.type(torch.float32)

    return g


def build_prepare_graph_batch(model, prepare_image_batch):
    """Close over atom localization model and image batch prep."""
    """
    Running the pixel classifier like this in the dataloader
    is not the most efficient. It might be viable to put this
    inside a closure used for a dataloader collate_fn. This would
    assemble the full batch, run the pixel classifier, and then
    construct the atomistic graphs. Hopefully this is all
    done during DataLoader prefetch still.

    Depending on the quality of the label predictions,
    this could potentially need label smoothing as well.


    example: initialize GCN-only ALIGNN with two atom input features
    cfg = alignn.ALIGNNConfig(name="alignn",
    alignn_layers=0, atom_input_features=2)
    model = alignn.ALIGNN(cfg)
    model(g)
    """

    def prepare_graph_batch(
        batch: Dict[str, torch.Tensor],
        device=None,
        non_blocking=False,
    ):
        """Extract image and mask from batch dictionary."""
        x, mask = prepare_image_batch(
            batch, device=device, non_blocking=non_blocking
        )

        with torch.no_grad():
            yhat = model(x).detach().cpu()

        predicted_mask = torch.sigmoid(yhat.squeeze()).numpy() > 0.5

        batch_size = x.size(0)
        graphs = [
            atom_mask_to_graph(
                predicted_mask[idx],
                batch["image"][idx, 0].numpy(),
                batch["px_scale"][idx].item(),
            )[0]
            for idx in range(batch_size)
        ]
        graphs = [to_dgl(g) for g in graphs]

        return dgl.batch(graphs).to(device), batch["crys"].to(device)

    return prepare_graph_batch
