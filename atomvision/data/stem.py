import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from skimage import draw

from jarvis.core.atoms import Atoms
from jarvis.core.specie import chem_data
from jarvis.db.figshare import data, get_jid_data

from atomvision.data.stemconv import STEMConv

LABEL_MODES = {"delta", "radius"}

# atomic radii
pt = pd.DataFrame(chem_data).T
pt = pt.sort_values(by="Z")
RADII = {int(row.Z): row.atom_rad for id, row in pt.iterrows()}


def atomic_radius_mask(shape, X, N, px_scale=0.1):
    labels = np.zeros(shape, dtype=int)
    for x, n in zip(X, N):

        rr, cc = draw.disk(tuple(x), 0.5 * RADII[n] / px_scale, shape=labels.shape)
        labels[rr, cc] = n

    return labels


class Jarvis2dSTEMDataset:
    """Simulated STEM dataset (jarvis dft_2d)"""

    def __init__(self, px_scale=0.1, label_mode="delta"):

        if label_mode not in LABEL_MODES:
            raise NotImplementedError(f"label mode {label_mode} not supported")

        self.px_scale = px_scale
        self.label_mode = label_mode

        self.df = pd.DataFrame(data("dft_2d"))
        self.stem = STEMConv(output_size=[256, 256])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        a = Atoms.from_dict(row.atoms)

        image, label, pos, nb = self.stem.simulate_surface(
            a, px_scale=self.px_scale, eps=0.6, rot=0, shift=[0, 0]
        )

        if self.label_mode == "radius":
            label = atomic_radius_mask(image.shape, pos, nb, self.px_scale)

        sample = {"image": image, "label": label, "coords": pos, "id": row.jid}
        return sample
