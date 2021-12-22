"""Simulated STEM pytorch dataloader for atom localization and crystal classification."""
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
    """Atom localization masks, with footprints scaled to atomic radii.

    Atoms occluding each other along the Z (transmission) dimension are
    not guaranteed to be masked nicely; these are not multilabel masks
    """
    labels = np.zeros(shape, dtype=int)
    for x, n in zip(X, N):

        rr, cc = draw.disk(
            tuple(x), 0.5 * RADII[n] / px_scale, shape=labels.shape
        )
        labels[rr, cc] = n

    return labels


"""
#excluded=['JVASP-76418','JVASP-76611','JVASP-19999','JVASP-76567','JVASP-652','JVASP-6379','JVASP-60567','JVASP-60331','JVASP-8981','JVASP-8984','JVASP-60475','JVASP-31368','JVASP-75366','JVASP-75078','JVASP-60353','JVASP-27957','JVASP-6346','JVASP-676','JVASP-76604']
excluded=['JVASP-60433']
my_data=[]
for i in data("dft_2d"):
    if i['jid'] not in excluded and len(my_data)<129:
        my_data.append(i)

"""
# my_data = data("dft_2d")[0:128]


class Jarvis2dSTEMDataset:
    """Simulated STEM dataset (jarvis dft_2d)"""

    def __init__(self, px_scale=0.1, label_mode="delta", image_data=[]):
        """Simulated STEM dataset, jarvis-2d data

        px_scale: pixel size in angstroms
        label_mode: `delta` or `radius`, controls atom localization mask style
        """

        if label_mode not in LABEL_MODES:
            raise NotImplementedError(f"label mode {label_mode} not supported")

        self.px_scale = px_scale
        self.label_mode = label_mode

        self.df = pd.DataFrame(image_data)
        self.stem = STEMConv(output_size=[256, 256])
        # self.stem = STEMConv(output_size=[256, 256])

    def __len__(self):
        """Datset size: len(jarvis_2d)"""
        return self.df.shape[0]

    def __getitem__(self, idx):
        """Sample: image, label mask, atomic coords, numbers, structure ids."""
        row = self.df.iloc[idx]
        # print (row.jid)
        a = Atoms.from_dict(row.atoms)

        image, label, pos, nb = self.stem.simulate_surface(
            a, px_scale=self.px_scale, eps=0.6, rot=0, shift=[0, 0]
        )

        if self.label_mode == "radius":
            label = atomic_radius_mask(image.shape, pos, nb, self.px_scale)

        sample = {
            "image": torch.tensor(image, dtype=torch.float32),
            "label": label,
            "id": row.jid,
        }
        # sample = {"image": image, "label": label, "coords": pos, "id": row.jid}
        return sample
