# import matplotlib
from sklearn.model_selection import train_test_split
from jarvis.core.lattice import get_2d_lattice
import matplotlib.pyplot as plt
import numpy as np
from jarvis.db.figshare import data
from atomvision.data.stemconv import STEMConv
from jarvis.core.atoms import Atoms
from tqdm import tqdm
import os


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
cwd = os.getcwd()
os.makedirs("train_folder")
os.chdir("train_folder")
for i in tqdm(dft_2d):
    if i["jid"] in tr:
        structure = Atoms.from_dict(i["atoms"])
        img = STEMConv(output_size=[256, 256]).simulate_surface(structure)[0]
        label = get_2d_lattice(structure.to_dict())[1]
        if not os.path.exists(str(label)):
            os.makedirs(str(label))
        filename = str(label) + "/" + str(i["jid"]) + ".jpg"
        plt.imshow(img, interpolation="gaussian", cmap="gray")
        plt.tight_layout()
        plt.axis("off")
        plt.savefig(filename)
        plt.close()

        labels.append(label)
        # g, lg = image_to_dgl_graph_blob(image_data=img,device=device)
        # graphs.append(g)
        # line_graphs.append(lg)

os.chdir(cwd)
os.makedirs("test_folder")
os.chdir("test_folder")
labels = []
graphs = []
line_graphs = []
for i in tqdm(dft_2d):
    if i["jid"] in ts:
        structure = Atoms.from_dict(i["atoms"])
        img = STEMConv(output_size=[256, 256]).simulate_surface(structure)[0]
        label = get_2d_lattice(structure.to_dict())[1]
        if not os.path.exists(str(label)):
            os.makedirs(str(label))
        filename = str(label) + "/" + str(i["jid"]) + ".jpg"
        plt.imshow(img, interpolation="gaussian", cmap="gray")
        plt.tight_layout()
        plt.axis("off")
        plt.savefig(filename)
        plt.close()
        labels.append(label)
        # g, lg = image_to_dgl_graph_blob(image_data=img,device=device)
        # graphs.append(g)
        # line_graphs.append(lg)


os.chdir(cwd)
