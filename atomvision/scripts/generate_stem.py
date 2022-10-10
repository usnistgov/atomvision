"""Module to generate dataset."""
from sklearn.model_selection import train_test_split
from jarvis.core.lattice import get_2d_lattice
import matplotlib.pyplot as plt

# import numpy as np
from jarvis.db.figshare import data
from atomvision.data.stemconv import STEMConv
from jarvis.core.atoms import Atoms
from tqdm import tqdm
import os
import pandas as pd


def dft_2d_only():
    """Get JARVIS-DFT-2D dataset only."""
    # images = []
    labels = []
    # graphs = []
    # line_graphs = []
    # batch_size = 32
    test_size = 0.25
    dft_2d = data("dft_2d")  # [0:100]
    mem = []
    labels = []
    for i in dft_2d:
        mem.append(i["jid"])
        labels.append(get_2d_lattice(i["atoms"])[1])
    tr, ts = train_test_split(mem, stratify=labels, test_size=test_size)

    labels = []
    # graphs = []
    # line_graphs = []
    cwd = os.getcwd()
    os.makedirs("train_folder")
    os.chdir("train_folder")
    for i in tqdm(dft_2d):
        if i["jid"] in tr:
            structure = Atoms.from_dict(i["atoms"])
            img = STEMConv(output_size=[256, 256]).simulate_surface(structure)[
                0
            ]
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
    # graphs = []
    # line_graphs = []
    for i in tqdm(dft_2d):
        if i["jid"] in ts:
            structure = Atoms.from_dict(i["atoms"])
            img = STEMConv(output_size=[256, 256]).simulate_surface(structure)[
                0
            ]
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


def get_combined_data():
    """Get JARVIS-DFT-2D+C2DB+TWOD_MAT_Ped dataset."""
    dft_2d = pd.DataFrame(data("dft_2d"))
    c2_db = pd.DataFrame(data("c2db"))
    twod_matp = pd.DataFrame(data("twod_matpd"))
    c2_db["spg_formula"] = c2_db["atoms"].apply(
        lambda x: (Atoms.from_dict(x)).composition.reduced_formula
        + "_"
        + str((Atoms.from_dict(x)).get_spacegroup[0])
    )
    twod_matp["spg_formula"] = twod_matp["atoms"].apply(
        lambda x: (Atoms.from_dict(x)).composition.reduced_formula
        + "_"
        + str((Atoms.from_dict(x)).get_spacegroup[0])
    )
    dft_2d["spg_formula"] = dft_2d["atoms"].apply(
        lambda x: (Atoms.from_dict(x)).composition.reduced_formula
        + "_"
        + str((Atoms.from_dict(x)).get_spacegroup[0])
    )
    dft_2d["id"] = dft_2d["jid"]
    df1 = (
        pd.concat([dft_2d, c2_db, twod_matp])
        .drop_duplicates("spg_formula")
        .reset_index(drop=True)
    )
    print(len(dft_2d) + len(c2_db) + len(twod_matp))
    df1["lat"] = df1["atoms"].apply(lambda x: get_2d_lattice(x)[1])
    plt.hist(df1["lat"].values)
    plt.xticks([0, 1, 2, 3, 4])
    plt.savefig("dist.png")
    plt.close()
    # images = []
    labels = []
    # graphs = []
    # line_graphs = []
    # batch_size = 32
    test_size = 0.25
    # dft_2d = data("dft_2d") #[0:100]
    # c2_db = data("c2db")
    # twod_matp = data("twod_matpd")

    mem = []
    labels = []
    for ii, i in df1.iterrows():
        mem.append(i["id"])
        labels.append(get_2d_lattice(i["atoms"])[1])

    tr, ts = train_test_split(mem, stratify=labels, test_size=test_size)
    labels = []
    # graphs = []
    # line_graphs = []
    cwd = os.getcwd()
    os.makedirs("train_folder")
    os.chdir("train_folder")
    for ii, i in tqdm(df1.iterrows()):
        if i["id"] in tr:
            structure = Atoms.from_dict(i["atoms"])
            img = STEMConv(output_size=[256, 256]).simulate_surface(structure)[
                0
            ]
            label = get_2d_lattice(structure.to_dict())[1]
            if not os.path.exists(str(label)):
                os.makedirs(str(label))
            filename = str(label) + "/" + str(i["id"]) + ".jpg"
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
    # graphs = []
    # line_graphs = []
    for ii, i in tqdm(df1.iterrows()):
        # for i in tqdm(dft_2d):
        if i["id"] in ts:
            structure = Atoms.from_dict(i["atoms"])
            img = STEMConv(output_size=[256, 256]).simulate_surface(structure)[
                0
            ]
            label = get_2d_lattice(structure.to_dict())[1]
            if not os.path.exists(str(label)):
                os.makedirs(str(label))
            filename = str(label) + "/" + str(i["id"]) + ".jpg"
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
