from jarvis.core.atoms import crop_square
import matplotlib.pyplot as plt
import os
import glob
from jarvis.analysis.stem.convolution_apprx import STEMConv
from jarvis.db.figshare import data  # , get_jid_data
from jarvis.core.atoms import Atoms, get_supercell_dims
from jarvis.core.lattice import get_2d_lattice
from sklearn.model_selection import train_test_split
from collections import defaultdict

plt.switch_backend("agg")


dft_2d = data("dft_2d")
for i in dft_2d:
    a = Atoms.from_dict(i["atoms"])
    jid = i["jid"]
    dims = get_supercell_dims(atoms=a, enforce_c_size=50)
    s = a.make_supercell_matrix(dims)
    c = crop_square(s, csize=20)
    p = STEMConv(atoms=c, output_size=[200, 200]).simulate_surface()
    plt.imshow(p[50:150, 50:150], interpolation="gaussian", cmap="gray")
    plt.tight_layout()
    plt.axis("off")
    filename = jid + ".jpg"
    plt.savefig(filename)
    plt.close()


lat_dict = defaultdict()
for i in dft_2d:
    lat_dict[i["jid"]] = get_2d_lattice(i["atoms"])[1]
lat_dict_data = defaultdict(list)

for i in glob.glob("JVASP-*.jpg"):
    jid = i.split(".jpg")[0]
    lat_type = lat_dict[jid]
    lat_dict_data[lat_type].append(i)
train_folder = "train_folder"
test_folder = "test_folder"
test_size = 0.2
if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

for i, j in lat_dict_data.items():
    splits = train_test_split(list(j), test_size=test_size)
    train_samples = splits[0]
    cwd = os.getcwd()
    fold_name = os.path.join(str(train_folder), str(i))
    if not os.path.exists(fold_name):
        os.makedirs(fold_name)
    for k in train_samples:
        cmd = "mv " + k + " " + train_folder + "/" + str(i)
        os.system(cmd)
    os.chdir(cwd)
    test_samples = splits[1]
    fold_name = os.path.join(str(test_folder), str(i))
    if not os.path.exists(fold_name):
        os.makedirs(fold_name)
    for k in test_samples:
        cmd = "mv " + k + " " + test_folder + "/" + str(i)
        os.system(cmd)


"""
for ii, i in enumerate(dft_2d):
    a = Atoms.from_dict(i["atoms"])
    lat_type=str(get_2d_lattice(a.to_dict())[1])
    if not os.path.exists(lat_type):
        os.makedirs(lat_type)
    jid = i["jid"]
    print(ii,jid,lat_type)
    dims = get_supercell_dims(atoms=a, enforce_c_size=50)
    s = a.make_supercell_matrix(dims)
    c = crop_square(s, csize=20)
    p = STEMConv(atoms=c, output_size=[200, 200]).simulate_surface()
    plt.imshow(p[50:150, 50:150], interpolation="gaussian", cmap="gray")
    plt.tight_layout()
    plt.axis("off")
    filename = lat_type+"/STEM-" + jid + ".jpg"
    plt.savefig(filename)
    plt.close()
    # break
"""
