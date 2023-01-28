"""Module to perform t-SNE on imamges."""
import sys
import argparse
import cv2
import torchvision.datasets as dset
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFile

from atomvision.scripts.image_to_graph import (
    crop_image,
    get_blob_positions,
    blob_list_to_graph,
)

# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# import torchvision.transforms as transforms
# https://raw.githubusercontent.com/GunhoChoi/PyTorch-FastCampus/master/07_Transfer_Learning/2_T-SNE/color_tsne.py
# python train_tsne.py --data_dir /wrk/knc6/AtomVision/Combined/train_folder
# %matplotlib inline
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_tsne(
    data_dir="New_stem_2d/train_folder",
    image_size=256,
    perplexity=30,
    filename=None,
):
    """Get T-SNE clustering."""
    #     transform = transforms.Compose(
    #         [
    #             transforms.ToTensor(),
    #             transforms.Resize(255),
    #             transforms.CenterCrop(224),
    #             transforms.Normalize((0.5,), (0.5,)),
    #         ]
    #     )

    data = dset.ImageFolder(data_dir)  # ,transform=transform)

    print(data.class_to_idx)
    img_list = []
    for i in data.imgs:
        img_list.append(i[0])

    total_arr = []
    label_arr = []
    for idx, (image_path, label) in enumerate(data.imgs):
        image = cv2.imread(image_path)

        image2 = cv2.resize(
            image, [image_size, image_size], interpolation=cv2.INTER_AREA
        ).T  # [np.newaxis,...]
        max_val = 1  # np.max(image2)

        i = (
            image2 / max_val
        ).flatten()  # torch.tensor(np.tile(image, (3,1, 1)))#[np.newaxis,...])

        total_arr.append(i.reshape(-1))
        label_arr.append(label)

        # print(idx)

    print(label_arr, len(label_arr))

    # Apply TSNE

    print("\n------Starting TSNE------\n")

    model = TSNE(
        n_components=2, perplexity=perplexity, random_state=1, init="warn"
    )
    result = model.fit_transform(np.array(total_arr))

    print("\n------TSNE Done------\n")

    print("\n------Starting the plot------\n")
    X_embedded = result

    # %matplotlib inline
    plt.rcParams.update({"font.size": 14})
    plt.figure(figsize=(8, 8))
    X = X_embedded
    x = X[:, 0]
    y = X[:, 1]

    term_list = list(np.array(label_arr))
    term_set = list(set(term_list))
    term_list = [term_set.index(term) for term in term_list]

    color_list = plt.cm.tab10(term_list)

    lbls = []
    xyz = []
    for i, j, k, p in zip(x, y, term_list, color_list):
        if k not in lbls:
            lbls.append(k)
            xyz.append([i, j, k])
            plt.scatter(i, j, s=10, c=p, label=term_set[k])

    plt.scatter(x, y, s=10, c=color_list)  # ,label=term_list)
    plt.legend(loc="lower left")

    plt.xticks([])
    plt.yticks([])
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


parser = argparse.ArgumentParser(description="AtomVison package.")
parser.add_argument(
    "--data_dir",
    default="data_dir",
    help="Folder with training images. Each class should have its own folder.",
)


def labelled_images_to_graphs(images, labels, border_pxl=0, saveto=""):
    """
    Convert labelled images to graphs.

    Args:
         images: list of image arrays
         labels: list of corresponding class labels
    """
    graphs = []
    line_graphs = []
    n = 0
    for img, lbl in zip(images, labels):
        if border_pxl != 0:
            img = crop_image(img, border_pxl)
        blob_list = get_blob_positions(img)
        g, lg = blob_list_to_graph(img, blob_list)
        graphs.append(g)
        line_graphs.append(lg)
        n = n + 1
    return graphs, line_graphs


def train_tsne_graph(
    data_dir="New_stem_2d/train_folder",
    image_size=256,
    border_pxl=0,
    perplexity=30,
    filename=None,
):
    """Train T-SNE with Graph features."""
    data = dset.ImageFolder(data_dir)

    image_arr = []
    label_arr = []

    for idx, (image_path, label) in enumerate(data.imgs):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        image2 = cv2.resize(
            image, [image_size, image_size], interpolation=cv2.INTER_AREA
        ).T  # [np.newaxis,...]
        max_val = 1  # np.max(image2)

        i = (
            image2 / max_val
        )  # torch.tensor(np.tile(image, (3,1, 1)))#[np.newaxis,...])

        image_arr.append(i)
        label_arr.append(label)
    plt.imshow(image_arr[0])
    plt.savefig("first_image.pdf", bbox_inches="tight")
    print("first image", image_arr[0])
    print("image_dim", image_arr[0].shape)
    print("image_arr len", len(image_arr))
    print("\n------Starting Graph Generation------\n")

    g_arr, lg_arr = labelled_images_to_graphs(
        np.array(image_arr), label_arr, border_pxl=50
    )

    edges = []
    for lg in lg_arr:
        angles = lg.edata["h"].numpy()
        hist, bin_edges = np.histogram(angles, 200, (-1, 1))
        edges.append(hist)
    print("\n------Starting TSNE------\n")

    model = TSNE(
        n_components=2, perplexity=perplexity, random_state=1, init="warn"
    )

    result = model.fit_transform(np.array(edges))

    print("\n------TSNE Done------\n")
    X_embedded = result

    # %matplotlib inline
    plt.rcParams.update({"font.size": 14})
    plt.figure(figsize=(8, 8))
    X = X_embedded
    x = X[:, 0]
    y = X[:, 1]

    term_list = list(np.array(label_arr))
    term_set = list(set(term_list))
    term_list = [term_set.index(term) for term in term_list]

    color_list = plt.cm.tab10(term_list)

    lbls = []
    xyz = []
    for i, j, k, p in zip(x, y, term_list, color_list):
        if k not in lbls:
            lbls.append(k)
            xyz.append([i, j, k])
            plt.scatter(i, j, s=10, c=p, label=term_set[k])

    plt.scatter(x, y, s=10, c=color_list)  # ,label=term_list)
    plt.legend(loc="lower left")

    plt.xticks([])
    plt.yticks([])
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    data_dir = str(args.data_dir)

    train_tsne_graph(data_dir=data_dir, filename="tsne_graph.pdf")
