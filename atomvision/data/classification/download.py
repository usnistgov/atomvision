"""Download data from figshare."""
import requests
import zipfile
import os
from tqdm import tqdm


def get_db_info():
    """Get DB info."""
    db_info = {
        # https://doi.org/10.6084/m9.figshare.16788268
        "stem_c2db": [
            "https://figshare.com/ndownloader/files/31054516",
            "stem_c2db.zip",
            "Obtaining C2DB 2D STEM image dataset 3.5k ...",
            "https://iopscience.iop.org/article/10.1088/2053-1583/aacfc1",
        ],
        "stem_jv2d": [
            "https://figshare.com/ndownloader/files/31054519",
            "stem_jv2d.zip",
            "Obtaining JARVIS-DFT 2D STEM image dataset 1.1k ...",
            "https://www.nature.com/articles/s41524-020-00440-1",
        ],
        "stm_jv2d": [
            "https://figshare.com/ndownloader/files/31054525",
            "stm_jv2d.zip",
            "Obtaining JARVIS-DFT 2D STM image dataset 1.4k ...",
            "https://www.nature.com/articles/s41524-020-00440-1",
        ],
        "stm_tdmp": [
            "",
            "stm_jv2d.zip",
            "Obtaining 2DMatPefia 2D STM image dataset 6k ...",
            "https://www.nature.com/articles/s41597-019-0097-3",
        ],
    }
    return db_info


def data(
    dataset="stem_jv2d",
    my_path=None,
):
    """Get data with progress bar."""
    db_info = get_db_info()
    if dataset not in list(db_info.keys()):
        raise ValueError("Check DB name options.")

    zfile = dataset + ".zip"
    url = db_info[dataset][0]

    path = str(os.path.join(os.path.dirname(__file__), zfile))
    # path = str(os.path.join(os.path.dirname(__file__), js_tag))
    if not os.path.isfile(path):
        # zfile = str(os.path.join(os.path.dirname(__file__), "tmp.zip"))
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(
            total=total_size_in_bytes, unit="iB", unit_scale=True
        )
        with open(path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        # f = open(zfile, "wb")
        # f.write(r.content)
        # f.close()
    if my_path is None:
        my_path = os.path.abspath(
            str(os.path.join(os.path.dirname(__file__), "."))
        )
        print("Found extract_path as None.")
    print("Loading and extracting the zipfile...")
    final_path = os.path.join(my_path, dataset)
    if not os.path.exists(final_path):
        info = zipfile.ZipFile(path).extractall(my_path)
        print("Here info", info)
    print("Files are be kept at:", final_path)
    print("Process completed.")

    return final_path
