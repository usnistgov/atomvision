from pathlib import Path
from atomvision.data.stemconv import STEMConv
import os
from jarvis.core.atoms import Atoms, crop_square
from jarvis.db.figshare import get_jid_data
import matplotlib.pyplot as plt

config_json_file = str(os.path.join(os.path.dirname(__file__), "config.json"))


def test_stemconv():
    plt.switch_backend("agg")

    a = Atoms.from_dict(get_jid_data("JVASP-667")["atoms"])
    c = crop_square(a)
    # c = a.make_supercell_matrix([2, 2, 1])
    p = STEMConv(atoms=c).simulate_surface(c)


#def test_gcn():
#    x = localization(Path(config_json_file))
#    y = gcn(Path(config_json_file))


# test_stemconv()
# test_gcn()
