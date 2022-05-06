from pathlib import Path
from atomvision.models.gcn import localization, gcn
from atomvision.models.cnn_classifiers import densenet
import os

config_json_file = str(os.path.join(os.path.dirname(__file__), "config.json"))


def test_gcn():
    x = localization(Path(config_json_file))
    y = gcn(Path(config_json_file))


test_gcn()
