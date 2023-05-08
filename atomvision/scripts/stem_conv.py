#!/usr/bin/env python
from atomvision.data.stemconv import STEMConv
import sys
import argparse
from jarvis.core.atoms import Atoms
import matplotlib.pyplot as plt
from jarvis.analysis.defects.surface import Surface


parser = argparse.ArgumentParser(
    description="Generating STEM images using convolution approximation."
)
parser.add_argument(
    "--file_path",
    default="POSCAR",
    help="Input crystallographic file.",
)

parser.add_argument(
    "--file_format", default="poscar", help="poscar/cif/xyz/pdb file format."
)


parser.add_argument(
    "--output_path", default="STEM.png", help="Output filepath."
)
parser.add_argument("--output_size", default=256, help="Output pizel size.")
parser.add_argument(
    "--px_scale", default=0.2, help="Pixel ratio in angstrom/pixel."
)
parser.add_argument(
    "--surface_layers", default=1, help="Number of layers of surface."
)
parser.add_argument(
    "--power_factor", default=1.7, help="Power factor for STEM-Conv image."
)
parser.add_argument(
    "--miller_index",
    default="0_0_1",
    help="Miller index (as string input, separated by _).",
)


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    file_path = str(args.file_path)
    output_path = str(args.output_path)
    file_format = str(args.file_format)
    output_size = int(args.output_size)
    power_factor = float(args.power_factor)
    px_scale = float(args.px_scale)
    surface_layers = int(args.surface_layers)

    miller_index = [int(i) for i in (args.miller_index).split("_")]
    atoms = None
    if file_format == "poscar":
        atoms = Atoms.from_poscar(file_path)
    elif file_format == "cif":
        atoms = Atoms.from_cif(file_path)
    elif file_format == "xyz":
        # Note using 500 angstrom as box size
        atoms = Atoms.from_xyz(file_path, box_size=500)
    elif file_format == "pdb":
        # Note using 500 angstrom as box size
        # Recommended install pytraj
        # conda install -c ambermd pytraj
        atoms = Atoms.from_pdb(file_path, max_lat=500)
    else:
        raise NotImplementedError("File format not implemented", file_format)
    surface = Surface(
        atoms=atoms, indices=miller_index, layers=surface_layers
    ).make_surface()
    out = STEMConv(
        output_size=[output_size, output_size], power_factor=power_factor
    ).simulate_surface(surface, px_scale=px_scale)

    plt.imshow(out[0], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(output_path)
    plt.close()
