"""Module to simulate STEM images using convoltuin approximation."""
# Adapted from https://github.com/jacobjma/fourier-scale-calibration
import numpy as np
from scipy.interpolate import interp1d

# from numbers import Number
from jarvis.core.utils import gaussian
from jarvis.core.utils import lorentzian2 as lorentzian
from jarvis.core.atoms import Atoms, get_supercell_dims, crop_square


def make_supercell(atoms=None, csize=10):
    """Crop a square portion from a surface/2D system."""
    sz = csize / 2
    # sz = csize
    # just to make sure we have enough material to crop from
    enforce_c_size = sz * 3
    dims = get_supercell_dims(atoms, enforce_c_size=enforce_c_size)
    b = atoms.make_supercell_matrix(dims).center_around_origin()
    lat_mat = [
        [enforce_c_size, 0, 0],
        [0, enforce_c_size, 0],
        [0, 0, b.lattice_mat[2][2]],
    ]

    els = []
    coords = []
    for i, j in zip(b.cart_coords, b.elements):
        if True:  # i[0] <= sz and i[0] >= -sz and i[1] <= sz and i[1] >= -sz:
            els.append(j)
            coords.append(i)
    coords = np.array(coords)
    # new_mat = (
    #    [max(coords[:, 0]) - min(coords[:, 0]) + tol, 0, 0],
    #    [0, max(coords[:, 1]) - min(coords[:, 1]) + tol, 0],
    #    [0, 0, b.lattice_mat[2][2]],
    # )
    new_atoms = Atoms(
        lattice_mat=lat_mat, elements=els, coords=coords, cartesian=True
    ).center_around_origin([0.5, 0.5, 0.5])
    return new_atoms


class STEMConv(object):
    """Module to simulate STEM images using convoltuin approximation."""

    def __init__(
        self,
        atoms=None,
        output_size=[50, 50],
        power_factor=1.7,
        gaussian_width=0.5,
        lorentzian_width=0.5,
        intensity_ratio=0.5,
        nbins=100,
        tol=0.5,
        crop=False,
    ):
        """
        Intitialize the class.
        """
        self.atoms = atoms
        self.output_size = output_size
        self.power_factor = power_factor
        self.gaussian_width = gaussian_width
        self.lorentzian_width = lorentzian_width
        self.intensity_ratio = intensity_ratio
        self.nbins = nbins
        self.tol = tol
        self.crop = crop

    def superpose_deltas(self, positions, array):
        """Superpose deltas."""
        z = 0
        shape = array.shape[-2:]
        rounded = np.floor(positions).astype(np.int32)
        rows, cols = rounded[:, 0], rounded[:, 1]

        array[z, rows, cols] += (1 - (positions[:, 0] - rows)) * (
            1 - (positions[:, 1] - cols)
        )
        array[z, (rows + 1) % shape[0], cols] += (positions[:, 0] - rows) * (
            1 - (positions[:, 1] - cols)
        )
        array[z, rows, (cols + 1) % shape[1]] += (1 - (positions[:, 0] - rows)) * (
            positions[:, 1] - cols
        )
        array[z, (rows + 1) % shape[0], (cols + 1) % shape[1]] += (
            rows - positions[:, 0]
        ) * (cols - positions[:, 1])
        return array

    def simulate_surface(self):
        """Simulate a STEM image."""

        output_size = np.squeeze(self.output_size)
        extent = np.diag(self.atoms.lattice_mat)[:2]
        sampling = tuple(extent / output_size)
        sampling = (sampling[0], sampling[1])

        margin = int(np.ceil(5 / min(sampling)))  # int like 20
        shape_w_margin = tuple(output_size + 2 * margin)

        # Set up real-space grid (in angstroms?)
        x = np.fft.fftfreq(shape_w_margin[0]) * shape_w_margin[0] * sampling[0]
        y = np.fft.fftfreq(shape_w_margin[1]) * shape_w_margin[1] * sampling[1]
        r = np.sqrt(x[:, None] ** 2 + y[None] ** 2)

        # construct the probe profile centered at (0,0) on the periodic spatial grid
        x = np.linspace(0, 4 * self.lorentzian_width, self.nbins)
        profile = gaussian(x, self.gaussian_width) + self.intensity_ratio * lorentzian(
            x, self.lorentzian_width
        )

        profile /= profile.max()
        f = interp1d(x, profile, fill_value=0, bounds_error=False)
        intensity = f(r)

        # project atomic coordinates onto the image
        positions = self.atoms.cart_coords[:, :2] / sampling - self.tol

        # Check if atoms are within the specified range
        inside = (
            (positions[:, 0] > -margin)
            & (positions[:, 1] > -margin)
            & (positions[:, 0] < self.output_size[0] + margin)
            & (positions[:, 1] < self.output_size[1] + margin)
        )

        inside = np.ones(positions.shape[0], dtype=bool)
        positions = positions[inside] + margin

        numbers = np.array(self.atoms.atomic_numbers)[inside]

        array = np.zeros((1,) + shape_w_margin)  # adding extra 1
        mask = np.zeros((1,) + shape_w_margin)
        for number in np.unique(np.array(self.atoms.atomic_numbers)):

            temp = np.zeros((1,) + shape_w_margin)
            temp = self.superpose_deltas(positions[numbers == number], temp)
            array += temp * number ** self.power_factor
            temp = np.where(temp > 0, number, temp)
            mask += temp[0]

        array = np.fft.ifft2(np.fft.fft2(array) * np.fft.fft2(intensity)).real

        scale = 1
        sel = slice(scale * margin, -scale * margin)
        # sel = slice(1, -1)
        array = array[0, sel, sel]
        mask = mask[0, sel, sel]
        positions = positions - scale * margin

        # array = array[0, margin:-margin, margin:-margin]
        # mask = mask[0, margin:-margin, margin:-margin]

        if self.crop:
            sel = slice(margin, -margin)
            return array[sel, sel], mask[sel, sel], positions - 2 * margin

        return array, mask, positions

    def simulate_surface_2(self, atoms, px_scale=0.2, eps=0.1):
        """Simulate a STEM image.

        px_scale: pixel size in angstroms/px
        """

        # instead of padding and all this first,
        # construct the probe array with the output target size
        # fftshift, pad, un-fftshift

        # px_scale = 0.2  # angstrom/px
        output_px = np.squeeze(self.output_size)  # px
        view_size = px_scale * output_px  # angstrom
        print(f"{view_size=}")

        extent = np.diag(atoms.lattice_mat)[:2]
        print(f"{extent=}")
        cells = ((view_size // extent) + 1).astype(int)
        print(f"{cells=}")
        atoms = atoms.make_supercell_matrix(
            (cells[0], cells[1], 1)
        ).center_around_origin()
        extent = np.diag(atoms.lattice_mat)[:2]

        sampling = (px_scale, px_scale)
        print(f"{sampling=}")

        margin = int(np.ceil(5 / min(sampling)))  # int like 20
        shape_w_margin = tuple(output_px + 2 * margin)

        # Set up real-space grid (in angstroms?)
        x = np.fft.fftfreq(output_px[0]) * output_px[0] * px_scale
        y = np.fft.fftfreq(output_px[1]) * output_px[1] * px_scale
        r = np.sqrt(x[:, None] ** 2 + y[None] ** 2)

        # construct the probe profile centered at (0,0) on the periodic spatial grid
        x = np.linspace(0, 4 * self.lorentzian_width, self.nbins)
        profile = gaussian(x, self.gaussian_width) + self.intensity_ratio * lorentzian(
            x, self.lorentzian_width
        )

        profile /= profile.max()
        f = interp1d(x, profile, fill_value=0, bounds_error=False)
        intensity = f(r)

        # shift the probe profile to the center
        # apply zero-padding, and shift back to the origin
        intensity = np.fft.fftshift(intensity)
        intensity = np.pad(intensity, (margin, margin))
        intensity = np.fft.fftshift(intensity)

        # project atomic coordinates onto the image
        # center them as well
        centroid = np.mean(atoms.cart_coords[:, :2], axis=0)
        print(f"{centroid=}")

        # center atom positions around (0,0)
        pos = atoms.cart_coords[:, :2] - centroid

        # shift to center of image
        pos += view_size / 2

        # select only atoms in field of view
        in_view = (
            (pos[:, 0] > -eps)
            & (pos[:, 0] < view_size[0] + eps)
            & (pos[:, 1] > -eps)
            & (pos[:, 1] < view_size[1] + eps)
        )
        # pos = pos[in_view]
        numbers = np.array(atoms.atomic_numbers)
        # numbers = numbers[in_view]

        atom_px = pos / px_scale  # AA / (AA/px) -> px

        # shift atomic positions to offset zero padding
        atom_px = atom_px + margin

        # initialize arrays with zero padding
        array = np.zeros((1,) + intensity.shape)  # adding extra 1
        mask = np.zeros((1,) + intensity.shape)
        for number in np.unique(np.array(atoms.atomic_numbers)):

            temp = np.zeros((1,) + shape_w_margin)
            temp = self.superpose_deltas(atom_px[numbers == number], temp)
            array += temp * number ** self.power_factor
            temp = np.where(temp > 0, number, temp)
            mask += temp[0]

        # FFT convolution of beam profile and atom position delta functions
        array = np.fft.ifft2(np.fft.fft2(array) * np.fft.fft2(intensity)).real

        scale = 1
        sel = slice(scale * margin, -scale * margin)
        # sel = slice(0, -1)
        array = array[0, sel, sel]

        mask = mask[0, sel, sel]
        atom_px = atom_px - scale * margin

        atom_px = pos[in_view] / px_scale
        numbers = numbers[in_view]

        return array, mask, atom_px, margin


"""
if __name__ == "__main__":
    from jarvis.core.atoms import crop_squre
    from jarvis.db.figshare import data, get_jid_data
    import matplotlib.pyplot as plt
    from jarvis.core.atoms import Atoms, ase_to_atoms, get_supercell_dims

    plt.switch_backend("agg")

    a = Atoms.from_dict(get_jid_data("JVASP-667")["atoms"])
    c = crop_square(a)
    # c = a.make_supercell_matrix([2, 2, 1])
    p = STEMConv(atoms=c).simulate_surface()
    plt.imshow(p, interpolation="gaussian", cmap="plasma")
    plt.savefig("stem_example.png")
    plt.close()
"""
