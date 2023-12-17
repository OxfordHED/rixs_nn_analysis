from pathlib import Path
import copy
import time
import json

import click

import torch
from torch import nn, optim

import matplotlib.pyplot as plt


from diff_rixs import RIXSModel

from physics_util import (
    DensityOfStates,
    NeuralDoS,
    DEFAULT_TEMPERATURE,
    Material,
    Spectra,
    get_thermals
)

@click.command(help="Generate RIXS data at different temperatures")
@click.option("--chem-pot", "-cp", default="simple", type=click.Choice(["simple", "full"]))
@click.argument("data_path", type=click.Path(exists=True))
def main(chem_pot, data_path):
    config = {"subsets": (4,)}
    data_path = Path(data_path)
    material = Material.Fe2O3()
    base_density = DensityOfStates.load(data_path / "dos.pkl")

    xfels = Spectra.load(data_path / "xfel_spectra.pkl").pick_subsets(config["subsets"])

    for t in range(4, 11):
        rixs = Spectra.load(data_path / "rixs_spectra" / f"rixs_temp_{t}.pkl")

        thermals = get_thermals(
            base_density,
            torch.tensor(t).float(),
            material.density_per_unit_cell,
            thermals_type=("approximate" if chem_pot == "simple" else "exact")
        )

        model = RIXSModel(
            dos_function=base_density.function,
            dos_energies=base_density.energies,
            material=material,
            thermodynamic_props=thermals,
            rixs_energies=rixs.energies,
            xfel_energies=xfels.energies,
        )

        out = model(xfels.signal())

        out_rixs = Spectra(
            energies=rixs.energies,
            signal=out.detach()
        )

        out_rixs.save(data_path / "rixs_spectra" / f"rixs_temp_{t}.pkl")

        plt.plot(rixs.energies, rixs.signal()[0, :], label="original")
        plt.plot(out_rixs.energies, out_rixs.signal()[0, :], label="new")
        plt.legend()
        plt.show()

if __name__ == '__main__':
    main()
