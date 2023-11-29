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
    ThermodynamicalProperties
)

@click.command(help="Generate RIXS data at different temperatures")
def main():
    config = {"subsets": (4,)}
    data_path = Path("data") / "temperature"
    material = Material.Fe2O3()
    base_density = DensityOfStates.load(data_path / "dos_0.pkl")

    xfels = Spectra.load(data_path / "xfel_spectra.pkl").pick_subsets(config["subsets"])
    rixs = (
        Spectra.load(data_path / "rixs_0.pkl")
        .pick_subsets(config["subsets"])
    )

    for t in range(10, 11):
        thermals = ThermodynamicalProperties.from_dos(
            dos=base_density,
            temperature=torch.tensor(t),
            electron_density=material.density_per_unit_cell,
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
            signal=out
        )

        out_rixs.save(data_path / f"rixs_temp_{t}.pkl")

    plt.plot(model.dos_energies, base_density.function(model.dos_energies))
    plt.show()

    plt.plot(rixs.energies, rixs.signal()[0, :], label="original")
    plt.plot(out_rixs.energies, out_rixs.signal()[0, :], label="new")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
