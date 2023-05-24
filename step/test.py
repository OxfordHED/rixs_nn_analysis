import click

import torch
from torch import nn

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("TkAgg")

from physics_util import DensityOfStates, NeuralDoS, DEFAULT_TEMPERATURE, Material

@click.command(help="Test the STEP estimator against the original DoS.")
@click.option("--loss", "-l", default="MSE", type=click.Choice(["MSE", "MAE"]), help="Loss function (MSE).")
@click.argument("model", type=click.Path(exists=True))
@click.argument("target", type=click.Path(exists=True))
def test(model, target, loss):

    material = Material.Fe2O3()

    base_dos = DensityOfStates.load(target)
    neural_dos = NeuralDoS.load(model).vacant(DEFAULT_TEMPERATURE, material.density_per_unit_cell, base_dos=base_dos)
    base_dos = base_dos.vacant(DEFAULT_TEMPERATURE, material.density_per_unit_cell)


    loss_fn = nn.MSELoss() if loss == "MSE" else nn.L1Loss()

    assert torch.allclose(base_dos.energies, neural_dos.energies), "Can only compare two DoS on the same energy axis."

    with torch.no_grad():
        loss = loss_fn(base_dos.density, neural_dos.function(neural_dos.energies[:, None])[:, 0])
    print(loss)

    plt.plot(base_dos.energies, base_dos.density)
    plt.plot(neural_dos.energies, neural_dos.function(neural_dos.energies[:, None])[:, 0].detach())
    plt.show()

if __name__ == '__main__':
    test()
