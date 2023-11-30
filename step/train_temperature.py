from pathlib import Path
import copy
import time
import json

import click

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim

from torchviz import make_dot

from diff_rixs import RIXSModel

from physics_util import (
    DensityOfStates,
    NeuralDoS,
    DEFAULT_TEMPERATURE,
    Material,
    Spectra,
    ThermodynamicalProperties
)

@click.command(help="Train the STEP estimator on a DoS.")
@click.option("--subsets", "-s", default=(4,), multiple=True, help="Subsets of XFEL pulses to use ((4,)).")
@click.option("--noise", "-n", default=0., help="Noise level (0.).")
@click.option("--batch-size", "-bs", default=8, help="Number of samples per batch (8).")
@click.option("--epochs", "-e", default=10_000, help="Number of epochs (10_000).")
@click.option("--learning-rate", "-lr", default=1e-3, help="Learning rate (1e-3).")
@click.option("--loss", "-l", default="MSE", type=click.Choice(["MSE", "MAE"]), help="Loss function (MSE).")
@click.option("--regularization", "-reg", default=0., help="Level of L2 reg. (0.).")
@click.argument("dataset", type=click.Path(exists=True))
def train(**config):
    data_path = Path(config["dataset"])
    out_path = Path("runs") / ("step_train_temperature" + str(int(time.time())))
    out_path.mkdir(exist_ok=True)

    # Save the configuration for later reference
    with (out_path / "config.json").open("w", encoding="utf-8") as out_file:
        json.dump(config, out_file, indent=2)

    material = Material.Fe2O3()

    true_temps = torch.tensor(np.loadtxt(data_path / "temperatures.csv").T).squeeze().float()

    xfels = Spectra.load(data_path / "xfel_spectra.pkl").pick_subsets(config["subsets"])
    for t in true_temps:
        print(f"Fitting temperature: {t} eV")

        fname = f"rixs_temp_{int(t)}.pkl"
        rixs = (
            Spectra.load(data_path / "rixs_spectra" / fname)
            .noisy(config["noise"])
        )

        base_density = DensityOfStates.load(data_path / "dos.pkl")

        thermals = ThermodynamicalProperties.from_fermi_energy(
            dos=base_density,
            temperature=2 + torch.rand(1)*8,
            electron_density=material.density_per_unit_cell,
        )

        model = RIXSModel(
            dos_function=base_density.function,
            dos_energies=base_density.energies,
            material=material,
            thermodynamic_props=thermals,
            rixs_energies=rixs.energies,
            xfel_energies=xfels.energies,
            thermal_backprop=True,
            fit_temp=True
        )
        loss_fn = nn.MSELoss() if config["loss"] == "MSE" else nn.L1Loss()
        optimizer = optim.Adam(
            [thermals.temperature],
            lr=config["learning_rate"],
            weight_decay=config["regularization"]
        )

        samples_per_batch = config["batch_size"]
        n_batches = len(xfels) // samples_per_batch

        best_loss = 1e6
        best_temp = None
        best_epoch = 0

        history = {"loss": []}

        log_interval = 10
        start = time.time()
        for epoch in range(config["epochs"]):
            epoch_loss = 0
            perm = torch.arange(len(xfels)).int()  # torch.randperm(len(xfels))
            for i in range(n_batches):
                optimizer.zero_grad()
                out = model(
                    xfels.signal()[perm][
                    i * samples_per_batch: (i + 1) * samples_per_batch
                    ]
                )
                loss = loss_fn(
                    out,
                    rixs.signal()[perm][
                    i * samples_per_batch: (i + 1) * samples_per_batch, :],
                )
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if epoch_loss < best_loss:
                with torch.no_grad():
                    best_loss = epoch_loss
                    best_epoch = epoch
                    best_temp = thermals.temperature.detach().numpy()[0]

            history["loss"].append(float(epoch_loss))
            if epoch % log_interval == 0 and epoch > 0:
                avg = (time.time() - start) / log_interval
                print(
                    f"Epoch {epoch}, t/it: {avg:.2f}s, current: {epoch_loss:.2e}, best: {best_loss:.2e}, current T: {thermals.temperature[0]:.2e}eV, best T: {best_temp:.2e}eV"
                )
                start = time.time()

        print(f"Best loss: {best_loss:.2e} at epoch {best_epoch}")

        history["best_loss"] = float(best_loss)
        history["best_epoch"] = best_epoch
        history["best_temp"] = float(best_temp)
        with (out_path / f"history_{t}.json").open("w", encoding="utf-8") as hist_file:
            json.dump(history, hist_file)

if __name__ == "__main__":
    train()
