from pathlib import Path
import copy
import time
import json

import click

import torch
from torch import nn, optim

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
@click.option("--nn-width", "-w", default=40, help="Number of neurons per layer (40).")
@click.option("--nn-depth", "-d", default=4, help="Number of layers (4).")
@click.option("--batch-size", "-bs", default=8, help="Number of samples per batch (8).")
@click.option("--epochs", "-e", default=10_000, help="Number of epochs (10_000).")
@click.option("--learning-rate", "-lr", default=1e-3, help="Learning rate (1e-3).")
@click.option("--loss", "-l", default="MSE", type=click.Choice(["MSE", "MAE"]), help="Loss function (MSE).")
@click.option("--regularization", "-reg", default=0., help="Level of L2 reg. (0.).")
@click.argument("dataset", type=click.Path(exists=True))
def train(**config):
    data_path = Path(config["dataset"])
    out_path = Path("runs") / ("step_train_" + str(int(time.time())))
    out_path.mkdir(exist_ok=True)

    # Save the configuration for later reference
    with (out_path / "config.json").open("w", encoding="utf-8") as out_file:
        json.dump(config, out_file, indent=2)

    material = Material.Fe2O3()

    for dos in (data_path / "density_of_states").iterdir():
        print(f"Fitting on dos {dos}...")
        idx = str(dos).split(".")[0].split("_")[-1]
        base_density = DensityOfStates.load(dos).vacant(
            DEFAULT_TEMPERATURE,
            material.density_per_unit_cell
        )
        xfels = Spectra.load(data_path / "xfel_spectra.pkl").pick_subsets(config["subsets"])
        rixs = (
            Spectra.load(data_path / "rixs_spectra" / f"rixs_{idx}.pkl")
            .pick_subsets(config["subsets"])
            .noisy(config["noise"])
        )

        energies = base_density.energies

        neural_dos = NeuralDoS.create(energies)

        # Won't be used for vacant fit but needs to be passed
        thermals = ThermodynamicalProperties.from_dos(
            dos=base_density,
            temperature=DEFAULT_TEMPERATURE,
            electron_density=material.density_per_unit_cell,
        )

        model = RIXSModel(
            dos_function=neural_dos.function,
            dos_energies=neural_dos.energies,
            material=material,
            thermodynamic_props=thermals,
            rixs_energies=rixs.energies,
            xfel_energies=xfels.energies,
            vacant=True,
        )

        loss_fn = nn.MSELoss() if config["loss"] == "MSE" else nn.L1Loss()
        optimizer = optim.Adam(
            neural_dos.estimator.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["regularization"]
        )

        samples_per_batch = config["batch_size"]
        n_batches = len(xfels) // samples_per_batch

        best_loss = 1e6
        best_state = None
        best_epoch = 0

        history = {"loss": []}

        log_interval = 10
        start = time.time()
        for epoch in range(config["epochs"]):
            epoch_loss = 0
            perm = torch.randperm(len(xfels))
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
                    best_state = copy.deepcopy(neural_dos.estimator.state_dict())
                    best_epoch = epoch

            history["loss"].append(float(epoch_loss))
            if epoch % log_interval == 0 and epoch > 0:
                avg = (time.time() - start) / log_interval
                print(f"Epoch {epoch}, t/it: {avg:.2f}s, best: {best_loss:.2e}")
                start = time.time()

        print(f"Best loss: {best_loss:.2e} at epoch {best_epoch}")
        best_model = NeuralDoS.create(neural_dos.energies)
        best_model.estimator.load_state_dict(best_state)

        history["best_loss"] = float(best_loss)
        history["best_epoch"] = best_epoch
        with (out_path / f"history_{idx}.json").open("w", encoding="utf-8") as hist_file:
            json.dump(history, hist_file)
        best_model.save(out_path / "neural_dos_" + idx + ".pt")


if __name__ == '__main__':
    train()
