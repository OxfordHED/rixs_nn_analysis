"""
Module containing the training endpoint for the end-to-end model.
"""
import time as t
from pathlib import Path
import json

import click
from torch import nn, optim
import matplotlib.pyplot as plt

from end_to_end import EndToEndDataset, JointModel, split_data
from physics_util import Material, DEFAULT_TEMPERATURE

@click.command()
@click.option("--subsets", "-s", default=(4,), multiple=True, help="Subsets of XFEL pulses to use ((4,)).")
@click.option("--subset-truncate", "-st", default=None, multiple=True, help="Truncate subsets at index (None).")
@click.option("--chem-potential", "-cp", default=0., help="Override chemical potential (0.).")
@click.option("--noise", "-n", default=0., help="Noise level (0.).")
@click.option("--validation-split", "-vs", default=0.25, help="Fraction of data for validation (0.2).")
@click.option("--batch-size", "-bs", default=100, help="Number of samples per batch (100).")
@click.option("--conv-layers", "-cl", default=4, help="Number of convolutional layers (4).")
@click.option("--conv-channels", "-cc", default=8, help="Number of channels per conv. layer (8).")
@click.option("--fcn-shape", "-fs", default=(200, 200, 100, 100), multiple=True, help="FCN architecture ((200, 200, 100, 100)).")
@click.option("--epochs", "-e", default=5_000, help="Number of epochs (5_000).")
@click.option("--learning-rate", "-lr", default=1e-4, help="Learning rate (1e-4).")
@click.option("--loss", "-l", default="MSE", type=click.Choice(["MSE", "MAE"]), help="Loss function (MSE).")
@click.option("--regularization", "-reg", default=0., help="Level of L2 reg. (0.).")
@click.argument("dataset", type=click.Path(exists=True))
def train(**config: dict[str, object]) -> None:
    """Training the end-to-end estimator on DATASET.

    DATASET: path, root directory of the training data, e.g. data/trainig.
    """
    # Generate a record of the run
    out_path = Path("runs") / ("end_to_end_train_" + str(int(t.time())))
    out_path.mkdir(exist_ok=True)

    # Set up data and save plots of the datasets
    train_dos, val_dos = split_data(Path(config["dataset"]), config["validation_split"])
    print(f"# Training DoS: {len(train_dos)}, # Validation DoS: {len(val_dos)}, Overlapping: {any(v in train_dos for v in val_dos)}")
    print(f"# Training DoS w/ sqrt: {len([t for t in train_dos if t < 93])}")

    config["train_dos"] = train_dos
    config["val_dos"] = val_dos

    # Save the configuration for later reference
    with (out_path / "config.json").open("w", encoding="utf-8") as out_file:
        json.dump(config, out_file, indent=2)

    train_ds = EndToEndDataset(
        Path(config["dataset"]),
        train_dos,
        material=Material.Fe2O3(),
        temperature=DEFAULT_TEMPERATURE,
        subset_filter_range=config["subset_truncate"],
        pick_subsets=config["subsets"],
        out_noise=config["noise"],
        chemical_potential=config["chem_potential"],
    )
    train_ds.plot(save_to=out_path / "train_ds.png")

    train_loaded = train_ds.load(config["batch_size"])

    val_ds = EndToEndDataset(
        Path(config["dataset"]),
        val_dos,
        material=Material.Fe2O3(),
        temperature=DEFAULT_TEMPERATURE,
        subset_filter_range=config["subset_truncate"],
        pick_subsets=config["subsets"],
        out_noise=config["noise"],
        chemical_potential=config["chem_potential"],
    )
    val_ds.plot(save_to=out_path / "val_ds.png")

    val_loaded = val_ds.load(config["batch_size"])

    model = JointModel.setup(dict(
        **config,
        feat_count=train_ds.feat_count,
        label_count=train_ds.label_count
    ))

    loss_fn = nn.MSELoss() if config["loss"] == "MSE" else nn.L1Loss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["regularization"],
    )

    results = {
        "loss": [],
        "val_loss": [],
    }

    start = t.time()
    for epoch in range(config["epochs"]):
        last = t.time()
        epoch_loss = 0
        non_normed_loss = 0

        # To actually do training
        for feat, label in train_loaded:
            optimizer.zero_grad()
            out = model(feat)
            non_normed_loss += loss_fn(
                out * train_ds.max_dos, label * train_ds.max_dos
            ).item()
            loss = loss_fn(out, label)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        val_loss = 0
        non_normed_val_loss = 0

        # To evaluate performance
        for feat, label in val_loaded:
            out = model(feat)
            loss = loss_fn(out, label)
            non_normed_val_loss += loss_fn(
                out * val_ds.max_dos, label * val_ds.max_dos
            ).item()
            val_loss += loss.item()

        epoch_loss /= len(train_loaded)
        val_loss /= len(val_loaded)

        # keep track of these to record data
        non_normed_loss = non_normed_loss / len(train_loaded)
        non_normed_val_loss = non_normed_val_loss / len(val_loaded)

        results["loss"].append(non_normed_loss)
        results["val_loss"].append(non_normed_val_loss)
        print(f"Epoch: {epoch}, Loss: {non_normed_loss:.2e}, Val. Loss: {non_normed_val_loss:.2e}, t/it: {t.time() - last:.2f}")

    end = t.time()
    print(f"Finished training in {round(end - start, 2)}s!")

    # save model
    model.save(out_path / "model.pt")

    # save history
    with (out_path / "history.json").open("w", encoding="utf-8") as out_file:
        json.dump(results, out_file, indent=2)

    # Plot history
    plt.figure(figsize=(6, 6))
    plt.plot(results["loss"], "b", label="Training Loss")
    plt.plot(results["val_loss"], "magenta", label="Validation Loss")
    plt.xlabel("# Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(out_path / "history.png")

if __name__ == '__main__':
    train()