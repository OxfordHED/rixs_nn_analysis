from pathlib import Path
import time as t
import json

import click
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

from end_to_end import EndToEndDataset, split_data, JointModel

from physics_util import DEFAULT_TEMPERATURE, Material

@click.command()
@click.option("--subsets", "-s", default=(4,), multiple=True, help="Subsets of XFEL pulses to use ((4,)).")
@click.option("--subset-truncate", "-st", default=None, multiple=True, help="Truncate subsets at index (None).")
@click.option("--chem-potential", "-cp", default=None, type=float, help="Override chemical potential (None).")
@click.option("--loss", "-l", default="MSE", type=click.Choice(["MSE", "MAE"]), help="Loss function (MSE).")
@click.option("--batch-size", "-bs", default=100, help="Number of samples per batch (100).")
@click.argument("model", type=click.Path(exists=True))
@click.argument("dataset", type=click.Path(exists=True))
def test(**config):
    """Testing the trained end-to-end MODEL on DATASET.

    MODEL: path, model file, e.g. models/noise00/end_to_end.pt.

    DATASET: path, testing dataset, e.g. data/testing.
    """
    with torch.no_grad():
        results = {}
        test_loss = 0
        test_labels = []
        test_out = []

        out_path = Path("runs") / ("end_to_end_test_" + str(int(t.time())))
        out_path.mkdir(exist_ok=True)

        test_dos, _ = split_data(Path(config["dataset"]), 0.)

        test_ds = EndToEndDataset(
            Path(config["dataset"]),
            test_dos,
            material=Material.Fe2O3(),
            temperature=DEFAULT_TEMPERATURE,
            subset_filter_range=config["subset_truncate"],
            pick_subsets=config["subsets"],
            out_noise=0.,
            chemical_potential=None if config["chem_potential"] is None else torch.tensor(config["chem_potential"]),
        )
        test_ds.plot(save_to=out_path / "test_ds.png")

        test_loaded = test_ds.load(config["batch_size"])

        model = JointModel.load(Path(config["model"]))

        loss_fn = nn.MSELoss() if config["loss"] == "MSE" else nn.L1Loss()

        # For saving data
        for feat, label in test_loaded:
            out = model(feat)
            label *= test_ds.max_dos
            out *= test_ds.max_dos
            test_labels.append(label.tolist())
            test_out.append(out.tolist())

        # For loss evaluation
        for feat, label in test_loaded:
            out = model(feat)
            test_loss += loss_fn(
                out * test_ds.max_dos, label * test_ds.max_dos
            ).item()

        test_loss /= len(test_loaded)

        print(f"Test loss: {test_loss:.2e}")

        # Write to results file
        results["test_loss"] = test_loss
        results["test_labels"] = test_labels
        results["test_pred"] = test_out

        # plot labels against predictions
        test_labels = np.concatenate([np.array(label) for label in test_labels], axis=0)
        test_out = np.concatenate([np.array(out) for out in test_out], axis=0)

        unique_labels = np.unique(test_labels, axis=0)
        label_indices = [
            np.where((test_labels == label).all(axis=1))[0] for label in unique_labels
        ]

        _, axes = plt.subplots(
            1, len(unique_labels), figsize=(len(unique_labels) * 4, 4)
        )

        mean_preds = []
        std_preds = []
        for i, label in enumerate(unique_labels):
            index = label_indices[i]
            axis = axes[i]
            outputs = test_out[index, :]
            pred_mean = np.mean(outputs, axis=0)
            mean_preds.append(pred_mean)
            pred_std = np.std(outputs, axis=0)
            std_preds.append(pred_std)
            axis.fill_between(
                range(outputs.shape[1]),
                pred_mean + pred_std,
                pred_mean - pred_std,
                alpha=0.5,
                color="magenta",
            )
            axis.plot(np.mean(outputs, axis=0), "b", label="CNN Prediction")
            axis.plot(label, "k", label="Label")
            axis.legend(loc="upper left")
        plt.savefig(out_path / "predictions.png")

        results["unique_labels"] = unique_labels.tolist()
        results["mean_preds"] = np.stack(mean_preds).tolist()
        results["std_preds"] = np.stack(std_preds).tolist()

        with (out_path / "results.json").open("w", encoding="utf-8") as out_file:
            json.dump(results, out_file, indent=2)


if __name__ == "__main__":
    test()
