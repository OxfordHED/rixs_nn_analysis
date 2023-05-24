from pathlib import Path
import json

import click

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("TkAgg")

COLORS = ["black", "darkturquoise", "sandybrown"]

@click.command()
def figure_5():
    """Plot figure 5 from the main text."""
    data_path = Path("plot") / "figure_5_data.json"

    with data_path.open("r", encoding="utf-8") as infile:
        data = json.load(infile)

    plt.rcParams.update({"font.size": 22})

    _, axes = plt.subplots(1, len(data), figsize=(6*len(data), 6))

    for i, dos in enumerate(data):

        for key, val in dos.items():
            dos[key] = np.array(val)

        axes[i].plot(
            dos["energies"], dos["label"], label="Target", color=COLORS[0], alpha=0.75, linestyle="-"
        )
        axes[i].plot(
            dos["energies"], dos["mean_step"], color=COLORS[1], label="STEP"
        )
        axes[i].fill_between(
            dos["energies"],
            dos["mean_step"] + dos["std_step"],
            dos["mean_step"] - dos["std_step"],
            alpha=0.2, color=COLORS[1]
        )

        axes[i].fill_between(
            dos["energies"],
            dos["mean_ete"] + dos["std_ete"],
            dos["mean_ete"] - dos["std_ete"],
            alpha=0.2, color=COLORS[2]
        )
        axes[i].plot(dos["energies"], dos["mean_ete"], label="CNN", color=COLORS[2])
        axes[i].text(
            -45 if i == 1 else 0,
            7 if i == 1 else 14,
            "(a)" if i == 0 else "(b)",
            fontweight="bold",
            fontsize=24
        )
        axes[i].legend(loc="upper right")
        axes[i].set_xlabel(r"$\Delta$ (eV)")
        axes[i].set_xlim(-50 if i == 1 else -5, 140)
        axes[i].set_ylim(0, 8 if i == 1 else 16)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    figure_5()