from pathlib import Path
import json

import click

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("TkAgg")

ALPHA = 0.3
NOISE_LEVELS = [0, 1, 2, 3]
LEGEND_LABELS = ["CNN", "STEP"]
COLORS = ["sandybrown", "darkturquoise"]


def formatters(color: str, alpha: float) -> dict[str, dict]:
    """Returns formatting dict for boxplot with a given color.

    Args:
        color: str, name of the color for the boxplot

    Returns:
        dict containing the formatting dicts for a boxplot with the given color.
    """
    return {
        "boxprops": {"edgecolor": "black", "facecolor": color, "alpha": alpha},
        "whiskerprops": {"alpha": alpha},
        "capprops": {"alpha": alpha},
        "medianprops": {"color": "black", "alpha": alpha},
        "flierprops": {
            "marker": "o",
            "markerfacecolor": color,
            "alpha": alpha,
            "markeredgecolor": "black",
        },
    }



@click.command()
def figure_6():
    """Plot figure 6 from the main text."""
    data_path = Path("plot") / "figure_6_data.json"

    with data_path.open("r", encoding="utf-8") as infile:
        data = json.load(infile)

    ete_losses = np.array(data["ete"])  # end-to-end losses
    step_losses = np.array(data["step"]) # step losses
    plt.rcParams.update({"font.size": 16})
    plt.figure(figsize=(10, 4))
    plt.boxplot(
        ete_losses.T,
        positions=[i - 1.5 for i in range(0, step_losses.shape[0] * 10, 10)],
        widths=2.5,
        patch_artist=True,
        **formatters(COLORS[0], ALPHA),
    )
    plt.boxplot(
        step_losses.T,
        positions=[i + 1.5 for i in range(0, step_losses.shape[0] * 10, 10)],
        widths=2.5,
        patch_artist=True,
        **formatters(COLORS[1], ALPHA),
    )

    for i in range(step_losses.shape[0]):
        (p1,) = plt.plot(
            [10 * (i - 0.25) + j / 2.2 for j in range(ete_losses.shape[1])],
            sorted(ete_losses[i]),
            "^",
            color=COLORS[0],
            markeredgecolor="black",
            markersize=8,
        )
        (p2,) = plt.plot(
            [10 * i + j / 2 for j in range(step_losses.shape[1])],
            sorted(step_losses[i]),
            "^",
            color=COLORS[1],
            markeredgecolor="black",
            markersize=8,
        )

    # Add labels and title
    plt.xlabel(r"Noise ($\epsilon$)")
    plt.xlim(-5, 40)
    plt.xticks(range(0, 40, 10), labels=[0, 0.1, 0.2, 0.3])
    plt.ylabel(r"MSE $\mathcal{L}(\rho_{eff}, \tilde{\rho}_{eff})$")
    plt.yscale("log")
    ax = plt.gca()
    ax.xaxis.set_label_coords(0.45, -0.1)
    legend_handles = [p1, p2]
    plt.legend(legend_handles, LEGEND_LABELS, loc="lower right")

    plt.tight_layout()
    # Show the plot
    plt.show()

if __name__ == '__main__':
    figure_6()