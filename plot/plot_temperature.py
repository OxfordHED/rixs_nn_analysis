import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import click

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
def main():
    run_path = Path("runs")

    noise_level_str = "temp_n_0."
    all_estimates = []

    for noise in range(4):
        noise_estimates = []
        noise_path = run_path / (noise_level_str + str(noise))
        assert noise_path.exists() and noise_path.is_dir()
        for temp in range(4, 11):
            temp_estimates = []
            temp_hist = f"history_{temp}.0_"
            for j in range(5):
                iter_hist = temp_hist + str(j) + ".json"
                with (noise_path / iter_hist).open("r", encoding="utf-8") as infile:
                    iter_data = json.load(infile)
                    temp_estimates.append((iter_data["best_temp"] - temp) / temp)
            noise_estimates.append(temp_estimates)
        all_estimates.append(noise_estimates)

    reduced_estimates = [
        [np.mean(np.abs(temp_el)) for temp_el in noise_el]
        for noise_el in all_estimates
    ]
    x_axis = np.array([[i * 10 + j - 3 for j in range(7)] for i in range(4)]).flatten()
    plt.plot(
        x_axis,
        np.sort(np.array(reduced_estimates), axis=1).flatten(),
        "^",
        color="darkturquoise",
        markeredgecolor="black"
    )
    plt.boxplot(
        np.array(reduced_estimates).T,
        positions=[i*10 for i in range(4)],
        widths=7,
        patch_artist=True,
        **formatters("darkturquoise", .3)
    )
    plt.xlabel(r"Noise ($\varepsilon$)")
    plt.ylabel(r"MARE $\mathcal{L}(T, \tilde{T})$")
    plt.show()


if __name__ == "__main__":
    main()
