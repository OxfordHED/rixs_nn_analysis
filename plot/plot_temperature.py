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

def get_estimates(run_path: Path, noise_level_str: str) -> tuple[np.ndarray, np.ndarray]:
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
                    temp_estimates.append((iter_data["best_temp"]))#  - temp))  #  / temp)
            noise_estimates.append(temp_estimates)
        all_estimates.append(noise_estimates)

    return all_estimates

@click.command()
def main():
    run_path = Path("runs")

    x_axis, correct_est = get_estimates(run_path, "low_lr_temp_n_0.")
    _, incorrect_est = get_estimates(run_path, "alt_temp_n_0.")


    plt.plot(
        x_axis + 3.5,
        np.array(correct_est).flatten(),  # np.sort(np.array(correct_est), axis=1).flatten(),
        "^",
        color="darkturquoise",
        markeredgecolor="black",
        label="Correct model"
    )
    plt.plot(
        x_axis - 3.5,
        np.array(incorrect_est).flatten(),  # np.sort(, axis=1).flatten(),
        "^",
        color="sandybrown",
        markeredgecolor="black",
        label="Incorrect model"
    )
    plt.boxplot(
        np.array(correct_est).T,
        positions=[i*10 + 3.5 for i in range(4)],
        widths=3,
        patch_artist=True,
        **formatters("darkturquoise", .3)
    )
    plt.boxplot(
        np.array(incorrect_est).T,
        positions=[i*10 - 3.5 for i in range(4)],
        widths=3,
        patch_artist=True,
        **formatters("sandybrown", .3)
    )
    plt.xlabel(r"Noise ($\varepsilon$)")
    plt.ylabel(r"MAE $\mathcal{L}(T, \tilde{T})$")
    plt.xticks(range(0, 40, 10), labels=[0, 0.1, 0.2, 0.3])
    plt.legend()
    # plt.yscale("log")
    plt.show()


if __name__ == "__main__":
    main()
