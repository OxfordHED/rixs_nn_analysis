import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def get_estimates(run_path: Path, noise_level_str: str) -> np.ndarray:
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

    return np.array(all_estimates)

def get_mean_std(estimates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.mean(estimates, axis=2)
    stds = np.std(estimates, axis=2)

    return means, stds

if __name__ == "__main__":
    run_path = Path("runs")
    x = np.arange(4, 11, 1)
    alt_model = get_estimates(run_path, "alt_temp_n_0.")
    true_model = get_estimates(run_path, "low_lr_temp_n_0.")

    alt_means, alt_stds = get_mean_std(alt_model)
    true_means, true_stds = get_mean_std(true_model)

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

    for i in range(4):
        plt.sca(axes[i//2, i%2])
        plt.xlim(3, 11)
        plt.ylim(3, 11)
        plt.plot([3, 11], [3, 11], "k--")
        plt.title(f"Noise {i*10}%")
        if i // 2:
            plt.xlabel("T [eV]")
        if not i % 2:
            plt.ylabel(r"$\tilde{\rm T}$ [eV]")
        plt.errorbar(
            x,
            alt_means[i],
            yerr=alt_stds[i],
            marker="x",
            color="sandybrown",
            ls="",
            capsize=3,
            label=r"Approximate $\mu$"
        )
        plt.errorbar(
            x,
            true_means[i],
            yerr=true_stds[i],
            marker="o",
            color="darkturquoise",
            ls="",
            capsize=3,
            label=r"Exact $\mu$"
        )
        plt.text(10, 3.5, "(%s)" % "abcd"[i], fontweight=1000, fontsize=14)
        plt.legend()
    plt.tight_layout()
    plt.show()
