from pathlib import Path
import matplotlib.pyplot as plt
from physics_util import Spectra, get_thermals, Material, DEFAULT_TEMPERATURE, DensityOfStates

if __name__ == "__main__":
    data_path = Path("data") / "testing"
    material = Material.Fe2O3()
    xfels = Spectra.load(data_path / "xfel_spectra.pkl").pick_subsets((4,))
    rixs = (Spectra.load(data_path / "rixs_spectra" / f"rixs_0.pkl")
            .pick_subsets((4,)).noisy(0.15))
    base_density = DensityOfStates.load(data_path / "density_of_states" / "dos_0.pkl").vacant(
        DEFAULT_TEMPERATURE,
        material.density_per_unit_cell
    )

    plt.rcParams["font.size"] = 25

    # fig = plt.figure(figsize=(4, 3))
    plt.xlabel(r"$\omega_1$ (a.u.)")
    plt.ylabel(r"$\Phi$ (a.u.)")
    plt.plot(xfels.energies / 100, xfels.signal()[0, :] * 1_000, "b")
    plt.tight_layout()
    plt.savefig("figure_4_0.png", dpi=300)
    plt.clf()

    # fig = plt.figure(figsize=(4, 3))
    plt.xlabel(r"$\omega_2$ (a.u.)")
    plt.ylabel("I (a.u.)")
    plt.plot(rixs.energies / 100, rixs.signal()[0, :], "b")
    plt.tight_layout()
    plt.savefig("figure_4_1.png", dpi=300)
    plt.clf()

    # fig = plt.figure(figsize=(4, 3))
    plt.xlabel(r"$\Delta$ (a.u.)")
    plt.ylabel(r"$\rho$ (a.u.)")
    plt.plot(base_density.energies / 100, base_density.density, "b")
    plt.tight_layout()
    plt.savefig("figure_4_2.png", dpi=300)
