from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from physics_util import Spectra, get_thermals, Material, DEFAULT_TEMPERATURE, DensityOfStates
from diff_rixs import RIXSModel

def get_ax(cell):
    return plt.subplot(cell)

if __name__ == '__main__':
    data_path = Path("data") / "testing"
    material = Material.Fe2O3()
    outer_spacing = 0.35
    inner_spacing = 0.05
    xfels = Spectra.load(data_path / "xfel_spectra.pkl").pick_subsets((4,))
    left_outer_gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2], wspace=outer_spacing)
    left_col_gs = gridspec.GridSpecFromSubplotSpec(2, 1, hspace=outer_spacing, height_ratios=[1, 2], subplot_spec=left_outer_gs[0])
    left_inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, hspace=inner_spacing, subplot_spec=left_col_gs[1])
    top_outer_gs = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[1, 2], hspace=outer_spacing, subplot_spec=left_outer_gs[1])
    top_inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, wspace=inner_spacing, subplot_spec=top_outer_gs[0])
    inner_gs = gridspec.GridSpecFromSubplotSpec(2, 2, wspace=inner_spacing, hspace=inner_spacing, subplot_spec=top_outer_gs[1])
    # axs[0, 0].axis("off")
    get_ax(left_col_gs[0]).axis("off")
    for i, cell in enumerate(left_inner_gs):
        plt.sca(get_ax(cell))
        plt.plot(xfels.energies / 100, xfels.signal()[i-1] * 1000, "b")
        plt.ylabel(r"$\Phi$ (a.u.)")
    plt.xlabel(r"$\omega_1$ (a.u.)")

    get_ax(left_inner_gs[1]).sharex(get_ax(left_inner_gs[0]))
    plt.setp(get_ax(left_inner_gs[0]).get_xticklabels(), visible=False)


    for idx, dos in enumerate((data_path / "density_of_states").iterdir()):
        if idx > 1:
            break
        base_density = DensityOfStates.load(dos).vacant(
            DEFAULT_TEMPERATURE,
            material.density_per_unit_cell
        )
        rixs = (Spectra.load(data_path / "rixs_spectra" / f"rixs_{idx}.pkl")
                .pick_subsets((4,)))
        noise_rixs = rixs.noisy(0.3)
        plt.sca(get_ax(top_inner_gs[idx]))
        plt.plot(base_density.energies / 100, base_density.density, "b")
        plt.xlabel(r"$\Delta$ (a.u.)")
        if idx == 0:
            plt.ylabel(r"$\rho$ (a.u.)")
        for j in range(2):
            ax = get_ax(inner_gs[idx, j])
            plt.sca(ax)
            if j == 0:
                plt.ylabel("I (a.u.)")
            if idx == 1:
                plt.xlabel(r"$\omega_2$ (a.u.)")
            plt.plot(rixs.energies / 100, rixs.signal()[j], "k")
            plt.plot(noise_rixs.energies / 100, noise_rixs.signal()[j], "r", alpha=0.2)

    get_ax(top_inner_gs[1]).sharey(get_ax(top_inner_gs[0]))
    plt.setp(get_ax(top_inner_gs[1]).get_yticklabels(), visible=False)
    get_ax(inner_gs[1, 0]).sharex(get_ax(inner_gs[0, 0]))
    plt.setp(get_ax(inner_gs[0, 0]).get_xticklabels(), visible=False)
    get_ax(inner_gs[1, 1]).sharex(get_ax(inner_gs[0, 1]))
    plt.setp(get_ax(inner_gs[0, 1]).get_xticklabels(), visible=False)
    get_ax(inner_gs[0, 1]).sharey(get_ax(inner_gs[0, 0]))
    plt.setp(get_ax(inner_gs[0, 1]).get_yticklabels(), visible=False)
    get_ax(inner_gs[1, 1]).sharey(get_ax(inner_gs[1, 0]))
    plt.setp(get_ax(inner_gs[1, 1]).get_yticklabels(), visible=False)
    plt.subplots_adjust(0.1, 0.11, 0.99, 0.94)
    plt.savefig("figure_3.png", dpi=300)
