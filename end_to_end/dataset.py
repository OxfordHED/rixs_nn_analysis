from __future__ import annotations
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from physics_util import Spectra, DensityOfStates, Material


def split_data(path: Path, valid_split: float) -> tuple[list[int], list[int]]:
    try:
        dos_path = path / "density_of_states"
        indices = [int(str(dos).split(".")[0].split("_")[-1]) for dos in dos_path.iterdir()]
    except Exception as error:
        raise Exception("Couldn't load dataset") from error
    valid = random.sample(indices, int(len(indices) * valid_split))
    train = [ind for ind in indices if ind not in valid]
    return train, valid

class EndToEndDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        dos_indices: list[int],
        normalize: bool = True,
        log_rixs: bool = False,
        vacant: bool = True,  # material, temperature and chem pot only matter if vacant is True
        material: list[Material] | Material | None = None,
        temperature: list[torch.Tensor] | torch.Tensor | None = None,
        # Used to override calculated chem pot
        chemical_potential: torch.Tensor | None = None,
        subset_filter_range: tuple | None = None,
        pick_subsets: tuple | None = None,
        out_noise: float = 0,
        in_noise: float = 0,
    ) -> None:
        # Assertions to ensure all entered data matches if vacant DoS taken
        if vacant:
            as_list = False
            assert material is not None and temperature is not None
            if not isinstance(material, Material):
                assert len(material) == len(temperature)
                as_list = True

        base_path = Path(path)

        self._dos_list = dos_indices
        self.in_noise = in_noise
        self.out_noise = out_noise

        # Load XFEL spectra, optionally filter them and apply noise
        self._xfels = Spectra.load(base_path / "xfel_spectra.pkl")
        self._xfel_energies = self._xfels.energies
        if pick_subsets:
            self._xfels = self._xfels.pick_subsets(pick_subsets)
        if subset_filter_range:
            self._xfels = self._xfels.filter_on_subsets(*subset_filter_range)
        self._xfels = self._xfels.noisy(in_noise)

        self._dos = []
        self._rixs = []

        # For every DoS, load DoS and RIXS
        for i in dos_indices:
            # No data_processing of DoS other than potentially using the vacant DoS
            dos = DensityOfStates.load(base_path / "density_of_states" / f"dos_{i}.pkl")
            self._dos_energies = dos.energies

            # Usually set this to true, as the vacant DoS is what really affects the Spectra
            if vacant:
                if as_list:
                    dos = dos.vacant(
                        temperature=temperature[i],
                        density_per_unit_cell=material[i].density_per_unit_cell,
                        chemical_potential=chemical_potential,
                    )
                else:
                    dos = dos.vacant(
                        temperature=temperature,
                        density_per_unit_cell=material.density_per_unit_cell,
                        chemical_potential=chemical_potential,
                    )
            self._dos.append(dos)

            # Load RIXS and process them exactly as we did the XFELs
            rixs = Spectra.load(base_path / "rixs_spectra" / f"rixs_{i}.pkl")
            self._rixs_energies = rixs.energies
            if pick_subsets:
                rixs = rixs.pick_subsets(pick_subsets)
            if subset_filter_range:
                rixs = rixs.filter_on_subsets(*subset_filter_range)
            rixs = rixs.noisy(self.out_noise)
            self._rixs.append(rixs)

        # Repeat XFEL signatures to match length of the RIXS data (length n_dos * n_xfel)
        self._joint_xfels = self._xfels.signal().repeat(len(self._rixs), 1)

        # Start with the first element in RIXS and DoS lists, append the others iteratively
        self._joint_rixs = self._rixs[0].signal()
        # Expand to length n_xfel
        self._joint_dos = self._dos[0].density.unsqueeze(0).repeat(len(self._xfels), 1)

        for i, rixs in enumerate(self._rixs[1:]):  # repeat n_dos - 1 times
            # Append other RIXS and DoS to total length n_dos * n_xfel
            self._joint_rixs = torch.cat((self._joint_rixs, rixs.signal()), axis=0)
            self._joint_dos = torch.cat(
                (
                    self._joint_dos,
                    self._dos[i + 1].density.unsqueeze(0).repeat(len(self._xfels), 1),
                ),
                axis=0,
            )

        # Confirm all the lengths match
        assert self._joint_dos.shape[0] == self._joint_rixs.shape[0]
        assert self._joint_rixs.shape[0] == self._joint_xfels.shape[0]

        if log_rixs:  # Use this to mitigate large range of RIXS data
            self.rixs_logged = True
            self._joint_rixs = torch.log(1 + self._joint_rixs)

        if normalize:  # Standard normalization for ML
            self.normalized = True
            self._dos_max = self._joint_dos.max()
            self._rixs_max = self._joint_rixs.max()
            self._xfel_max = self._joint_xfels.max()

            self._joint_dos /= self._dos_max
            self._joint_rixs /= self._rixs_max
            self._joint_xfels /= self._xfel_max

    def undo_rixs_transform(self, transformed_data: torch.Tensor) -> torch.Tensor:
        if self.normalized:
            transformed_data = transformed_data * self._rixs_max
        if self.rixs_logged:
            transformed_data = torch.exp(transformed_data - 1)
        return transformed_data

    @property
    def feat_count(self) -> int:
        return self._joint_rixs.shape[1] + self._joint_xfels.shape[1]

    @property
    def label_count(self) -> int:
        return self._joint_dos.shape[1]

    @property
    def max_dos(self) -> torch.Tensor:
        return self._dos_max

    def __len__(self) -> int:
        return self._joint_rixs.shape[0]

    def __getitem__(
        self, item
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return (
            (self._joint_xfels[item, :], self._joint_rixs[item, :]),
            self._joint_dos[item, :],
        )

    def plot(self, save_to: Path | str | None = None) -> None:
        num_dos = len(self._dos_list)
        fig = plt.figure(figsize=(num_dos * 2, (len(self._xfels.partitions) + 1.5) * 2))

        width = num_dos + 2

        left_margin = 0.5 / width
        right_margin = 1 - left_margin
        xfels_right = left_margin + (0.7 / width)
        rixs_left = xfels_right + (0.3 / width)

        gs_xfels = plt.GridSpec(
            2,
            1,
            left=left_margin,
            right=xfels_right,
            top=0.95,
            bottom=0.05,
            height_ratios=[len(self._xfels.partitions), 1],
        )
        gs_rixs = plt.GridSpec(
            2,
            num_dos,
            left=rixs_left,
            right=right_margin,
            top=0.95,
            bottom=0.05,
            height_ratios=[len(self._xfels.partitions), 1],
        )
        ax_xfels = fig.add_subplot(gs_xfels[0, 0])
        ax_xfels.imshow(self._xfels.signal(), aspect="auto", cmap="PuRd")
        ax_xfels.set_ylabel("Shot #")
        ax_xfels.set_title("XFEL Spectra")

        for i in range(num_dos):
            rixs = self._rixs[i].signal()
            dos = self._dos[i]

            ax_1 = fig.add_subplot(gs_rixs[0, i])
            ax_1.imshow(rixs, aspect="auto", cmap="PuRd")
            ax_1.set_yticklabels([])
            ax_1.set_title(f"RIXS Spectra {i}")

            ax_2 = fig.add_subplot(gs_rixs[1, i])
            ax_2.plot(dos.energies, dos.density, c="k")
            ax_2.set_yticklabels([])
            ax_2.set_title(f"DoS {i}")

        if save_to is not None:
            plt.savefig(save_to)

    def load(self, batch_size, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)