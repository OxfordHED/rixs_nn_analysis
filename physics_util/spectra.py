from __future__ import annotations
from pathlib import Path
from typing import Generator
import pickle
from deprecated import deprecated
import numpy as np
import torch
from scipy import interpolate

from .material import Material


def infer_dos_range(
    xfel_energies: torch.Tensor,
    rixs_energies: torch.Tensor,
    material: Material,
    resolution: float = 0.5,
) -> torch.Tensor:
    lower = xfel_energies.min() - rixs_energies.max() + material.energy_vacancies.min()
    upper = xfel_energies.max() - rixs_energies.min() + material.energy_vacancies.max()

    addition = resolution if not ((upper - lower) // resolution) % 2 else 0

    return torch.arange(lower, upper + addition, resolution)


class Spectra:
    @deprecated
    @staticmethod
    def align_energies(
        energies: torch.Tensor,
        signal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert (
            len(energies.shape) == 2 and energies.shape[0] > 1
        ), "Only intended for shifted energy axes"
        # For identical, shifted pulses where energies is (n x r) and signal is (1 x r)

        # expand axis rather than picking one index
        interp_axis = torch.arange(energies[0, 0], energies[-1, -1], 0.5)

        # if there is multiple but with the same energy axis (xfel, extra)
        if len(signal.shape) == 2 and signal.shape[0] > 1:
            interpolator = interpolate.interp1d(
                energies[energies.shape[0] // 2, :],
                signal,
                bounds_error=False,
                fill_value=0,
            )
            return interp_axis, torch.from_numpy(interpolator(interp_axis)).float()

        signal_out = torch.zeros((energies.shape[0], interp_axis.shape[0]))

        for i in range(energies.shape[0]):
            interpolator = interpolate.interp1d(
                energies[i, :], signal, bounds_error=False, fill_value=0
            )

            signal_out[i, :] = torch.from_numpy(interpolator(interp_axis)).float()

        energies = interp_axis
        signal = signal_out

        return energies, signal

    def __init__(
        self,
        energies: torch.Tensor,
        signal: torch.Tensor,
        weights: torch.Tensor | None = None,
        partitions: list[int] | None = None,
        noise: float = 0,
    ) -> None:
        # called for xfel, but not rixs
        if len(energies.shape) == 2:
            raise DeprecationWarning(
                f"The data format supplied (len(energies.shape) > 1) is being deprecated."
            )
            energies, signal = Spectra.align_energies(energies, signal)

        self.noise = noise

        self._signal = signal

        self.energies = energies
        self.weights = weights
        if partitions is None:
            self.partitions = [0]
        else:
            self.partitions = partitions

        self.function = interpolate.interp1d(
            self.energies, self._signal, bounds_error=False, fill_value="extrapolate"
        )

    def signal(self, energies: torch.Tensor | None = None) -> torch.Tensor:
        if energies is None:
            return self._signal

        return torch.from_numpy(self.function(energies)).float()

    def resample(self, energies: torch.Tensor) -> Spectra:
        new_signal = self.signal(energies)

        return Spectra(energies, new_signal, self.weights)

    def append(self, other: Spectra) -> None:
        assert torch.allclose(self.energies, other.energies)
        self.partitions.append(len(self))
        self._signal = torch.cat((self._signal, other.signal()), axis=0)
        self.function = interpolate.interp1d(
            self.energies, self._signal, bounds_error=False, fill_value="extrapolate"
        )
        if self.weights is None:
            assert other.weights is None
        else:
            self.weights = torch.cat((self.weights, other.weights), axis=0)

    def subset(self, idx: int) -> Spectra:
        # if there are no underlying partitions
        if idx == 0 and len(self.partitions) == 1:
            return self
        assert 0 <= idx < len(self.partitions)
        lower_idx = self.partitions[idx]
        if idx == len(self.partitions) - 1:
            upper_idx = None
        else:
            upper_idx = self.partitions[idx + 1]
        return Spectra(
            self.energies,
            self._signal[lower_idx:upper_idx, :],
            self.weights if self.weights is None else self.weights[lower_idx:upper_idx],
        )

    def gen_subsets(self) -> Generator[Spectra, None, None]:
        for i in range(len(self.partitions)):
            yield self.subset(i)

    def filter_on_subsets(
        self, start: int | None, end: int | None, step: int = 1
    ) -> Spectra:
        out = None
        for subset in self.gen_subsets():
            if out is None:
                out = subset[start:end:step]
            else:
                out.append(subset[start:end:step])
        return out

    def pick_subsets(self, subsets: list[int]) -> None | Spectra:
        out = None
        for i, subset in enumerate(self.gen_subsets()):
            if i not in subsets:
                continue
            if out is None:
                out = subset
            else:
                out.append(subset)
        return out

    def noisy(self, noise_level: float) -> Spectra:
        noise = self._signal.max(dim=1, keepdim=True).values * noise_level
        return Spectra(
            energies=self.energies,
            signal=torch.normal(self._signal, noise),
            weights=self.weights,
            partitions=self.partitions,
            noise=noise_level,
        )

    def __len__(self) -> int:
        return self._signal.shape[0]

    def __getitem__(self, item: slice) -> Spectra:
        partitions = self.partitions.copy()
        if isinstance(item, slice):
            partitions = [
                p - item.start for p in partitions if item.start <= p < item.stop
            ]
        else:
            raise NotImplementedError(
                f"Calling [] on Spectra with unsupported argument {item}."
            )
        return Spectra(
            self.energies,
            self._signal[item, :],
            self.weights if self.weights is None else self.weights[item],
            partitions=partitions,
        )

    def save(self, filename: str | Path) -> None:
        assert self.noise == 0, "Don't save noisy spectra."
        obj_dict = self.__dict__.copy()
        del obj_dict["function"]
        with open(filename, "wb") as file:
            pickle.dump(obj_dict, file)

    @classmethod
    def load(cls, filename: str | Path) -> Spectra:
        with open(filename, "rb") as file:
            obj = pickle.load(file)

        obj["signal"] = obj["_signal"].clone()
        del obj["_signal"]

        return cls(**obj)

    @classmethod
    @deprecated
    def from_files(
        cls,
        x_path: str | Path,
        y_path: str | Path,
        seeded: bool = False,
    ) -> Spectra:
        signal_data = torch.from_numpy(np.load(y_path)).float()
        if seeded:
            xfel_signal = signal_data[:, :-1]
            xfel_weights = signal_data[:, -1]
        else:
            xfel_signal = signal_data
            xfel_weights = None
        xfel_energies = torch.from_numpy(np.load(x_path)).float()
        return cls(
            xfel_energies,
            xfel_signal,
            xfel_weights,
        )
