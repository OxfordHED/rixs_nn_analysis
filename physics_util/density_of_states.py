from __future__ import annotations
import pickle
from pathlib import Path
from typing import Callable, Any
from deprecated import deprecated

import numpy as np
from scipy import interpolate
import torch
from torch import nn

from diff_rixs import FunctionEstimator, SSP, Siren
from .thermodynamics import get_thermals


def gaussian(energy: np.ndarray, sigma: np.ndarray, mu_center: np.ndarray) -> Any:
    return np.exp(-0.5 * np.power((energy - mu_center) / sigma, 2.0)) / (
        np.sqrt(2 * np.pi) * sigma
    )


class DensityOfStates:
    def __init__(
        self,
        energies: torch.Tensor,
        density: torch.Tensor,
        function: Callable[[torch.Tensor], torch.Tensor] | None = None,
        interpolate_function: bool | None = False,
    ):
        self.energies = energies.float()
        self.density = density.float()
        self.function = function
        self.interpolate_function = interpolate_function

        if self.function is None and interpolate_function:
            interpolator = interpolate.interp1d(
                self.energies, self.density, bounds_error=False, fill_value=0
            )

            self.function = lambda x: torch.from_numpy(interpolator(x)).float()

    @classmethod
    @deprecated
    def from_files(
        cls,
        x_path: str | Path,
        y_path: str | Path,
        interpolate_function: bool | None = False,
    ) -> DensityOfStates:
        energies = torch.from_numpy(np.load(x_path)).float()
        dos = torch.from_numpy(np.load(y_path)).float()
        if len(energies.shape) == 2 and energies.shape[0] > 1:
            if energies.shape == dos.shape:
                energies = energies[-1, :]
                dos = dos[-1, :]
            else:
                energies = energies[energies.shape[0] // 2, :]
                dos = dos[-1, :]
        return cls(energies, dos, interpolate_function=interpolate_function)

    @classmethod
    def from_function(
        cls, energies: torch.Tensor, function: Callable
    ) -> DensityOfStates:
        dos = function(energies)
        return cls(energies.float(), dos.float(), function)

    @classmethod
    def synthetic(
        cls,
        energies: torch.Tensor,
        gaussian_centers: list[float],
        gaussian_widths: list[float],
        gaussian_intensities: list[float],
        sqrt_amplitude: float = 0.1,
        sqrt_offset: float = 0,
    ) -> DensityOfStates:
        assert len(gaussian_centers) == len(gaussian_widths)
        assert len(gaussian_intensities) == len(gaussian_widths)

        def function(energy: torch.Tensor) -> torch.Tensor:
            out = 0
            for c, s, i in zip(gaussian_centers, gaussian_widths, gaussian_intensities):
                out += i * gaussian(energy, s, c)

            sqrt_vals = np.zeros(energy.shape)
            np.sqrt(
                energy - sqrt_offset,
                out=sqrt_vals,
                where=np.heaviside(energy - sqrt_offset, 1),
            )

            return torch.tensor(out + sqrt_amplitude * sqrt_vals)

        return cls.from_function(energies, function)

    @classmethod
    def random_synthetic(
        cls,
        energies: torch.Tensor,
        gauss_amp_range: tuple[float, float],
        gauss_width_range: tuple[float, float],
        gauss_number: int,
        sqrt_amp_range: tuple[float, float] | float,
        sqrt_offset_range: tuple[float, float] | float,
    ) -> DensityOfStates:
        gauss_energy_range = (energies[0], energies[-1])

        return cls.synthetic(
            energies=energies,
            gaussian_centers=np.random.uniform(
                *gauss_energy_range, gauss_number
            ).tolist(),
            gaussian_widths=np.random.uniform(
                *gauss_width_range, gauss_number
            ).tolist(),
            gaussian_intensities=np.random.uniform(
                *gauss_amp_range, gauss_number
            ).tolist(),
            sqrt_amplitude=(
                sqrt_amp_range
                if isinstance(sqrt_amp_range, float)
                else np.random.uniform(*sqrt_amp_range, 1)[0]
            ),
            sqrt_offset=(
                sqrt_offset_range
                if isinstance(sqrt_offset_range, float)
                else np.random.uniform(*sqrt_offset_range, 1)[0]
            ),
        )

    def resample(self, energies: torch.Tensor) -> DensityOfStates:
        assert self.function is not None
        new_density = self.function(energies)

        if self.interpolate_function:
            return DensityOfStates(energies, new_density, interpolate_function=True)

        return DensityOfStates(energies, new_density, function=self.function)

    def vacant(
        self,
        temperature: torch.Tensor,
        density_per_unit_cell: torch.Tensor,
        chemical_potential: torch.Tensor | None = None,
        base_dos: DensityOfStates | None = None  # well-behaved dos to find thermal factors
    ) -> DensityOfStates:
        thermals = get_thermals(
            self if base_dos is None else base_dos,
            temperature=temperature,
            electron_density=density_per_unit_cell,
            kwargs={"chemical_potential": chemical_potential},
        )

        new_density = self.density * thermals.thermal_factor(self.energies).float()

        init_dict = {
            "energies": self.energies,
            "density": new_density,
            "interpolate_function": self.interpolate_function,
        }

        if self.function is not None:
            init_dict["function"] = (
                lambda x: self.function(x) * thermals.thermal_factor(x).float()
            )

        return DensityOfStates(**init_dict)

    def save(self, filename: str | Path) -> None:
        obj_dict = self.__dict__.copy()
        del obj_dict["function"]
        # cannot pickle the generic function attribute, so we interpolate it after saving
        if not obj_dict["interpolate_function"]:
            obj_dict["interpolate_function"] = True
        with open(filename, "wb") as file:
            pickle.dump(obj_dict, file)

    @classmethod
    def load(cls, filename: str | Path) -> DensityOfStates:
        with open(filename, "rb") as file:
            obj = pickle.load(file)

        return cls(**obj)


class NeuralDoS(DensityOfStates):
    def __init__(self, estimator: nn.Module, lower_energy=None, upper_energy=None, already_vacant=True, **kwargs):
        super().__init__(**kwargs)
        self.estimator = estimator
        self.lower_energy = lower_energy
        self.upper_energy = upper_energy
        self.already_vacant = already_vacant


    def vacant(self, *args, **kwargs):
        print(self.already_vacant, args, kwargs)
        if self.already_vacant:
            return self
        return super().vacant(*args, **kwargs)

    @classmethod
    def create(
        cls,
        energies,
        num_layers: int = 4,
        num_hidden_units: int = 40,
        activation: Callable = SSP(),
        lower_energy = None,
        upper_energy = None,
        already_vacant = True
    ):
        estimator = FunctionEstimator(
            activation, num_hidden_units=num_hidden_units, num_layers=num_layers
        )

        lower_energy = -100 if lower_energy is None else lower_energy
        upper_energy = 200 if upper_energy is None else upper_energy

        def function(x):
            x = (100 * x / (upper_energy - lower_energy)) - (lower_energy + upper_energy) / 2
            return estimator(x)

        density = function(energies.unsqueeze(1))[:, 0]

        return cls(
            estimator,
            energies=energies,
            density=density,
            function=function,
            lower_energy=lower_energy,
            upper_energy=upper_energy,
            already_vacant=already_vacant
        )

    @classmethod
    def create_as_siren(
        cls,
        energies,
        num_layers: int = 4,
        num_hidden_units: int = 40,
        lower_energy = None,
        upper_energy = None,
        already_vacant = True
    ):
        estimator = Siren(num_hidden_units=num_hidden_units, num_layers=num_layers)
        lower_energy = -1 if lower_energy is None else lower_energy
        upper_energy = 1 if upper_energy is None else upper_energy

        def function(x):
            x = (100 * x / (upper_energy - lower_energy)) - (lower_energy + upper_energy) / 2
            return estimator(x)

        density = function(energies.unsqueeze(1))[:, 0]

        return cls(
            estimator,
            energies=energies,
            density=density,
            function=function,
            lower_energy=lower_energy,
            upper_energy=upper_energy,
            already_vacant=already_vacant
        )

    def save(self, filename: str | Path) -> None:
        saved_state = {
            "state_dict": self.estimator.state_dict(),
            "energies": self.energies,
            "num_units": self.estimator.num_hidden_units,
            "num_layers": self.estimator.num_layers,
            "activation": self.estimator.activation,
            "lower_energy": self.lower_energy,
            "upper_energy": self.upper_energy,
            "already_vacant": self.already_vacant
        }

        with open(filename, "wb") as file:
            torch.save(saved_state, file)

    @classmethod
    def load(cls, filename: str | Path) -> NeuralDoS:
        with open(filename, "rb") as file:
            state = torch.load(file)

        neural_dos = cls.create(
            state["energies"],
            state["num_layers"],
            state["num_units"],
            state["activation"],
            state.get("lower_energy"),
            state.get("upper_energy"),
            state.get("already_vacant", True)
        )
        neural_dos.estimator.load_state_dict(state["state_dict"])

        return neural_dos
