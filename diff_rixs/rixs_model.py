from typing import Callable, TYPE_CHECKING

import torch
from torch import nn
from torch.autograd import Function

if TYPE_CHECKING:
    from physics_util import Material, ThermodynamicalProperties
from .torch_components import (
    FunctionEstimator,
    SSP,
    LorentzianProduct,
    ModBackpropLorentzianProduct,
    FullCrossSection,
    NoThermalBackpropCrossSection,
    VacantCrossSection,
)


def round_to_resolution(value: float, resolution: float) -> float:
    # (m // r) * r rounds down to the nearest point on a grid with spacing r
    return (value // resolution) * resolution


class RIXSModel(nn.Module):
    def __init__(
        self,
        dos_function: Callable,
        material: "Material",
        thermodynamic_props: "ThermodynamicalProperties",
        xfel_energies: torch.Tensor,
        dos_energies: torch.Tensor,
        rixs_energies: torch.Tensor,
        resolution: float = 0.5,
        vacant: bool = False,
        thermal_backprop: bool = False,
        lor_backprop: bool = False,
    ) -> None:
        super().__init__()
        self.material = material
        self.thermodynamic_props = thermodynamic_props

        self.resolution = resolution

        # how many grid spacings the different transitions are shifted by
        self.deltas = (
            (self.material.energy_vacancies - self.material.energy_vacancies.min())
            // self.resolution
        ).int()

        self.max_delta = self.deltas.max()
        self.uppers = [
            (d - self.max_delta) if (d - self.max_delta) < 0 else None
            for d in self.deltas
        ]

        self.raw_dos_energies = dos_energies
        self.raw_xfel_energies = xfel_energies
        self.rixs_energies = rixs_energies

        # initialize in __init__ to make pylint happy
        self._dos_energies: torch.Tensor | None = None
        self._xfel_energies: torch.Tensor | None = None
        self._full_energies: torch.Tensor | None = None
        self.output_map: torch.Tensor | None = None
        self.align_energies()

        # Thermal factors and lorentzians can be initialised once and used in every calculation
        self.thermal_factors = self.thermodynamic_props.thermal_factor(
            self._dos_energies
        )

        self.lorentzians = torch.zeros(
            (self.material.num_transitions, self._full_energies.shape[0])
        )
        for i in range(self.material.num_transitions):
            self.lorentzians[i, :] = self.material.lor_factor(self._full_energies, i)

        # options for backpropagation
        if vacant:
            self.cross_section = VacantCrossSection  # No thermal factors at all
        else:
            if thermal_backprop:
                self.cross_section = FullCrossSection  # Thermal factors always
            else:
                # Thermal factors only in forward pass (to avoid vanishing gradients)
                self.cross_section = NoThermalBackpropCrossSection

        if lor_backprop:
            self.lor_product = LorentzianProduct
        else:
            # Lorentzian factor only in forward pass (to avoid vanishing gradients)
            self.lor_product = ModBackpropLorentzianProduct

        # Can be a generic callable, i.e. a defined function or a nn.Module call
        self.dos = dos_function

    @property
    def dos_energies(self) -> torch.Tensor:
        return self._dos_energies

    @property
    def xfel_energies(self) -> torch.Tensor:
        return self._xfel_energies

    def align_energies(self) -> None:
        # We need a symmetric kernel to utilize convolution frameworks
        symmetric_dos_range = round_to_resolution(
            self.raw_dos_energies.abs().max(), self.resolution
        )

        self._dos_energies = torch.arange(
            -symmetric_dos_range, symmetric_dos_range + self.resolution, self.resolution
        )

        new_min_xfel_energy = round_to_resolution(
            self.raw_xfel_energies.min(), self.resolution
        )
        new_max_xfel_energy = round_to_resolution(
            self.raw_xfel_energies.max(), self.resolution
        )

        self._xfel_energies = torch.arange(
            new_min_xfel_energy, new_max_xfel_energy + self.resolution, self.resolution
        )

        min_vac_energy = round_to_resolution(
            self.material.energy_vacancies.min(), self.resolution
        )
        max_vac_energy = round_to_resolution(
            self.material.energy_vacancies.max(), self.resolution
        )

        difference = (
            self._xfel_energies[-1]
            + symmetric_dos_range
            + max_vac_energy
            - (self._xfel_energies[0] - symmetric_dos_range + min_vac_energy)
        ) // self.resolution

        addition = self.resolution if not difference % 2 else 0

        # full aligned energy range to map to rixs spectrum
        self._full_energies = torch.arange(
            self._xfel_energies[0] - symmetric_dos_range + min_vac_energy,
            self._xfel_energies[-1] + symmetric_dos_range + max_vac_energy + addition,
            torch.tensor(self.resolution),
        )

        # matrix to map from convolution output to RIXS energies, allows for different resolutions
        self.output_map = torch.zeros(
            (self.rixs_energies.shape[0], self._full_energies.shape[0])
        )

        for i, energy in enumerate(self.rixs_energies):
            closest_ind = (self._full_energies - energy).abs().argmin()
            self.output_map[i, closest_ind] = 1

        assert self.output_map.sum() == self.rixs_energies.shape[0]

    def forward(self, xfel_pulses: torch.Tensor) -> torch.Tensor:
        assert (
            self.dos is not None
        ), "Please initialize density of states before you continue"
        assert len(xfel_pulses.shape) == 2
        dos = self.dos(self.dos_energies[:, None]).squeeze()
        cross_sections = self.cross_section.apply(
            dos, self.material.oscillator_strengths, self.thermal_factors
        ).T.float()
        convolved = nn.functional.conv1d(
            input=xfel_pulses.unsqueeze(1).repeat(1, self.material.num_transitions, 1),
            weight=cross_sections.unsqueeze(1),
            padding=cross_sections.shape[1] - 1,
            groups=self.material.num_transitions,
        )

        # Full shape of the output convolution, accounting for shift in different energy transitions
        aligned_convolved = torch.zeros(
            (
                xfel_pulses.shape[0],
                self.material.num_transitions,
                self._full_energies.shape[0],
            )
        )

        # map convolutions onto the same axes again
        for i, d in enumerate(self.deltas):
            aligned_convolved[:, i, d : self.uppers[i]] = convolved[:, i, :]

        high_res_spectrum = self.lor_product.apply(
            aligned_convolved,
            self.lorentzians.unsqueeze(0),
            self._full_energies.unsqueeze(0),
        )

        # We normalize to previous value of resolution 0.5 for backwards compat
        return torch.matmul(self.output_map, high_res_spectrum.T).T * (
            self.resolution / 0.5
        )
