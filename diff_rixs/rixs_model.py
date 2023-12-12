from typing import Callable, TYPE_CHECKING

import torch
from torch import nn
from torch.autograd import Function


# This avoids type checking failures
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
    """A helper function to round values onto a predefined grid.

    Args:
        value: float, the value to round
        resolution: float, the resolution of the grid

    Returns:
        float, the nearest grid point to value (rounded down).
    """
    # (m // r) * r rounds down to the nearest point on a grid with spacing r
    return (value // resolution) * resolution


class RIXSModel(nn.Module):
    """
    A subclass of nn.Module, used to perform forward and backwards passes through the RIXS physics
    model.
    """
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
        fit_temp: bool = False,
    ) -> None:
        """Initial setup of RIXSmodel.

        Args:
            dos_function: Callable, the function mapping from energy space to DOS.
            material: "Material" instance, contains physical information of the scattering material.
            thermodynamic_props: "ThermodynamicalProperties" instance, contains temperature and
                chemical potential information.
            xfel_energies: torch.Tensor, energy range across which XFEL spectra are recorded.
            dos_energies: torch.Tensor, energy range across which DoS will be taken.
            rixs_energies: torch.Tensor, energy range across which RIXS spectra are recorded.
            resolution: float, resolution in which to process spectra, in eV. Default 0.5.
            vacant: bool, whether the provided DoS is already vacant. If False, thermal factors will
                be applied in forward pass. Default False.
            thermal_backprop: bool, whether to apply thermal factors in backward pass. No effect if
                vacant is True. Set False to avoid vanishing gradients. Default False.
            lor_backprop: bool, whether to apply Lorentzian factors in backward pass. Set False to
                avoid vanishing gradients. Default False.
            fit_temp: bool, whether to fit the temperature through the model. Substantial
                performance improvement if False, as chemical potentials will not be reevaluated.
                Default False.
        """
        super().__init__()

        # Store relevant inputs
        self.material = material
        self.thermodynamic_props = thermodynamic_props
        self.resolution = resolution

        # As there are different final states in e.g. iron, there are multiple overlaid spectra,
        # each shifted by a certain amount.
        # how many grid spacings the different transitions are shifted by.
        self.deltas = (
            (self.material.energy_vacancies - self.material.energy_vacancies.min())
            // self.resolution
        ).int()

        # Required to recombine the spectra from different final states.
        self.max_delta = self.deltas.max()
        self.uppers = [
            (d - self.max_delta) if (d - self.max_delta) < 0 else None
            for d in self.deltas
        ]

        # Store the relevant energy ranges
        self.raw_dos_energies = dos_energies
        self.raw_xfel_energies = xfel_energies
        self.rixs_energies = rixs_energies

        # initialize in __init__ to make pylint happy
        self._dos_energies: torch.Tensor | None = None
        self._xfel_energies: torch.Tensor | None = None
        self._full_energies: torch.Tensor | None = None
        self.output_map: torch.Tensor | None = None

        # Allows for the combination of different sampling schemes and resolutions, aligns them.
        self.align_energies()

        # Thermal factors and Lorentzians can be initialised once and used in every calculation
        self.thermal_factors = self.thermodynamic_props.thermal_factor(
            self._dos_energies
        )
        # This flag will cause thermal factors to be regenerated for each forward pass, required to
        # fit temperatures.
        self._fit_temp = fit_temp


        # Evaluate the Lorentzian factors with respect to each transition.
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
            self.lor_product = LorentzianProduct  # Lorentzian factors always
        else:
            # Lorentzian factor only in forward pass (to avoid vanishing gradients)
            self.lor_product = ModBackpropLorentzianProduct

        # Can be a generic callable, i.e. a defined function or a nn.Module (Neural Network) call
        self.dos = dos_function

    @property
    def dos_energies(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor, the range of DoS energies after alignment.
        """
        return self._dos_energies

    @property
    def xfel_energies(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor, the range of XFEL spectra energies after alignment.
        """
        return self._xfel_energies

    def align_energies(self) -> None:
        """This helper function aligns sampling ranges and resolutions, to allow for the use of
        kernel methods in calculating the RIXS spectra (significant performance boost).

        Returns:
            None.
        """
        # We need a symmetric kernel to utilize convolution frameworks
        symmetric_dos_range = round_to_resolution(
            self.raw_dos_energies.abs().max(), self.resolution
        )

        # Expand the range of DoS energies to be symmetric around 0.
        self._dos_energies = torch.arange(
            -symmetric_dos_range, symmetric_dos_range + self.resolution, self.resolution
        )

        # Find the correct XFEL energy range using the fixed resolution.
        new_min_xfel_energy = round_to_resolution(
            self.raw_xfel_energies.min(), self.resolution
        )
        new_max_xfel_energy = round_to_resolution(
            self.raw_xfel_energies.max(), self.resolution
        )

        # Resample XFEL energies with the correct resolution.
        self._xfel_energies = torch.arange(
            new_min_xfel_energy, new_max_xfel_energy + self.resolution, self.resolution
        )

        # Find the minimum and maximum energies in final states.
        # # This is used to shift the underlying spectra correctly.
        min_vac_energy = round_to_resolution(
            self.material.energy_vacancies.min(), self.resolution
        )
        max_vac_energy = round_to_resolution(
            self.material.energy_vacancies.max(), self.resolution
        )

        # Helper calculated to avoid shape off by 1 error.
        difference = (
            self._xfel_energies[-1]
            + symmetric_dos_range
            + max_vac_energy
            - (self._xfel_energies[0] - symmetric_dos_range + min_vac_energy)
        ) // self.resolution

        # Helper calculated to avoid shape off by 1 error.
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

        # Fill in mapping matrix.
        for i, energy in enumerate(self.rixs_energies):
            closest_ind = (self._full_energies - energy).abs().argmin()
            self.output_map[i, closest_ind] = 1

        # Check whether the map has been constructed to preserve intensity.
        assert self.output_map.sum() == self.rixs_energies.shape[0]

    def forward(self, xfel_pulses: torch.Tensor) -> torch.Tensor:
        """The forward pass through the RIXS model. This simulates a RIXS response for the relevant
        material being exposed to the incoming XFEL spectra.

        Args:
            xfel_pulses: torch.Tensor (2d), spectra aligned with the XFEL energies provided in
                initializing the RIXS model.

        Returns:
            torch.Tensor (2d), the RIXS spectra generated from the corresponding XFEL pulses.
        """

        # Ensure everything is initialized
        assert (
            self.dos is not None
        ), "Please initialize density of states before you continue"

        assert len(xfel_pulses.shape) == 2  # Correct shape for batched spectra

        # Correct dimensions (2d) must be provided if self.dos is a nn.Module
        dos = self.dos(self.dos_energies[:, None]).squeeze()

        # We need to regenerate thermal factors regularly to fit temperatures
        if self._fit_temp:
            self.thermal_factors = self.thermodynamic_props.thermal_factor(
                self._dos_energies
            )
        # Calculate the cross-section for each final state and input energy in a 2d matrix.
        cross_sections = self.cross_section.apply(
            dos, self.material.oscillator_strengths, self.thermal_factors
        ).T.float()
        # Use a 1d Convolution to model the inelasticity of the scattering.
        convolved = nn.functional.conv1d(
            # Cast into 2d to align with different final state cross-sections
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

        # map convolutions onto the same axes again.
        for i, d in enumerate(self.deltas):
            aligned_convolved[:, i, d : self.uppers[i]] = convolved[:, i, :]

        # Apply Lorentzian factors to account for Resonances.
        high_res_spectrum = self.lor_product.apply(
            aligned_convolved,
            self.lorentzians.unsqueeze(0),
            self._full_energies.unsqueeze(0),
        )

        # We normalize to previous value of resolution 0.5 for backwards compatibility.
        # We also map back onto a 1d spectrum.
        output = torch.matmul(self.output_map, high_res_spectrum.T).T * (
            self.resolution / 0.5
        )

        return output