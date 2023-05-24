from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class Material:
    """
    Dataclass to contain the material properties relevant for the RIXS process
    """

    energy_transition: torch.Tensor
    energy_vacancies: torch.Tensor
    gamma_factors: torch.Tensor
    oscillator_strengths: torch.Tensor
    density_per_unit_cell: torch.Tensor

    def lor_factor(self, energy: torch.Tensor, vacancy_index: int = 0) -> Any:
        """Lorentzian factor for a transition from the band energy_vacancies[vacancy_index] to
        the band energy_transition.

        Parameters
        ----------
        energy: energy for which to obtain the Lorentzian factor.

        Returns
        -------
        Lorentzian factor.
        """
        delta = energy - (self.energy_vacancies[vacancy_index] - self.energy_transition)
        gamma = self.gamma_factors[vacancy_index]

        return 1.0 / (delta**2 + gamma**2)

    @property
    def num_transitions(self) -> int:
        return len(self.energy_vacancies)

    @classmethod
    def Fe(cls) -> Material:
        """
        Returns
        -------
        Material dataclass populated with tabulated values of Iron
        """
        return cls(
            energy_transition=torch.tensor([-7112]).float(),
            energy_vacancies=torch.tensor([-723, -708.5]).float(),
            gamma_factors=torch.tensor([3, 2.55]).float(),
            oscillator_strengths=torch.tensor([50, 100]).float(),
            density_per_unit_cell=torch.tensor(16).float(),
        )

    @classmethod
    def Fe2O3(cls) -> Material | None:
        material = cls.Fe()
        material.density_per_unit_cell = torch.tensor(150).float()
        return material

    @classmethod
    def FeO(cls) -> Material | None:
        raise NotImplementedError()

    @classmethod
    def Ni(cls) -> Material | None:
        raise NotImplementedError()
