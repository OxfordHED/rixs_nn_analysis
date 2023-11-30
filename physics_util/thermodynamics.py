from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
import torch

DEFAULT_TEMPERATURE = torch.Tensor([8])

@dataclass
class ThermodynamicalProperties:
    temperature: torch.Tensor
    electron_density: torch.Tensor
    _chemical_potential: torch.Tensor
    chemical_potential_error: torch.Tensor
    dos: object
    fermi_energy: torch.Tensor | None = None

    @property
    def chemical_potential(self):
        if self.fermi_energy is None:
            return self._chemical_potential

        return ThermodynamicalProperties.chemical_potential_fit(
            self.temperature,
            self.fermi_energy.detach()
        )

    @staticmethod
    def generic_thermal_factor(
        energy: torch.Tensor | float,
        chemical_potential: float,
        temperature: float
    ) -> torch.Tensor | float:
        """signal()
        Parameters
        ----------
        energy : energy of the electronic level [eV]
        chemical_potential : chemical potential [eV]
        temperature : Temperature of the system [eV]

        Returns
        -------
        Thermal factor of energy

        """
        if not isinstance(chemical_potential, torch.Tensor):
            chemical_potential = torch.tensor(chemical_potential)
        return 1.0 / (1 + torch.exp((chemical_potential - energy) / temperature))

    @staticmethod
    def chemical_potential_solver(
        temperature: float,
        chem_pot_guess: float,
        rho: float,
        dos: object,
        energies: torch.Tensor
    ) -> tuple[float, float]:
        """
        Parameters
        ----------
        temperature : temperature of the system (in eV)
        rho : density of the system
              (# of electrons in the bands present in the DOS taken into consideration / V)
        dos : Density of states (# states/ (eV*V))

        Warning: The volume for rho and the DOS must be expressed in the same units

        This routine may be sped up.
        (look 'Calculating the integral of the density of states for electrons in metals' on google)


        Returns
        -------
        Chemical potential

        """
        temperature = temperature.detach().numpy()
        rho = rho.detach().numpy()
        dos = dos.detach()
        energies = energies.detach().numpy()

        delta = energies[1] - energies[0]

        def integrand(mu_prime):
            # vectorized version should be faster than loop
            def partial_thermal(energy):
                return ThermodynamicalProperties.generic_thermal_factor(
                    energy, mu_prime, temperature
                )

            return dos * (1 - partial_thermal(energies))

        def difference(chemical_potential):
            integral = integrate.trapz(integrand(chemical_potential), dx=delta)

            return rho - integral

        vfunc = np.vectorize(difference)

        chem_pot_exit = optimize.fsolve(vfunc, chem_pot_guess)

        error = difference(chem_pot_exit)

        return chem_pot_exit, error

    @staticmethod
    def fermi_energy_solver(
        dos: object,
        electron_density: float,
        fermi_guess: float = 0.
    ) -> torch.Tensor:
        energies = dos.energies.detach().numpy()
        delta = energies[1] - energies[0]

        def integrand(fermi_energy):
            return np.where(energies < fermi_energy, dos.density, 0)

        def difference(fermi_energy):
            integral = integrate.trapz(integrand(fermi_energy), dx=delta)

            return electron_density - integral

        vfunc = np.vectorize(difference)

        fermi_energy_exit = optimize.fsolve(vfunc, fermi_guess)

        error = difference(fermi_energy_exit)
        return torch.tensor(fermi_energy_exit).float()


    @staticmethod
    def chemical_potential_fit(temp: float, fermi_energy: float) -> float:

        '''
        Parameters
        ----------
        temp: temperature of the system
        fermi_energy: fermi energy of the system


        Returns
        -------
        Chemical potential approximation for the given fermi energy and temperature
        '''

        # uniform electron gas
        # fermi_energy = hartree * 0.5 * (3 * np.pi ** 2 * electron_density) ** (2 / 3)

        part_1 = -0.28468 - 1.5 * torch.log(temp / fermi_energy)

        part_2_num = (
            0.25945 * (temp / fermi_energy) ** (-1.858)
            + 0.072 * (temp / fermi_energy) ** (-0.929)
        )

        part_2_denom = (1 + 0.25945 * (temp / fermi_energy) ** (-0.858))

        eta = part_1 + part_2_num / part_2_denom
        return eta * temp

    def thermal_factor(self, energy: torch.Tensor | float) -> torch.Tensor | float:
        thermal_fac = ThermodynamicalProperties.generic_thermal_factor(
            energy.detach(), self.chemical_potential, self.temperature
        )
        return thermal_fac

    @classmethod
    def from_dos(
        cls,
        dos: object,
        temperature: float,
        electron_density: float,
        guess: float = 1,
        chemical_potential: torch.Tensor | None = None,
    ) -> ThermodynamicalProperties:
        if chemical_potential is None:
            (
                chemical_potential,
                error,
            ) = ThermodynamicalProperties.chemical_potential_solver(
                temperature=temperature,
                chem_pot_guess=guess,
                rho=electron_density,
                dos=dos.density,
                energies=dos.energies,
            )

            # assert abs(error / chemical_potential) < 1e-5
            chemical_potential = torch.tensor(chemical_potential)
        else:
            error = 0.0

        return cls(
            temperature=temperature,
            electron_density=electron_density,
            _chemical_potential=chemical_potential,
            chemical_potential_error=torch.tensor(error),
            dos=dos,
        )

    @classmethod
    def from_fermi_energy(
        cls,
        dos: object,
        temperature: float,
        electron_density: float,
    ) -> ThermodynamicalProperties:
        # This separates out the fitting of the fermi energy from the chemical potential
        fermi_energy = ThermodynamicalProperties.fermi_energy_solver(dos, electron_density)
        chemical_potential = ThermodynamicalProperties.chemical_potential_fit(
            temperature,
            fermi_energy
        )

        return cls(
            temperature=temperature.requires_grad_().float(),
            electron_density=electron_density,
            _chemical_potential=chemical_potential,
            chemical_potential_error=0.0,  # This is not accurate (there is a propagated error)
            dos=dos,
            fermi_energy=fermi_energy
        )
