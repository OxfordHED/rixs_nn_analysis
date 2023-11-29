from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy import integrate, optimize
import torch

DEFAULT_TEMPERATURE = torch.Tensor([8])

@dataclass
class ThermodynamicalProperties:
    temperature: torch.Tensor
    electron_density: torch.Tensor
    chemical_potential: torch.Tensor
    chemical_potential_error: torch.Tensor
    dos: object

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
        return 1.0 / (1 + np.exp((chemical_potential - energy) / temperature))

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
        dos = dos.detach().numpy()
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
    def fermi_energy_solver(dos: object, electron_density: float) -> float:
        pass

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

        hartree = 27.211383

        # uniform electron gas
        # fermi_energy = hartree * 0.5 * (3 * np.pi ** 2 * electron_density) ** (2 / 3)

        part_1 = -0.28468 - 1.5 * torch.log(temp / fermi_energy)

        part_2_num = (
            0.25945 * (temp / fermi_energy) ** -1.858
            + 0.072 * (temp / fermi_energy) ** -0.929
        )

        part_2_denom = (1 + 0.25945 * (temp / fermi_energy) ** -0.858)

        return part_1 + part_2_num / part_2_denom

    def thermal_factor(self, energy: torch.Tensor | float) -> torch.Tensor | float:
        return ThermodynamicalProperties.generic_thermal_factor(
            energy, self.chemical_potential, self.temperature
        )

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
            chemical_potential=chemical_potential,
            chemical_potential_error=torch.tensor(error),
            dos=dos,
        )

    @classmethod
    def from_dos_fit(
        cls,
        dos: object,
        temperature: float,
        electron_density: float,
    ) -> ThermodynamicalProperties:
        # This seperates out the fitting of the fermi energy from the chemical potential
        fermi_energy = ThermodynamicalProperties.fermi_energy_solver(dos, electron_density)
        chemical_potential = ThermodynamicalProperties.chemical_potential_fit(
            temperature,
            fermi_energy
        )

        chemical_potential = torch.tensor(chemical_potential)

        return cls(
            temperature=temperature,
            electron_density=electron_density,
            chemical_potential=chemical_potential,
            chemical_potential_error=0.0,  # This is not accurate (there is a propagated error)
            dos=dos
        )
