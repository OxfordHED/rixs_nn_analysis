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
    def generic_thermal_factor(energy, chemical_potential, temperature):
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
    def chemical_potential_solver(temperature, chem_pot_guess, rho, dos, energies):
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

    def thermal_factor(self, energy):
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
