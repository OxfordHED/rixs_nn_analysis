"""
Helper functions to construct the RIXS model nn.Module with different options, and to create NN DoS
objects.
"""
from typing import Callable

import numpy as np
import torch  # for all things PyTorch
from torch import nn  # for torch.nn.Module, PyTorch models
import torch.nn.functional as F  # for the activation function
from torch.autograd import Function
import matplotlib.pyplot as plt


class SSP(nn.Softplus):
    """
    Modified Softplus activation which can go negative.
    """
    def __init__(self, beta: int = 1, threshold: int = 20) -> None:
        super(SSP, self).__init__(beta, threshold)

    def forward(self, input_val: torch.Tensor) -> torch.Tensor:
        sp0 = F.softplus(torch.zeros(1), self.beta, self.threshold).item()
        # subtraction of Softplus(0) allows for negative activation.
        return F.softplus(input_val, self.beta, self.threshold) - sp0

PREFACTOR_SINE = 30 # SIREN paper uses 30

# From https://arxiv.org/pdf/2006.09661.pdf
class Sine(nn.Module):
    """
    Sine function module for use with SIREN architecture.
    """
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(PREFACTOR_SINE * input)

# From https://arxiv.org/pdf/2006.09661.pdf
def sine_init(m: nn.Module):
    """Initialize the weights for the sine NN correctly (for SIREN).

    Args:
        m: nn.Module, layer for which to set weights.

    Returns:
        None.
    """
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(
                -np.sqrt(6 / num_input) / PREFACTOR_SINE,
                np.sqrt(6 / num_input) / PREFACTOR_SINE
            )

# From https://arxiv.org/pdf/2006.09661.pdf
def first_layer_sine_init(m: nn.Module):
    """ Correctly initialize the SIREN weights for the first layer.

    Args:
        m: nn.Module, layer for which to initialze weights.

    Returns:
        None.
    """
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph
            # and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class FunctionEstimator(nn.Module):
    """
    A simple feed-forward neural network used to estimate the DoS function.
    """
    def __init__(
        self, activation: Callable, num_hidden_units: int = 40, num_layers: int = 4
    ) -> None:
        """

        Args:
            activation: Callable, the activation function to apply after each layer.
            num_hidden_units: int, the number of units per hidden layer.
            num_layers: int, the number of hidden layers.
        """
        super(FunctionEstimator, self).__init__()
        self.num_hidden_units = num_hidden_units
        self.num_layers = num_layers
        self.fc1 = nn.Linear(1, num_hidden_units, bias=True)
        self.fc2 = nn.ModuleList()
        for _ in range(num_layers):
            self.fc2.append(nn.Linear(num_hidden_units, num_hidden_units, bias=True))
        self.fc3 = nn.Linear(num_hidden_units, 1)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the neural network to inputs x.

        Args:
            x: torch.Tensor (2d), inputs to the NN.

        Returns:
            torch.Tensor, output from the NN.
        """
        x = self.fc1(x)
        x = self.activation(x)
        for fc in self.fc2:
            x = fc(x)
            x = self.activation(x)
        x = self.fc3(x)
        return x

# From https://arxiv.org/pdf/2006.09661.pdf
class Siren(FunctionEstimator):
    """
    A modified neural network with sine activation functions and correct initialization for SIREN.
    """

    def __init__(self, num_hidden_units: int = 40, num_layers: int = 4):
        super().__init__(Sine(), num_hidden_units, num_layers)
        self.fc1.apply(first_layer_sine_init)
        self.fc2.apply(sine_init)
        # self.fc3.apply(sine_init)


class FullCrossSection(Function):
    """
    Calculates the cross-section of RIXS scattering through vacant intermediate states.
    Backpropagation includes thermal factor.
    """
    @staticmethod
    def forward(
        ctx,
        dos: torch.Tensor,
        oscillator_strengths: torch.Tensor,
        thermal_factors: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the cross-section for each transition given by oscillator strengths.

        Args:
            ctx: Context object to store data for backwards pass.
            dos: torch.Tensor, density of states evaluated at the relevant energies.
            oscillator_strengths: torch.Tensor, the strength of transitions to each final state.
            thermal_factors: torch.Tensor, the thermal factors at each DoS energy.
                Must have the same shape as dos.

        Returns:
            torch.Tensor, the cross-section including thermal factors and oscillator strengths.
        """
        ctx.save_for_backward(oscillator_strengths, thermal_factors, dos)
        return torch.outer(thermal_factors * dos, oscillator_strengths)

    @staticmethod
    def backward(ctx, grad_output):
        """Calculate the full backwards pass for cross-sections (including thermal factors).

        Args:
            ctx: Context object to retrieve data from forward pass.
            grad_output: torch.Tensor, the gradient of the loss w.r.t. the cross-section.

        Returns:
            tuple[torch.Tensor], the gradients of the loss w.r.t. each relevant input. Fitting of
            oscillator strengths not implemented.
        """
        oscillator_strengths, thermal_factors, dos = ctx.saved_tensors

        grad_inputs = thermal_factors * torch.mv(grad_output, oscillator_strengths)

        grad_thermal = torch.mv(grad_output, oscillator_strengths)
        return grad_inputs, None, grad_thermal


class NoThermalBackpropCrossSection(FullCrossSection):
    """
    Modified version of `FullCrossSection` without thermal factors in backwards pass.
    """
    @staticmethod
    def backward(ctx, grad_output):
        """Modified backwards pass without thermal factors.

        Args:
            ctx: Context object to retrieve data from forward pass.
            grad_output: torch.Tensor, the gradient of the loss w.r.t. the cross-section.

        Returns:
            tuple[torch.Tensor], the gradients of the loss w.r.t. each relevant input. Fitting of
            oscillator strengths and temperatures not implemented. Thermal factor removed in dos
            gradient.
        """
        oscillator_strengths, _ = ctx.saved_tensors
        grad_inputs = torch.mv(grad_output, oscillator_strengths)

        return grad_inputs, None, None


class VacantCrossSection(NoThermalBackpropCrossSection):
    """
    Modified version of `NoThermalBackpropCrossSection` with thermal factors also removed in forward
    pass. The dos input is expected to be the density of accessible states, with thermal factors
    included.
    """
    @staticmethod
    def forward(ctx, dos, oscillator_strengths, thermal_factors):
        ctx.save_for_backward(oscillator_strengths, thermal_factors)
        return torch.outer(
            dos, oscillator_strengths
        )  # don't include thermal factor in forward


class LorentzianProduct(Function):
    """
    Torch autograd function to calculate Lorentzian factor products. Lorentzian factors are included
    in backwards pass.
    """
    @staticmethod
    def forward(
        ctx,
        convolved: torch.Tensor,
        lorentzians: torch.Tensor,
        out_energies: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the contributions to RIXS spectra weighted by Lorentzian factors.

        Args:
            ctx: Context object to store data for backwards pass.
            convolved: torch.Tensor, the spectra components after convolving with the cross-section
                contributions.
            lorentzians: torch.Tensor, pre-calculated Lorentzian factors to account for resonances.
            out_energies: torch.Tensor, for omega_2 energy prefactor in RIXS spectra.

        Returns:
            torch.Tensor, the spectra components weighted by Lorentzian factors and energies.
        """
        ctx.save_for_backward(lorentzians, out_energies, convolved)
        lorentzian_weighting = (convolved * lorentzians).sum(axis=1)
        return lorentzian_weighting * out_energies

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        """Backwards pass through Lorentzian weighting.

        Args:
            ctx: Context object to retrieve data from forward pass.
            grad_output: torch.Tensor, the gradient of the loss w.r.t. the output.

        Returns:
            tuple[torch.Tensor], gradients of loss w.r.t. inputs of Lorentzian calculation.
        """
        lorentzians, _, convolved = ctx.saved_tensors
        # Only propagate back gradients on "convolved", the others can be added if required.
        return (lorentzians * grad_output).repeat(convolved.shape[0], 1, 1), None, None


class ModBackpropLorentzianProduct(LorentzianProduct):
    """
    Modified version of `LorentzianProduct` with Lorentzian factors switched off for backprop.
    """
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        """Backwards pass through Lorentzian weighting. Lorentzian factors are set to unity to avoid
        vanishing gradients.

        Args:
            ctx: Context object to retrieve data from forward pass.
            grad_output: torch.Tensor, the gradient of the loss w.r.t. the output.

        Returns:
            tuple[torch.Tensor], gradients of loss w.r.t. inputs of Lorentzian calculation.
        """
        lorentzians, _, convolved = ctx.saved_tensors

        num_transitions = lorentzians.shape[1]

        return (
            grad_output[:, None, :].repeat(1, num_transitions, 1),
            None,
            None,
        )  # match shape of input
