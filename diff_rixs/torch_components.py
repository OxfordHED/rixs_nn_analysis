"""
Program for computing the forward diff_model to insert then in neural network computations
(for the fitting of theoretical spectrum and experimental one)
"""
from typing import Callable

import numpy as np
import torch  # for all things PyTorch
from torch import nn  # for torch.nn.Module, PyTorch models
import torch.nn.functional as F  # for the activation function
from torch.autograd import Function
import matplotlib.pyplot as plt


class SSP(nn.Softplus):
    def __init__(self, beta: int = 1, threshold: int = 20) -> None:
        super(SSP, self).__init__(beta, threshold)

    def forward(self, input_val: torch.Tensor) -> torch.Tensor:
        sp0 = F.softplus(torch.zeros(1), self.beta, self.threshold).item()
        return F.softplus(input_val, self.beta, self.threshold) - sp0

PREFACTOR_SINE = 30 # Paper uses 30

# From https://arxiv.org/pdf/2006.09661.pdf
class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(PREFACTOR_SINE * input)

# From https://arxiv.org/pdf/2006.09661.pdf
def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(
                -np.sqrt(6 / num_input) / PREFACTOR_SINE,
                np.sqrt(6 / num_input) / PREFACTOR_SINE
            )

# From https://arxiv.org/pdf/2006.09661.pdf
def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class FunctionEstimator(nn.Module):
    def __init__(
        self, activation: Callable, num_hidden_units: int = 40, num_layers: int = 4
    ) -> None:
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
        x = self.fc1(x)
        x = self.activation(x)
        for fc in self.fc2:
            x = fc(x)
            x = self.activation(x)
        x = self.fc3(x)
        return x

# From https://arxiv.org/pdf/2006.09661.pdf
class Siren(FunctionEstimator):

    def __init__(self, num_hidden_units: int = 40, num_layers: int = 4):
        super().__init__(Sine(), num_hidden_units, num_layers)
        self.fc1.apply(first_layer_sine_init)
        self.fc2.apply(sine_init)
        # self.fc3.apply(sine_init)


class FullCrossSection(Function):
    @staticmethod
    def forward(ctx, dos, oscillator_strengths, thermal_factors):
        ctx.save_for_backward(oscillator_strengths, thermal_factors, dos)
        return torch.outer(thermal_factors * dos, oscillator_strengths)

    @staticmethod
    def backward(ctx, grad_output):
        oscillator_strengths, thermal_factors, dos = ctx.saved_tensors

        grad_inputs = thermal_factors * torch.mv(grad_output, oscillator_strengths)

        grad_thermal = torch.mv(grad_output, oscillator_strengths)
        return grad_inputs, None, grad_thermal


class NoThermalBackpropCrossSection(FullCrossSection):
    @staticmethod
    def backward(ctx, grad_dos):
        oscillator_strengths, _ = ctx.saved_tensors
        grad_inputs = torch.mv(grad_dos, oscillator_strengths)

        return grad_inputs, None, None


class VacantCrossSection(
    NoThermalBackpropCrossSection
):  # don't include thermal factor in backprop
    @staticmethod
    def forward(ctx, dos, oscillator_strengths, thermal_factors):
        ctx.save_for_backward(oscillator_strengths, thermal_factors)
        return torch.outer(
            dos, oscillator_strengths
        )  # don't include thermal factor in forward


class LorentzianProduct(Function):
    @staticmethod
    def forward(ctx, convolved, lorentzians, out_energies):
        ctx.save_for_backward(lorentzians, out_energies, convolved)
        lorentzian_weighting = (convolved * lorentzians).sum(axis=1)
        return lorentzian_weighting * out_energies

    @staticmethod
    def backward(ctx, grad_output):
        lorentzians, _, convolved = ctx.saved_tensors
        return (lorentzians * grad_output).repeat(convolved.shape[0], 1, 1), None, None


class ModBackpropLorentzianProduct(LorentzianProduct):
    @staticmethod
    def backward(ctx, grad_output):
        lorentzians, _, convolved = ctx.saved_tensors

        num_transitions = lorentzians.shape[1]

        return (
            grad_output[:, None, :].repeat(1, num_transitions, 1),
            None,
            None,
        )  # match shape of input
