"""
Program for computing the forward diff_model to insert then in neural network computations
(for the fitting of theoretical spectrum and experimental one)
"""
from typing import Callable

import torch  # for all things PyTorch
from torch import nn  # for torch.nn.Module, PyTorch models
import torch.nn.functional as F  # for the activation function
from torch.autograd import Function


class SSP(nn.Softplus):
    def __init__(self, beta: int = 1, threshold: int = 20) -> None:
        super(SSP, self).__init__(beta, threshold)

    def forward(self, input_val: torch.Tensor) -> torch.Tensor:
        sp0 = F.softplus(torch.zeros(1), self.beta, self.threshold).item()
        return F.softplus(input_val, self.beta, self.threshold) - sp0


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


class FullCrossSection(Function):
    @staticmethod
    def forward(ctx, dos, oscillator_strengths, thermal_factors):
        ctx.save_for_backward(oscillator_strengths, thermal_factors)
        return torch.outer(thermal_factors * dos, oscillator_strengths)

    @staticmethod
    def backward(ctx, grad_output):
        oscillator_strengths, thermal_factors = ctx.saved_tensors

        grad_inputs = thermal_factors * torch.mv(grad_output, oscillator_strengths)

        return grad_inputs, None, None


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
