from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .model_conv import FlexibleCNN
from .model_dense import FCN


class JointModel(nn.Module):
    def __init__(
        self, xfel_cnn: nn.Module, rixs_cnn: nn.Module, joiner_nn: nn.Module, config=None
    ) -> None:
        super().__init__()

        self.xfel_cnn = xfel_cnn
        self.rixs_cnn = rixs_cnn
        self.joiner_nn = joiner_nn
        self.config = config

    def forward(self, X: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        xfel, rixs = X

        xfel_out = self.xfel_cnn(xfel)
        rixs_out = self.rixs_cnn(rixs)

        out = torch.cat((xfel_out, rixs_out), axis=-1)

        # xfel_out, rixs_out have shape [batch_size, output_channels, sequence_length]

        return self.joiner_nn(out)

    def get_avg_pred(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self(X).detach()
        std = out.std(axis=0)
        mean = out.mean(axis=0)
        return mean, std

    @classmethod
    def setup(cls, config):
        xfel_cnn = FlexibleCNN(
            input_channels=1,
            num_layers=config["conv_layers"],
            output_channels=config["conv_channels"],
            kernel_size=7,
            padding=3,
        )

        rixs_cnn = FlexibleCNN(
            input_channels=1,
            num_layers=config["conv_layers"],
            output_channels=config["conv_channels"],
            kernel_size=7,
            padding=3,
        )

        joiner = FCN(
            (config["conv_channels"], config["feat_count"]),
            (config["label_count"],),
            config["fcn_shape"],
        )

        return cls(xfel_cnn, rixs_cnn, joiner, config=config)

    def save(self, filename: str | Path) -> None:
        assert self.config is not None, "Can only save with config object present"

        saved_state = {
            "xfel_state_dict": self.xfel_cnn.state_dict(),
            "rixs_state_dict": self.rixs_cnn.state_dict(),
            "joiner_state_dict": self.joiner_nn.state_dict(),
            "config": self.config
        }

        with open(filename, "wb") as file:
            torch.save(saved_state, file)

    @classmethod
    def load(cls, filename: str | Path) -> JointModel:

        with open(filename, "rb") as file:
            saved_state = torch.load(file)

        joint_model = cls.setup(saved_state["config"])
        joint_model.xfel_cnn.load_state_dict(saved_state["xfel_state_dict"])
        joint_model.rixs_cnn.load_state_dict(saved_state["rixs_state_dict"])
        joint_model.joiner_nn.load_state_dict(saved_state["joiner_state_dict"])

        return joint_model