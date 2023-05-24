import torch
from torch import nn


# Generated with https://chat.openai.com/chat
class FlexibleCNN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_layers: int,
        output_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        cnn_layers: nn.Module = []
        in_channels = input_channels
        for _ in range(num_layers):
            cnn_layers.append(
                nn.Conv1d(
                    in_channels, output_channels, kernel_size, stride, padding, dilation
                )
            )
            cnn_layers.append(nn.ReLU())
            in_channels = output_channels
        self.cnn = nn.Sequential(*cnn_layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.unsqueeze(1)
        X = self.cnn(X)
        X = X.squeeze()
        return X
