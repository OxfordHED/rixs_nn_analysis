import torch
import torch.nn as nn


class FCN(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int],
        output_shape: tuple[int],
        hidden_sizes: tuple[int],
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        # Compute the flattened input and output sizes
        input_size = torch.prod(torch.tensor(input_shape))
        output_size = torch.prod(torch.tensor(output_shape))

        # Define the layers of the network
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))

        # Define the network as a sequence of layers
        self.network = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Flatten the input tensor
        flattened_input = X.view(X.size(0), -1)

        # Pass the flattened input through the network
        output = self.network(flattened_input)

        # Reshape the output tensor to match the output shape
        output = output.view(X.size(0), *self.output_shape)

        return output
