import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Simple residual block used by the policy network."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        return torch.relu(x)


class PolicyNet(nn.Module):
    """
    Convolutional policy network that outputs logits over the 4096-from-to
    chess move encoding used in chess_env_v2.py.
    """

    def __init__(
        self,
        input_shape=(13, 8, 8),
        num_actions: int = 4096,
        channels: int = 64,
        num_res_blocks: int = 3,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.conv_in = nn.Conv2d(input_shape[0], channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(channels)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )

        flat_size = channels * input_shape[1] * input_shape[2]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)
        logits = self.head(x)
        return logits
