import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(MLP, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, dropout: float, batch_norm: bool
    ):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pooling_kernel_size: int,
        pooling_stride: int,
        dropout: float,
        batch_norm: bool,
    ):
        super(UpConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                in_channels,
                kernel_size=pooling_kernel_size,
                stride=pooling_stride,
            ),
            ConvBlock(in_channels, out_channels, dropout, batch_norm),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pooling_kernel_size: int,
        pooling_stride: int,
        dropout: float,
        batch_norm: bool,
    ):
        super(DownConvBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels, dropout, batch_norm),
            nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
