from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    action_dim: int  # set by Config
    horizon: int  # set by Config

    input_channels: int = 3  # R, G, B
    output_channels: int = 3  # R, G, B

    hidden_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    bottleneck_channels: int = 512

    dropout: float = 0.2
    batch_norm: bool = True

    pooling_kernel_size: int = 2
    pooling_stride: int = 2

    t_cond_stages: List = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8])
    act_cond_stages: List = field(default_factory=lambda: [3, 5])
    obs_cond_stages: List = field(default_factory=lambda: [2, 6])
