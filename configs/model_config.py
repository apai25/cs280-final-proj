from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    action_dim: int  # set by Config
    horizon: int  # set by Config
    img_channels: int # set by Config

    # UNet config
    hidden_channels: List[int] = field(default_factory=lambda: [128, 256, 512])
    bottleneck_channels: int = 1024

    dropout: float = 0.0
    batch_norm: bool = True

    pooling_kernel_size: int = 2
    pooling_stride: int = 2

    t_cond_stages: List = field(default_factory=lambda: [4])
    act_cond_stages: List = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])
    obs_cond_stages: List = field(default_factory=lambda: [0])

    # FM Config
    p_uncond: float = 0.0
