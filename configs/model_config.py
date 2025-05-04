from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    action_dim: int  # set by Config
    horizon: int  # set by Config

    # UNet config
    input_channels: int = 3  # R, G, B
    output_channels: int = 3  # R, G, B

    hidden_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    bottleneck_channels: int = 256

    dropout: float = 0.0
    batch_norm: bool = False

    pooling_kernel_size: int = 2
    pooling_stride: int = 2

    t_cond_stages: List = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])
    act_cond_stages: List = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])
    obs_cond_stages: List = field(default_factory=lambda: [0, 1, 4])

    # time_embed_dim: int = 128

    # DDPM Config
    # num_timesteps: int = 1000
    # beta1: float = 1e-4
    # beta2: float = 0.02

    # FM Config
    p_uncond: float = 0.0
