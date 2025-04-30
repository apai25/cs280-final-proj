from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    input_channels: int  # horizon x 3
    output_channels: int = 3  # R, G, B

    hidden_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    bottleneck_channels: int = 512

    dropout: float = 0.2
    batch_norm: bool = True

    pooling_kernel_size: int = 2
    pooling_stride: int = 2

    cond_stages: List[int] = field(
        default_factory=lambda: [3, 5]
    )  # enc/dec layer whose input
    # should be conditioned on diffusion timestep + act (0-indexed)
