from dataclasses import dataclass


@dataclass
class TrainConfig:
    batch_size: int = 32
    num_epochs: int = 10
    lr: float = 1e-4
    loss_fn: str = "mse"
