from dataclasses import dataclass


@dataclass
class TrainConfig:
    batch_size: int = 8
    epochs: int = 10
    lr: float = 1e-4
    loss_fn: str = "mse"

    noise_std: float = 1.0

    num_workers: int = 8
    pin_memory: bool = True

    outputs_dir: str = "outputs"
