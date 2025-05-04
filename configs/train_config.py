from dataclasses import dataclass


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 50
    init_lr: float = 1e-3
    min_lr: float = 1e-5
    loss_fn: str = "mse"

    num_workers: int = 64
    pin_memory: bool = True

    outputs_dir: str = "outputs"
