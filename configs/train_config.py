from dataclasses import dataclass


@dataclass
class TrainConfig:
    batch_size: int = 256
    epochs: int = 50
    init_lr: float = 1e-3
    min_lr: float = 1e-5
    loss_fn: str = "mse"

    grad_clip: float = 1.0

    num_workers: int = 64
    pin_memory: bool = True

    outputs_dir: str = "outputs"
    save_every: int = 10
