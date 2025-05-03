from dataclasses import dataclass


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 10
    lr: float = 1e-4
    loss_fn: str = "mse"

    num_timesteps: int = 1000
    beta1 = 1e-4
    beta2 = 0.02

    num_workers: int = 40
    pin_memory: bool = True

    outputs_dir: str = "outputs"
