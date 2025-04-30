from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    data_path: Path = Path("data").resolve()
    dataset_name: str = "droid_100"
    horizon: int = 4

    action_dim: int = 7
