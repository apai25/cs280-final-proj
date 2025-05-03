from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class DataConfig:
    data_dir: Path = Path("data").resolve()
    dataset_name: str = "droid_100"
    horizon: int = 4

    action_dim: int = 7
    img_size: Tuple[int, int] = (64, 64)
