from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class DataConfig:
    data_dir: Path = Path("data").resolve()
    # data_dir: Path = Path("/nfs/kun2/datasets").resolve()
    dataset_name: str = "droid_100"
    # dataset_name: str = "droid"
    # camera: str = "wrist_image_left"
    camera: str = "exterior_image_1_left"
    horizon: int = 5

    action_dim: int = 7
    img_size: Tuple[int, int] = (64, 64)
    img_channels: int = 3  # R, G, B
