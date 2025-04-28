from dataclasses import dataclass 
from pathlib import Path 

@dataclass 
class DataConfig:
    data_path: Path = Path("data/droid_100").resolve()
