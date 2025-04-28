from dataclasses import dataclass  

from configs.data_config import DataConfig

@dataclass 
class Config:
    data: DataConfig = DataConfig()
