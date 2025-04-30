from dataclasses import dataclass, field

from configs.data_config import DataConfig
from configs.model_config import ModelConfig
from configs.train_config import TrainConfig


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(init=False)

    def __post_init__(self):
        self.model = ModelConfig(
            input_channels=self.data.horizon * 3,
            action_dim=self.data.action_dim,
            loss_fn=self.train.loss_fn,
        )
