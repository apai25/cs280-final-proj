from configs.config import Config
from src.trainer import Trainer

if __name__ == "__main__":
    cfg = Config()
    trainer = Trainer(cfg)
    trainer.train()
