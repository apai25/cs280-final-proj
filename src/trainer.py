import os
from datetime import datetime

import torch
from tqdm import tqdm

from configs.config import Config
from src.datasets.droid_dataset import DroidDatasetIndexed as DroidDataset
from src.models.model import Model


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Data
        self.dataset = DroidDataset(
            data_path=self.cfg.data_dir,
            dataset_name=self.cfg.dataset_name,
            horizon=self.cfg.horizon,
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,  # <-- fixed typo from self.ds
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )

        # Model
        self.model = Model(self.cfg.model)
        self.model.to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.loss_fn = torch.nn.MSELoss()  # or whatever you're using

        # Init training vars, dirs
        self.ep = 0
        self.train_losses = []
        self.train_rmses = []

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.outputs_dir = os.path.join(self.cfg.outputs_dir, self.cfg.model, timestamp)
        self.checkpoint_dir = os.path.join(self.outputs_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Save configs
        torch.save(vars(self.cfg), os.path.join(self.outputs_dir, "trainer_config.pth"))

        print(f"Device: {self.device}")
        print(f"Outputs directory: {self.outputs_dir}")

    def train(self):
        print("Starting training...")
        for ep in range(self.cfg.epochs):
            self.model.train()
            self.ep = ep + 1

            train_loss = 0.0
            for data in tqdm(
                self.dataloader, desc=f"Epoch {self.ep} / {self.cfg.epochs}"
            ):
                context_obs = data["context_obs"].to(self.device)
                context_act = data["context_act"].to(self.device)
                obs = data["obs"].to(self.device)

                t = torch.rand(obs.shape[0], device=self.device)
                eps = torch.randn_like(obs, device=self.device) * self.cfg.noise_std
                x_t = t * eps + (1 - t) * obs

                self.optim.zero_grad()
                out = self.model(
                    context_obs,
                    context_act,
                    x_t,
                    t,
                )
                out.loss.backward()
                self.optim.step()

                train_loss += out.loss.item()

            avg_loss = train_loss / len(self.dataloader)
            self.train_losses.append(avg_loss)
            self.train_rmses.append((avg_loss**0.5))

            print(f"Epoch {self.ep} - Train Loss: {avg_loss:.4f}")
            self.save_model()

        torch.save(
            {"train_losses": self.train_losses, "train_rmses": self.train_rmses},
            os.path.join(self.outputs_dir, "metrics.pth"),
        )
        print("Training completed.")

    def save_model(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_{self.ep}.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
