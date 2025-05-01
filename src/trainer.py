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
            data_path=self.cfg.data.data_dir,
            dataset_name=self.cfg.data.dataset_name,
            horizon=self.cfg.data.horizon,
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,  # <-- fixed typo from self.ds
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=self.cfg.train.pin_memory,
        )

        # Model
        self.model = Model(self.cfg.model)
        self.model.to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.train.lr)
        self.loss_fn = torch.nn.MSELoss()

        # Init training vars, dirs
        self.ep = 0
        self.train_losses = []

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.outputs_dir = os.path.join(self.cfg.train.outputs_dir, timestamp)
        self.checkpoint_dir = os.path.join(self.outputs_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Save configs
        torch.save(
            vars(self.cfg.train), os.path.join(self.outputs_dir, "trainer_config.pth")
        )
        torch.save(
            vars(self.cfg.model), os.path.join(self.outputs_dir, "model_config.pth")
        )
        torch.save(
            vars(self.cfg.data), os.path.join(self.outputs_dir, "data_config.pth")
        )

        print(f"Device: {self.device}")
        print(f"Outputs directory: {self.outputs_dir}")

    def train(self):
        print("Starting training...")
        for ep in range(self.cfg.train.epochs):
            self.model.train()
            self.ep = ep + 1

            train_loss = 0.0
            for data in tqdm(
                self.dataloader, desc=f"Epoch {self.ep} / {self.cfg.train.epochs}"
            ):
                context_obs = data["context_obs"].to(self.device)
                context_obs = context_obs.view(
                    context_obs.shape[0], -1, context_obs.shape[3], context_obs.shape[4]
                )
                context_acts = data["context_act"].to(self.device)
                context_acts = context_acts.view(context_acts.shape[0], -1)
                x_1 = data["obs"].to(self.device)

                t = torch.rand((x_1.shape[0], 1), device=self.device)
                x_0 = (
                    torch.randn_like(x_1, device=self.device) * self.cfg.train.noise_std
                )
                x_t = (
                    t.view(t.shape[0], 1, 1, 1) * x_0
                    + (1 - t).view(t.shape[0], 1, 1, 1) * x_1
                )

                self.optim.zero_grad()
                u_t = self.model(
                    x_t,
                    t,
                    context_acts,
                    context_obs,
                )

                loss = self.loss_fn(u_t, (x_1 - x_0))
                loss.backward()
                self.optim.step()

                train_loss += loss.item()

            avg_loss = train_loss / len(self.dataloader)
            self.train_losses.append(avg_loss)

            print(f"Epoch {self.ep} - Train Loss: {avg_loss:.4f}")
            self.save_model()

        torch.save(
            {"train_losses": self.train_losses},
            os.path.join(self.outputs_dir, "metrics.pth"),
        )
        print("Training completed.")

    def save_model(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_{self.ep}.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
