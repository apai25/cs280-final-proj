import os
from datetime import datetime

import torch
from tqdm import tqdm

from configs.config import Config
from src.datasets.droid_dataset import DroidDataset
from src.models.fm import FM


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Data
        self.dataset = DroidDataset(
            data_path=cfg.data.data_dir,
            dataset_name=cfg.data.dataset_name,
            camera=cfg.data.camera,
            horizon=cfg.data.horizon,
            img_size=cfg.data.img_size,
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=cfg.train.num_workers,
            pin_memory=cfg.train.pin_memory,
        )

        # Model
        self.fm = FM(cfg.model).to(self.device)
        self.optim = torch.optim.Adam(self.fm.parameters(), lr=cfg.train.init_lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim,
            T_max=self.cfg.train.epochs * len(self.dataloader),
            eta_min=cfg.train.min_lr,
        )
        self.loss_fn = torch.nn.MSELoss()

        # Init training vars, dirs
        self.ep = 0
        self.train_losses = []

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.outputs_dir = os.path.join(cfg.train.outputs_dir, timestamp)
        self.checkpoint_dir = os.path.join(self.outputs_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Save configs
        torch.save(
            vars(cfg.train), os.path.join(self.outputs_dir, "trainer_config.pth")
        )
        torch.save(vars(cfg.model), os.path.join(self.outputs_dir, "model_config.pth"))
        torch.save(vars(cfg.data), os.path.join(self.outputs_dir, "data_config.pth"))

        print(f"Device: {self.device}")
        print(f"Outputs directory: {self.outputs_dir}")

    def train(self):
        print("Starting training...")
        for ep in range(self.cfg.train.epochs):
            self.fm.train()
            self.ep = ep + 1

            train_loss = 0.0
            for data in tqdm(
                self.dataloader, desc=f"Epoch {self.ep} / {self.cfg.train.epochs}"
            ):
                B = data["context_obs"].shape[0]

                context_obs = (
                    data["context_obs"]
                    .to(self.device)
                    .view(
                        B,
                        -1,
                        data["context_obs"].shape[-2],
                        data["context_obs"].shape[-1],
                    )
                )
                context_acts = data["context_acts"].to(self.device).view(B, -1)
                x_1 = data["obs"].to(self.device)
                x_0 = torch.randn_like(x_1)

                t = torch.rand((B, 1), device=self.device)

                u_t = self.fm(
                    x_1=x_1,
                    x_0=x_0,
                    t=t,
                    context_acts=context_acts,
                    context_obs=context_obs,
                )

                loss = self.loss_fn(u_t, x_1 - x_0)
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.fm.parameters(), self.cfg.train.grad_clip)
                self.optim.step()
                self.scheduler.step()

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
        torch.save(self.fm.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
