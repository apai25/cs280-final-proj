from collections import deque

import tensorflow_datasets as tfds
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class DroidDatasetIndexed(Dataset):
    def __init__(
        self,
        data_path: str,
        dataset_name: str = "droid_100",
        horizon: int = 4,
        img_size: tuple[int, int] = (64, 64),
    ):
        self.horizon = horizon
        self.samples = []
        self.img_size = img_size

        episodes = tfds.as_numpy(
            tfds.load(dataset_name, data_dir=data_path, split="train", download=False)
        )

        for episode in episodes:
            buf = deque(maxlen=self.horizon)

            for step in episode["steps"]:
                obs = step["observation"]["wrist_image_left"]
                act = step["action"]

                if len(buf) == self.horizon:
                    context_obs = [o for o, _ in buf]
                    context_act = [a for _, a in buf]
                    self.samples.append(
                        {
                            "context_obs": context_obs,
                            "context_acts": context_act,
                            "obs": obs,
                        }
                    )

                buf.append((obs, act))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        context_obs = torch.stack(
            [
                F.interpolate(
                    (
                        torch.from_numpy(o).permute(2, 0, 1).float() / 127.5 - 1.0
                    ).unsqueeze(0),
                    size=self.img_size,
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                ).squeeze(0)
                for o in sample["context_obs"]
            ]
        )

        context_acts = torch.stack(
            [torch.from_numpy(a).float() for a in sample["context_acts"]]
        )

        obs = torch.from_numpy(sample["obs"]).permute(2, 0, 1).float()
        obs = obs / 127.5 - 1.0
        obs = F.interpolate(
            obs.unsqueeze(0),
            size=self.img_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)

        return {"context_obs": context_obs, "context_acts": context_acts, "obs": obs}
