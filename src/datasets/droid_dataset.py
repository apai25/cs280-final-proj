from collections import deque

import tensorflow_datasets as tfds
import torch
from torch.utils.data import Dataset, IterableDataset


class DroidDatasetIterable(IterableDataset):
    def __init__(
        self, data_path: str, dataset_name: str = "droid_100", horizon: int = 4
    ):
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.horizon = horizon

        self.episodes = tfds.as_numpy(
            tfds.load(dataset_name, data_dir=data_path, split="train", download=False)
        )

    def __iter__(self):
        for episode in self.episodes:
            buf = deque(maxlen=self.horizon)

            for step in episode["steps"]:
                obs = (
                    torch.from_numpy(step["observation"]["exterior_image_1_left"])
                    .permute(2, 0, 1)
                    .float()
                    / 255.0
                )
                act = torch.from_numpy(step["action"]).float()

                if len(buf) == self.horizon:
                    context = list(buf)
                    context_obs = torch.stack([obs for obs, _ in context], dim=0)
                    context_act = torch.stack([act for _, act in context], dim=0)

                    yield {
                        "context_obs": context_obs,
                        "context_act": context_act,
                        "obs": obs,
                    }

                buf.append((obs, act))


class DroidDatasetIndexed(Dataset):
    def __init__(
        self, data_path: str, dataset_name: str = "droid_100", horizon: int = 4
    ):
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.horizon = horizon

        self.samples = []
        self.episodes = tfds.as_numpy(
            tfds.load(dataset_name, data_dir=data_path, split="train", download=False)
        )

        for episode in self.episodes:
            buf = deque(maxlen=self.horizon)

            for step in episode["steps"]:
                obs = (
                    torch.from_numpy(step["observation"]["exterior_image_1_left"])
                    .permute(2, 0, 1)
                    .float()
                    / 255.0
                )
                act = torch.from_numpy(step["action"]).float()

                if len(buf) == self.horizon:
                    context_obs = torch.stack([o for o, _ in buf], dim=0)
                    context_act = torch.stack([a for _, a in buf], dim=0)
                    self.samples.append(
                        {
                            "context_obs": context_obs,
                            "context_act": context_act,
                            "obs": obs,
                        }
                    )

                buf.append((obs, act))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
