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
        self.horizon = horizon
        self.samples = []

        episodes = tfds.as_numpy(
            tfds.load(dataset_name, data_dir=data_path, split="train", download=False)
        )

        for episode in episodes:
            buf = deque(maxlen=self.horizon)

            for step in episode["steps"]:
                obs = step["observation"]["exterior_image_1_left"]
                act = step["action"]

                if len(buf) == self.horizon:
                    context_obs = [o for o, _ in buf]
                    context_act = [a for _, a in buf]
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
        sample = self.samples[idx]

        context_obs = torch.stack(
            [
                torch.from_numpy(o).permute(2, 0, 1).float() / 255.0
                for o in sample["context_obs"]
            ]
        )
        context_act = torch.stack(
            [torch.from_numpy(a).float() for a in sample["context_act"]]
        )
        obs = torch.from_numpy(sample["obs"]).permute(2, 0, 1).float() / 255.0

        return {"context_obs": context_obs, "context_act": context_act, "obs": obs}
