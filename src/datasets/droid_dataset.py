from collections import deque

import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from torch.utils.data import IterableDataset


class DroidDataset(IterableDataset):
    def __init__(
        self, data_path: str, dataset_name: str = "droid_100", horizon: int = 4
    ):
        self.data_path = data_path
        self.tf_ds = tfds.load(
            dataset_name, data_dir=data_path, split="train", download=False
        )
        self.tf_ds = self.tf_ds.flat_map(lambda x: x["steps"])

        self.horizon = horizon

    def __iter__(self):
        buf = deque(maxlen=self.horizon)

        for sample in self.tf_ds:
            if sample["is_first"]:
                buf.clear()

            sample = tf.nest.map_structure(lambda x: x.numpy(), sample)

            obs = (
                torch.from_numpy(sample["observation"]["exterior_image_1_left"])
                .permute(2, 0, 1)
                .float()
                / 255.0
            )
            act = torch.from_numpy(sample["action"]).float()

            if len(buf) < self.horizon:
                buf.append((obs, act))
                continue

            horizon_seq = [(pair[0], pair[1]) for pair in buf]
            buf.append((obs, act))

            yield horizon_seq, obs, act
