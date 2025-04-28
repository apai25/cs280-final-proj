import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from torch.utils.data import IterableDataset

class DroidDataset(IterableDataset):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.tf_ds = tfds.load(
            "droid_100", data_dir=data_path, split="train", download=False
        )
        self.tf_ds = self.tf_ds.flat_map(lambda x: x["steps"])

    def __iter__(self):
        prev_obs = None
        prev_act = None

        for sample in self.tf_ds:
            sample = tf.nest.map_structure(lambda x: x.numpy(), sample)

            obs = torch.from_numpy(sample["observation"]["exterior_image_1_left"]).permute(2, 0, 1).float() / 255.0
            act = torch.from_numpy(sample["action"]).float()

            if prev_obs is not None and prev_act is not None:
                yield prev_obs, prev_act, obs

            prev_obs = obs
            prev_act = act
