import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from torch.utils.data import IterableDataset


class DroidDataset(IterableDataset):
    def __init__(self, data_path: str, batch_size: int = 32):
        self.data_path = data_path
        self.batch_size = batch_size
        self.tf_ds = tfds.load(
            "droid_100", data_dir=data_path, split="train", download=False
        )
        self.tf_ds = self.tf_ds.flat_map(lambda x: x["steps"])
        self.tf_ds = self.tf_ds.shuffle(10000).batch(batch_size)

    def __iter__(self):
        for batch in self.tf_ds:
            batch = tf.nest.map_structure(lambda x: x.numpy(), batch)

            images = (
                torch.from_numpy(batch["observation"]["exterior_image_1_left"])
                .permute(0, 3, 1, 2)
                .float()
                / 255.0
            )
            actions = torch.from_numpy(batch["action"]).float()

            yield images, actions
