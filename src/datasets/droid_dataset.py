import json
from collections import deque
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
import torch
from PIL import Image
from torch.utils.data import Dataset


class DroidDataset(Dataset):
    SUPPORTED_CAMERAS = [
        "wrist_image_left",
        "exterior_image_1_left",
        "exterior_image_2_left",
    ]

    def __init__(
        self,
        data_path: str,
        dataset_name: str = "droid_100",
        camera: str = "exterior_image_1_left",
        horizon: int = 4,
        img_size: tuple[int, int] = (64, 64),
    ):
        if camera not in self.SUPPORTED_CAMERAS:
            raise ValueError(f"Unsupported camera: {camera}")

        self.camera = camera
        self.horizon = horizon
        self.img_size = img_size
        self.dataset_name = dataset_name

        # Directory for all camera memmaps
        self.root = (
            Path("/raid/users/ryan_cheng")
            / f"{dataset_name}_memmap_h{horizon}_{img_size[0]}x{img_size[1]}"
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root / "meta.json"

        if not self.meta_path.exists():
            self._build_memmaps(data_path)

        with open(self.meta_path) as f:
            meta = json.load(f)
            self.N = meta["num_samples"]
            self.action_dim = meta["action_dim"]
            self.num_shards = meta["num_shards"]
            self.shard_size = meta["shard_size"]

        self.obs = []
        self.context_obs = []
        self.context_acts = []
        self.shard_offsets = []

        cam_key = self.camera
        total = 0
        for shard_idx in range(self.num_shards):
            shard_len = min(self.shard_size, self.N - total)
            self.shard_offsets.append(total)
            total += shard_len

            obs_shard = np.memmap(
                self.root / f"obs_{cam_key}_{shard_idx:03d}.dat",
                dtype="uint8",
                mode="r",
                shape=(shard_len, 3, *self.img_size),
            )
            ctx_obs_shard = np.memmap(
                self.root / f"context_obs_{cam_key}_{shard_idx:03d}.dat",
                dtype="uint8",
                mode="r",
                shape=(shard_len, self.horizon, 3, *self.img_size),
            )
            ctx_acts_shard = np.memmap(
                self.root / f"context_acts_{shard_idx:03d}.dat",
                dtype="float32",
                mode="r",
                shape=(shard_len, self.horizon, self.action_dim),
            )

            self.obs.append(obs_shard)
            self.context_obs.append(ctx_obs_shard)
            self.context_acts.append(ctx_acts_shard)

    def _resize(self, img):
        return np.array(
            Image.fromarray(img).resize(self.img_size[::-1], Image.Resampling.BILINEAR)
        )

    def _build_memmaps(self, data_path, shard_size: int = 100_000):
        print("[DroidDataset] Building memmaps from TFDS…")
        episodes = tfds.as_numpy(
            tfds.load(
                self.dataset_name, split="train", data_dir=data_path, download=False
            )
        )

        buffer = deque(maxlen=self.horizon)
        obs_by_cam = {cam: [] for cam in self.SUPPORTED_CAMERAS}
        ctx_obs_by_cam = {cam: [] for cam in self.SUPPORTED_CAMERAS}
        ctx_acts = []

        action_dim = None
        sample_count = 0
        shard_index = 0
        H, W = self.img_size

        def write_shard():
            nonlocal shard_index, sample_count
            N = len(ctx_acts)
            if N == 0:
                return

            for cam in self.SUPPORTED_CAMERAS:
                np.memmap(
                    self.root / f"obs_{cam}_{shard_index:03d}.dat",
                    mode="w+",
                    dtype="uint8",
                    shape=(N, 3, H, W),
                )[:] = np.stack(obs_by_cam[cam]).transpose(0, 3, 1, 2)
                np.memmap(
                    self.root / f"context_obs_{cam}_{shard_index:03d}.dat",
                    mode="w+",
                    dtype="uint8",
                    shape=(N, self.horizon, 3, H, W),
                )[:] = np.stack([x.transpose(0, 3, 1, 2) for x in ctx_obs_by_cam[cam]])

            np.memmap(
                self.root / f"context_acts_{shard_index:03d}.dat",
                mode="w+",
                dtype="float32",
                shape=(N, self.horizon, action_dim),
            )[:] = np.stack(ctx_acts)

            print(f"[DroidDataset] Wrote shard {shard_index} with {N} samples.")
            sample_count += N
            shard_index += 1

            # Clear for next shard
            for cam in self.SUPPORTED_CAMERAS:
                obs_by_cam[cam].clear()
                ctx_obs_by_cam[cam].clear()
            ctx_acts.clear()

        for episode in episodes:
            buffer.clear()
            for step in episode["steps"]:
                obs_all = {
                    cam: step["observation"][cam] for cam in self.SUPPORTED_CAMERAS
                }
                act = step["action"]
                if action_dim is None:
                    action_dim = act.shape[0]

                buffer.append((obs_all, act))
                if len(buffer) < self.horizon:
                    continue

                ctx_acts.append(np.stack([a for (_, a) in buffer]))
                for cam in self.SUPPORTED_CAMERAS:
                    ctx_obs = [self._resize(obs_all[cam]) for (obs_all, _) in buffer]
                    ctx_obs_by_cam[cam].append(np.stack(ctx_obs))
                    obs_by_cam[cam].append(self._resize(obs_all[cam]))

                if len(ctx_acts) >= shard_size:
                    write_shard()

        # Write any remaining data
        write_shard()

        with open(self.meta_path, "w") as f:
            json.dump({
                "num_samples": sample_count,
                "action_dim": action_dim,
                "num_shards": shard_index,
                "shard_size": shard_size
            }, f)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # Binary search to find the right shard
        shard_idx = max(i for i, offset in enumerate(self.shard_offsets) if offset <= idx)
        local_idx = idx - self.shard_offsets[shard_idx]

        obs = torch.from_numpy((self.obs[shard_idx][local_idx].astype(np.float32) / 127.5 - 1.0))
        context_obs = torch.from_numpy(
            self.context_obs[shard_idx][local_idx].astype(np.float32) / 127.5 - 1.0
        )
        context_acts = torch.from_numpy(self.context_acts[shard_idx][local_idx].copy())
        return {
            "obs": obs,
            "context_obs": context_obs,
            "context_acts": context_acts,
        }
    # def __getitem__(self, idx):
    #     obs = torch.from_numpy((self.obs[idx].astype(np.float32) / 127.5 - 1.0))
    #     context_obs = torch.from_numpy(
    #         self.context_obs[idx].astype(np.float32) / 127.5 - 1.0
    #     )
    #     context_acts = torch.from_numpy(self.context_acts[idx].copy())
    #     return {
    #         "obs": obs,
    #         "context_obs": context_obs,
    #         "context_acts": context_acts,
    #     }
