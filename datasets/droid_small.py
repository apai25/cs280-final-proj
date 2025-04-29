import os
import glob
import torch
import torchvision
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset

class DroidDataset(Dataset):
    def __init__(self, k: int = 4, frame_size: tuple[int, int] = None, dir: str = "data/droid_small", video_source: str = "observation.images.wrist_left"):
        """
        Args:
            k: Number of past frames to condition on (including current frame).
            dir: Root directory of droid dataset.
            video_source: Video source name for view of robot.
        """
        self.dir = dir
        self.k = k
        self.video_source = video_source
        self.frame_size = (64, 64) if frame_size is None else frame_size
        self.resize_frame = False if frame_size is None else True

        # Collect data
        self.samples = []

        data_paths = glob.glob(os.path.join(self.dir, "data", "**", "*.parquet"), recursive=True)
        for data_path in sorted(data_paths):
            df = pd.read_parquet(data_path)

            path_parts = os.path.normpath(data_path).split(os.sep)
            chunk_name = path_parts[-2]
            file_name = path_parts[-1]
            chunk_index = int(chunk_name.split("-")[1])
            file_index = int(file_name.split("-")[1].replace(".parquet", ""))

            for idx, row in df.iterrows():
                sample = {
                    "chunk_index": chunk_index,
                    "file_index": file_index,
                    "frame_index": row["index"],
                    "episode_index": row["episode_index"],
                    "episode_frame_index": row["frame_index"],
                    "task_index": row["task_index"],
                    "is_episode_end": row["is_last"],
                    "action": row["action"],
                }
                self.samples.append(sample)

        # Collect valid indices (start of window)
        self.items = []
        for idx in range(len(self.samples) - self.k):
            valid = True
            for ki in range(self.k):
                if self.samples[idx + ki]["is_episode_end"]:
                    valid = False
                    break
            if valid:
                self.items.append(idx)

        # Video frame loading and processing
        self.video_cache = {}
        self.resize = torchvision.transforms.Resize(self.frame_size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sample_idx = self.items[idx]
        start_sample = self.samples[sample_idx]

        chunk_idx, file_idx = start_sample["chunk_index"], start_sample["file_index"]
        video_path = os.path.join(
            self.dir, "videos", self.video_source, 
            f"chunk-{chunk_idx:03d}", f"file-{file_idx:03d}.mp4"
        )
        start_frame_idx = start_sample["frame_index"]

        frames = self._load_k_frames(video_path, start_frame_idx, self.k + 1)
        frames_cond, frame_cur, frame_next = frames[:-1], frames[-2], frames[-1]

        actions = []
        for i in range(self.k):
            action_i = self.samples[sample_idx + i]["action"]
            actions.append(torch.tensor(action_i, dtype=torch.float32))
        actions = torch.stack(actions, dim=0)  # shape: (k, action_dim)

        label = frame_next - frame_cur

        return frames_cond, actions, label, frame_next

    def _get_video_capture(self, video_path: str):
        if video_path not in self.video_cache:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video: {video_path}")
            self.video_cache[video_path] = cap
        return self.video_cache[video_path]

    def _load_k_frames(self, video_path: str, start_idx: int, k: int):
        cap = self._get_video_capture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        frames = []
        for _ in range(k):
            ret, frame = cap.read()
            if not ret:
                raise IOError(f"Failed to read frame {start_idx} from {video_path}")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
            if self.resize_frame:
                frame_tensor = self.resize(frame_tensor)
            frames.append(frame_tensor)
        return torch.stack(frames, dim=0)

    def close(self):
        for cap in self.video_cache.values():
            cap.release()
        self.video_cache.clear()