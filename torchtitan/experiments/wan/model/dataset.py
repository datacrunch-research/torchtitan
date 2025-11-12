import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

import decord


class RawVideoDataset(Dataset):
    """Dataset for loading sharded video and robot state data.

    This dataset loads video frames and corresponding robot states from sharded data files.
    Each sample contains a sequence of `clip_length` consecutive frames from the same video segment.

    Args:
        data_dir (str): Path to the data directory containing sharded data
        train (bool): Whether to load training or validation data
        clip_length (int, optional): Number of consecutive frames per sample. Defaults to 77.

    The data directory should have the following structure:
        data_dir/
        ├── metadata.json           # Overall dataset metadata
        ├── videos/                 # Sharded video data
        ├── robot_states/          # Sharded robot state data
        ├── metadata/              # Per-shard metadata
        └── segment_indices/       # Frame-to-segment mapping

    Each sample is a tuple containing:
        - Video frames tensor of shape [clip_length, H, W, C]
        - Robot states tensor of shape [clip_length, state_dim]
    """

    def __init__(
        self,
        data_dir: str,
        downsampled: int,
        clip_length: int = 77,
        repeat: int = 1,
        window_size: int = 8,
        robot_temporal_mode: str = "downsample",
    ):
        assert downsampled in [1, 4, 2], "downsampled must be 1, 4, or 2"
        self.window_size = window_size
        data_dir = Path(data_dir)
        with open(data_dir / "metadata.json", "r") as f:
            self.metadata = json.load(f)
        self.num_shards = self.metadata["num_shards"]
        self.train = "train_v2.0_raw" in str(data_dir)

        if self.train:
            self.videos_path = data_dir / "videos"
            self.robot_states_path = data_dir / "robot_states"
            self.metadata_path = data_dir / "metadata"
            self.segment_indices_path = data_dir / "segment_indices"
        else:
            self.videos_path = data_dir
            self.robot_states_path = data_dir
            self.metadata_path = data_dir
            self.segment_indices_path = data_dir

        self.clip_length = clip_length
        self.repeat = repeat
        self.downsampled = downsampled
        self.robot_temporal_mode = robot_temporal_mode

        if self.downsampled == 4:
            self.frame_idxs = list(range(clip_length))[::4] + [76]
        elif self.downsampled == 2:
            self.frame_idxs = [round(i * (77 - 1) / (41 - 1)) for i in range(41)]
            # print(f"frame_idxs: {self.frame_idxs}")
        self.num_frames_per_shard = {}

        self.clip_start_idxs = []  # (shard_idx, frame_id)
        for shard_idx in range(self.num_shards):
            with open(self.metadata_path / f"metadata_{shard_idx}.json", "r") as f:
                shard_metadata = json.load(f)
            num_frames = shard_metadata["shard_num_frames"]

            self.num_frames_per_shard[shard_idx] = num_frames

            segment_idxs = np.memmap(
                self.segment_indices_path / f"segment_idx_{shard_idx}.bin",
                dtype=np.int32,
                mode="r",
                shape=(num_frames,),
            )

            start_idx = 0
            while start_idx < num_frames:
                if start_idx + clip_length >= num_frames:
                    break

                if segment_idxs[start_idx] == segment_idxs[start_idx + clip_length]:
                    self.clip_start_idxs.append((shard_idx, start_idx))
                    start_idx = start_idx + self.window_size
                else:
                    start_idx = start_idx + 1

    def __len__(self):
        return len(self.clip_start_idxs) * self.repeat

    def __getitem__(self, idx):
        shard_idx, start_idx = self.clip_start_idxs[idx % len(self.clip_start_idxs)]

        num_frames = self.num_frames_per_shard[shard_idx]
        robot_states = np.memmap(
            self.robot_states_path / f"states_{shard_idx}.bin",
            dtype=np.float32,
            mode="r",
            shape=(num_frames, 25),
        )[start_idx : start_idx + self.clip_length]

        vr = decord.VideoReader(str(self.videos_path / f"video_{shard_idx}.mp4"))
        video_frames = vr.get_batch(
            range(start_idx, start_idx + self.clip_length)
        ).asnumpy()
        # video_frames = video_frames.permute(0, 3, 1, 2)
        if self.downsampled:
            video_frames = video_frames[self.frame_idxs]
            if self.robot_temporal_mode == "downsample":
                robot_states = robot_states[self.frame_idxs]
        video_frames = torch.from_numpy(video_frames)

        # Note: this could be put here to do all the frame preprocessing at once
        # video_frames = torch.tensor(video_frames, dtype=torch.bfloat16)
        # # Normalize video frames between -1 and 1
        # max_value = 1.
        # min_value = -1.
        # video_frames = video_frames * ((max_value - min_value) / 255.) + min_value
        # # T, H, W, C -> T, C, H, W
        # video_frames = video_frames.permute(0, 3, 1, 2)

        return (video_frames, torch.tensor(robot_states))


if __name__ == "__main__":
    train_dataset = RawVideoDataset(
        data_dir="./dataset/world_model_raw_data/train_v2.0_raw",
        downsampled=4,
    )
    print(f"train_dataset length: {len(train_dataset)}")

    val_dataset = RawVideoDataset(
        data_dir="./dataset/world_model_raw_data/val_v2.0_raw",
        downsampled=4,
    )
    print(f"val_dataset length: {len(val_dataset)}")

    indices = [0, 200, -1]

    for idx in indices:
        frames, states = train_dataset[idx]
        print(f"frames.shape: \t{frames.shape}")
        print(f"state.shape: \t{states.shape}")

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=2, num_workers=4, shuffle=True, drop_last=True
    )
    for i, batch in enumerate(train_loader):
        print(i, batch[0].shape)
        break
