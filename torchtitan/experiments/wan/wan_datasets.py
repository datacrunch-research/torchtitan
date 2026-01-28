# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle
from pathlib import Path
# import icecream
from typing import Any, Callable, Optional

import torch


from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

# from typing import Any

from torchtitan.components.dataloader import BaseDataLoader

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.experiments.wan.model.dataset import (
    RawVideoDataset as BaseRawVideoDataset,
)
from torchtitan.experiments.wan.tokenizer import build_wan_tokenizer, WanTokenizer
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.tools.logging import logger


class ParallelAwareDataloader(StatefulDataLoader, BaseDataLoader):
    """Dataloader that is aware of distributed data parallelism.

    This dataloader is used to load data in a distributed data parallel fashion. It also
    utilizes ``torchdata.stateful_dataloader.StatefulDataLoader`` to implement the necessary
    methods such as ``__iter__``.

    Args:
        dataset (IterableDataset): The dataset to iterate over.
        dp_rank: Data parallelism rank for this dataloader.
        dp_world_size: The world size of the data parallelism.
        batch_size: The batch size to use for each iteration.
        collate_fn: Optional function to collate samples in a batch.
        num_workers: Number of worker processes for data loading.
        persistent_workers: Whether to keep workers alive across epochs.
        pin_memory: Whether to pin memory for faster GPU transfer.
        prefetch_factor: Number of batches each worker prefetches ahead.
    """

    dp_rank: int
    dp_world_size: int
    batch_size: int

    def __init__(
        self,
        dataset: IterableDataset,
        dp_rank: int,
        dp_world_size: int,
        batch_size: int,
        collate_fn: Callable | None = None,
        num_workers: int = 0,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
    ):
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.batch_size = batch_size
        # Pass DataLoader parameters to StatefulDataLoader which will forward them to the underlying DataLoader
        super().__init__(
            dataset,
            batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )
        self._rank_id = f"dp_rank_{dp_rank}"

    def state_dict(self) -> dict[str, Any]:
        """Store state only for dp rank to avoid replicating the same state across other dimensions."""
        return {
            # We don't have to use pickle as DCP will serialize the state_dict. However,
            # we have to keep this for backward compatibility.
            self._rank_id: pickle.dumps(super().state_dict()),
            "world_size": self.dp_world_size,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dict for this dataloader."""
        # State being empty is valid.
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(
                f"DataLoader state is empty for dp rank {self.dp_rank}, "
                f"expected key {self._rank_id}"
            )
            return

        assert self.dp_world_size == state_dict["world_size"], (
            "dp_degree is inconsistent before and after checkpoint, "
            "dataloader resharding is not supported yet."
        )
        # We don't have to use pickle as DCP will serialize the state_dict. However, we have to
        # keep this for backward compatibility.
        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))


def _load_raw_video_dataset(
    dataset_path: str,
    downsampled: int = 4,
    repeat: int = 1,
    window_size: int = 8,
    robot_temporal_mode: str = "downsample",
) -> BaseRawVideoDataset:
    """
    Load RawVideoDataset from local directory.

    Args:
        dataset_path: Path to the dataset directory
        downsampled: Downsampling factor (1, 2, or 4)
        repeat: Number of times to repeat the dataset
        robot_temporal_mode: How to handle robot state temporal alignment

    Returns:
        RawVideoDataset instance
    """
    logger.info(f"Loading RawVideoDataset from {dataset_path}")
    return BaseRawVideoDataset(
        data_dir=dataset_path,
        downsampled=downsampled,
        repeat=repeat,
        window_size=window_size,
        robot_temporal_mode=robot_temporal_mode,
    )


def _process_raw_video_sample(
    sample: tuple[torch.Tensor, torch.Tensor],
    t5_tokenizer: WanTokenizer,
    job_config: Optional[JobConfig] = None,
    t5_empty_tokens: Optional[list[int]] = None,
) -> dict[str, Any]:
    """
    Process RawVideoDataset sample (video frames and robot states).

    Args:
        sample: Tuple of (video_frames, robot_states)
            - video_frames: Tensor of shape [T, H, W, C]
            - robot_states: Tensor of shape [T, state_dim]
        t5_tokenizer: T5 tokenizer for text encoding
        job_config: Job configuration (optional, for accessing config params)
        t5_empty_tokens: Precomputed empty string tokens for T5 (optional, for efficiency)

    Returns:
        Dictionary with processed video frames, robot states, and tokens
    """
    video_frames, robot_states = sample

    # For video datasets, we might not have text prompts
    # Use empty string or generate from robot states if needed
    # For now, using empty prompt - you can customize this
    prompt = ""  # Can be customized based on robot states or other metadata

    # Use precomputed empty tokens if provided, otherwise encode on the fly
    if t5_empty_tokens is not None:
        t5_tokens = t5_empty_tokens
    else:
        t5_tokens = t5_tokenizer.encode(prompt) if prompt else t5_tokenizer.encode("")

    return {
        "video_frames": video_frames,  # Shape: [T, H, W, C]
        "robot_states": robot_states,  # Shape: [T, state_dim]
        "t5_tokens": t5_tokens,
        "prompt": prompt,
    }


DATASETS = {
    "1xwm": DatasetConfig(
        path="",  # Path will be provided via dataset_path in config
        loader=_load_raw_video_dataset,
        sample_processor=_process_raw_video_sample,
    ),
}


def _validate_dataset(
    dataset_name: str, dataset_path: Optional[str] = None
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path."""
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.sample_processor


class LatentDatasetWrapper(IterableDataset, Stateful):
    """
    Dataset that loads pre-encoded VAE latents from disk.

    This is much faster than encoding videos on-the-fly since it avoids
    VAE forward passes entirely.

    Each latent file: latent_XXXXXXXX.pt containing tensor (C, T, H, W) = (48, 20, 32, 32)
    Robot states are loaded from the raw dataset for conditioning.

    Args:
        latents_dir: Path to directory containing latent_XXXXXXXX.pt files
        raw_dataset_path: Path to raw dataset (for robot_states)
        t5_tokenizer: T5 tokenizer for text encoding
        job_config: Job configuration
        dp_rank: Data parallel rank
        dp_world_size: Data parallel world size
        infinite: Whether to loop infinitely
    """

    def __init__(
        self,
        latents_dir: str,
        raw_dataset_path: str,
        t5_tokenizer: BaseTokenizer,
        job_config: JobConfig,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = True,
    ) -> None:
        logger.info("=" * 60)
        logger.info("Initializing LatentDatasetWrapper")
        logger.info("=" * 60)
        logger.info(f"  Latents directory: {latents_dir}")
        logger.info(f"  Raw dataset path: {raw_dataset_path}")
        logger.info(f"  DP rank: {dp_rank}, DP world size: {dp_world_size}")

        self.latents_dir = Path(latents_dir)
        self.job_config = job_config
        self.infinite = infinite

        # Discover available latent files and extract indices
        logger.info("  Scanning for latent files...")
        latent_files = sorted(self.latents_dir.glob("latent_*.pt"))
        all_indices = []
        for f in latent_files:
            # Extract index from filename like "latent_00000123.pt"
            idx_str = f.stem.split("_")[1]
            all_indices.append(int(idx_str))

        self._dataset_len = len(all_indices)
        logger.info(f"  Found {self._dataset_len} latent files")

        # Split indices across data parallel ranks
        indices_per_rank = self._dataset_len // dp_world_size
        start_idx = dp_rank * indices_per_rank
        end_idx = start_idx + indices_per_rank if dp_rank < dp_world_size - 1 else self._dataset_len

        self._all_indices = all_indices
        self._start_idx = start_idx
        self._end_idx = end_idx
        self._my_indices = all_indices[start_idx:end_idx]
        logger.info(f"  DP sharding: rank {dp_rank} gets indices [{start_idx}:{end_idx}] ({len(self._my_indices)} samples)")

        # Load raw dataset for robot_states only
        # We need the same clip extraction logic to get aligned robot states
        logger.info("  Loading raw dataset for robot_states...")
        downsampled = getattr(job_config.training, "downsampled", 4)
        clip_length = getattr(job_config.training, "clip_length", 77)
        window_size = getattr(job_config.training, "window_size", 8)
        robot_temporal_mode = getattr(job_config.training, "robot_temporal_mode", "downsample")

        self._raw_dataset = BaseRawVideoDataset(
            data_dir=raw_dataset_path,
            downsampled=downsampled,
            clip_length=clip_length,
            window_size=window_size,
            robot_temporal_mode=robot_temporal_mode,
        )
        logger.info(f"  Raw dataset loaded: {len(self._raw_dataset)} samples")

        # Tokenizer for T5 (precompute empty string tokens)
        self._t5_tokenizer = t5_tokenizer
        self._t5_empty_token = t5_tokenizer.encode("")

        # Checkpoint state
        self._sample_idx = 0  # Index within _my_indices

        logger.info("  LatentDatasetWrapper initialization complete")
        logger.info("=" * 60)

    def _get_data_iter(self):
        """Get iterator over indices for this rank."""
        if self._sample_idx >= len(self._my_indices):
            return iter([])
        return iter(range(self._sample_idx, len(self._my_indices)))

    def __iter__(self):
        """Iterate over the dataset."""
        while True:
            indices_iter = self._get_data_iter()

            for local_idx in indices_iter:
                global_idx = self._my_indices[local_idx]

                # Load pre-encoded latent from disk
                latent_path = self.latents_dir / f"latent_{global_idx:08d}.pt"
                latent = torch.load(latent_path, weights_only=True)

                # Load robot_states from raw dataset
                # The raw dataset returns (video_frames, robot_states)
                _, robot_states = self._raw_dataset[global_idx]

                # Build sample dict (similar to RawVideoDatasetWrapper)
                sample_dict = {
                    "latents": latent,  # Shape: (C, T, H, W) = (48, 20, 32, 32)
                    "robot_states": robot_states,
                    "t5_tokens": self._t5_empty_token,
                    "sample_idx": global_idx,
                }

                self._sample_idx = local_idx + 1

                # Yield sample_dict and latent (latent serves as "labels" placeholder)
                # Note: For latent dataset, we don't have video_frames, so we pass latent
                yield sample_dict, latent

            if not self.infinite:
                logger.warning(
                    "Latent dataset has run out of data. "
                    "This might cause NCCL timeout if data parallelism is enabled."
                )
                break
            else:
                # Reset for next iteration
                self._sample_idx = 0
                logger.warning("Latent dataset is being re-looped.")
                continue

    def load_state_dict(self, state_dict):
        """Load checkpoint state."""
        self._sample_idx = state_dict.get("sample_idx", 0)
        # Ensure sample_idx is within valid range
        if self._sample_idx < 0:
            self._sample_idx = 0
        if self._sample_idx >= len(self._my_indices):
            self._sample_idx = len(self._my_indices) - 1

    def state_dict(self):
        """Save checkpoint state."""
        return {"sample_idx": self._sample_idx}

    def __len__(self):
        return self._dataset_len


class RawVideoDatasetWrapper(IterableDataset, Stateful):
    """
    Wrapper for RawVideoDataset to make it compatible with IterableDataset pattern.

    This wraps the map-style RawVideoDataset (PyTorch Dataset) and makes it work
    as an IterableDataset for distributed training.

    Args:
        dataset_name: Name of the dataset
        dataset_path: Path to the dataset directory
        t5_tokenizer: T5 tokenizer
        job_config: Job configuration
        dp_rank: Data parallel rank
        dp_world_size: Data parallel world size
        infinite: Whether to loop infinitely
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
        t5_tokenizer: BaseTokenizer,
        job_config: Optional[JobConfig] = None,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, data_processor = _validate_dataset(
            dataset_name, dataset_path
        )

        # Load the RawVideoDataset with parameters from job_config if available
        if job_config and hasattr(job_config.training, "downsampled"):
            downsampled = getattr(job_config.training, "downsampled", 4)
        else:
            downsampled = 4

        if job_config and hasattr(job_config.training, "window_size"):
            window_size = getattr(job_config.training, "window_size", 8)
        else:
            window_size = 8

        if job_config and hasattr(job_config.training, "robot_temporal_mode"):
            robot_temporal_mode = getattr(
                job_config.training, "robot_temporal_mode", "downsample"
            )
        else:
            robot_temporal_mode = "downsample"

        # Create the underlying RawVideoDataset
        # The loader for 1x-wmds accepts additional parameters
        if dataset_name == "1xwm":
            # ic(downsampled)
            # ic(window_size)
            raw_dataset = dataset_loader(
                path,
                downsampled=downsampled,
                window_size=window_size,
                robot_temporal_mode=robot_temporal_mode,
            )
        else:
            # For HuggingFace datasets, loader only takes path
            raw_dataset = dataset_loader(path)

        # Split dataset across data parallel ranks
        # For map-style datasets, we need to manually split indices
        dataset_len = len(raw_dataset)
        indices_per_rank = dataset_len // dp_world_size
        start_idx = dp_rank * indices_per_rank
        end_idx = (
            start_idx + indices_per_rank if dp_rank < dp_world_size - 1 else dataset_len
        )

        self.dataset_name = dataset_name
        self._raw_dataset = raw_dataset
        self._start_idx = start_idx
        self._end_idx = end_idx
        self._dataset_len = dataset_len

        self._t5_tokenizer = t5_tokenizer
        self._t5_empty_token = t5_tokenizer.encode("")
        self._data_processor = data_processor
        self.job_config = job_config

        self.infinite = infinite

        # Variables for checkpointing
        self._sample_idx = start_idx

    def _get_data_iter(self):
        """Get iterator over the dataset for this rank."""
        # Start from checkpointed position to resume correctly
        # load_state_dict already ensures _sample_idx is within valid range
        if self._sample_idx >= self._end_idx:
            return iter([])
        indices = list(range(self._sample_idx, self._end_idx))
        return iter(indices)

    def __iter__(self):
        """Iterate over the dataset."""
        while True:
            indices = self._get_data_iter()

            for idx in indices:
                # Get sample from the underlying RawVideoDataset
                sample = self._raw_dataset[idx]

                # Process the sample using the data processor
                # Pass precomputed empty tokens for efficiency
                sample_dict = self._data_processor(
                    sample,
                    self._t5_tokenizer,
                    job_config=self.job_config,
                    t5_empty_tokens=self._t5_empty_token,
                )

                self._sample_idx = idx + 1

                # Yield the processed sample
                # For video datasets, we might yield video_frames and robot_states separately
                video_frames = sample_dict["video_frames"]  # TODO: is this zero copy?
                # robot_states = sample_dict.pop("robot_states")

                # Add sample index for latent caching
                sample_dict["sample_idx"] = idx

                yield sample_dict, video_frames

            if not self.infinite:
                logger.warning(
                    f"Dataset {self.dataset_name} has run out of data. "
                    f"This might cause NCCL timeout if data parallelism is enabled."
                )
                break
            else:
                # Reset for next iteration
                self._sample_idx = self._start_idx
                logger.warning(f"Dataset {self.dataset_name} is being re-looped.")
                # Continue to restart the while loop
                continue

    def load_state_dict(self, state_dict):
        """Load checkpoint state."""
        self._sample_idx = state_dict.get("sample_idx", self._start_idx)
        # Ensure sample_idx is within valid range
        if self._sample_idx < self._start_idx:
            self._sample_idx = self._start_idx
        if self._sample_idx >= self._end_idx:
            self._sample_idx = self._end_idx - 1

    def state_dict(self):
        """Save checkpoint state."""
        return {"sample_idx": self._sample_idx}
    
    def __len__(self):
        return len(self._raw_dataset)


def build_wan_dataloader(
    dp_world_size: int,
    dp_rank: int,
    job_config: JobConfig,
    # This parameter is not used, keep it for compatibility
    tokenizer: WanTokenizer | None,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """
    Build a data loader for WAN video datasets.

    Currently only supports 1xwm dataset via RawVideoDatasetWrapper.
    """
    dataset_name = job_config.training.dataset.lower()
    dataset_path = job_config.training.dataset_path
    logger.info(
        f"Building TRAINING dataloader: dataset={dataset_name}, path={dataset_path}"
    )
    batch_size = job_config.training.local_batch_size
    num_workers = job_config.training.num_workers
    persistent_workers = job_config.training.persistent_workers
    pin_memory = job_config.training.pin_memory
    prefetch_factor = job_config.training.prefetch_factor

    # TODO: Without this the dataset_wan test fails
    # TODO: Add the code to pass the correct prefetch_factor/num_workers/persistent_workers when testing
    if num_workers == 0 and prefetch_factor != 1.0:
        prefetch_factor = None

    t5_tokenizer = build_wan_tokenizer(job_config)

    # Only support 1xwm video dataset
    if dataset_name != "1xwm":
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. Only '1xwm' is currently supported."
        )

    ds = RawVideoDatasetWrapper(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        t5_tokenizer=t5_tokenizer,
        job_config=job_config,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    return ParallelAwareDataloader(
        dataset=ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )


def build_wan_latent_dataloader(
    dp_world_size: int,
    dp_rank: int,
    job_config: JobConfig,
    tokenizer: WanTokenizer | None,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """
    Build a data loader that loads pre-encoded latents from disk.

    Requires:
    - job_config.training.latents_path: Path to latent files (latent_XXXXXXXX.pt)
    - job_config.training.dataset_path: Path to raw dataset (for robot_states)
    """
    latents_path = job_config.training.latents_path
    raw_dataset_path = job_config.training.dataset_path
    logger.info(
        f"Building LATENT dataloader: latents={latents_path}, raw={raw_dataset_path}"
    )
    batch_size = job_config.training.local_batch_size
    num_workers = job_config.training.num_workers
    persistent_workers = job_config.training.persistent_workers
    pin_memory = job_config.training.pin_memory
    prefetch_factor = job_config.training.prefetch_factor

    if num_workers == 0 and prefetch_factor != 1.0:
        prefetch_factor = None

    t5_tokenizer = build_wan_tokenizer(job_config)

    ds = LatentDatasetWrapper(
        latents_dir=latents_path,
        raw_dataset_path=raw_dataset_path,
        t5_tokenizer=t5_tokenizer,
        job_config=job_config,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    return ParallelAwareDataloader(
        dataset=ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )


class ValidationLatentDatasetWrapper(IterableDataset, Stateful):
    """
    Dataset for validation that loads BOTH pre-encoded latents AND video frames.

    Unlike training (which only needs latents), validation needs both:
    - Pre-encoded latents: Skip VAE encoding during loss computation
    - Original video frames: Required for PSNR computation against generated videos

    Args:
        latents_dir: Path to directory containing latent_XXXXXXXX.pt files
        raw_dataset_path: Path to raw dataset (for video_frames and robot_states)
        t5_tokenizer: T5 tokenizer for text encoding
        job_config: Job configuration
        dp_rank: Data parallel rank
        dp_world_size: Data parallel world size
        infinite: Whether to loop infinitely
    """

    def __init__(
        self,
        latents_dir: str,
        raw_dataset_path: str,
        t5_tokenizer: BaseTokenizer,
        job_config: JobConfig,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        logger.info("=" * 60)
        logger.info("Initializing ValidationLatentDatasetWrapper")
        logger.info("=" * 60)
        logger.info(f"  Latents directory: {latents_dir}")
        logger.info(f"  Raw dataset path: {raw_dataset_path}")
        logger.info(f"  DP rank: {dp_rank}, DP world size: {dp_world_size}")

        self.latents_dir = Path(latents_dir)
        self.job_config = job_config
        self.infinite = infinite

        # Discover available latent files and extract indices
        logger.info("  Scanning for validation latent files...")
        latent_files = sorted(self.latents_dir.glob("latent_*.pt"))
        all_indices = []
        for f in latent_files:
            idx_str = f.stem.split("_")[1]
            all_indices.append(int(idx_str))

        self._dataset_len = len(all_indices)
        logger.info(f"  Found {self._dataset_len} validation latent files")

        # Split indices across data parallel ranks
        indices_per_rank = self._dataset_len // dp_world_size
        start_idx = dp_rank * indices_per_rank
        end_idx = start_idx + indices_per_rank if dp_rank < dp_world_size - 1 else self._dataset_len

        self._all_indices = all_indices
        self._start_idx = start_idx
        self._end_idx = end_idx
        self._my_indices = all_indices[start_idx:end_idx]
        logger.info(f"  DP sharding: rank {dp_rank} gets indices [{start_idx}:{end_idx}] ({len(self._my_indices)} samples)")

        # Load raw dataset for video_frames and robot_states
        logger.info("  Loading raw dataset for video_frames and robot_states...")
        downsampled = getattr(job_config.training, "downsampled", 4)
        clip_length = getattr(job_config.training, "clip_length", 77)
        window_size = getattr(job_config.training, "window_size", 8)
        robot_temporal_mode = getattr(job_config.training, "robot_temporal_mode", "downsample")

        self._raw_dataset = BaseRawVideoDataset(
            data_dir=raw_dataset_path,
            downsampled=downsampled,
            clip_length=clip_length,
            window_size=window_size,
            robot_temporal_mode=robot_temporal_mode,
        )
        logger.info(f"  Raw dataset loaded: {len(self._raw_dataset)} samples")

        # Tokenizer for T5 (precompute empty string tokens)
        self._t5_tokenizer = t5_tokenizer
        self._t5_empty_token = t5_tokenizer.encode("")

        # Checkpoint state
        self._sample_idx = 0

        logger.info("  ValidationLatentDatasetWrapper initialization complete")
        logger.info("=" * 60)

    def _get_data_iter(self):
        if self._sample_idx >= len(self._my_indices):
            return iter([])
        return iter(range(self._sample_idx, len(self._my_indices)))

    def __iter__(self):
        while True:
            indices_iter = self._get_data_iter()

            for local_idx in indices_iter:
                global_idx = self._my_indices[local_idx]

                # Load pre-encoded latent from disk
                latent_path = self.latents_dir / f"latent_{global_idx:08d}.pt"
                latent = torch.load(latent_path, weights_only=True)

                # Load video_frames and robot_states from raw dataset
                video_frames, robot_states = self._raw_dataset[global_idx]

                # Build sample dict with BOTH latents and video_frames
                sample_dict = {
                    "latents": latent,  # Shape: (C, T, H, W) = (48, 20, 32, 32)
                    "video_frames": video_frames,  # Shape: (T, H, W, C) for PSNR
                    "robot_states": robot_states,
                    "t5_tokens": self._t5_empty_token,
                    "sample_idx": global_idx,
                }

                self._sample_idx = local_idx + 1

                # Yield sample_dict and video_frames as labels (for compatibility)
                yield sample_dict, video_frames

            if not self.infinite:
                logger.warning(
                    "Validation latent dataset exhausted. "
                    "This might cause NCCL timeout if data parallelism is enabled."
                )
                break
            else:
                self._sample_idx = 0
                logger.warning("Validation latent dataset is being re-looped.")
                continue

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict.get("sample_idx", 0)
        if self._sample_idx < 0:
            self._sample_idx = 0
        if self._sample_idx >= len(self._my_indices):
            self._sample_idx = len(self._my_indices) - 1

    def state_dict(self):
        return {"sample_idx": self._sample_idx}

    def __len__(self):
        return self._dataset_len


def build_wan_validation_dataloader(
    dp_world_size: int,
    dp_rank: int,
    job_config: JobConfig,
    # This parameter is not used, keep it for compatibility
    tokenizer: BaseTokenizer | None,
    generate_timestamps: bool = True,
    infinite: bool = False,
) -> ParallelAwareDataloader:
    """
    Build a data loader for WAN video validation datasets.

    If validation.latents_path is set, loads pre-encoded latents from disk
    (along with video frames for PSNR computation).
    Otherwise, loads raw videos for on-the-fly VAE encoding.
    """
    dataset_name = job_config.validation.dataset.lower()
    dataset_path = job_config.validation.dataset_path
    latents_path = getattr(job_config.validation, "latents_path", "")

    batch_size = job_config.validation.local_batch_size
    # Use training dataloader settings for validation as well
    num_workers = job_config.training.num_workers
    persistent_workers = job_config.training.persistent_workers
    pin_memory = job_config.training.pin_memory
    # prefetch_factor must be None when num_workers=0 (single-process loading)
    prefetch_factor = job_config.training.prefetch_factor if num_workers > 0 else None

    t5_tokenizer = build_wan_tokenizer(job_config)

    # Only support 1xwm video dataset
    if dataset_name != "1xwm":
        raise ValueError(
            f"Unsupported validation dataset: {dataset_name}. Only '1xwm' is currently supported."
        )

    # Use latent dataloader if latents_path is provided
    if latents_path:
        logger.info(
            f"Building VALIDATION dataloader with PRE-ENCODED LATENTS: latents={latents_path}, raw={dataset_path}"
        )
        ds = ValidationLatentDatasetWrapper(
            latents_dir=latents_path,
            raw_dataset_path=dataset_path,
            t5_tokenizer=t5_tokenizer,
            job_config=job_config,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=infinite,
        )
    else:
        logger.info(
            f"Building VALIDATION dataloader with RAW VIDEOS: dataset={dataset_name}, path={dataset_path}"
        )
        ds = RawVideoDatasetWrapper(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            t5_tokenizer=t5_tokenizer,
            job_config=job_config,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=infinite,
        )

    return ParallelAwareDataloader(
        dataset=ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
