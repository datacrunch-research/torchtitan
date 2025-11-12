# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import math
from functools import partial
from typing import Any, Callable, Optional

import json
import pathlib
import numpy as np
import PIL

import torch
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node

from torch.distributed.checkpoint.stateful import Stateful

from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import Job, JobConfig
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.experiments.wan.tokenizer import build_wan_tokenizer, WanTokenizer
from torchtitan.tools.logging import logger
from torchtitan.experiments.wan.model.dataset import RawVideoDataset as BaseRawVideoDataset


def _process_cc12m_image(
    img: PIL.Image.Image,
    output_size: int = 256,
) -> Optional[torch.Tensor]:
    """Process CC12M image to the desired size."""

    width, height = img.size
    # Skip low resolution images
    if width < output_size or height < output_size:
        return None

    if width >= height:
        # resize height to be equal to output_size, then crop
        new_width, new_height = math.ceil(output_size / height * width), output_size
        img = img.resize((new_width, new_height))
        left = torch.randint(0, new_width - output_size + 1, (1,)).item()
        resized_img = img.crop((left, 0, left + output_size, output_size))
    else:
        # resize width to be equal to output_size, the crop
        new_width, new_height = (
            output_size,
            math.ceil(output_size / width * height),
        )
        img = img.resize((new_width, new_height))
        lower = torch.randint(0, new_height - output_size + 1, (1,)).item()
        resized_img = img.crop((0, lower, output_size, lower + output_size))

    assert resized_img.size[0] == resized_img.size[1] == output_size

    # Convert grayscale images, and RGBA, CMYK images
    if resized_img.mode != "RGB":
        resized_img = resized_img.convert("RGB")

    # Normalize the image to [-1, 1]
    np_img = np.array(resized_img).transpose((2, 0, 1))
    tensor_img = torch.tensor(np_img).float() / 255.0 * 2.0 - 1.0

    # NOTE: The following commented code is an alternative way
    # img_transform = transforms.Compose(
    #     [
    #         transforms.Resize(max(output_size, output_size)),
    #         transforms.CenterCrop((output_size, output_size)),
    #         transforms.ToTensor(),
    #     ]
    # )
    # tensor_img = img_transform(img)

    return tensor_img


def _cc12m_wds_data_processor(
    sample: dict[str, Any],
    t5_tokenizer: WanTokenizer,
    clip_tokenizer: WanTokenizer,
    output_size: int = 256,
) -> dict[str, Any]:
    """
    Preprocess CC12M dataset sample image and text for Flux model.

    Args:
        sample: A sample from dataset
        t5_encoder: T5 encoder
        clip_encoder: CLIP encoder
        output_size: The output image size

    """
    img = _process_cc12m_image(sample["jpg"], output_size=output_size)
    t5_tokens = t5_tokenizer.encode(sample["txt"])
    clip_tokens = clip_tokenizer.encode(sample["txt"])

    return {
        "image": img,
        "clip_tokens": clip_tokens,  # type: List[int]
        "t5_tokens": t5_tokens,  # type: List[int]
        "prompt": sample["txt"],  # type: str
    }


def _coco_data_processor(
    sample: dict[str, Any],
    t5_tokenizer: WanTokenizer,
    clip_tokenizer: WanTokenizer,
    output_size: int = 256,
) -> dict[str, Any]:
    """
    Preprocess COCO dataset sample image and text for Flux model.

    Args:
        sample: A sample from dataset
        t5_encoder: T5 encoder
        clip_encoder: CLIP encoder
        output_size: The output image size

    """
    img = _process_cc12m_image(sample["image"], output_size=output_size)
    prompt = sample["caption"]
    if isinstance(prompt, list):
        prompt = prompt[0]
    t5_tokens = t5_tokenizer.encode(prompt)
    clip_tokens = clip_tokenizer.encode(prompt)

    return {
        "image": img,
        "clip_tokens": clip_tokens,  # type: List[int]
        "t5_tokens": t5_tokens,  # type: List[int]
        "prompt": prompt,  # type: str
    }

def _load_raw_video_dataset(
    dataset_path: str,
    downsampled: int = 4,
    clip_length: int = 77,
    repeat: int = 1,
    window_size: int = 8,
    robot_temporal_mode: str = "downsample",
) -> BaseRawVideoDataset:
    """
    Load RawVideoDataset from local directory.
    
    Args:
        dataset_path: Path to the dataset directory
        downsampled: Downsampling factor (1, 2, or 4)
        clip_length: Number of frames per clip
        repeat: Number of times to repeat the dataset
        window_size: Window size for sampling clips
        robot_temporal_mode: How to handle robot state temporal alignment
        
    Returns:
        RawVideoDataset instance
    """
    logger.info(f"Loading RawVideoDataset from {dataset_path}")
    return BaseRawVideoDataset(
        data_dir=dataset_path,
        downsampled=downsampled,
        clip_length=clip_length,
        repeat=repeat,
        window_size=window_size,
        robot_temporal_mode=robot_temporal_mode,
    )


def _process_raw_video_sample(
    sample: tuple[torch.Tensor, torch.Tensor],
    t5_tokenizer: WanTokenizer,
    clip_tokenizer: WanTokenizer,
    job_config: Optional[JobConfig] = None,
    t5_empty_tokens: Optional[list[int]] = None,
    clip_empty_tokens: Optional[list[int]] = None,
) -> dict[str, Any]:
    """
    Process RawVideoDataset sample (video frames and robot states).
    
    Args:
        sample: Tuple of (video_frames, robot_states)
            - video_frames: Tensor of shape [T, H, W, C]
            - robot_states: Tensor of shape [T, state_dim]
        t5_tokenizer: T5 tokenizer for text encoding
        clip_tokenizer: CLIP tokenizer for text encoding
        job_config: Job configuration (optional, for accessing config params)
        t5_empty_tokens: Precomputed empty string tokens for T5 (optional, for efficiency)
        clip_empty_tokens: Precomputed empty string tokens for CLIP (optional, for efficiency)
        
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
    
    if clip_empty_tokens is not None:
        clip_tokens = clip_empty_tokens
    else:
        clip_tokens = clip_tokenizer.encode(prompt) if prompt else clip_tokenizer.encode("")
    
    return {
        "video_frames": video_frames,  # Shape: [T, H, W, C]
        "robot_states": robot_states,  # Shape: [T, state_dim]
        "clip_tokens": clip_tokens,
        "t5_tokens": t5_tokens,
        "prompt": prompt,
    }


DATASETS = {
    "1xwm": DatasetConfig(
        path="",  # Path will be provided via dataset_path in config
        loader=_load_raw_video_dataset,
        sample_processor=_process_raw_video_sample,
    ),
    # "cc12m-wds": DatasetConfig(
    #     path="pixparse/cc12m-wds",
    #     loader=lambda path: load_dataset(path, split="train", streaming=True),
    #     sample_processor=_cc12m_wds_data_processor,
    # ),
    # "cc12m-test": DatasetConfig(
    #     path="tests/assets/cc12m_test",
    #     loader=lambda path: load_dataset(
    #         path, split="train", data_files={"train": "*.tar"}, streaming=True
    #     ),
    #     sample_processor=_cc12m_wds_data_processor,
    # ),
    # "coco-validation": DatasetConfig(
    #     path="howard-hou/COCO-Text",
    #     loader=lambda path: load_dataset(path, split="validation", streaming=True),
    #     sample_processor=_coco_data_processor,
    # ),
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

class RawVideoDatasetWrapper(IterableDataset, Stateful):
    """
    Wrapper for RawVideoDataset to make it compatible with IterableDataset pattern.
    
    This wraps the map-style RawVideoDataset (PyTorch Dataset) and makes it work
    as an IterableDataset for distributed training.
    
    Args:
        dataset_name: Name of the dataset
        dataset_path: Path to the dataset directory
        t5_tokenizer: T5 tokenizer
        clip_tokenizer: CLIP tokenizer
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
        clip_tokenizer: BaseTokenizer,
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
        if job_config and hasattr(job_config.training, 'downsampled'):
            downsampled = getattr(job_config.training, 'downsampled', 4)
        else:
            downsampled = 4
            
        if job_config and hasattr(job_config.training, 'clip_length'):
            clip_length = getattr(job_config.training, 'clip_length', 77)
        else:
            clip_length = 77
            
        if job_config and hasattr(job_config.training, 'window_size'):
            window_size = getattr(job_config.training, 'window_size', 8)
        else:
            window_size = 8
            
        if job_config and hasattr(job_config.training, 'robot_temporal_mode'):
            robot_temporal_mode = getattr(job_config.training, 'robot_temporal_mode', 'downsample')
        else:
            robot_temporal_mode = 'downsample'
        
        # Create the underlying RawVideoDataset
        # The loader for 1x-wmds accepts additional parameters
        if dataset_name == "1xwm":
            raw_dataset = dataset_loader(
                path,
                downsampled=downsampled,
                clip_length=clip_length,
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
        end_idx = start_idx + indices_per_rank if dp_rank < dp_world_size - 1 else dataset_len
        
        self.dataset_name = dataset_name
        self._raw_dataset = raw_dataset
        self._start_idx = start_idx
        self._end_idx = end_idx
        self._dataset_len = dataset_len
        
        self._t5_tokenizer = t5_tokenizer
        self._t5_empty_token = t5_tokenizer.encode("")
        self._clip_tokenizer = clip_tokenizer
        self._clip_empty_token = clip_tokenizer.encode("")
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
                    self._clip_tokenizer,
                    job_config=self.job_config,
                    t5_empty_tokens=self._t5_empty_token,
                    clip_empty_tokens=self._clip_empty_token,
                )
                
                self._sample_idx = idx + 1
                
                # Yield the processed sample
                # For video datasets, we might yield video_frames and robot_states separately
                video_frames = sample_dict["video_frames"] # TODO: is this zero copy?
                # robot_states = sample_dict.pop("robot_states")
                
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
    batch_size = job_config.training.local_batch_size

    t5_tokenizer, clip_tokenizer = build_wan_tokenizer(job_config)

    # Only support 1xwm video dataset
    if dataset_name != "1xwm":
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. Only '1xwm' is currently supported."
        )

    ds = RawVideoDatasetWrapper(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        t5_tokenizer=t5_tokenizer,
        clip_tokenizer=clip_tokenizer,
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
    )


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
    
    Currently only supports 1xwm dataset via RawVideoDatasetWrapper.
    """
    dataset_name = job_config.validation.dataset.lower()
    dataset_path = job_config.validation.dataset_path
    batch_size = job_config.validation.local_batch_size

    t5_tokenizer, clip_tokenizer = build_wan_tokenizer(job_config)

    # Only support 1xwm video dataset
    if dataset_name != "1xwm":
        raise ValueError(
            f"Unsupported validation dataset: {dataset_name}. Only '1xwm' is currently supported."
        )

    ds = RawVideoDatasetWrapper(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        t5_tokenizer=t5_tokenizer,
        clip_tokenizer=clip_tokenizer,
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
    )
