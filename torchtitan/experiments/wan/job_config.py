# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field


@dataclass
class Training:
    classifier_free_guidance_prob: float = 0.0
    """Classifier-free guidance with probability `p` to dropout each text encoding independently.
    If `n` text encoders are used, the unconditional model is trained in `p ^ n` of all steps.
    For example, if `n = 2` and `p = 0.447`, the unconditional model is trained in 20% of all steps"""
    img_size: int = 512
    """Image width to sample"""
    test_mode: bool = False
    """Whether to use integration test mode, which will randomly initialize the encoder and use a dummy tokenizer"""
    # Dataset-specific parameters for RawVideoDataset (1x-wmds)
    downsampled: int = 4
    """Downsampling factor for video frames: 1, 2, or 4. Only used for 1x-wmds dataset."""
    clip_length: int = 77
    """Number of frames per clip. Only used for 1x-wmds dataset."""
    window_size: int = 8
    """Window size for sampling clips. Only used for 1x-wmds dataset."""
    robot_temporal_mode: str = "downsample"
    """How to handle robot state temporal alignment: 'downsample' or other modes. Only used for 1x-wmds dataset."""


@dataclass
class Encoder:
    t5_encoder: str = "google/t5-v1_1-small"
    """T5 encoder to use, HuggingFace model name. This field could be either a local folder path,
        or a Huggingface repo name."""
    clip_encoder: str = "openai/clip-vit-large-patch14"
    """Clip encoder to use, HuggingFace model name. This field could be either a local folder path,
        or a Huggingface repo name."""
    autoencoder_path: str = (
        "torchtitan/models/flux/assets/autoencoder/ae.safetensors"
    )
    """Autoencoder checkpoint path to load. This should be a local path referring to a safetensors file."""
    wan_vae_path: str = "Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    """Wan Video VAE checkpoint path to load. This should be a local path referring to a .pth or .safetensors file."""
    max_t5_encoding_len: int = 256
    """Maximum length of the T5 encoding."""


@dataclass
class Validation:
    enable_classifier_free_guidance: bool = False
    """Whether to use classifier-free guidance during sampling"""
    classifier_free_guidance_scale: float = 5.0
    """Classifier-free guidance scale when sampling"""
    denoising_steps: int = 50
    """How many denoising steps to sample when generating an image"""
    eval_freq: int = 100
    """Frequency of evaluation/sampling during training"""
    save_img_count: int = 1
    """ How many images to generate and save during validation, starting from
    the beginning of validation set, -1 means generate on all samples"""
    save_img_folder: str = "img"
    """Directory to save image generated/sampled from the model"""
    all_timesteps: bool = False
    """Whether to generate all stratified timesteps per sample or use round robin"""


@dataclass
class Inference:
    """Inference configuration"""

    save_img_folder: str = "inference_results"
    """Path to save the inference results"""
    prompts_path: str = "./torchtitan/experiments/wan/inference/prompts.txt"
    """Path to file with newline separated prompts to generate images for"""
    local_batch_size: int = 2
    """Batch size for inference"""
    img_size: int = 256
    """Image size for inference"""


@dataclass
class JobConfig:
    """
    Extend the tyro parser with custom config classes for Wan model.
    """

    training: Training = field(default_factory=Training)
    encoder: Encoder = field(default_factory=Encoder)
    validation: Validation = field(default_factory=Validation)
    inference: Inference = field(default_factory=Inference)
