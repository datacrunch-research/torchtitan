# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.tools import utils
from torchtitan.tools.logging import logger


def parallelize_wan(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    *args,  # Accept extra args for compatibility (init_device, buffer_device, etc.)
    **kwargs,  # Accept extra kwargs for compatibility
):
    """
    Apply parallelism to Wan model. For Wan models, weights must be loaded BEFORE FSDP wrapping.
    
    This function automatically handles:
    - Moving model from meta device to target device
    - Loading pretrained weights (if available) before FSDP wrapping
    - Applying FSDP/parallelism
    
    Args:
        model: The model to parallelize (on meta device)
        parallel_dims: Parallelism dimensions
        job_config: Job configuration
        *args: Extra positional arguments (for compatibility, accepts init_device, buffer_device)
        **kwargs: Extra keyword arguments (for compatibility)
    
    Returns:
        The parallelized model
    """
    # Determine init_device and buffer_device from config (same logic as base Trainer)
    device_type = utils.device_type
    if job_config.checkpoint.create_seed_checkpoint:
        init_device = "cpu"
        buffer_device = None
    elif job_config.training.enable_cpu_offload:
        init_device = "cpu"
        buffer_device = device_type
    else:
        init_device = device_type
        buffer_device = None
    
    # Get pretrained weights path from config or kwargs
    pretrained_weights_path = kwargs.get('pretrained_weights_path', None)
    if pretrained_weights_path is None:
        pretrained_weights_path = getattr(job_config.encoder, 'pretrained_weights_path', None)
    
    # Override with args if provided (for backward compatibility)
    if len(args) >= 1:
        init_device = args[0]
    if len(args) >= 2:
        buffer_device = args[1]
    if len(args) >= 3:
        pretrained_weights_path = args[2]
    
    # Wan-specific: Load weights BEFORE applying FSDP wrapping
    # This avoids DTensor complications when loading weights
    # Check if model is on meta device (needs initialization)
    if hasattr(model, 'parameters') and next(model.parameters(), None) is not None:
        first_param = next(model.parameters())
        is_meta_device = first_param.device.type == "meta"
    else:
        # If no parameters, check buffers or assume meta
        is_meta_device = True
    
    if is_meta_device:
        logger.info("  [parallelize_wan] Moving model to device and loading weights (BEFORE FSDP)...")
        model.to_empty(device=init_device)
        
        # Load pretrained weights if model supports it
        if hasattr(model, 'init_weights'):
            if pretrained_weights_path is not None:
                logger.info(f"  [parallelize_wan] Loading pretrained weights from: {pretrained_weights_path}")
                with torch.no_grad():
                    model.init_weights(buffer_device=buffer_device, pretrained_weights_path=pretrained_weights_path)
                logger.info("  [parallelize_wan] ✓ Pretrained weights loaded successfully")
            else:
                # Still call init_weights for default initialization (if needed)
                # Some models might need this even without pretrained weights
                logger.debug("  [parallelize_wan] No pretrained weights path, using default initialization")
                with torch.no_grad():
                    model.init_weights(buffer_device=buffer_device)
    
    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint)

    if parallel_dims.fsdp_enabled:
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)

        apply_fsdp(
            model,
            parallel_dims.world_mesh[tuple(dp_mesh_dim_names)],
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            cpu_offload=job_config.training.enable_cpu_offload,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if parallel_dims.cp_enabled:
            # The attention in Flux does not use causal mask.
            # Currently, load_balance must be disabled in order to support Context Parallelism
            # in Pytorch's experimental ring attention module
            # https://github.com/pytorch/pytorch/blob/v2.9.0/torch/distributed/tensor/experimental/_attention.py#L395
            from torch.distributed.tensor.experimental._attention import _cp_options

            _cp_options.enable_load_balance = False
            logger.info("Applied Context Parallel to the model")

    return model


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool = False,
):
    """
    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reduction operations.
        cpu_offload (bool): Whether to offload model parameters to CPU. Defaults to False.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    linear_layers = [
        model.text_embedding,
        model.time_embedding,
        model.time_projection,
    ]
    for layer in linear_layers:
        fully_shard(layer, **fsdp_config)

    for block in model.blocks:
        fully_shard(
            block,
            **fsdp_config,
        )
    # apply FSDP to last layer. Set reshard_after_forward=False for last layer to avoid gather right after reshard
    fully_shard(model.head, **fsdp_config, reshard_after_forward=False)

    # Wrap all the rest of model
    fully_shard(model, **fsdp_config)


def apply_ac(model: nn.Module, ac_config):
    """Apply activation checkpointing to the model."""

    for layer_id, block in model.blocks.named_children():
        block = ptd_checkpoint_wrapper(block, preserve_rng_state=False)
        model.blocks.register_module(layer_id, block)

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the model")


def parallelize_encoders(
    t5_model: nn.Module,
    wan_video_vae: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    if parallel_dims.dp_shard_enabled:  # apply FSDP or HSDP
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard")
        else:
            dp_mesh_dim_names = ("dp_shard",)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
        )
        fsdp_config = {
            "mesh": parallel_dims.world_mesh[tuple(dp_mesh_dim_names)],
            "mp_policy": mp_policy,
        }
        if job_config.training.enable_cpu_offload:
            fsdp_config["offload_policy"] = CPUOffloadPolicy()

        # Apply FSDP to the T5 encoder
        for block in t5_model.hf_module.encoder.block:
            fully_shard(block, **fsdp_config)
        fully_shard(t5_model.hf_module, **fsdp_config)

        # Apply FSDP to the Wan Video VAE if provided
        # The VAE is large enough (z_dim=48, dim=160) to benefit from FSDP
        if wan_video_vae is not None:
            # Apply FSDP to the VAE model (encoder and decoder)
            if hasattr(wan_video_vae, 'model'):
                # WanVideoVAE38 wraps the model in self.model
                fully_shard(wan_video_vae.model.encoder, **fsdp_config)
                fully_shard(wan_video_vae.model.decoder, **fsdp_config)
                # Also wrap the conv layers
                fully_shard(wan_video_vae.model.conv1, **fsdp_config)
                fully_shard(wan_video_vae.model.conv2, **fsdp_config)
                logger.info("Applied FSDP to the Wan Video VAE")
            else:
                # Fallback: wrap the entire VAE
                fully_shard(wan_video_vae, **fsdp_config)
                logger.info("Applied FSDP to the Wan Video VAE (full model)")

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the T5 encoder model")
        else:
            logger.info("Applied FSDP to the T5 encoder model")

    return t5_model, wan_video_vae
