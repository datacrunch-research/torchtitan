# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import sys

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.config.job_config import Compile as CompileConfig
from torchtitan.distributed import ParallelDims
from torchtitan.tools import utils
from torchtitan.tools.logging import logger

# # Import LoRA functions (only when needed)
# try:
#     from torchtitan.experiments.wan.model.lora import apply_lora_to_linear, freeze_base_weights
#     LORA_AVAILABLE = True
# except ImportError:
#     LORA_AVAILABLE = False


def parallelize_wan(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    *args,  # Accept extra args for compatibility (unused now)
    **kwargs,  # Accept extra kwargs for compatibility (unused now)
):
    """
    Apply parallelism to Wan model. This function only applies parallelism (FSDP, activation
    checkpointing, compile, etc.). Weight initialization is handled separately in train.py
    after parallelization, which allows weights to be loaded into FSDP-wrapped models using
    PyTorch's distributed checkpoint APIs.
    
    Args:
        model: The model to parallelize (on meta device)
        parallel_dims: Parallelism dimensions
        job_config: Job configuration
        *args: Extra positional arguments (for compatibility, unused)
        **kwargs: Extra keyword arguments (for compatibility, unused)
    
    Returns:
        The parallelized model (still on meta device, weights not initialized)
    """
    
    # Apply activation checkpointing if enabled
    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint)

    # Apply torch.compile if enabled
    # Note: Compile should be applied AFTER activation checkpointing and BEFORE FSDP
    # for best performance and compatibility
    if job_config.compile.enable and "model" in job_config.compile.components:
        logger.info("  [parallelize_wan] Applying torch.compile to model...")
        logger.info(f"    - Backend: {job_config.compile.backend}")
        logger.info(f"    - Components: {job_config.compile.components}")
        sys.stdout.flush()
        apply_compile(model, job_config.compile)
        logger.info("  [parallelize_wan] ✓ Model compilation completed")
        sys.stdout.flush()

    if parallel_dims.fsdp_enabled:
        names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(names)

        logger.info("  [parallelize_wan] Applying FSDP wrapping (this may take a minute)...")
        sys.stdout.flush()
        
        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            cpu_offload=job_config.training.enable_cpu_offload,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")
        sys.stdout.flush()

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


def apply_compile(model: nn.Module, compile_config: CompileConfig):
    """
    Apply torch.compile to the Wan model.
    
    For Wan models, we compile each transformer block individually for efficiency,
    similar to how other models handle compilation. This allows for better optimization
    of the repeated transformer structure.
    
    Args:
        model: The WanModel to compile
        compile_config: Compile configuration from job_config
    """
    # Compile each transformer block individually
    # This is more efficient than compiling the whole model due to repeated structure
    compiled_blocks = 0
    for layer_id, block in model.blocks.named_children():
        compiled_block = torch.compile(
            block,
            backend=compile_config.backend,
            fullgraph=True,  # Require full graph for better optimization
        )
        model.blocks.register_module(layer_id, compiled_block)
        compiled_blocks += 1
    
    logger.info(
        f"Compiled {compiled_blocks} transformer blocks with torch.compile "
        f"(backend: {compile_config.backend})"
    )


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
