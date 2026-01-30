# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Generator, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.pipelining.schedules import _PipelineSchedule
from torchmetrics.functional.image import peak_signal_noise_ratio

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.loss import LossFunction
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.components.validate import Validator
from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.wan.inference.sampling import generate_video, save_video
from torchtitan.experiments.wan.model.hf_embedder import WanEmbedder

from torchtitan.experiments.wan.model.wan_vae import WanVideoVAE
from torchtitan.experiments.wan.tokenizer import build_wan_tokenizer
from torchtitan.experiments.wan.utils import (
    # create_position_encoding_for_latents,
    pack_latents,
    preprocess_data,
)
from torchtitan.experiments.wan.wan_datasets import build_wan_validation_dataloader

from torchtitan.tools.logging import init_logger, logger


class WanValidator(Validator):
    """
    Simple validator focused on correctness and integration.

    Args:
        job_config: Job configuration
        validation_dataloader: The validation dataloader
        loss_fn: Loss function to use for validation
        model: The model to validate (single model, no parallelism)
    """

    validation_dataloader: BaseDataLoader

    def __init__(
        self,
        job_config: JobConfig,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        parallel_dims: ParallelDims,
        loss_fn: LossFunction,
        validation_context: Generator[None, None, None],
        maybe_enable_amp: Generator[None, None, None],
        metrics_processor: MetricsProcessor | None = None,
        pp_schedule: _PipelineSchedule | None = None,
        pp_has_first_stage: bool | None = None,
        pp_has_last_stage: bool | None = None,
    ):
        self.job_config = job_config
        self.parallel_dims = parallel_dims
        self.loss_fn = loss_fn
        self.all_timesteps = self.job_config.validation.all_timesteps
        self.validation_dataloader = build_wan_validation_dataloader(
            job_config=job_config,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            tokenizer=tokenizer,
            generate_timestamps=not self.all_timesteps,
            infinite=self.job_config.validation.steps != -1,
        )
        self.validation_context = validation_context
        self.maybe_enable_amp = maybe_enable_amp
        self.metrics_processor = metrics_processor

        self.t5_tokenizer = build_wan_tokenizer(self.job_config)
        self.precomputed_t5_embedding = None
        # with torch.no_grad():
        #     empty_t5_tokens_tensor = self.t5_tokenizer.encode("").to(device=self.device)
        #     self._precomputed_t5_embedding = self.t5_encoder(empty_t5_tokens_tensor).to(dtype=self._dtype)
        #     self._precomputed_t5_embedding = self._precomputed_t5_embedding.squeeze(0)  # [seq_len, hidden_dim]

        if self.job_config.validation.steps == -1:
            logger.warning(
                "Setting validation steps to -1 might cause hangs because of "
                "unequal sample counts across ranks when dataset is exhausted."
            )

    def wan_init(
        self,
        device: torch.device,
        _dtype: torch.dtype,
        wan_video_vae: WanVideoVAE,
        t5_encoder: Optional[WanEmbedder] = None,
        precomputed_t5_embedding: Optional[Tensor] = None,
    ):
        self.device = device
        self._dtype = _dtype
        self.wan_video_vae = wan_video_vae
        self.t5_encoder = t5_encoder
        self.precomputed_t5_embedding = precomputed_t5_embedding

        # Overfit mode: use same batch as training for validation
        self._overfit_batch: tuple[dict[str, Tensor], Tensor] | None = None
        self._overfit_mode = getattr(
            self.job_config.training, "overfit_single_sample", True
        )

    def set_overfit_batch(self, input_dict: dict[str, Tensor], labels: Tensor) -> None:
        """Set the cached batch for overfit mode validation."""
        self._overfit_batch = (
            {
                k: v.clone() if isinstance(v, Tensor) else v
                for k, v in input_dict.items()
            },
            labels.clone(),
        )
        logger.info("Validator: cached overfit batch for validation")

    @torch.no_grad()
    def validate(
        self,
        model_parts: list[nn.Module],
        step: int,
    ) -> None:
        """
        Run validation during training.

        This generates videos using the diffusion model and computes PSNR
        against ground truth to measure model performance.
        """

        # Set model to eval mode
        # TODO: currently does not support pipeline parallelism
        model = model_parts[0]
        model.eval()

        # Disable cfg dropout during validation
        training_cfg_prob = self.job_config.training.classifier_free_guidance_prob
        self.job_config.training.classifier_free_guidance_prob = 0.0

        parallel_dims = self.parallel_dims
        num_cond_frames = self.job_config.validation.num_cond_frames

        all_psnr_values = []
        accumulated_losses = []
        num_steps = 0

        # Overfit mode: use cached training batch instead of validation dataloader
        if self._overfit_mode and self._overfit_batch is not None:
            logger.info("Validation using OVERFIT batch (same as training)")
            data_source = [self._overfit_batch]  # Single-item iterator
        else:
            logger.info(
                f"len(validation_dataloader): {len(self.validation_dataloader)}"
            )
            data_source = self.validation_dataloader

        for input_dict, labels in data_source:
            if (
                self.job_config.validation.steps != -1
                and num_steps >= self.job_config.validation.steps
            ):
                break

            # Overfit mode: clone tensors to avoid mutation
            if self._overfit_mode and self._overfit_batch is not None:
                input_dict = {
                    k: v.clone() if isinstance(v, Tensor) else v
                    for k, v in input_dict.items()
                }
                labels = labels.clone()

                # When overfit_single_sample=True, use only first element (bsz=1) for validation
                if self.job_config.training.overfit_single_sample:
                    input_dict = {
                        k: v[:1] if isinstance(v, Tensor) else v
                        for k, v in input_dict.items()
                    }
                    labels = labels[:1]

            # Check if we have video frames (not available when using pre-encoded latents)
            has_video_frames = "video_frames" in input_dict

            # Store original video frames for PSNR computation (if available)
            original_video_frames = (
                input_dict["video_frames"].clone() if has_video_frames else None
            )

            # Compute MSE loss similar to training
            # Preprocess data: generate t5 embeddings, encode video with VAE
            processed_input = preprocess_data(
                device=self.device,
                dtype=self._dtype,
                wan_video_vae=self.wan_video_vae,
                t5_encoder=self.t5_encoder,
                batch=input_dict,
                precomputed_t5_embedding=self.precomputed_t5_embedding,
            )
            t5_encodings = processed_input["t5_encodings"]
            latents = processed_input["latents"]  # Ground truth latents
            logger.info(f"{latents.shape}")

            bsz = latents.shape[0]

            # Get number of conditioning frames to keep clean (no noise)
            num_latent_cond = (
                model.num_latent_cond if hasattr(model, "num_latent_cond") else 2
            )

            # Generate noise and timesteps for diffusion (same as training)
            noise = torch.randn_like(latents, dtype=self._dtype)
            timesteps = torch.rand((bsz,), dtype=self._dtype, device=self.device)
            sigmas = timesteps.view(-1, 1, 1, 1, 1)
            # Mix clean latents with noise based on timesteps
            noisy_latents = (1 - sigmas) * latents + sigmas * noise

            # Masking: Keep first num_latent_cond frames clean (no noise) for conditioning
            if num_latent_cond > 0 and latents.shape[2] > num_latent_cond:
                noisy_latents[:, :, :num_latent_cond, :, :] = latents[
                    :, :, :num_latent_cond, :, :
                ]

            # Compute target: noise - labels for frames that need prediction
            target_noise_diff = noise - latents
            if num_latent_cond > 0 and latents.shape[2] > num_latent_cond:
                # Set target to zero for conditioning frames (no noise prediction needed)
                target_noise_diff[:, :, :num_latent_cond, :, :] = 0.0
            target = pack_latents(target_noise_diff)

            # TODO: Context Parallel is not currently used. This context is created but
            # validation_context() doesn't accept it as an argument. To enable CP, follow the
            # Flux pattern: use cp_shard() to shard tensors before validation_context() instead
            # of passing a context manager. See torchtitan/models/flux/train.py for reference.
            optional_context_parallel_ctx = (
                None  # noqa: F841 - kept for future CP implementation
            )
            if parallel_dims.cp_enabled:
                # Pack latents for context parallel
                latents_p = pack_latents(noisy_latents)
                POSITION_DIM = 3
                text_pos_enc = torch.zeros(
                    bsz, t5_encodings.shape[1], POSITION_DIM, device=self.device
                )

                optional_context_parallel_ctx = dist_utils.create_context_parallel_ctx(
                    cp_mesh=parallel_dims.get_mesh("cp"),
                    cp_buffers=[latents_p, t5_encodings, text_pos_enc, target],
                    cp_seq_dims=[1, 1, 1, 1],
                    cp_no_restore_buffers={
                        latents_p,
                        t5_encodings,
                        text_pos_enc,
                        target,
                    },
                    cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
                )

            # Forward pass through the model
            with self.validation_context():
                with self.maybe_enable_amp:
                    # Model forward: predict noise in latents
                    latent_noise_pred = model(
                        x=noisy_latents,
                        timesteps=timesteps,
                        context=t5_encodings,
                        robot_states=input_dict.get("robot_states"),
                    )

                    # Pack the model output to match the target format
                    latent_noise_pred = pack_latents(latent_noise_pred)

                    # Compute MSE loss between predicted noise and target
                    batch_loss = self.loss_fn(latent_noise_pred, target)

            accumulated_losses.append(batch_loss.detach())

            # Video generation and PSNR computation require video_frames
            # Skip when using pre-encoded latents (overfit mode with latents_path)
            if has_video_frames:
                # Set num_cond_frames for generation
                input_dict["num_cond_frames"] = num_cond_frames

                # Generate video using the diffusion model
                generated_video = generate_video(
                    device=self.device,
                    dtype=self._dtype,
                    job_config=self.job_config,
                    model=model,
                    input_dict=input_dict,
                    wan_video_vae=self.wan_video_vae,
                    t5_tokenizer=self.t5_tokenizer,
                    t5_encoder=self.t5_encoder,
                    precomputed_t5_embedding=self.precomputed_t5_embedding,
                )

                # Save video periodically
                output_dir = os.path.join(
                    self.job_config.job.dump_folder,
                    self.job_config.validation.save_img_folder,
                )
                save_video(
                    name=f"video_rank{str(torch.distributed.get_rank())}_step{step}.mp4",
                    output_dir=output_dir,
                    video=generated_video,
                    add_sampling_metadata=True,
                )

                # Compute PSNR for generated video vs ground truth
                # Prepare ground truth: (B, T, H, W, C) uint8 [0,255] -> (B, C, T, H, W) float [-1,1]
                gt_video = original_video_frames.float()
                gt_video = (gt_video / 127.5) - 1.0
                gt_video = gt_video.to(device=self.device, dtype=self._dtype)
                gt_video = gt_video.permute(
                    0, 4, 1, 2, 3
                )  # (B, T, H, W, C) -> (B, C, T, H, W)

                # Save ground truth/target video for comparison
                save_video(
                    name=f"target_video_rank{str(torch.distributed.get_rank())}_step{step}.mp4",
                    output_dir=output_dir,
                    video=gt_video,
                    add_sampling_metadata=False,
                )

                # Clamp generated video
                gen_video = generated_video.clamp(-1.0, 1.0)

                # Convert to CPU float32 for PSNR
                gt_cpu = gt_video.cpu().float()
                gen_cpu = gen_video.cpu().float()

                B, C, T, H, W = gt_cpu.shape

                # Compute PSNR for non-conditioning frames only
                batch_psnr_values = []
                for t in range(num_cond_frames, T):
                    gt_frame = gt_cpu[:, :, t, :, :]
                    gen_frame = gen_cpu[:, :, t, :, :]
                    psnr_frame = peak_signal_noise_ratio(
                        gen_frame,
                        gt_frame,
                        data_range=2.0,
                        reduction="none",
                        dim=(1, 2, 3),
                    )
                    if psnr_frame.dim() == 0:
                        psnr_frame = psnr_frame.unsqueeze(0)
                    batch_psnr_values.append(psnr_frame)

                if batch_psnr_values:
                    batch_psnr = torch.stack(batch_psnr_values).mean()
                    all_psnr_values.append(batch_psnr)
            else:
                # No video_frames available (using pre-encoded latents)
                # Decode ground truth latents to get pseudo-ground-truth video for PSNR
                logger.info("Using decoded latents as ground truth for PSNR")

                # Set num_cond_frames for generation
                input_dict["num_cond_frames"] = num_cond_frames

                # Generate video using the diffusion model
                generated_video = generate_video(
                    device=self.device,
                    dtype=self._dtype,
                    job_config=self.job_config,
                    model=model,
                    input_dict=input_dict,
                    wan_video_vae=self.wan_video_vae,
                    t5_tokenizer=self.t5_tokenizer,
                    t5_encoder=self.t5_encoder,
                    precomputed_t5_embedding=self.precomputed_t5_embedding,
                )

                # Save generated video
                output_dir = os.path.join(
                    self.job_config.job.dump_folder,
                    self.job_config.validation.save_img_folder,
                )
                save_video(
                    name=f"video_rank{str(torch.distributed.get_rank())}_step{step}.mp4",
                    output_dir=output_dir,
                    video=generated_video,
                    add_sampling_metadata=True,
                )

                # Decode ground truth latents to video for PSNR comparison
                # latents shape: (B, C, T, H, W)
                gt_video = self.wan_video_vae.decode(
                    hidden_states=latents,
                    device=self.device,
                    tiled=True,
                )
                gt_video = gt_video.clamp(-1.0, 1.0)

                # Save decoded ground truth video for comparison
                save_video(
                    name=f"target_video_rank{str(torch.distributed.get_rank())}_step{step}.mp4",
                    output_dir=output_dir,
                    video=gt_video,
                    add_sampling_metadata=False,
                )

                # Clamp generated video
                gen_video = generated_video.clamp(-1.0, 1.0)

                # Convert to CPU float32 for PSNR
                gt_cpu = gt_video.cpu().float()
                gen_cpu = gen_video.cpu().float()

                B, C, T, H, W = gt_cpu.shape

                # Compute PSNR for non-conditioning frames only
                batch_psnr_values = []
                for t in range(num_cond_frames, T):
                    gt_frame = gt_cpu[:, :, t, :, :]
                    gen_frame = gen_cpu[:, :, t, :, :]
                    psnr_frame = peak_signal_noise_ratio(
                        gen_frame,
                        gt_frame,
                        data_range=2.0,
                        reduction="none",
                        dim=(1, 2, 3),
                    )
                    if psnr_frame.dim() == 0:
                        psnr_frame = psnr_frame.unsqueeze(0)
                    batch_psnr_values.append(psnr_frame)

                if batch_psnr_values:
                    batch_psnr = torch.stack(batch_psnr_values).mean()
                    all_psnr_values.append(batch_psnr)

            # Update token count for metrics
            if self.metrics_processor is not None:
                self.metrics_processor.ntokens_since_last_log += labels.numel()

            num_steps += 1

        # Compute average MSE loss across all batches
        if accumulated_losses:
            avg_loss = torch.stack(accumulated_losses).mean()
        else:
            avg_loss = torch.tensor(0.0, device=self.device, dtype=self._dtype)

        # Compute average PSNR across all batches
        if all_psnr_values:
            avg_psnr = torch.stack(all_psnr_values).mean().to(device=self.device)
        else:
            avg_psnr = torch.tensor(0.0, device=self.device)

        # Gather across distributed processes if needed
        if parallel_dims.dp_cp_enabled:
            # Use "loss" mesh which includes dp_replicate, dp_shard, and cp
            loss_mesh = parallel_dims.get_optional_mesh("loss")
            if loss_mesh is not None:
                avg_loss = dist_utils.dist_mean(avg_loss, loss_mesh)
                avg_psnr = dist_utils.dist_mean(avg_psnr, loss_mesh)

        # Convert to Python scalars for logging
        if isinstance(avg_loss, torch.Tensor):
            avg_loss = avg_loss.item()
        if isinstance(avg_psnr, torch.Tensor):
            avg_psnr = avg_psnr.item()

        # Log validation metrics
        logger.info(
            f"Validation Step {step}: Loss = {avg_loss:.6f}, PSNR = {avg_psnr:.4f} dB"
        )
        self.metrics_processor.log_validation(
            loss=avg_loss,
            step=step,
            extra_metrics={"validation_metrics/avg_psnr": avg_psnr},
        )

        # Set model back to train mode
        model.train()

        # Re-enable cfg dropout for training
        self.job_config.training.classifier_free_guidance_prob = training_cfg_prob


def build_wan_validator(
    job_config: JobConfig,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    parallel_dims: ParallelDims,
    loss_fn: LossFunction,
    validation_context: Generator[None, None, None],
    maybe_enable_amp: Generator[None, None, None],
    metrics_processor: MetricsProcessor | None = None,
    pp_schedule: _PipelineSchedule | None = None,
    pp_has_first_stage: bool | None = None,
    pp_has_last_stage: bool | None = None,
) -> WanValidator:
    """Build a simple validator focused on correctness."""
    return WanValidator(
        job_config=job_config,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        parallel_dims=parallel_dims,
        loss_fn=loss_fn,
        validation_context=validation_context,
        maybe_enable_amp=maybe_enable_amp,
        metrics_processor=metrics_processor,
        pp_schedule=pp_schedule,
        pp_has_first_stage=pp_has_first_stage,
        pp_has_last_stage=pp_has_last_stage,
    )


if __name__ == "__main__":
    """
    Test script for the WanValidator.

    This script tests the validation dataloader and basic setup without requiring
    distributed training. It initializes components with random weights for testing.

    Usage:
        python -m torchtitan.experiments.wan.validate
    """
    import random
    from datetime import datetime

    # from icecream import ic
    from PIL import Image

    from torchtitan.config.manager import ConfigManager
    from torchtitan.experiments.wan import get_train_spec, wan_configs, WanModel
    from torchtitan.experiments.wan.model.wan_vae import load_wan_vae

    # Initialize logger for standalone execution
    init_logger()
    logger.info("Starting WanValidator test script")

    # Parse config from CLI arguments (passed via torchrun or command line)
    # Usage: torchrun ... -m torchtitan.experiments.wan.validate --job.config_file=<path>
    config_manager = ConfigManager()
    job_config = config_manager.parse_args()  # Uses sys.argv automatically
    logger.info(f"Config loaded from: {job_config.job.config_file}")
    logger.info("Config loaded successfully")
    train_spec = get_train_spec()

    # Initialize distributed environment (same pattern as train.py)
    logger.info("Initializing distributed environment...")
    world_size = dist_utils.init_distributed(
        job_config.comm,
        enable_cpu_backend=job_config.training.enable_cpu_offload,
        base_folder=job_config.job.dump_folder,
    )
    logger.info(f"Distributed environment initialized with world_size={world_size}")

    # Get rank from distributed environment
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    logger.info(f"Global rank={rank}, local_rank={local_rank}")

    # Set device based on LOCAL_RANK (same pattern as forge/engine.py)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    if device_type == "cuda":
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device(device_type)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    logger.info(f"Using device: {device}, dtype: {dtype}")

    # Create ParallelDims from config (same pattern as train.py)
    parallelism_config = job_config.parallelism
    parallel_dims = ParallelDims(
        dp_shard=parallelism_config.data_parallel_shard_degree,
        dp_replicate=parallelism_config.data_parallel_replicate_degree,
        cp=parallelism_config.context_parallel_degree,
        tp=parallelism_config.tensor_parallel_degree,
        pp=parallelism_config.pipeline_parallel_degree,
        ep=parallelism_config.expert_parallel_degree,
        etp=parallelism_config.expert_tensor_parallel_degree,
        world_size=world_size,
    )
    logger.info(f"ParallelDims initialized: {parallel_dims}")

    # Extract DP world size and rank from batch mesh (same pattern as forge/engine.py)
    if parallel_dims.dp_enabled:
        # Original code commented out due to "Backend fake does not yet support sequence numbers" error
        # batch_mesh = parallel_dims.get_mesh("batch")
        # dp_world_size = batch_mesh.size()
        # dp_rank = batch_mesh.get_local_rank()
        # Compute batch_rank without using get_local_rank() on fake backend
        dp_world_size = parallel_dims.dp_replicate * parallel_dims.dp_shard
        world_rank = torch.distributed.get_rank()
        dp_rank = (world_rank // (parallel_dims.cp * parallel_dims.tp)) % dp_world_size
    else:
        dp_world_size = 1
        dp_rank = 0
    logger.info(f"Data parallel: dp_world_size={dp_world_size}, dp_rank={dp_rank}")

    # Set random seeds for reproducibility using the same utility as train.py
    # This ensures consistency with training pipeline
    logger.info("Setting random seed for reproducibility...")
    distinct_seed_mesh_dims = []
    if parallel_dims.dp_enabled:
        # Use distinct seeds across DP ranks like in train.py
        distinct_seed_mesh_dims = ["dp_replicate", "fsdp"]
    dist_utils.set_determinism(
        parallel_dims=parallel_dims,
        device=device,
        debug_config=job_config.debug,
        distinct_seed_mesh_dims=distinct_seed_mesh_dims,
    )

    # Build tokenizer
    logger.info("Building tokenizer...")
    tokenizer = build_wan_tokenizer(job_config)
    logger.info("Tokenizer built successfully")

    # Build validator (this will also create the dataloader)
    logger.info("Building validator...")
    validator = build_wan_validator(
        job_config=job_config,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        parallel_dims=parallel_dims,
        loss_fn=None,  # Not needed for dataloader test
        validation_context=None,
        maybe_enable_amp=None,
    )
    logger.info("Validator built successfully")
    # ic(validator)

    # Get model args for VAE params
    model_args = wan_configs[job_config.model.flavor]
    # ic(model_args)

    logger.info(f"Loading VAE from: {job_config.encoder.wan_vae_path}")
    wan_video_vae = load_wan_vae(
        chkpt_path=job_config.encoder.wan_vae_path,
        wan_vae_params=model_args.wan_video_vae_params,
        device=device,
        dtype=dtype,
    )
    logger.info("VAE loaded successfully")

    logger.info(f"Loading T5 encoder: {job_config.encoder.t5_encoder}")
    t5_encoder = WanEmbedder(
        version=job_config.encoder.t5_encoder,
    ).to(device=device, dtype=dtype)
    logger.info("T5 encoder loaded successfully")

    # Initialize validator with encoders
    validator.wan_init(
        device=device,
        _dtype=dtype,
        wan_video_vae=wan_video_vae,
        t5_encoder=t5_encoder,
        precomputed_t5_embedding=None,
    )
    logger.info("Validator initialized with encoders")

    # Load the WanModel (diffusion model)
    logger.info("=" * 80)
    logger.info("Loading WanModel (diffusion model)...")
    logger.info("=" * 80)
    model = WanModel(model_args)

    # Load pretrained weights if not in test mode
    if not job_config.training.test_mode:
        weights_path = "assets/hf/Wan2.2-TI2V-5B"
        logger.info(f"Loading pretrained weights from: {weights_path}")
        model.init_weights(pretrained_weights_path=weights_path)
        logger.info("Pretrained weights loaded successfully")
    else:
        logger.info("Using random initialization (test_mode=True)")

    model = model.to(device=device, dtype=dtype)
    model.eval()
    logger.info(f"WanModel loaded and moved to {device} with dtype {dtype}")

    # Test the dataloader by iterating through a few batches
    logger.info("Testing validation dataloader...")

    num_batches_to_test = 3
    for batch_idx, (input_dict, labels) in enumerate(validator.validation_dataloader):
        if batch_idx >= num_batches_to_test:
            break
        # ic(input_dict.keys())
        # ic(input_dict["video_frames"].shape)
        # ic(labels.shape)
        assert torch.all(input_dict["video_frames"] == labels)
        with torch.inference_mode():
            processed = preprocess_data(
                device=device,
                dtype=dtype,
                wan_video_vae=wan_video_vae,
                t5_encoder=t5_encoder,
                batch=input_dict,
                precomputed_t5_embedding=None,
            )
            latents = processed["latents"]
            # ic(latents.shape)
            latentes = latents.permute(0, 2, 1, 3, 4)
            # ic(latentes.shape)
            # TODO: fix the tiled execution path
            reconstructed_video = wan_video_vae.decode(
                hidden_states=latents, device=device, tiled=True
            )
        reconstructed_video = reconstructed_video.clamp(-1.0, 1.0)

        # Get normalized video frames for PSNR comparison
        # preprocess_data normalizes and converts to (B, C, T, H, W) format
        # We need to reconstruct this from the original input to compare
        video_frames_original = input_dict["video_frames"]
        # Normalize and convert to same format as preprocess_data output
        video_frames = video_frames_original.float()  # Convert uint8 to float
        video_frames = (video_frames / 127.5) - 1.0  # Normalize [0, 255] -> [-1, 1]
        video_frames = video_frames.to(device=device, dtype=dtype)
        video_frames = video_frames.permute(
            0, 4, 1, 2, 3
        )  # (B, T, H, W, C) -> (B, C, T, H, W)

        # Convert back to same dtype as original for PSNR calculation
        # Both should be float32 for accurate PSNR calculation
        video_frames_cpu = video_frames.cpu().float()
        reconstructed_cpu = reconstructed_video.cpu().float()

        # Calculate PSNR per frame
        # Both videos are in (B, C, T, H, W) format
        B, C, T, H, W = video_frames_cpu.shape
        psnr_values = []

        # logger.info(f"\nCalculating PSNR for {T} frames...")
        for t in range(T):
            # Extract frame t: (B, C, H, W)
            original_frame = video_frames_cpu[:, :, t, :, :]  # (B, C, H, W)
            reconstructed_frame = reconstructed_cpu[:, :, t, :, :]  # (B, C, H, W)

            # Compute PSNR for this frame across batch
            # data_range=2.0 because values are in [-1, 1] range (range = 2.0)
            # reduction="none" to get per-sample PSNR, dim=(1,2,3) to reduce over C, H, W
            psnr_frame = peak_signal_noise_ratio(
                reconstructed_frame,
                original_frame,
                data_range=2.0,  # Range is [-1, 1] = 2.0
                reduction="none",
                dim=(1, 2, 3),  # Reduce over C, H, W, keep batch dimension
            )

            # Ensure it's 1D tensor: (B,)
            if psnr_frame.dim() == 0:
                psnr_frame = psnr_frame.unsqueeze(0)
            psnr_values.append(psnr_frame)

        # Stack to get (T, B) then transpose to (B, T)
        if len(psnr_values) > 0:
            psnr_values = torch.stack(psnr_values, dim=0)  # (T, B)
            if psnr_values.dim() == 2:
                psnr_values = psnr_values.transpose(0, 1)  # (B, T)

        # psnr_values shape: (B, T) - PSNR for each batch and frame
        logger.info("\nPSNR Results:")
        logger.info(f"  - Overall PSNR (mean): {psnr_values.mean().item():.4f} dB")
        logger.info(f"  - PSNR min: {psnr_values.min().item():.4f} dB")
        logger.info(f"  - PSNR max: {psnr_values.max().item():.4f} dB")

        # This PSNR represents the upper bound - best possible reconstruction
        # Any model-generated video should have PSNR <= this value
        logger.info("VAE PSNR Test Summary:")
        logger.info(
            f"  - This PSNR ({psnr_values.mean().item():.4f} dB) is the UPPER BOUND"
        )
        logger.info("  - Model-generated videos should have PSNR <= this value")
        logger.info("  - Higher PSNR = better reconstruction quality")
        # Save frames with highest and lowest PSNR for each video in batch
        logger.info("Saving frames with highest and lowest PSNR...")

        # Generate timestamp and random string for folder name
        # Format: YYYYMMDD_HHMMSS_randomstring
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_str = random.randint(0, 1000000)  # 8 character random hex string
        folder_name = f"{timestamp}_{random_str}_vae_psnr_frames"

        # Get logs directory from job config or use default
        logs_dir = os.path.join(job_config.job.dump_folder, folder_name)
        os.makedirs(logs_dir, exist_ok=True)
        # logger.info(f"Saving frames to: {logs_dir}")

        # psnr_values shape: (B, T) - PSNR for each batch and frame
        for b in range(B):
            # Get PSNR values for this video: (T,)
            video_psnr = psnr_values[b]  # Shape: (T,)

            # Find indices of highest and lowest PSNR frames
            max_psnr_idx = video_psnr.argmax().item()
            min_psnr_idx = video_psnr.argmin().item()

            max_psnr_value = video_psnr[max_psnr_idx].item()
            min_psnr_value = video_psnr[min_psnr_idx].item()

            # logger.info(f"\nVideo {b}:")
            # logger.info(f"  - Highest PSNR: frame {max_psnr_idx}, value: {max_psnr_value:.4f} dB")
            # logger.info(f"  - Lowest PSNR: frame {min_psnr_idx}, value: {min_psnr_value:.4f} dB")

            # Extract frames from original and reconstructed videos
            # Both are in (B, C, T, H, W) format
            original_max_frame = video_frames_cpu[b, :, max_psnr_idx, :, :]  # (C, H, W)
            reconstructed_max_frame = reconstructed_cpu[
                b, :, max_psnr_idx, :, :
            ]  # (C, H, W)

            original_min_frame = video_frames_cpu[b, :, min_psnr_idx, :, :]  # (C, H, W)
            reconstructed_min_frame = reconstructed_cpu[
                b, :, min_psnr_idx, :, :
            ]  # (C, H, W)

            # Convert from [-1, 1] float to [0, 255] uint8 for saving
            def frame_to_uint8(frame):
                """Convert frame from [-1, 1] float to [0, 255] uint8."""
                frame = frame.clamp(-1.0, 1.0)
                frame = (frame + 1.0) * 127.5  # Map [-1, 1] to [0, 255]
                frame = frame.clamp(0, 255)
                return frame.to(dtype=torch.uint8)

            # Convert to numpy and then to PIL Image
            def tensor_to_pil(tensor):
                """Convert tensor (H, W, C) to PIL Image."""
                numpy_array = tensor.cpu().numpy()
                return Image.fromarray(numpy_array)

            # Convert frames to uint8
            original_max_frame_uint8 = frame_to_uint8(original_max_frame)
            reconstructed_max_frame_uint8 = frame_to_uint8(reconstructed_max_frame)
            original_min_frame_uint8 = frame_to_uint8(original_min_frame)
            reconstructed_min_frame_uint8 = frame_to_uint8(reconstructed_min_frame)

            original_max_frame_hwc = original_max_frame_uint8.permute(1, 2, 0)
            reconstructed_max_frame_hwc = reconstructed_max_frame_uint8.permute(1, 2, 0)
            original_min_frame_hwc = original_min_frame_uint8.permute(1, 2, 0)
            reconstructed_min_frame_hwc = reconstructed_min_frame_uint8.permute(1, 2, 0)

            # Save highest PSNR frames
            original_max_img = tensor_to_pil(original_max_frame_hwc)
            reconstructed_max_img = tensor_to_pil(reconstructed_max_frame_hwc)

            original_max_path = os.path.join(
                logs_dir,
                f"batch{b}_frame{max_psnr_idx}_original_max_psnr_{max_psnr_value:.2f}dB.png",
            )
            reconstructed_max_path = os.path.join(
                logs_dir,
                f"batch{b}_frame{max_psnr_idx}_reconstructed_max_psnr_{max_psnr_value:.2f}dB.png",
            )

            original_max_img.save(original_max_path)
            reconstructed_max_img.save(reconstructed_max_path)
            # logger.info(f"  - Saved max PSNR frames: {original_max_path}, {reconstructed_max_path}")

            # Save lowest PSNR frames
            original_min_img = tensor_to_pil(original_min_frame_hwc)
            reconstructed_min_img = tensor_to_pil(reconstructed_min_frame_hwc)

            original_min_path = os.path.join(
                logs_dir,
                f"batch{b}_frame{min_psnr_idx}_original_min_psnr_{min_psnr_value:.2f}dB.png",
            )
            reconstructed_min_path = os.path.join(
                logs_dir,
                f"batch{b}_frame{min_psnr_idx}_reconstructed_min_psnr_{min_psnr_value:.2f}dB.png",
            )

            original_min_img.save(original_min_path)
            reconstructed_min_img.save(reconstructed_min_path)
            # logger.info(f"  - Saved min PSNR frames: {original_min_path}, {reconstructed_min_path}")

        logger.info(f"\n✓ All frames saved to: {logs_dir}")

        # ========================================
        # Video Generation Test (using diffusion model)
        # ========================================
        logger.info("=" * 80)
        logger.info(
            f"Batch {batch_idx}: Testing video generation with diffusion model..."
        )
        logger.info("=" * 80)

        # Set num_cond_frames for the generation
        input_dict["num_cond_frames"] = job_config.validation.num_cond_frames

        with torch.inference_mode():
            generated_video = generate_video(
                device=device,
                dtype=dtype,
                job_config=job_config,
                model=model,
                input_dict=input_dict,
                wan_video_vae=wan_video_vae,
                t5_tokenizer=validator.t5_tokenizer,
                t5_encoder=t5_encoder,
                precomputed_t5_embedding=None,
            )

        logger.info(f"Generated video shape: {generated_video.shape}")

        # Save the generated video
        generated_video_path = os.path.join(
            logs_dir, f"batch{batch_idx}_generated_video.mp4"
        )
        save_video(
            name=f"batch{batch_idx}_generated_video.mp4",
            output_dir=logs_dir,
            video=generated_video,
            add_sampling_metadata=True,
        )
        logger.info(f"✓ Generated video saved to: {generated_video_path}")

        # ========================================
        # Compute PSNR for generated video
        # ========================================
        # Compare generated video against ground truth
        # Only compare non-conditioning frames (the model should predict these)
        num_cond_frames = job_config.validation.num_cond_frames

        # Get ground truth video in same format as generated
        # video_frames is (B, T, H, W, C) uint8 [0, 255]
        # generated_video is (B, C, T, H, W) float [-1, 1]
        gt_video = input_dict["video_frames"].float()
        gt_video = (gt_video / 127.5) - 1.0  # Normalize to [-1, 1]
        gt_video = gt_video.to(device=device, dtype=dtype)
        gt_video = gt_video.permute(0, 4, 1, 2, 3)  # (B, T, H, W, C) -> (B, C, T, H, W)

        # Clamp generated video to valid range
        generated_video_clamped = generated_video.clamp(-1.0, 1.0)

        # Convert to CPU float32 for PSNR calculation
        gt_video_cpu = gt_video.cpu().float()
        gen_video_cpu = generated_video_clamped.cpu().float()

        # Calculate PSNR for non-conditioning frames only
        # These are the frames the model actually predicted
        B, C, T, H, W = gt_video_cpu.shape

        gen_psnr_values = []
        for t in range(num_cond_frames, T):  # Skip conditioning frames
            gt_frame = gt_video_cpu[:, :, t, :, :]
            gen_frame = gen_video_cpu[:, :, t, :, :]

            psnr_frame = peak_signal_noise_ratio(
                gen_frame,
                gt_frame,
                data_range=2.0,  # Range is [-1, 1] = 2.0
                reduction="none",
                dim=(1, 2, 3),
            )
            if psnr_frame.dim() == 0:
                psnr_frame = psnr_frame.unsqueeze(0)
            gen_psnr_values.append(psnr_frame)

        if len(gen_psnr_values) > 0:
            gen_psnr_tensor = torch.stack(gen_psnr_values, dim=0)  # (T-cond, B)
            if gen_psnr_tensor.dim() == 2:
                gen_psnr_tensor = gen_psnr_tensor.transpose(0, 1)  # (B, T-cond)

            logger.info("\n" + "=" * 60)
            logger.info("GENERATED VIDEO PSNR Results:")
            logger.info("=" * 60)
            logger.info(
                f"  - Overall PSNR (mean): {gen_psnr_tensor.mean().item():.4f} dB"
            )
            logger.info(f"  - PSNR min: {gen_psnr_tensor.min().item():.4f} dB")
            logger.info(f"  - PSNR max: {gen_psnr_tensor.max().item():.4f} dB")
            logger.info(
                f"  - Frames evaluated: {T - num_cond_frames} (skipped {num_cond_frames} conditioning frames)"
            )
            logger.info("=" * 60)

            # Also compute last-frame PSNR (common metric for video prediction)
            last_frame_psnr = peak_signal_noise_ratio(
                gen_video_cpu[:, :, -1, :, :],
                gt_video_cpu[:, :, -1, :, :],
                data_range=2.0,
            )
            logger.info(f"  - Last frame PSNR: {last_frame_psnr.item():.4f} dB")
        else:
            logger.warning("No non-conditioning frames to evaluate!")
